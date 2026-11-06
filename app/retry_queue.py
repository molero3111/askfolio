"""Questions waiting for a late answer, kept in an owner-only file until answered or expired.

Holds the recruiter's question text, so logs here carry counts and exception types only.
Not thread-safe; the bot processes messages on a single thread.
"""
import json
import logging
import math
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import Enum
from functools import cache
from pathlib import Path

from app import config
from app.secure_io import atomic_write_secret

logger = logging.getLogger(__name__)

_FORMAT_VERSION = 1


@dataclass
class QueueItem:
    id: str
    chat_id: int
    message_id: int
    question: str
    created_at: float
    attempts: int = 0


class AddResult(Enum):
    ADDED = "added"
    DUPLICATE = "duplicate"
    FULL = "full"
    FAILED = "failed"


def _same_question(a: str, b: str) -> bool:
    return a.strip().casefold() == b.strip().casefold()


def _item_from_dict(raw: dict) -> QueueItem:
    item = QueueItem(
        id=raw["id"],
        chat_id=raw["chat_id"],
        message_id=raw["message_id"],
        question=raw["question"],
        created_at=raw["created_at"],
        attempts=raw["attempts"],
    )
    valid = (
        isinstance(item.id, str)
        and isinstance(item.chat_id, int)
        and isinstance(item.message_id, int)
        and isinstance(item.question, str)
        and isinstance(item.created_at, int | float)
        and isinstance(item.attempts, int)
    )
    if not valid or not math.isfinite(item.created_at):
        raise TypeError("bad field value")
    return item


def _next_attempt_from(raw: object) -> float:
    """A bad stored time means retry on the next cycle; it must not cost the queued questions."""
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return value if math.isfinite(value) else 0.0


class RetryQueue:
    def __init__(
        self,
        path: Path,
        *,
        interval: int,
        after_success: int,
        max_age: int,
        max_items: int,
        max_per_chat: int,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._path = path
        self._interval = interval
        self._after_success = after_success
        self._max_age = max_age
        self._max_items = max_items
        self._max_per_chat = max_per_chat
        self._clock = clock
        self._items: list[QueueItem] = []
        self._next_attempt_at = 0.0
        self._load()

    def __len__(self) -> int:
        return len(self._items)

    def _load(self) -> None:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            raw_items = data["items"]
            if not isinstance(raw_items, list):
                raise TypeError("items is not a list")
        except FileNotFoundError:
            return
        except Exception as exc:
            # Whatever is in this file, it must not stop the bot from starting
            logger.warning("[retry_queue] queue file unusable (%s); starting empty", type(exc).__name__)
            return
        self._next_attempt_at = _next_attempt_from(data.get("next_attempt_at"))
        skipped = 0
        for raw in raw_items:
            try:
                self._items.append(_item_from_dict(raw))
            except (KeyError, TypeError, OverflowError):
                skipped += 1
        if skipped:
            logger.warning("[retry_queue] skipped %s invalid items in the queue file", skipped)

    def _save(self) -> bool:
        try:
            if self._items:
                data = {
                    "version": _FORMAT_VERSION,
                    "next_attempt_at": self._next_attempt_at,
                    "items": [asdict(item) for item in self._items],
                }
                atomic_write_secret(self._path, json.dumps(data))
            else:
                self._path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("[retry_queue] could not save the queue file (%s)", type(exc).__name__)
            return False
        return True

    def add(self, chat_id: int, message_id: int, question: str) -> AddResult:
        """Keep a question. Only ADDED and DUPLICATE mean it will be retried."""
        same_chat = [item for item in self._items if item.chat_id == chat_id]
        if any(_same_question(item.question, question) for item in same_chat):
            return AddResult.DUPLICATE
        if len(self._items) >= self._max_items or len(same_chat) >= self._max_per_chat:
            return AddResult.FULL
        now = self._clock()
        previous_next = self._next_attempt_at
        if not self._items:
            self._next_attempt_at = now + self._interval
        self._items.append(QueueItem(uuid.uuid4().hex, chat_id, message_id, question, now))
        if self._save():
            return AddResult.ADDED
        self._items.pop()
        self._next_attempt_at = previous_next
        return AddResult.FAILED

    def discard(self, chat_id: int, question: str) -> None:
        """Drop a queued question that has since been answered live."""
        kept = [
            item
            for item in self._items
            if not (item.chat_id == chat_id and _same_question(item.question, question))
        ]
        if len(kept) == len(self._items):
            return
        self._items = kept
        if not self._items:
            self._next_attempt_at = 0.0
        self._save()

    def expired(self) -> list[QueueItem]:
        """The items that reached the maximum age; they stay queued until removed."""
        cutoff = self._clock() - self._max_age
        return [item for item in self._items if item.created_at <= cutoff]

    def remove(self, item: QueueItem) -> None:
        """Drop an item without changing when the next attempt runs."""
        self._items = [other for other in self._items if other.id != item.id]
        if not self._items:
            self._next_attempt_at = 0.0
        self._save()

    def next_due(self) -> QueueItem | None:
        """The item to try now: fewest attempts, then oldest, so one bad question can't block the rest."""
        now = self._clock()
        if now < self._next_attempt_at:
            return None
        live = [item for item in self._items if item.created_at > now - self._max_age]
        return min(live, key=lambda item: (item.attempts, item.created_at), default=None)

    def age_seconds(self, item: QueueItem) -> int:
        return int(self._clock() - item.created_at)

    def mark_failed(self, item: QueueItem) -> None:
        item.attempts += 1
        self._next_attempt_at = self._clock() + self._interval
        self._save()

    def mark_done(self, item: QueueItem) -> None:
        self._items = [other for other in self._items if other.id != item.id]
        self._next_attempt_at = self._clock() + self._after_success if self._items else 0.0
        self._save()


@cache
def get_queue() -> RetryQueue:
    """The bot's queue, loaded once from the configured file."""
    queue = RetryQueue(
        config.resolve_path(config.RETRY_QUEUE_FILE),
        interval=config.RETRY_QUEUE_INTERVAL_SECONDS,
        after_success=config.RETRY_QUEUE_AFTER_SUCCESS_SECONDS,
        max_age=config.RETRY_QUEUE_MAX_AGE_SECONDS,
        max_items=config.RETRY_QUEUE_MAX_ITEMS,
        max_per_chat=config.RETRY_QUEUE_MAX_PER_CHAT,
    )
    if len(queue):
        logger.info("[retry_queue] resumed %s queued questions", len(queue))
    return queue
