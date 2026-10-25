"""In-memory rate limiters for incoming messages and meeting bookings.

State lives in process memory and resets on restart. The public interfaces
(``MessageLimiter.check``, ``BookingLimiter.can_book``/``record``) are small so
a Postgres-backed implementation can replace these classes later.
"""

import time
from collections import defaultdict, deque
from collections.abc import Callable
from enum import Enum

_DAY_SECONDS = 86400
_MINUTE_SECONDS = 60


def _utc_day(now: float) -> int:
    """UTC calendar day index for a Unix timestamp."""
    return int(now // _DAY_SECONDS)


class Decision(Enum):
    ALLOWED = "allowed"
    LIMITED_NOTIFY = "limited_notify"  # first refusal in a window: tell the user
    LIMITED_SILENT = "limited_silent"  # later refusals: stay quiet


class MessageLimiter:
    """Per-chat sliding-minute and per-day limits, plus a global daily cap."""

    def __init__(
        self,
        per_chat_per_minute: int,
        per_chat_per_day: int,
        global_per_day: int,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._per_minute = per_chat_per_minute
        self._per_day = per_chat_per_day
        self._global_per_day = global_per_day
        self._clock = clock
        self._day = _utc_day(clock())
        self._global_count = 0
        self._chat_day: dict[str, int] = defaultdict(int)
        self._recent: dict[str, deque[float]] = defaultdict(deque)
        # Refusal scopes already notified in their current window.
        self._notified: set[tuple[str, str]] = set()

    def _roll_day(self, now: float) -> None:
        day = _utc_day(now)
        if day != self._day:
            self._day = day
            self._global_count = 0
            self._chat_day.clear()
            self._notified = {k for k in self._notified if k[0] == "minute"}

    def _refuse(self, scope: str, chat_id: str) -> Decision:
        key = (scope, "" if scope == "global" else chat_id)
        if key in self._notified:
            return Decision.LIMITED_SILENT
        self._notified.add(key)
        return Decision.LIMITED_NOTIFY

    def check(self, chat_id: str) -> Decision:
        """Return the decision for one incoming message; counts it if allowed."""
        now = self._clock()
        self._roll_day(now)

        recent = self._recent[chat_id]
        while recent and recent[0] <= now - _MINUTE_SECONDS:
            recent.popleft()

        if self._global_count >= self._global_per_day:
            return self._refuse("global", chat_id)
        if self._chat_day[chat_id] >= self._per_day:
            return self._refuse("day", chat_id)
        if len(recent) >= self._per_minute:
            return self._refuse("minute", chat_id)

        recent.append(now)
        self._chat_day[chat_id] += 1
        self._global_count += 1
        self._notified.discard(("minute", chat_id))
        return Decision.ALLOWED


class BookingDecision(Enum):
    ALLOWED = "allowed"
    CHAT_CAP = "chat_cap"
    GLOBAL_CAP = "global_cap"


class BookingLimiter:
    """Per-chat and global daily caps on meeting bookings (UTC day)."""

    def __init__(
        self,
        per_chat_per_day: int,
        global_per_day: int,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._per_chat = per_chat_per_day
        self._global_cap = global_per_day
        self._clock = clock
        self._day = _utc_day(clock())
        self._global_count = 0
        self._chat_count: dict[str, int] = defaultdict(int)

    def _roll_day(self) -> None:
        day = _utc_day(self._clock())
        if day != self._day:
            self._day = day
            self._global_count = 0
            self._chat_count.clear()

    def can_book(self, chat_id: str) -> BookingDecision:
        """Read-only check; does not count anything."""
        self._roll_day()
        if self._global_count >= self._global_cap:
            return BookingDecision.GLOBAL_CAP
        if self._chat_count.get(chat_id, 0) >= self._per_chat:
            return BookingDecision.CHAT_CAP
        return BookingDecision.ALLOWED

    def record(self, chat_id: str) -> tuple[int, int]:
        """Count a booking; returns (global today, chat today)."""
        self._roll_day()
        self._global_count += 1
        self._chat_count[chat_id] += 1
        return self._global_count, self._chat_count[chat_id]
