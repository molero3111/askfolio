"""Optional PostHog usage analytics: anonymous ids, scrubbed questions, never raises.

Off unless both POSTHOG_KEY and ANALYTICS_SALT are set. Config is read at call time so
tests can monkeypatch it.
"""
import hashlib
import hmac
import logging

from posthog import Posthog

from app import config
from app.secure_io import atomic_write_secret
from ingest.scrub import scrub_text

logger = logging.getLogger(__name__)

PROJECT_SOURCE = "askfolio"
QUESTION_MAX_CHARS = 500

_client = None
# Hashed ids of chats already seen; loaded from SEEN_CHATS_FILE on first use
_seen: set[str] | None = None


def enabled() -> bool:
    return bool(config.POSTHOG_KEY and config.ANALYTICS_SALT)


def distinct_id(chat_id: int | str) -> str:
    """Salted HMAC of the chat id: stable per chat, not reversible without the salt."""
    digest = hmac.new(config.ANALYTICS_SALT.encode(), str(chat_id).encode(), hashlib.sha256)
    return digest.hexdigest()[:32]


def clean_question(text: str) -> str:
    # Scrub before the cut, so a cut can't leave half an email or phone number behind
    scrubbed, _ = scrub_text(text)
    return scrubbed[:QUESTION_MAX_CHARS]


def _get_client():
    global _client
    if _client is None:
        # Async batching on a background thread: capture() only enqueues. A short timeout and
        # one retry keep the flush at shutdown within a few seconds when PostHog is down.
        _client = Posthog(
            config.POSTHOG_KEY,
            host=config.POSTHOG_HOST,
            timeout=5,
            max_retries=1,
            max_queue_size=1000,
        )
    return _client


def _capture(event: str, chat_id: int | str, properties: dict) -> None:
    try:
        props = {
            "project_source": PROJECT_SOURCE,
            "environment": config.ANALYTICS_ENV,
            "is_internal": str(chat_id) in config.ANALYTICS_INTERNAL_CHAT_IDS,
            "$ip": None,
            "$process_person_profile": False,
            **properties,
        }
        _get_client().capture(event, distinct_id=distinct_id(chat_id), properties=props)
    except Exception as e:
        # Type only: the message could carry the question or the key
        logger.warning("Analytics capture of %s failed (%s)", event, type(e).__name__)


def track_question(
    chat_id: int | str,
    question: str,
    outcome: str,
    *,
    scope: str | None = None,
    latency_ms: int | None = None,
) -> None:
    if not enabled():
        return
    try:
        properties: dict = {"question": clean_question(question), "outcome": outcome}
        if scope is not None:
            properties["rate_limit_scope"] = scope
        if latency_ms is not None:
            properties["latency_ms"] = latency_ms
    except Exception as e:
        logger.warning("Analytics could not prepare question_asked (%s)", type(e).__name__)
        return
    _capture("question_asked", chat_id, properties)


def _load_seen() -> set[str]:
    try:
        text = config.resolve_path(config.SEEN_CHATS_FILE).read_text(encoding="utf-8")
    except FileNotFoundError:
        return set()
    except (OSError, UnicodeDecodeError) as e:
        # At worst a first-seen event repeats once
        logger.warning("Could not read the seen-chats file (%s); starting empty", type(e).__name__)
        return set()
    return {line.strip() for line in text.splitlines() if line.strip()}


def track_first_seen(chat_id: int | str) -> None:
    """Capture recruiter_first_seen the first time a chat writes. Only hashed ids are stored."""
    global _seen
    if not enabled():
        return
    try:
        if _seen is None:
            _seen = _load_seen()
        did = distinct_id(chat_id)
        if did in _seen:
            return
        _seen.add(did)
        try:
            path = config.resolve_path(config.SEEN_CHATS_FILE)
            atomic_write_secret(path, "".join(f"{s}\n" for s in sorted(_seen)))
        except OSError as e:
            logger.warning("Could not save the seen-chats file (%s)", type(e).__name__)
    except Exception as e:
        logger.warning("Analytics first-seen check failed (%s)", type(e).__name__)
        return
    _capture("recruiter_first_seen", chat_id, {})


def shutdown() -> None:
    """Flush queued events and stop the client. Safe when disabled or called twice."""
    global _client
    client, _client = _client, None
    if client is None:
        return
    try:
        client.shutdown()
    except Exception as e:
        logger.warning("Analytics shutdown failed (%s)", type(e).__name__)
