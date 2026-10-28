import logging
import re
import time
from collections.abc import Sequence
from datetime import UTC
from typing import TypedDict

import requests
from langgraph.graph import END, START, StateGraph

from app import analytics, heartbeat
from app.calendar_client import (
    create_meeting,
    format_slot_label,
    get_available_slots,
    is_calendar_configured,
    resolve_timezone,
)
from app.config import (
    BOOKING_ENABLED,
    CALENDAR_DISPLAY_TIMEZONE,
    LANGSMITH_PROJECT,
    LLM_MODEL,
    RATE_LIMIT_GLOBAL_PER_DAY,
    RATE_LIMIT_PER_CHAT_PER_DAY,
    RATE_LIMIT_PER_CHAT_PER_MINUTE,
    TIMEZONE_ALIAS_FAST_PATH,
    is_langsmith_enabled,
)
from app.conversation import ConversationState, SchedulingMode
from app.llm_client import LLMCreditExhausted, LLMError, LLMResult, LLMUnavailable, chat  # noqa: F401
from app.llm_messages import build_qa_messages, build_timezone_messages
from app.observability import is_langsmith_exception, traceable, tracing_context
from app.offset import save_offset
from app.rag import get_relevant_context
from app.rate_limit import Decision, MessageLimiter
from app.telegram_client import TelegramError, get_updates, send_message
from app.text_format import to_plain_text

logger = logging.getLogger(__name__)

RATE_LIMIT_REPLY = (
    "You're sending messages faster than I can answer. Please wait a minute and try again."
)
DAY_LIMIT_REPLY = (
    "You've reached today's message limit. "
    "Please come back tomorrow, or reach out via the links on the portfolio."
)
GLOBAL_LIMIT_REPLY = (
    "I've answered as many questions as I can today. "
    "Please try again tomorrow, or reach out via the links on the portfolio."
)
_LIMIT_REPLIES = {
    "minute": RATE_LIMIT_REPLY,
    "chat_day": DAY_LIMIT_REPLY,
    "global": GLOBAL_LIMIT_REPLY,
}
BOOKING_DISABLED_REPLY = (
    "Scheduling isn't available right now. Please reach out via the links on the portfolio."
)

_message_limiter = MessageLimiter(
    per_chat_per_minute=RATE_LIMIT_PER_CHAT_PER_MINUTE,
    per_chat_per_day=RATE_LIMIT_PER_CHAT_PER_DAY,
    global_per_day=RATE_LIMIT_GLOBAL_PER_DAY,
)


def _track(fn, *args, **kwargs) -> None:
    """Call an analytics function; it never raises, but a bug there must not cost a reply."""
    try:
        fn(*args, **kwargs)
    except Exception:
        logger.warning("[analytics] %s failed", getattr(fn, "__name__", "call"))


def _message_allowed(chat_id: int, message_id: int, question: str | None = None) -> bool:
    """Gate an LLM-backed path. Notifies once on refusal; later refusals are silent.

    A refused Q&A question (passed as question) is tracked; scheduling refusals are not.
    """
    result = _message_limiter.check(str(chat_id))
    if result.decision is Decision.ALLOWED:
        return True
    if result.decision is Decision.LIMITED_NOTIFY:
        _send_telegram(chat_id, _LIMIT_REPLIES.get(result.scope, RATE_LIMIT_REPLY), message_id)
    if question is not None:
        _track(analytics.track_question, chat_id, question, "rate_limited", scope=result.scope)
    return False

_SCHEDULING_KEYWORDS = frozenset(
    {
        "schedule", "meeting", "book", "appointment", "calendar",
        "available", "availability", "slot", "call", "interview",
        "set up", "when can", "catch up", "talk", "arrange",
    }
)
_CANCEL_KEYWORDS = frozenset(
    {
        "cancel", "nevermind", "never mind", "stop", "forget it",
        "no thanks", "not interested", "exit", "quit", "discard", "nope",
    }
)


def _is_scheduling_intent(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in _SCHEDULING_KEYWORDS)


def _is_cancel_intent(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in _CANCEL_KEYWORDS)


def _strip_reasoning(text: str) -> str:
    """Remove <think>...</think> blocks so only the final answer is sent."""
    out = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    out = re.sub(r"<think>[\s\S]*", "", out, flags=re.IGNORECASE)
    return out.strip()


def _format_history(history: list[dict]) -> str:
    if not history:
        return "(None)"
    return "\n".join(
        f"{'Recruiter' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in history
    )


class AgentState(TypedDict):
    offset: int | None
    # (chat_id, message_id to reply to, text)
    pending_messages: Sequence[tuple[int, int, str]]
    # str(chat_id) -> ConversationState.to_dict()
    conversations: dict


UNAVAILABLE_REPLY = (
    "Sorry, I'm temporarily unavailable. Please try again later."
)
ERROR_REPLY = "Sorry, there was an error. I couldn't process that."


@traceable(name="retrieve_context", run_type="retriever")
def _retrieve_context(user_input: str) -> str:
    return get_relevant_context(user_input)


@traceable(name="call_llm_http", run_type="llm")
def _call_llm(messages: list[dict]) -> str:
    return chat(messages).content


@traceable(name="telegram_send_message", run_type="tool")
def _send_telegram(chat_id: int, reply: str, reply_to_message_id: int) -> None:
    try:
        send_message(chat_id, to_plain_text(reply), reply_to_message_id=reply_to_message_id)
    except (requests.RequestException, TelegramError) as exc:
        logger.warning("[telegram] sendMessage failed (chat_id=%s): %s", chat_id, exc)


_IANA_KEY_RE = re.compile(r"[A-Za-z]+(?:[_/][A-Za-z_]+)+")


@traceable(name="resolve_timezone_llm", run_type="llm")
def _resolve_timezone_llm(text: str) -> str | None:
    """Use the LLM to map a free-form timezone description to a valid IANA key.

    This is the primary timezone resolver. The alias dict in calendar_client.py
    acts as an optional fast-path only when TIMEZONE_ALIAS_FAST_PATH=true.
    Returns None if the LLM cannot determine a valid timezone.
    """
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
    try:
        raw = _call_llm(build_timezone_messages(text))
        cleaned = _strip_reasoning(raw).strip()
        if not cleaned or cleaned.upper() == "UNKNOWN":
            return None
        # Primary: use the response verbatim if it's a valid IANA key
        try:
            ZoneInfo(cleaned)
            return cleaned
        except ZoneInfoNotFoundError:
            pass
        # Fallback: some models prepend prose — scan for the first Region/City token
        match = _IANA_KEY_RE.search(cleaned)
        if match:
            candidate = match.group(0)
            try:
                ZoneInfo(candidate)
                return candidate
            except ZoneInfoNotFoundError:
                pass
        logger.warning("[timezone_llm] LLM response not a valid IANA key (input len=%s)", len(text))
        return None
    except LLMError:
        logger.exception("[timezone_llm] LLM request failed (input len=%s)", len(text))
        return None
    except Exception:
        logger.exception("[timezone_llm] unexpected error resolving tz (input len=%s)", len(text))
        return None


def _tz_display_name(iana_key: str) -> str:
    """Return a readable timezone label, e.g. 'Eastern Time (ET)' or 'America/New_York'."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    try:
        abbr = datetime.now(UTC).astimezone(ZoneInfo(iana_key)).strftime("%Z")
        # Build a friendly name from the IANA key city part
        city = iana_key.split("/")[-1].replace("_", " ")
        return f"{city} ({abbr})"
    except Exception:
        return iana_key


def _handle_scheduling(
    conv: ConversationState,
    user_input: str,
    chat_id: int,
    message_id: int,
) -> None:
    """Drive the scheduling state machine for a single message. Mutates conv in place."""

    if _is_cancel_intent(user_input):
        reply = "No problem! Feel free to ask me anything else about Emmanuel's background."
        _send_telegram(chat_id, reply, message_id)
        conv.add_message("user", user_input)
        conv.add_message("assistant", reply)
        conv.mode = SchedulingMode.CANCELLED
        conv.reset_scheduling()
        return

    if conv.mode == SchedulingMode.IDLE:
        if not BOOKING_ENABLED:
            _send_telegram(chat_id, BOOKING_DISABLED_REPLY, message_id)
            conv.add_message("user", user_input)
            conv.add_message("assistant", BOOKING_DISABLED_REPLY)
            return

        if not is_calendar_configured():
            reply = "Meeting scheduling isn't configured yet. Please contact Emmanuel directly."
            _send_telegram(chat_id, reply, message_id)
            conv.add_message("user", user_input)
            conv.add_message("assistant", reply)
            return

        display_tz_key = conv.recruiter_display_tz or CALENDAR_DISPLAY_TIMEZONE
        try:
            slots = get_available_slots(display_tz_key=display_tz_key)
        except Exception:
            logger.exception("[scheduling] Failed to fetch calendar slots")
            reply = "I couldn't retrieve available slots right now. Please try again later or contact Emmanuel directly."
            _send_telegram(chat_id, reply, message_id)
            conv.add_message("user", user_input)
            conv.add_message("assistant", reply)
            return

        if not slots:
            reply = (
                "There are no available slots in the next 7 days. "
                "Please try again later or contact Emmanuel directly."
            )
            _send_telegram(chat_id, reply, message_id)
            conv.add_message("user", user_input)
            conv.add_message("assistant", reply)
            return

        conv.proposed_slots = slots
        conv.mode = SchedulingMode.AWAITING_SLOT_SELECTION
        slot_list = "\n".join(f"{i + 1}. {s['label']}" for i, s in enumerate(slots))
        tz_label = _tz_display_name(display_tz_key)
        reply = (
            f"Here are Emmanuel's available time slots (all times in {tz_label}):\n\n"
            f"{slot_list}\n\n"
            "Reply with the number of your preferred slot. "
            "If you'd like to see these in a different timezone, just say so — "
            "for example: \"show in Pacific Time\" or \"convert to CET\"."
        )
        _send_telegram(chat_id, reply, message_id)
        conv.add_message("user", user_input)
        conv.add_message("assistant", reply)
        return

    if conv.mode == SchedulingMode.AWAITING_SLOT_SELECTION:
        # Slot number takes priority
        number_match = re.search(r"\b([1-6])\b", user_input)
        if number_match:
            idx = int(number_match.group(1)) - 1
            if 0 <= idx < len(conv.proposed_slots):
                conv.selected_slot = conv.proposed_slots[idx]
                conv.mode = SchedulingMode.AWAITING_RECRUITER_INFO
                reply = (
                    f"Great choice! I've noted: {conv.selected_slot['label']}.\n\n"
                    "Could you share your name and email so I can send you a calendar invite?\n\n"
                    "Please reply in this format:\n"
                    "Name: Your Name\n"
                    "Email: your@email.com"
                )
                _send_telegram(chat_id, reply, message_id)
                conv.add_message("user", user_input)
                conv.add_message("assistant", reply)
                return

        # No valid number — check if it's a timezone change request
        # Alias fast-path is optional (TIMEZONE_ALIAS_FAST_PATH=true); LLM is primary
        resolved_tz = resolve_timezone(user_input) if TIMEZONE_ALIAS_FAST_PATH else None
        if not resolved_tz:
            if not _message_allowed(chat_id, message_id):
                return
            resolved_tz = _resolve_timezone_llm(user_input)
        if resolved_tz:
            from zoneinfo import ZoneInfo
            conv.recruiter_display_tz = resolved_tz
            display_tz = ZoneInfo(resolved_tz)
            slot_list = "\n".join(
                f"{i + 1}. {format_slot_label(s['start'], display_tz)}"
                for i, s in enumerate(conv.proposed_slots)
            )
            tz_label = _tz_display_name(resolved_tz)
            reply = (
                f"Here are the same slots converted to {tz_label}:\n\n{slot_list}\n\n"
                "Reply with the number of your preferred slot."
            )
            _send_telegram(chat_id, reply, message_id)
            conv.add_message("user", user_input)
            conv.add_message("assistant", reply)
            return

        slot_count = len(conv.proposed_slots)
        default_label = _tz_display_name(CALENDAR_DISPLAY_TIMEZONE)
        reply = (
            f"I couldn't determine that timezone. Times are shown in {default_label}.\n\n"
            f"Please reply with a number between 1 and {slot_count} to select your slot, "
            "or try a different timezone (e.g. \"Pacific Time\", \"London\", \"Asia/Tokyo\")."
        )
        _send_telegram(chat_id, reply, message_id)
        conv.add_message("user", user_input)
        conv.add_message("assistant", reply)
        return

    if conv.mode == SchedulingMode.AWAITING_RECRUITER_INFO:
        email_match = re.search(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", user_input)
        name_match = re.search(r"(?i)name\s*:\s*(.+)", user_input)

        name = name_match.group(1).strip() if name_match else None
        email = email_match.group(0) if email_match else None

        if not name or not email:
            missing = []
            if not name:
                missing.append("Name: Your Name")
            if not email:
                missing.append("Email: your@email.com")
            reply = "I couldn't parse that. Please reply with:\n" + "\n".join(missing)
            _send_telegram(chat_id, reply, message_id)
            conv.add_message("user", user_input)
            conv.add_message("assistant", reply)
            return

        conv.recruiter_name = name
        conv.recruiter_email = email

        try:
            event_link = create_meeting(conv.selected_slot, name, email)
            reply = (
                f"Meeting booked!\n\n"
                f"Date: {conv.selected_slot['label']}\n"
                f"Name: {name}\n"
                f"Email: {email}\n\n"
                f"A calendar invite has been sent to {email}."
            )
            if event_link:
                reply += f"\n\nEvent details: {event_link}"
        except Exception:
            logger.exception("[scheduling] Failed to create calendar event")
            reply = (
                "Sorry, there was an error creating the calendar event. "
                "Please try again or contact Emmanuel directly."
            )

        _send_telegram(chat_id, reply, message_id)
        conv.add_message("user", user_input)
        conv.add_message("assistant", reply)
        conv.reset_scheduling()


def get_telegram_updates_node(state: AgentState) -> dict:
    """Node: fetch new Telegram messages and return (chat_id, message_id, text) list."""
    offset = state.get("offset")
    updates = get_updates(offset=offset)
    num_updates = len(updates)
    logger.info("[get_telegram_updates] offset=%s, raw updates received=%s", offset, num_updates)

    pending = []
    max_update_id = offset or 0
    for u in updates:
        max_update_id = max(max_update_id, u["update_id"])
        msg = u.get("message") or u.get("edited_message")
        if not msg:
            continue
        text = (msg.get("text") or "").strip()
        if not text:
            continue
        pending.append((msg["chat"]["id"], msg["message_id"], text))

    logger.info("[get_telegram_updates] new messages with text=%s", len(pending))
    for i, (cid, mid, text) in enumerate(pending):
        logger.info(
            "[get_telegram_updates] message %s: chat_id=%s message_id=%s text_len=%s",
            i + 1,
            cid,
            mid,
            len(text),
        )

    next_offset = max_update_id + 1 if num_updates > 0 else offset
    if num_updates > 0 and next_offset is not None:
        # Persist immediately so a restart mid-processing never replays these updates.
        save_offset(next_offset)
    return {
        "pending_messages": pending,
        "offset": next_offset,
    }


def process_message_node(state: AgentState) -> dict:
    """Node: process the first pending message — scheduling state machine or LLM Q&A.

    A crash while handling one message must not drop the rest of the batch: the offset
    on disk is already past the whole batch, so log it, apologise, and move on.
    """
    heartbeat.touch()
    pending = list(state.get("pending_messages") or [])
    if not pending:
        logger.info("[process_message] no pending messages, skipping")
        return {"pending_messages": []}

    conversations: dict = dict(state.get("conversations") or {})
    chat_id, reply_to_message_id, user_input = pending[0]
    try:
        return _process_first(pending, conversations)
    except Exception as exc:
        if is_langsmith_exception(exc):
            # Let run_once see it and switch tracing off instead of failing every Q&A.
            raise
        logger.exception(
            "[process_message] failed (chat_id=%s, message_id=%s); skipping message",
            chat_id,
            reply_to_message_id,
        )
        _send_telegram(chat_id, ERROR_REPLY, reply_to_message_id)
        _track(analytics.track_question, chat_id, user_input, "error")
        return {"pending_messages": pending[1:], "conversations": conversations}


def _process_first(pending: list, conversations: dict) -> dict:
    chat_id, reply_to_message_id, user_input = pending[0]
    remaining = len(pending) - 1
    logger.info(
        "[process_message] processing message (chat_id=%s, reply_to=%s, %s more, text_len=%s)",
        chat_id,
        reply_to_message_id,
        remaining,
        len(user_input),
    )

    _track(analytics.track_first_seen, chat_id)
    key = str(chat_id)
    conv = ConversationState.from_dict(conversations.get(key, {}))

    # Route to scheduling if already in a scheduling flow or intent detected
    in_scheduling_flow = conv.mode not in (SchedulingMode.IDLE, SchedulingMode.CANCELLED)
    # With booking off, keyword intent must not hijack Q&A ("can we talk about Python?").
    if in_scheduling_flow or (BOOKING_ENABLED and _is_scheduling_intent(user_input)):
        if not in_scheduling_flow and not _message_allowed(chat_id, reply_to_message_id):
            # Cheap non-LLM path, but still capped so it can't hammer the calendar API.
            conversations[key] = conv.to_dict()
            return {"pending_messages": pending[1:], "conversations": conversations}
        logger.info("[process_message] scheduling path (mode=%s)", conv.mode.value)
        _handle_scheduling(conv, user_input, chat_id, reply_to_message_id)
        conversations[key] = conv.to_dict()
        return {"pending_messages": pending[1:], "conversations": conversations}

    if not _message_allowed(chat_id, reply_to_message_id, user_input):
        conversations[key] = conv.to_dict()
        return {"pending_messages": pending[1:], "conversations": conversations}

    # Profile Q&A path — RAG + LLM with conversation history
    trace_meta = {
        "chat_id": chat_id,
        "message_id": reply_to_message_id,
        "message_text": user_input,
        "llm_model": LLM_MODEL,
    }
    with tracing_context(
        enabled=is_langsmith_enabled(),
        project_name=LANGSMITH_PROJECT,
        tags=["telegram", "recruiter-bot"],
        metadata=trace_meta,
    ):
        context = _retrieve_context(user_input)
        logger.info("[process_message] retrieved context length=%s chars", len(context))

        chat_history = _format_history(conv.history)
        messages = build_qa_messages(context, chat_history, user_input)
        with tracing_context(
            enabled=is_langsmith_enabled(),
            project_name=LANGSMITH_PROJECT,
            metadata={
                "context_length": len(context),
                "context_retrieved": context[:6000],
            },
        ):
            latency_ms = None
            try:
                started = time.monotonic()
                raw = _call_llm(messages)
                reply = to_plain_text(_strip_reasoning(raw))
                if reply:
                    outcome = "answered"
                    latency_ms = int((time.monotonic() - started) * 1000)
                else:
                    logger.warning("[process_message] LLM reply empty after stripping reasoning")
                    reply = ERROR_REPLY
                    outcome = "error"
                logger.info("[process_message] LLM reply length=%s chars", len(reply))
            except (LLMCreditExhausted, LLMUnavailable) as e:
                reply = UNAVAILABLE_REPLY
                outcome = "llm_unavailable"
                logger.error("[process_message] LLM unavailable: %s", e)
            except Exception as e:
                reply = ERROR_REPLY
                outcome = "error"
                logger.exception("[process_message] LLM request failed: %s", e)
            _send_telegram(chat_id, reply, reply_to_message_id)
            _track(analytics.track_question, chat_id, user_input, outcome, latency_ms=latency_ms)

    conv.add_message("user", user_input)
    conv.add_message("assistant", reply)
    conversations[key] = conv.to_dict()
    return {"pending_messages": pending[1:], "conversations": conversations}


def should_continue(state: AgentState) -> str:
    return "process_message" if state.get("pending_messages") else "end"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("get_telegram_updates", get_telegram_updates_node)
    graph.add_node("process_message", process_message_node)
    graph.add_edge(START, "get_telegram_updates")
    graph.add_conditional_edges(
        "get_telegram_updates",
        should_continue,
        {"process_message": "process_message", "end": END},
    )
    graph.add_conditional_edges(
        "process_message",
        should_continue,
        {"process_message": "process_message", "end": END},
    )
    return graph.compile()
