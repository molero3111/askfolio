import logging
import re
import requests
from typing import TypedDict, Sequence

from langgraph.graph import StateGraph, START, END

from app.calendar_client import (
    create_meeting,
    format_slot_label,
    get_available_slots,
    is_calendar_configured,
    resolve_timezone,
)
from app.config import (
    CALENDAR_DISPLAY_TIMEZONE,
    LANGSMITH_PROJECT,
    LLM_API_KEY,
    LLM_API_URL,
    LLM_MODEL,
    LLM_REQUEST_TIMEOUT,
    RECRUITER_PROMPT_PATH,
    TIMEZONE_ALIAS_FAST_PATH,
    TIMEZONE_PROMPT_PATH,
    is_langsmith_enabled,
)
from app.conversation import ConversationState, SchedulingMode
from app.observability import traceable, tracing_context
from app.rag import get_relevant_context
from app.telegram_client import get_updates, send_message

logger = logging.getLogger(__name__)

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
    out = out.strip()
    return out if out else "(No reply could be extracted.)"


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


@traceable(name="retrieve_context", run_type="retriever")
def _retrieve_context(user_input: str) -> str:
    return get_relevant_context(user_input)


@traceable(name="call_llm_http", run_type="llm")
def _call_llm(prompt_content: str) -> str:
    headers = {"Content-Type": "application/json"}
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt_content}],
    }
    r = requests.post(LLM_API_URL, headers=headers, json=payload, timeout=LLM_REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


@traceable(name="telegram_send_message", run_type="tool")
def _send_telegram(chat_id: int, reply: str, reply_to_message_id: int) -> None:
    send_message(chat_id, reply, reply_to_message_id=reply_to_message_id)


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
        prompt = TIMEZONE_PROMPT_PATH.read_text(encoding="utf-8").format(user_input=text)
        raw = _call_llm(prompt)
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
        logger.warning("[timezone_llm] LLM response not a valid IANA key for input=%r: %r", text, cleaned)
        return None
    except requests.exceptions.RequestException:
        logger.exception("[timezone_llm] LLM request failed for input=%r", text)
        return None
    except Exception:
        logger.exception("[timezone_llm] unexpected error resolving tz for input=%r", text)
        return None


def _tz_display_name(iana_key: str) -> str:
    """Return a readable timezone label, e.g. 'Eastern Time (ET)' or 'America/New_York'."""
    from zoneinfo import ZoneInfo
    from datetime import datetime, timezone as _tz
    try:
        abbr = datetime.now(_tz.utc).astimezone(ZoneInfo(iana_key)).strftime("%Z")
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
        resolved_tz = (resolve_timezone(user_input) if TIMEZONE_ALIAS_FAST_PATH else None) or _resolve_timezone_llm(user_input)
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
            "[get_telegram_updates] message %s: chat_id=%s message_id=%s text=%s",
            i + 1,
            cid,
            mid,
            text[:80] + "..." if len(text) > 80 else text,
        )

    next_offset = max_update_id + 1 if num_updates > 0 else offset
    return {
        "pending_messages": pending,
        "offset": next_offset,
    }


def process_message_node(state: AgentState) -> dict:
    """Node: process the first pending message — scheduling state machine or LLM Q&A."""
    pending = list(state.get("pending_messages") or [])
    if not pending:
        logger.info("[process_message] no pending messages, skipping")
        return {"pending_messages": []}

    conversations: dict = dict(state.get("conversations") or {})
    chat_id, reply_to_message_id, user_input = pending[0]
    remaining = len(pending) - 1
    logger.info(
        "[process_message] processing message (chat_id=%s, reply_to=%s, %s more). input=%s",
        chat_id,
        reply_to_message_id,
        remaining,
        user_input[:80] + "..." if len(user_input) > 80 else user_input,
    )

    key = str(chat_id)
    conv = ConversationState.from_dict(conversations.get(key, {}))

    # Route to scheduling if already in a scheduling flow or intent detected
    in_scheduling_flow = conv.mode not in (SchedulingMode.IDLE, SchedulingMode.CANCELLED)
    if in_scheduling_flow or _is_scheduling_intent(user_input):
        logger.info("[process_message] scheduling path (mode=%s)", conv.mode.value)
        _handle_scheduling(conv, user_input, chat_id, reply_to_message_id)
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

        prompt_template = RECRUITER_PROMPT_PATH.read_text(encoding="utf-8")
        chat_history = _format_history(conv.history)
        prompt_content = prompt_template.format(
            context=context,
            user_input=user_input,
            chat_history=chat_history,
        )
        with tracing_context(
            enabled=is_langsmith_enabled(),
            project_name=LANGSMITH_PROJECT,
            metadata={
                "context_length": len(context),
                "context_retrieved": context[:6000],
            },
        ):
            try:
                raw = _call_llm(prompt_content)
                reply = _strip_reasoning(raw)
                logger.info("[process_message] LLM reply length=%s chars", len(reply))
            except Exception as e:
                reply = "Sorry, there was an error. I couldn't process that."
                logger.exception("[process_message] LLM request failed: %s", e)
                if isinstance(
                    e,
                    (requests.exceptions.ConnectTimeout, requests.exceptions.ConnectionError),
                ):
                    logger.error(
                        "[process_message] Cannot reach LLM at %s. "
                        "If Ollama is on the host and the app runs in Docker, bind Ollama on all interfaces "
                        "(e.g. set OLLAMA_HOST=0.0.0.0:11434 for the Ollama service, restart Ollama), "
                        "then check on the host: curl -sS http://127.0.0.1:11434/api/tags",
                        LLM_API_URL,
                    )
            _send_telegram(chat_id, reply, reply_to_message_id)

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
