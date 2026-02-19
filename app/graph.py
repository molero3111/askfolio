import logging
import re
import requests
from typing import TypedDict, Sequence

from langgraph.graph import StateGraph, START, END

from app.config import LLM_API_KEY, LLM_API_URL, LLM_MODEL, LLM_REQUEST_TIMEOUT, RECRUITER_PROMPT_PATH
from app.rag import get_relevant_context
from app.telegram_client import get_updates, send_message

logger = logging.getLogger(__name__)


def _strip_reasoning(text: str) -> str:
    """Remove <think>...</think> blocks and similar reasoning so only the final answer is sent to Telegram."""
    # Remove <think>...</think> (and unclosed <think> at end)
    out = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    out = re.sub(r"<think>[\s\S]*", "", out, flags=re.IGNORECASE)
    out = out.strip()
    return out if out else "(No reply could be extracted.)"


class AgentState(TypedDict):
    offset: int | None
    pending_messages: Sequence[tuple[int, str]]


def get_telegram_updates_node(state: AgentState) -> dict:
    """Node: fetch new Telegram messages and return (chat_id, text) list."""
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
        pending.append((msg["chat"]["id"], text))

    logger.info("[get_telegram_updates] new messages with text=%s", len(pending))
    for i, (cid, text) in enumerate(pending):
        logger.info("[get_telegram_updates] message %s: chat_id=%s, text=%s", i + 1, cid, text[:80] + "..." if len(text) > 80 else text)

    # Only advance offset when we received updates (ack to Telegram so they are not returned again)
    next_offset = max_update_id + 1 if num_updates > 0 else offset
    return {
        "pending_messages": pending,
        "offset": next_offset,
    }


def llm_reply_node(state: AgentState) -> dict:
    """Node: for the first pending message, retrieve context, call LLM, send reply."""
    pending = list(state.get("pending_messages") or [])
    if not pending:
        logger.info("[llm_reply] no pending messages, skipping")
        return {"pending_messages": []}

    chat_id, user_input = pending[0]
    remaining = len(pending) - 1
    logger.info("[llm_reply] processing 1 message (chat_id=%s, %s more in queue). user_input=%s", chat_id, remaining, user_input[:80] + "..." if len(user_input) > 80 else user_input)

    context = get_relevant_context(user_input)
    logger.info("[llm_reply] retrieved context length=%s chars", len(context))

    prompt_template = RECRUITER_PROMPT_PATH.read_text(encoding="utf-8")
    prompt_content = prompt_template.format(context=context, user_input=user_input)
    headers = {"Content-Type": "application/json"}
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt_content}],
    }
    try:
        r = requests.post(LLM_API_URL, headers=headers, json=payload, timeout=LLM_REQUEST_TIMEOUT)
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"].strip()
        reply = _strip_reasoning(raw)
        logger.info("[llm_reply] LLM reply length=%s chars, sending to chat_id=%s", len(reply), chat_id)
    except Exception as e:
        reply = f"Sorry, I couldn't process that: {e!s}"
        logger.exception("[llm_reply] LLM request failed: %s", e)
    send_message(chat_id, reply)
    return {"pending_messages": pending[1:]}


def should_continue(state: AgentState) -> str:
    return "llm_reply" if state.get("pending_messages") else "end"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("get_telegram_updates", get_telegram_updates_node)
    graph.add_node("llm_reply", llm_reply_node)
    graph.add_edge(START, "get_telegram_updates")
    graph.add_conditional_edges(
        "get_telegram_updates", should_continue, {"llm_reply": "llm_reply", "end": END}
    )
    graph.add_conditional_edges(
        "llm_reply", should_continue, {"llm_reply": "llm_reply", "end": END}
    )
    return graph.compile()
