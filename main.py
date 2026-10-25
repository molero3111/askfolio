"""Run the Telegram RAG agent: poll for updates and process with LangGraph."""
import logging
import os
import time
import traceback

from dotenv import load_dotenv

load_dotenv()

from app import heartbeat
from app.config import (
    LANGSMITH_API_KEY,
    LANGSMITH_ENABLED,
    LANGSMITH_ENDPOINT,
    LANGSMITH_PROJECT,
    LANGSMITH_RUNTIME_FLAG,
    LLM_API_KEY,
    LLM_MODEL,
    POLL_INTERVAL_SECONDS,
    TELEGRAM_BOT_TOKEN,
)
from app.graph import build_graph
from app.log_redact import install_redaction
from app.offset import load_offset, save_offset
from app.rag import get_vectorstore

if LANGSMITH_ENABLED:
    # Disable global auto-tracing (LangGraph node-level noise on empty polls).
    # We only emit manual spans via tracing_context() in llm_reply flow.
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY
    os.environ["LANGSMITH_PROJECT"] = LANGSMITH_PROJECT
    os.environ["LANGSMITH_ENDPOINT"] = LANGSMITH_ENDPOINT
    os.environ[LANGSMITH_RUNTIME_FLAG] = "true"
else:
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ[LANGSMITH_RUNTIME_FLAG] = "false"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
install_redaction([TELEGRAM_BOT_TOKEN, LLM_API_KEY])


def _is_langsmith_exception(exc: BaseException) -> bool:
    """Best-effort detection for tracing backend/runtime errors."""
    stack = traceback.format_exception(type(exc), exc, exc.__traceback__)
    text = "".join(stack).lower()
    if "langsmith" in text or "smith.langchain.com" in text:
        return True
    cause = exc.__cause__ or exc.__context__
    if cause is None:
        return False
    return _is_langsmith_exception(cause)


def startup() -> None:
    """Validate config, prove the DB and embedding model work, then mark alive."""
    if not TELEGRAM_BOT_TOKEN:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN in .env")
    if not LLM_MODEL:
        raise SystemExit("Set LLM_MODEL in .env (e.g. an OpenRouter model ID)")
    get_vectorstore()
    heartbeat.touch()


def _resume_state(state: dict) -> dict:
    """State to continue from after a failed cycle.

    The on-disk offset is already advanced past the fetched batch, so resuming from it
    skips a crashing message instead of replaying it (duplicate replies) forever.
    """
    disk_offset = load_offset()
    return {
        **state,
        "offset": disk_offset if disk_offset is not None else state.get("offset"),
        "pending_messages": [],
    }


def run_once(graph, state: dict, sleep=time.sleep) -> dict:
    """Run one poll cycle. Never raises: on failure resume from the disk offset."""
    logger = logging.getLogger(__name__)
    try:
        new_state = graph.invoke(state)
    except Exception as exc:
        if _is_langsmith_exception(exc):
            logger.warning(
                "LangSmith tracing error detected; disabling tracing and continuing. Error: %s",
                exc,
            )
            os.environ[LANGSMITH_RUNTIME_FLAG] = "false"
            return _resume_state(state)
        logger.exception("Poll cycle failed (%s); retrying in 5s", type(exc).__name__)
        sleep(5)
        return _resume_state(state)
    if new_state.get("offset") is not None:
        save_offset(new_state["offset"])
    heartbeat.touch()
    return new_state


def main():
    logger = logging.getLogger(__name__)
    logger.info(
        "LangSmith tracing: %s (manual spans only, project=%s)",
        "enabled" if LANGSMITH_ENABLED else "disabled",
        LANGSMITH_PROJECT,
    )
    startup()
    graph = build_graph()
    state: dict = {"offset": load_offset(), "pending_messages": [], "conversations": {}}
    while True:
        state = run_once(graph, state)
        if POLL_INTERVAL_SECONDS:
            time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
