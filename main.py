"""Run the Telegram RAG agent: poll for updates and process with LangGraph."""
import logging
import os
import time
import traceback

from dotenv import load_dotenv

load_dotenv()

from app.config import (
    LANGSMITH_API_KEY,
    LANGSMITH_ENABLED,
    LANGSMITH_ENDPOINT,
    LANGSMITH_PROJECT,
    LANGSMITH_RUNTIME_FLAG,
    POLL_INTERVAL_SECONDS,
    TELEGRAM_BOT_TOKEN,
)
from app.graph import build_graph
from app.offset import load_offset, save_offset

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


def main():
    logger = logging.getLogger(__name__)
    logger.info(
        "LangSmith tracing: %s (manual spans only, project=%s)",
        "enabled" if LANGSMITH_ENABLED else "disabled",
        LANGSMITH_PROJECT,
    )
    if not TELEGRAM_BOT_TOKEN:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN in .env")
    graph = build_graph()
    state: dict = {"offset": load_offset(), "pending_messages": [], "conversations": {}}
    while True:
        try:
            state = graph.invoke(state)
        except Exception as exc:
            if _is_langsmith_exception(exc):
                logger.warning(
                    "LangSmith tracing error detected; disabling tracing and continuing. Error: %s",
                    exc,
                )
                os.environ[LANGSMITH_RUNTIME_FLAG] = "false"
                continue
            raise
        if state.get("offset") is not None:
            save_offset(state["offset"])
        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
