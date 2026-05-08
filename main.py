"""Run the Telegram RAG agent: poll for updates and process with LangGraph."""
import logging
import os
import time
import traceback
from pathlib import Path

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
    TELEGRAM_OFFSET_FILE,
)
from app.graph import build_graph

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


def _offset_path() -> Path:
    p = Path(TELEGRAM_OFFSET_FILE)
    return p if p.is_absolute() else Path.cwd() / p


def _load_offset() -> int | None:
    path = _offset_path()
    if not path.exists():
        return None
    try:
        raw = path.read_text().strip()
        return int(raw) if raw else None
    except (ValueError, OSError):
        return None


def _save_offset(offset: int) -> None:
    path = _offset_path()
    try:
        path.write_text(str(offset))
    except OSError:
        logging.getLogger(__name__).warning("Could not save offset to %s", path)


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
    state: dict = {"offset": _load_offset(), "pending_messages": [], "conversations": {}}
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
            _save_offset(state["offset"])
        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
