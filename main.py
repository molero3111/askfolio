"""Run the Telegram RAG agent: poll for updates and process with LangGraph."""
import logging
import os
import signal
import time

from app import heartbeat
from app.config import (
    ANALYTICS_SALT,
    LANGSMITH_API_KEY,
    LANGSMITH_ENABLED,
    LANGSMITH_ENDPOINT,
    LANGSMITH_PROJECT,
    LANGSMITH_RUNTIME_FLAG,
    LLM_API_KEY,
    LLM_MODEL,
    POLL_INTERVAL_SECONDS,
    POSTHOG_KEY,
    TELEGRAM_BOT_TOKEN,
)
from app.graph import build_graph
from app.log_redact import install_excepthook, install_redaction
from app.observability import is_langsmith_exception
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


def configure_process() -> None:
    """Install logging, secret redaction and the excepthook. Called once from main()."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Google OAuth client secret / refresh token are only read lazily from secrets/ when
    # booking runs, so they are not in memory at startup and are not listed here.
    install_redaction([TELEGRAM_BOT_TOKEN, LLM_API_KEY, LANGSMITH_API_KEY, POSTHOG_KEY, ANALYTICS_SALT])
    install_excepthook()
    logging.captureWarnings(True)  # route warnings.warn through the filtered handlers


def handle_sigterm(signum, frame) -> None:
    """Exit cleanly on SIGTERM (docker stop).

    The offset is saved when a batch is fetched, so any messages of the current batch that
    haven't been answered yet are dropped, not replayed (at-most-once delivery).
    """
    raise SystemExit(0)


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
        if is_langsmith_exception(exc):
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
    configure_process()
    signal.signal(signal.SIGTERM, handle_sigterm)
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
