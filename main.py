"""Run the Telegram RAG agent: poll for updates and process with LangGraph."""
import logging
import time
from pathlib import Path

from dotenv import load_dotenv

from app.config import (
    POLL_INTERVAL_SECONDS,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_OFFSET_FILE,
)
from app.graph import build_graph

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


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
    if not TELEGRAM_BOT_TOKEN:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN in .env")
    graph = build_graph()
    state: dict = {"offset": _load_offset(), "pending_messages": []}
    while True:
        state = graph.invoke(state)
        if state.get("offset") is not None:
            _save_offset(state["offset"])
        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
