"""Telegram update-offset persistence shared by main.py and the graph."""
import logging
from pathlib import Path

from app import config
from app.secure_io import atomic_write_secret

logger = logging.getLogger(__name__)


def offset_path() -> Path:
    p = Path(config.TELEGRAM_OFFSET_FILE)
    return p if p.is_absolute() else Path.cwd() / p


def load_offset() -> int | None:
    try:
        raw = offset_path().read_text().strip()
        return int(raw) if raw else None
    except (ValueError, OSError):
        return None


def save_offset(offset: int) -> None:
    """Atomically persist the offset (see secure_io.atomic_write_secret)."""
    path = offset_path()
    try:
        atomic_write_secret(path, str(offset))
    except OSError:
        logger.warning("Could not save offset to %s", path, exc_info=True)
