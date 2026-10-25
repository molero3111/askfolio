"""Telegram update-offset persistence shared by main.py and the graph."""
import logging
import os
import tempfile
from pathlib import Path

from app import config

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
    """Atomically persist the offset (temp file in the same dir + os.replace)."""
    path = offset_path()
    tmp_name = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
        with os.fdopen(fd, "w") as f:
            f.write(str(offset))
        os.replace(tmp_name, path)
    except OSError:
        logger.warning("Could not save offset to %s", path)
        if tmp_name:
            Path(tmp_name).unlink(missing_ok=True)
