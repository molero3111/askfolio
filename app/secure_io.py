"""Atomic, owner-only file writes for secrets and small state files."""
import os
import tempfile
from pathlib import Path


def atomic_write_secret(path: Path, text: str) -> None:
    """Write text to path atomically with 0600 perms (dir created 0700).

    A temp file in the same directory is fsynced then os.replace'd, so a crash
    never leaves a truncated file. Raises OSError on failure, after removing
    the temp file and leaving any existing file intact.
    """
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
