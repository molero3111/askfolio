"""Liveness heartbeat. Stdlib only; reads HEARTBEAT_FILE directly (not via app.config).

Lives in the container's /tmp, not on the data volume, so a new container never
inherits an old container's fresh file during a deploy swap.
"""
import os
from pathlib import Path


def touch() -> None:
    path = Path(os.environ.get("HEARTBEAT_FILE", "/tmp/heartbeat"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()  # touch() updates the mtime of an existing file
