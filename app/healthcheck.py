"""Container healthcheck: exit 0 if the heartbeat is fresh, else 1.

Stdlib only (os, sys, time, pathlib): it runs every 10s under a memory limit, so
it must not import app.config, requests, torch or LangChain.
"""
import os
import sys
import time
from pathlib import Path


def main() -> int:
    path = Path(os.environ.get("HEARTBEAT_FILE", "/tmp/heartbeat"))
    max_age = 3 * int(os.environ.get("GET_UPDATES_TIMEOUT", "60"))
    try:
        age = time.time() - path.stat().st_mtime
    except OSError:
        return 1
    return 0 if age < max_age else 1


if __name__ == "__main__":
    sys.exit(main())
