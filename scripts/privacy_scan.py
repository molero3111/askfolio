"""Fail CI when a tracked file contains a term from the PRIVACY_DENYLIST secret.

Matched terms are never printed; only ``path:line`` locations are reported.
"""

import os
import re
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path


def scan(files: Iterable[Path], terms: list[str]) -> list[tuple[str, int]]:
    """Return (path, line_no) for every line containing any term, case-insensitive."""
    needles = [t.lower() for t in terms if t.strip()]
    hits: list[tuple[str, int]] = []
    if not needles:
        return hits
    for path in files:
        try:
            text = Path(path).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            lowered = line.lower()
            if any(n in lowered for n in needles):
                hits.append((str(path), line_no))
    return hits


def tracked_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "-z"], check=True, capture_output=True, text=True
    ).stdout
    return [Path(p) for p in out.split("\0") if p]


def main() -> int:
    raw = os.environ.get("PRIVACY_DENYLIST", "")
    terms = [t.strip() for t in re.split(r"[,\r\n]+", raw)]
    terms = [t for t in terms if t]
    if raw and not terms:
        print("::error::PRIVACY_DENYLIST is set but contains no usable terms")
        return 2
    if not terms:
        print("::notice::PRIVACY_DENYLIST not set; privacy scan skipped")
        return 0
    hits = scan(tracked_files(), terms)
    for path, line_no in hits:
        print(f"{path}:{line_no}")
    if hits:
        print(f"Privacy scan failed: {len(hits)} hit(s)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
