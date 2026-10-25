import os
import subprocess
import sys
import time

from app import healthcheck, heartbeat


def test_touch_creates_file(tmp_path, monkeypatch):
    path = tmp_path / "hb"
    monkeypatch.setenv("HEARTBEAT_FILE", str(path))
    heartbeat.touch()
    assert path.exists()


def test_healthcheck_fresh_stale_missing(tmp_path, monkeypatch):
    path = tmp_path / "hb"
    monkeypatch.setenv("HEARTBEAT_FILE", str(path))
    monkeypatch.setenv("GET_UPDATES_TIMEOUT", "60")
    assert healthcheck.main() == 1  # missing
    path.touch()
    assert healthcheck.main() == 0  # fresh
    old = time.time() - 200
    os.utime(path, (old, old))
    assert healthcheck.main() == 1  # stale


def test_healthcheck_is_stdlib_only():
    code = (
        "import sys, app.healthcheck; "
        "bad={'app.config','torch','langchain_core','requests'} & set(sys.modules); "
        "assert not bad, bad"
    )
    result = subprocess.run([sys.executable, "-c", code], check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
