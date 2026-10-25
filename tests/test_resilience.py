import logging
import os
import subprocess
import sys

import pytest
import requests

import main
from app import graph


def test_send_telegram_swallows_errors(monkeypatch, caplog):
    def boom(*a, **k):
        raise requests.HTTPError("500")

    monkeypatch.setattr(graph, "send_message", boom)
    with caplog.at_level(logging.WARNING):
        graph._send_telegram(1, "hi", 2)
    assert "500" in caplog.text


class _BoomGraph:
    def invoke(self, state):
        raise RuntimeError("boom")


class _OkGraph:
    def invoke(self, state):
        return {**state, "offset": 7}


def test_run_once_survives_exception(monkeypatch, tmp_path):
    monkeypatch.setenv("HEARTBEAT_FILE", str(tmp_path / "hb"))
    sleeps = []
    prev = {"offset": 1, "pending_messages": [], "conversations": {}}
    assert main.run_once(_BoomGraph(), prev, sleep=sleeps.append) is prev
    assert sleeps == [5]
    assert not (tmp_path / "hb").exists()


def test_run_once_touches_heartbeat_on_success(monkeypatch, tmp_path):
    monkeypatch.setenv("HEARTBEAT_FILE", str(tmp_path / "hb"))
    monkeypatch.setattr(main, "save_offset", lambda o: None)
    out = main.run_once(_OkGraph(), {"offset": 1}, sleep=lambda s: None)
    assert out["offset"] == 7
    assert (tmp_path / "hb").exists()


def test_startup_without_calendar(monkeypatch, tmp_path):
    for k in list(os.environ):
        if k.startswith("GOOGLE_"):
            monkeypatch.delenv(k)
    (tmp_path / "secrets").mkdir()
    monkeypatch.setenv("BOOKING_ENABLED", "false")
    monkeypatch.setenv("HEARTBEAT_FILE", str(tmp_path / "hb"))
    monkeypatch.setattr(main, "TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setattr(main, "LLM_MODEL", "some/model")
    called = []
    monkeypatch.setattr(main, "get_vectorstore", lambda: called.append(1))
    main.startup()
    assert called == [1]
    assert (tmp_path / "hb").exists()


def test_startup_requires_token_and_model(monkeypatch):
    monkeypatch.setattr(main, "TELEGRAM_BOT_TOKEN", "")
    with pytest.raises(SystemExit):
        main.startup()
    monkeypatch.setattr(main, "TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setattr(main, "LLM_MODEL", "")
    with pytest.raises(SystemExit):
        main.startup()


def test_import_graph_and_calendar_does_not_touch_secrets():
    code = (
        "import builtins, sys\n"
        "orig = builtins.open\n"
        "def guard(f, *a, **k):\n"
        "    assert 'secrets' not in str(f), f\n"
        "    return orig(f, *a, **k)\n"
        "builtins.open = guard\n"
        "import app.graph, app.calendar_client\n"
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith("GOOGLE_")}
    r = subprocess.run([sys.executable, "-c", code], check=False, capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
