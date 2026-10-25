import logging
import os
import subprocess
import sys

import pytest
import requests

import main
from app import graph, log_redact, telegram_client


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
    out = main.run_once(_BoomGraph(), prev, sleep=sleeps.append)
    assert out["conversations"] == prev["conversations"]
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


def test_run_once_skips_poison_update(monkeypatch, tmp_path):
    monkeypatch.setenv("HEARTBEAT_FILE", str(tmp_path / "hb"))
    monkeypatch.setattr(main, "load_offset", lambda: 99)
    prev = {"offset": 1, "pending_messages": [(1, 2, "x")], "conversations": {"1": {}}}
    out = main.run_once(_BoomGraph(), prev, sleep=lambda s: None)
    assert out["offset"] == 99
    assert out["pending_messages"] == []
    assert out["conversations"] == {"1": {}}


def test_run_once_failure_does_not_touch_heartbeat(monkeypatch):
    touched = []
    monkeypatch.setattr(main.heartbeat, "touch", lambda: touched.append(1))
    main.run_once(_BoomGraph(), {"offset": None}, sleep=lambda s: None)
    assert touched == []


def test_startup_propagates_vectorstore_failure(monkeypatch, tmp_path):
    monkeypatch.setenv("HEARTBEAT_FILE", str(tmp_path / "hb"))
    monkeypatch.setattr(main, "TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setattr(main, "LLM_MODEL", "m")

    def boom():
        raise RuntimeError("db down")

    monkeypatch.setattr(main, "get_vectorstore", boom)
    with pytest.raises(RuntimeError):
        main.startup()
    assert not (tmp_path / "hb").exists()


TOKEN = "123456789:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"


class _Resp:
    def __init__(self, data, status=200):
        self._data, self.status_code = data, status

    def json(self):
        return self._data


def _get_returning(resp):
    return lambda *a, **k: resp


def test_get_updates_connection_error_wrapped_without_token(monkeypatch):
    def boom(*a, **k):
        raise requests.ConnectionError(f"Max retries exceeded with url: /bot{TOKEN}/getUpdates")

    monkeypatch.setattr(telegram_client.requests, "get", boom)
    with pytest.raises(telegram_client.TelegramError) as ei:
        telegram_client.get_updates()
    assert TOKEN not in str(ei.value)
    assert ei.value.__cause__ is None and ei.value.__suppress_context__


def test_get_updates_http_error_without_token(monkeypatch):
    monkeypatch.setattr(telegram_client.requests, "get", _get_returning(_Resp({}, 401)))
    with pytest.raises(telegram_client.TelegramError, match="HTTP 401") as ei:
        telegram_client.get_updates()
    assert TOKEN not in str(ei.value)


def test_get_updates_409_conflict_message(monkeypatch):
    monkeypatch.setattr(telegram_client.requests, "get", _get_returning(_Resp({}, 409)))
    with pytest.raises(telegram_client.TelegramError) as ei:
        telegram_client.get_updates()
    assert str(ei.value) == (
        "getUpdates conflict (HTTP 409): another poller or a webhook is using this bot token"
    )


def test_get_updates_raises_on_not_ok(monkeypatch):
    monkeypatch.setattr(
        telegram_client.requests, "get", _get_returning(_Resp({"ok": False}))
    )
    with pytest.raises(telegram_client.TelegramError):
        telegram_client.get_updates()


def test_send_message_error_without_token(monkeypatch):
    def boom(*a, **k):
        raise requests.ConnectionError(f"url /bot{TOKEN}/sendMessage")

    monkeypatch.setattr(telegram_client.requests, "post", boom)
    with pytest.raises(telegram_client.TelegramError) as ei:
        telegram_client.send_message(1, "x")
    assert TOKEN not in str(ei.value)


@pytest.fixture
def restore_root_logging():
    root = logging.getLogger()
    handlers, filters = list(root.handlers), list(root.filters)
    handler_filters = {h: list(h.filters) for h in handlers}
    yield
    root.handlers[:] = handlers
    root.filters[:] = filters
    for h, fl in handler_filters.items():
        h.filters[:] = fl


def test_run_once_log_never_contains_token(monkeypatch, caplog, tmp_path, restore_root_logging):
    monkeypatch.setenv("HEARTBEAT_FILE", str(tmp_path / "hb"))

    def boom(*a, **k):
        raise requests.ConnectionError(f"Max retries exceeded with url: /bot{TOKEN}/getUpdates")

    monkeypatch.setattr(telegram_client.requests, "get", boom)

    class G:
        def invoke(self, state):
            telegram_client.get_updates()

    log_redact.install_redaction([TOKEN])
    with caplog.at_level(logging.INFO):
        main.run_once(G(), {"offset": None}, sleep=lambda s: None)
    assert "Poll cycle failed" in caplog.text
    assert TOKEN not in caplog.text
    assert TOKEN not in "".join(r.getMessage() + (r.exc_text or "") for r in caplog.records)


def test_filter_redacts_args_and_exc_info():
    flt = log_redact.RedactionFilter(["sekret-key", "  ", ""])
    try:
        raise ValueError(f"bad url /bot{TOKEN}/x and sekret-key")
    except ValueError:
        import sys

        rec = logging.LogRecord(
            "t", logging.ERROR, __file__, 1, "got %s", ("sekret-key",), sys.exc_info()
        )
    flt.filter(rec)
    out = logging.Formatter("%(message)s").format(rec)
    assert "sekret-key" not in out
    assert TOKEN not in out
    assert "got ***" in out
    assert "bot***" in out


def test_get_updates_read_timeout_returns_empty(monkeypatch):
    def slow(*a, **k):
        raise requests.ReadTimeout("slow")

    monkeypatch.setattr(telegram_client.requests, "get", slow)
    assert telegram_client.get_updates() == []


def test_filter_survives_malformed_log_call():
    flt = log_redact.RedactionFilter([])
    rec = logging.LogRecord("t", logging.INFO, __file__, 1, f"two %s %s /bot{TOKEN}", ("one",), None)
    assert flt.filter(rec) is True
    out = logging.Formatter("%(message)s").format(rec)
    assert TOKEN not in out
    assert "bot***" in out


def test_langsmith_error_resumes_from_disk_offset(monkeypatch, tmp_path):
    monkeypatch.setenv("HEARTBEAT_FILE", str(tmp_path / "hb"))
    monkeypatch.setattr(main, "load_offset", lambda: 99)
    monkeypatch.setenv(main.LANGSMITH_RUNTIME_FLAG, "true")  # restored on teardown

    class LangsmithGraph:
        def invoke(self, state):
            raise RuntimeError("langsmith upload failed")

    prev = {"offset": 1, "pending_messages": [(1, 2, "x")], "conversations": {"1": {}}}
    out = main.run_once(LangsmithGraph(), prev, sleep=lambda s: None)
    assert out["offset"] == 99
    assert out["pending_messages"] == []
    assert out["conversations"] == {"1": {}}
    assert os.environ[main.LANGSMITH_RUNTIME_FLAG] == "false"


LS_KEY = "lsv2_pt_" + "f" * 32


def test_langsmith_shaped_key_is_redacted(caplog, restore_root_logging):
    log_redact.install_redaction([LS_KEY])
    with caplog.at_level(logging.INFO):
        logging.getLogger("t").error("upload failed with key %s", LS_KEY)
    assert LS_KEY not in caplog.text
    assert "***" in caplog.text


def test_captured_warning_is_redacted(caplog, restore_root_logging):
    import warnings

    log_redact.install_redaction([LS_KEY])
    # importing main already enabled capture at collection time; reinstall it fresh
    logging.captureWarnings(False)
    logging.captureWarnings(True)
    try:
        with caplog.at_level(logging.WARNING):
            # What warnings.warn() ends up calling once capture is on.
            warnings.showwarning(
                UserWarning(f"deprecated call with {LS_KEY} and /bot{TOKEN}/x"),
                UserWarning,
                "somefile.py",
                1,
            )
    finally:
        logging.captureWarnings(False)
    assert "deprecated call" in caplog.text
    assert LS_KEY not in caplog.text
    assert TOKEN not in caplog.text


def test_uncaught_exception_is_logged_redacted(caplog, restore_root_logging, monkeypatch):
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)  # restored on teardown
    log_redact.install_redaction([LS_KEY])
    log_redact.install_excepthook()
    try:
        raise RuntimeError(f"boom {LS_KEY}")
    except RuntimeError:
        with caplog.at_level(logging.CRITICAL):
            sys.excepthook(*sys.exc_info())
    assert "Uncaught exception" in caplog.text
    assert LS_KEY not in caplog.text
