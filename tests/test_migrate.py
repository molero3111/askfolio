import pytest

from ingest.migrate import with_retry


def test_retries_connect_errors_then_succeeds():
    calls, sleeps = [], []

    def fake_connect():
        calls.append(1)
        if len(calls) <= 2:
            raise ConnectionRefusedError("refused")
        return "ok"

    assert with_retry(fake_connect, sleep=sleeps.append) == "ok"
    assert len(calls) == 3
    assert sleeps == [1, 2]


def test_gives_up_after_attempts():
    sleeps = []

    def always():
        raise ConnectionRefusedError("refused")

    with pytest.raises(ConnectionRefusedError):
        with_retry(always, attempts=3, sleep=sleeps.append)
    assert sleeps == [1, 2]


def test_non_connect_errors_not_retried():
    sleeps = []

    def bad():
        raise ValueError("boom")

    with pytest.raises(ValueError):
        with_retry(bad, sleep=sleeps.append)
    assert sleeps == []


@pytest.mark.parametrize("name", ["rag_docs", "_t", "Docs2", "a"])
def test_valid_table_names(name):
    from ingest.migrate import valid_table_name

    assert valid_table_name(name)


@pytest.mark.parametrize("name", ["", "1docs", "a-b", "a b", 'a"; DROP TABLE x;--', "schema.table", "a\n", "é"])
def test_invalid_table_names(name):
    from ingest.migrate import valid_table_name

    assert not valid_table_name(name)


def test_main_exits_2_on_bad_table_name(monkeypatch, capsys):
    import ingest.migrate as m

    monkeypatch.setattr(m, "DB_CONNECTION_URL_ASYNC", "postgresql+asyncpg://x")
    monkeypatch.setattr(m, "PGVECTOR_COLLECTION_NAME", "bad-name")
    assert m.main() == 2
    assert "PGVECTOR_COLLECTION_NAME" in capsys.readouterr().out


# --- _is_connect_error ---

def test_is_connect_error_classification():
    import asyncpg
    import sqlalchemy

    from ingest.migrate import _is_connect_error

    assert _is_connect_error(OSError("x"))
    assert _is_connect_error(ConnectionRefusedError())
    assert _is_connect_error(asyncpg.CannotConnectNowError())
    assert _is_connect_error(asyncpg.PostgresConnectionError())
    assert _is_connect_error(sqlalchemy.exc.OperationalError("s", {}, Exception("x")))
    assert not _is_connect_error(asyncpg.InvalidPasswordError())
    assert not _is_connect_error(sqlalchemy.exc.ProgrammingError("s", {}, Exception("x")))
    assert not _is_connect_error(ValueError("x"))


# --- main() with a fake engine ---

class _FakeEngine:
    def __init__(self, init_exc=None, close_exc=None):
        self.init_exc, self.close_exc, self.closed = init_exc, close_exc, False

    def init_vectorstore_table(self, **kw):
        if self.init_exc:
            raise self.init_exc

    async def close(self):
        self.closed = True
        if self.close_exc:
            raise self.close_exc

    def _run_as_sync(self, coro):
        import asyncio

        return asyncio.run(coro)


def _patch_main(monkeypatch, engine):
    import ingest.migrate as m

    monkeypatch.setattr(m, "DB_CONNECTION_URL_ASYNC", "postgresql+asyncpg://x")
    monkeypatch.setattr(m, "PGVECTOR_COLLECTION_NAME", "docs")
    monkeypatch.setattr(m, "_create_extension", lambda: None)
    monkeypatch.setattr(m.PGEngine, "from_connection_string", staticmethod(lambda url: engine))
    return m


def _programming_error(msg):
    import sqlalchemy

    return sqlalchemy.exc.ProgrammingError("stmt", {}, Exception(msg))


def test_already_exists_is_skipped(monkeypatch, capsys):
    eng = _FakeEngine(init_exc=_programming_error('relation "docs" already exists'))
    m = _patch_main(monkeypatch, eng)
    assert m.main() == 0
    assert "already exists, skipping" in capsys.readouterr().out
    assert eng.closed


def test_other_programming_error_propagates(monkeypatch):
    import sqlalchemy

    m = _patch_main(monkeypatch, _FakeEngine(init_exc=_programming_error("syntax error")))
    with pytest.raises(sqlalchemy.exc.ProgrammingError):
        m.main()


def test_close_failure_does_not_mask_success(monkeypatch, capsys):
    m = _patch_main(monkeypatch, _FakeEngine(close_exc=RuntimeError("boom postgresql://secret")))
    assert m.main() == 0
    out = capsys.readouterr().out
    assert "engine close failed: RuntimeError" in out
    assert "secret" not in out


def test_close_failure_does_not_replace_original_error(monkeypatch):
    import sqlalchemy

    eng = _FakeEngine(init_exc=_programming_error("syntax error"), close_exc=RuntimeError("boom"))
    m = _patch_main(monkeypatch, eng)
    with pytest.raises(sqlalchemy.exc.ProgrammingError):
        m.main()
