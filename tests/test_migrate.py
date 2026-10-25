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
