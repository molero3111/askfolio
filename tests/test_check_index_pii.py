from scripts.check_index_pii import build_queries, split_contacts


def test_split_contacts_separates_email_and_digit_phone():
    emails, phones = split_contacts(["jane@example.com", "+99 (12) 3456-7890", "  "])
    assert emails == ["jane@example.com"]
    assert phones == ["991234567890"]


def test_split_contacts_empty():
    assert split_contacts([]) == ([], [])


def test_build_queries_quotes_table_and_uses_params():
    email_sql, phone_sql = build_queries('we"ird')
    assert '"we""ird"' in email_sql and "$1" in email_sql and "$1" in phone_sql
    assert "langchain_metadata::text" in phone_sql


def _setup(monkeypatch, tmp_path, cv):
    from scripts import check_index_pii as m

    kb = tmp_path / "cv.json"
    kb.write_text(__import__("json").dumps(cv))
    monkeypatch.setattr(m, "DB_CONNECTION_URL_ASYNC", "postgresql+asyncpg://u:p@h/d")
    monkeypatch.setattr(m, "resolve_path", lambda _p: kb)
    return m


def test_main_returns_2_when_no_contacts(monkeypatch, tmp_path, capsys):
    m = _setup(monkeypatch, tmp_path, {"personal_info": {}})
    assert m.main([]) == 2
    assert "nothing to check" in capsys.readouterr().out


def test_main_returns_2_on_db_error_and_prints_class_only(monkeypatch, tmp_path, capsys):
    m = _setup(monkeypatch, tmp_path, {"personal_info": {"email": "jane@example.com"}})

    async def boom(*a, **k):
        raise ConnectionRefusedError("secret query context")

    monkeypatch.setattr(m, "_count", boom)
    assert m.main([]) == 2
    out = capsys.readouterr().out
    assert "ConnectionRefusedError" in out and "secret" not in out
