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


def test_count_obfuscated_email_rows():
    from scripts.check_index_pii import count_obfuscated_email_rows

    rows = [
        "reach jane.doe at example dot com",
        "jane.doe [at] example.com",
        "jane.doe%40example.com",
        "jane.doe&#64;example.com",
        "jane.doe＠example.com",
        "unrelated jane.doe mentioned without a domain",
        "jane dot doe at example dot com",
        "jane . doe @ example . com",
        "someone.else@example.com",
    ]
    assert count_obfuscated_email_rows(rows, ["jane.doe@example.com"]) == 7
    assert count_obfuscated_email_rows([], ["jane.doe@example.com"]) == 0


def test_candidate_query_uses_local_part_ilike():
    from scripts.check_index_pii import build_candidate_query, email_local_part

    assert email_local_part("Jane.Doe@example.com") == "Jane.Doe"
    sql = build_candidate_query('we"ird')
    assert '"we""ird"' in sql and "ILIKE" in sql and "$1" in sql


def test_email_candidate_key_uses_first_dot_segment():
    from scripts.check_index_pii import email_candidate_key

    assert email_candidate_key(" jane.doe@example.com") == "jane"


def test_phone_variants_adds_national_form_for_long_numbers():
    from scripts.check_index_pii import phone_variants

    assert phone_variants("991234567890") == [
        "991234567890", "1234567890", "01234567890", "234567890", "0234567890",
    ]
    assert phone_variants("1234567890") == ["1234567890", "234567890", "0234567890"]
    # Fake 11-digit number: 2-digit country code + 9-digit national number.
    assert "123456789" in phone_variants("99123456789")
    assert "0123456789" in phone_variants("99123456789")


def test_build_queries_one_bind_param_per_phone_variant():
    _, sql = build_queries("t", 2)
    assert "$1" in sql and "$2" in sql and "$3" not in sql
    assert sql.count("LIKE") == 4


def test_main_returns_2_on_unreadable_knowledge_json(monkeypatch, tmp_path, capsys):
    m = _setup(monkeypatch, tmp_path, {})
    (tmp_path / "cv.json").write_text("{not json")
    assert m.main([]) == 2
    out = capsys.readouterr().out
    assert out.strip() == "cannot read knowledge JSON: JSONDecodeError"

    monkeypatch.setattr(m, "resolve_path", lambda _p: tmp_path / "missing.json")
    assert m.main([]) == 2
    assert "cannot read knowledge JSON: FileNotFoundError" in capsys.readouterr().out
