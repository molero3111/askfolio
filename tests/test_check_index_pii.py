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
