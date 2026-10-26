import json

import pytest

import ingest.ingest_pgvector as ingest


def test_missing_allowlisted_repo_exits_before_touching_the_table(tmp_path, monkeypatch):
    kb = tmp_path / "cv.json"
    kb.write_text(json.dumps({"personal_info": {"name": "Test Person"}}))
    gh = tmp_path / "projects.json"
    gh.write_text(json.dumps({"github_projects": [{"name": "alpha", "url": "https://example.com/alpha"}]}))

    def fail(*_args, **_kwargs):
        raise AssertionError("the table must not be touched")

    monkeypatch.setattr(ingest, "INGEST_KNOWLEDGE_JSON", str(kb))
    monkeypatch.setattr(ingest, "INGEST_GITHUB_JSON", str(gh))
    monkeypatch.setattr(ingest, "INGEST_GITHUB_REPOS", ["alpha", "typo-repo"])
    monkeypatch.setattr(ingest, "DB_CONNECTION_URL_ASYNC", "postgresql+psycopg://u:p@localhost/db")
    monkeypatch.setattr(ingest, "_clear_collection_rows", fail)
    monkeypatch.setattr(ingest, "split_documents_semantically", fail)
    monkeypatch.setattr("sys.argv", ["ingest_pgvector.py", "--reinstall"])

    with pytest.raises(SystemExit) as exc:
        ingest.main()
    assert exc.value.code == 1
