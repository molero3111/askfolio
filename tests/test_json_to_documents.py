from ingest.json_to_documents import _personal_info_doc


def test_personal_info_allowlist():
    doc = _personal_info_doc({
        "name": "Jane Doe", "title": "Engineer", "location": "Nowhere",
        "email": "jane.doe@example.com", "phone": "+99 12 3456 7890",
        "summary": "Builds things.", "links": {"github": "https://github.com/janedoe"},
        "passport": "X123",
    })
    text = doc.page_content
    assert "Jane Doe" in text and "https://github.com/janedoe" in text and "Builds things." in text
    for leaked in ("jane.doe@example.com", "3456", "Email:", "Phone:", "X123"):
        assert leaked not in text


def _write_projects(tmp_path, names):
    import json

    path = tmp_path / "projects.json"
    path.write_text(json.dumps({"github_projects": [{"name": n, "url": f"https://example.com/{n}"} for n in names]}))
    return path


def _repo_names(docs):
    return {d.metadata["repo"] for d in docs if d.metadata.get("section") == "github_project"}


def test_github_allowlist_keeps_only_listed_repos(tmp_path):
    from ingest.json_to_documents import documents_from_github_projects

    path = _write_projects(tmp_path, ["alpha", "beta", "gamma"])
    docs = documents_from_github_projects(path, allowlist=["alpha", "gamma"])
    assert _repo_names(docs) == {"alpha", "gamma"}
    index = " ".join(d.page_content for d in docs if d.metadata["section"].startswith("github_index"))
    assert "beta" not in index


def test_github_allowlist_empty_keeps_everything(tmp_path):
    from ingest.json_to_documents import documents_from_github_projects

    path = _write_projects(tmp_path, ["alpha", "beta"])
    assert _repo_names(documents_from_github_projects(path)) == {"alpha", "beta"}
    assert _repo_names(documents_from_github_projects(path, allowlist=[])) == {"alpha", "beta"}


def test_github_allowlist_is_case_insensitive(tmp_path):
    from ingest.json_to_documents import documents_from_github_projects

    path = _write_projects(tmp_path, ["Alpha", "beta"])
    assert _repo_names(documents_from_github_projects(path, allowlist=["alpha"])) == {"Alpha"}


def test_parse_name_list():
    from app.config import _name_list

    assert _name_list("") == []
    assert _name_list(" alpha, Beta ,,gamma ") == ["alpha", "Beta", "gamma"]


def test_missing_repos_reports_names_absent_from_the_file(tmp_path):
    from ingest.json_to_documents import documents_from_github_projects, missing_repos

    path = _write_projects(tmp_path, ["Alpha", "beta"])
    allowlist = ["alpha", "typo-repo"]
    docs = documents_from_github_projects(path, allowlist=allowlist)
    assert missing_repos(docs, allowlist) == ["typo-repo"]
    assert missing_repos(docs, ["ALPHA"]) == []
