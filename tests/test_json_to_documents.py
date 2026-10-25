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
