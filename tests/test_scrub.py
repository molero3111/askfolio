import pytest
from langchain_core.documents import Document

from ingest.scrub import REPLACEMENT, contact_values, scrub_documents, scrub_text

PHONE = "+99 12 3456 7890"
EMAIL = "jane.doe@example.com"


@pytest.mark.parametrize(
    "src",
    [
        f"mail me: {EMAIL}",
        f"[Email](mailto:{EMAIL})",
        f"call {PHONE} now",
        "call +991234567890",
        "call (12) 3456-7890",
        "call 12.3456.7890",
    ],
)
def test_removes_contacts(src):
    out, kinds = scrub_text(src, exact=[EMAIL, PHONE])
    assert EMAIL not in out and "34567890" not in "".join(c for c in out if c.isdigit())
    assert kinds


@pytest.mark.parametrize(
    "keep",
    [
        "2015 - 2020",
        "Jan 2023 – Mar 2024",
        "2023-01-15",
        "https://www.udemy.com/certificate/UC-ad904fa5-36b2-4c03-aa30-3524904a63ab/",
        "https://github.com/janedoe/repo",
        "github: janedoe",
        "v1.2.3",
        "port 5432",
        "2018-2020 (2 years)",
        "Q1 2020 2021 2022 results",
        "2019\n2020\n2021",
        "Python 3.10 3.11 3.12 3.13",
        "192.168.100.200",
        "git clone git@github.com:janedoe/repo.git",
        "https://janedoe@host.com/x",
    ],
)
def test_keeps_non_contacts(keep):
    out, kinds = scrub_text(keep, exact=[EMAIL, PHONE])
    assert out == keep and kinds == []


def test_contact_values():
    assert contact_values({"personal_info": {"email": EMAIL, "phone": PHONE}}) == [EMAIL, PHONE]
    assert contact_values({}) == []


def test_replacement_marker():
    out, _ = scrub_text(f"x {EMAIL} y")
    assert out == f"x {REPLACEMENT} y"


def test_scrub_documents_content_metadata_and_log(capsys):
    docs = [
        Document(
            page_content=f"reach {EMAIL}",
            metadata={"source": "github", "section": "readme", "url": f"mailto:{EMAIL}", "n": 3},
        )
    ]
    out = scrub_documents(docs, [EMAIL, PHONE])
    assert EMAIL not in out[0].page_content
    assert EMAIL not in out[0].metadata["url"]
    assert out[0].metadata["n"] == 3
    assert EMAIL in docs[0].page_content  # input not mutated
    printed = capsys.readouterr().out
    assert "[scrub] removed exact from github/readme" in printed
    assert EMAIL not in printed
