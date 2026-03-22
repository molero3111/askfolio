"""
Build LangChain Documents from knowledge base + GitHub JSON with section metadata.
"""
import json
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Prefer splitting at markdown headings and paragraphs before arbitrary cuts
MARKDOWN_FRIENDLY_SEPARATORS = [
    "\n## ",
    "\n### ",
    "\n#### ",
    "\n\n",
    "\n",
    ". ",
    " ",
    "",
]


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"JSON not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _personal_info_doc(obj: dict) -> Document:
    links = obj.get("links") or {}
    lines = [
        "# Personal profile",
        f"Name: {obj.get('name', '')}",
        f"Title: {obj.get('title', '')}",
        f"Location: {obj.get('location', '')}",
        f"Email: {obj.get('email', '')}",
        f"Phone: {obj.get('phone', '')}",
        f"Summary: {obj.get('summary', '')}",
    ]
    if links:
        lines.append("Links:")
        for k, v in links.items():
            lines.append(f"- {k}: {v}")
    return Document(
        page_content="\n".join(lines),
        metadata={"source": "knowledge_base", "section": "personal_info"},
    )


def _skills_docs(skills: dict) -> list[Document]:
    docs = []
    for category, items in skills.items():
        if not isinstance(items, list):
            continue
        label = str(category).replace("_", " ").title()
        body = "\n".join(f"- {item}" for item in items)
        content = f"## Skills: {label}\n\n{body}"
        docs.append(
            Document(
                page_content=content,
                metadata={
                    "source": "knowledge_base",
                    "section": f"skills.{category}",
                },
            )
        )
    return docs


def _languages_doc(items: list) -> Document:
    lines = ["## Languages"]
    for row in items:
        if isinstance(row, dict):
            lines.append(
                f"- {row.get('language', '')}: {row.get('level', '')}"
                + (f" (certificate: {row.get('certificate')})" if row.get("certificate") else "")
            )
    return Document(
        page_content="\n".join(lines),
        metadata={"source": "knowledge_base", "section": "languages"},
    )


def _education_docs(items: list) -> list[Document]:
    docs = []
    for i, row in enumerate(items):
        if not isinstance(row, dict):
            continue
        content = "\n".join(
            [
                "## Education",
                f"Degree: {row.get('degree', '')}",
                f"Institution: {row.get('institution', '')}",
                f"Period: {row.get('period', '')}",
                f"Location: {row.get('location', '')}",
            ]
        )
        docs.append(
            Document(
                page_content=content,
                metadata={"source": "knowledge_base", "section": f"education.{i}"},
            )
        )
    return docs


def _certification_docs(items: list) -> list[Document]:
    docs = []
    for i, row in enumerate(items):
        if not isinstance(row, dict):
            continue
        content = "\n".join(
            [
                "## Certification",
                f"Title: {row.get('title', '')}",
                f"Institution: {row.get('institution', '')}",
                f"Year: {row.get('year', '')}",
                f"Certificate: {row.get('certificate', '')}",
            ]
        )
        docs.append(
            Document(
                page_content=content,
                metadata={"source": "knowledge_base", "section": f"certifications.{i}"},
            )
        )
    return docs


def _work_experience_docs(items: list) -> list[Document]:
    docs = []
    for i, job in enumerate(items):
        if not isinstance(job, dict):
            continue
        period = job.get("period") or {}
        if isinstance(period, dict):
            period_s = f"{period.get('start', '')} – {period.get('end', '')}"
        else:
            period_s = str(period)
        lines = [
            "## Work experience",
            f"Role: {job.get('role', '')}",
            f"Company: {job.get('company', '')}",
            f"Location: {job.get('location', '')}",
            f"Period: {period_s}",
            "",
            "Responsibilities:",
        ]
        for r in job.get("responsibilities") or []:
            lines.append(f"- {r}")
        notes = (job.get("notes") or "").strip()
        if notes:
            lines.extend(["", f"Notes: {notes}"])
        company = str(job.get("company", f"role_{i}"))
        docs.append(
            Document(
                page_content="\n".join(lines),
                metadata={
                    "source": "knowledge_base",
                    "section": "work_experience",
                    "company": company,
                },
            )
        )
    return docs


def documents_from_knowledge_base(path: Path) -> list[Document]:
    data = _load_json(path)
    docs: list[Document] = []

    if "personal_info" in data and isinstance(data["personal_info"], dict):
        docs.append(_personal_info_doc(data["personal_info"]))

    if "skills" in data and isinstance(data["skills"], dict):
        docs.extend(_skills_docs(data["skills"]))

    if "languages" in data and isinstance(data["languages"], list):
        docs.append(_languages_doc(data["languages"]))

    if "education" in data and isinstance(data["education"], list):
        docs.extend(_education_docs(data["education"]))

    if "certifications" in data and isinstance(data["certifications"], list):
        docs.extend(_certification_docs(data["certifications"]))

    if "work_experience" in data and isinstance(data["work_experience"], list):
        docs.extend(_work_experience_docs(data["work_experience"]))

    return docs


def _github_index_documents(projects: list) -> list[Document]:
    """
    Dense index chunk(s) so queries like "Python projects" retrieve many repo
    URLs, not only whichever single README embedding matched best.
    """
    rows: list[dict] = [p for p in projects if isinstance(p, dict)]
    if not rows:
        return []

    intro = (
        "# GitHub repository index\n\n"
        "Complete list of Emmanuel's public repositories (name, primary language on GitHub, "
        "short description, URL). When asked which projects exist or which use a language, "
        "use every matching line below and include each URL.\n"
    )

    def row_line(proj: dict) -> str:
        name = proj.get("name") or "unknown"
        url = proj.get("url") or ""
        lang = proj.get("language") or "N/A"
        desc = (proj.get("description") or "").strip() or "N/A"
        return f"- **{name}** | Language: {lang} | {desc} | URL: {url}"

    # Batch so each index chunk stays embed-friendly (~15–20 repos per part)
    batch_size = 18
    docs: list[Document] = []
    for batch_num, i in enumerate(range(0, len(rows), batch_size), start=1):
        batch = rows[i : i + batch_size]
        header = intro if i == 0 else f"# GitHub repository index (continued, part {batch_num})\n\n"
        body = "\n".join(row_line(p) for p in batch)
        content = header + "\n" + body
        docs.append(
            Document(
                page_content=content,
                metadata={
                    "source": "github",
                    "section": "github_index" if i == 0 else f"github_index_{batch_num}",
                },
            )
        )
    return docs


def documents_from_github_projects(path: Path) -> list[Document]:
    data = _load_json(path)
    projects = data.get("github_projects") or []
    docs: list[Document] = []
    docs.extend(_github_index_documents(projects))
    for proj in projects:
        if not isinstance(proj, dict):
            continue
        name = proj.get("name") or "unknown"
        topics = proj.get("topics") or []
        topics_s = ", ".join(topics) if topics else "(none)"
        header = "\n".join(
            [
                f"# GitHub project: {name}",
                f"URL: {proj.get('url', '')}",
                f"Description: {proj.get('description') or 'N/A'}",
                f"Primary language: {proj.get('language') or 'N/A'}",
                f"Topics: {topics_s}",
                "",
            ]
        )
        readme = proj.get("readme") or ""
        content = header + (readme if readme else "_No README in repository._")
        docs.append(
            Document(
                page_content=content,
                metadata={
                    "source": "github",
                    "section": "github_project",
                    "repo": name,
                },
            )
        )
    return docs


def split_documents_semantically(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """
    Split only oversized documents, using markdown/paragraph-aware separators.
    Metadata is copied onto each chunk.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=MARKDOWN_FRIENDLY_SEPARATORS,
        length_function=len,
        is_separator_regex=False,
    )
    final: list[Document] = []
    for doc in documents:
        if len(doc.page_content) <= chunk_size:
            final.append(doc)
            continue
        chunks = splitter.split_documents([doc])
        final.extend(chunks)
    return final
