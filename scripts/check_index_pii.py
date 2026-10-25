"""Fail if the owner's email or phone is still present in the pgvector table.

Prints only match counts, never the contact values themselves.
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from app.config import (
    DB_CONNECTION_URL_ASYNC,
    INGEST_KNOWLEDGE_JSON,
    PGVECTOR_COLLECTION_NAME,
    resolve_path,
)
from ingest.scrub import contact_values, normalize, owner_email_pattern

CONTENT_COL = "content"
METADATA_COL = "langchain_metadata"


def split_contacts(values: list[str]) -> tuple[list[str], list[str]]:
    """Return (emails, phone digit strings); values without digits are ignored."""
    emails = [v.strip() for v in values if "@" in v]
    phones = ["".join(c for c in v if c.isdigit()) for v in values if "@" not in v]
    return emails, [p for p in phones if p]


def build_queries(table: str) -> tuple[str, str]:
    t = table.replace('"', '""')
    meta = f"{METADATA_COL}::text"
    email_sql = (
        f'SELECT count(*) FROM "{t}" WHERE {CONTENT_COL} ILIKE \'%\' || $1 || \'%\' '
        f"OR {meta} ILIKE '%' || $1 || '%'"
    )
    phone_sql = (
        f'SELECT count(*) FROM "{t}" WHERE '
        f"regexp_replace({CONTENT_COL}, '\\D', '', 'g') LIKE '%' || $1 || '%' "
        f"OR regexp_replace({meta}, '\\D', '', 'g') LIKE '%' || $1 || '%'"
    )
    return email_sql, phone_sql


def email_local_part(email: str) -> str:
    return email.strip().partition("@")[0]


def build_candidate_query(table: str) -> str:
    t = table.replace('"', '""')
    return (
        f"SELECT {CONTENT_COL} || ' ' || {METADATA_COL}::text FROM \"{t}\" "
        f"WHERE {CONTENT_COL} ILIKE '%' || $1 || '%' OR {METADATA_COL}::text ILIKE '%' || $1 || '%'"
    )


def count_obfuscated_email_rows(rows: list[str], emails: list[str]) -> int:
    """Count rows matching any owner email, incl. [at] / ' at ' / ' dot ' / encoded spellings."""
    patterns = [p for p in (owner_email_pattern(e) for e in emails) if p]
    return sum(any(p.search(normalize(row)) for p in patterns) for row in rows)


async def _count(dsn: str, table: str, emails: list[str], phones: list[str]) -> tuple[int, int]:
    import asyncpg

    email_sql, phone_sql = build_queries(table)
    candidate_sql = build_candidate_query(table)
    conn = await asyncpg.connect(dsn)
    try:
        email_rows = 0
        for e in emails:
            email_rows += await conn.fetchval(email_sql, e)
        # Obfuscated spellings: pull rows mentioning the local part, match in Python.
        candidates: dict[str, list[str]] = {}
        for e in emails:
            local = email_local_part(e)
            if local:
                candidates[e] = [r[0] for r in await conn.fetch(candidate_sql, local)]
        obfuscated = sum(count_obfuscated_email_rows(rows, [e]) for e, rows in candidates.items())
        email_rows = max(email_rows, obfuscated)
        phone_rows = 0
        for p in phones:
            phone_rows += await conn.fetchval(phone_sql, p)
    finally:
        await conn.close()
    return email_rows, phone_rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", default=PGVECTOR_COLLECTION_NAME, help="table to check")
    args = parser.parse_args(argv)

    if not DB_CONNECTION_URL_ASYNC:
        print("Postgres env missing: set POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB.")
        return 2
    kb_path = resolve_path(INGEST_KNOWLEDGE_JSON)
    cv = json.loads(kb_path.read_text(encoding="utf-8"))
    emails, phones = split_contacts(contact_values(cv))
    if not emails and not phones:
        print("no contact values in knowledge JSON; nothing to check")
        return 2
    dsn = DB_CONNECTION_URL_ASYNC.replace("postgresql+asyncpg://", "postgresql://", 1)
    try:
        email_rows, phone_rows = asyncio.run(_count(dsn, args.table, emails, phones))
    except Exception as exc:
        print(f"check failed: {type(exc).__name__}")
        return 2
    print(f"email: {email_rows} rows, phone: {phone_rows} rows")
    return 1 if email_rows or phone_rows else 0


if __name__ == "__main__":
    sys.exit(main())
