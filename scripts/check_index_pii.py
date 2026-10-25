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
from ingest.scrub import contact_values

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


async def _count(dsn: str, table: str, emails: list[str], phones: list[str]) -> tuple[int, int]:
    import asyncpg

    email_sql, phone_sql = build_queries(table)
    conn = await asyncpg.connect(dsn)
    try:
        email_rows = sum([await conn.fetchval(email_sql, e) for e in emails])
        phone_rows = sum([await conn.fetchval(phone_sql, p) for p in phones])
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
    dsn = DB_CONNECTION_URL_ASYNC.replace("postgresql+asyncpg://", "postgresql://", 1)
    email_rows, phone_rows = asyncio.run(_count(dsn, args.table, emails, phones))
    print(f"email: {email_rows} rows, phone: {phone_rows} rows")
    return 1 if email_rows or phone_rows else 0


if __name__ == "__main__":
    sys.exit(main())
