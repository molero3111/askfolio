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

def phone_variants(digits: str) -> list[str]:
    """Digit strings to look for: the full number plus national forms without the country code.

    National numbers are 9 or 10 digits in most countries, so the last 9 and last 10 digits are
    both checked, along with each written with a leading trunk ``0``.
    """
    variants = [digits]
    for n in (10, 9):
        if len(digits) > n:
            national = digits[-n:]
            variants += [national, "0" + national]
    return list(dict.fromkeys(variants))

def build_queries(table: str, phone_params: int = 1) -> tuple[str, str]:
    t = table.replace('"', '""')
    meta = f"{METADATA_COL}::text"
    email_sql = (
        f'SELECT count(*) FROM "{t}" WHERE {CONTENT_COL} ILIKE \'%\' || $1 || \'%\' '
        f"OR {meta} ILIKE '%' || $1 || '%'"
    )
    conds = []
    for i in range(1, phone_params + 1):
        conds.append(f"regexp_replace({CONTENT_COL}, '\\D', '', 'g') LIKE '%' || ${i} || '%'")
        conds.append(f"regexp_replace({meta}, '\\D', '', 'g') LIKE '%' || ${i} || '%'")
    phone_sql = f'SELECT count(*) FROM "{t}" WHERE ' + " OR ".join(conds)
    return email_sql, phone_sql

def email_local_part(email: str) -> str:
    return email.strip().partition("@")[0]

def email_candidate_key(email: str) -> str:
    """First dot-separated segment of the local part, so 'jane dot doe' spellings are candidates too."""
    return email_local_part(email).split(".")[0]

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

    email_sql, _ = build_queries(table)
    candidate_sql = build_candidate_query(table)
    conn = await asyncpg.connect(dsn)
    try:
        email_rows = 0
        for e in emails:
            email_rows += await conn.fetchval(email_sql, e)
        # Obfuscated spellings: pull rows mentioning the local part, match in Python.
        candidates: dict[str, list[str]] = {}
        for e in emails:
            local = email_candidate_key(e)
            if local:
                candidates[e] = [r[0] for r in await conn.fetch(candidate_sql, local)]
        obfuscated = sum(count_obfuscated_email_rows(rows, [e]) for e, rows in candidates.items())
        email_rows = max(email_rows, obfuscated)
        phone_rows = 0
        for p in phones:
            variants = phone_variants(p)
            phone_sql = build_queries(table, len(variants))[1]
            phone_rows += await conn.fetchval(phone_sql, *variants)
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
    try:
        cv = json.loads(kb_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"cannot read knowledge JSON: {type(exc).__name__}")
        return 2
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
