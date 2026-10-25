"""Idempotent schema setup: pgvector extension + vector table. Never deletes or embeds.

Run as ``python -m ingest.migrate`` (from the project root / ``/app``).
"""
import asyncio
import sys
from pathlib import Path

# Keep direct execution (python ingest/migrate.py) working as well as -m.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

import sqlalchemy
from langchain_postgres import PGEngine

from app.config import DB_CONNECTION_URL_ASYNC, PGVECTOR_COLLECTION_NAME, VECTOR_SIZE


def _create_extension() -> None:
    dsn = DB_CONNECTION_URL_ASYNC.replace("postgresql+asyncpg://", "postgresql://", 1)

    async def _run():
        import asyncpg

        conn = await asyncpg.connect(dsn)
        try:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        finally:
            await conn.close()

    asyncio.run(_run())


def main() -> int:
    if not DB_CONNECTION_URL_ASYNC:
        print(
            "Postgres env missing: set POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB "
            "(and POSTGRES_HOST / POSTGRES_PORT if not using defaults)."
        )
        return 1

    _create_extension()
    print("Extension 'vector' ready.")

    engine = PGEngine.from_connection_string(url=DB_CONNECTION_URL_ASYNC)
    try:
        engine.init_vectorstore_table(
            table_name=PGVECTOR_COLLECTION_NAME,
            vector_size=VECTOR_SIZE,
        )
        print(f'Created table "{PGVECTOR_COLLECTION_NAME}".')
    except sqlalchemy.exc.ProgrammingError as e:
        err_msg = str(e).lower()
        if "already exists" in err_msg or "duplicatetable" in err_msg:
            print(f'Table "{PGVECTOR_COLLECTION_NAME}" already exists, skipping create.')
        else:
            raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
