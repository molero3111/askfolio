"""Idempotent schema setup: pgvector extension + vector table. Never deletes or embeds.

Run as ``python -m ingest.migrate`` (from the project root / ``/app``).
"""
import asyncio
import re
import sys
import time
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

TABLE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
RETRY_ATTEMPTS = 6  # 1 try + 5 retries, sleeping 1..5s (~15s)


def valid_table_name(name: str) -> bool:
    """True if name is a plain SQL identifier (letters, digits, underscore; no leading digit)."""
    return bool(TABLE_NAME_RE.fullmatch(name or ""))


def _is_connect_error(exc: BaseException) -> bool:
    """True for connection-refused / cannot-connect failures (not SQL errors)."""
    if isinstance(exc, (ConnectionError, OSError)):
        return True
    if isinstance(exc, sqlalchemy.exc.OperationalError):
        return True
    try:
        import asyncpg

        return isinstance(exc, (asyncpg.CannotConnectNowError, asyncpg.PostgresConnectionError))
    except ImportError:
        return False


def with_retry(fn, attempts=RETRY_ATTEMPTS, sleep=time.sleep, is_retryable=_is_connect_error):
    """Call fn(); on a connect error sleep 1, 2, 3... s and retry (~15s for 5 attempts)."""
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == attempts or not is_retryable(e):
                raise
            print(f"Database not reachable ({type(e).__name__}); retry {attempt}/{attempts - 1} in {attempt}s...")
            sleep(attempt)


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
    if not valid_table_name(PGVECTOR_COLLECTION_NAME):
        print(
            f"Invalid PGVECTOR_COLLECTION_NAME {PGVECTOR_COLLECTION_NAME!r}: "
            "use only letters, digits and underscores, not starting with a digit."
        )
        return 2

    def _migrate() -> None:
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
        finally:
            # PGEngine.close() is async but must run on the engine's own background loop.
            # Never raise from here: it would replace the original exception.
            try:
                engine._run_as_sync(engine.close())
            except Exception as close_err:
                print(f"engine close failed: {type(close_err).__name__}")

    with_retry(_migrate)
    return 0


if __name__ == "__main__":
    sys.exit(main())
