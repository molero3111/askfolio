"""Ingest knowledge base + GitHub JSON into pgvector with section-aware chunking."""
import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

import sqlalchemy
from langchain_postgres import PGEngine, PGVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DB_CONNECTION_URL_ASYNC,
    EMBEDDING_MODEL,
    INGEST_GITHUB_JSON,
    INGEST_KNOWLEDGE_JSON,
    PGVECTOR_COLLECTION_NAME,
    VECTOR_SIZE,
)
from ingest.json_to_documents import (
    documents_from_github_projects,
    documents_from_knowledge_base,
    split_documents_semantically,
)


def _resolve_path(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else ROOT / path


def _clear_collection_rows() -> None:
    """Delete all rows from the vector table (keeps table/schema)."""
    dsn = DB_CONNECTION_URL_ASYNC.replace("postgresql+asyncpg://", "postgresql://", 1)

    async def _run():
        import asyncpg

        conn = await asyncpg.connect(dsn)
        try:
            await conn.execute(f'DELETE FROM "{PGVECTOR_COLLECTION_NAME}"')
        finally:
            await conn.close()

    asyncio.run(_run())


def main():
    parser = argparse.ArgumentParser(
        description="Ingest knowledge base + GitHub JSON into pgvector.",
    )
    parser.add_argument(
        "--reinstall",
        action="store_true",
        help="Drop the collection table, recreate it, then ingest (full reset).",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Do not delete existing vectors before ingest (may duplicate if you run twice). "
        "Default: delete all rows in the collection table, then ingest fresh.",
    )
    args = parser.parse_args()

    kb_path = _resolve_path(INGEST_KNOWLEDGE_JSON)
    gh_path = _resolve_path(INGEST_GITHUB_JSON)

    if not kb_path.exists():
        print(f"Knowledge base JSON not found: {kb_path}")
        sys.exit(1)
    if not gh_path.exists():
        print(f"GitHub projects JSON not found: {gh_path}")
        sys.exit(1)

    if not DB_CONNECTION_URL_ASYNC:
        print(
            "Postgres env missing: set POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB "
            "(and POSTGRES_HOST / POSTGRES_PORT if not using defaults)."
        )
        sys.exit(1)

    print(f"Loading documents from:\n  - {kb_path}\n  - {gh_path}")
    all_docs = []
    all_docs.extend(documents_from_knowledge_base(kb_path))
    all_docs.extend(documents_from_github_projects(gh_path))
    print(f"Built {len(all_docs)} section-level documents.")

    chunks = split_documents_semantically(
        all_docs,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    print(f"After semantic chunking: {len(chunks)} chunks (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    engine = PGEngine.from_connection_string(url=DB_CONNECTION_URL_ASYNC)

    if args.reinstall:
        print(f'Dropping table "{PGVECTOR_COLLECTION_NAME}" (if exists)...')
        dsn = DB_CONNECTION_URL_ASYNC.replace("postgresql+asyncpg://", "postgresql://", 1)

        async def _drop():
            import asyncpg

            conn = await asyncpg.connect(dsn)
            try:
                await conn.execute(f'DROP TABLE IF EXISTS "{PGVECTOR_COLLECTION_NAME}"')
            finally:
                await conn.close()

        asyncio.run(_drop())
        print("Dropped.")

    try:
        engine.init_vectorstore_table(
            table_name=PGVECTOR_COLLECTION_NAME,
            vector_size=VECTOR_SIZE,
        )
    except sqlalchemy.exc.ProgrammingError as e:
        err_msg = str(e).lower()
        if "already exists" in err_msg or "duplicatetable" in err_msg:
            print(f'Table "{PGVECTOR_COLLECTION_NAME}" already exists, skipping create.')
        else:
            raise

    if not args.append:
        print(f'Clearing existing rows in "{PGVECTOR_COLLECTION_NAME}"...')
        _clear_collection_rows()
        print("Cleared.")
    else:
        print("Append mode: keeping existing vectors.")

    vs = PGVectorStore.create_sync(
        engine=engine,
        table_name=PGVECTOR_COLLECTION_NAME,
        embedding_service=embeddings,
    )
    vs.add_documents(chunks)
    print("Ingestion complete.")


if __name__ == "__main__":
    main()
