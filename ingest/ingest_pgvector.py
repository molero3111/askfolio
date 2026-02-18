"""Ingest PDFs from resources/pdfs into pgvector. Run from project root."""
import argparse
import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

import sqlalchemy
from langchain_community.document_loaders import PyPDFLoader
from langchain_postgres import PGEngine, PGVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DB_CONNECTION_URL_ASYNC,
    EMBEDDING_MODEL,
    PDF_DIR,
    PGVECTOR_COLLECTION_NAME,
    VECTOR_SIZE,
)


def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs into pgvector.")
    parser.add_argument(
        "--reinstall",
        action="store_true",
        help="Drop the collection table first, then create and ingest (clean re-ingest).",
    )
    args = parser.parse_args()

    pdf_dir = Path(PDF_DIR)
    if not pdf_dir.is_absolute():
        pdf_dir = ROOT / pdf_dir
    if not pdf_dir.exists():
        print(f"PDF directory not found: {pdf_dir}")
        sys.exit(1)
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"No PDFs in {pdf_dir}")
        sys.exit(1)
    if not DB_CONNECTION_URL_ASYNC:
        print("DB_CONNECTION_URL must be set in .env")
        sys.exit(1)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    engine = PGEngine.from_connection_string(url=DB_CONNECTION_URL_ASYNC)

    if args.reinstall:
        print(f'Dropping table "{PGVECTOR_COLLECTION_NAME}" (if exists)...')
        # Use asyncpg directly so we don't rely on PGEngine internals
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
    all_docs = []
    for f in pdf_files:
        loader = PyPDFLoader(str(pdf_dir / f))
        all_docs.extend(loader.load())
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(all_docs)
    vs = PGVectorStore.create_sync(
        engine=engine,
        table_name=PGVECTOR_COLLECTION_NAME,
        embedding_service=embeddings,
    )
    vs.add_documents(chunks)
    print("Ingestion complete.")


if __name__ == "__main__":
    main()
