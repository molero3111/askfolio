from langchain_postgres import PGEngine, PGVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import (
    DB_CONNECTION_URL_ASYNC,
    EMBEDDING_MODEL,
    PGVECTOR_COLLECTION_NAME,
    RAG_TOP_K,
)

_engine = None
_vectorstore = None


def get_vectorstore():
    global _engine, _vectorstore
    if _vectorstore is None:
        if not DB_CONNECTION_URL_ASYNC:
            raise ValueError(
                "Postgres connection missing: set POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB "
                "(and POSTGRES_HOST / POSTGRES_PORT if not using defaults)."
            )
        _engine = PGEngine.from_connection_string(url=DB_CONNECTION_URL_ASYNC)
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        _vectorstore = PGVectorStore.create_sync(
            engine=_engine,
            table_name=PGVECTOR_COLLECTION_NAME,
            embedding_service=embeddings,
        )
    return _vectorstore


def get_relevant_context(query: str, k: int | None = None) -> str:
    vs = get_vectorstore()
    n = k if k is not None else RAG_TOP_K
    docs = vs.similarity_search(query, k=n)
    return "\n\n---\n\n".join(doc.page_content for doc in docs)
