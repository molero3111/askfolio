from langchain_postgres import PGEngine, PGVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import (
    DB_CONNECTION_URL_ASYNC,
    PGVECTOR_COLLECTION_NAME,
    EMBEDDING_MODEL,
)

_engine = None
_vectorstore = None


def get_vectorstore():
    global _engine, _vectorstore
    if _vectorstore is None:
        if not DB_CONNECTION_URL_ASYNC:
            raise ValueError("DB_CONNECTION_URL must be set.")
        _engine = PGEngine.from_connection_string(url=DB_CONNECTION_URL_ASYNC)
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        _vectorstore = PGVectorStore.create_sync(
            engine=_engine,
            table_name=PGVECTOR_COLLECTION_NAME,
            embedding_service=embeddings,
        )
    return _vectorstore


def get_relevant_context(query: str, k: int = 4) -> str:
    vs = get_vectorstore()
    docs = vs.similarity_search(query, k=k)
    return "\n\n".join(doc.page_content for doc in docs)
