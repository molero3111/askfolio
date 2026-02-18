import os
from pathlib import Path


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


TELEGRAM_BOT_TOKEN = _env("TELEGRAM_BOT_TOKEN")
# File to persist last Telegram update_id (so we send offset on next getUpdates and after restart)
TELEGRAM_OFFSET_FILE = _env("TELEGRAM_OFFSET_FILE", "telegram_offset.txt")
LLM_API_URL = _env("LLM_API_URL", "http://host.docker.internal:1234/v1/chat/completions")
LLM_MODEL = _env("LLM_MODEL", "local-model")
LLM_API_KEY = _env("LLM_API_KEY", "")
DB_CONNECTION_URL = _env("DB_CONNECTION_URL")
# PGEngine uses async SQLAlchemy; it requires postgresql+asyncpg:// (asyncpg driver)
DB_CONNECTION_URL_ASYNC = (
    DB_CONNECTION_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
    if DB_CONNECTION_URL and "postgresql+asyncpg" not in DB_CONNECTION_URL
    else DB_CONNECTION_URL
)
PGVECTOR_COLLECTION_NAME = _env("PGVECTOR_COLLECTION_NAME", "rag_docs")
VECTOR_SIZE = int(_env("VECTOR_SIZE", "384"))
EMBEDDING_MODEL = _env("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(_env("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(_env("CHUNK_OVERLAP", "50"))
PDF_DIR = _env("PDF_DIR", "./resources/pdfs")
# Seconds to sleep after each poll cycle (getUpdates is long-polling with timeout; this is extra delay between cycles)
POLL_INTERVAL_SECONDS = int(_env("POLL_INTERVAL_SECONDS", "300"))  # 5 minutes
# getUpdates long-poll timeout (request blocks up to this many seconds for new updates)
GET_UPDATES_TIMEOUT = int(_env("GET_UPDATES_TIMEOUT", "30"))

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
RECRUITER_PROMPT_PATH = PROMPTS_DIR / "recruiter_prompt.txt"
