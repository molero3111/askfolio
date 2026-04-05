import os
from pathlib import Path
from urllib.parse import quote


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _env_bool(key: str, default: bool = False) -> bool:
    val = _env(key, "true" if default else "false").strip().lower()
    return val in {"1", "true", "yes", "on"}


def _postgres_dsn() -> str:
    """Build postgresql:// from POSTGRES_* (same vars as the db container)."""
    user = _env("POSTGRES_USER")
    password = _env("POSTGRES_PASSWORD")
    db = _env("POSTGRES_DB")
    if not user or not password or not db:
        return ""
    host = _env("POSTGRES_HOST", "db-askfolio")
    port = _env("POSTGRES_PORT", "5432")
    u = quote(user, safe="")
    p = quote(password, safe="")
    d = quote(db, safe="")
    return f"postgresql://{u}:{p}@{host}:{port}/{d}"


TELEGRAM_BOT_TOKEN = _env("TELEGRAM_BOT_TOKEN")
# File to persist last Telegram update_id (so we send offset on next getUpdates and after restart)
TELEGRAM_OFFSET_FILE = _env("TELEGRAM_OFFSET_FILE", "telegram_offset.txt")
LLM_API_URL = _env(
    "LLM_API_URL",
    "http://host.docker.internal:11434/v1/chat/completions",
)
LLM_MODEL = _env("LLM_MODEL", "local-model")
LLM_API_KEY = _env("LLM_API_KEY", "")
# Timeout in seconds for the HTTP request to the LLM API (default 600 = 10 min for slow local models)
LLM_REQUEST_TIMEOUT = int(_env("LLM_REQUEST_TIMEOUT", "600"))
DB_CONNECTION_URL = _postgres_dsn()
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
# JSON sources for vector ingest (paths relative to project root unless absolute)
INGEST_KNOWLEDGE_JSON = _env(
    "INGEST_KNOWLEDGE_JSON",
    "resources/json/cv.json",
)
INGEST_GITHUB_JSON = _env(
    "INGEST_GITHUB_JSON",
    "resources/json/github_projects.json",
)
# How many vector chunks to retrieve per question (higher = more repos/context; default 20)
RAG_TOP_K = int(_env("RAG_TOP_K", "20"))
# Seconds to sleep after each poll cycle (getUpdates is long-polling with timeout; this is extra delay between cycles)
POLL_INTERVAL_SECONDS = int(_env("POLL_INTERVAL_SECONDS", "300"))  # 5 minutes
# getUpdates long-poll timeout (request blocks up to this many seconds for new updates)
GET_UPDATES_TIMEOUT = int(_env("GET_UPDATES_TIMEOUT", "60"))

# LangSmith tracing (optional)
LANGSMITH_API_KEY = _env("LANGSMITH_API_KEY")
LANGSMITH_TRACING = _env_bool("LANGSMITH_TRACING", True)
LANGSMITH_PROJECT = _env("LANGSMITH_PROJECT", "askfolio-dev")
LANGSMITH_ENDPOINT = _env("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
LANGSMITH_ENABLED = bool(LANGSMITH_API_KEY and LANGSMITH_TRACING)
LANGSMITH_RUNTIME_FLAG = "ASKFOLIO_LANGSMITH_ENABLED"


def is_langsmith_enabled() -> bool:
    """Runtime LangSmith toggle for manual spans (supports fail-open disable)."""
    return _env_bool(LANGSMITH_RUNTIME_FLAG, LANGSMITH_ENABLED)


PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
RECRUITER_PROMPT_PATH = PROMPTS_DIR / "recruiter_prompt.txt"
