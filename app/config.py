import os
from pathlib import Path
from urllib.parse import quote

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(p: str) -> Path:
    """Resolve a configured path against the project root (absolute paths pass through)."""
    path = Path(p)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _env_bool(key: str, default: bool = False) -> bool:
    val = _env(key, "true" if default else "false").strip().lower()
    return val in {"1", "true", "yes", "on"}


def _name_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


# Tests set ASKFOLIO_SKIP_DOTENV=1 so a developer's real .env is never loaded.
SKIP_DOTENV = _env_bool("ASKFOLIO_SKIP_DOTENV")
if not SKIP_DOTENV:
    load_dotenv(PROJECT_ROOT / ".env")


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
TELEGRAM_OFFSET_FILE = _env("TELEGRAM_OFFSET_FILE", "data/telegram_offset.txt")
LLM_API_URL = _env("LLM_API_URL", "https://openrouter.ai/api/v1/chat/completions")
# No default: set LLM_MODEL in .env (main.py exits when empty)
LLM_MODEL = _env("LLM_MODEL", "")
# Comma-separated OpenRouter model IDs tried in order if the primary fails
LLM_FALLBACK_MODELS = [m.strip() for m in _env("LLM_FALLBACK_MODELS", "").split(",") if m.strip()]
LLM_API_KEY = _env("LLM_API_KEY", "")
POSTGRES_PASSWORD = _env("POSTGRES_PASSWORD")
# Timeout in seconds for the HTTP request to the LLM API
LLM_REQUEST_TIMEOUT = int(_env("LLM_REQUEST_TIMEOUT", "60"))
LLM_MAX_TOKENS = int(_env("LLM_MAX_TOKENS", "800"))
# OpenRouter provider.data_collection: "deny" refuses providers that may train on prompts
LLM_DATA_COLLECTION = _env("LLM_DATA_COLLECTION", "deny")
# OpenRouter provider.only: comma-separated provider slugs allowed to serve requests (empty = any)
LLM_PROVIDER_ONLY = _name_list(_env("LLM_PROVIDER_ONLY", ""))
LLM_APP_URL = _env("LLM_APP_URL", "")
LLM_APP_NAME = _env("LLM_APP_NAME", "askfolio")
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
# JSON sources for vector ingest (paths relative to project root unless absolute)
INGEST_KNOWLEDGE_JSON = _env(
    "INGEST_KNOWLEDGE_JSON",
    "resources/json/cv.json",
)
INGEST_GITHUB_JSON = _env(
    "INGEST_GITHUB_JSON",
    "resources/json/github_projects.json",
)
# Comma-separated repo names to ingest from INGEST_GITHUB_JSON; empty ingests every repo
INGEST_GITHUB_REPOS = _name_list(_env("INGEST_GITHUB_REPOS", ""))
# How many vector chunks to retrieve per question (higher = more repos/context; default 20)
RAG_TOP_K = int(_env("RAG_TOP_K", "20"))
# Seconds to sleep after each poll cycle (getUpdates is long-polling with timeout; this is extra delay between cycles)
POLL_INTERVAL_SECONDS = int(_env("POLL_INTERVAL_SECONDS", "0"))
# getUpdates long-poll timeout (request blocks up to this many seconds for new updates)
GET_UPDATES_TIMEOUT = int(_env("GET_UPDATES_TIMEOUT", "60"))

# LangSmith tracing (optional)
LANGSMITH_API_KEY = _env("LANGSMITH_API_KEY")
LANGSMITH_TRACING = _env_bool("LANGSMITH_TRACING", True)
LANGSMITH_PROJECT = _env("LANGSMITH_PROJECT", "askfolio-dev")
LANGSMITH_ENDPOINT = _env("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
LANGSMITH_ENABLED = bool(LANGSMITH_API_KEY and LANGSMITH_TRACING)
LANGSMITH_RUNTIME_FLAG = "ASKFOLIO_LANGSMITH_ENABLED"


# PostHog usage analytics (optional): off unless both POSTHOG_KEY and ANALYTICS_SALT are set
POSTHOG_KEY = _env("POSTHOG_KEY")
POSTHOG_HOST = _env("POSTHOG_HOST", "https://us.i.posthog.com")
# Secret for hashing chat ids into anonymous ids; without it ids would be guessable
ANALYTICS_SALT = _env("ANALYTICS_SALT")
ANALYTICS_ENV = _env("ANALYTICS_ENV", "local")
# Chat ids whose events are tagged is_internal (the owner's own tests)
ANALYTICS_INTERNAL_CHAT_IDS = _name_list(_env("ANALYTICS_INTERNAL_CHAT_IDS", ""))
# Hashed ids of chats already seen, so first contact is counted once across restarts
SEEN_CHATS_FILE = _env("SEEN_CHATS_FILE", "data/seen_chats")


def is_langsmith_enabled() -> bool:
    """Runtime LangSmith toggle for manual spans (supports fail-open disable)."""
    return _env_bool(LANGSMITH_RUNTIME_FLAG, LANGSMITH_ENABLED)


PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
SYSTEM_PROMPT_PATH = PROMPTS_DIR / "system_prompt.txt"
USER_PROMPT_PATH = PROMPTS_DIR / "user_prompt.txt"
TIMEZONE_SYSTEM_PROMPT_PATH = PROMPTS_DIR / "timezone_system_prompt.txt"
TIMEZONE_PROMPT_PATH = PROMPTS_DIR / "timezone_prompt.txt"
# When true, the alias dict in calendar_client.py acts as a fast-path before the LLM call.
# Default false: LLM is the primary (sole) resolver for timezone input.
TIMEZONE_ALIAS_FAST_PATH = _env_bool("TIMEZONE_ALIAS_FAST_PATH", False)

# Google Calendar
GOOGLE_CREDENTIALS_PATH = _env("GOOGLE_CREDENTIALS_PATH", "secrets/credentials.json")
GOOGLE_TOKEN_PATH = _env("GOOGLE_TOKEN_PATH", "secrets/token.json")
GOOGLE_CALENDAR_ID = _env("GOOGLE_CALENDAR_ID", "primary")
# Developer's local timezone — used to calculate working hours on the calendar
CALENDAR_TIMEZONE = _env("CALENDAR_TIMEZONE", "UTC")
# Timezone shown to recruiters by default in slot labels
CALENDAR_DISPLAY_TIMEZONE = _env("CALENDAR_DISPLAY_TIMEZONE", "America/New_York")
CALENDAR_SLOT_DURATION_MINUTES = int(_env("CALENDAR_SLOT_DURATION_MINUTES", "60"))
CALENDAR_WORKING_HOURS_START = int(_env("CALENDAR_WORKING_HOURS_START", "9"))
CALENDAR_WORKING_HOURS_END = int(_env("CALENDAR_WORKING_HOURS_END", "18"))
CALENDAR_DAYS_AHEAD = int(_env("CALENDAR_DAYS_AHEAD", "7"))

# Rate limiting (in-memory). The global daily limit must stay below the OpenRouter
# free-model daily cap (50 requests/day) so the bot's own limit triggers first.
RATE_LIMIT_PER_CHAT_PER_MINUTE = int(_env("RATE_LIMIT_PER_CHAT_PER_MINUTE", "5"))
RATE_LIMIT_PER_CHAT_PER_DAY = int(_env("RATE_LIMIT_PER_CHAT_PER_DAY", "50"))
RATE_LIMIT_GLOBAL_PER_DAY = int(_env("RATE_LIMIT_GLOBAL_PER_DAY", "45"))
RATE_LIMIT_BOOKINGS_PER_CHAT_PER_DAY = int(_env("RATE_LIMIT_BOOKINGS_PER_CHAT_PER_DAY", "1"))
RATE_LIMIT_BOOKINGS_GLOBAL_PER_DAY = int(_env("RATE_LIMIT_BOOKINGS_GLOBAL_PER_DAY", "5"))
# Master switch for meeting booking (the MVP .env.example ships it disabled)
BOOKING_ENABLED = _env_bool("BOOKING_ENABLED", False)
