# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

**askfolio** — a Telegram bot that answers recruiter questions about a developer's profile using RAG (retrieval-augmented generation) and can schedule Google Calendar meetings. It runs as a long-polling loop driven by a LangGraph `StateGraph`.

## Running the bot

```bash
# First run, in this order (the app crash-loops if the table doesn't exist yet)
docker compose up -d db-askfolio
docker compose run --rm app-askfolio python -m ingest.migrate
docker compose run --rm app-askfolio python ingest/ingest_pgvector.py
docker compose up -d

# Ingest/refresh the knowledge base into pgvector (run after changing cv.json or github_projects.json)
docker compose exec app-askfolio python ingest/ingest_pgvector.py

# Full reset (drops and recreates the vector table)
docker compose exec app-askfolio python ingest/ingest_pgvector.py --reinstall

# Run bot directly (requires .env and a running Postgres)
uv run python main.py
```

### One-time Google Calendar setup

```bash
# Run locally (opens browser for OAuth — must NOT run inside Docker)
uv run python scripts/authorize_calendar.py
# Generates secrets/token.json; Docker reads it via the ./secrets:/app/secrets volume mount
```

## Architecture

```
main.py
  └─ build_graph() → LangGraph StateGraph loop (poll → process → repeat)

app/graph.py          — StateGraph nodes and routing logic
  ├─ get_telegram_updates_node  — calls Telegram getUpdates, returns pending messages
  └─ process_message_node       — routes each message to scheduling SM or LLM Q&A

app/conversation.py   — ConversationState dataclass + SchedulingMode enum
app/calendar_client.py — Google Calendar freebusy query, slot formatting, meeting creation, tz resolution
app/rag.py            — pgvector similarity search via langchain-postgres
app/telegram_client.py — send_message / get_updates wrappers
app/observability.py  — optional LangSmith tracing decorators (no-op when key absent)
app/config.py         — all env var reads (also loads .env unless ASKFOLIO_SKIP_DOTENV); import config values from here, never os.environ directly. Only healthcheck.py and heartbeat.py read env themselves
app/llm_client.py     — OpenRouter-aware chat completions client with fallback models and typed errors
app/llm_messages.py   — builds the Q&A/timezone message lists; sanitizes input; fills {scheduling_hint} from BOOKING_ENABLED
app/rate_limit.py     — in-memory MessageLimiter / BookingLimiter
app/analytics.py      — optional PostHog events (question_asked, recruiter_first_seen); off unless POSTHOG_KEY and ANALYTICS_SALT are set; never raises
app/heartbeat.py      — touches HEARTBEAT_FILE each cycle (stdlib only)
app/healthcheck.py    — `python -m app.healthcheck`: exit 0 if the heartbeat is fresh (container healthcheck)
app/log_redact.py     — logging filter and excepthook that keep secrets out of logs
app/offset.py         — load/save the Telegram update offset
app/secure_io.py      — atomic owner-only file writes
app/text_normalize.py — folds look-alike/invisible/encoded text before sanitizing or scrubbing

ingest/
  ├─ ingest_pgvector.py    — CLI: load JSON → chunk → embed → upsert into pgvector
  ├─ json_to_documents.py  — document builders for cv.json and github_projects.json
  ├─ migrate.py            — `python -m ingest.migrate`: idempotent pgvector extension + table setup
  └─ scrub.py              — strips emails and phone numbers from documents before embedding

scripts/
  ├─ check_index_pii.py    — fails if the owner's email/phone is still in the vector table
  ├─ privacy_scan.py       — CI: fails if a tracked file contains a PRIVACY_DENYLIST term
  └─ check_image.sh        — CI: asserts the built image has no secrets and runs non-root

deploy/compose.yml         — production stack: GHCR image, ASKFOLIO_* host paths, read-only hardened app
.github/workflows/         — ci.yml (lint, tests, privacy scan, image check) and deploy.yml (build, push, deploy)

resources/json/
  ├─ cv.json               — knowledge base (skills, experience, education)
  └─ github_projects.json  — repo index with descriptions and languages

app/prompts/system_prompt.txt + user_prompt.txt (user: {chat_history}, {context}, {user_input}); app/llm_messages.py builds the message lists
```

### Message routing in `process_message_node`

1. Keyword-match checks `_SCHEDULING_KEYWORDS` / `_CANCEL_KEYWORDS` (frozensets in `graph.py`). Scheduling keywords only count when `BOOKING_ENABLED` is true (default false); with booking off every message goes to Q&A.
2. If `conv.mode != IDLE`, or booking is on and scheduling intent is detected → `_handle_scheduling()` state machine. Starting a flow from idle passes the `MessageLimiter` first; replies inside a flow are not limited.
3. Otherwise → `MessageLimiter` check (per-chat/minute, per-chat/day, global/day; one notice, then silent) → RAG retrieval (`get_relevant_context`) + LLM HTTP call → Telegram reply.

### Scheduling state machine (`SchedulingMode`)

`IDLE → AWAITING_SLOT_SELECTION → AWAITING_RECRUITER_INFO → IDLE` (reset_scheduling on success or cancel).  
Any cancel-keyword input at any step → `CANCELLED` → back to `IDLE`.

### State persistence

`AgentState` (TypedDict) is carried between LangGraph invocations in a plain `dict`.  
`conversations: dict` maps `str(chat_id)` → `ConversationState.to_dict()` for per-chat memory (10-message sliding window).  
`offset: int | None` is saved to `data/telegram_offset.txt` between restarts.

## Key configuration

All config lives in `app/config.py` and is driven by env vars (see `.env.example`).  
Critical vars:

| Var | Purpose |
|---|---|
| `TELEGRAM_BOT_TOKEN` | Required; bot will not start without it |
| `LLM_API_URL` / `LLM_MODEL` | OpenAI-compatible endpoint (default: OpenRouter); `LLM_MODEL` is required |
| `LLM_FALLBACK_MODELS` / `LLM_API_KEY` | Fallback model IDs tried in order; provider API key |
| `BOOKING_ENABLED` | Master switch for scheduling (default `false`) |
| `RATE_LIMIT_*` | In-memory per-chat, global and booking caps |
| `POSTGRES_*` | DSN built in `config._postgres_dsn()` |
| `CALENDAR_TIMEZONE` | Developer's IANA tz for working-hours computation |
| `CALENDAR_DISPLAY_TIMEZONE` | Timezone shown to recruiters (default `America/New_York`) |
| `LANGSMITH_API_KEY` | Optional; tracing is silently disabled when absent |

The LLM is called via plain HTTP (`requests.post`) to an OpenAI-compatible `/v1/chat/completions` endpoint — **not** via LangChain's LLM abstractions. This keeps the LLM call transparent and avoids deprecation churn.

## Timezone notes

`CALENDAR_TIMEZONE` = where developer is (used to compute which hours are "working hours").  
`CALENDAR_DISPLAY_TIMEZONE` = what recruiters see in slot labels (independent).  
`resolve_timezone(text)` in `calendar_client.py` maps fuzzy user input ("Pacific Time", "CET", "Tokyo") to IANA keys via `_TZ_ALIASES` dict, with direct `ZoneInfo` fallback.  
Docker requires the `tzdata` package (a dependency in `pyproject.toml`) since `python:3.14-slim` has no system tz data.

## Ingest details

`ingest_pgvector.py` reads two JSON sources configured by `INGEST_KNOWLEDGE_JSON` and `INGEST_GITHUB_JSON`.  
Default mode: deletes all rows in the collection, then re-inserts. Use `--append` to keep existing vectors.  
Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (384-dim). Changing `VECTOR_SIZE` requires `--reinstall`.

## README.md

Please read if you need more info: @README.md
