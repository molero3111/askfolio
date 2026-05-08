# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

**askfolio** — a Telegram bot that answers recruiter questions about Emmanuel's developer profile using RAG (retrieval-augmented generation) and can schedule Google Calendar meetings. It runs as a long-polling loop driven by a LangGraph `StateGraph`.

## Running the bot

```bash
# Start Postgres (pgvector) + the bot
docker compose up -d

# Ingest/refresh the knowledge base into pgvector (run after changing cv.json or github_projects.json)
docker compose exec app-askfolio python ingest/ingest_pgvector.py

# Full reset (drops and recreates the vector table)
docker compose exec app-askfolio python ingest/ingest_pgvector.py --reinstall

# Run bot directly (requires .env and a running Postgres)
python main.py
```

### One-time Google Calendar setup

```bash
# Run locally (opens browser for OAuth — must NOT run inside Docker)
python scripts/authorize_calendar.py
# Generates token.json in project root; Docker reads it via the . :/app volume mount
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
app/config.py         — all env var reads; import config values from here, never os.environ directly

ingest/
  ├─ ingest_pgvector.py    — CLI: load JSON → chunk → embed → upsert into pgvector
  └─ json_to_documents.py  — document builders for cv.json and github_projects.json

resources/json/
  ├─ cv.json               — knowledge base (skills, experience, education)
  └─ github_projects.json  — repo index with descriptions and languages

app/prompts/recruiter_prompt.txt — system prompt template ({chat_history}, {context}, {user_input})
```

### Message routing in `process_message_node`

1. Keyword-match checks `_SCHEDULING_KEYWORDS` / `_CANCEL_KEYWORDS` (frozensets in `graph.py`).
2. If `conv.mode != IDLE` or scheduling intent detected → `_handle_scheduling()` state machine.
3. Otherwise → RAG retrieval (`get_relevant_context`) + LLM HTTP call → Telegram reply.

### Scheduling state machine (`SchedulingMode`)

`IDLE → AWAITING_SLOT_SELECTION → AWAITING_RECRUITER_INFO → IDLE` (reset_scheduling on success or cancel).  
Any cancel-keyword input at any step → `CANCELLED` → back to `IDLE`.

### State persistence

`AgentState` (TypedDict) is carried between LangGraph invocations in a plain `dict`.  
`conversations: dict` maps `str(chat_id)` → `ConversationState.to_dict()` for per-chat memory (10-message sliding window).  
`offset: int | None` is saved to `telegram_offset.txt` between restarts.

## Key configuration

All config lives in `app/config.py` and is driven by env vars (see `.env.example`).  
Critical vars:

| Var | Purpose |
|---|---|
| `TELEGRAM_BOT_TOKEN` | Required; bot will not start without it |
| `LLM_API_URL` / `LLM_MODEL` | OpenAI-compatible endpoint (default: Ollama on host) |
| `POSTGRES_*` | DSN built in `config._postgres_dsn()` |
| `CALENDAR_TIMEZONE` | Developer's IANA tz for working-hours computation |
| `CALENDAR_DISPLAY_TIMEZONE` | Timezone shown to recruiters (default `America/New_York`) |
| `LANGSMITH_API_KEY` | Optional; tracing is silently disabled when absent |

The LLM is called via plain HTTP (`requests.post`) to an OpenAI-compatible `/v1/chat/completions` endpoint — **not** via LangChain's LLM abstractions. This keeps the LLM call transparent and avoids deprecation churn.

## Timezone notes

`CALENDAR_TIMEZONE` = where Emmanuel is (used to compute which hours are "working hours").  
`CALENDAR_DISPLAY_TIMEZONE` = what recruiters see in slot labels (independent).  
`resolve_timezone(text)` in `calendar_client.py` maps fuzzy user input ("Pacific Time", "CET", "Tokyo") to IANA keys via `_TZ_ALIASES` dict, with direct `ZoneInfo` fallback.  
Docker requires the `tzdata` package (included in `requirements.txt`) since `python:3.11-slim` has no system tz data.

## Ingest details

`ingest_pgvector.py` reads two JSON sources configured by `INGEST_KNOWLEDGE_JSON` and `INGEST_GITHUB_JSON`.  
Default mode: deletes all rows in the collection, then re-inserts. Use `--append` to keep existing vectors.  
Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (384-dim). Changing `VECTOR_SIZE` requires `--reinstall`.
