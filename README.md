# Telegram RAG Agent (Recruiter bot)

A Telegram bot that answers recruiter-style questions using **RAG** over **pgvector**, and can **schedule Google Calendar meetings** with available time slots. Knowledge comes from a generic **`cv.json`** (profile / CV) plus **`github_projects.json`**. Built with **LangGraph** so you can extend to more agents or tools later.

## Features

- **LangGraph agent**: Two nodes — `get_telegram_updates` (fetch new messages) and `process_message` (route to scheduling or Q&A, then reply).
- **RAG Q&A**: `resources/json/cv.json` and `resources/json/github_projects.json` are ingested into pgvector (section-aware chunking). Each question is answered using retrieved context.
- **Per-chat memory**: Last 10 messages per chat are kept in a sliding window and injected into the prompt so the LLM has conversation context.
- **Meeting scheduling** (state machine):
  - Recruiter says they want to schedule → bot fetches real free slots from Google Calendar.
  - Slots are shown in **Eastern Time** by default. Recruiter can request any other timezone ("show in Pacific Time", "convert to CET") and the bot re-displays them converted.
  - Recruiter picks a slot number → bot asks for name and email → creates the Google Calendar event and emails the invite automatically.
  - Recruiter can cancel at any point with words like "cancel", "never mind", "no thanks".
- **LLM**: [OpenRouter](https://openrouter.ai/) by default, with optional fallback models; any other OpenAI-compatible endpoint works too.
- **Rate limits**: per-chat and global message caps, plus booking caps (see [Rate limits](#rate-limits)).

## CV data (`resources/json/cv.json`)

This file is your **structured CV / profile** for retrieval. It is **not** tied to a person's name in the filename — use **`cv.json`** (or override with `INGEST_KNOWLEDGE_JSON`).

Use JSON sections such as:

| Section | Purpose |
|--------|---------|
| **`personal_info`** | Name, title, location, contact, links, short **summary** |
| **`skills`** | Grouped lists (e.g. backend, frontend, AI, databases, DevOps) |
| **`languages`** | Spoken languages and levels |
| **`education`** | Degrees, institutions, periods |
| **`certifications`** | Titles, issuers, links |
| **`work_experience`** | Roles, companies, periods, **responsibilities** |
| **`projects`** | Name, year, URL, short description |

The ingest pipeline turns each logical block into documents with metadata so search stays section-aware. Match the **shape** expected by `ingest/json_to_documents.py` (see that file for exact keys), or extend the parser if you add fields.

## Setup

### 1. Copy environment file

```bash
cp .env.example .env
mkdir -p secrets data          # must exist before the first `docker compose up`, or Docker creates them owned by root
```

Set `TELEGRAM_BOT_TOKEN` in `.env` (create a bot via [@BotFather](https://t.me/BotFather)), and set `POSTGRES_PASSWORD`. Compose refuses to start until it's set.

The app container runs as your host user, so it can write to `secrets/` (Google token refresh) and `data/` (Telegram offset). Compose reads `UID`/`GID` from the environment and falls back to `1000:1000`. If your user isn't uid 1000, run `export UID GID=$(id -g)` first (bash has no `GID` variable), or add `UID=` and `GID=` to `.env`.

### 2. Prepare JSON

- Add or edit **`resources/json/cv.json`** with your profile (sections above).
- Add **`resources/json/github_projects.json`** (e.g. run `uv run python scripts/fetch_github_projects.py` with `GITHUB_TOKEN` set to avoid rate limits).
- Optional: set `INGEST_KNOWLEDGE_JSON` / `INGEST_GITHUB_JSON` in `.env` for custom paths, and `INGEST_GITHUB_REPOS` (comma-separated names) to ingest only the repos you want to show.

### 3. LLM: OpenRouter (default)

The bot calls an OpenAI-compatible chat completions endpoint over plain HTTP. OpenRouter is the default; set these in `.env`:

| Var | Purpose |
|-----|---------|
| `LLM_API_URL` | Chat completions URL (default `https://openrouter.ai/api/v1/chat/completions`) |
| `LLM_MODEL` | **Required.** OpenRouter model ID; the bot exits at startup if empty |
| `LLM_FALLBACK_MODELS` | Comma-separated model IDs tried in order if the primary fails |
| `LLM_API_KEY` | Your OpenRouter API key |
| `LLM_DATA_COLLECTION` | `deny` (default) skips providers that may train on prompts |
| `LLM_PROVIDER_ONLY` | Optional comma-separated OpenRouter provider slugs allowed to serve requests (e.g. `google-ai-studio`); empty allows any. With a provider key added in OpenRouter (BYOK), this keeps requests on your key instead of OpenRouter credits |

Any other OpenAI-compatible endpoint (for example a local Ollama) works by overriding `LLM_API_URL`. The OpenRouter-specific `provider` field and the attribution fields (`LLM_APP_URL`, `LLM_APP_NAME`) are only sent to `openrouter.ai`.

### 4. Google Calendar integration (optional — for meeting scheduling)

Scheduling is **opt-in** and off by default: the bot works fine without it. To enable it, follow these steps once, then set **`BOOKING_ENABLED=true`** in `.env`. While it is false, every message goes to the Q&A path and the assistant says scheduling isn't available.

#### 4a. Google Cloud Console setup

1. Go to [console.cloud.google.com](https://console.cloud.google.com/)
2. **Create a project** — top navbar dropdown → "New Project" → name it → Create.
3. **Enable Calendar API** — search "Google Calendar API" → click it → "Enable".
4. **Configure OAuth consent screen** — "APIs & Services" → "OAuth consent screen" → choose **External** → fill in App name and your email → Save and Continue.
5. **Add yourself as a test user** — on the OAuth consent screen page → "Test users" → add your Google account email → Save.
6. **Create credentials** — "APIs & Services" → "Credentials" → "Create Credentials" → "OAuth client ID" → Application type: **Desktop app** → Create.
7. **Download** — click the download icon next to the credential → save it as **`secrets/credentials.json`** (create the `secrets/` directory in the project root; it is git-ignored).

#### 4b. Create a dedicated test calendar (recommended)

Using a separate calendar keeps the bot away from your real schedule.

1. Go to [calendar.google.com](https://calendar.google.com)
2. Left sidebar → "Other calendars" → **+** → "Create new calendar" → name it (e.g. `Emmanuel Bot`).
3. Hover over it → three dots → "Settings and sharing" → scroll to **"Integrate calendar"** → copy the **Calendar ID** (looks like `abc123@group.calendar.google.com`).
4. Set in `.env`: `GOOGLE_CALENDAR_ID=abc123@group.calendar.google.com`

#### 4c. Authorize and generate token

Run this **once locally** (not in Docker — it opens a browser):

```bash
uv sync
uv run python scripts/authorize_calendar.py
```

A browser tab opens. Sign in, grant Calendar access, and `secrets/token.json` is written. The Docker container reads it automatically via the volume mount — **you never need to re-run this** unless you revoke access.

#### 4d. Calendar environment variables

In `.env`:

```env
GOOGLE_CREDENTIALS_PATH=secrets/credentials.json
GOOGLE_TOKEN_PATH=secrets/token.json
GOOGLE_CALENDAR_ID=primary            # or your dedicated calendar ID
CALENDAR_TIMEZONE=America/Argentina/Buenos_Aires   # developer's local tz (for working hours)
CALENDAR_DISPLAY_TIMEZONE=America/New_York         # tz shown to recruiters by default
CALENDAR_SLOT_DURATION_MINUTES=60
CALENDAR_WORKING_HOURS_START=9
CALENDAR_WORKING_HOURS_END=18
CALENDAR_DAYS_AHEAD=7
BOOKING_ENABLED=true                  # master switch; default false
```

`CALENDAR_TIMEZONE` must be a valid IANA key (e.g. `America/Argentina/Buenos_Aires`, `Europe/Madrid`). `CALENDAR_DISPLAY_TIMEZONE` controls how slot times appear to recruiters; they can always ask the bot to convert to their preferred timezone.

### 5. Start the stack and ingest

Order matters: starting the app before the vector table exists makes it crash-loop.

```bash
mkdir -p secrets data                                           # 1. directories
cp .env.example .env                                            # 2. then edit .env
docker compose up -d db-askfolio                                # 3. database only
docker compose run --rm app-askfolio python -m ingest.migrate   # 4. create extension + table
docker compose run --rm app-askfolio python ingest/ingest_pgvector.py   # 5. ingest
docker compose up -d --build                                    # 6. start the bot
```

`python -m ingest.migrate` is idempotent: it creates the pgvector extension and the vector table, and never deletes or embeds anything.

Ingest options:

- **Default:** Deletes all existing vectors in the collection table, then ingests fresh.
- **`--append`:** Skip the delete step (can duplicate if run twice).
- **`--reinstall`:** Drop and recreate the collection table, then ingest.

---

**Offset persistence:** The last Telegram `update_id` is saved to `data/telegram_offset.txt` so after a restart the bot only fetches new updates.

**Embedding model:** `EMBEDDING_MODEL` is a Docker build arg (the model is baked into the image and loaded offline). Changing it requires a rebuild (`docker compose build app-askfolio`) and a re-ingest; if `VECTOR_SIZE` changes too, re-ingest with `--reinstall`.

**After changing code or dependencies:**
```bash
docker compose build app-askfolio && docker compose up -d app-askfolio
```

## Meeting scheduling flow

```
Recruiter: I'd like to schedule a meeting with Emmanuel

Bot: Here are Emmanuel's available time slots (all times in New York (EDT)):

1. Monday, May 11 at 10:00 AM (EDT)
2. Monday, May 11 at 02:00 PM (EDT)
3. Tuesday, May 12 at 09:00 AM (EDT)
...

Reply with the number of your preferred slot.
If you'd like to see these in a different timezone, just say so —
for example: "show in Pacific Time" or "convert to CET".

Recruiter: show in Pacific Time

Bot: Here are the same slots converted to Los Angeles (PDT):

1. Monday, May 11 at 07:00 AM (PDT)
...

Recruiter: 2

Bot: Great choice! I've noted: Monday, May 11 at 11:00 AM (PDT).

Could you share your name and email so I can send you a calendar invite?

Name: Your Name
Email: your@email.com

Recruiter: Name: Jane Smith
           Email: jane@company.com

Bot: Meeting booked!

Date: Monday, May 11 at 11:00 AM (PDT)
Name: Jane Smith
Email: jane@company.com

A calendar invite has been sent to jane@company.com.
```

**Timezone conversion:** The bot understands fuzzy timezone names — `"pacific"`, `"PST"`, `"Tokyo"`, `"CET"`, `"Buenos Aires"`, full IANA keys, and more. If a name can't be resolved, it asks the recruiter to try again.

**Cancellation:** At any point, saying "cancel", "never mind", "no thanks", etc. exits the flow gracefully.

## LLM configuration notes

`LLM_REQUEST_TIMEOUT` defaults to 60 seconds; raise it for slow self-hosted models. Use **`RAG_TOP_K`** to control how many chunks are passed into the prompt (default 20).

## Answer check (eval)

`scripts/run_eval.py` asks the bot's own Q&A path (same retrieval, prompt, `max_tokens` and provider settings) a list of questions and checks each answer for text it must and must not contain. Use it to compare models and to catch regressions after changing prompts, retrieval or the knowledge JSON.

Write a golden YAML file from your own CV; `evals/golden.example.yaml` shows the format. Other `evals/golden*.yaml` files and `evals/results/` are git-ignored. The owner's email and phone from the knowledge JSON are always checked as forbidden (including spellings like `[at]`), and are removed from every reply that is printed or saved, so never write them into the golden file.

It needs the vector table (run the ingest first) and an LLM key, and it calls the paid API, so it is not part of CI. Run it from the host with the localhost database override described in [Running without Docker](#running-without-docker):

```bash
uv run python scripts/run_eval.py --golden path/to/golden.yaml                 # LLM_MODEL, with fallbacks
uv run python scripts/run_eval.py --golden path/to/golden.yaml --model a/b --model c/d   # compare, fallbacks off
uv run python scripts/run_eval.py --golden path/to/golden.yaml --case years-experience  # one case
```

A case with `language: en` or `language: es` also fails when the reply is in the other language. The runner refuses to start when the knowledge JSON lacks an email or a phone (pass `--allow-missing-contact` to run without that check).

Each case is `PASS`, `FAIL`, `ERROR` (the LLM call failed) or `MISROUTED` (the bot would send this message to the scheduling flow instead; this follows `BOOKING_ENABLED`, and `--booking` checks routing as if it were on). The summary per model counts passed cases, `contact` and `injection` cases, mean latency and cost when the provider reports it. Results are saved to `evals/results/<timestamp>.json`. Exit code: 0 when every case passed, 1 otherwise, 2 for a bad golden file or arguments.

## Rate limits

In-memory limits (reset on restart; day windows are UTC calendar days). A message over a limit gets one notice, then is ignored silently.

| Var | Scope |
|-----|-------|
| `RATE_LIMIT_PER_CHAT_PER_MINUTE` | Messages per chat in any sliding 60 seconds (default 5) |
| `RATE_LIMIT_PER_CHAT_PER_DAY` | Messages per chat per UTC day (default 50) |
| `RATE_LIMIT_GLOBAL_PER_DAY` | Messages across all chats per UTC day (default 45). It counts messages, not upstream requests, so keep it under your provider's daily cap with headroom |
| `RATE_LIMIT_BOOKINGS_PER_CHAT_PER_DAY` | Meetings one chat can book per UTC day (default 1) |
| `RATE_LIMIT_BOOKINGS_GLOBAL_PER_DAY` | Meetings booked across all chats per UTC day (default 5) |

## Health and operations

- **Heartbeat:** the bot touches `HEARTBEAT_FILE` (default `/tmp/heartbeat`) after startup and after each successful poll cycle.
- **Healthcheck:** `python -m app.healthcheck` exits 0 while the heartbeat is fresh and 1 otherwise. The production compose uses it as the container healthcheck.
- **Shutdown:** the app exits with status 0 on SIGTERM, and the compose files run it with `init: true` so the signal reaches Python.
- **Privacy check:** `uv run python scripts/check_index_pii.py` fails (exit 1) if the owner's email or phone from the knowledge JSON is still present in the vector table. It prints only match counts, never the values.
- **Production compose:** `deploy/compose.yml` is the production stack. It pulls the app image from GHCR (`IMAGE_TAG`), takes host paths from `ASKFOLIO_*` env vars (`ASKFOLIO_STATE_DIR`, `ASKFOLIO_RESOURCES_DIR`, `ASKFOLIO_DB_DIR`), and runs the app read-only with all capabilities dropped. CI and deploy live in `.github/workflows`.

## Usage analytics (optional)

Optional, off unless `POSTHOG_KEY` and `ANALYTICS_SALT` are set; sends the question with emails and phone numbers removed, an anonymous id, and the outcome; never the answer.

| Var | Purpose |
|-----|---------|
| `POSTHOG_KEY` | PostHog project API key |
| `POSTHOG_HOST` | PostHog host (default `https://us.i.posthog.com`); must be `https://`, or analytics stays off |
| `ANALYTICS_SALT` | Secret for hashing chat ids into anonymous ids (`openssl rand -hex 32`) |
| `ANALYTICS_ENV` | `environment` property on every event (default `local`) |
| `ANALYTICS_INTERNAL_CHAT_IDS` | Comma-separated chat ids whose events are tagged `is_internal` |
| `SEEN_CHATS_FILE` | Hashed ids of chats already seen (default `data/seen_chats`) |

Events:

- `question_asked`: `question`, `outcome` (`answered`, `rate_limited`, `llm_unavailable` or `error`), `rate_limit_scope` when rate-limited, `latency_ms` when answered.
- `recruiter_first_seen`: the first message from a chat.

Every event carries `project_source`, `environment` and `is_internal`, with no IP and no person profile. Queued events are flushed when the bot stops.

## LangSmith tracing (optional)

This bot can send traces to LangSmith for better observability (runs, nested spans, latency, errors).

### 1) Create a LangSmith API key

1. Open [LangSmith](https://smith.langchain.com/) and sign in.
2. Click your profile/avatar → **Settings** → **API Keys** → **Create API Key**.
3. Copy the key (starts with `lsv2_...`) and store it safely.

### 2) Configure environment

In `.env`:

```env
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=askfolio-dev
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

Tracing is **optional** and only turns on when `LANGSMITH_API_KEY` is set.

### 3) Rebuild/restart app

```bash
docker compose up -d --build app-askfolio
```

What gets traced:
- Telegram message metadata (`chat_id`, `message_id`, message text)
- Retrieved RAG context (length + bounded context snippet)
- LLM HTTP call span
- Telegram `sendMessage` span

## Troubleshooting

### `ZoneInfoNotFoundError` for `CALENDAR_TIMEZONE`

Use a full IANA timezone key. Common examples:

| Region | Key |
|--------|-----|
| Argentina | `America/Argentina/Buenos_Aires` |
| Eastern US | `America/New_York` |
| Pacific US | `America/Los_Angeles` |
| London | `Europe/London` |
| Madrid | `Europe/Madrid` |

Full list: [Wikipedia — tz database time zones](https://en.wikipedia.org/wiki/List_of_tz_database_time_zones)

## Project layout

```
├── app/
│   ├── config.py              # Settings from env
│   ├── llm_client.py          # OpenRouter-aware chat completions client
│   ├── rate_limit.py          # In-memory message and booking limiters
│   ├── healthcheck.py         # Container healthcheck (reads the heartbeat)
│   ├── graph.py               # LangGraph: get_telegram_updates → process_message
│   ├── conversation.py        # ConversationState, SchedulingMode (state machine types)
│   ├── calendar_client.py     # Google Calendar: slot fetching, meeting creation, tz resolution
│   ├── rag.py                 # pgvector retrieval
│   ├── telegram_client.py     # Telegram getUpdates / sendMessage
│   └── prompts/
│       └── system_prompt.txt, user_prompt.txt
├── ingest/
│   ├── ingest_pgvector.py     # Ingest JSON → pgvector
│   ├── migrate.py             # Idempotent schema setup
│   └── json_to_documents.py
├── resources/
│   └── json/                  # cv.json, github_projects.json
├── scripts/
│   ├── authorize_calendar.py  # One-time Google OAuth — run locally to generate secrets/token.json
│   ├── check_index_pii.py     # Fail if owner contact data is in the index
│   ├── fetch_github_projects.py
│   └── run_eval.py            # Answer check against a golden YAML set
├── evals/
│   └── golden.example.yaml    # Example golden set (real sets stay out of the repo)
├── deploy/compose.yml         # Production stack (GHCR image)
├── main.py
├── docker-compose.yml         # Dev stack
└── Dockerfile
```

The database uses the **official [pgvector/pgvector](https://hub.docker.com/r/pgvector/pgvector)** image (`pg17`). Upgrading from an older setup on `pg15`: the dev stack uses a new volume (`askfolio_pg17_data`), so run the ingest again after the first `docker compose up`.

## Running without Docker

```bash
uv sync
docker compose up db-askfolio -d          # Postgres only
uv run python -m ingest.migrate
uv run python ingest/ingest_pgvector.py
uv run python main.py
```

The dev `db-askfolio` publishes no port. To reach it from the host, add a `docker-compose.override.yml` (localhost-only binding):

```yaml
services:
  db-askfolio:
    ports: ["127.0.0.1:5432:5432"]
```

then set `POSTGRES_HOST=localhost` and `POSTGRES_PORT=5432` in `.env`. Alternatively run the app inside the stack with `docker compose run --rm app-askfolio python main.py`.

## License

MIT
