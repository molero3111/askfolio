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
- **LLM**: **Ollama** on the host (recommended for local dev), or any **OpenAI-compatible** HTTP API (cloud).

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

The ingest pipeline turns each logical block into documents with metadata so search stays section-aware. Match the **shape** expected by `ingest/json_to_documents.py` (see that file for exact keys), or extend the parser if you add fields.

## Setup

### 1. Copy environment file

```bash
cp .env.example .env
```

Set `TELEGRAM_BOT_TOKEN` in `.env` (create a bot via [@BotFather](https://t.me/BotFather)).

### 2. Prepare JSON

- Add or edit **`resources/json/cv.json`** with your profile (sections above).
- Add **`resources/json/github_projects.json`** (e.g. run `python scripts/fetch_github_projects.py` with `GITHUB_TOKEN` set to avoid rate limits).
- Optional: set `INGEST_KNOWLEDGE_JSON` / `INGEST_GITHUB_JSON` in `.env` for custom paths.

### 3. Local LLM: Ollama (recommended)

Install [Ollama](https://ollama.com/) on the host, pull a model (e.g. `ollama pull qwen2.5:14b`), and keep the service running (`systemctl enable --now ollama` on Linux).

- Default API URL points at **`http://host.docker.internal:11434/v1/chat/completions`** (Ollama's OpenAI-compatible endpoint).
- Set **`LLM_MODEL`** to the Ollama model tag (e.g. `qwen2.5:14b`).
- Leave **`LLM_API_KEY`** empty for local Ollama.
- If the app runs in Docker, Ollama must listen beyond localhost: set `OLLAMA_HOST=0.0.0.0:11434` in the Ollama systemd override.

Other OpenAI-compatible servers (different host/port) work too: set `LLM_API_URL` and `LLM_MODEL` accordingly.

### 4. Google Calendar integration (optional — for meeting scheduling)

Scheduling is **opt-in**: the bot works fine without it. To enable it, follow these steps once.

#### 4a. Google Cloud Console setup

1. Go to [console.cloud.google.com](https://console.cloud.google.com/)
2. **Create a project** — top navbar dropdown → "New Project" → name it → Create.
3. **Enable Calendar API** — search "Google Calendar API" → click it → "Enable".
4. **Configure OAuth consent screen** — "APIs & Services" → "OAuth consent screen" → choose **External** → fill in App name and your email → Save and Continue.
5. **Add yourself as a test user** — on the OAuth consent screen page → "Test users" → add your Google account email → Save.
6. **Create credentials** — "APIs & Services" → "Credentials" → "Create Credentials" → "OAuth client ID" → Application type: **Desktop app** → Create.
7. **Download** — click the download icon next to the credential → save it as **`credentials.json`** in the project root.

#### 4b. Create a dedicated test calendar (recommended)

Using a separate calendar keeps the bot away from your real schedule.

1. Go to [calendar.google.com](https://calendar.google.com)
2. Left sidebar → "Other calendars" → **+** → "Create new calendar" → name it (e.g. `Emmanuel Bot`).
3. Hover over it → three dots → "Settings and sharing" → scroll to **"Integrate calendar"** → copy the **Calendar ID** (looks like `abc123@group.calendar.google.com`).
4. Set in `.env`: `GOOGLE_CALENDAR_ID=abc123@group.calendar.google.com`

#### 4c. Authorize and generate token

Run this **once locally** (not in Docker — it opens a browser):

```bash
# Using the project venv:
venv/bin/pip install google-auth-oauthlib   # if not already installed
venv/bin/python scripts/authorize_calendar.py

# Or install it temporarily in any local env:
pip install google-auth-oauthlib
python scripts/authorize_calendar.py
```

A browser tab opens. Sign in, grant Calendar access, and `token.json` is written to the project root. The Docker container reads it automatically via the volume mount — **you never need to re-run this** unless you revoke access.

#### 4d. Calendar environment variables

In `.env`:

```env
GOOGLE_CREDENTIALS_PATH=credentials.json
GOOGLE_TOKEN_PATH=token.json
GOOGLE_CALENDAR_ID=primary            # or your dedicated calendar ID
CALENDAR_TIMEZONE=America/Argentina/Buenos_Aires   # developer's local tz (for working hours)
CALENDAR_DISPLAY_TIMEZONE=America/New_York         # tz shown to recruiters by default
CALENDAR_SLOT_DURATION_MINUTES=60
CALENDAR_WORKING_HOURS_START=9
CALENDAR_WORKING_HOURS_END=18
CALENDAR_DAYS_AHEAD=7
```

`CALENDAR_TIMEZONE` must be a valid IANA key (e.g. `America/Argentina/Buenos_Aires`, `Europe/Madrid`). `CALENDAR_DISPLAY_TIMEZONE` controls how slot times appear to recruiters; they can always ask the bot to convert to their preferred timezone.

### 5. Start the stack

```bash
docker compose up --build -d
```

### 6. Ingest into pgvector (one-off, after DB is up)

```bash
docker compose run --rm app-askfolio python ingest/ingest_pgvector.py
```

- **Default:** Deletes all existing vectors in the collection table, then ingests fresh.
- **`--append`:** Skip the delete step (can duplicate if run twice).
- **`--reinstall`:** Drop and recreate the collection table, then ingest.

---

**Offset persistence:** The last Telegram `update_id` is saved to `telegram_offset.txt` so after a restart the bot only fetches new updates.

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

## LLM configuration summary

| Setup | `LLM_API_URL` (example) | `LLM_MODEL` | `LLM_API_KEY` |
|-------|---------------------------|-------------|---------------|
| **Ollama (Docker app → host)** | `http://host.docker.internal:11434/v1/chat/completions` | e.g. `qwen2.5:14b` | empty |
| **Ollama (app on host)** | `http://127.0.0.1:11434/v1/chat/completions` | same as `ollama list` | empty |
| **Cloud (OpenAI, DeepSeek, …)** | Provider chat completions URL | Provider model id | API key |

Use **`LLM_REQUEST_TIMEOUT`** for slow local models (default 600s). Use **`RAG_TOP_K`** to control how many chunks are passed into the prompt (default 20).

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

### `ConnectTimeout` / `host.docker.internal:11434`

The app container talks to Ollama on the **host**. A timeout usually means Ollama only listens on `127.0.0.1`.

1. **Bind Ollama on all interfaces** (Linux systemd):
   ```bash
   sudo systemctl edit ollama.service
   ```
   Add:
   ```ini
   [Service]
   Environment="OLLAMA_HOST=0.0.0.0:11434"
   ```
   Then: `sudo systemctl daemon-reload && sudo systemctl restart ollama`
2. On the **host**: `curl -sS http://127.0.0.1:11434/api/tags`
3. From the **app container**:
   ```bash
   docker compose exec app-askfolio python -c \
     "import urllib.request; urllib.request.urlopen('http://host.docker.internal:11434/api/tags', timeout=5).read()"
   ```

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
│   ├── graph.py               # LangGraph: get_telegram_updates → process_message
│   ├── conversation.py        # ConversationState, SchedulingMode (state machine types)
│   ├── calendar_client.py     # Google Calendar: slot fetching, meeting creation, tz resolution
│   ├── rag.py                 # pgvector retrieval
│   ├── telegram_client.py     # Telegram getUpdates / sendMessage
│   └── prompts/
│       └── recruiter_prompt.txt
├── ingest/
│   ├── ingest_pgvector.py     # Ingest JSON → pgvector
│   └── json_to_documents.py
├── resources/
│   └── json/                  # cv.json, github_projects.json
├── scripts/
│   ├── authorize_calendar.py  # One-time Google OAuth — run locally to generate token.json
│   └── fetch_github_projects.py
├── main.py
├── docker-compose.yml
└── Dockerfile
```

The database uses the **official [pgvector/pgvector](https://hub.docker.com/r/pgvector/pgvector)** image (`pg15`).

## Running without Docker

```bash
pip install -r requirements.txt
docker compose up db-askfolio -d          # Postgres only
python ingest/ingest_pgvector.py
python main.py
```

Set `.env` `POSTGRES_HOST=localhost` and `POSTGRES_PORT=5432` if running Postgres via Docker but the app on the host. Point `LLM_API_URL` at `http://127.0.0.1:11434/v1/chat/completions` for local Ollama.

## License

MIT
