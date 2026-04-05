# Telegram RAG Agent (Recruiter bot)

MVP: A Telegram bot that answers recruiter-style questions using **RAG** over **pgvector**. Knowledge comes from a generic **`cv.json`** (profile / CV) plus **`github_projects.json`**. Built with **LangGraph** nodes so you can extend to more agents or tools later.

## Features

- **Single-agent LangGraph**: Two nodes — `get_telegram_updates` (fetch new messages) and `llm_reply` (retrieve context from pgvector, call LLM, send reply).
- **RAG**: `resources/json/cv.json` and `resources/json/github_projects.json` are ingested into pgvector (section-aware chunking). Each user message is answered using retrieved context.
- **LLM**: **Ollama** on the host (recommended for local dev), or any **OpenAI-compatible** HTTP API (cloud) when configured.

## CV data (`resources/json/cv.json`)

This file is your **structured CV / profile** for retrieval. It is **not** tied to a person’s name in the filename—use **`cv.json`** (or override with `INGEST_KNOWLEDGE_JSON`).

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

1. **Copy environment file**
   ```bash
   cp .env.example .env
   ```
   Set `TELEGRAM_BOT_TOKEN` in `.env` (create a bot via [@BotFather](https://t.me/BotFather)).

2. **Prepare JSON**
   - Add or edit **`resources/json/cv.json`** with your profile (sections above).
   - Add **`resources/json/github_projects.json`** (e.g. run `python scripts/fetch_github_projects.py` with `GITHUB_TOKEN` set to avoid rate limits).
   - Optional: set `INGEST_KNOWLEDGE_JSON` / `INGEST_GITHUB_JSON` in `.env` for custom paths.

3. **Local LLM: Ollama (recommended)**

   Install [Ollama](https://ollama.com/) on the host, pull a model (e.g. `ollama pull qwen2.5:14b`), and keep the service running (`systemctl enable --now ollama` on Linux).

   - Default API URL in this project points at **`http://host.docker.internal:11434/v1/chat/completions`** (Ollama’s **OpenAI-compatible** endpoint on port **11434**).
   - Set **`LLM_MODEL`** to the Ollama model tag (e.g. `qwen2.5:14b`).
   - Leave **`LLM_API_KEY`** empty for local Ollama.
   - If the app runs in Docker, Ollama must listen beyond localhost: set e.g. `OLLAMA_HOST=0.0.0.0:11434` in the Ollama systemd override so `host.docker.internal` can connect.

   Other OpenAI-compatible servers (different host/port) work too: set `LLM_API_URL` and `LLM_MODEL` accordingly.

4. **Start the stack**
   ```bash
   docker compose up --build -d
   ```

5. **Ingest into pgvector** (one-off, after DB is up)
   ```bash
   docker compose run --rm app-askfolio python ingest/ingest_pgvector.py
   ```
   - **Default:** Deletes all existing vectors in the collection table, then ingests fresh from `cv.json` and `github_projects.json`.
   - **`--append`:** Skip the delete step (can duplicate if run twice).
   - **`--reinstall`:** Drop and recreate the collection table, then ingest:
     ```bash
     docker compose run --rm app-askfolio python ingest/ingest_pgvector.py --reinstall
     ```

- **Offset persistence:** The last Telegram `update_id` is saved to `telegram_offset.txt` (or `TELEGRAM_OFFSET_FILE`) so after a restart the bot only fetches new updates.

- **After changing code or dependencies:**
  ```bash
  docker compose build app-askfolio && docker compose up -d app-askfolio
  ```

## LLM configuration summary

| Setup | `LLM_API_URL` (example) | `LLM_MODEL` | `LLM_API_KEY` |
|-------|---------------------------|-------------|---------------|
| **Ollama (Docker app → host)** | `http://host.docker.internal:11434/v1/chat/completions` | e.g. `qwen2.5:14b` | empty |
| **Ollama (app on host)** | `http://127.0.0.1:11434/v1/chat/completions` | same as `ollama list` | empty |
| **Cloud (OpenAI, DeepSeek, …)** | Provider chat completions URL | Provider model id | API key |

Use **`LLM_REQUEST_TIMEOUT`** for slow local models (default 600s). Use **`RAG_TOP_K`** to control how many chunks are passed into the prompt (default 20).

## LangSmith tracing (optional)

This bot can send traces to LangSmith for better observability than raw logs (runs, nested spans, latency, and errors).

### 1) Create a LangSmith API key

1. Open [LangSmith](https://smith.langchain.com/) and sign in.
2. Click your profile/avatar (top-right) and open **Settings**.
3. Go to **API Keys**.
4. Click **Create API Key**, copy the key (starts with `lsv2_...`), and store it safely.

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

What gets traced in this project:
- Telegram message metadata (`chat_id`, `message_id`, message text)
- Retrieved RAG context (length + bounded context snippet)
- LLM HTTP call span
- Telegram sendMessage span

### Troubleshooting: `ConnectTimeout` / `host.docker.internal:11434`

The app container talks to Ollama on the **host**. A **timeout** (not “connection refused”) usually means nothing is accepting TCP on the host address Docker uses—often because **Ollama only listens on `127.0.0.1`**, while traffic from the container arrives on another host interface.

1. **Bind Ollama on all interfaces** (Linux systemd example):
   - `sudo systemctl edit ollama.service` and add:
     ```ini
     [Service]
     Environment="OLLAMA_HOST=0.0.0.0:11434"
     ```
   - `sudo systemctl daemon-reload && sudo systemctl restart ollama`
2. On the **host**, confirm: `curl -sS http://127.0.0.1:11434/api/tags`
3. From the **app container**, confirm:  
   `docker compose exec app-askfolio python -c "import urllib.request; urllib.request.urlopen('http://host.docker.internal:11434/api/tags', timeout=5).read()"`

If step 3 still fails, check host firewalls and that `extra_hosts: host.docker.internal:host-gateway` is present in `docker-compose.yml` (already set for this project).

## Project layout

```
├── app/
│   ├── config.py           # Settings from env
│   ├── graph.py            # LangGraph: get_telegram_updates → llm_reply
│   ├── rag.py              # pgvector retrieval
│   ├── telegram_client.py  # Telegram getUpdates / sendMessage
│   └── prompts/
│       └── recruiter_prompt.txt
├── ingest/
│   ├── ingest_pgvector.py  # Ingest JSON → pgvector
│   └── json_to_documents.py
├── resources/
│   └── json/               # cv.json, github_projects.json
├── scripts/
│   └── fetch_github_projects.py
├── main.py
├── docker-compose.yml
└── Dockerfile
```

The database uses the **official [pgvector/pgvector](https://hub.docker.com/r/pgvector/pgvector)** image (`pg15`).

## Running without Docker

- Install dependencies: `pip install -r requirements.txt`
- Run Postgres with pgvector (e.g. `docker compose up db-askfolio -d`)
- Set `.env` with `POSTGRES_HOST=localhost`, `POSTGRES_PORT` matching `POSTGRES_PUBLISH_PORT` from Compose, and the same `POSTGRES_*` credentials as the db container (see `.env.example`)
- Run Ollama on the host; point `LLM_API_URL` at `http://127.0.0.1:11434/v1/chat/completions`
- Run ingestion: `python ingest/ingest_pgvector.py`
- Run bot: `python main.py`

## License

MIT
