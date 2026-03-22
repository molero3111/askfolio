# Telegram RAG Agent (Recruiter bot)

MVP: A Telegram bot that answers questions about you (CV, experience, projects) using RAG over pgvector. Built with LangGraph nodes so you can extend to multiple agents or tools later.

## Features

- **Single-agent LangGraph**: Two nodes — `get_telegram_updates` (fetch new messages) and `llm_reply` (retrieve context from pgvector, call LLM, send reply).
- **RAG**: `resources/json/emmanuel_molero_knowledge_base.json` and `resources/json/github_projects.json` are ingested into pgvector (section-aware chunking); each user message is answered using retrieved context.
- **LLM**: LM Studio locally (default) or any OpenAI-compatible API when `LLM_API_KEY` is set.

## Setup

1. **Copy environment file**
   ```bash
   cp .env.example .env
   ```
   Set `TELEGRAM_BOT_TOKEN` in `.env` (create a bot via [@BotFather](https://t.me/BotFather)).

2. **Knowledge + GitHub JSON**  
   Ensure `resources/json/emmanuel_molero_knowledge_base.json` and `resources/json/github_projects.json` exist (generate the latter with `python scripts/fetch_github_projects.py` if needed). Optional env: `INGEST_KNOWLEDGE_JSON`, `INGEST_GITHUB_JSON`.

3. **Start the stack**
   ```bash
   docker compose up --build -d
   ```

4. **Ingest into pgvector** (one-off, after DB is up)
   ```bash
   docker compose run --rm emmanuelai-app python ingest/ingest_pgvector.py
   ```
   - **Default:** Deletes all existing vectors in the collection table, then ingests fresh from the two JSON files (no duplicate runs).
   - **`--append`:** Skip the delete step; new chunks are added on top (can duplicate if you run twice).
   - **`--reinstall`:** Drop and recreate the collection table, then ingest (use if schema/table is broken):
     ```bash
     docker compose run --rm emmanuelai-app python ingest/ingest_pgvector.py --reinstall
     ```

- **Offset persistence:** The last Telegram `update_id` is saved to `telegram_offset.txt` (or `TELEGRAM_OFFSET_FILE`) so after a restart the bot only requests new updates. No Redis needed.

- **After changing code or dependencies:** Rebuild the app image so the running bot has the latest code and packages:
  ```bash
  docker compose build emmanuelai-app && docker compose up -d emmanuelai-app
  ```

## LLM configuration

- **Local (LM Studio)**  
  Leave `LLM_API_KEY` empty. Set:
  - `LLM_API_URL=http://host.docker.internal:1234/v1/chat/completions`
  - `LLM_MODEL` to your model name in LM Studio  
  Enable “Serve on local network” (and CORS if needed) in LM Studio.

- **Cloud (OpenAI, DeepSeek, etc.)**  
  Set `LLM_API_KEY` and `LLM_API_URL` (e.g. `https://api.openai.com/v1/chat/completions` or DeepSeek endpoint). Set `LLM_MODEL` to the model name.

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
│   └── json/               # knowledge base + github_projects.json
├── main.py                 # Polling loop that invokes the graph
├── docker-compose.yml
└── Dockerfile
```

The database uses the **official [pgvector/pgvector](https://hub.docker.com/r/pgvector/pgvector)** image (`pg15`), so no custom DB image is built.

## Running without Docker

- Install dependencies: `pip install -r requirements.txt`
- Run Postgres with pgvector (e.g. use Docker only for DB: `docker compose up emmanuelai-db -d`)
- Set `.env` with `DB_CONNECTION_URL` pointing at `localhost:5432` (see `.env.example` for user/password/db name)
- Run ingestion: `python ingest/ingest_pgvector.py` (default clears vectors then reloads JSON; `--append` to skip clear; `--reinstall` to drop table)
- Run bot: `python main.py`

## License

MIT
