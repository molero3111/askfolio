# Telegram RAG Agent (Recruiter bot)

MVP: A Telegram bot that answers questions about you (CV, experience, projects) using RAG over pgvector. Built with LangGraph nodes so you can extend to multiple agents or tools later.

## Features

- **Single-agent LangGraph**: Two nodes — `get_telegram_updates` (fetch new messages) and `llm_reply` (retrieve context from pgvector, call LLM, send reply).
- **RAG**: PDF(s) in `resources/pdfs/` are ingested into pgvector; each user message is answered using retrieved context.
- **LLM**: LM Studio locally (default) or any OpenAI-compatible API when `LLM_API_KEY` is set.

## Setup

1. **Copy environment file**
   ```bash
   cp .env.example .env
   ```
   Set `TELEGRAM_BOT_TOKEN` in `.env` (create a bot via [@BotFather](https://t.me/BotFather)).

2. **Add your CV PDF**  
   Place your PDF (CV + project info) in `resources/pdfs/`.

3. **Start the stack**
   ```bash
   docker compose up --build -d
   ```

4. **Ingest PDFs** (one-off, after DB is up)
   ```bash
   docker compose run --rm emmanuelai-app python ingest/ingest_pgvector.py
   ```
   - **First run or append:** Table is created if missing; documents are added. Re-running appends more documents to the existing table.
   - **Clean re-ingest (drop table and re-create):** use `--reinstall` to drop the collection table first, then create and ingest:
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
│   └── ingest_pgvector.py  # Ingest PDFs into pgvector
├── resources/
│   └── pdfs/               # Put your CV PDF here
├── main.py                 # Polling loop that invokes the graph
├── docker-compose.yml
└── Dockerfile
```

The database uses the **official [pgvector/pgvector](https://hub.docker.com/r/pgvector/pgvector)** image (`pg15`), so no custom DB image is built.

## Running without Docker

- Install dependencies: `pip install -r requirements.txt`
- Run Postgres with pgvector (e.g. use Docker only for DB: `docker compose up emmanuelai-db -d`)
- Set `.env` with `DB_CONNECTION_URL` pointing at `localhost:5432` (see `.env.example` for user/password/db name)
- Run ingestion: `python ingest/ingest_pgvector.py` (add `--reinstall` to drop the table and do a clean re-ingest)
- Run bot: `python main.py`

## License

MIT
