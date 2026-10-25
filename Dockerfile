FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.21 /uv /uvx /usr/local/bin/

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    PATH=/opt/venv/bin:$PATH \
    HF_HOME=/opt/models \
    SENTENCE_TRANSFORMERS_HOME=/opt/models

# Embedding model is a build arg: changing it needs a rebuild and a re-ingest.
ARG EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
ENV EMBEDDING_MODEL=$EMBEDDING_MODEL

# Dependencies first so this layer is cached until the lockfile changes
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev

# Bake the embedding model into the image, then run offline
RUN python -c "import os; from sentence_transformers import SentenceTransformer; SentenceTransformer(os.environ['EMBEDDING_MODEL'])" \
    && chmod -R a+rX /opt/models
ENV HF_HUB_OFFLINE=1

# Non-root runtime user
RUN groupadd -g 10001 app && useradd -u 10001 -g 10001 -M -s /usr/sbin/nologin app

COPY . .

USER 10001:10001

CMD ["python", "main.py"]
