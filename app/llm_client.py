"""Small OpenRouter-aware client for OpenAI-compatible chat completions."""

import logging
import time
from dataclasses import dataclass
from urllib.parse import urlparse

import requests

from app import config

logger = logging.getLogger(__name__)

_DEFAULT_RETRY_SLEEP = 2
_MAX_RETRY_SLEEP = 10


class LLMError(Exception):
    """Base class for LLM client failures."""


class LLMUnavailable(LLMError):
    """The LLM could not produce an answer (transient or routing failure)."""


class LLMCreditExhausted(LLMError):
    """The provider reported the credit limit is reached (HTTP 402)."""


@dataclass
class LLMResult:
    content: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    cost: float | None
    latency_s: float


def _is_openrouter(url: str) -> bool:
    host = urlparse(url).hostname or ""
    return host == "openrouter.ai" or host.endswith(".openrouter.ai")


def _error_message(resp) -> str:
    try:
        body = resp.json()
    except Exception:
        return ""
    err = body.get("error") if isinstance(body, dict) else None
    if isinstance(err, dict):
        return str(err.get("message") or "")
    return str(err or "")


def _retry_delay(resp) -> int:
    raw = resp.headers.get("Retry-After") if resp.headers else None
    try:
        return max(0, min(int(raw), _MAX_RETRY_SLEEP))
    except (TypeError, ValueError):
        return _DEFAULT_RETRY_SLEEP


def chat(
    messages: list[dict],
    *,
    model: str | None = None,
    session=None,
    sleep=time.sleep,
) -> LLMResult:
    """Send a chat completion request; retry once on 429/5xx."""
    session = session if session is not None else requests
    url = config.LLM_API_URL
    primary = model or config.LLM_MODEL
    openrouter = _is_openrouter(url)

    headers = {"Content-Type": "application/json"}
    if config.LLM_API_KEY:
        headers["Authorization"] = f"Bearer {config.LLM_API_KEY}"
    payload: dict = {
        "model": primary,
        "messages": messages,
        "max_tokens": config.LLM_MAX_TOKENS,
    }
    if openrouter:
        payload["provider"] = {"data_collection": config.LLM_DATA_COLLECTION}
        headers["X-OpenRouter-Title"] = config.LLM_APP_NAME
        if config.LLM_APP_URL:
            headers["HTTP-Referer"] = config.LLM_APP_URL
        if config.LLM_FALLBACK_MODELS:
            payload["models"] = [primary, *config.LLM_FALLBACK_MODELS]

    start = time.monotonic()
    resp = None
    for attempt in range(2):
        try:
            resp = session.post(
                url, headers=headers, json=payload, timeout=config.LLM_REQUEST_TIMEOUT
            )
        except requests.Timeout as exc:
            raise LLMUnavailable("LLM request timed out") from exc
        except requests.RequestException as exc:
            raise LLMUnavailable(f"LLM request failed: {type(exc).__name__}") from exc

        status = resp.status_code
        if status == 402:
            logger.error("[llm] OpenRouter credit limit reached")
            raise LLMCreditExhausted("credit limit reached")
        if status in (400, 404):
            message = _error_message(resp)
            lowered = message.lower()
            if "data policy" in lowered or "no endpoints found" in lowered:
                logger.error(
                    "[llm] no provider for %s matches data_collection=%s",
                    primary,
                    config.LLM_DATA_COLLECTION,
                )
                raise LLMUnavailable("no provider matches the data policy")
            raise LLMUnavailable(f"LLM request rejected (HTTP {status})")
        if status == 429 or status >= 500:
            if attempt == 0:
                delay = _retry_delay(resp)
                logger.warning("[llm] HTTP %s, retrying in %ss", status, delay)
                sleep(delay)
                continue
            raise LLMUnavailable(f"LLM unavailable (HTTP {status})")
        if status >= 400:
            raise LLMUnavailable(f"LLM request failed (HTTP {status})")
        break

    try:
        body = resp.json()
    except Exception as exc:
        raise LLMUnavailable("LLM returned a non-JSON response") from exc
    if not isinstance(body, dict) or body.get("error") or not body.get("choices"):
        raise LLMUnavailable("LLM response had no choices")
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMUnavailable("LLM response malformed") from exc
    if not isinstance(content, str) or not content.strip():
        raise LLMUnavailable("LLM returned empty content")

    latency = time.monotonic() - start
    usage = body.get("usage") or {}
    result = LLMResult(
        content=content.strip(),
        model=body.get("model") or primary,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        cost=usage.get("cost"),
        latency_s=latency,
    )
    logger.info(
        "[llm] model=%s prompt=%s completion=%s cost=%s latency=%.1fs",
        result.model,
        result.prompt_tokens,
        result.completion_tokens,
        result.cost,
        result.latency_s,
    )
    return result
