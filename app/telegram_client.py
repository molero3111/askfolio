import logging
from typing import Any

import requests

from app.config import GET_UPDATES_TIMEOUT, TELEGRAM_BOT_TOKEN

BASE = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
logger = logging.getLogger(__name__)


class TelegramError(RuntimeError):
    """Telegram API failure. The message never contains the request URL."""


def get_updates(offset: int | None = None) -> list[dict[str, Any]]:
    """Fetch new updates (messages). Long-poll: request blocks up to GET_UPDATES_TIMEOUT seconds for new updates."""
    url = f"{BASE}/getUpdates"
    params: dict[str, Any] = {"timeout": GET_UPDATES_TIMEOUT}
    if offset is not None:
        params["offset"] = offset
    try:
        # Keep read timeout above long-poll timeout to reduce false client timeouts.
        r = requests.get(url, params=params, timeout=(10, GET_UPDATES_TIMEOUT + 30))
    except requests.exceptions.ReadTimeout as exc:
        # Normal for a long poll with no new updates.
        logger.warning("[telegram_client] getUpdates read timeout (%s)", type(exc).__name__)
        return []
    except requests.RequestException as exc:
        # Never re-raise the original: its text carries the URL, which holds the bot token.
        raise TelegramError(f"getUpdates failed: {type(exc).__name__}") from None
    if r.status_code == 409:
        raise TelegramError(
            "getUpdates conflict (HTTP 409): another poller or a webhook is using this bot token"
        )
    if r.status_code >= 400:
        raise TelegramError(f"getUpdates failed: HTTP {r.status_code}")
    try:
        data = r.json()
    except ValueError:
        raise TelegramError("getUpdates failed: invalid JSON") from None
    if not data.get("ok"):
        raise TelegramError("getUpdates failed: ok=false")
    return data.get("result", [])


def send_message(
    chat_id: int, text: str, *, reply_to_message_id: int | None = None
) -> None:
    url = f"{BASE}/sendMessage"
    payload: dict[str, Any] = {"chat_id": chat_id, "text": text}
    if reply_to_message_id is not None:
        payload["reply_parameters"] = {"message_id": reply_to_message_id}
    try:
        r = requests.post(url, json=payload, timeout=10)
    except requests.RequestException as exc:
        raise TelegramError(f"sendMessage failed: {type(exc).__name__}") from None
    if r.status_code >= 400:
        raise TelegramError(f"sendMessage failed: HTTP {r.status_code}")
