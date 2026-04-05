import logging
import requests
from typing import Any

from app.config import TELEGRAM_BOT_TOKEN, GET_UPDATES_TIMEOUT

BASE = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
logger = logging.getLogger(__name__)


def get_updates(offset: int | None = None) -> list[dict[str, Any]]:
    """Fetch new updates (messages). Long-poll: request blocks up to GET_UPDATES_TIMEOUT seconds for new updates."""
    url = f"{BASE}/getUpdates"
    params: dict[str, Any] = {"timeout": GET_UPDATES_TIMEOUT}
    if offset is not None:
        params["offset"] = offset
    try:
        # Keep read timeout above long-poll timeout to reduce false client timeouts.
        r = requests.get(url, params=params, timeout=(10, GET_UPDATES_TIMEOUT + 30))
        r.raise_for_status()
        data = r.json()
    except requests.exceptions.Timeout as exc:
        logger.warning("[telegram_client] getUpdates timeout: %s", exc)
        return []
    except requests.exceptions.RequestException as exc:
        logger.warning("[telegram_client] getUpdates request failed: %s", exc)
        return []
    if not data.get("ok"):
        return []
    return data.get("result", [])


def send_message(
    chat_id: int, text: str, *, reply_to_message_id: int | None = None
) -> None:
    url = f"{BASE}/sendMessage"
    payload: dict[str, Any] = {"chat_id": chat_id, "text": text}
    if reply_to_message_id is not None:
        payload["reply_parameters"] = {"message_id": reply_to_message_id}
    r = requests.post(url, json=payload, timeout=10)
    r.raise_for_status()
