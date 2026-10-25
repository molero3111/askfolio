import logging
from typing import Any

import requests

from app.config import GET_UPDATES_TIMEOUT, TELEGRAM_BOT_TOKEN

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
    except requests.exceptions.ReadTimeout as exc:
        # Normal for a long poll with no new updates.
        logger.warning("[telegram_client] getUpdates read timeout: %s", exc)
        return []
    # Any other failure (connect error, DNS, 4xx/5xx, ok:false) raises so the
    # caller backs off and the heartbeat goes stale during a sustained outage.
    r.raise_for_status()
    data = r.json()
    if not data.get("ok"):
        raise RuntimeError(f"getUpdates returned ok=false: {str(data)[:200]}")
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
