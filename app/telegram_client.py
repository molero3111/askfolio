import requests
from typing import Any

from app.config import TELEGRAM_BOT_TOKEN, GET_UPDATES_TIMEOUT

BASE = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"


def get_updates(offset: int | None = None) -> list[dict[str, Any]]:
    """Fetch new updates (messages). Long-poll: request blocks up to GET_UPDATES_TIMEOUT seconds for new updates."""
    url = f"{BASE}/getUpdates"
    params: dict[str, Any] = {"timeout": GET_UPDATES_TIMEOUT}
    if offset is not None:
        params["offset"] = offset
    r = requests.get(url, params=params, timeout=GET_UPDATES_TIMEOUT + 10)
    r.raise_for_status()
    data = r.json()
    if not data.get("ok"):
        return []
    return data.get("result", [])


def send_message(chat_id: int, text: str) -> None:
    url = f"{BASE}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    r = requests.post(url, json=payload, timeout=10)
    r.raise_for_status()
