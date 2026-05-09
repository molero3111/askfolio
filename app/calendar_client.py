import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from app.config import (
    CALENDAR_DAYS_AHEAD,
    CALENDAR_DISPLAY_TIMEZONE,
    CALENDAR_SLOT_DURATION_MINUTES,
    CALENDAR_TIMEZONE,
    CALENDAR_WORKING_HOURS_END,
    CALENDAR_WORKING_HOURS_START,
    GOOGLE_CALENDAR_ID,
    GOOGLE_TOKEN_PATH,
)

logger = logging.getLogger(__name__)

_SCOPES = ["https://www.googleapis.com/auth/calendar"]

# Fuzzy name → IANA key. Covers the most common recruiter timezones.
_TZ_ALIASES: dict[str, str] = {
    # United States
    "eastern": "America/New_York",
    "eastern time": "America/New_York",
    "est": "America/New_York",
    "edt": "America/New_York",
    "et": "America/New_York",
    "new york": "America/New_York",
    "central": "America/Chicago",
    "central time": "America/Chicago",
    "cst": "America/Chicago",
    "cdt": "America/Chicago",
    "ct": "America/Chicago",
    "chicago": "America/Chicago",
    "mountain": "America/Denver",
    "mountain time": "America/Denver",
    "mst": "America/Denver",
    "mdt": "America/Denver",
    "mt": "America/Denver",
    "denver": "America/Denver",
    "pacific": "America/Los_Angeles",
    "pacific time": "America/Los_Angeles",
    "pst": "America/Los_Angeles",
    "pdt": "America/Los_Angeles",
    "pt": "America/Los_Angeles",
    "los angeles": "America/Los_Angeles",
    "la": "America/Los_Angeles",
    "alaska": "America/Anchorage",
    "akst": "America/Anchorage",
    "hawaii": "Pacific/Honolulu",
    "hst": "Pacific/Honolulu",
    # Canada
    "toronto": "America/Toronto",
    "vancouver": "America/Vancouver",
    # UK / Ireland
    "london": "Europe/London",
    "gmt": "Europe/London",
    "bst": "Europe/London",
    "uk": "Europe/London",
    "ireland": "Europe/Dublin",
    # Europe
    "paris": "Europe/Paris",
    "cet": "Europe/Paris",
    "cest": "Europe/Paris",
    "berlin": "Europe/Berlin",
    "germany": "Europe/Berlin",
    "madrid": "Europe/Madrid",
    "spain": "Europe/Madrid",
    "amsterdam": "Europe/Amsterdam",
    "netherlands": "Europe/Amsterdam",
    "rome": "Europe/Rome",
    "italy": "Europe/Rome",
    "stockholm": "Europe/Stockholm",
    "sweden": "Europe/Stockholm",
    "zurich": "Europe/Zurich",
    "switzerland": "Europe/Zurich",
    "brussels": "Europe/Brussels",
    "warsaw": "Europe/Warsaw",
    "poland": "Europe/Warsaw",
    "lisbon": "Europe/Lisbon",
    "portugal": "Europe/Lisbon",
    "helsinki": "Europe/Helsinki",
    "athens": "Europe/Athens",
    "bucharest": "Europe/Bucharest",
    "moscow": "Europe/Moscow",
    "msk": "Europe/Moscow",
    # Middle East
    "dubai": "Asia/Dubai",
    "gulf": "Asia/Dubai",
    # Asia
    "india": "Asia/Kolkata",
    "kolkata": "Asia/Kolkata",
    "mumbai": "Asia/Kolkata",
    "delhi": "Asia/Kolkata",
    "ist": "Asia/Kolkata",
    "singapore": "Asia/Singapore",
    "sgt": "Asia/Singapore",
    "hong kong": "Asia/Hong_Kong",
    "hkt": "Asia/Hong_Kong",
    "beijing": "Asia/Shanghai",
    "shanghai": "Asia/Shanghai",
    "china": "Asia/Shanghai",
    "tokyo": "Asia/Tokyo",
    "japan": "Asia/Tokyo",
    "jst": "Asia/Tokyo",
    "seoul": "Asia/Seoul",
    "korea": "Asia/Seoul",
    "kst": "Asia/Seoul",
    # Oceania
    "sydney": "Australia/Sydney",
    "aest": "Australia/Sydney",
    "aedt": "Australia/Sydney",
    "melbourne": "Australia/Melbourne",
    "brisbane": "Australia/Brisbane",
    "perth": "Australia/Perth",
    "awst": "Australia/Perth",
    "auckland": "Pacific/Auckland",
    "nzst": "Pacific/Auckland",
    # South America
    "buenos aires": "America/Argentina/Buenos_Aires",
    "argentina": "America/Argentina/Buenos_Aires",
    "sao paulo": "America/Sao_Paulo",
    "brazil": "America/Sao_Paulo",
    "bogota": "America/Bogota",
    "colombia": "America/Bogota",
    "santiago": "America/Santiago",
    "chile": "America/Santiago",
    "lima": "America/Lima",
    "peru": "America/Lima",
    "mexico city": "America/Mexico_City",
    "mexico": "America/Mexico_City",
    # UTC
    "utc": "UTC",
    "universal": "UTC",
    "z": "UTC",
}


def resolve_timezone(text: str) -> str | None:
    """Map a fuzzy timezone string to a valid IANA key using the alias dict.

    The LLM (_resolve_timezone_llm in graph.py) is the primary resolver by default.
    This function acts as an optional fast-path only when TIMEZONE_ALIAS_FAST_PATH=true.
    Tries: alias map (whole phrase, then word-by-word), then direct ZoneInfo lookup.
    Returns None if nothing matches.
    """
    cleaned = text.strip()

    # Try alias on whole input
    lower = cleaned.lower()
    if lower in _TZ_ALIASES:
        return _TZ_ALIASES[lower]

    # Try alias on each word and consecutive two-word pairs
    words = lower.split()
    for i, word in enumerate(words):
        if word in _TZ_ALIASES:
            return _TZ_ALIASES[word]
        if i < len(words) - 1:
            pair = f"{word} {words[i + 1]}"
            if pair in _TZ_ALIASES:
                return _TZ_ALIASES[pair]

    # Try raw string as a direct IANA key
    try:
        ZoneInfo(cleaned)
        return cleaned
    except Exception:
        pass

    return None


def format_slot_label(slot_start_iso: str, display_tz: ZoneInfo) -> str:
    """Format a UTC ISO timestamp as a human-readable label in the given display timezone."""
    dt = datetime.fromisoformat(slot_start_iso).astimezone(display_tz)
    tz_abbr = dt.strftime("%Z")
    return dt.strftime(f"%A, %B %d at %I:%M %p ({tz_abbr})")


def is_calendar_configured() -> bool:
    return Path(GOOGLE_TOKEN_PATH).exists()


def _get_credentials() -> Credentials:
    token_path = Path(GOOGLE_TOKEN_PATH)
    creds = None

    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), _SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            token_path.write_text(creds.to_json())
        else:
            raise RuntimeError(
                f"Google Calendar token not found at {token_path}. "
                "Run scripts/authorize_calendar.py locally to generate it."
            )

    return creds


def _service():
    return build("calendar", "v3", credentials=_get_credentials())


def get_available_slots(display_tz_key: str | None = None) -> list[dict]:
    """
    Return up to 6 free meeting slots.
    Working hours are computed in CALENDAR_TIMEZONE (developer's local tz).
    Slot labels are formatted in display_tz_key (defaults to CALENDAR_DISPLAY_TIMEZONE).
    """
    try:
        dev_tz = ZoneInfo(CALENDAR_TIMEZONE)
    except Exception as exc:
        raise RuntimeError(
            f"Invalid CALENDAR_TIMEZONE '{CALENDAR_TIMEZONE}'. "
            "Use a full IANA key, e.g. America/Argentina/Buenos_Aires, Europe/Madrid, UTC."
        ) from exc

    display_tz = ZoneInfo(display_tz_key or CALENDAR_DISPLAY_TIMEZONE)
    duration = timedelta(minutes=CALENDAR_SLOT_DURATION_MINUTES)
    now = datetime.now(timezone.utc)

    # Start from tomorrow so the recruiter isn't pressured into same-day booking
    start = (now + timedelta(days=1)).astimezone(dev_tz).replace(
        hour=CALENDAR_WORKING_HOURS_START, minute=0, second=0, microsecond=0
    )
    end = start + timedelta(days=CALENDAR_DAYS_AHEAD)

    freebusy = (
        _service()
        .freebusy()
        .query(
            body={
                "timeMin": start.astimezone(timezone.utc).isoformat(),
                "timeMax": end.astimezone(timezone.utc).isoformat(),
                "items": [{"id": GOOGLE_CALENDAR_ID}],
            }
        )
        .execute()
    )

    busy = [
        (
            datetime.fromisoformat(b["start"].replace("Z", "+00:00")),
            datetime.fromisoformat(b["end"].replace("Z", "+00:00")),
        )
        for b in freebusy["calendars"][GOOGLE_CALENDAR_ID]["busy"]
    ]

    slots: list[dict] = []
    cursor = start

    while cursor.astimezone(timezone.utc) < end.astimezone(timezone.utc) and len(slots) < 6:
        if cursor.weekday() >= 5:
            cursor = (cursor + timedelta(days=1)).replace(
                hour=CALENDAR_WORKING_HOURS_START, minute=0, second=0, microsecond=0
            )
            continue

        if cursor.hour >= CALENDAR_WORKING_HOURS_END:
            cursor = (cursor + timedelta(days=1)).replace(
                hour=CALENDAR_WORKING_HOURS_START, minute=0, second=0, microsecond=0
            )
            continue

        slot_end = cursor + duration
        if slot_end.hour > CALENDAR_WORKING_HOURS_END or (
            slot_end.hour == CALENDAR_WORKING_HOURS_END and slot_end.minute > 0
        ):
            cursor = (cursor + timedelta(days=1)).replace(
                hour=CALENDAR_WORKING_HOURS_START, minute=0, second=0, microsecond=0
            )
            continue

        cursor_utc = cursor.astimezone(timezone.utc)
        slot_end_utc = slot_end.astimezone(timezone.utc)

        if not any(b_start < slot_end_utc and b_end > cursor_utc for b_start, b_end in busy):
            slots.append(
                {
                    "start": cursor_utc.isoformat(),
                    "end": slot_end_utc.isoformat(),
                    "label": format_slot_label(cursor_utc.isoformat(), display_tz),
                }
            )

        cursor += duration

    logger.info("[calendar] found %s available slots (display tz: %s)", len(slots), display_tz)
    return slots


def create_meeting(slot: dict, recruiter_name: str, recruiter_email: str) -> str:
    """Create a Google Calendar event and return its HTML link."""
    event = {
        "summary": f"Meeting with {recruiter_name}",
        "description": (
            f"Scheduled via Emmanuel's AI assistant.\n"
            f"Recruiter: {recruiter_name} ({recruiter_email})"
        ),
        "start": {"dateTime": slot["start"], "timeZone": "UTC"},
        "end": {"dateTime": slot["end"], "timeZone": "UTC"},
        "attendees": [{"email": recruiter_email}],
        "reminders": {"useDefault": True},
    }
    result = (
        _service()
        .events()
        .insert(calendarId=GOOGLE_CALENDAR_ID, body=event, sendUpdates="all")
        .execute()
    )
    link = result.get("htmlLink", "")
    logger.info("[calendar] created event: %s", link)
    return link
