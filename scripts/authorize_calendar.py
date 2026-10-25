"""
Run this script once locally to authorize Google Calendar access and generate secrets/token.json.

Usage:
    uv run python scripts/authorize_calendar.py

A browser window will open asking you to sign in with your Google account and grant
Calendar access. After you approve, the token is written to secrets/token.json.
The app will use secrets/token.json automatically from that point on — no re-auth needed
unless you revoke access in your Google account settings.
"""
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from google_auth_oauthlib.flow import InstalledAppFlow

from app.config import GOOGLE_CREDENTIALS_PATH, GOOGLE_TOKEN_PATH, resolve_path
from app.secure_io import atomic_write_secret

_SCOPES = ["https://www.googleapis.com/auth/calendar"]
_CREDS_PATH = resolve_path(GOOGLE_CREDENTIALS_PATH)
_TOKEN_PATH = resolve_path(GOOGLE_TOKEN_PATH)

if not _CREDS_PATH.exists():
    raise SystemExit(
        f"Google credentials not found at {_CREDS_PATH.resolve()}.\n"
        "Download it from Google Cloud Console:\n"
        "  APIs & Services → Credentials → your OAuth 2.0 client → download icon"
    )

flow = InstalledAppFlow.from_client_secrets_file(str(_CREDS_PATH), _SCOPES)
creds = flow.run_local_server(port=0)
atomic_write_secret(_TOKEN_PATH, creds.to_json())
print(f"\nAuthorization complete. Token saved to {_TOKEN_PATH.resolve()}")
print("You can now start the application — scheduling is ready.")
