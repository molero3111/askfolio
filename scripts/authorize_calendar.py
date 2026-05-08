"""
Run this script once locally to authorize Google Calendar access and generate token.json.

Usage:
    python scripts/authorize_calendar.py

A browser window will open asking you to sign in with your Google account and grant
Calendar access. After you approve, token.json is written to the project root.
The app will use token.json automatically from that point on — no re-auth needed
unless you revoke access in your Google account settings.
"""
from pathlib import Path

from google_auth_oauthlib.flow import InstalledAppFlow

_SCOPES = ["https://www.googleapis.com/auth/calendar"]
_CREDS_PATH = Path("credentials.json")
_TOKEN_PATH = Path("token.json")

if not _CREDS_PATH.exists():
    raise SystemExit(
        f"credentials.json not found at {_CREDS_PATH.resolve()}.\n"
        "Download it from Google Cloud Console:\n"
        "  APIs & Services → Credentials → your OAuth 2.0 client → download icon"
    )

flow = InstalledAppFlow.from_client_secrets_file(str(_CREDS_PATH), _SCOPES)
creds = flow.run_local_server(port=0)
_TOKEN_PATH.write_text(creds.to_json())
print(f"\nAuthorization complete. Token saved to {_TOKEN_PATH.resolve()}")
print("You can now start the application — scheduling is ready.")
