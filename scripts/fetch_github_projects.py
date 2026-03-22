import base64
import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

# Repo root (parent of scripts/)
ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = ROOT / "resources" / "json" / "github_projects.json"

# Load .env from repo root so GITHUB_TOKEN is available
load_dotenv(ROOT / ".env")

GITHUB_USERNAME = "molero3111"
REPOS_URL = f"https://api.github.com/users/{GITHUB_USERNAME}/repos"


def _headers() -> dict[str, str]:
    h = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def fetch_readme(repo_name: str) -> str | None:
    url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo_name}/readme"
    response = requests.get(url, headers=_headers())

    if response.status_code == 404:
        return None  # Repo has no README

    if response.status_code != 200:
        print(f"  Warning: could not fetch README for {repo_name} (status {response.status_code})")
        return None

    data = response.json()
    encoded_content = data.get("content", "")
    decoded = base64.b64decode(encoded_content).decode("utf-8")
    return decoded


def fetch_repos() -> list[dict]:
    projects = []
    page = 1

    while True:
        response = requests.get(
            REPOS_URL,
            headers=_headers(),
            params={
                "per_page": 100,
                "page": page,
                "type": "owner",  # Exclude forks
                "sort": "updated"
            }
        )

        if response.status_code != 200:
            print(f"Error fetching repos: {response.status_code}")
            break

        repos = response.json()
        if not repos:
            break  # No more pages

        for repo in repos:
            name = repo["name"]
            print(f"Processing: {name}...")

            readme = fetch_readme(name)
            time.sleep(0.3)  # Be polite to the API

            project = {
                "name": name,
                "url": repo["html_url"],
                "description": repo["description"],
                "language": repo["language"],
                "topics": repo["topics"],
                "readme": readme
            }

            projects.append(project)

        page += 1

    return projects


if __name__ == "__main__":
    if not os.environ.get("GITHUB_TOKEN", "").strip():
        print(
            "Warning: GITHUB_TOKEN is not set. Unauthenticated GitHub API is limited to ~60 requests/hour.\n"
            "Create a token (classic) with scope 'public_repo' (or fine-grained: read on your repos) and add to .env:\n"
            "  GITHUB_TOKEN=ghp_...\n"
            "See: https://github.com/settings/tokens\n",
            file=sys.stderr,
        )
    print(f"Fetching repos for {GITHUB_USERNAME}...\n")
    projects = fetch_repos()

    output = {"github_projects": projects}

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nDone! {len(projects)} repos saved to {OUTPUT_PATH} (overwritten).")
