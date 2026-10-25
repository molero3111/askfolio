#!/usr/bin/env bash
# Decides what the production deploy ships (used by .github/workflows/deploy.yml).
#   - no SHA requested: mode=build, sha=GITHUB_SHA (CI, then build, then deploy);
#   - REQUESTED_SHA set (the workflow_dispatch input, a rollback or redeploy): mode=redeploy,
#     after checking that it is a full SHA on main's history, that CI's "test" job
#     (from GitHub Actions) passed for it, and that its image exists in GHCR. Nothing is built.
# Writes mode= and sha= to $GITHUB_OUTPUT. Needs a full-history checkout of main, `gh`
# (GH_TOKEN with checks: read) and `docker` logged in to GHCR for the redeploy checks.
# Env: REF, GITHUB_SHA, REQUESTED_SHA, IMAGE_NAME, REPO, GITHUB_OUTPUT.
set -euo pipefail

fail() {
  echo "::error::$*"
  exit 1
}

[ "${REF:-}" = "refs/heads/main" ] || fail "deploys run only from main (got ${REF:-unset})"

sha="${REQUESTED_SHA:-}"

if [ -z "$sha" ]; then
  mode=build
  sha="$GITHUB_SHA"
else
  mode=redeploy
  [[ "$sha" =~ ^[0-9a-f]{40}$ ]] || fail "sha must be a full 40-character lowercase commit SHA"
  git cat-file -e "${sha}^{commit}" 2>/dev/null || fail "commit $sha is not in this repository"
  git merge-base --is-ancestor "$sha" origin/main || fail "commit $sha is not on main's history"

  # ci.yml's "test" job succeeded on the commit. Its check run is named "test" in ci.yml's own
  # run (pull_request), or "<caller job> / test" when the deploy workflow called ci.yml; only GitHub
  # Actions' own runs count, so another app cannot satisfy the guard with a look-alike name.
  runs=$(gh api "repos/$REPO/commits/$sha/check-runs?per_page=100" \
    --jq '[.check_runs[] | select(.app.slug == "github-actions" and .conclusion == "success") | .name]')
  jq -e 'any(.[]; test("(^|/ )test$"))' <<<"$runs" >/dev/null ||
    fail "CI (test) has no successful GitHub Actions run for $sha"

  docker manifest inspect "$IMAGE_NAME:$sha" >/dev/null 2>&1 ||
    fail "image $IMAGE_NAME:$sha does not exist (or this job cannot read it)"
fi

echo "Deploy plan: mode=$mode sha=$sha"
{
  echo "mode=$mode"
  echo "sha=$sha"
} >>"$GITHUB_OUTPUT"
