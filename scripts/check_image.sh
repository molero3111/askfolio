#!/usr/bin/env bash
# Assert the built image contains no secrets/private data and runs as non-root.
# Usage: scripts/check_image.sh IMAGE
set -uo pipefail

IMAGE="${1:?usage: check_image.sh IMAGE}"
fail=0
err() { echo "FAIL: $*" >&2; fail=1; }
run() { docker run --rm --entrypoint "" "$IMAGE" "$@"; }

docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "FAIL: image not found: $IMAGE" >&2; exit 1; }

for name in .env token.json credentials.json venv secrets data; do
    run test -e "/app/$name"
    rc=$?
    if [ "$rc" -eq 0 ]; then
        err "/app/$name exists in image"
    elif [ "$rc" -ne 1 ]; then
        err "could not check /app/$name (docker/exec exit $rc)"
    fi
done

json_listing=$(run sh -c 'ls -A /app/resources/json' 2>&1) || err "cannot list /app/resources/json"
if [ "$json_listing" != ".gitkeep" ]; then
    err "/app/resources/json must contain only .gitkeep (found: $(echo "$json_listing" | tr '\n' ' '))"
fi

uid=$(run id -u 2>&1)
if [ "$uid" != "10001" ]; then
    err "image runs as uid '$uid', expected 10001"
fi

# find exits non-zero on unreadable /proc entries, so guard against docker
# failures via a sentinel line instead of find's own status.
found=$(run sh -c "find / \( -name token.json -o -name credentials.json -o -name '.env' \) 2>/dev/null; echo __done__")
rc=$?
if [ "$rc" -ne 0 ] || ! printf '%s\n' "$found" | grep -qx '__done__'; then
    err "could not run find in image (exit $rc)"
else
    found=$(printf '%s\n' "$found" | grep -vx '__done__')
    if [ -n "$found" ]; then
        err "secret files found in image: $(echo "$found" | tr '\n' ' ')"
    fi
fi

if [ "$fail" -ne 0 ]; then
    echo "Image check FAILED for $IMAGE" >&2
    exit 1
fi
echo "Image check passed for $IMAGE"
