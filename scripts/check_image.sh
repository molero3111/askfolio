#!/usr/bin/env bash
# Assert the built image contains no secrets/private data and runs as non-root.
# Usage: scripts/check_image.sh IMAGE
set -uo pipefail

IMAGE="${1:?usage: check_image.sh IMAGE}"
fail=0
err() { echo "FAIL: $*" >&2; fail=1; }
run() { docker run --rm --entrypoint "" "$IMAGE" "$@"; }

docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "FAIL: image not found: $IMAGE" >&2; exit 1; }

for name in .env token.json credentials.json venv .venv secrets data; do
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

# Backstop: search the whole image for secret-like files. Runs as root so unreadable
# directories cannot hide anything; the uid check above still uses the default user.
# find's own status is echoed as __rc=N: 0 (clean) and 1 (some paths unreadable) are
# accepted, anything else, or a missing sentinel, fails. Only *.pem/*.key are excluded, and
# only under paths that legitimately hold public CA material: Python package dirs
# (/opt/venv, /usr/local/lib/python*, /usr/lib/python3*: certifi/pip CA bundles, test
# fixtures), the system CA store (/etc/ssl/certs, /usr/lib/ssl, /usr/share/ca-certificates).
# Every other name is searched everywhere.
out=$(docker run --rm --user 0 --entrypoint "" "$IMAGE" sh -c "find / \\( -path /proc -o -path /sys -o -path /dev \\) -prune -o \\( \\
    -name token.json -o -name credentials.json \\
    -o \\( -name '.env*' ! -name .env.example \\) \\
    -o \\( \\( -name '*.pem' -o -name '*.key' \\) \\
        ! -path '/opt/venv/*' ! -path '/usr/local/lib/python*' ! -path '/usr/lib/python3*' \\
        ! -path '/etc/ssl/certs/*' ! -path '/usr/lib/ssl/*' ! -path '/usr/share/ca-certificates/*' \\) \\
    -o -name 'id_rsa*' -o -name cv.json -o -name github_projects.json \\
    -o -path '*/secrets/*' \\) -print 2>/dev/null; echo __rc=\$?")
docker_rc=$?
find_rc=$(printf '%s\n' "$out" | sed -n 's/^__rc=\([0-9][0-9]*\)$/\1/p' | tail -n 1)
if [ "$docker_rc" -ne 0 ] || [ -z "$find_rc" ]; then
    err "could not run find in image (docker exit $docker_rc)"
elif [ "$find_rc" -ne 0 ] && [ "$find_rc" -ne 1 ]; then
    err "find failed in image (exit $find_rc)"
else
    found=$(printf '%s\n' "$out" | grep -v '^__rc=[0-9]*$')
    if [ -n "$found" ]; then
        err "secret files found in image: $(echo "$found" | tr '\n' ' ')"
    fi
fi

if [ "$fail" -ne 0 ]; then
    echo "Image check FAILED for $IMAGE" >&2
    exit 1
fi
echo "Image check passed for $IMAGE"
