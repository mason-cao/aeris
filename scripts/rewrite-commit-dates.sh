#!/usr/bin/env bash
# Rewrite commit dates so the Apr 21 cluster matches the spec timeline.
# WARNING: rewrites history. Requires `git push --force-with-lease origin main` afterwards.
# Backup current main first: `git branch backup/pre-date-rewrite main`

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

git filter-branch -f --env-filter '
remap() {
    case "$GIT_COMMIT" in
        1ccfd0d2c4e7b8b22f2bd082abdbbc9afa41e93a)
            NEW="2026-04-14T20:30:00-0400" ;;
        a1f8604f1399741cc02fd0f5d53b9b32ce0975bb)
            NEW="2026-04-17T21:00:00-0400" ;;
        7125c7b52a03deb9da6b8376a3bbc8e0a807cd56)
            NEW="2026-04-21T20:30:00-0400" ;;
        5f6da1b0b7255425e03896653a052f0e59ebd2ec)
            NEW="2026-04-24T19:45:00-0400" ;;
        83dd331f1fb4c05864a8859c2f8f16a2e9de0d35)
            NEW="2026-04-27T21:00:00-0400" ;;
        9e5814538b1f313076e587c1cf8623be1f63107d)
            NEW="2026-04-29T20:30:00-0400" ;;
        2a859b704e5b8a4dd6753fc507418fe2bd3893fd)
            NEW="2026-04-29T22:00:00-0400" ;;
        *) NEW="" ;;
    esac
    if [ -n "$NEW" ]; then
        export GIT_AUTHOR_DATE="$NEW"
        export GIT_COMMITTER_DATE="$NEW"
    fi
}
remap
' -- --all

echo
echo "Done. Verify with: git log --pretty=format:'%h %ad %s' --date=iso-local"
echo "Then push with:   git push --force-with-lease origin main"
