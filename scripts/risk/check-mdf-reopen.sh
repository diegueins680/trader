#!/usr/bin/env bash
# check-mdf-reopen.sh
#
# Risk-control gate (RISK standing NO-GO: "PERMANENTLY CLOSED mdfLastOpenTimeMs").
# Reject any diff that REINTRODUCES the `mdfLastOpenTimeMs` symbol into the
# Haskell tree. The symbol was retired as part of the bull-reopen HELD CLOSED
# decision (CIO/CEO, multi-cron standing); resurrecting it is a silent reopen
# of that lane.
#
# Inputs:
#   $1 : git revision range (default: HEAD~1..HEAD)
#
# Exit codes:
#   0  : safe — `mdfLastOpenTimeMs` not reintroduced in the range
#   1  : VIOLATION — `mdfLastOpenTimeMs` added on a `+` line under haskell/
#        WITHOUT an explicit `Mdf-Reopen-Ack:` trailer in the latest commit
#        in the range (emergency hand-merge clause)
#   2  : usage / environment error
#
# Narrow by design. No CI polling. Local risk gate only.
set -euo pipefail

RANGE="${1:-HEAD~1..HEAD}"
SYMBOL='mdfLastOpenTimeMs'
SCOPE_RE='^\+\+\+ b/haskell/'

if ! command -v git >/dev/null 2>&1; then
  echo "check-mdf-reopen: git not found" >&2
  exit 2
fi

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "check-mdf-reopen: not inside a git work tree" >&2
  exit 2
fi

# Get the diff restricted to haskell/.
DIFF_BODY=$(git diff --unified=0 "$RANGE" -- 'haskell/**' 2>/dev/null || true)

if [ -z "$DIFF_BODY" ]; then
  echo "OK: no haskell/ changes in $RANGE"
  exit 0
fi

# Look for `+` lines (additions, not file headers) containing the symbol.
ADDED=$(printf '%s\n' "$DIFF_BODY" \
  | grep -E '^\+[^+]' \
  | grep -F "$SYMBOL" || true)

if [ -z "$ADDED" ]; then
  echo "OK: $SYMBOL not reintroduced in $RANGE"
  exit 0
fi

# Reintroduction detected. Check for explicit ack trailer in the most recent
# commit in the range.
LATEST=$(git rev-list -n1 "$RANGE" 2>/dev/null || true)
if [ -n "$LATEST" ] && git log -1 --format=%B "$LATEST" 2>/dev/null \
     | grep -qE '^Mdf-Reopen-Ack:[[:space:]]+\S'; then
  echo "OK: $SYMBOL reintroduced WITH Mdf-Reopen-Ack: trailer in $LATEST"
  exit 0
fi

echo "VIOLATION: $SYMBOL reintroduced under haskell/ without Mdf-Reopen-Ack: trailer" >&2
printf '%s\n' "$ADDED" | sed -e 's/^/  /' >&2
exit 1
