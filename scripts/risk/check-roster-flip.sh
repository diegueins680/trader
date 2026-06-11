#!/usr/bin/env bash
# check-roster-flip.sh
#
# Risk-control gate (RISK standing NO-GO: "01/10/blend roster flip").
# Flag any change that mutates the canonical method roster mapping
# ("01" / "10" / "blend") in haskell/app/Trader/Method.hs (the source-of-truth
# for MethodLstmOnly / MethodKalmanOnly / MethodBlend tags) WITHOUT an
# accompanying explicit roster-policy artifact:
#
#   - a roster ADR/note under docs/risk/roster/ , or
#   - a CHANGELOG.md entry mentioning "roster", or
#   - an explicit "Roster-Flip-Ack:" trailer in the most recent commit
#     in the range (for emergency hand-merges).
#
# Inputs:
#   $1 : git revision range (default: HEAD~1..HEAD)
#
# Exit codes:
#   0  : safe — no roster line touched, OR roster touch accompanied
#        by an explicit policy artifact / ack trailer
#   1  : VIOLATION — silent roster flip detected
#   2  : usage / environment error
#
# Narrow by design. No GitHub Actions. No CI polling. Local risk gate only.
set -euo pipefail

RANGE="${1:-HEAD~1..HEAD}"
METHOD_FILE="haskell/app/Trader/Method.hs"
ROSTER_RE='"(01|10|blend)"'

if ! command -v git >/dev/null 2>&1; then
  echo "check-roster-flip: git not found" >&2
  exit 2
fi

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "check-roster-flip: not inside a git work tree" >&2
  exit 2
fi

# 1. Did the range touch Method.hs at all?
if ! git diff --name-only "$RANGE" -- "$METHOD_FILE" 2>/dev/null | grep -q .; then
  echo "OK: $METHOD_FILE untouched in $RANGE"
  exit 0
fi

# 2. Do the +/- lines in that diff actually hit a roster token?
DIFF_BODY=$(git diff --unified=0 "$RANGE" -- "$METHOD_FILE" 2>/dev/null || true)
ROSTER_HUNK=$(printf '%s\n' "$DIFF_BODY" \
  | grep -E '^[+-][^+-]' \
  | grep -E -c "$ROSTER_RE" || true)

if [ "${ROSTER_HUNK:-0}" -eq 0 ]; then
  echo "OK: $METHOD_FILE touched in $RANGE but no 01/10/blend roster line"
  exit 0
fi

# 3. Roster touched. Require an explicit policy artifact.
POLICY_TOUCHED=$(git diff --name-only "$RANGE" -- 'docs/risk/roster/' 'CHANGELOG.md' 2>/dev/null | grep -c . || true)
ACK_TRAILER=$(git log -1 --format=%B "$(git rev-parse "${RANGE##*..}" 2>/dev/null || echo HEAD)" 2>/dev/null \
  | grep -c -E '^Roster-Flip-Ack:' || true)

if [ "${POLICY_TOUCHED:-0}" -gt 0 ] || [ "${ACK_TRAILER:-0}" -gt 0 ]; then
  echo "OK: roster hunk in $METHOD_FILE accompanied by policy artifact (paths=$POLICY_TOUCHED ack=$ACK_TRAILER)"
  exit 0
fi

cat >&2 <<EOF
REJECT: silent 01/10/blend roster flip detected.
  range          : $RANGE
  method file    : $METHOD_FILE
  roster tokens  : "01" | "10" | "blend"
  required       : at least one of
                    - file under docs/risk/roster/
                    - CHANGELOG.md entry
                    - commit trailer "Roster-Flip-Ack: <reason>"

Standing risk-control NO-GO: roster-flip changes to MethodLstmOnly /
MethodKalmanOnly / MethodBlend must ship with an explicit policy artifact.
Either revert the roster mutation or attach an ack/ADR before merge.
EOF
exit 1
