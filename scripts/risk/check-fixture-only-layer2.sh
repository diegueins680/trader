#!/usr/bin/env bash
# check-fixture-only-layer2.sh
#
# Risk-control gate (RISK-CADENCE-001, NO-GO appended 2026-06-10):
#   Flag any change that touches `haskell/test/TestMain.hs` Layer (2) without
#   a matching production change in `haskell/app/Trader/*`. Fixture-only
#   mutations to Layer (2) are REJECT — they mask missing production
#   invariants (cf. cycles 928→931 bounded-overflow / capped-attribution
#   fixture rewrite churn).
#
# Inputs:
#   $1 : git revision range (default: HEAD~1..HEAD)
#
# Exit codes:
#   0  : safe — either no Layer-(2) touch, or Layer-(2) touch accompanied
#        by a production change under haskell/app/Trader/
#   1  : VIOLATION — fixture-only Layer-(2) mutation detected
#   2  : usage / environment error
#
# Narrow by design. No GitHub Actions. No CI polling. Local risk gate only.
set -euo pipefail

RANGE="${1:-HEAD~1..HEAD}"
TEST_FILE="haskell/test/TestMain.hs"
PROD_PREFIX="haskell/app/Trader/"
LAYER2_MARKER='-- Layer (2)'

if ! command -v git >/dev/null 2>&1; then
  echo "check-fixture-only-layer2: git not found" >&2
  exit 2
fi

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "check-fixture-only-layer2: not inside a git work tree" >&2
  exit 2
fi

# 1. Did the range touch the test file at all?
if ! git diff --name-only "$RANGE" -- "$TEST_FILE" 2>/dev/null | grep -q .; then
  echo "OK: $TEST_FILE untouched in $RANGE"
  exit 0
fi

# 2. Does the diff actually mutate Layer (2)? Use --unified=0 + range-context
#    grep so we only fire when added/removed lines are near a Layer (2) marker.
DIFF_BODY=$(git diff --unified=10 "$RANGE" -- "$TEST_FILE" 2>/dev/null || true)
if ! printf '%s\n' "$DIFF_BODY" | grep -F -q -e "$LAYER2_MARKER"; then
  echo "OK: $TEST_FILE touched in $RANGE but no Layer (2) hunk"
  exit 0
fi

# 3. Layer (2) hunk present. Require a matching production change.
PROD_TOUCHED=$(git diff --name-only "$RANGE" -- "$PROD_PREFIX" 2>/dev/null | grep -c . || true)
if [ "${PROD_TOUCHED:-0}" -gt 0 ]; then
  echo "OK: Layer (2) hunk in $TEST_FILE accompanied by $PROD_TOUCHED production change(s) under $PROD_PREFIX"
  exit 0
fi

cat >&2 <<EOF
REJECT: fixture-only Layer (2) mutation detected.
  range          : $RANGE
  test file      : $TEST_FILE
  layer marker   : $LAYER2_MARKER
  required path  : $PROD_PREFIX (no files touched)

Standing risk-control NO-GO (RISK-CADENCE-001): PRs that mutate
haskell/test/TestMain.hs Layer (2) without a matching haskell/app/Trader/*
production change are REJECT. Repair the production invariant
(simulateEnsembleWithHLChecked / bcaRealizedSlippageCost / capped bucket
allocation) instead of rewriting the fixture text.
EOF
exit 1
