#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HASKELL_DIR="${ROOT_DIR}/haskell"
WEB_DIR="${HASKELL_DIR}/web"

TARGET="${1:-full}"

usage() {
  cat <<'EOF'
Usage: bash scripts/verify.sh [target]

Targets:
  haskell     Run Haskell format, lint, build, smoke, and test checks.
  web         Run web typecheck, tests, and build.
  automation  Run the root autoloop regression tests.
  smoke       Run only the Haskell smoke checks.
  full        Run haskell + web + automation checks.
EOF
}

require_cmd() {
  local cmd="$1"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    printf 'Missing required command: %s\n' "${cmd}" >&2
    exit 1
  fi
}

run_step() {
  local display="$1"
  shift
  printf '\n==> %s\n' "${display}"
  "$@"
}

run_haskell_shell() {
  local command="$1"
  run_step "cd haskell && ${command}" bash -c "cd \"$HASKELL_DIR\" && ${command}"
}

run_web_shell() {
  local command="$1"
  run_step "cd haskell/web && ${command}" bash -c "cd \"$WEB_DIR\" && ${command}"
}

verify_haskell() {
  require_cmd bash
  require_cmd cabal
  require_cmd fourmolu
  require_cmd hlint

  # Build first: a compile break must surface on its own, not be hidden behind a
  # formatting/lint failure that aborts the script early (set -e). This mirrors
  # the autoloop merge build gate.
  run_haskell_shell "cabal build"
  run_haskell_shell "find app test bench -name '*.hs' -print0 | xargs -0 fourmolu --mode check"
  run_haskell_shell "bash scripts/hlint_check.sh"
  run_haskell_shell "bash scripts/ci_smoke.sh"
  run_haskell_shell "cabal test --test-show-details=direct"
}

verify_web() {
  require_cmd bash
  require_cmd npm

  run_web_shell "npm --workspaces=false run typecheck"
  run_web_shell "npm --workspaces=false run test"
  run_web_shell "npm --workspaces=false run build"
}

verify_automation() {
  require_cmd node

  run_step "node --test test/autoloop.test.mjs" bash -c "cd \"$ROOT_DIR\" && node --test test/autoloop.test.mjs"
}

verify_smoke() {
  require_cmd bash
  require_cmd cabal

  run_haskell_shell "bash scripts/ci_smoke.sh"
}

case "${TARGET}" in
  haskell)
    verify_haskell
    ;;
  web)
    verify_web
    ;;
  automation)
    verify_automation
    ;;
  smoke)
    verify_smoke
    ;;
  full)
    verify_haskell
    verify_web
    verify_automation
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    printf 'Unknown verify target: %s\n\n' "${TARGET}" >&2
    usage >&2
    exit 1
    ;;
esac
