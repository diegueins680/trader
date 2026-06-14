#!/usr/bin/env bash
# restart-local-stack.sh — Refresh the locally running trader API + Web UI so
# they match the current git HEAD.
#
# Used by the Continuous Improvement auto-loop after a green push to main:
# the LaunchAgents are the source of truth, so we just (a) refresh
# `haskell/.build-commit` so the /health endpoint reports the right SHA and
# (b) `launchctl kickstart -k` the API and Web services. Each step is
# best-effort and never fails the caller.
#
# Usage:
#   scripts/restart-local-stack.sh [--api-label LABEL] [--web-label LABEL] [--quiet]
#
# Env overrides:
#   TRADER_API_LAUNCHD_LABEL   default: ai.openclaw.trader.api
#   TRADER_WEB_LAUNCHD_LABEL   default: ai.openclaw.trader.web
#   TRADER_API_BASE_URL        default: http://127.0.0.1:8090   (used only for the post-restart probe)
#   TRADER_LOCAL_STACK_QUIET=1 suppress non-error output

set -uo pipefail

API_LABEL="${TRADER_API_LAUNCHD_LABEL:-ai.openclaw.trader.api}"
WEB_LABEL="${TRADER_WEB_LAUNCHD_LABEL:-ai.openclaw.trader.web}"
API_BASE="${TRADER_API_BASE_URL:-http://127.0.0.1:8090}"
QUIET="${TRADER_LOCAL_STACK_QUIET:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --api-label) API_LABEL="$2"; shift 2 ;;
    --web-label) WEB_LABEL="$2"; shift 2 ;;
    --api-base)  API_BASE="$2";  shift 2 ;;
    --quiet)     QUIET=1;        shift   ;;
    *) echo "restart-local-stack.sh: unknown arg: $1" >&2; exit 2 ;;
  esac
done

log() { [[ "$QUIET" == "1" ]] || echo "[restart-local-stack] $*"; }
warn() { echo "[restart-local-stack] WARN: $*" >&2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

# Only run on macOS where launchctl is the supervisor.
if [[ "$(uname -s)" != "Darwin" ]]; then
  log "non-Darwin host; nothing to do."
  exit 0
fi

if ! command -v launchctl >/dev/null 2>&1; then
  warn "launchctl not on PATH; skipping local stack refresh."
  exit 0
fi

UID_NUM="$(id -u)"
GUI_DOMAIN="gui/${UID_NUM}"

HEAD_SHA="$(git rev-parse HEAD 2>/dev/null || true)"
if [[ -z "${HEAD_SHA}" ]]; then
  warn "git rev-parse HEAD failed; aborting refresh."
  exit 0
fi
log "git HEAD = ${HEAD_SHA}"

# Refresh the build-commit marker the API reads on startup so /health reports
# the SHA the binary was built from. The cabal build already produced the
# binary; we just keep the marker in sync with HEAD.
mkdir -p haskell
printf '%s\n' "${HEAD_SHA}" > haskell/.build-commit
log "wrote haskell/.build-commit"

kick() {
  local label="$1"
  if launchctl print "${GUI_DOMAIN}/${label}" >/dev/null 2>&1; then
    if launchctl kickstart -k "${GUI_DOMAIN}/${label}" >/dev/null 2>&1; then
      log "kicked ${label}"
    else
      warn "launchctl kickstart -k ${label} failed (rc=$?)"
    fi
  else
    log "service ${label} not loaded; skipping."
  fi
}

kick "${API_LABEL}"
kick "${WEB_LABEL}"

# Best-effort health probe so the loop can see the new commit.
if command -v curl >/dev/null 2>&1; then
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    sleep 1
    body="$(curl -fsS --max-time 2 "${API_BASE%/}/health" 2>/dev/null || true)"
    if [[ -n "${body}" ]]; then
      log "API /health responded: $(printf '%s' "${body}" | head -c 200)"
      break
    fi
  done
fi

exit 0
