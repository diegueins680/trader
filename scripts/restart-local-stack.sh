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
#   TRADER_TRADE_LOG           default: .tmp/trader/live_trades.ndjson
#   TRADER_LOCAL_STACK_QUIET=1 suppress non-error output
#   TRADER_BOT_START_*         live bot startup preset written into the API LaunchAgent

set -uo pipefail

API_LABEL="${TRADER_API_LAUNCHD_LABEL:-ai.openclaw.trader.api}"
WEB_LABEL="${TRADER_WEB_LAUNCHD_LABEL:-ai.openclaw.trader.web}"
API_BASE="${TRADER_API_BASE_URL:-http://127.0.0.1:8090}"
API_PLIST="${TRADER_API_LAUNCHD_PLIST:-${HOME}/Library/LaunchAgents/${API_LABEL}.plist}"
TRADE_LOG="${TRADER_TRADE_LOG:-.tmp/trader/live_trades.ndjson}"
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

API_PLIST_CHANGED=0
ensure_api_trade_log_arg() {
  local plist="$1"
  if [[ ! -f "${plist}" ]]; then
    return 0
  fi
  if [[ ! -x /usr/libexec/PlistBuddy ]]; then
    warn "PlistBuddy not found; cannot ensure API LaunchAgent --trade-log."
    return 0
  fi
  if /usr/libexec/PlistBuddy -c "Print :ProgramArguments" "${plist}" 2>/dev/null | grep -q -- "--trade-log"; then
    return 0
  fi

  local count
  count="$(
    /usr/libexec/PlistBuddy -c "Print :ProgramArguments" "${plist}" 2>/dev/null |
      awk '
        /^Array[[:space:]]*\{/ { in_array = 1; next }
        /^\}/ { in_array = 0; next }
        in_array && $0 ~ /[^[:space:]]/ { n++ }
        END { print n + 0 }
      '
  )"
  if [[ "${count}" -le 0 ]]; then
    warn "API LaunchAgent has no ProgramArguments; cannot add --trade-log."
    return 0
  fi

  mkdir -p "$(dirname "${TRADE_LOG}")"
  if /usr/libexec/PlistBuddy \
    -c "Add :ProgramArguments:${count} string --trade-log" \
    -c "Add :ProgramArguments:$((count + 1)) string ${TRADE_LOG}" \
    "${plist}" >/dev/null 2>&1; then
    API_PLIST_CHANGED=1
    log "added --trade-log ${TRADE_LOG} to ${plist}"
  else
    warn "failed to add --trade-log to ${plist}"
  fi
}

ensure_api_trade_log_arg "${API_PLIST}"

ensure_api_env_var() {
  local plist="$1"
  local key="$2"
  local value="$3"

  if [[ ! -f "${plist}" ]]; then
    return 0
  fi
  if [[ ! -x /usr/libexec/PlistBuddy ]]; then
    warn "PlistBuddy not found; cannot ensure API LaunchAgent ${key}."
    return 0
  fi

  if ! /usr/libexec/PlistBuddy -c "Print :EnvironmentVariables" "${plist}" >/dev/null 2>&1; then
    if /usr/libexec/PlistBuddy -c "Add :EnvironmentVariables dict" "${plist}" >/dev/null 2>&1; then
      API_PLIST_CHANGED=1
    else
      warn "failed to add EnvironmentVariables to ${plist}"
      return 0
    fi
  fi

  local current=""
  current="$(/usr/libexec/PlistBuddy -c "Print :EnvironmentVariables:${key}" "${plist}" 2>/dev/null || true)"
  if [[ "${current}" == "${value}" ]]; then
    return 0
  fi

  if [[ -n "${current}" ]]; then
    if /usr/libexec/PlistBuddy -c "Set :EnvironmentVariables:${key} ${value}" "${plist}" >/dev/null 2>&1; then
      API_PLIST_CHANGED=1
      log "set ${key}=${value} in ${plist}"
    else
      warn "failed to set ${key} in ${plist}"
    fi
  else
    if /usr/libexec/PlistBuddy -c "Add :EnvironmentVariables:${key} string ${value}" "${plist}" >/dev/null 2>&1; then
      API_PLIST_CHANGED=1
      log "added ${key}=${value} to ${plist}"
    else
      warn "failed to add ${key} to ${plist}"
    fi
  fi
}

ensure_api_startup_preset_env() {
  local plist="$1"
  ensure_api_env_var "${plist}" "TRADER_BOT_START_METHOD" "${TRADER_BOT_START_METHOD:-ta_regime_switch}"
  ensure_api_env_var "${plist}" "TRADER_BOT_START_VOL_CONF_GATE" "${TRADER_BOT_START_VOL_CONF_GATE:-disabled}"
  ensure_api_env_var "${plist}" "TRADER_BOT_START_COST_AWARE_EDGE" "${TRADER_BOT_START_COST_AWARE_EDGE:-false}"
  ensure_api_env_var "${plist}" "TRADER_BOT_START_MIN_EDGE" "${TRADER_BOT_START_MIN_EDGE:-0}"
  ensure_api_env_var "${plist}" "TRADER_BOT_START_MIN_SIGNAL_TO_NOISE" "${TRADER_BOT_START_MIN_SIGNAL_TO_NOISE:-0}"
  ensure_api_env_var "${plist}" "TRADER_BOT_START_OPEN_THRESHOLD" "${TRADER_BOT_START_OPEN_THRESHOLD:-0.001}"
  ensure_api_env_var "${plist}" "TRADER_BOT_START_CLOSE_THRESHOLD" "${TRADER_BOT_START_CLOSE_THRESHOLD:-0.001}"
  ensure_api_env_var "${plist}" "TRADER_BOT_START_TOP_COMBO_ADOPTION" "${TRADER_BOT_START_TOP_COMBO_ADOPTION:-true}"
}

ensure_api_startup_preset_env "${API_PLIST}"

if [[ "${API_PLIST_CHANGED}" == "1" ]]; then
  if launchctl print "${GUI_DOMAIN}/${API_LABEL}" >/dev/null 2>&1; then
    launchctl bootout "${GUI_DOMAIN}/${API_LABEL}" >/dev/null 2>&1 || true
  fi
  if launchctl bootstrap "${GUI_DOMAIN}" "${API_PLIST}" >/dev/null 2>&1; then
    log "reloaded ${API_LABEL} from ${API_PLIST}"
  else
    warn "launchctl bootstrap ${API_LABEL} failed after plist update"
  fi
fi

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
