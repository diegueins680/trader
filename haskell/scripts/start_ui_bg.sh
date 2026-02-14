#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WEB_DIR="$ROOT_DIR/web"

ENV_FILE="$ROOT_DIR/../.env"
if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
fi

TRADER_API_TARGET="${TRADER_API_TARGET:-http://127.0.0.1:8080}"
TRADER_API_LOG_FILE="${TRADER_API_LOG_FILE:-/tmp/trader-api.log}"
TRADER_UI_PORT="${TRADER_UI_PORT:-5174}"
TRADER_UI_HOST="${TRADER_UI_HOST:-127.0.0.1}"
TRADER_UI_LOG_FILE="${TRADER_UI_LOG_FILE:-/tmp/trader-ui.log}"
TRADER_UI_WAIT_TIMEOUT_SEC="${TRADER_UI_WAIT_TIMEOUT_SEC:-30}"

api_port_from_target() {
  local target="$1"
  local port="${target##*:}"
  case "$port" in
    '' | *[!0-9]*)
      return 0
      ;;
    *)
      echo "$port"
      ;;
  esac
}

report_api_debug() {
  local port
  port="$(api_port_from_target "${TRADER_API_TARGET}")"
  if [ -n "${port}" ] && command -v lsof >/dev/null 2>&1; then
    if lsof -nP -iTCP:"${port}" -sTCP:LISTEN >/dev/null 2>&1; then
      echo "NOTE: Port ${port} is listening, but /health did not respond."
    else
      echo "NOTE: No process is listening on port ${port}."
    fi
  fi

  if [ -f "${TRADER_API_LOG_FILE}" ]; then
    echo "Last 20 lines of ${TRADER_API_LOG_FILE}:"
    tail -n 20 "${TRADER_API_LOG_FILE}" || true
  fi
}

wait_for_api() {
  local start
  start="$(date +%s)"
  while true; do
    if command -v curl >/dev/null 2>&1; then
      if curl -sf "${TRADER_API_TARGET}/health" >/dev/null 2>&1; then
        return 0
      fi
    fi
    local now
    now="$(date +%s)"
    if [ $((now - start)) -ge "${TRADER_UI_WAIT_TIMEOUT_SEC}" ]; then
      return 1
    fi
    sleep 1
  done
}

if ! wait_for_api; then
  echo "WARN: API not reachable at ${TRADER_API_TARGET}/health after ${TRADER_UI_WAIT_TIMEOUT_SEC}s. Starting UI anyway."
  report_api_debug
fi

cd "$WEB_DIR"
nohup env TRADER_API_TARGET="${TRADER_API_TARGET}" TRADER_API_TOKEN="${TRADER_API_TOKEN:-}" \
  npm run dev -- --host "${TRADER_UI_HOST}" --port "${TRADER_UI_PORT}" --strictPort \
  > "${TRADER_UI_LOG_FILE}" 2>&1 &

echo "Started trader-web on http://${TRADER_UI_HOST}:${TRADER_UI_PORT} (log ${TRADER_UI_LOG_FILE})."
