#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

ENV_FILE="$ROOT_DIR/../.env"
if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
fi

LOG_FILE="${TRADER_API_LOG_FILE:-/tmp/trader-api.log}"
export TRADER_API_PORT="${TRADER_API_PORT:-8080}"
PORT="$TRADER_API_PORT"
export TRADER_API_BIND_HOST="${TRADER_API_BIND_HOST:-127.0.0.1}"
export TRADER_API_RESTART_ON_EXIT="${TRADER_API_RESTART_ON_EXIT:-0}"
export TRADER_API_RESTART_DELAY_SEC="${TRADER_API_RESTART_DELAY_SEC:-3}"
export TRADER_OPTIMIZER_MAX_TRIALS="${TRADER_OPTIMIZER_MAX_TRIALS:-30}"
export TRADER_OPTIMIZER_MAX_TIMEOUT_SEC="${TRADER_OPTIMIZER_MAX_TIMEOUT_SEC:-1200}"
export TRADER_STATE_DIR="${TRADER_STATE_DIR:-$ROOT_DIR/../tmp/state}"
export TRADER_OPTIMIZER_COMBOS_DIR="${TRADER_OPTIMIZER_COMBOS_DIR:-$ROOT_DIR/../tmp}"

# Ensure state/combos dirs exist so restarts don't lose top-combos.json.
mkdir -p "$TRADER_STATE_DIR" "$TRADER_OPTIMIZER_COMBOS_DIR"

nohup "$ROOT_DIR/scripts/run_api_with_db.sh" > "$LOG_FILE" 2>&1 &
PID=$!
sleep 1
CHILD_PID=""
if command -v lsof >/dev/null 2>&1; then
  CHILD_PID="$(lsof -nP -iTCP:${PORT} -sTCP:LISTEN 2>/dev/null | awk 'NR==2 {print $2}' || true)"
fi
if [ -z "${CHILD_PID}" ] && command -v pgrep >/dev/null 2>&1; then
  CHILD_PID="$(pgrep -f "/trader-hs .*--serve.*--port ${PORT}" | head -n 1 || true)"
fi
if [ -n "${CHILD_PID}" ]; then
  echo "Started trader-hs (PID $CHILD_PID), port $PORT, log $LOG_FILE"
  echo "$CHILD_PID"
else
  echo "Started trader-hs (PID $PID), port $PORT, log $LOG_FILE"
  echo "$PID"
fi
