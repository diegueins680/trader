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

: "${TRADER_OUTBOX_LOG_FILE:=/tmp/trader-outbox.log}"
: "${TRADER_OUTBOX_POLL_MS:=1000}"
: "${TRADER_OUTBOX_BATCH_SIZE:=100}"
: "${TRADER_OUTBOX_PUBLISHER_MODE:=noop}"
: "${TRADER_OUTBOX_PUBLISHING_TIMEOUT_MS:=60000}"

nohup env \
  TRADER_DB_URL="${TRADER_DB_URL:-}" \
  DATABASE_URL="${DATABASE_URL:-}" \
  TRADER_OUTBOX_PUBLISHER_MODE="$TRADER_OUTBOX_PUBLISHER_MODE" \
  TRADER_OUTBOX_POLL_MS="$TRADER_OUTBOX_POLL_MS" \
  TRADER_OUTBOX_BATCH_SIZE="$TRADER_OUTBOX_BATCH_SIZE" \
  TRADER_OUTBOX_PUBLISHING_TIMEOUT_MS="$TRADER_OUTBOX_PUBLISHING_TIMEOUT_MS" \
  cabal run outbox-publisher \
  > "$TRADER_OUTBOX_LOG_FILE" 2>&1 &

PID=$!
echo "Started outbox-publisher (PID $PID, mode $TRADER_OUTBOX_PUBLISHER_MODE, log $TRADER_OUTBOX_LOG_FILE)"
echo "$PID"
