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

DB_URL_DEFAULT="postgres://${USER}@127.0.0.1:5432/trader?sslmode=disable"
export TRADER_DB_URL="${TRADER_DB_URL:-$DB_URL_DEFAULT}"

# Basic local Postgres readiness checks to avoid silent persistence disable.
check_pg_ready() {
  if command -v pg_isready >/dev/null 2>&1; then
    if pg_isready -q -h 127.0.0.1 -p 5432; then
      return 0
    fi
  fi

  if command -v lsof >/dev/null 2>&1; then
    if lsof -nP -iTCP:5432 -sTCP:LISTEN >/dev/null 2>&1; then
      return 0
    fi
  fi

  if command -v nc >/dev/null 2>&1; then
    if nc -z 127.0.0.1 5432 >/dev/null 2>&1; then
      return 0
    fi
  fi

  return 1
}

cleanup_stale_postmaster() {
  local brew_prefix=""
  local formula_prefix=""
  local data_root=""
  local pid_file=""
  local stale_pid=""
  local process_comm=""
  local reason=""
  local -a pid_candidates=(
    "/opt/homebrew/var/postgresql@16/postmaster.pid"
    "/usr/local/var/postgresql@16/postmaster.pid"
  )

  if command -v brew >/dev/null 2>&1; then
    brew_prefix="$(brew --prefix 2>/dev/null || true)"
    if [ -n "${brew_prefix}" ]; then
      pid_candidates=("${brew_prefix}/var/postgresql@16/postmaster.pid" "${pid_candidates[@]}")
    fi

    formula_prefix="$(brew --prefix postgresql@16 2>/dev/null || true)"
    if [ -n "${formula_prefix}" ]; then
      data_root="$(dirname "$(dirname "${formula_prefix}")")"
      pid_candidates=("${data_root}/var/postgresql@16/postmaster.pid" "${pid_candidates[@]}")
    fi
  fi

  for pid_file in "${pid_candidates[@]}"; do
    [ -f "${pid_file}" ] || continue
    stale_pid="$(head -n 1 "${pid_file}" 2>/dev/null || true)"
    process_comm=""
    reason=""

    if [ -z "${stale_pid}" ]; then
      reason="missing PID"
    else
      process_comm="$(ps -p "${stale_pid}" -o comm= 2>/dev/null || true)"
      if [ -z "${process_comm}" ]; then
        reason="PID ${stale_pid} not running"
      elif [[ "${process_comm}" != *postgres* ]]; then
        reason="PID ${stale_pid} belongs to ${process_comm}"
      fi
    fi

    if [ -n "${reason}" ]; then
      echo "Removing stale postmaster.pid at ${pid_file} (${reason})..."
      rm -f "${pid_file}" >/dev/null 2>&1 || true
    fi
  done
}

if ! check_pg_ready; then
  cleanup_stale_postmaster
  if command -v brew >/dev/null 2>&1; then
    echo "Postgres not ready; starting via Homebrew..."
    brew services start postgresql >/dev/null 2>&1 || true
    if ! check_pg_ready; then
      # Try versioned formula if the default service isn't installed.
      pg_formula="$(brew list --formula 2>/dev/null | rg -m1 '^postgresql@' || true)"
      if [ -n "${pg_formula}" ]; then
        echo "Postgres still not ready; starting via Homebrew formula ${pg_formula}..."
        brew services start "${pg_formula}" >/dev/null 2>&1 || true
      fi
    fi
  else
    cat <<'EOF'
ERROR: Cannot verify Postgres availability (pg_isready/lsof/nc not found).
Install Postgres (Homebrew: brew install postgresql) and start it.
EOF
    exit 1
  fi
fi

if ! check_pg_ready; then
  cat <<'EOF'
ERROR: Postgres is not running on 127.0.0.1:5432.
Homebrew quickstart:
  brew services start postgresql
  createdb trader
If Homebrew reports a launchctl error, try:
  brew services stop postgresql@16
  launchctl unload -w ~/Library/LaunchAgents/homebrew.mxcl.postgresql@16.plist
  rm -f ~/Library/LaunchAgents/homebrew.mxcl.postgresql@16.plist
  brew services start postgresql@16
Or set TRADER_DB_URL to your Postgres instance.
EOF
  exit 1
fi

# Create the default database if possible (no-op if it already exists).
if command -v createdb >/dev/null 2>&1; then
  if ! createdb trader >/dev/null 2>&1; then
    if command -v psql >/dev/null 2>&1; then
      if ! psql -h 127.0.0.1 -lqt 2>/dev/null | awk '{print $1}' | rg -q '^trader$'; then
        echo "WARN: failed to create database 'trader'. Continuing without verification."
      fi
    fi
  fi
fi

if ! check_pg_ready; then
  cat <<'EOF'
ERROR: Postgres is still not reachable after attempted startup.
Set TRADER_DB_URL to a reachable Postgres instance.
EOF
  exit 1
fi

run_api_once() {
  cabal run -v0 trader-hs -- --serve --port "${TRADER_API_PORT:-8080}" --platform binance --futures
}

restart_on_exit="${TRADER_API_RESTART_ON_EXIT:-0}"
restart_delay_sec="${TRADER_API_RESTART_DELAY_SEC:-3}"

if [ "${restart_on_exit}" = "1" ] || [ "${restart_on_exit}" = "true" ]; then
  echo "API restart loop enabled (delay ${restart_delay_sec}s)."
  while true; do
    set +e
    run_api_once
    exit_code=$?
    set -e
    echo "API exited (code ${exit_code}). Restarting in ${restart_delay_sec}s..."
    sleep "${restart_delay_sec}"
  done
else
  run_api_once
fi
