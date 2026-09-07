#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
STATE_DIR="${TRADER_AUTOLOOP_STATE_DIR:-/Users/diegosaa/.openclaw/orgs/trader-firm/runtime/trader-autoloop-live/.tmp/autoloop}"
PID_FILE="${STATE_DIR}/runner.pid"
STOP_FILE="${STATE_DIR}/stop"
STATUS_FILE="${STATE_DIR}/status.json"
LAUNCHER_LOG="${STATE_DIR}/launcher.log"

load_repo_env() {
  if [[ -f "${ROOT_DIR}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${ROOT_DIR}/.env"
    set +a
  fi
}

runner_pid() {
  if [[ -f "${PID_FILE}" ]]; then
    tr -d '[:space:]' < "${PID_FILE}"
  fi
}

pid_exists() {
  local pid="$1"
  python3 - "${pid}" <<'PY'
import errno
import os
import sys

try:
    pid = int(sys.argv[1])
except (IndexError, TypeError, ValueError):
    raise SystemExit(1)

if pid <= 0:
    raise SystemExit(1)

try:
    os.kill(pid, 0)
except ProcessLookupError:
    raise SystemExit(1)
except PermissionError:
    # POSIX EPERM proves that the PID exists; only the signal permission is
    # denied. Treating it as absent can report a live runner as dead and can
    # let a second launcher race the existing process.
    raise SystemExit(0)
except OSError as exc:
    raise SystemExit(0 if exc.errno == errno.EPERM else 1)
raise SystemExit(0)
PY
}

runner_alive() {
  local pid
  pid="$(runner_pid || true)"
  [[ -n "${pid}" ]] && pid_exists "${pid}"
}

start_runner() {
  mkdir -p "${STATE_DIR}"
  if runner_alive; then
    printf 'Autoloop runner already active (PID %s).\n' "$(runner_pid)"
    exit 1
  fi
  load_repo_env
  (
    cd "${ROOT_DIR}"
    nohup node scripts/autoloop-forever.mjs "$@" >> "${LAUNCHER_LOG}" 2>&1 < /dev/null &
  )
  sleep 1
  if runner_alive; then
    printf 'Started autoloop runner (PID %s). Status: %s\n' "$(runner_pid)" "${STATUS_FILE}"
    exit 0
  fi
  printf 'Autoloop runner did not stay up. Check %s or %s.\n' "${LAUNCHER_LOG}" "${STATUS_FILE}" >&2
  exit 1
}

run_foreground() {
  load_repo_env
  cd "${ROOT_DIR}"
  exec node scripts/autoloop-forever.mjs "$@"
}

stop_runner() {
  mkdir -p "${STATE_DIR}"
  printf '%s stop requested by operator\n' "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" > "${STOP_FILE}"
  if runner_alive; then
    kill -TERM "$(runner_pid)" 2>/dev/null || true
    printf 'Stop requested for autoloop runner (PID %s).\n' "$(runner_pid)"
    exit 0
  fi
  printf 'Stop file written at %s.\n' "${STOP_FILE}"
}

show_status() {
  if [[ ! -f "${STATUS_FILE}" ]]; then
    printf 'No autoloop status file found at %s.\n' "${STATUS_FILE}" >&2
    exit 1
  fi

  local alive_flag=0
  if runner_alive; then
    alive_flag=1
  fi

  python3 - "${STATUS_FILE}" "${PID_FILE}" "${alive_flag}" <<'PY'
import json
import sys

status_path, pid_path, alive_raw = sys.argv[1:4]
with open(status_path, "r", encoding="utf-8") as handle:
    status = json.load(handle)

pid = status.get("pid")
try:
    with open(pid_path, "r", encoding="utf-8") as handle:
        raw = handle.read().strip()
    if raw:
        pid = int(raw)
except Exception:
    pass

alive = alive_raw == "1"

status["pid"] = pid
status["pidAlive"] = alive
status["live"] = alive and status.get("state") not in {"stopped", "error", "dead"}
if not alive and status.get("state") not in {"stopped", "error", "dead"}:
    status["state"] = "dead"
    status["nextRunAt"] = None
    status["statusReason"] = "runner pid recorded in status is not alive"

print(json.dumps(status, indent=2))
PY
}

command="${1:-run}"
if [[ "${command}" == "start" || "${command}" == "run" || "${command}" == "stop" || "${command}" == "status" ]]; then
  shift || true
else
  command="run"
fi

case "${command}" in
  start)
    start_runner "$@"
    ;;
  run)
    run_foreground "$@"
    ;;
  stop)
    stop_runner
    ;;
  status)
    show_status
    ;;
esac
