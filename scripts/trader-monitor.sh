#!/usr/bin/env bash
# trader-monitor.sh — Health-check trader API and restart via launchd if down
#
# Usage:
#   scripts/trader-monitor.sh once          # Single health check
#   scripts/trader-monitor.sh loop          # Run forever (sleep 120s between checks)
#   scripts/trader-monitor.sh install       # Install LaunchAgent plist
#   scripts/trader-monitor.sh uninstall     # Remove LaunchAgent plist
#   scripts/trader-monitor.sh status        # Show LaunchAgent status
#
# Environment:
#   TRADER_API_BASE_URL   default: http://127.0.0.1:8080
#   TRADER_API_LAUNCHD_LABEL  default: ai.openclaw.trader.api
#   TRADER_MONITOR_LOG_DIR    default: /tmp/trader-monitor

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

API_BASE="${TRADER_API_BASE_URL:-http://127.0.0.1:8080}"
API_BASE="${API_BASE%/}"
API_LABEL="${TRADER_API_LAUNCHD_LABEL:-ai.openclaw.trader.api}"
LOG_DIR="${TRADER_MONITOR_LOG_DIR:-/tmp/trader-monitor}"
DIAG_LOG="${LOG_DIR}/diagnostics.log"
RESTART_LOG="${LOG_DIR}/restarts.log"
CHECK_INTERVAL_SEC="${TRADER_MONITOR_INTERVAL:-120}"
GUI_DOMAIN="gui/$(id -u)"
API_SERVICE_TARGET="${GUI_DOMAIN}/${API_LABEL}"

PLIST_DIR="${HOME}/Library/LaunchAgents"
MONITOR_LABEL="ai.openclaw.trader.monitor"
MONITOR_PLIST="${PLIST_DIR}/${MONITOR_LABEL}.plist"
MONITOR_SERVICE_TARGET="${GUI_DOMAIN}/${MONITOR_LABEL}"

mkdir -p "${LOG_DIR}"

log_diag() {
  printf '%s [diag] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')" "$*" >>"${DIAG_LOG}"
}

log_restart() {
  printf '%s [restart] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')" "$*" >>"${RESTART_LOG}"
}

health_check() {
  local url="${API_BASE}/health"
  local resp=""
  local http_code=""

  if ! command -v curl >/dev/null 2>&1; then
    log_diag "curl not found; skipping health check"
    return 1
  fi

  http_code="$(curl -fsS -o /dev/null -w '%{http_code}' --max-time 10 "${url}" 2>/dev/null || true)"

  if [[ "${http_code}" == "200" ]]; then
    log_diag "OK ${url} (${http_code})"
    return 0
  fi

  if [[ -z "${http_code}" ]]; then
    log_diag "FAIL ${url} (no response)"
  else
    log_diag "FAIL ${url} (HTTP ${http_code})"
  fi
  return 1
}

restart_api() {
  log_restart "Triggering launchctl kickstart -k ${API_SERVICE_TARGET}"
  if launchctl kickstart -k "${API_SERVICE_TARGET}" 2>>"${RESTART_LOG}"; then
    log_restart "kickstart succeeded"
    return 0
  else
    log_restart "kickstart failed"
    return 1
  fi
}

check_once() {
  if health_check; then
    return 0
  fi

  # Give it one more quick chance before restarting
  sleep 3
  if health_check; then
    return 0
  fi

  restart_api
}

check_loop() {
  log_diag "Starting monitor loop (interval=${CHECK_INTERVAL_SEC}s)"
  while true; do
    check_once || true
    sleep "${CHECK_INTERVAL_SEC}"
  done
}

write_monitor_plist() {
  mkdir -p "${PLIST_DIR}"
  cat > "${MONITOR_PLIST}" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
  <dict>
    <key>Label</key>
    <string>${MONITOR_LABEL}</string>

    <key>ProgramArguments</key>
    <array>
      <string>/bin/bash</string>
      <string>${ROOT_DIR}/scripts/trader-monitor.sh</string>
      <string>loop</string>
    </array>

    <key>WorkingDirectory</key>
    <string>${ROOT_DIR}</string>

    <key>EnvironmentVariables</key>
    <dict>
      <key>HOME</key>
      <string>${HOME}</string>
      <key>PATH</key>
      <string>${PATH:-/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin}</string>
    </dict>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <true/>

    <key>StandardOutPath</key>
    <string>${LOG_DIR}/launchd.stdout.log</string>

    <key>StandardErrorPath</key>
    <string>${LOG_DIR}/launchd.stderr.log</string>
  </dict>
</plist>
EOF
}

install_agent() {
  write_monitor_plist
  launchctl bootout "${MONITOR_SERVICE_TARGET}" >/dev/null 2>&1 || true
  launchctl bootstrap "${GUI_DOMAIN}" "${MONITOR_PLIST}"
  launchctl kickstart -k "${MONITOR_SERVICE_TARGET}"
  printf 'Installed and started %s\n' "${MONITOR_SERVICE_TARGET}"
  printf 'Plist: %s\n' "${MONITOR_PLIST}"
}

uninstall_agent() {
  launchctl bootout "${MONITOR_SERVICE_TARGET}" >/dev/null 2>&1 || true
  rm -f "${MONITOR_PLIST}"
  printf 'Removed %s\n' "${MONITOR_PLIST}"
}

show_status() {
  printf 'Monitor LaunchAgent: %s\n' "${MONITOR_SERVICE_TARGET}"
  printf 'Plist: %s\n' "${MONITOR_PLIST}"
  if [[ -f "${MONITOR_PLIST}" ]]; then
    printf '\n[launchctl]\n'
    launchctl print "${MONITOR_SERVICE_TARGET}" 2>&1 || true
  else
    printf '\nMonitor LaunchAgent plist is not installed.\n'
  fi
  printf '\n[API LaunchAgent]\n'
  launchctl print "${API_SERVICE_TARGET}" 2>&1 || true
  printf '\n[recent diagnostics]\n'
  if [[ -f "${DIAG_LOG}" ]]; then
    tail -n 10 "${DIAG_LOG}"
  else
    printf 'No diagnostics log yet.\n'
  fi
}

command="${1:-once}"
case "${command}" in
  once)
    check_once
    ;;
  loop)
    check_loop
    ;;
  install)
    install_agent
    ;;
  uninstall)
    uninstall_agent
    ;;
  status)
    show_status
    ;;
  *)
    echo "Usage: $(basename "$0") {once|loop|install|uninstall|status}" >&2
    exit 1
    ;;
esac
