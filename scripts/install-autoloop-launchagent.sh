#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
STATE_DIR="${ROOT_DIR}/.tmp/autoloop"
LABEL="${AUTOLOOP_LAUNCHD_LABEL:-ai.openclaw.trader.autoloop.forever}"
PLIST_DIR="${HOME}/Library/LaunchAgents"
PLIST_PATH="${PLIST_DIR}/${LABEL}.plist"
LAUNCHD_OUT="${STATE_DIR}/launchd.stdout.log"
LAUNCHD_ERR="${STATE_DIR}/launchd.stderr.log"
GUI_DOMAIN="gui/$(id -u)"
SERVICE_TARGET="${GUI_DOMAIN}/${LABEL}"
PATH_VALUE="${PATH:-/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin}"

usage() {
  cat <<EOF
Usage: $(basename "$0") <install|uninstall|restart|status|print-plist-path>

install          Write the LaunchAgent plist, bootstrap it, and kickstart the runner.
uninstall        Boot out the LaunchAgent and remove the plist.
restart          Kickstart the installed LaunchAgent.
status           Show launchd status plus repo-local autoloop status.
print-plist-path Print the exact LaunchAgent plist path.
EOF
}

ensure_dirs() {
  mkdir -p "${PLIST_DIR}" "${STATE_DIR}"
}

write_plist() {
  ensure_dirs
  cat > "${PLIST_PATH}" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
  <dict>
    <key>Label</key>
    <string>${LABEL}</string>

    <key>ProgramArguments</key>
    <array>
      <string>/bin/bash</string>
      <string>${ROOT_DIR}/scripts/autoloop-forever.sh</string>
      <string>run</string>
    </array>

    <key>WorkingDirectory</key>
    <string>${ROOT_DIR}</string>

    <key>EnvironmentVariables</key>
    <dict>
      <key>HOME</key>
      <string>${HOME}</string>
      <key>PATH</key>
      <string>${PATH_VALUE}</string>
    </dict>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <true/>

    <key>StandardOutPath</key>
    <string>${LAUNCHD_OUT}</string>

    <key>StandardErrorPath</key>
    <string>${LAUNCHD_ERR}</string>
  </dict>
</plist>
EOF
}

install_agent() {
  write_plist
  launchctl bootout "${SERVICE_TARGET}" >/dev/null 2>&1 || true
  launchctl bootstrap "${GUI_DOMAIN}" "${PLIST_PATH}"
  launchctl kickstart -k "${SERVICE_TARGET}"
  printf 'Installed and started %s\n' "${SERVICE_TARGET}"
  printf 'Plist: %s\n' "${PLIST_PATH}"
}

uninstall_agent() {
  launchctl bootout "${SERVICE_TARGET}" >/dev/null 2>&1 || true
  rm -f "${PLIST_PATH}"
  printf 'Removed %s\n' "${PLIST_PATH}"
}

restart_agent() {
  if [[ ! -f "${PLIST_PATH}" ]]; then
    printf 'LaunchAgent plist not found at %s\n' "${PLIST_PATH}" >&2
    exit 1
  fi
  launchctl kickstart -k "${SERVICE_TARGET}"
  printf 'Restarted %s\n' "${SERVICE_TARGET}"
}

show_status() {
  printf 'LaunchAgent: %s\n' "${SERVICE_TARGET}"
  printf 'Plist: %s\n' "${PLIST_PATH}"
  if [[ -f "${PLIST_PATH}" ]]; then
    printf '\n[launchctl]\n'
    launchctl print "${SERVICE_TARGET}" 2>&1 || true
  else
    printf '\nLaunchAgent plist is not installed.\n'
  fi
  if [[ -x "${ROOT_DIR}/scripts/autoloop-forever.sh" ]]; then
    printf '\n[repo autoloop status]\n'
    "${ROOT_DIR}/scripts/autoloop-forever.sh" status 2>&1 || true
  fi
}

command="${1:-status}"
case "${command}" in
  install)
    install_agent
    ;;
  uninstall)
    uninstall_agent
    ;;
  restart)
    restart_agent
    ;;
  status)
    show_status
    ;;
  print-plist-path)
    printf '%s\n' "${PLIST_PATH}"
    ;;
  *)
    usage >&2
    exit 1
    ;;
esac
