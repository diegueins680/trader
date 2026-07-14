#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNNER_PATH="${ROOT_DIR}/scripts/research/collect_datafeed.py"
DEFAULT_LABEL="ai.openclaw.trader.research-datafeed"
LABEL="${TRADER_RESEARCH_LAUNCHD_LABEL:-${DEFAULT_LABEL}}"
CACHE_DIR="${TRADER_RESEARCH_CACHE:-${ROOT_DIR}/data/research}"
PLIST_DIR="${HOME}/Library/LaunchAgents"
PLIST_PATH="${PLIST_DIR}/${DEFAULT_LABEL}.plist"
GUI_DOMAIN="gui/$(id -u)"
PATH_VALUE="${PATH:-/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin}"
SYMBOLS_VALUE="${TRADER_RESEARCH_SYMBOLS:-BTCUSDT ETHUSDT SOLUSDT BNBUSDT XRPUSDT DOGEUSDT ADAUSDT AVAXUSDT LINKUSDT LTCUSDT}"
COLLECT_MINUTE="${TRADER_RESEARCH_COLLECT_MINUTE:-10}"
MAX_RUN_SECONDS="${TRADER_RESEARCH_MAX_RUN_SECONDS:-3000}"
PYTHON_BIN="${TRADER_RESEARCH_PYTHON:-}"
LABEL_OVERRIDDEN="${TRADER_RESEARCH_LAUNCHD_LABEL:+1}"
CACHE_OVERRIDDEN="${TRADER_RESEARCH_CACHE:+1}"
SYMBOLS_OVERRIDDEN="${TRADER_RESEARCH_SYMBOLS:+1}"
MINUTE_OVERRIDDEN="${TRADER_RESEARCH_COLLECT_MINUTE:+1}"
MAX_RUN_OVERRIDDEN="${TRADER_RESEARCH_MAX_RUN_SECONDS:+1}"
PYTHON_OVERRIDDEN="${TRADER_RESEARCH_PYTHON:+1}"
INSTALLED_LABEL=""
command="${1:-status}"

usage() {
  cat <<EOF
Usage: $(basename "$0") <install|uninstall|restart|status|write-plist|print-plist-path>

install          Validate dependencies, write the plist, bootstrap it, and run now.
uninstall        Boot out the LaunchAgent and remove only its plist.
restart          Kickstart the installed LaunchAgent immediately.
status           Show launchd state and the collector's last-run record.
write-plist      Validate and write the plist without loading it.
print-plist-path Print the exact LaunchAgent plist path.
EOF
}

resolve_python() {
  local candidate="${PYTHON_BIN:-python3}"
  local resolved
  resolved="$(command -v "${candidate}" 2>/dev/null || true)"
  if [[ -z "${resolved}" || ! -x "${resolved}" ]]; then
    printf 'A runnable python3 is required; set TRADER_RESEARCH_PYTHON.\n' >&2
    exit 1
  fi
  if [[ "${resolved}" != /* ]]; then
    resolved="$(cd "$(dirname "${resolved}")" && pwd)/$(basename "${resolved}")"
  fi
  PYTHON_BIN="${resolved}"
}

read_installed_value() {
  local key="$1"
  if [[ -x /usr/libexec/PlistBuddy ]]; then
    /usr/libexec/PlistBuddy -c "Print :${key}" "${PLIST_PATH}"
    return
  fi
  local parser="${TRADER_RESEARCH_PYTHON:-}"
  if [[ -z "${parser}" ]]; then
    parser="$(command -v python3 || true)"
  fi
  if [[ -z "${parser}" || ! -x "${parser}" ]]; then
    printf 'Reading the installed plist requires PlistBuddy or python3.\n' >&2
    exit 1
  fi
  "${parser}" -c \
    'import plistlib, sys
with open(sys.argv[1], "rb") as handle:
    value = plistlib.load(handle)
for key in sys.argv[2].split(":"):
    value = value[int(key)] if isinstance(value, list) else value[key]
print(value)' \
    "${PLIST_PATH}" "${key}"
}

load_installed_configuration() {
  [[ -f "${PLIST_PATH}" ]] || return 0
  INSTALLED_LABEL="$(read_installed_value Label)"
  local installed_cache installed_symbols installed_minute installed_max_run installed_python
  installed_cache="$(read_installed_value EnvironmentVariables:TRADER_RESEARCH_CACHE)"
  installed_symbols="$(read_installed_value EnvironmentVariables:TRADER_RESEARCH_SYMBOLS)"
  installed_minute="$(read_installed_value StartCalendarInterval:Minute)"
  installed_max_run="$(read_installed_value EnvironmentVariables:TRADER_RESEARCH_MAX_RUN_SECONDS)"
  installed_python="$(read_installed_value ProgramArguments:0)"

  if [[ "${command}" == "status" || "${command}" == "uninstall" || "${command}" == "restart" ]]; then
    LABEL="${INSTALLED_LABEL}"
    CACHE_DIR="${installed_cache}"
    SYMBOLS_VALUE="${installed_symbols}"
    COLLECT_MINUTE="${installed_minute}"
    MAX_RUN_SECONDS="${installed_max_run}"
    PYTHON_BIN="${installed_python}"
    return
  fi
  [[ -n "${LABEL_OVERRIDDEN}" ]] || LABEL="${INSTALLED_LABEL}"
  [[ -n "${CACHE_OVERRIDDEN}" ]] || CACHE_DIR="${installed_cache}"
  [[ -n "${SYMBOLS_OVERRIDDEN}" ]] || SYMBOLS_VALUE="${installed_symbols}"
  [[ -n "${MINUTE_OVERRIDDEN}" ]] || COLLECT_MINUTE="${installed_minute}"
  [[ -n "${MAX_RUN_OVERRIDDEN}" ]] || MAX_RUN_SECONDS="${installed_max_run}"
  [[ -n "${PYTHON_OVERRIDDEN}" ]] || PYTHON_BIN="${installed_python}"
}

refresh_paths() {
  STATE_DIR="${CACHE_DIR}/.collector"
  STDOUT_PATH="${STATE_DIR}/launchd.stdout.log"
  STDERR_PATH="${STATE_DIR}/launchd.stderr.log"
  STATUS_PATH="${STATE_DIR}/last-run.json"
  SERVICE_TARGET="${GUI_DOMAIN}/${LABEL}"
}

validate_label() {
  if [[ ! "${LABEL}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    printf 'TRADER_RESEARCH_LAUNCHD_LABEL contains invalid characters.\n' >&2
    exit 1
  fi
}

validate_configuration() {
  resolve_python
  if [[ "${CACHE_DIR}" != /* ]]; then
    printf 'TRADER_RESEARCH_CACHE must be an absolute path for launchd.\n' >&2
    exit 1
  fi
  if [[ ! "${COLLECT_MINUTE}" =~ ^[0-9]+$ || ${#COLLECT_MINUTE} -gt 2 ]]; then
    printf 'TRADER_RESEARCH_COLLECT_MINUTE must be an integer from 0 through 59.\n' >&2
    exit 1
  fi
  COLLECT_MINUTE="$((10#${COLLECT_MINUTE}))"
  if (( COLLECT_MINUTE > 59 )); then
    printf 'TRADER_RESEARCH_COLLECT_MINUTE must be an integer from 0 through 59.\n' >&2
    exit 1
  fi
  if [[ ! "${MAX_RUN_SECONDS}" =~ ^[0-9]+$ || ${#MAX_RUN_SECONDS} -gt 4 ]]; then
    printf 'TRADER_RESEARCH_MAX_RUN_SECONDS must be an integer from 60 through 3500.\n' >&2
    exit 1
  fi
  MAX_RUN_SECONDS="$((10#${MAX_RUN_SECONDS}))"
  if (( MAX_RUN_SECONDS < 60 || MAX_RUN_SECONDS > 3500 )); then
    printf 'TRADER_RESEARCH_MAX_RUN_SECONDS must be an integer from 60 through 3500.\n' >&2
    exit 1
  fi
  if [[ ! -f "${RUNNER_PATH}" ]]; then
    printf 'Research collector runner not found: %s\n' "${RUNNER_PATH}" >&2
    exit 1
  fi
  "${PYTHON_BIN}" -c 'import numpy, pandas' >/dev/null
  TRADER_RESEARCH_SYMBOLS="${SYMBOLS_VALUE}" \
    "${PYTHON_BIN}" -c \
      'import sys; sys.path.insert(0, sys.argv[1]); import collect_datafeed; collect_datafeed._symbols_from_environment()' \
      "${ROOT_DIR}/scripts/research"
}

ensure_dirs() {
  mkdir -p "${PLIST_DIR}" "${CACHE_DIR}" "${STATE_DIR}"
  if [[ ! -w "${CACHE_DIR}" || ! -w "${STATE_DIR}" ]]; then
    printf 'Research cache/state directory is not writable: %s\n' "${CACHE_DIR}" >&2
    exit 1
  fi
}

render_plist() {
  local destination="$1"
  validate_configuration
  ensure_dirs
  "${PYTHON_BIN}" - \
    "${destination}" \
    "${LABEL}" \
    "${PYTHON_BIN}" \
    "${RUNNER_PATH}" \
    "${ROOT_DIR}" \
    "${HOME}" \
    "${PATH_VALUE}" \
    "${CACHE_DIR}" \
    "${SYMBOLS_VALUE}" \
    "${COLLECT_MINUTE}" \
    "${MAX_RUN_SECONDS}" \
    "${STDOUT_PATH}" \
    "${STDERR_PATH}" <<'PY'
import os
import plistlib
import sys
import tempfile

(
    plist_path,
    label,
    python_bin,
    runner_path,
    root_dir,
    home,
    path_value,
    cache_dir,
    symbols,
    collect_minute,
    max_run_seconds,
    stdout_path,
    stderr_path,
) = sys.argv[1:]

payload = {
    "Label": label,
    "ProgramArguments": [python_bin, runner_path],
    "WorkingDirectory": root_dir,
    "EnvironmentVariables": {
        "HOME": home,
        "PATH": path_value,
        "TRADER_RESEARCH_CACHE": cache_dir,
        "TRADER_RESEARCH_SYMBOLS": symbols,
        "TRADER_RESEARCH_MAX_RUN_SECONDS": max_run_seconds,
    },
    "RunAtLoad": True,
    "StartCalendarInterval": {"Minute": int(collect_minute)},
    "ProcessType": "Background",
    "StandardOutPath": stdout_path,
    "StandardErrorPath": stderr_path,
}


def replace_atomically(path, mode, writer, validator):
    descriptor, temporary_path = tempfile.mkstemp(
        dir=os.path.dirname(path), prefix=f".{os.path.basename(path)}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, mode) as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        validator(temporary_path)
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def validate_plist(path):
    with open(path, "rb") as handle:
        generated = plistlib.load(handle)
    if generated != payload:
        raise RuntimeError("generated LaunchAgent plist did not round-trip")


replace_atomically(
    plist_path,
    "w+b",
    lambda handle: plistlib.dump(payload, handle, fmt=plistlib.FMT_XML, sort_keys=False),
    validate_plist,
)
PY
  if command -v plutil >/dev/null 2>&1; then
    plutil -lint "${destination}" >/dev/null
  fi
}

write_plist() {
  if command -v launchctl >/dev/null 2>&1 && [[ -n "${INSTALLED_LABEL}" ]]; then
    local installed_target="${GUI_DOMAIN}/${INSTALLED_LABEL}"
    if launchagent_is_loaded "${installed_target}"; then
      printf 'Refusing to overwrite the plist while %s is loaded; use install.\n' \
        "${installed_target}" >&2
      return 1
    else
      local state_status=$?
      if (( state_status != 1 )); then
        return "${state_status}"
      fi
    fi
  fi
  render_plist "${PLIST_PATH}"
  printf 'Wrote %s\n' "${PLIST_PATH}"
}

require_launchctl() {
  if ! command -v launchctl >/dev/null 2>&1; then
    printf 'launchctl is required to manage the research-data LaunchAgent.\n' >&2
    exit 1
  fi
}

launchagent_is_loaded() {
  local target="$1"
  local print_output
  local print_status
  if print_output="$(launchctl print "${target}" 2>&1)"; then
    print_status=0
  else
    print_status=$?
  fi
  if (( print_status != 0 )); then
    if (( print_status == 113 )) || [[ "${print_output}" == *"Could not find service"* ]]; then
      return 1
    fi
    printf 'Unable to determine LaunchAgent state for %s; definition retained.\n' "${target}" >&2
    return 2
  fi
  return 0
}

bootout_if_loaded() {
  local target="$1"
  if launchagent_is_loaded "${target}"; then
    :
  else
    local state_status=$?
    if (( state_status == 1 )); then
      return 0
    fi
    return "${state_status}"
  fi
  if ! launchctl bootout "${target}" >/dev/null 2>&1; then
    printf 'Failed to stop loaded LaunchAgent %s; definition retained.\n' "${target}" >&2
    return 1
  fi
}

restore_previous_install() {
  local backup_path="$1"
  local had_previous="$2"
  local previous_was_loaded="$3"
  local replacement_target="$4"

  bootout_if_loaded "${replacement_target}" || return 1
  if (( had_previous == 1 )); then
    mv -f "${backup_path}" "${PLIST_PATH}"
    if (( previous_was_loaded == 1 )); then
      if ! launchctl bootstrap "${GUI_DOMAIN}" "${PLIST_PATH}"; then
        printf 'Rollback restored the prior plist but could not restart it.\n' >&2
        return 1
      fi
    fi
  else
    rm -f "${PLIST_PATH}" "${backup_path}"
  fi
}

install_agent() {
  require_launchctl
  local staged_path="${PLIST_PATH}.staged.$$"
  local backup_path="${PLIST_PATH}.backup.$$"
  local previous_target=""
  local previous_was_loaded=0
  local had_previous=0
  if [[ -n "${INSTALLED_LABEL}" ]]; then
    previous_target="${GUI_DOMAIN}/${INSTALLED_LABEL}"
  fi
  render_plist "${staged_path}"
  if [[ -f "${PLIST_PATH}" ]]; then
    cp -p "${PLIST_PATH}" "${backup_path}"
    had_previous=1
  fi
  if [[ -n "${previous_target}" ]]; then
    if launchagent_is_loaded "${previous_target}"; then
      previous_was_loaded=1
    else
      local previous_state=$?
      if (( previous_state != 1 )); then
        rm -f "${staged_path}" "${backup_path}"
        return "${previous_state}"
      fi
    fi
  fi
  if [[ "${SERVICE_TARGET}" != "${previous_target}" ]]; then
    if launchagent_is_loaded "${SERVICE_TARGET}"; then
      printf 'Replacement target %s is already loaded outside this installer.\n' \
        "${SERVICE_TARGET}" >&2
      rm -f "${staged_path}" "${backup_path}"
      return 1
    else
      local replacement_state=$?
      if (( replacement_state != 1 )); then
        rm -f "${staged_path}" "${backup_path}"
        return "${replacement_state}"
      fi
    fi
  fi
  if (( previous_was_loaded == 1 )); then
    if ! launchctl bootout "${previous_target}" >/dev/null 2>&1; then
      printf 'Failed to stop loaded LaunchAgent %s; definition retained.\n' \
        "${previous_target}" >&2
      rm -f "${staged_path}" "${backup_path}"
      return 1
    fi
  fi
  if ! mv -f "${staged_path}" "${PLIST_PATH}"; then
    restore_previous_install \
      "${backup_path}" "${had_previous}" "${previous_was_loaded}" "${SERVICE_TARGET}"
    return 1
  fi
  if ! launchctl bootstrap "${GUI_DOMAIN}" "${PLIST_PATH}"; then
    printf 'Replacement bootstrap failed; restoring prior LaunchAgent.\n' >&2
    restore_previous_install \
      "${backup_path}" "${had_previous}" "${previous_was_loaded}" "${SERVICE_TARGET}"
    return 1
  fi
  rm -f "${backup_path}"
  printf 'Installed and started %s\n' "${SERVICE_TARGET}"
  printf 'Cache: %s\n' "${CACHE_DIR}"
}

uninstall_agent() {
  require_launchctl
  bootout_if_loaded "${SERVICE_TARGET}"
  rm -f "${PLIST_PATH}"
  printf 'Removed %s; cache retained at %s\n' "${PLIST_PATH}" "${CACHE_DIR}"
}

restart_agent() {
  require_launchctl
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
  printf 'Cache: %s\n' "${CACHE_DIR}"
  if command -v launchctl >/dev/null 2>&1 && [[ -f "${PLIST_PATH}" ]]; then
    printf '\n[launchctl]\n'
    launchctl print "${SERVICE_TARGET}" 2>&1 || true
  else
    printf '\nLaunchAgent plist is not installed.\n'
  fi
  if [[ -f "${STATUS_PATH}" ]]; then
    printf '\n[last run]\n'
    cat "${STATUS_PATH}"
  else
    printf '\nNo collector run has been recorded.\n'
  fi
}

load_installed_configuration
refresh_paths
validate_label
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
  write-plist)
    write_plist
    ;;
  print-plist-path)
    printf '%s\n' "${PLIST_PATH}"
    ;;
  *)
    usage >&2
    exit 1
    ;;
esac
