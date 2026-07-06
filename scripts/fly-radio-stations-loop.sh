#!/bin/sh

set -eu

interval="${RADIO_STATION_INTERVAL_SECONDS:-900}"
stations_file="${RADIO_STATIONS_FILE:-/var/lib/trader/state/radio-stations.json}"

case "$interval" in
  ''|*[!0-9]*)
    echo "radio-stations invalid RADIO_STATION_INTERVAL_SECONDS=$interval; expected integer seconds" >&2
    exit 2
    ;;
esac

mkdir -p "$(dirname "$stations_file")"

while true; do
  started_at="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  echo "radio-stations loop started_at=$started_at interval_seconds=$interval stations_file=$stations_file"
  if ! node /opt/trader/scripts/maintain-radio-stations.mjs --stations-file "$stations_file" --json; then
    echo "radio-stations maintenance failed; retrying after ${interval}s" >&2
  fi
  sleep "$interval"
done
