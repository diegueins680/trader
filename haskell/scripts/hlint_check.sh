#!/usr/bin/env bash

set -euo pipefail

run_hlint() {
  printf '\n==> hlint'
  printf ' %q' "$@"
  printf '\n'
  hlint "$@"
}

run_hlint_batch() {
  local max_batch="${HLINT_CHECK_BATCH_SIZE:-12}"
  local batch=()
  local file

  while IFS= read -r file; do
    batch+=("${file}")
    if [ "${#batch[@]}" -ge "${max_batch}" ]; then
      run_hlint "${batch[@]}"
      batch=()
    fi
  done < <(find app test bench -name '*.hs' ! -path 'app/Main.hs' -print | sort)

  if [ "${#batch[@]}" -gt 0 ]; then
    run_hlint "${batch[@]}"
  fi
}

# Main imports most of the application and is slow enough to trip long
# monolithic HLint runs in local agent sessions, so lint it separately.
run_hlint app/Main.hs
run_hlint_batch
