#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python3 scripts/validate-toml.py

# GitHub's Ubuntu runner includes Docker Compose, so CI validates interpolation,
# required variables, and the complete service graph. Local environments without
# the Compose plugin still get strict TOML parsing above.
if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
  for env_file in \
    deploy/hetzner/trader.trading.env.example \
    deploy/hetzner/trader.research.env.example
  do
    docker compose \
      -f deploy/hetzner/docker-compose.yml \
      --env-file "$env_file" \
      config --quiet
    printf 'validated Compose config: %s\n' "$env_file"
  done
else
  printf 'Docker Compose unavailable; skipped Compose interpolation validation.\n' >&2
fi
