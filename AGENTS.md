# AGENTS.md (CI agents)

This file provides contributor guidance for CI agents working in this repository.

## General expectations (repo-wide)
- Follow best Haskell practices: clear, total functions where feasible, explicit types for exported functions, and small, well-named helpers.
- Keep changes minimal and focused.
- For any user-facing change or CLI/API behavior change, update **README.md** and **CHANGELOG.md**.
- Never commit secrets. Use environment variables or `.env` files (see `.env.example`) and keep credentials out of logs.

## Canonical commands
- Prefer repo-root verification wrappers over ad hoc command sequences:
  - `bash scripts/verify.sh haskell`
  - `bash scripts/verify.sh web`
  - `bash scripts/verify.sh automation`
  - `bash scripts/verify.sh full`
- Equivalent npm aliases exist:
  - `npm run verify:haskell`
  - `npm run verify:web`
  - `npm run verify:automation`
  - `npm run verify`

## Required checks before PR
- Run `bash scripts/verify.sh full` from the repo root.
- At minimum, run the targeted wrapper for the area you changed:
  - Haskell backend / CLI / smoke-path changes: `bash scripts/verify.sh haskell`
  - Web UI changes: `bash scripts/verify.sh web`
  - Autoloop / root automation changes: `bash scripts/verify.sh automation`

## Recommended tools
- **Formatting:** prefer `fourmolu` (or `ormolu` if `fourmolu` is unavailable).
  - Example: `fourmolu -i $(rg --files -g '*.hs')`
- **Linting:** `hlint`.
  - Example: `cd haskell && hlint app test bench`
- **Testing:** `cabal test`

Avoid mixing formatters (e.g., do not run `stylish-haskell` alongside `ormolu/fourmolu`).

## Toolchain
- Local and CI targets are pinned to:
  - GHC `9.4.8`
  - Cabal `3.12.1.0`
  - Node `20.19.0`
  - `fourmolu 0.15.0.0`
  - `hlint 3.8`
- Prefer the checked-in `.tool-versions` and `.nvmrc` when setting up a local or agent workspace.

## Directory-specific guidance

### `haskell/`
- Build/run/test from this directory:
  - `cabal build`
  - `cabal run trader-hs -- --version`
  - `cabal test`
- Keep new modules organized by feature (e.g., predictors in `app/Trader/Predictors.hs`).
- When adding new CLI flags, update the README usage section and ensure JSON output remains stable.

### `scripts/` and `test/`
- Root automation code lives in `scripts/autoloop*.mjs`, `scripts/codex-logical-correctness-loop.sh`, and `test/autoloop.test.mjs`.
- Changes there should run `bash scripts/verify.sh automation`.
- If you add a new automation verification command, keep it aligned with the allowlist in `scripts/autoloop.mjs`.

### `deploy/` and `deploy-aws-*`
- Treat deployment scripts as sensitive; avoid embedding secrets and document required environment variables.

## Environment setup
- Install GHC and Cabal via **ghcup** (recommended), and prefer the checked-in toolchain pins.
- From the repo root:
  - `cd haskell`
  - `cabal update`
  - `cabal build`
  - `npm --workspaces=false --prefix haskell/web ci`
- Copy `.env.example` to `.env` for local secrets as needed.
- AI automation toggles for local use are also documented in `.env.example`.

## CI/CD notes (suggested)
- CI should run the same repo-root verification wrappers that local agents use.
- CI should cover Haskell, web, and root automation tests separately.
- Any deployment scripts should assume credentials are injected via environment variables (never committed).

## Gotchas & tips
- Most low-level Haskell commands still run from the `haskell/` directory, but prefer the repo-root verification wrappers when they exist.
- Binance credentials must be set via `BINANCE_API_KEY` / `BINANCE_API_SECRET` and should never be logged.
- GitHub-hosted autoloop additionally requires `OPENAI_API_KEY` and `AUTOLOOP_PUSH_TOKEN` secrets.

## Templates (examples)
- Verify Haskell: `bash scripts/verify.sh haskell`
- Verify web: `bash scripts/verify.sh web`
- Verify automation: `bash scripts/verify.sh automation`
- Verify all: `bash scripts/verify.sh full`
- Run: `cd haskell && cabal run trader-hs -- --data ../data/sample_prices.csv --price-column close`
