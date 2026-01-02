# AGENTS.md (CI agents)

This file provides contributor guidance for CI agents working in this repository.

## General expectations (repo-wide)
- Follow best Haskell practices: clear, total functions where feasible, explicit types for exported functions, and small, well-named helpers.
- Keep changes minimal and focused.
- For any user-facing change or CLI/API behavior change, update **README.md** and **CHANGELOG.md**.
- Never commit secrets. Use environment variables or `.env` files (see `.env.example`) and keep credentials out of logs.

## Required checks before PR
- Ensure the code compiles:
  - `cd haskell`
  - `cabal build`

## Recommended tools
- **Formatting:** prefer `fourmolu` (or `ormolu` if `fourmolu` is unavailable).
  - Example: `fourmolu -i $(rg --files -g '*.hs')`
- **Linting:** `hlint`.
  - Example: `cd haskell && hlint app test`
- **Testing:** `cabal test`

Avoid mixing formatters (e.g., do not run `stylish-haskell` alongside `ormolu/fourmolu`).

## Directory-specific guidance

### `haskell/`
- Build/run/test from this directory:
  - `cabal build`
  - `cabal run trader-hs -- --version`
  - `cabal test`
- Keep new modules organized by feature (e.g., predictors in `app/Trader/Predictors.hs`).
- When adding new CLI flags, update the README usage section and ensure JSON output remains stable.

### `deploy/` and `deploy-aws-*`
- Treat deployment scripts as sensitive; avoid embedding secrets and document required environment variables.

## Environment setup
- Install GHC and Cabal via **ghcup** (recommended).
- From the repo root:
  - `cd haskell`
  - `cabal update`
  - `cabal build`
- Copy `.env.example` to `.env` for local secrets as needed.

## CI/CD notes (suggested)
- CI should run `cabal build` and `cabal test`.
- Any deployment scripts should assume credentials are injected via environment variables (never committed).

## Gotchas & tips
- Most commands must be run from the `haskell/` directory (paths in README assume this).
- Binance credentials must be set via `BINANCE_API_KEY` / `BINANCE_API_SECRET` and should never be logged.

## Templates (examples)
- Build: `cd haskell && cabal build`
- Test: `cd haskell && cabal test`
- Run: `cd haskell && cabal run trader-hs -- --data ../data/sample_prices.csv --price-column close`
