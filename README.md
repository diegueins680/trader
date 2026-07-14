# Trader

Trader is a Haskell trading research and execution system with a React operations UI. It supports exchange-backed signals, cost-aware backtests, optimizer research, live-bot supervision, formal safety checks, and Fly/Hetzner deployment.

> Live trading can place real orders. Keep `TRADER_BINANCE_LIVE=false` and `TRADER_BOT_TRADE=false` until credentials, risk limits, deployment identity, and exchange permissions have been verified. The browser starts in paper mode and never starts a bot merely because a combo was applied.

## What is included

- Haskell CLI, REST API, backtester, optimizer, and supervised live bots
- Binance, Coinbase, Kraken, Poloniex, and CSV market-data paths
- LSTM, Kalman, online-neural, technical-analysis, and ensemble methods
- Cost, exposure, drawdown, stale-data, capital-preservation, and execution gates
- Walk-forward and overfit-aware optimizer evidence with top-combo adoption controls
- React monitoring UI for signals, bots, positions, trades, optimizer results, and operations
- Machine-checked formal-spec registry and repo-wide verification wrappers

See [CHANGELOG.md](CHANGELOG.md) for detailed feature history and [.env.example](.env.example) for the complete runtime configuration surface.

## Toolchain

The checked-in versions are authoritative:

- GHC `9.4.8`
- Cabal `3.12.1.0`
- Node `20.19.0`
- fourmolu `0.15.0.0`
- HLint `3.8`

Use `.tool-versions` and `.nvmrc` where possible.

## Quick start

Install the pinned toolchain, then from the repository root:

```bash
cp .env.example .env

cd haskell
cabal update
cabal build
cabal run trader-hs -- --version
cabal run trader-hs -- --data ../data/sample_prices.csv --price-column close
```

Install and build the web UI:

```bash
npm --workspaces=false --prefix haskell/web ci
npm --workspaces=false --prefix haskell/web run build
```

For local API and UI development, start PostgreSQL first. Serve mode requires ops persistence:

```bash
bash haskell/scripts/start_api_bg.sh
bash haskell/scripts/start_ui_bg.sh
```

Defaults are API `http://127.0.0.1:8080` and UI `http://127.0.0.1:5174`. Logs default to `/tmp/trader-api.log` and `/tmp/trader-ui.log`.

## Verification

Use the repo-root wrappers; these are also the CI contract:

```bash
bash scripts/verify.sh haskell
bash scripts/verify.sh web
bash scripts/verify.sh automation
bash scripts/verify.sh full
```

Equivalent npm commands are `npm run verify:haskell`, `npm run verify:web`, `npm run verify:automation`, and `npm run verify`.

The automation gate validates Fly TOML, both Hetzner Compose roles, formal-spec coverage, the canonical risk register, and root automation regressions. Formal feature contracts live in `formal/specifications.json`; canonical risk IDs and lifecycle state live in `formal/risk-register.json`; the verification model is explained in `docs/formal-specifications.md`.

## CLI and API

List the complete CLI surface with:

```bash
cd haskell
cabal run trader-hs -- --help
cabal run optimize-equity -- --help
```

Serve mode requires `TRADER_DB_URL` or `DATABASE_URL`. The local wrapper supplies the conventional local PostgreSQL URL when possible.

Important operational endpoints:

- `GET /health` — process liveness and build commit
- `GET /ready` — readiness; returns HTTP 503 while the server is draining
- `GET /admin/health` — authenticated job and queue detail
- `GET /metrics` — runtime metrics
- `POST /signal`, `/backtest`, `/trade` — synchronous work
- `/signal/async`, `/backtest/async`, `/trade/async` — persisted asynchronous work
- `/bot/start`, `/bot/status`, `/bot/stop` — live-bot lifecycle

On SIGTERM or SIGINT, serve mode marks readiness as draining, rejects new compute/order/bot-start work, persists `server.stop` and bot snapshots, stops workers and jobs, closes PostgreSQL, and observes `TRADER_API_SHUTDOWN_TIMEOUT_SEC` (default `20`).

The web client runtime-validates the safety envelope returned by `/bot/start`, `/bot/status`, and `/bot/stop` before updating bot state. Malformed successful responses become explicit errors and are not retried as mutations.

Optimizer result filters remain authoritative: a filtered trial is never emitted as an eligible result. Trials rejected only by selected soft performance filters may remain internal search parents when they have final equity and present walk-forward stability evidence within the search cap; the stricter survivor activity and annual-return floors still govern exploitation parent selection. Configuration, risk, data, edge, Kelly, and missing-walk-forward failures remain hard exclusions.

## Edge research campaign

Run the pre-registered residual-momentum campaign from the repository root:

```bash
python3 scripts/research/run_edge_campaign.py \
  BTCUSDT ETHUSDT SOLUSDT BNBUSDT XRPUSDT \
  DOGEUSDT ADAUSDT AVAXUSDT LINKUSDT LTCUSDT
```

The campaign evaluates exactly 15 causal trials: `24h`, `72h`, and `168h` residual momentum crossed with base, funding/basis, open-interest, taker-flow, and all-feature ablations. Close-derived signals activate one full bar later. Final selection and every nested outer fold use the same expanding inner-OOS Sharpe rule with an exact label-horizon embargo; the stitched OOS path charges direct cross-fold position turnover. Cost and additional-delay stresses keep the base outer-fold selections frozen and must clear block-bootstrap OOS confidence gates. Formal deflated Sharpe and balanced CSCV probability-of-backtest-overfitting diagnostics use complete daily-compounded trial returns and fail closed when evidence is incomplete, sparse, or degenerate.

Generated evidence is written to `.tmp/research/edge-campaign/`, including an immutable code/config/data manifest, panel hash, complete trial ledger, gross/net/turnover/weight paths, aligned return matrices, separate daily DSR and CSCV/PBO matrices, final-selection and inner/outer fold records, nested OOS returns, stressed OOS paths, regime/fold metrics, and promotion-gate results. Cached data is used unless `--refresh` is passed. The final chronological 20% holdout is sealed by default; `--open-final-holdout` evaluates it only after all sample-size, activity, joint finite derivatives-coverage, uncertainty, fold, regime, DSR/PBO, doubled-cost, and additional-delay gates pass. An overlap-aware registry under `.tmp/research/edge-campaign-holdouts/` makes successful or interrupted openings one-shot across output directories for intersecting symbol/time windows, including the final candle's outcome interval; blocked requests do not consume it, and completed entries reference atomically persisted evidence.

The Haskell backtester can opt into the same point-in-time Binance derivatives inputs:

```bash
cd haskell
cabal run trader-hs -- \
  --symbol BTCUSDT --futures --exogenous-derivatives --json
```

`--exogenous-derivatives` is research-backtest-only. CLI validation rejects CSV, spot, trade-only, order, live, and server combinations, and unavailable fetches leave the existing neutral features unchanged.

## Live-trading safety

Real Binance orders require all applicable controls to agree:

- the server was launched with live capability (`--binance-live` or the deployment wrapper equivalent)
- `TRADER_BINANCE_LIVE=true`
- `TRADER_BOT_TRADE=true` for bot execution
- valid `BINANCE_API_KEY` and `BINANCE_API_SECRET`
- the browser/request explicitly enables and arms live trading

Applying an optimizer combo changes parameters only. Starting a live bot is a separate explicit action, and stale, unauthenticated, or failed API states disable that action.

Credentials must remain in environment variables or ignored `.env` files. Never put credentials in committed files or logs.

Tenant identity is derived consistently by the backend, browser, and AWS deployment helper. Credential boundaries trim only ASCII whitespace so all three runtimes agree; separator-free tuples, including Unicode credentials, retain their existing `platform:<hash>` key. Tuples containing `:` use collision-resistant `platform:v2:<hash>` framing. This intentionally changes the key for those exceptional existing tenants: migrate tenant-scoped database rows, bot snapshots/object keys, and any explicit `TRADER_STATE_SYNC_TENANT_KEY` together. When updating an existing App Runner service to a v2 target, set `TRADER_STATE_SYNC_SOURCE_TENANT_KEY` to the tenant key used by the deployed service; the deploy exports from that source and imports into the v2 target, and refuses the update when the source is omitted. An old ambiguous alias cannot be accepted safely.

## Neural-governor rollout

The online neural governor is separated into three rollout modes:

- `observe` — train and report scores without evaluating or applying policy actions
- `shadow` — evaluate entry/sizing counterfactuals without changing live orders; this is the default
- `enforce` — apply the policy only after its sample and counterfactual-advantage promotion gates pass

An enforced policy automatically rolls back to non-enforcing behavior when post-promotion counterfactual advantage breaches the configured floor. The current mode, promotion state, rollback state, evaluation count, and advantage are available in `/bot/status` and the UI.

Core settings:

```dotenv
TRADER_BOT_NEURAL_GOVERNOR_ROLLOUT_MODE=shadow
TRADER_BOT_NEURAL_GOVERNOR_PROMOTION_MIN_TRADES=30
TRADER_BOT_NEURAL_GOVERNOR_PROMOTION_MIN_ADVANTAGE=0.01
TRADER_BOT_NEURAL_GOVERNOR_ROLLBACK_MIN_TRADES=10
TRADER_BOT_NEURAL_GOVERNOR_ROLLBACK_ADVANTAGE_FLOOR=-0.02
```

## Deployment

- Hetzner: `deploy/hetzner/` and [deploy/COMBOS_SYNC.md](deploy/COMBOS_SYNC.md)
- AWS: [DEPLOY_AWS.md](DEPLOY_AWS.md) and [DEPLOY_AWS_QUICKSTART.md](DEPLOY_AWS_QUICKSTART.md)
- Render: [deploy/render/README.md](deploy/render/README.md)
- Fly: `fly.toml`, `fly.research.toml`, and `haskell/web/fly.frontend.toml`

For Hetzner CI, both trading and research boxes are mandatory. The workflow requires the SSH key, both role hosts, and pinned `known_hosts` entries, and deploys only the latest green commit. Deployments retain the previous API image, wait for container health, verify the exact `/health` commit, and roll back automatically if health or attestation fails.

## Automation and maintenance

- `npm run autoloop` runs one bounded repo automation cycle.
- `scripts/autoloop-forever.sh` and `scripts/install-autoloop-launchagent.sh` provide the repo-native persistent macOS lane.
- `npm run radio:maintain` checks configured radio streams and refreshes the station file; see `scripts/radio-stations.cron.example`.
- [docs/ai-workflow.md](docs/ai-workflow.md) documents the AI/bootstrap workflow.

Autoloop never auto-resolves merge conflicts. Conflicted branches remain for operator review, and branch promotion must pass the matching canonical verification wrapper before push.

## Project layout

```text
haskell/app/          Haskell executables and Trader modules
haskell/test/         Haskell regression and formal-verification tests
haskell/web/          React/Vite operations UI
scripts/              Verification, automation, research, and maintenance tools
test/                 Root Node automation tests
formal/               Machine-readable feature contracts
deploy/               Deployment configuration and operator scripts
data/                 Sample and stress datasets
artifacts/, reports/  Research, risk, and engineering records
```

## License

See [LICENSE](LICENSE).
