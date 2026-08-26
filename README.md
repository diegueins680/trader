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
- Persisted bot ownership for exchange positions, so system-opened positions remain attributable across bot restarts
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

## Alternative data

The research pipeline can ingest non-exchange CSV, JSON, and RSS feeds for macro data, policy releases, news, company fundamentals and filings, on-chain activity, stablecoins, institutional/custody flows, miner and validator activity, developer activity, governance and token supply, search/web attention, social activity, prediction markets, real-world operations, security incidents, CFTC positioning, options volatility, and archival microstructure. The checked-in [source configuration](scripts/research/alternative-data.example.json) contains a disabled example for every family; enable only feeds whose timestamps, licenses, and credentials you have verified.

Each cached observation records the underlying event time separately from the time it became available. Sources with provider release or vintage evidence must name it with `availableTimeField`; sources without that field become available only when the collector first sees them, and a later changed value is recorded as a new first-seen release instead of being backdated to the historical event. Unchanged re-fetches preserve the original first-seen timestamp. The panel builder uses only availability timestamps, applies publication lags and maximum ages, causally normalizes each metric before combining metrics, and emits coverage alongside every family. Stateful readings are carried forward only until their configured age limit; event/flow readings reset between bars.

Use `aggregation=last` for state, or `sum`, `mean`, and `count` for observations that belong only to their publication bar. `transform` supports `zscore`, `delta`, `pct_change`, and `raw`; `raw` should be reserved for an already bounded/standardized signal. `polarity=-1` makes adverse readings consistently negative, while `minHistory` and `maxAgeSeconds` control causal warm-up and staleness.

After configuring sources, collect them and build a feature panel against the exact price-bar grid:

```bash
python3 scripts/research/alternative_data.py collect \
  --config scripts/research/alternative-data.example.json

python3 scripts/research/alternative_data.py panel \
  --cache data/research/alternative-observations.csv \
  --bars data/research/BTCUSDT_1h.csv \
  --symbol BTCUSDT \
  --output data/research/BTCUSDT_1h-alternative.csv \
  --manifest data/research/BTCUSDT_1h-alternative.json
```

`run` combines collection and panel construction. A provider failure is isolated and reported as degraded; successful sources are still cached atomically, while the command returns non-zero unless `--allow-partial` is explicit. Secrets referenced by `queryFromEnv` or `headersFromEnv` are read only from the environment and are never stored or printed.

Attach one or more generated panels to Haskell research, backtests, or bot predictors with `--external-data --external-data-csv PATH`, or set:

```dotenv
TRADER_EXTERNAL_DATA=true
TRADER_EXTERNAL_DATA_CSVS=data/research/BTCUSDT_1h-alternative.csv
```

In source configuration, `entity`/`entityField` identifies the target trading asset, not an internal provider record ID; an empty entity is global. Panel construction requires either `--symbol` (which includes that full/base asset plus global observations) or an explicit `--global`; generated symbol rows apply only to that full or base asset. Enabling alternative data changes the predictor feature dimension, so models must be retrained and validated through walk-forward/holdout tests. Collection alone does not authorize a signal for live trading, and missing data stays neutral rather than being interpreted as a bullish or bearish observation.

## CLI and API

List the complete CLI surface with:

```bash
cd haskell
cabal run trader-hs -- --help
cabal run optimize-equity -- --help
```

Serve mode requires `TRADER_DB_URL` or `DATABASE_URL`. The local wrapper supplies the conventional local PostgreSQL URL when possible.
Database-backed endpoints serialize access to their shared PostgreSQL connection and automatically replace it after libpq connection failures, including a connection left busy by an interrupted command.

Important operational endpoints:

- `GET /health` — process liveness and build commit
- `GET /ready` — readiness; returns HTTP 503 while the server is draining or a live executor has not yet reconciled exchange inventory and registered an owner for every open-position symbol
- `GET /admin/health` — authenticated job and queue detail
- `GET /metrics` — runtime metrics
- `POST /signal`, `/backtest`, `/trade` — synchronous work
- `/signal/async`, `/backtest/async`, `/trade/async` — persisted asynchronous work
- `POST /binance/revenue` — futures income, execution, and cost reconciliation (signed Binance credentials required)
- `POST /binance/listenKey` and `GET /binance/listenKey/stream` — start and relay a tenant-scoped Binance user-data stream
- `/bot/start`, `/bot/status`, `/bot/stop` — live-bot lifecycle

On SIGTERM or SIGINT, serve mode marks readiness as draining, rejects new compute/order/bot-start work, persists `server.stop` and bot snapshots, stops workers and jobs, closes PostgreSQL, and observes `TRADER_API_SHUTDOWN_TIMEOUT_SEC` (default `20`). Hetzner is the sole live executor; Fly is checked in as a read-only standby and cannot claim positions or place orders. The live profile persists bot state and LSTM weights under `/var/lib/trader`, reuses compatible saved models, and runs five explicitly reviewed relaxed candidates for AVAX, UNI, SUI, ETC, and ADA under independent `shadow` selection. Each adopted strategy is capped at 5%, bounding the five-worker configuration to 25% aggregate configured exposure; an inherited minimum-size floor above 5% is reduced to the adopted maximum so it cannot make the worker structurally unable to enter. The pinned fleet sets `TRADER_BOT_ONLINE_OPTIMIZER_ENABLED=false`, preventing per-bot re-optimization from replacing a reviewed UUID and triggering the pinning controller to restart it. After 30 complete UTC live-evidence days, the server automatically reviews fleet net return, drawdown, order-submission reliability, status health, and strict selector admission; it atomically graduates to effective `enforce` mode and stops using the UUID target allowlist only when every gate passes. Any missing or failing evidence retains the existing shadow safeguards. Flat workers rotate when the selected combo changes, while workers managing an open position retain its origin combo until flat. On replacement startup, `/ready` stays at HTTP 503 with `status=recovering_positions` until exchange inventory has been inspected and every open-position symbol has a registered live owner. Inventory errors fail closed; adoption bypasses portfolio, per-cycle start, backoff, and new-entry circuit caps and precedes new flat-symbol exposure. Backoff filtering happens before start throttling, so one invalid or cooling-down target cannot pin every later bot. Position-owner upserts explicitly target the partial `positions.bot_id` index, making adoption visible immediately to the Positions UI instead of leaving a live runtime labeled orphaned. Bot status remains independent of slower position/database enrichment; until the first successful status response, the Overview displays activity as unknown rather than asserting that the fleet is stopped, and a later timeout preserves the last observed fleet.

The web client runtime-validates the safety envelope returned by `/bot/start`, `/bot/status`, and `/bot/stop` before updating bot state. Malformed successful responses become explicit errors and are not retried as mutations.

`POST /binance/listenKey` returns the tenant key that owns the newly created stream. The web client uses that authoritative value for the immediate SSE request, avoiding a startup race while the browser is still deriving the same tenant identity from inline credentials.

## Revenue accounting and Strategy Assurance

The Binance account page includes an exchange-reconciled futures revenue ledger. It treats Binance income history as the accounting authority, separates realized P&L, funding, signed commissions, rebates, and other operating income, excludes transfers and bonuses from net revenue, and uses account trades only for maker/taker execution metrics. Current unrealized P&L is opt-in. A settlement asset and infrastructure cost can be supplied for an explicit net-revenue view.

The API defaults to the latest seven days and accepts ranges up to 90 days. A response that reaches either 1,000-record request limit is marked as potentially truncated so callers can shorten the period instead of mistaking a partial response for complete accounting.

```bash
curl -sS http://127.0.0.1:8080/binance/revenue \
  -H 'Content-Type: application/json' \
  -d '{"market":"futures","asset":"USDT","infrastructureCost":25}'
```

Keep signed Binance credentials in `BINANCE_API_KEY` and `BINANCE_API_SECRET`; the request body does not need to carry them when the API process already has them.

This evidence is also the financial baseline for the productized [Strategy Assurance review](docs/strategy-assurance.md): a fixed-scope technical and economic audit for algorithmic-trading deployments, with a one-time review and optional recurring monitoring. The dashboard exports a client-ready Markdown snapshot, versioned evidence JSON, and daily/symbol CSV files; the offer includes reusable [proposal/SOW](docs/strategy-assurance-proposal-template.md), [decision memo](docs/strategy-assurance-report-template.md), [findings ledger](docs/strategy-assurance-findings-template.json), and [pre-live evidence checklist](docs/strategy-assurance-pre-live-checklist.md) templates. A dated, source-backed [prospecting brief](docs/strategy-assurance-prospecting-brief.md) and [local outreach queue](docs/strategy-assurance-prospect-queue.json) rank organization-level partner and community routes without scraping members, sending outreach, or treating public fit as buyer intent.

Prepare a human-review package for at most three exact queue organizations:

```bash
npm run assurance:outreach -- \
  --provider "Provider legal name" \
  --sender "Name, title" \
  --prospect "OctoBot"
```

The command writes initial and single-follow-up drafts, source snapshots, exact message hashes, null outcome fields, and a copy of the checklist under `.tmp/strategy-assurance/outreach/`. It rejects unknown and unacknowledged watchlist prospects and performs no send, submission, community join, affiliation claim, member-data collection, or pipeline transition. Run `npm run assurance:outreach -- --help` for queue, checklist, date, output, and overwrite options.

Track only real acquisition events in the separate local pre-proposal registry:

```bash
npm run assurance:acquisition -- import \
  --campaign .tmp/strategy-assurance/outreach/CAMPAIGN/campaign.json

npm run assurance:acquisition -- advance \
  --id LEAD_ID \
  --status contacted \
  --channel "Official organization contact form" \
  --evidence "Receipt or sent-record reference" \
  --at 2026-08-20

npm run assurance:acquisition -- summary
```

Campaign import validates the source package and its content hashes, is idempotent, and prevents a second campaign for the same organization. Forward transitions require dated evidence; contact and follow-up events require the channel, wait periods are based on the actual contact/follow-up dates, and `proposed` requires a matching generated `engagement.json`. Summaries expose exact acquisition conversions, performance by queue source kind, eligible follow-ups and closures, and next actions. The default registry is `.tmp/strategy-assurance/acquisition.json`; this command records events but sends nothing and does not mutate the commercial pipeline. Run `npm run assurance:acquisition -- --help` for the full lifecycle and path options.

Generate a prospect-specific commercial package without sending data or making any network request:

```bash
npm run assurance:kit -- \
  --client "Client legal name" \
  --provider "Provider legal name" \
  --decision-owner "Name, title" \
  --strategy "Strategy name/version" \
  --deployment "Deployment name/region"
```

The command writes a tailored proposal/SOW, evidence request, initial and follow-up outreach copy, pro-forma payment request in Markdown and JSON, and a versioned engagement JSON record under `.tmp/strategy-assurance/`. The payment request is explicitly not a tax invoice or receipt, contains no bank or payment-provider credentials, and records that no external payment action occurred. The generator refuses to replace an existing package unless `--force` is supplied. Run `npm run assurance:kit -- --help` for scope, period, pricing, validity, and output options.

For a qualified acquisition lead, commit the reviewed proposal to both local registries with one recoverable, idempotent handoff:

```bash
npm run assurance:handoff -- commit \
  --lead LEAD_ID \
  --engagement .tmp/strategy-assurance/CLIENT-DATE/engagement.json \
  --evidence "Reviewed proposal sent-record reference" \
  --at 2026-08-21

npm run assurance:handoff -- reconcile --as-of 2026-08-21
```

The handoff validates the complete acquisition and commercial-pipeline states before writing, imports the engagement first, and then records the proposal link. If the second atomic file write is interrupted, rerunning the identical command completes the link without duplicating either record. Reconciliation reports qualified leads awaiting proposals, missing or inconsistent imports, exact linked review value and net cash, and pipeline engagements that need direct/referral provenance. It performs no proposal generation, sending, signing, payment request, charge, or external action.

Import that engagement record into the local commercial pipeline, then advance it only when the real-world event has occurred:

```bash
npm run assurance:pipeline -- import \
  --engagement .tmp/strategy-assurance/CLIENT-DATE/engagement.json

npm run assurance:pipeline -- advance \
  --id ENGAGEMENT_ID \
  --status accepted \
  --at 2026-08-14

npm run assurance:pipeline -- advance \
  --id ENGAGEMENT_ID \
  --status paid \
  --amount 2500 \
  --at 2026-08-15

npm run assurance:pipeline -- summary
```

Use direct pipeline import for referrals or legacy engagements without an acquisition lead; otherwise prefer `assurance:handoff`. The pipeline records an append-only lifecycle from proposal through acceptance, payment, delivery, and optional monitoring. A `paid` event requires the actual USD cash amount, a `refunded` event requires the actual refund, and a `delivered` event requires actual delivery hours. Its versioned summary separates open and booked contract value from gross cash, refunds, net cash, delivered net cash, delivery hours, realized review revenue per hour, contracted monitoring MRR/ARR, expired proposals, exact funnel conversions, and the next action for every active engagement. Status alone never manufactures collected cash. The default registry is `.tmp/strategy-assurance/pipeline.json`; use `npm run assurance:pipeline -- --help` for alternate paths, event evidence, terminal statuses, and JSON output.

After a standard review reaches `delivered`, generate a monitoring order for the recurring-revenue offer:

```bash
npm run assurance:renewal -- \
  --id ENGAGEMENT_ID \
  --offer-date 2026-08-15 \
  --start 2026-09-01 \
  --months 3
```

The renewal command reads the delivered scope and quoted monitoring price from the local pipeline, then writes a Markdown monitoring order and versioned JSON offer with the exact initial contract value. It refuses non-delivered engagements and implicit overwrites. Generation does not send, sign, invoice, charge, or move the engagement to `monitoring`; record that transition only after real acceptance. Run `npm run assurance:renewal -- --help` for validity, registry, and output options.

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

Run the independent historical funding-only campaign without opening its final holdout:

```bash
python3 scripts/research/run_historical_funding_campaign.py --acquire
```

Its committed registration at `research-notes/registrations/residual-momentum-funding-only-v1.json` fixes ten survivors, `6,138` eight-hour bars from `2020-09-23T00:00:00Z` through `2026-04-30T16:00:00Z`, a `4,910`-bar development window, and exactly six `24h`/`72h`/`168h` base-versus-funding-only trials. Acquisition uses only bounded, conservatively rate-paced public Binance contract, funding-event, and one-hour mark-price endpoints; successful and throttled response headers reconcile IP-wide usage into the local one-minute budget, a separate five-minute funding-request budget stays below the endpoint quota, and per-artifact progress makes an interrupted download resumable. Every raw artifact, aligned price row, resolved settlement, registration file, and implementation file is hashed before returns are produced. Blank historical funding marks use the containing one-hour mark candle's open, never its future close; every endpoint-returned event in each `(left close, right close]` interval is charged with the position's sign and mark-to-left-close ratio. Missing pages, grid rows, causal marks, duplicate returned events, or funding gaps larger than the registered tolerance fail closed. The endpoint does not publish a reconstructable historical cadence schedule, so the resolution fraction is explicitly scoped to events returned by bounded pagination. Modeled turnover charges absolute target-weight changes, not price-drift rebalancing or terminal liquidation, and can therefore understate executable costs.

Evidence is written to `.tmp/research/historical-funding-campaign-v1/`; the immutable raw snapshot is stored separately under `.tmp/research/historical-funding-snapshot-v1/`, while ordinary campaign CSVs contain development rows only. Nested rolling-origin selection, cost/delay stresses, daily DSR and CSCV/PBO, a Bonferroni-simultaneous paired funding-versus-base test, and a conservative Bonferroni-adjusted probabilistic Sharpe gate account for all `15` earlier plus `6` new trials. The shared overlap-aware registry procedurally keeps the final `1,227` returns unevaluated by the runner. `--open-final-holdout` opens them only after every development gate passes and consumes no registry entry when blocked. Because the source is public history, this is a reproducible process control rather than access control; an operator could independently fetch the same dates. The fixed ex-post survivor universe also does not model delistings, so a passing result would not by itself establish performance for a contemporaneously investable universe.

Run the locked residual-reversal turnover campaign against those exact development inputs:

```sh
python3 scripts/research/run_historical_reversal_campaign.py
```

The registration at `research-notes/registrations/residual-reversal-turnover-v1.json` pins the prior campaign directory, snapshot directory, source manifest and registration hashes, and the exact development-panel and settlement CSV hashes. The runner does not reacquire or realign data. It evaluates six primary residual-reversal trials: `24h`, `72h`, and `168h` horizons crossed with one-bar and three-bar rebalancing at phase zero on a fixed absolute-time grid. Funding remains a realized signed cashflow but cannot filter or rank symbols. Between delayed activations, futures positions remain open, so their effective weights drift with prices and with equity after funding and costs; scheduled turnover is charged from the drifted pretrade weights to the new targets, and only terminal liquidation is uncharged. Only the three-bar trials may become champion; the matching one-bar trials are paired controls.

For each horizon, the paired estimand is the annualized Sharpe of the row-wise net-return spread `r_3bar - r_1bar`, not the difference between the two standalone Sharpes. Promotion requires the selected three-bar trial's three-hypothesis Bonferroni-simultaneous spread-Sharpe lower bound to exceed zero and its mean turnover across all development evaluation rows to be no more than `0.5` of its matched control. Phase-one and phase-two stresses rerun all three three-bar horizons with primary final and outer-fold selections frozen. For each phase, both the frozen outer-fold composite path and the frozen final champion on the same final-selection OOS rows must have block-bootstrap Sharpe confidence lower bounds above zero. The existing sample, activity, funding-resolution, outer-OOS confidence, fold, regime, doubled-cost, and extra-delay gates also remain mandatory. Evidence is written to `.tmp/research/historical-reversal-campaign-v1/`.

This is an adaptive follow-up, not an independent replication or a clean preregistration relative to development: the reversal direction, three-bar execution treatment, phase stresses, and turnover gate were chosen after inspecting the predecessor's complete development result. A prohibited pre-merge implementation smoke then accessed the registered development window before this protocol was committed. It exposed phase-zero portfolio-equity exhaustion for `resrev_24h_rebalance_3bar` over the interval whose left close is `1611907199999` and outcome close is `1611935999999`; phase-zero maximum turnover of about `208.51` and `234.69` for the completed `72h` and `168h` three-bar paths as equity approached zero; and maximum turnover of about `2.4` to `3.24` across the completed phase-one and phase-two three-bar paths. No final-holdout row was accessed. The pre-smoke strategy, trial set, phase choices, thresholds, selection rule, and holdout success rule remain unchanged and are locked before any final-holdout access. Post-smoke work is explicitly limited to conservative fail-closed bankruptcy handling, implementation conformance for the already stated frozen-final phase gate, and accounting corrections: exact state propagation through outer-fold switches, plus a cash-start holdout that uses full causal feature history and charges entry into the frozen target. None can rescue the observed bankrupt path.

Any primary phase-zero development path for which `1 + netReturn <= 0` makes the entire campaign `mechanically_invalid`. The runner emits a structured mechanical rejection without clipping the return, imposing an absorbing post-bankruptcy path, restarting the portfolio, substituting a trial, or tuning parameters. Bankruptcy-free execution of all six primary development paths is mandatory, and a mechanical rejection leaves the final holdout reserved and unopened even if `--open-final-holdout` is supplied. Equity exhaustion on a derived development path instead produces structured `insufficient_evidence` and also leaves the holdout unopened. Exhaustion after an authorized holdout reservation produces a structured failed result and completes the registry lifecycle, so the consumed window cannot be retried. If development did pass, holdout execution would start from cash at its first left close, use the frozen champion target computed from full causal history, charge that entry turnover, and then follow the absolute registered cadence. Nested validation cannot undo either the earlier research-process contamination or the disclosed smoke. All six primary and six horizon-by-phase configurations remain in current-campaign DSR and CSCV/PBO matrices and multiple-testing accounting, so the new-trial count remains `12` and the lifetime count remains `33` if the campaign reaches those analytics.

The fixed one-day bootstrap block may not capture dependence from 21-bar overlapping signals or persistent portfolio state, so its intervals can understate uncertainty. Endpoint solvency also does not establish exchange executability: maintenance margin, intrabar liquidation, leverage/equity buffers, market impact, and absolute turnover limits are not modeled. These are disclosures only; changing the locked block length or adding gates after the development smoke would require a new campaign.

Run the locked risk-controlled rank-hysteresis follow-up without opening its final holdout:

```sh
python3 scripts/research/run_historical_risk_controlled_reversal_campaign.py
```

The registration at `research-notes/registrations/residual-reversal-rank-hysteresis-risk-v1.json` pins the rejected reversal campaign's registration, implementation, manifest, failure, and summary evidence plus the complete upstream funding-data hash chain; it records result-note merge `226292304ba06cbc12212ebbc6e00f3dcdb2bb6f` as lineage metadata, not a data input. This is another adaptive, development-contaminated campaign, but it had no pre-merge development run. It fixes six trials: `24h`, `72h`, and `168h` residual reversal crossed with an exit-rank `1` matched control and exit-rank `3` hysteresis treatment; only treatments are selectable. Every-bar decisions activate one bar later, use stable rank-one entries, flatten both sides when either signed side is unavailable, and reset retained identities to `0.50` gross exposure split `+0.25/-0.25`. Charged turnover costs are `10` bps, the frozen stress uses `20` bps, and every completed path charges both cash entry and terminal liquidation.

All six paths must remain complete and pass fixed endpoint risk rules for equity, drawdown, gross and per-symbol exposure, activation and terminal turnover, and a simultaneous long-down/short-up `25%` diagnostic shock. Shocked equity must retain half of pre-shock equity and cover twice the combined `10%` maintenance plus `1%` liquidation reserve on shocked gross. These checks use close-only endpoints and do not simulate intrabar liquidation, fills, impact, or order-book liquidity. A breach stops as typed `risk_invalid` evidence with modeled immediate close-liquidation turnover and cost, never a forced-cash continuation; a successful path charges actual terminal liquidation and rechecks equity and drawdown. Promotion requires all `2,443` nested OOS rows, at least `50%` active observations, drawdown no greater than `20%`, at least `5/7` positive folds, worst-fold and per-regime losses no greater than `5%`, at least `100` observations per regime, DSR and lifetime-`39` Bonferroni PSR probabilities of at least `0.95`, PBO no greater than `0.20`, and champion mean/per-fold turnover ratios no greater than `0.70`/`0.85`. Nested, `20`-bps cost, one-extra-bar delay, and three-way Bonferroni matched-spread confidence lower bounds must all exceed zero at each circular moving-block length `21`, `42`, and `63` using `10,000` replications and seed `20260715`. Evidence is written to `.tmp/research/historical-risk-controlled-reversal-campaign-v1/`. The same `1,227` final returns stay sealed by registry version `3`; opening requires every development gate plus the explicit flag, reserves the overlapping window before any snapshot read, starts from cash with full causal history, and charges entry and terminal liquidation costs. This campaign's records bind both strict holdout identity and resolved output-directory digests. Only the three named predecessor schemas may omit `panelSha256`; their recorded campaign manifests must recover the panel digest and match campaign, registration, manifest, filename, window, and output provenance before they block overlap or repeat output, so unrelated valid legacy markers do not poison a run. The holdout passes only with no risk breach, positive total return, drawdown no greater than `20%`, and all three block confidence lower bounds above zero.

V1 stopped on its first required path when the `0.50`-gross `resrev_24h_exit1_control` reached `21.1756%` development drawdown against the fixed `20%` limit. The result is `risk_invalid`; the other five paths and statistical gates were not evaluated, and the final `1,227` returns remain unopened. A post-run audit found that v1 alphabetized close columns rather than consuming its registered nonalphabetical stable tie order. A development-only registered-order replay found no exact score ties or target differences through the stop and reproduced the same breach, so the immutable rejection stands, but the original execution is not relabeled bit-for-bit conforming.

Run the separately locked v2 adaptive screening campaign without opening its final holdout:

```sh
python3 scripts/research/run_historical_risk_controlled_reversal_campaign_v2.py
```

The v2 registration at `research-notes/registrations/residual-reversal-rank-hysteresis-risk-v2.json` pins the complete v1 rejection and upstream data lineage. It makes exactly one risk intervention: every base, derived, stress, and eventual holdout path uses fixed `0.25` gross exposure split `+0.125/-0.125`; there is no exposure sweep or volatility-target selection. All six horizons/treatments, the `20%` drawdown rule, other endpoint and shock constraints, selection, confidence, fold, regime, turnover, cost, delay, DSR, PBO, and promotion gates remain unchanged. Attempt accounting advances from `39` prior to `45` lifetime trials. Reusing the same `4,910` development bars is explicitly contaminated adaptive screening, not independent confirmation or evidence of optimal sizing; a failure stops rather than trying another gross setting on those rows.

V2 validates and applies the exact registered symbol order before residual construction. Its version-`3` registry resolves from Git's common directory, so linked worktrees in the same clone share the canonical `.tmp/research/edge-campaign-holdouts/`; malformed Git metadata, explicit noncanonical test injection, or unreconciled linked-worktree JSON markers fail closed, and `TRADER_EDGE_HOLDOUT_REGISTRY` is ignored by production imports. This is still process control within one clone, not data access control across machines. The first official v2 invocation must omit both `--open-final-holdout` and `--development-audit-sha256`; only a complete passing development result, separate audit, and later explicit one-shot invocation could authorize opening.

When every gate passes, that first invocation writes byte-identical `summary.json` and immutable `development-ready-summary.json`, then exclusively writes `development-ready-index.json` over the manifest, ready summary, risk ledger, and every development analysis CSV. The ready index binds the exact all-true gate list, frozen champion, registration, implementation, manifest, artifact paths, and artifact hashes. After separately reviewing that frozen evidence, the later invocation must supply the lowercase SHA-256 of the ready index as an explicit process attestation:

```sh
python3 scripts/research/run_historical_risk_controlled_reversal_campaign_v2.py \
  --open-final-holdout \
  --development-audit-sha256 <sha256-of-development-ready-index.json>
```

The opening invocation validates the receipt without rerunning or rewriting development analysis, rehashes its inputs and implementation, records the audit digest in the irreversible reservation, and only then may read the snapshot. Supplying a digest is process evidence, not proof of human cognition. Every current runner using this shared module defaults to the canonical registry. A separate-Git-directory checkout with linked worktrees fails closed because Git cannot safely identify its clone-wide primary checkout; independent clones and outdated or already-running implementations that still use worktree-local registries are outside the synchronization boundary and must not run during authorization or holdout evaluation.

The next protocol is prospective rather than another repair on the reused residual-reversal history. `research-notes/registrations/cross-sectional-funding-carry-v1.json` fixes one nonselectable daily funding-carry path before `2026-07-17T00:00:00Z`: long the most negative causally settled funding rate and short the most positive only when both sides receive carry, then activate one hour later at `0.25` gross. It uses no residual price signal, basis, premium, OI, taker flow, trend filter, threshold grid, exposure sweep, or rank hysteresis. Actual funding events, drifted positions, `10` bps turnover, cash entry, and terminal liquidation are mandatory; risk, `20` bps cost, and eight-hour-delay stresses are fixed.

This campaign must wait for exactly `4,500` contiguous hourly returns ending at `2027-01-20T13:00:00Z`. Earlier data, every `.tmp/research` predecessor input, and the still-sealed `1,227`-return window are prohibited. Before the cutoff, automation may inspect only acquisition metadata such as hashes, byte counts, boundaries, row counts, and missing-grid locations. It may not calculate returns, ranks, weights, PnL, risk, or performance. After the cutoff, acquisition integrity is frozen in a receipt; a separate digest-authorized invocation must reserve the canonical one-shot registry before parsing any performance value. The fixed path is lifetime attempt `46`, and every risk, fold, regime, dependence-conservative confidence, doubled-cost, delay, DSR, and lifetime-corrected gate is conjunctive.

The public OI, basis, and taker endpoints retain only about 30 days. The collector retrieves a safely bounded version of that retained window in fixed, last-closed-bar chunks and merges it into the append-only cache, but it cannot backfill days that have already expired. Run one locked refresh of the fixed ten-symbol, 1-hour campaign cache with:

```bash
python3 scripts/research/collect_datafeed.py
```

On macOS, install the repo-native hourly LaunchAgent after exporting any optional public values documented in `.env.example`:

```bash
scripts/install-research-datafeed-launchagent.sh install
scripts/install-research-datafeed-launchagent.sh status
# To stop collection without deleting the cache:
scripts/install-research-datafeed-launchagent.sh uninstall
```

The job runs at minute `10` of each hour, does not load exchange credentials, never uses `KeepAlive`, and records bounded per-symbol status under `data/research/.collector/last-run.json` by default. `TRADER_RESEARCH_CACHE` may point to a persistent absolute path, `TRADER_RESEARCH_SYMBOLS` overrides the fixed universe, `TRADER_RESEARCH_COLLECT_MINUTE` changes the minute after the hour, and `TRADER_RESEARCH_MAX_RUN_SECONDS` sets the wall-clock deadline (default `3000`; the installer accepts `60` through `3500`). The atomic plist is the installed configuration source, so later `status`, `restart`, and `uninstall` commands do not require those overrides or another Python executable on `PATH`.

Every cache-mutating entry point shares one writer lock, and panel reads hold it across the complete cross-symbol snapshot; scheduled overlap exits cleanly while direct and campaign work waits instead of racing a read/merge/replace transaction. CSV, status, and plist files use same-filesystem atomic replacement, so interruption cannot expose a partial file. A symbol is `ok` only when the refresh fetched a current closed kline and every derivatives series returned recent finite observations. Empty/stale klines and empty, stale, or unavailable funding, OI, basis, or taker evidence make the run a nonzero `partial_failure` without starving later symbols. New point-in-time alignment stops carrying an observation past its freshness bound, while merge precedence retains newer non-null evidence already accumulated from an earlier successful response.

Each derivatives series refresh is isolated on an API/page error. Missing intervals in otherwise successful pages remain null. In both cases the merge preserves previously cached point-in-time values instead of forward-filling incomplete fresh evidence over them. SIGINT/SIGTERM records `interrupted`; the deadline records `timeout` and exits `124`; a manual `restart` starts a new bounded run. A loaded service must stop successfully before uninstall removes its plist. The local LaunchAgent is an interim accumulator because laptop sleep/offline periods still lose collection opportunities; archival ingestion or an always-on persistent collector remains necessary before sparse-history, PBO, OOS-count, and confidence gates can pass.

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

## Top-combo portfolio selection

Optimizer exports include timestamped daily returns derived from the net out-of-sample equity curve. The selector defaults to at least 180 aligned days plus the existing freshness, venue-cost, 20-trade, walk-forward Sharpe, stability, position-size, and quarantine gates; legacy combos without the new evidence remain visible but cannot receive portfolio capital. The managed five-worker graduation explicitly uses a 30-day selector floor, while retaining every other strict admission gate and adding an independent 30-complete-day live review.

Leaderboard `processing.tier=deployable` uses the same fail-closed evidence contract: `params.minEdge >= 0.0018`, at least 20 trades, walk-forward mean Sharpe at least `0.3`, and walk-forward Sharpe standard deviation at most `1.5`. Missing or failing readings keep the row at `candidate`/`raw` and are listed in `processing.reasons`. `TRADER_TOP_COMBO_DEPLOYABLE_OVERRIDE_UUIDS` is a bounded operator exception: only named candidate UUIDs are promoted, `raw` and `quarantined` rows remain blocked, and the bypassed strict gates remain visible under `processing.relaxedReasons` with `processing.relaxed=true`.

Scheduled and startup re-backtests backfill the same timestamped net daily-return evidence onto legacy combos. A refresh therefore makes an older combo assessable by the portfolio selector, but never bypasses the existing activity, walk-forward, cost, freshness, position-size, or drawdown gates.

Scheduled refreshes publish the five highest-ranked combos first and then persist the remainder in bounded 100-combo batches. This makes newly completed evidence durable during long leaderboard sweeps instead of waiting for every stale combo to finish.

The PostgreSQL leaderboard replica stores portfolio evidence inside `combos.metrics_json` and binds batch combo identifiers as `uuid[]`. Database fallback and cross-replica reconciliation therefore retain the same OOS return series as the JSON/S3 leaderboard instead of losing canary-selection evidence or failing on a `uuid = text` comparison.

Production research nodes own scheduled evidence backfills; trading nodes consume the synced leaderboard without duplicating that CPU-heavy work. A cached selector failure remains reusable for up to one hour only while the leaderboard evidence snapshot is unchanged, so newly synced evidence is evaluated on the next bot poll.

Production roles launch bot auto-start before the combo replica worker, then give replica synchronization a 30-second startup grace period. Bot auto-start can therefore assess the atomically persisted local leaderboard before the CPU-heavy 5,000-combo replica merge begins; recurring synchronization still runs every 60 seconds afterward.

Once per week, the server evaluates up to three unique-symbol bots jointly. It searches 5% weight increments, caps each bot at 25% and total deployed capital at 75%, and leaves unused capital in cash. A deterministic seven-day moving-block bootstrap maximizes the 10th-percentile annualized net return while requiring the 95th-percentile maximum drawdown to remain at or below 10%. Rotation additionally requires two percentage points of conservative annualized improvement and at least 90% paired-bootstrap outperformance probability.

The decision is atomically persisted beside `top-combos.json` as `portfolio-selection.json` and expires after eight days. Open/orphaned positions are always restored first for safe management: live startup scans exchange inventory before loading or rebuilding portfolio selection, skips selection and rotation while any position still needs adoption or its worker is initializing, and lets every adoption bypass ordinary portfolio, per-cycle start, stale-backoff, and new-entry circuit-breaker caps. In `canary` or `enforce`, an absent, expired, or invalid decision blocks new portfolio entries instead of falling back to independent combo ranking. Canary scales aggregate portfolio capital to 25%; enforce uses the selected weights up to the normal 75% ceiling. The default `shadow` mode reports the challenger through `/bot/status` and the Live Bot UI without changing the existing fleet.

The checked-in Hetzner trading profile is the sole live executor. Its current bounded relaxation pins five reviewed UUIDs for AVAX, UNI, SUI, ETC, and ADA, starts at most two workers per polling cycle, disables their online combo replacement, and initially runs independent `shadow` selection. The runtime rechecks each allowlisted row and refuses raw, quarantined, or unknown processing tiers even if an earlier annotation or operator list is stale. Each adopted combo is capped at 5%, and any larger inherited minimum-size floor is reduced to that cap, so the five-worker configuration can enter but cannot exceed 25% aggregate configured exposure; existing-position recovery still takes priority and bypasses ordinary fleet caps. Starting with the first full UTC day after `2026-08-21T14:20:00Z`, automatic graduation requires a valid boundary equity baseline for every reviewed UUID, 30 complete days present for every reviewed UUID, positive aggregate window-relative fleet return, maximum drawdown at or below 10%, at least 10 order attempts with 95% successful submission reliability, at least 99% healthy running status samples, and a latest status that is running, unhalted, and error-free for every UUID. Each worker is rebased to its boundary equity, so gains or losses before the review window cannot affect graduation. It then also requires the strict 30-day portfolio selector to produce a valid enforcement selection. The review runs hourly, persists pending evidence beside the leaderboard, and writes a durable `portfolio.graduated` decision before effective mode changes to `enforce`; database, evidence, selector, or persistence failure leaves shadow mode and the UUID target allowlist active. The 5%/25% exposure caps and normal risk controls are not removed by graduation. The override does not change optimizer discovery thresholds or promote any unnamed candidate. Fly omits `--binance-live` and sets `TRADER_BOT_TRADE=false`, making it a read-only standby even when CI recreates its app machine. Research profiles request 1,100-day lookbacks on 6h-or-longer intervals and reserve 64 of the bars actually fetched before deriving a safe capped window; exchange pagination and shorter symbol histories therefore cannot land just below the train/tune/backtest split requirement while enough OOS coverage remains for the default 180-day evidence floor. Long-horizon searches run as recurring batches of four broad primary trials and, when needed, six activity-focused recovery trials, so the two passes explore distinct neighborhoods and completed evidence is merged incrementally. The recovery edge band remains cost-safe at `0.0018..0.0024`. The managed Hetzner research profile optimizes Sharpe and forwards that objective to both outer candidate scoring and nested threshold tuning. Exposure metrics use the position held during each return interval, while the position series remains the end-of-bar state; same-bar exits therefore retain their real exposure without appearing open after the bar. Hetzner research caps each optimizer child at an 8 GiB Haskell heap with `TRADER_OPTIMIZER_TRIAL_HEAP_CAP=8g`, preserving headroom for the API and PostgreSQL while recording heap-overflow trials as failures. Hetzner research favors faster TA methods and refreshes only 20 stale incumbents daily after seven days, leaving most CPU for discovery. Optimizer telemetry distinguishes `optimizer.auto.admitted` from `optimizer.auto.no_admission`, includes board counts and newest timestamps, and reports database persistence failures instead of treating every successful merge process as a new combo.

Valid `TRADER_TOP_COMBO_DEPLOYABLE_OVERRIDE_UUIDS` values are canonicalized before lookup and deduplication, so equivalent uppercase and lowercase UUID spellings select the same reviewed candidate.

Hetzner releases rebuild `/ops/performance` transactionally after health and commit attestation (`TRADER_OPS_ROLLUP_ON_DEPLOY=true`). Futures entries remain maker-first with the existing 2 bps / 3 second defaults; `TRADER_EXECUTION_MAKER_*` exposes those settings, and persisted `bot.order` results report `executionPath` as `maker-filled`, `maker-partial`, `market-fallback`, or `maker-skipped` for canary review before tuning.

Every live Binance futures entry now passes a final, fail-closed market-risk boundary. The boundary requires a positive directional forecast edge—remeasured from the live order-book midpoint—at or above the configured minimum, walks the book for the actual requested quantity, enforces spread and expected-impact ceilings, charges side-adverse funding against the remaining edge budget, rejects extreme mark/index basis, and blocks high, stale, or unavailable symbol-level ADL evidence. Critical feeds are collected concurrently and checked against Binance-adjusted time with a bounded clock-skew allowance; shadow-only requests cannot delay the admission decision. It never blocks position reductions or protective orders: a reversal first sends and confirms an unconditional reduce-only close, calculates balance-fraction sizing after the account is flat, then subjects only the new opposite-side quantity to entry admission. Existing bot protection orders are removed only after the venue confirms that the closed side is flat; transient position-query failures retain that protection. The order result records the reduce-only effect and both reversal legs separately so bot state remains flat when the entry is rejected and reflects partial entry fills exactly. Current open interest and its interval change, taker buy/sell ratio, historical basis, and book imbalance are persisted as structured `marketRisk` order evidence but remain shadow-only until out-of-sample validation justifies using them as alpha. Missing, invalid, and stale shadow readings are reported without blocking; `TRADER_MARKET_RISK_MAX_SHADOW_AGE_SEC` controls their freshness horizon. The remaining `TRADER_MARKET_RISK_*` settings configure critical freshness and cost ceilings; setting `TRADER_MARKET_RISK_FAIL_CLOSED=false` is an explicit unsafe availability tradeoff. Binance klines also retain quote volume, trade count, and taker-buy base/quote volume instead of discarding those exchange fields.

```dotenv
TRADER_PORTFOLIO_SELECTOR_ROLLOUT_MODE=shadow
TRADER_PORTFOLIO_SELECTOR_MAX_BOTS=5
TRADER_PORTFOLIO_SELECTOR_MAX_BOT_WEIGHT=0.05
TRADER_PORTFOLIO_SELECTOR_MAX_GROSS_WEIGHT=0.25
TRADER_BOT_ONLINE_OPTIMIZER_ENABLED=false
TRADER_PORTFOLIO_SELECTOR_MAX_DRAWDOWN=0.10
TRADER_PORTFOLIO_SELECTOR_MIN_DAYS=30
TRADER_PORTFOLIO_SELECTOR_BOOTSTRAP_SAMPLES=1000
TRADER_PORTFOLIO_SELECTOR_BLOCK_DAYS=7
TRADER_PORTFOLIO_SELECTOR_ROTATION_IMPROVEMENT=0.02
TRADER_PORTFOLIO_SELECTOR_ROTATION_PROBABILITY=0.90
TRADER_PORTFOLIO_AUTO_GRADUATE_ENABLED=true
TRADER_PORTFOLIO_AUTO_GRADUATE_STARTED_AT_MS=1787322000000
TRADER_PORTFOLIO_AUTO_GRADUATE_MIN_DAILY_OBSERVATIONS=30
TRADER_PORTFOLIO_AUTO_GRADUATE_MIN_NET_RETURN=0
TRADER_PORTFOLIO_AUTO_GRADUATE_MAX_DRAWDOWN=0.10
TRADER_PORTFOLIO_AUTO_GRADUATE_MIN_EXECUTION_ATTEMPTS=10
TRADER_PORTFOLIO_AUTO_GRADUATE_MIN_EXECUTION_RELIABILITY=0.95
TRADER_PORTFOLIO_AUTO_GRADUATE_MIN_STATUS_RELIABILITY=0.99
TRADER_PORTFOLIO_AUTO_GRADUATE_REVIEW_EVERY_SEC=3600
```

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

For Hetzner CI, both trading and research boxes are mandatory. The workflow requires the SSH key, both role hosts, and pinned `known_hosts` entries, and deploys only the latest green commit. Each release explicitly builds and recreates the API, retains the previous image, waits for container health, and verifies the exact `/health` commit. It validates the incoming Caddyfile and recreates Caddy after the API is healthy so an `rsync`-replaced bind mount cannot leave stale proxy routes active. The SSH-streamed health probe detaches stdin so it cannot consume the remaining remote deployment program. The transactional performance rollup retries up to three times when it collides with freshly started platform/symbol registration; persistent failure still rolls back the release. CI rejects a deploy that exits without the remote attestation and rolls back on failed health or commit attestation.

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
