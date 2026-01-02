Haskell Trading Bot (Kalman + LSTM + Binance/Coinbase/Kraken/Poloniex)
=============================================

This repository contains a small Haskell trading demo that:
- Predicts the next price using a small **LSTM**, and a **multi-sensor Kalman fusion** layer that combines multiple model outputs into a single latent expected return signal.
- By default, only trades when Kalman and LSTM **agree on direction** (both predict up, or both predict down) — configurable via `--method` (including `router` to auto-select models).
- Can backtest on CSV data or pull klines from **Binance**, **Coinbase**, **Kraken**, or **Poloniex** (trading supports Binance + Coinbase spot).

Features
--------
- Multi-sensor Kalman fusion filter for latent expected return (`haskell/app/Trader/KalmanFusion.hs`).
- Multiple predictive methods feeding into Kalman as an observation vector (`haskell/app/Trader/Predictors.hs`):
  - Gradient-boosted trees (LightGBM/CatBoost style, simplified)
  - TCN / dilated 1D CNN (simplified)
  - Transformer-style attention predictor (kNN attention)
  - HMM / regime model (3 regimes)
  - Quantile regression (q10/q50/q90)
- Conformal interval wrapper (calibrated on a holdout split, sigma derived from alpha; omitted when calibration is empty)
- Predictor training validates fixed feature dimensions to avoid silent mismatches.
- Predictor outputs omit transformer/GBDT/quantile/conformal when the feature dataset is empty or features do not match the trained dimensions.
- Quantile outputs clamp the reported median inside the q10/q90 bounds and omit sigma when the interval is invalid; the sensor mean uses the raw median.
- Predictor training uses a train/calibration split so held-out calibration data is excluded from model training.
- LSTM next-step predictor with Adam, gradient clipping, and early stopping (`haskell/app/Trader/LSTM.hs`).
- Agreement-gated ensemble strategy (`haskell/app/Trader/Trading.hs`).
- Optional tri-layer entry gating: Kalman cloud trend + price-action reversal triggers (`haskell/app/Trader/Trading.hs`).
- Profitability, risk/volatility, trade execution, and efficiency metrics (incl. Sharpe, max drawdown) (`haskell/app/Trader/Metrics.hs`).
- Data sources: CSV or exchange klines (Binance/Coinbase/Kraken/Poloniex).
- Sample dataset in `data/sample_prices.csv`.

Quick start
-----------
Build and run with Cabal:
```
cd haskell
cabal run trader-hs -- --data ../data/sample_prices.csv --price-column close
```

Print the CLI version:
```
cd haskell
cabal run trader-hs -- --version
```

Benchmarks (dev)
----------------
Run the predictor microbench harness:
```
cd haskell
cabal run bench:trader-bench -- --samples 5000 --features 16 --trees 50
```

Example backtest with tighter model settings:
```
cd haskell
cabal run trader-hs -- \
  --data ../data/sample_prices.csv \
  --normalization standard \
  --interval 1h \
  --lookback-window 7d \
  --hidden-size 8 \
  --epochs 10 \
  --open-threshold 0.002 \
  --close-threshold 0.002 \
  --fee 0.0008
```

Using exchange klines
---------------------
Fetch klines from an exchange instead of a CSV (default platform is Binance):
```
cd haskell
cabal run trader-hs -- \
  --binance-symbol BTCUSDT \
  --interval 1h \
  --epochs 5
```

Coinbase example (products use BASE-QUOTE, e.g. BTC-USD):
```
cd haskell
cabal run trader-hs -- \
  --symbol BTC-USD \
  --platform coinbase \
  --interval 1h \
  --epochs 5
```

Kraken example:
```
cd haskell
cabal run trader-hs -- \
  --symbol XBTUSD \
  --platform kraken \
  --interval 1h \
  --epochs 5
```

Poloniex example (symbols use BASE_QUOTE, e.g. BTC_USDT):
```
cd haskell
cabal run trader-hs -- \
  --symbol BTC_USDT \
  --platform poloniex \
  --interval 2h \
  --epochs 5
```

Sending exchange orders (optional)
----------------------------------
Binance: live orders are the default. Use `--no-binance-live` to send test orders (`/api/v3/order/test` or `/fapi/v1/order/test`). Futures use `--futures` (uses `/fapi` endpoints). Margin uses `--margin` (requires live orders).
Coinbase: spot-only and live-only (no test endpoint). Use `--platform coinbase`.

Futures protection orders (live, manual trades only):
- When sending **LIVE futures** orders via the CLI (`--binance-trade`) or REST `/trade`, providing `--stop-loss` and/or `--take-profit` places exchange-native trigger orders (`STOP_MARKET` / `TAKE_PROFIT_MARKET`) with `closePosition=true`.
- The continuous `/bot` loop does not place exchange-native protection orders to avoid bot/exchange state desync.

Environment variables:
- `BINANCE_API_KEY`
- `BINANCE_API_SECRET`
- `COINBASE_API_KEY`
- `COINBASE_API_SECRET`
- `COINBASE_API_PASSPHRASE`

Getting Binance API keys:
- Binance → Profile → **API Management** → **Create API**
- Enable only what you need (Spot/Margin/Futures trading) and keep withdrawals disabled
- Prefer IP restrictions (allowlist your server IP) when possible
- Save the secret: Binance only shows it once

Example (test endpoint with `--no-binance-live`):
```
cd haskell
export BINANCE_API_KEY=...
export BINANCE_API_SECRET=...
cabal run trader-hs -- \
  --binance-symbol BTCUSDT \
  --interval 1h \
  --epochs 5 \
  --trade-only \
  --binance-trade \
  --no-binance-live \
  --order-quote 50
```

CLI parameters
--------------
You must provide exactly one data source: `--data` (CSV) or `--symbol`/`--binance-symbol` (exchange; default platform is Binance).

- Data source
  - `--data PATH` (default: none) CSV file containing prices
  - `--price-column close` CSV column name for price
  - `--high-column high` CSV column name for high (requires `--low-column`; enables intrabar stop-loss/take-profit/trailing-stop realism)
  - `--low-column low` CSV column name for low (requires `--high-column`)
  - `--symbol SYMBOL` (alias `--binance-symbol`) exchange symbol to fetch klines
  - `--platform binance` exchange platform for `--symbol` (`binance|coinbase|kraken|poloniex`)
    - Coinbase products use `BASE-QUOTE` (for example `BTC-USD`).
    - Poloniex symbols use `BASE_QUOTE` (for example `BTC_USDT`); legacy `USDT_BTC` is auto-swapped.

- Bars & lookback (defaults: `--interval 1h`, `--lookback-window 7d` → 168 bars, `--bars auto`)
- `--interval 1h` (alias `--binance-interval`) bar interval / exchange kline interval
- `--bars auto` (alias `--binance-limit`) number of bars/klines to use (`auto` = all CSV, or 500 for exchanges; CSV also supports `0` = all; Binance 2..1000)
  - `--lookback-window 7d` lookback window duration (converted to bars)
  - `--lookback-bars N` (alias `--lookback`) override the computed lookback bars
  - Lookback must be less than the total number of bars, otherwise the backtest errors.

- Trading (Binance + Coinbase spot)
  - Trading flags apply only when `--platform binance` or `--platform coinbase` (Coinbase is spot-only and has no test endpoint).
  - `--binance-testnet` (default: off) use Binance testnet base URL (Binance only)
  - `--futures` (default: off) use Binance USDT-M futures endpoints (data + orders; Binance only)
  - `--margin` (default: off) use Binance margin account endpoints for orders/balance (requires live orders; Binance only)
  - `--binance-api-key KEY` (default: none) or env `BINANCE_API_KEY`
  - `--binance-api-secret SECRET` (default: none) or env `BINANCE_API_SECRET`
  - `--binance-trade` (default: off) place a market order for the latest signal
  - `--binance-live` (default: on) send LIVE orders
  - `--no-binance-live` send TEST orders (Binance only; Coinbase has no test endpoint)
  - `--order-quote AMOUNT` (default: none) quote amount to spend on BUY (`quoteOrderQty`)
  - `--order-quantity QTY` (default: none) base quantity to trade (`quantity`)
  - `--order-quote-fraction F` (default: none) size BUY orders as a fraction of the free quote balance (`0 < F <= 1`)
  - `--max-order-quote Q` (default: none) cap the computed quote amount when using `--order-quote-fraction`
  - `--idempotency-key ID` (default: none) optional Binance `newClientOrderId` for idempotent orders
  - Sizing inputs are mutually exclusive: choose one of `--order-quantity`, `--order-quote`, or `--order-quote-fraction`.
  - Binance futures orders pre-check available balance (and leverage) and skip entries that exceed available margin.

- Coinbase API keys (optional; used for `/coinbase/keys` checks and Coinbase trades)
  - `--coinbase-api-key KEY` (default: none) or env `COINBASE_API_KEY`
  - `--coinbase-api-secret SECRET` (default: none) or env `COINBASE_API_SECRET`
  - `--coinbase-api-passphrase PASSPHRASE` (default: none) or env `COINBASE_API_PASSPHRASE`

- Normalization
  - `--normalization standard` one of `none|minmax|standard|log`
  - If the fit window is empty or only has non-finite values, `minmax`/`standard` fall back to no-op normalization.
  - `log` normalization falls back to no-op when the fit window is empty or contains non-finite/non-positive values.

- LSTM
  - Lookback bars come from `--lookback-window`/`--lookback-bars`
  - `--hidden-size 16` hidden size
  - `--epochs 30` training epochs (Adam)
  - `--lr 1e-3` learning rate
  - `--val-ratio 0.3` validation split ratio (within training set)
  - `--patience 10` early stopping patience (`0` disables)
  - `--grad-clip N` (default: none) gradient clipping max L2 norm
  - `--seed 42` random seed for init

- Kalman fusion (latent expected return)
  - `--kalman-dt 1.0` scales process noise per step
  - `--kalman-process-var 1e-5` process noise variance
  - `--kalman-measurement-var 1e-3` fallback measurement variance (and initial variance)
  - `--kalman-market-top-n 50` optional market context measurement; skipped if fewer than `min(N, 5)` symbols are available
  - `--predictors gbdt,tcn,transformer,hmm,quantile,conformal` comma-separated predictors to train/use (`all`/`none` accepted; default is all; `all` and `none` cannot be combined)
    - Conformal intervals use the GBDT model internally, even if `gbdt` isn't selected as a sensor.
    - HMM/quantile/conformal confirmations only apply when their predictors are enabled.
    - If a predictor output is missing for a bar, its confirmation/width gates are skipped (treated as pass).

- Strategy / costs
  - `--open-threshold 0.002` (or legacy `--threshold`) entry/open direction threshold (fractional deadband)
  - `--close-threshold 0.002` exit/close threshold (fractional deadband; defaults to open-threshold when omitted)
    - Live order placement uses `close-threshold` to decide exits when already in position, mirroring backtest logic.
  - `--min-edge F` minimum predicted return magnitude required to enter (`0` disables)
  - `--min-signal-to-noise F` require edge / per-bar sigma >= `F` (`0` disables; default: `0.8`)
    - `--cost-aware-edge` raises min-edge to cover estimated fees/slippage/spread (default on; disable with `--no-cost-aware-edge`)
    - `--edge-buffer 0.0002` optional extra buffer added on top of cost-aware edge
  - `--threshold-factor` enable dynamic threshold multipliers for open/close thresholds and min-edge/min-signal-to-noise (default off; disable with `--no-threshold-factor`)
    - `--threshold-factor-alpha 0.2` EMA update rate; `--threshold-factor-min/max 0.5/2.0` bounds; `--threshold-factor-floor 0` floor on adjusted thresholds
    - Weights: `--threshold-factor-edge-kal-weight`, `--threshold-factor-edge-lstm-weight`, `--threshold-factor-kalman-z-weight`, `--threshold-factor-high-vol-weight`, `--threshold-factor-conformal-weight`, `--threshold-factor-quantile-weight`, `--threshold-factor-lstm-conf-weight`, `--threshold-factor-lstm-health-weight`
  - `--method 11` choose `11`/`both` (Kalman+LSTM direction-agreement), `10`/`kalman` (Kalman only), `01`/`lstm` (LSTM only), `blend` (weighted average), `router` (adaptive model selection)
    - When using `--method 10`, the LSTM is disabled (not trained).
    - When using `--method 01`, the Kalman/predictors are disabled (not trained).
    - When using `--method router`, the bot picks Kalman/LSTM/blend per bar based on recent directional accuracy; Kalman confidence/risk gates apply only when Kalman is selected. Router scoring uses the effective open threshold (open-threshold plus any cost-aware min-edge floor).
    - `--blend-weight 0.5` Kalman weight for `blend` (`0..1`, default: `0.5`)
    - `--router-lookback 30` lookback bars for router scoring (`>= 2`)
    - `--router-min-score 0.25` minimum router score (accuracy × coverage) to accept a model (`0..1`)
- `--positioning long-flat` (default, alias `long-only`/`long`) or `--positioning long-short` (allows short positions; trading/live bot requires `--futures`)
  - `--optimize-operations` optimize `--method`, `--open-threshold`, and `--close-threshold` on the tune split (uses best combo for the latest signal; router is excluded)
  - `--sweep-threshold` sweep open/close thresholds on the tune split and pick the best by final equity
  - Sweeps/optimization validate prediction lengths and return errors if inputs are too short.
  - Threshold sweeps sample slightly below observed edges to avoid equality edge cases.
  - `--tune-objective equity-dd-turnover` objective used by `--optimize-operations` / `--sweep-threshold`:
    - `annualized-equity` | `final-equity` | `sharpe` | `calmar` | `equity-dd` | `equity-dd-turnover`
    - When `--threshold-factor` is enabled, the tune objective is forced to `annualized-equity`.
    - To maximize annualized equity, set `--tune-objective annualized-equity` (alias: `annualized-return`).
  - When sweep/optimization scores tie, the selector prefers higher final equity, then lower turnover, more round trips (excludes end-of-series EOD exits), and non-inverted hysteresis (close <= open) without reducing equity.
  - `--tune-penalty-max-drawdown 1.5` penalty weight for max drawdown (used by `equity-dd*` objectives)
  - `--tune-penalty-turnover 0.2` penalty weight for turnover (used by `equity-dd-turnover`)
  - `--min-round-trips N` (default: `0`) when optimizing/sweeping, require at least `N` non-EOD round trips in the tune split (helps avoid picking "no-trade" configs)
  - `--tune-stress-vol-mult F` volatility multiplier for stress scoring (`1` disables)
  - `--tune-stress-shock F` shock added to returns for stress scoring (`0` disables)
  - `--tune-stress-weight F` penalty weight for stress scoring (`0` disables)
  - `--walk-forward-folds 7` number of folds used to score the tune split and report backtest variability (`1` disables)
  - `--trade-only` skip backtest/metrics and only compute the latest signal (and optionally place an order)
  - `--fee 0.0008` fee applied when switching position
  - The CLI also prints an estimated **round-trip cost** (fee + slippage + spread) and warns when thresholds are below it.
  - `--stop-loss F` optional synthetic stop loss (`0 < F < 1`, e.g. `0.02` for 2%)
  - `--take-profit F` optional synthetic take profit (`0 < F < 1`)
  - `--trailing-stop F` optional synthetic trailing stop (`0 < F < 1`)
  - `--stop-loss-vol-mult F` optional: stop loss as per-bar sigma multiple (`0` disables; overrides `--stop-loss` when vol estimate is available)
  - `--take-profit-vol-mult F` optional: take profit as per-bar sigma multiple (`0` disables; overrides `--take-profit` when vol estimate is available)
  - `--trailing-stop-vol-mult F` optional: trailing stop as per-bar sigma multiple (`0` disables; overrides `--trailing-stop` when vol estimate is available)
  - Live-bot bracket exits honor the vol-mult overrides when exchange-native protection orders are not in use.
  - `--min-hold-bars N` minimum holding periods before allowing a signal-based exit (`0` disables; default: `4`; bracket exits still apply)
  - `--cooldown-bars N` after an exit to flat, wait `N` bars before allowing a new entry (`0` disables; default: `2`)
  - `--max-hold-bars N` force exit after holding for `N` bars (`0` disables; default: `36`; exit reason `MAX_HOLD`, then wait 1 bar before re-entry)
  - `--lstm-exit-flip-bars N` exit after `N` consecutive LSTM bars flip against the position (`0` disables; LSTM methods only)
  - `--lstm-exit-flip-grace-bars N` ignore LSTM flip exits during the first `N` bars of a trade (LSTM methods only)
  - `--lstm-exit-flip-strong` require strong LSTM confidence for flip exits (uses `--lstm-confidence-hard`; LSTM methods only)
  - `--trend-lookback N` simple moving average filter for entries (`0` disables; default: `30`)
  - `--tri-layer` enable Kalman cloud + price-action entry gating (uses the last candle and Kalman cloud trend)
    - `--tri-layer-fast-mult 0.5` fast Kalman measurement-variance multiplier for the cloud
    - `--tri-layer-slow-mult 2.0` slow Kalman measurement-variance multiplier for the cloud
    - `--tri-layer-cloud-padding 0.0` expand the cloud by this fraction of price when checking touches (`0` = strict)
    - `--tri-layer-cloud-slope 0.0` require cloud slope to exceed this fraction of price (`0` = sign-only)
    - `--tri-layer-cloud-width 0.0` block entries when the cloud width exceeds this fraction of price (`0` disables)
    - `--tri-layer-touch-lookback 1` allow cloud touches up to `N` bars back (`1` = current bar)
    - `--no-tri-layer-price-action` disable the candle-pattern trigger (cloud-only gating)
    - `--tri-layer-price-action-body 0.0` override min candle body fraction for price-action patterns (`0` = default)
    - `--tri-layer-exit-on-slow` exit when price crosses and closes on the opposite side of the slow Kalman line
    - `--kalman-band-lookback 0` rolling window (bars) for Kalman-band exits (`0` disables; must be >= 2; hits use candle high/low)
    - `--kalman-band-std-mult 0` band width in std devs for Kalman-band exits (`0` disables; `2` = PDF default)
    - Kalman-band exits do not require `--tri-layer`; enable them with the band flags and a lookback >= 2.
  - When high/low data is available (CSV `--high-column`/`--low-column` or exchange candles), tri-layer cloud touches and price-action checks use those highs/lows for latest signals/live bots.
  - `--max-position-size 0.8` cap position size/leverage (`1` = full size)
  - `--vol-target F` target annualized volatility for position sizing (`0` disables; default: `0.7`)
    - `--vol-lookback N` realized-vol lookback window (bars) when EWMA alpha is not set
    - `--vol-ewma-alpha F` use EWMA volatility estimate (overrides lookback)
    - `--vol-floor F` annualized vol floor for sizing (default: `0.15`)
    - `--vol-scale-max F` cap volatility scaling (limits leverage)
    - `--max-volatility F` block entries when annualized vol exceeds this (`0` disables; default: `1.5`)
  - `--rebalance-bars N` optional: resize open positions every `N` bars toward the target size (`0` disables rebalancing; backtests only; default: `24`, entry-anchored)
  - `--rebalance-threshold F` optional: minimum absolute size delta required to rebalance (`0` disables rebalancing; default: `0.05`)
  - `--rebalance-global` optional: anchor rebalance cadence to global bars instead of entry age
  - `--rebalance-reset-on-signal` optional: reset rebalance cadence when a same-side open signal updates size
  - `--funding-rate F` optional: annualized funding/borrow rate applied per bar in backtests (`0` disables; negative allowed; default: `0.1`)
  - `--funding-by-side` optional: apply funding sign by side (long pays positive, short receives)
    - Without `--funding-by-side`, the funding rate is applied uniformly (negative values credit both sides).
  - `--funding-on-open` optional: charge funding for bars opened with a position (even if exited intrabar)
  - Entries and latest-signal actions that use `--min-signal-to-noise`, `--max-volatility`, or `--vol-target` wait for a volatility estimate before entering.
  - `--kalman-z-min 0.5` minimum Kalman |mean|/std required to treat Kalman as directional (`0` disables)
  - `--kalman-z-max 3` Z-score mapped to full position size when confidence sizing is enabled
  - `--confirm-conformal` require conformal interval to agree with the chosen direction (default on; disable with `--no-confirm-conformal`)
  - `--confirm-quantiles` require quantiles to agree with the chosen direction (default on; disable with `--no-confirm-quantiles`)
  - `--confidence-sizing` scale entries by confidence (default on; disable with `--no-confidence-sizing`)
  - `--lstm-confidence-soft 0.6` soft LSTM confidence threshold for sizing (`0` disables the half-size step; requires confidence sizing)
  - `--lstm-confidence-hard 0.8` hard LSTM confidence threshold for sizing (`0` disables; requires confidence sizing)
  - `--min-position-size 0.15` minimum entry size after sizing/vol scaling (`0..1`; entries below this are skipped)
    - Must be <= `--max-position-size`.
  - When confidence sizing is enabled, live orders also scale entry size by LSTM confidence (score = clamp01(|lstmNext/current - 1| / (2 * openThreshold))): use `--lstm-confidence-hard/soft` thresholds (defaults 80%/60%).
  - The UI defaults to `orderQuote=100` so new setups clear common minQty/step sizes; adjust sizing to your account.
  - The UI auto-adjusts `bars` and `backtestRatio` on backtest/optimize requests when the split would be invalid (insufficient train/backtest/tune bars).
  - The UI error panel offers an Apply fix button for split errors that adjusts tune ratio to restore valid fit/tune windows.
  - Close-direction gating ignores `--min-position-size` so exits are not blocked by size floors.
  - Conformal/quantile confirmations apply the open threshold for entries and the close threshold for exits.
  - `--max-drawdown F` optional live-bot kill switch: halt if peak-to-trough drawdown exceeds `F`
  - `--max-daily-loss F` optional live-bot kill switch: halt if daily loss exceeds `F` (UTC day; resets each day)
    - Backtests use bar timestamps when available (exchange data or CSV time columns); otherwise they fall back to interval-based day keys.
    - If neither timestamps nor interval seconds are available, `--max-daily-loss` errors instead of silently disabling.
  - `--max-order-errors N` optional live-bot kill switch: halt after `N` consecutive order failures
  - Risk halts are evaluated on post-bar equity and can close open positions at the bar close.
  - Risk halts that occur while holding a position record `MAX_DRAWDOWN`/`MAX_DAILY_LOSS` as the exit reason.

- Metrics
  - `--backtest-ratio 0.2` holdout ratio (last portion of series; avoids lookahead)
    - The split must leave at least `lookback+1` training bars and 2 backtest bars, otherwise it errors.
  - `--periods-per-year N` (default: inferred from `--interval`)

- Output
  - `--version` (or `-V`) print `trader-hs` version
  - `--json` machine-readable JSON to stdout:
    - Trade-only: `{ "mode": "signal", "signal": ... }` or `{ "mode": "trade", "trade": ... }`
    - Backtest: `{ "mode": "backtest", "backtest": ... }` (includes `"baselines"` like `buy-hold` / `sma-cross(...)`, and `"trade"` if `--binance-trade` is set)
    - Backtest trades include `exitReason`; risk halts report `MAX_DRAWDOWN`/`MAX_DAILY_LOSS` when applicable.
    - Backtest `positions` reflect the bar-open position for t->t+1; `agreementOk` flags when Kalman/LSTM open-direction signals match with non-neutral directions; agreement rate only counts bars where both models emit a non-neutral open direction.
    - Latest signal output includes `closeDirection` to indicate the close-threshold direction (when available).
    - When confidence gating is enabled, `closeDirection` respects the gated signal direction (matching backtests).

Tests
-----
```
cd haskell
cabal test
```

REST API
--------
Run the bot as a REST API:
- Most endpoints are **stateless** (each request loads data and computes/trains as needed).
- The optional **live bot** endpoints (`/bot/*`) start a **stateful, non-stop** loop that ingests new bars, fine-tunes the model each bar, and (optionally) places orders until stopped.
- `GET /metrics` exposes a small Prometheus-style endpoint.
```
cd haskell
cabal run trader-hs -- --serve --port 8080
```

Optional auth (recommended for any deployment):
- Set `TRADER_API_TOKEN` to require a token on all endpoints except `/health`
- Send either `Authorization: Bearer <token>` or `X-API-Key: <token>`
- `/health` stays public, and reports `authRequired`/`authOk` when `TRADER_API_TOKEN` is set (useful for quickly checking auth wiring)

Build info:
- `GET /` and `GET /health` include `version` and optional `commit` (from env `TRADER_GIT_COMMIT` / `TRADER_COMMIT` / `GIT_COMMIT` / `COMMIT_SHA`).

Endpoints:
- `GET /` → basic endpoint list
- `GET /health`
- `GET /metrics`
- `GET /ops` → persisted operations feed (optional; enabled via `TRADER_OPS_DIR`)
- `GET /cache` → in-memory cache stats (entries + hit/miss)
- `POST /cache/clear` → clears the in-memory cache
- `POST /signal` → returns the latest signal (no orders)
- `POST /signal/async` → starts an async signal job
- `GET /signal/async/:jobId` → polls an async signal job (also accepts `POST` for proxy compatibility)
- `POST /trade` → returns the latest signal + attempts an order (Binance live orders by default; use `binanceLive=false` for test orders; Coinbase is live-only)
- `POST /trade/async` → starts an async trade job
- `GET /trade/async/:jobId` → polls an async trade job (also accepts `POST` for proxy compatibility)
- Signal endpoints validate request parameters the same way as the CLI; invalid ranges return 400.
- Use `predictors` in API payloads to select which predictors train/use (same format as `--predictors`).
- `POST /backtest` → runs a backtest and returns summary metrics
- `POST /backtest/async` → starts an async backtest job
- `GET /backtest/async/:jobId` → polls an async backtest job (also accepts `POST` for proxy compatibility)
- Backtest endpoints return 400 for inconsistent inputs (e.g., lookback >= bars, high/low length mismatches).
- `POST /optimizer/run` → runs the optimizer executable, merges the run into `top-combos.json`, and returns the last JSONL record
- `GET /optimizer/combos` → returns `top-combos.json` (UI helper; includes combo `operations` when available)
  - Top-combo merges rank by annualized equity (`metrics.annualizedReturn`), using score and final equity as tie-breakers.
  - `top-combos.json` also includes `bestOptimizationTechniques`, a curated list of optimization best practices with short explanations for downstream consumers, plus `optimizationTechniquesApplied`/`ensemble` sections that summarize the Sobol seeding, successive halving, Bayesian-inspired exploitation, walk-forward validation, and ensemble construction applied during a run.
- `POST /binance/keys` → checks key/secret presence and probes signed endpoints (test order quantity is rounded to the symbol step size; `tradeTest.skipped` indicates the test order was not attempted due to missing/invalid sizing or minNotional)
- `POST /binance/trades` → returns account trades (spot/margin require symbol; futures supports all symbols)
- `POST /binance/positions` → returns open Binance futures positions plus recent klines for charting
- `POST /coinbase/keys` → checks Coinbase key/secret/passphrase via a signed `/accounts` probe
- `POST /binance/listenKey` → creates a Binance user-data listenKey (returns WebSocket URL)
- `POST /binance/listenKey/keepAlive` → keep-alives a listenKey (required ~every 30 minutes)
- `POST /binance/listenKey/close` → closes a listenKey
- `POST /bot/start` → starts one or more live bot loops (Binance data only; use `botSymbols` for multi-symbol; errors include per-symbol details when all fail)
- `POST /bot/stop` → stops the live bot loop (`?symbol=BTCUSDT` stops one; omit to stop all)
- `GET /bot/status` → returns live bot status (`?symbol=BTCUSDT` for one; multi-bot returns `multi=true` + `bots[]`; `starting=true` includes `startingReason`; `tail=N` caps history, max 5000, and open trade entries are clamped to the tail).
- On API boot, the live bot auto-starts for `TRADER_BOT_SYMBOLS` (or `--binance-symbol`) and also keeps bots running for the current top 10 combos in `top-combos.json` (Binance only), prioritized by annualized equity (`metrics.annualizedReturn`) with trade count as a tie-breaker; it warns if fewer than 10 unique symbols exist to start all top-combo bots. Trading is enabled by default (requires Binance API keys) and missing bots restart on the next poll interval.

Always-on live bot (cron watchdog):
- Use `deploy/ensure-bot-running.sh` to check `/bot/status` and call `/bot/start` if the bot is not running.
- Provide a start payload with `TRADER_BOT_START_FILE` (JSON file) or `TRADER_BOT_START_BODY` (raw JSON string).
- If `bot-start.json` exists in the repo root, the script uses it as the default start payload.
- Optional: set `TRADER_BOT_SYMBOLS` or `TRADER_BOT_SYMBOL` to check specific symbols; otherwise it checks the global status.
- If no start payload is provided, the script builds a minimal one from `TRADER_BOT_SYMBOLS` or `TRADER_BOT_SYMBOL` (and optional `TRADER_BOT_TRADE`).
- The script reads `.env` by default (override with `TRADER_ENV_FILE`).
- Relative paths for `TRADER_ENV_FILE` and `TRADER_BOT_START_FILE` are resolved from the repo root.
- Requires `python3` for JSON parsing (override with `TRADER_BOT_PYTHON_BIN` if needed).
- Example cron entry:
```
*/2 * * * * TRADER_ENV_FILE=/path/to/.env TRADER_BOT_START_FILE=/path/to/bot-start.json /path/to/repo/deploy/ensure-bot-running.sh >> /var/log/trader-bot-cron.log 2>&1
```

Request limits:
- `TRADER_API_MAX_BODY_BYTES` (default 1048576) caps JSON request payload size; larger requests return 413.
- `TRADER_API_MAX_OPTIMIZER_OUTPUT_BYTES` (default 20000) truncates `/optimizer/run` stdout/stderr in responses.
- Truncated optimizer trial errors end with a `…` marker.
- Optimizer JSON output uses stable key ordering for easier diffs.
- When the backtest queue is busy, the API queues the request; the UI waits for the slot to clear and reports when the backtest finishes.

Backtest limits:
- `TRADER_API_MAX_BACKTEST_RUNNING` (default: `1`) caps concurrent backtests across sync/async requests and daily top-combo refreshes.
- `TRADER_API_BACKTEST_TIMEOUT_SEC` (default: `900`) cancels long-running backtests (sync returns 504; async jobs return an error).

Optimizer script tips:
- `optimize-equity` defaults to `--objective annualized-equity` (annualized return).
- `optimize-equity --quality` enables a deeper search (more trials, wider ranges, min round trips, smaller splits).
- `--auto-high-low` auto-detects CSV high/low columns to enable intrabar stops/TP/trailing.
- `--platform`/`--platforms` sample exchange platforms when using `--binance-symbol`/`--symbol` (default: binance; supports coinbase/kraken/poloniex).
- `--bars-auto-prob` and `--bars-distribution` tune how often bars=auto/all is sampled and how explicit bars are drawn.
- `--seed-trials`, `--seed-ratio`, `--survivor-fraction`, `--perturb-scale-*`, and `--early-stop-no-improve` tune search seeding, exploitation, and early stopping.
- `--min-hold-bars-min/max`, `--cooldown-bars-min/max`, and `--max-hold-bars-min/max` sample trade gating windows to reduce churn.
- `--min-win-rate`, `--min-profit-factor`, and `--min-exposure` filter out low-quality candidates.
- `--min-sharpe`, `--min-annualized-return`, `--min-calmar`, `--max-turnover`, `--min-wf-sharpe-mean`, and `--max-wf-sharpe-std` filter for higher/stabler candidates.
- `--min-edge-min/max`, `--min-signal-to-noise-min/max`, `--edge-buffer-min/max`, `--p-cost-aware-edge`, and `--trend-lookback-min/max` tune entry gating (edge-buffer > 0 enables cost-aware edge; set `--p-cost-aware-edge` to override).
- `--p-threshold-factor`, `--threshold-factor-alpha-min/max`, `--threshold-factor-min-min/max`, `--threshold-factor-max-min/max`, `--threshold-factor-floor-min/max`, and `--threshold-factor-weight-min/max` sample dynamic threshold-factor settings.
- `--p-intrabar-take-profit-first` mixes intrabar fill ordering when high/low data is available.
- `--p-tri-layer` plus `--tri-layer-fast-mult-min/max`, `--tri-layer-slow-mult-min/max`, `--tri-layer-cloud-padding-min/max`, `--tri-layer-cloud-slope-min/max`, `--tri-layer-cloud-width-min/max`, `--tri-layer-touch-lookback-min/max`, `--tri-layer-price-action-body-min/max`, `--tri-layer-exit-on-slow`, `--kalman-band-lookback-min/max`, `--kalman-band-std-mult-min/max`, `--p-tri-layer-price-action`, `--lstm-exit-flip-bars-min/max`, `--lstm-exit-flip-grace-bars-min/max`, and `--lstm-exit-flip-strong` sample tri-layer gating and LSTM flip exits (set `--p-tri-layer 1` to force tri-layer gating).
- `--stop-min/max`, `--tp-min/max`, `--trail-min/max`, and `--p-disable-stop/tp/trail` sample bracket exits; `--stop-vol-mult-min/max`, `--tp-vol-mult-min/max`, `--trail-vol-mult-min/max`, and `--p-disable-*-vol-mult` sample volatility-based brackets.
- `--max-position-size-min/max`, `--vol-target-*`, `--vol-lookback-*`/`--vol-ewma-alpha-*`, `--vol-floor-*`, `--vol-scale-max-*`, `--max-volatility-*`, and `--periods-per-year-*` tune sizing (use `--p-disable-vol-target`/`--p-disable-max-volatility` to mix disabled samples).
- `--p-disable-vol-ewma-alpha` mixes EWMA vs rolling vol when using `--vol-ewma-alpha-*`.
- `--funding-rate-min/max`, `--p-funding-by-side`, `--p-funding-on-open`, `--rebalance-bars-min/max`, `--rebalance-threshold-min/max`, `--p-rebalance-global`, and `--p-rebalance-reset-on-signal` sample funding and rebalance behavior.
- `--blend-weight-min/max` plus `--method-weight-blend` sample the blend method mix.
- `--kalman-market-top-n-min/max` tunes the Kalman market-context sample size (Binance only).
- `--kalman-z-min-min/max`, `--kalman-z-max-min/max`, `--max-high-vol-prob-min/max`, `--max-conformal-width-min/max`, `--max-quantile-width-min/max`, `--p-confirm-conformal`, `--p-confirm-quantiles`, `--p-confidence-sizing`, `--lstm-confidence-soft-min/max`, `--lstm-confidence-hard-min/max`, and `--min-position-size-min/max` tune confidence gating/sizing (use `--p-disable-max-*` to mix disabled samples).
- `--lr-min/max`, `--patience-max`, `--grad-clip-min/max`, and `--p-disable-grad-clip` tune LSTM training hyperparameters.
- `--tune-objective`, `--tune-penalty-*`, and `--tune-stress-*` align the internal threshold sweep objective (`--tune-stress-*-min/max` lets it sample ranges).
- `--walk-forward-folds-min/max` varies walk-forward fold counts in the tune stats.
- Auto optimizer biases `--p-long-short` to match existing open positions/orders (short requires long-short; spot/margin suppresses long-short).
- `/optimizer/run` accepts the same options via camelCase JSON fields (e.g., `barsAutoProb`, `seedTrials`, `seedRatio`, `survivorFraction`, `perturbScaleDouble`, `perturbScaleInt`, `earlyStopNoImprove`, `minHoldBarsMin`, `blendWeightMin`, `minWinRate`, `minAnnualizedReturn`, `minCalmar`, `maxTurnover`, `minSignalToNoiseMin`, `pThresholdFactor`, `thresholdFactorAlphaMin`, `thresholdFactorMinMin`, `thresholdFactorWeightMax`, `minSharpe`, `minWalkForwardSharpeMean`, `stopMin`, `pIntrabarTakeProfitFirst`, `pTriLayer`, `pTriLayerPriceAction`, `triLayerFastMultMin`, `triLayerCloudPaddingMin`, `triLayerCloudSlopeMin`, `triLayerCloudWidthMin`, `triLayerTouchLookbackMin`, `triLayerPriceActionBodyMin`, `triLayerExitOnSlow`, `fundingRateMin`, `fundingRateMax`, `rebalanceBarsMin`, `rebalanceBarsMax`, `rebalanceThresholdMin`, `rebalanceThresholdMax`, `pFundingBySide`, `pFundingOnOpen`, `pRebalanceGlobal`, `pRebalanceResetOnSignal`, `kalmanBandLookbackMin`, `kalmanBandStdMultMin`, `lstmExitFlipBarsMin`, `lstmExitFlipGraceBarsMin`, `lstmExitFlipStrong`, `lstmConfidenceSoftMin`, `lstmConfidenceHardMin`, `kalmanZMinMin`, `lrMin`, `platforms`); numeric fields may be JSON numbers or numeric strings (including `nan`/`inf`) for legacy compatibility.

State directory (recommended for persistence across deployments):
- Set `TRADER_STATE_DIR` to a shared writable directory to persist:
  - ops history (`ops.jsonl`)
  - JSONL journal events
  - live-bot status snapshots (`bot-state-<symbol>.json`)
  - optimizer top-combos (`top-combos.json`)
  - async job results (`/signal/async`, `/backtest/async`, `/trade/async`)
  - LSTM weights (for incremental training)
- Per-feature `TRADER_*_DIR` variables override the state directory; set any of them to an empty string to disable that feature.
- Docker image default: `TRADER_STATE_DIR=/var/lib/trader/state` (mount `/var/lib/trader` to durable storage to keep state across redeploys).
- For App Runner (no EFS support), use S3 persistence via `TRADER_STATE_S3_BUCKET` and keep `TRADER_STATE_DIR` for local-only state if desired.
- `deploy-aws-quick.sh` defaults `TRADER_STATE_DIR` to `/var/lib/trader/state`; you can add S3 state flags (`--state-s3-*`) and `--instance-role-arn`. When updating an existing App Runner service, it reuses the service's S3 state settings and instance role if you don't pass new values, defaults `TRADER_BOT_TRADE=true` unless overridden, and forwards `TRADER_BOT_SYMBOLS`/`TRADER_BOT_TRADE` plus `BINANCE_API_KEY`/`BINANCE_API_SECRET` when set.

S3 state (recommended for App Runner):
- Set `TRADER_STATE_S3_BUCKET` (optional `TRADER_STATE_S3_PREFIX`, `TRADER_STATE_S3_REGION`) to persist bot snapshots, optimizer top-combos, and ops logs in S3.
- Requires AWS credentials or an App Runner instance role with S3 access.
- Bot snapshots include orders/trades, so the UI can show history after restarts; journal/async/LSTM weights still use `TRADER_STATE_DIR`.

Optional journaling:
- Set `TRADER_JOURNAL_DIR` to a directory path to write JSONL events (server start/stop, bot start/stop, bot orders/halts, trade orders).
- If `TRADER_STATE_DIR` is set, defaults to `TRADER_STATE_DIR/journal`.

Optional webhooks (Discord-compatible):
- Set `TRADER_WEBHOOK_URL` to send notifications for live-bot and trade events.
- Payload: JSON with a `content` string (Discord webhook compatible).
- `TRADER_WEBHOOK_EVENTS` (comma-separated) filters which events are sent; when unset, all webhook events are sent.
- Event types: `bot.started`, `bot.start_failed`, `bot.stop`, `bot.order`, `bot.halt`, `trade.order`.

Optional ops persistence (powers `GET /ops` and the “operations” history):
- Set `TRADER_OPS_DIR` to a writable directory (writes `ops.jsonl`)
- If `TRADER_STATE_DIR` is set, defaults to `TRADER_STATE_DIR/ops`.
- `TRADER_OPS_MAX_IN_MEMORY` (default: `20000`) max operations kept in memory per process
- When S3 persistence is enabled via `TRADER_STATE_S3_BUCKET`, ops are restored from `ops/ops.jsonl` and synced back on a timer.
- `TRADER_OPS_S3_EVERY_SEC` (default: `60`) cadence for syncing ops to S3 (`0` disables S3 sync but still restores from S3).
- `GET /ops` query params:
  - `limit` (default: `200`, max: `5000`)
  - `since` (only return ops with `id > since`)
  - `kind` (exact match on operation kind)
- Ops log kinds include:
  - `binance.request` for every Binance API request (method/path/latency/status; signature/listenKey values are redacted).
- `bot.status` snapshots on start and every minute with running/live/halts/errors (used by the live/offline timeline chart in the UI).

Optional live-bot status snapshots (keeps `/bot/status` data across restarts):
- Set `TRADER_BOT_STATE_DIR` to a writable directory (writes `bot-state-<symbol>.json`; set empty to disable)
- When unset, defaults to `TRADER_STATE_DIR/bot` (if set) or `.tmp/bot` (local only).
- When S3 persistence is enabled, the API serves local snapshots first and only falls back to S3 when local data is missing.

Optional optimizer combo persistence (keeps `/optimizer/combos` data across restarts/deploys):
- Set `TRADER_OPTIMIZER_COMBOS_DIR` to a writable directory (writes `top-combos.json`)
- When unset, defaults to `TRADER_STATE_DIR/optimizer` (if set) or `.tmp/optimizer` (local only).
- `TRADER_OPTIMIZER_MAX_COMBOS` (default: `200`) caps the merged combo list size
- `TRADER_OPTIMIZER_COMBOS_HISTORY_DIR` (default: `<combos dir>/top-combos-history`) stores timestamped snapshots (set to `off`, `false`, or `0` to disable).
- When S3 persistence is enabled, new optimizer runs merge against the existing S3 `top-combos.json` so the best-ever combos are retained, and history snapshots are written under `optimizer/history/`.
- When S3 persistence is enabled, the API serves local `top-combos.json` first and only falls back to S3 when local data is missing.

Optional daily top-combo backtests (refreshes metrics for the best performers):
- `TRADER_TOP_COMBOS_BACKTEST_ENABLED` (default: `true`) enable daily refreshes of the top combos (plus per-candle attempts while a live bot is running).
- `TRADER_TOP_COMBOS_BACKTEST_TOP_N` (default: `5`, minimum: `5`) number of top combos to re-backtest per cycle.
- `TRADER_TOP_COMBOS_BACKTEST_EVERY_SEC` (default: `86400`) cadence in seconds.
- Uses the latest exchange data and writes updated `metrics`, `finalEquity`, `score`, and `operations` back into `top-combos.json` (and S3 when configured).
- The top 5 combos are always refreshed and overwrite prior metrics even if equity performance drops.
- Each new candle processed by a live bot triggers a top-5 backtest attempt to refresh operations.

Async-job persistence (default on; recommended if you run multiple instances behind a non-sticky load balancer, or want polling to survive restarts):
- Default directory: `TRADER_STATE_DIR/async` (if set) or `.tmp/async` (local only). Set `TRADER_API_ASYNC_DIR` to a shared writable directory (the API writes per-endpoint subdirectories under it), or set it empty to disable.

Optional in-memory caching (recommended for the Web UI’s repeated calls):
- `TRADER_API_CACHE_TTL_MS` (default: `30000`) cache TTL in milliseconds (`0` disables)
- `TRADER_API_CACHE_MAX_ENTRIES` (default: `64`) max cached entries (`0` disables)
  - To bypass cache for a single request, send `Cache-Control: no-cache` or add `?nocache=1`.

Optional API compute limits (useful on small instances):
- `TRADER_API_MAX_BARS_LSTM` (default: `1000`) max LSTM bars accepted by the API
- `TRADER_API_MAX_EPOCHS` (default: `100`) max LSTM epochs accepted by the API
- `TRADER_API_MAX_HIDDEN_SIZE` (default: `32`; set to `50` to allow larger LSTM hidden sizes)

Optional LSTM weight persistence (recommended for faster repeated backtests):
- `TRADER_LSTM_WEIGHTS_DIR` (default: `TRADER_STATE_DIR/lstm` if set, else `.tmp/lstm`) directory to persist LSTM weights between runs (set to an empty string to disable)
  - Used by both backtests and the live bot (online fine-tuning).
  - The persisted seed is only used when it was trained on **≤** the current training window (prevents lookahead leakage when you change tune/backtest splits).
  - Keys include the data source (platform+symbol, plus Binance market), interval, normalization, hidden size, and lookback.

Examples:
```
curl -s http://127.0.0.1:8080/health
```

```
curl -s -X POST http://127.0.0.1:8080/signal \
  -H 'Content-Type: application/json' \
  -d '{"binanceSymbol":"BTCUSDT","interval":"1h","bars":200,"method":"10","openThreshold":0.003838,"closeThreshold":0.003838}'
```

Optimize `method` and thresholds on the tune split (no orders):
```
curl -s -X POST http://127.0.0.1:8080/backtest \
  -H 'Content-Type: application/json' \
  -d '{"binanceSymbol":"BTCUSDT","interval":"1h","bars":1000,"optimizeOperations":true}'
```

```
export BINANCE_API_KEY=...
export BINANCE_API_SECRET=...
curl -s -X POST http://127.0.0.1:8080/trade \
  -H 'Content-Type: application/json' \
  -d '{"binanceSymbol":"BTCUSDT","interval":"1h","bars":200,"method":"10","openThreshold":0.003838,"closeThreshold":0.003838,"orderQuote":20,"binanceLive":false}'
```

Start the live bot (paper mode; no orders). `botTrade` defaults to `true`, so set `botTrade=false` explicitly for paper mode:
```
curl -s -X POST http://127.0.0.1:8080/bot/start \
  -H 'Content-Type: application/json' \
  -d '{"binanceSymbol":"BTCUSDT","interval":"1h","bars":500,"method":"11","openThreshold":0.002,"closeThreshold":0.002,"fee":0.0008,"botOnlineEpochs":1,"botTrade":false}'
```

Start multiple symbols (auto-apply latest top combo per symbol):
```
curl -s -X POST http://127.0.0.1:8080/bot/start \
  -H 'Content-Type: application/json' \
  -d '{"botSymbols":["BTCUSDT","ETHUSDT"],"interval":"1h","bars":500,"method":"11","openThreshold":0.002,"closeThreshold":0.002,"fee":0.0008,"botOnlineEpochs":1,"botTrade":false}'
```

Multi-symbol notes:
- Use `botSymbols` (array) or `TRADER_BOT_SYMBOLS=BTCUSDT,ETHUSDT` to define the bot symbol set.
- `GET /bot/status?symbol=BTCUSDT` returns a single bot; omit `symbol` to get `multi=true` with `bots[]`.
- `POST /bot/stop?symbol=BTCUSDT` stops one bot; omit `symbol` to stop all.

Live safety (startup position):
- When `botTrade=true`, `/bot/start` adopts any existing position or open exchange orders for the symbol (long or short, subject to positioning).
- Adopted positions use the gated `closeDirection` (closeThreshold + confidence filters) to decide hold/exit on startup.
- Live bot exit decisions during the run loop use the gated `closeDirection` logic as well.
- When `botTrade=true`, `/bot/start` also auto-starts bots for orphan open futures positions that have a matching top combo in `top-combos.json` (even if not listed in `botSymbols`).
- `botAdoptExistingPosition` is now implied and ignored if provided.
- If an existing position or open orders are detected, `/bot/start` waits for a top combo compatible with that operation before starting (e.g., shorts require `positioning=long-short`).

Auto-optimize after each buy/sell operation:
- Thresholds only: add `"sweepThreshold": true`
- Method + thresholds: add `"optimizeOperations": true`
- The live bot always syncs to the latest top combo from `top-combos.json` (poll interval `TRADER_BOT_COMBOS_POLL_SEC`, default 30s); interval-less combos are treated as compatible with the current interval.

Check status:
```
curl -s http://127.0.0.1:8080/bot/status
```

Stop it:
```
curl -s -X POST http://127.0.0.1:8080/bot/stop
```

Assumptions:
- Requests must include a data source: `data` (CSV path) or `binanceSymbol`.
- `method` is `"11"`/`"both"` (direction-agreement gated), `"10"`/`"kalman"` (Kalman only), `"01"`/`"lstm"` (LSTM only), `"blend"` (weighted average; see `--blend-weight`), or `"router"` (adaptive selection; Kalman confidence/risk gates apply only on Kalman-selected bars; see `--router-lookback` / `--router-min-score`).
- `positioning` is `"long-flat"` (default, alias `"long-only"`/`"long"`) or `"long-short"` (shorts require futures when placing orders or running the live bot).
- Hedge-mode long+short futures positions for the same symbol must be flattened to one side before bot start/adoption or futures trade requests.

Deploy to AWS
-------------
See `DEPLOY_AWS_QUICKSTART.md`, `DEPLOY_AWS.md`, and `deploy/aws/README.md`.

Note: `/bot/*` is stateful, and async endpoints persist job state to `TRADER_STATE_DIR/async` (if set) or `.tmp/async` by default (local only). For deployments behind non-sticky load balancers (including CloudFront `/api/*`), keep the backend **single-instance** unless you set `TRADER_API_ASYNC_DIR` (or `TRADER_STATE_DIR`) to a shared writable directory. If the UI reports "Async job not found", the backend likely restarted or the load balancer is not sticky; use shared async storage or run a single instance.

Web UI
------
A TypeScript web UI lives in `haskell/web` (Vite + React). It talks to the REST API and visualizes signals/backtests (including the equity curve).
The UI layout uses a refreshed header, section grouping, and spacing for faster scanning on desktop and mobile.
The UI styling now emphasizes a light-first palette, calmer surfaces, and updated typography for a cleaner read.
Configuration sections and result panels are collapsible; the UI remembers open/closed state locally, offers expand/collapse-all controls in the configuration panel, and starts low-signal panels (Data Log, Request preview) collapsed by default.
The overview card summarizes connection, execution mode, and the latest signal/backtest/trade results for quick scanning.
The platform selector includes Coinbase (symbols use BASE-QUOTE like `BTC-USD`); API keys are stored per platform, trading supports Binance + Coinbase spot, and the live bot remains Binance-only.
Symbol inputs are validated per platform (Binance `BTCUSDT`, Coinbase `BTC-USD`, Poloniex `BTC_USDT`).
The Latest signal card includes a decision-logic checklist that shows direction agreement, gating filters, and sizing behind the operate/hold outcome.
The Live bot panel includes visual aids for live data (price pulse, signal/position compass, and risk buffer).
Realtime telemetry and feed history are tracked per running bot so switching bots keeps each bot's live context.
When trading is armed, Long/Short positioning requires Futures market (the UI switches Market to Futures).
Optimizer combos are clamped to API compute limits reported by `/health`.
Optimizer combos only override Positioning when they include it; otherwise the current selection is preserved.
The UI shows whether combos are coming from the live API or the static fallback, their last update time, and how many combos are displayed; you can choose the combo count (default 5, up to the available combos).
Optimizer combos show when each combo was obtained, include annualized equity (default ordering), support ordering by date, and can be filtered by minimum final equity.
Optimizer run forms (including the Optimizer combos panel) launch `/optimizer/run` with constraints, accept advanced JSON overrides for `source`/`binanceSymbol`/`data` and `timeoutSec`, validate backtest/tune ratios, include an annualized-equity preset button, and surface equity-focused info popovers; complex parameters (method/thresholds/splits/LSTM/optimization) include info buttons.
Manual edits to Method/open/close thresholds are preserved when optimizer combos or optimization results apply.
The UI sends explicit zero/false values for default-on risk settings (e.g., min-hold/cooldown/max-hold, min SNR, vol target/max-vol, rebalancing, cost-aware edge, confidence gates) so disable toggles take effect.
Combos can be previewed without applying; Apply (or Apply top combo) loads values and auto-starts a live bot for the combo symbol (Binance only), selecting the existing bot if it is already running; top-combo auto-apply pauses while a manual Apply is starting a bot, and Refresh combos resyncs.
If a refresh fails, the last known combos remain visible with a warning banner.
The UI includes a “Binance account trades” panel that surfaces full exchange history via `/binance/trades`.
The UI includes an “Open positions” panel that charts every open Binance futures position via `/binance/positions` (auto-loads on page load, interval/market changes, and Binance key/auth updates including API token changes).
The UI includes an “Orphaned operations” panel that highlights open futures positions not currently adopted by a running/starting bot; matching is per-market and per-hedge side, and bots with `tradeEnabled=false` do not count as adopted (labeled as trade-off).
The bot state timeline shows the hovered timestamp.
Chart tooltips show the hovered bar timestamp when available.
Charts scale to use most of the viewport height for easier inspection.
The issue bar Fix button clamps bars/epochs/hidden size to the API limits when they are exceeded.
The Binance account trades panel requires a non-negative From ID when provided.
Binance account trades time filters accept unix ms timestamps or ISO-8601 dates (YYYY-MM-DD or YYYY-MM-DDTHH:MM).
Loading a profile clears manual override locks so combos can apply again.
Hover optimizer combos to inspect the operations captured for each top performer.
The configuration panel includes quick-jump buttons for major sections (API, market, lookback, thresholds, risk, optimization, live bot, trade).
Jump shortcuts move focus to the target section, with clearer focus rings for keyboard navigation.
The configuration panel keeps a sticky action bar with readiness status, run buttons, and issue shortcuts that jump/flash the relevant inputs.
The backtest/tune ratio inputs show a split preview with the minimum bars required for the current lookback.
The backtest summary chart includes a Download log button to export the backtest operations.
Backtest charts allow deeper zoom (mouse wheel down to ~6 bars) for close inspection.
When the UI is served via CloudFront with a `/api/*` behavior, `apiBaseUrl` must be `/api` to avoid CORS issues (the quick AWS deploy script enforces this when a distribution ID is provided unless `--ui-api-direct` is set). When the script knows the API URL and a CloudFront distribution is in play, it auto-fills `apiFallbackUrl` for failover: `/api` falls back to the API URL, and direct API URLs fall back to `/api` (override with `--ui-api-fallback`/`TRADER_UI_API_FALLBACK_URL` if needed). The script creates/updates the `/api/*` behavior to point at the API origin (disables caching, forwards auth headers, and excludes the Host header to avoid App Runner 404s) when a distribution ID is provided.
The UI auto-applies top combos when available and shows when a combo auto-applied; it also auto-starts missing bots for the top 5 combo symbols (Binance only), and manual override locks include an unlock button to let combos update those fields again.
The API panel includes quick actions to copy the base URL and open `/health`.
Numeric inputs accept comma decimals (e.g., 0,25) and ignore thousands separators.
The Data Log panel supports auto-scroll to keep the newest responses in view.
Filter the Data Log by label; Copy shown respects the current filter, and Jump to latest scrolls back down.

Run it:
```
# Terminal A (backend)
cd haskell
cabal run -v0 trader-hs -- --serve --port 8080

# Terminal B (frontend)
cd haskell/web
npm install
npm run dev
```

If your API uses a different port:
```
cd haskell/web
TRADER_API_TARGET=http://127.0.0.1:9090 npm run dev
```

Open `http://127.0.0.1:5173`.

Timeouts:
- Backend: set `TRADER_API_TIMEOUT_SEC` when starting `trader-hs` (default: `1800`).
- Frontend: set `timeoutsMs` in `haskell/web/public/trader-config.js` to increase UI request timeouts (e.g. long backtests).
- Frontend (dev proxy): set `TRADER_UI_PROXY_TIMEOUT_MS` to increase the Vite `/api` proxy timeout.

Proxying `/api/*` (CloudFront or similar): allow `GET`, `POST`, and `OPTIONS`; the UI will fall back to `GET` for async polling if `POST` hits proxy errors. Async signal/backtest starts retry transient 5xx/timeouts and can fail over to `apiFallbackUrl`; ensure the fallback points at the same backend to avoid mismatched job IDs.
If live bot start/status returns 502/503/504, verify the `/api/*` proxy target and origin health (CloudFront setups should keep `apiBaseUrl` at `/api` to avoid CORS).

If your backend has `TRADER_API_TOKEN` set, all endpoints except `/health` require auth.

- Web UI: `trader-config.js` is read at startup, so ensure it is served at `/trader-config.js` for static hosts.
- Web UI: set `apiToken` in `haskell/web/public/trader-config.js` (or `haskell/web/dist/trader-config.js` after build). The UI sends it as `Authorization: Bearer <token>` and `X-API-Key: <token>`. Only set `apiFallbackUrl` when your API supports CORS and you want explicit failover (quick deploy: `--ui-api-fallback`/`TRADER_UI_API_FALLBACK_URL`, or the script auto-fills it when a CloudFront distribution is used and the API URL is known). If the fallback host blocks CORS, the UI disables it for the session.
- Web UI (dev): set `TRADER_API_TOKEN` in `haskell/web/.env.local` to have the Vite `/api/*` proxy attach it automatically.

The UI also includes a “Live bot” panel to start/stop the continuous loop, show a chart per running live bot, and visualize each buy/sell operation on the selected bot chart (supports long/short on futures). It includes live/offline timeline charts with start/end controls when ops persistence is enabled: the selected bot shows the full timeline, and each running bot card shows a compact, shorter timeline. The chart reflects the available ops history and warns when the selected range extends beyond it.
When trading is armed, the UI blocks live bot start until Binance keys are provided or verified via “Check keys” (otherwise switch to paper mode).
When starting multi-symbol live bots, the UI uses the first bot symbol as the request symbol so `/bot/start` validation succeeds even if the main Symbol field is empty.
Optimizer combos are clamped to the API compute limits reported by `/health` when available.

Troubleshooting: “No live operations yet”
- The live bot only records an operation when it switches position (BUY/SELL). If the latest signal is `HOLD`/neutral, the operations list stays empty.
- Each new candle still triggers a trade attempt based on the desired position, even if it resolves to a no-op (already in position or neutral signal).
- A signal is neutral when the predicted next price is within the `openThreshold` deadband: it must be `> currentPrice*(1+openThreshold)` for UP or `< currentPrice*(1-openThreshold)` for DOWN.
- With `positioning=long-flat` (required by `/bot/start`), a DOWN signal while already flat does nothing; you’ll only see a SELL after you previously bought.
- If you want it to trade more often, lower `openThreshold`/`closeThreshold` (or run “Optimize thresholds/operations”) and/or use a higher timeframe.

Assumptions and limitations
---------------------------
- The strategy is intentionally simple (default long or flat; optional long-short for backtests and futures trade requests/live bot); it includes basic sizing/filters but is not a full portfolio/risk system or detailed transaction-cost model.
- Daily-loss resets prefer bar open timestamps when available; if timestamps are missing or misaligned, backtests fall back to interval-based day boundaries.
- If open timestamps are provided (CSV or API), their length must match the closes series; mismatches return an error to avoid misaligned day boundaries.
- Live order placement attempts to fetch/apply symbol filters (minQty/step size/minNotional), but is not exhaustive and may still be rejected by the exchange.
- This code is for experimentation and education only; it is **not** production-ready nor financial advice.
