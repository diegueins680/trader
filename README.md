Haskell Trading Bot (Kalman + LSTM + Binance)
=============================================

This repository contains a small Haskell trading demo that:
- Predicts the next price using a small **LSTM**, and a **multi-sensor Kalman fusion** layer that combines multiple model outputs into a single latent expected return signal.
- By default, only trades when Kalman and LSTM **agree on direction** (both predict up, or both predict down) — configurable via `--method`.
- Can backtest on CSV data or pull klines from **Binance** (and optionally place test/live market orders).

Features
--------
- Multi-sensor Kalman fusion filter for latent expected return (`haskell/app/Trader/KalmanFusion.hs`).
- Multiple predictive methods feeding into Kalman as an observation vector (`haskell/app/Trader/Predictors.hs`):
  - Gradient-boosted trees (LightGBM/CatBoost style, simplified)
  - TCN / dilated 1D CNN (simplified)
  - Transformer-style attention predictor (kNN attention)
  - HMM / regime model (3 regimes)
  - Quantile regression (q10/q50/q90)
  - Conformal interval wrapper
- LSTM next-step predictor with Adam, gradient clipping, and early stopping (`haskell/app/Trader/LSTM.hs`).
- Agreement-gated ensemble strategy (`haskell/app/Trader/Trading.hs`).
- Profitability, risk/volatility, trade execution, and efficiency metrics (incl. Sharpe, max drawdown) (`haskell/app/Trader/Metrics.hs`).
- Data sources: CSV or Binance klines (`haskell/app/Trader/Binance.hs`).
- Sample dataset in `data/sample_prices.csv`.

Quick start
-----------
Build and run with Cabal:
```
cd haskell
cabal run trader-hs -- --data ../data/sample_prices.csv --price-column close
```

Example backtest with tighter model settings:
```
cd haskell
cabal run trader-hs -- \
  --data ../data/sample_prices.csv \
  --normalization standard \
  --interval 5m \
  --lookback-window 6h \
  --hidden-size 8 \
  --epochs 10 \
  --open-threshold 0.001 \
  --close-threshold 0.001 \
  --fee 0.0005
```

Using Binance klines
--------------------
Fetch klines from Binance instead of a CSV:
```
cd haskell
cabal run trader-hs -- \
  --binance-symbol BTCUSDT \
  --interval 5m \
  --epochs 5
```

Sending Binance orders (optional)
---------------------------------
By default, orders are sent to `/api/v3/order/test`. Add `--binance-live` to send live orders.
For futures orders, add `--futures` (uses `/fapi` endpoints). For margin orders, add `--margin` (requires `--binance-live`).

Environment variables:
- `BINANCE_API_KEY`
- `BINANCE_API_SECRET`

Getting Binance API keys:
- Binance → Profile → **API Management** → **Create API**
- Enable only what you need (Spot/Margin/Futures trading) and keep withdrawals disabled
- Prefer IP restrictions (allowlist your server IP) when possible
- Save the secret: Binance only shows it once

Example (test endpoint):
```
cd haskell
export BINANCE_API_KEY=...
export BINANCE_API_SECRET=...
cabal run trader-hs -- \
  --binance-symbol BTCUSDT \
  --interval 5m \
  --epochs 5 \
  --trade-only \
  --binance-trade \
  --order-quote 50
```

CLI parameters
--------------
You must provide exactly one data source: `--data` (CSV) or `--binance-symbol` (Binance).

- Data source
  - `--data PATH` (default: none) CSV file containing prices
  - `--price-column close` CSV column name for price

- Bars & lookback (defaults: `--interval 5m`, `--lookback-window 24h` → 288 bars, `--bars auto`)
  - `--interval 5m` (alias `--binance-interval`) bar interval / Binance kline interval
  - `--bars auto` (alias `--binance-limit`) number of bars/klines to use (`auto` = all CSV, or 500 for Binance; CSV also supports `0` = all; Binance 2..1000)
  - `--lookback-window 24h` lookback window duration (converted to bars)
  - `--lookback-bars N` (alias `--lookback`) override the computed lookback bars

- Binance (price fetch / optional trading)
  - `--binance-symbol SYMBOL` (default: none) fetch klines from Binance (e.g., `BTCUSDT`)
  - `--binance-testnet` (default: off) use Binance testnet base URL
  - `--futures` (default: off) use Binance USDT-M futures endpoints (data + orders)
  - `--margin` (default: off) use Binance margin account endpoints for orders/balance (requires `--binance-live`)
  - `--binance-api-key KEY` (default: none) or env `BINANCE_API_KEY`
  - `--binance-api-secret SECRET` (default: none) or env `BINANCE_API_SECRET`
  - `--binance-trade` (default: off) place a market order for the latest signal
  - `--binance-live` (default: off) send LIVE orders (otherwise uses `/order/test`)
  - `--order-quote AMOUNT` (default: none) quote amount to spend on BUY (`quoteOrderQty`)
  - `--order-quantity QTY` (default: none) base quantity to trade (`quantity`)
  - `--order-quote-fraction F` (default: none) size BUY orders as a fraction of the free quote balance (`0 < F <= 1`)
  - `--max-order-quote Q` (default: none) cap the computed quote amount when using `--order-quote-fraction`
  - `--idempotency-key ID` (default: none) optional Binance `newClientOrderId` for idempotent orders
  - Sizing inputs are mutually exclusive: choose one of `--order-quantity`, `--order-quote`, or `--order-quote-fraction`.

- Normalization
  - `--normalization standard` one of `none|minmax|standard|log`

- LSTM
  - Lookback bars come from `--lookback-window`/`--lookback-bars`
  - `--hidden-size 16` hidden size
  - `--epochs 30` training epochs (Adam)
  - `--lr 1e-3` learning rate
  - `--val-ratio 0.2` validation split ratio (within training set)
  - `--patience 10` early stopping patience (`0` disables)
  - `--grad-clip N` (default: none) gradient clipping max L2 norm
  - `--seed 42` random seed for init

- Kalman fusion (latent expected return)
  - `--kalman-dt 1.0` scales process noise per step
  - `--kalman-process-var 1e-5` process noise variance
  - `--kalman-measurement-var 1e-3` fallback measurement variance (and initial variance)

- Strategy / costs
  - `--open-threshold 0.001` (or legacy `--threshold`) entry/open direction threshold (fractional deadband)
  - `--close-threshold 0.001` exit/close threshold (fractional deadband; defaults to open-threshold when omitted)
  - `--method 11` choose `11`/`both` (Kalman+LSTM direction-agreement), `10`/`kalman` (Kalman only), `01`/`lstm` (LSTM only)
    - When using `--method 10`, the LSTM is disabled (not trained).
    - When using `--method 01`, the Kalman/predictors are disabled (not trained).
  - `--positioning long-flat` (default) or `--positioning long-short` (allows short positions in backtests; if trading, requires `--futures`; live bot is long-flat only)
  - `--optimize-operations` optimize `--method`, `--open-threshold`, and `--close-threshold` on the tune split (uses best combo for the latest signal)
  - `--sweep-threshold` sweep open/close thresholds on the tune split and pick the best by final equity
  - `--trade-only` skip backtest/metrics and only compute the latest signal (and optionally place an order)
  - `--fee 0.0005` fee applied when switching position
  - `--stop-loss F` optional synthetic stop loss (`0 < F < 1`, e.g. `0.02` for 2%)
  - `--take-profit F` optional synthetic take profit (`0 < F < 1`)
  - `--trailing-stop F` optional synthetic trailing stop (`0 < F < 1`)
  - `--max-drawdown F` optional live-bot kill switch: halt if peak-to-trough drawdown exceeds `F`
  - `--max-daily-loss F` optional live-bot kill switch: halt if daily loss exceeds `F` (UTC day)
  - `--max-order-errors N` optional live-bot kill switch: halt after `N` consecutive order failures

- Metrics
  - `--backtest-ratio 0.2` holdout ratio (last portion of series; avoids lookahead)
  - `--periods-per-year N` (default: inferred from `--interval`)

- Output
  - `--json` machine-readable JSON to stdout:
    - Trade-only: `{ "mode": "signal", "signal": ... }` or `{ "mode": "trade", "trade": ... }`
    - Backtest: `{ "mode": "backtest", "backtest": ... }` (and includes `"trade"` if `--binance-trade` is set)

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

Endpoints:
- `GET /` → basic endpoint list
- `GET /health`
- `GET /metrics`
- `POST /signal` → returns the latest signal (no orders)
- `POST /signal/async` → starts an async signal job
- `GET /signal/async/:jobId` → polls an async signal job (also accepts `POST` for proxy compatibility)
- `POST /trade` → returns the latest signal + attempts an order
- `POST /trade/async` → starts an async trade job
- `GET /trade/async/:jobId` → polls an async trade job (also accepts `POST` for proxy compatibility)
- `POST /backtest` → runs a backtest and returns summary metrics
- `POST /backtest/async` → starts an async backtest job
- `GET /backtest/async/:jobId` → polls an async backtest job (also accepts `POST` for proxy compatibility)
- `POST /binance/keys` → checks key/secret presence and probes signed endpoints
- `POST /bot/start` → starts the live bot loop (Binance data only)
- `POST /bot/stop` → stops the live bot loop
- `GET /bot/status` → returns the live bot status + chart data (prices/equity/positions/operations)

Optional journaling:
- Set `TRADER_JOURNAL_DIR` to a directory path to write JSONL events (server start/stop, bot start/stop, bot orders/halts, trade orders).

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
  -d '{"binanceSymbol":"BTCUSDT","interval":"5m","bars":1000,"optimizeOperations":true}'
```

```
export BINANCE_API_KEY=...
export BINANCE_API_SECRET=...
curl -s -X POST http://127.0.0.1:8080/trade \
  -H 'Content-Type: application/json' \
  -d '{"binanceSymbol":"BTCUSDT","interval":"1h","bars":200,"method":"10","openThreshold":0.003838,"closeThreshold":0.003838,"orderQuote":20,"binanceLive":false}'
```

Start the live bot (paper mode; no orders):
```
curl -s -X POST http://127.0.0.1:8080/bot/start \
  -H 'Content-Type: application/json' \
  -d '{"binanceSymbol":"BTCUSDT","interval":"5m","bars":500,"method":"11","openThreshold":0.001,"closeThreshold":0.001,"fee":0.0005,"botOnlineEpochs":1,"botTrade":false}'
```

Auto-optimize after each buy/sell operation:
- Thresholds only: add `"sweepThreshold": true`
- Method + thresholds: add `"optimizeOperations": true`

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
- `method` is `"11"`/`"both"` (direction-agreement gated), `"10"`/`"kalman"` (Kalman only), or `"01"`/`"lstm"` (LSTM only).
- `positioning` is `"long-flat"` (default) or `"long-short"` (shorts require futures when placing orders; live bot is long-flat only).

Deploy to AWS
-------------
See `deploy/aws/README.md`.

Web UI
------
A TypeScript web UI lives in `haskell/web` (Vite + React). It talks to the REST API and visualizes signals/backtests (including the equity curve).

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

If your backend has `TRADER_API_TOKEN` set, all endpoints except `/health` require auth.

- Web UI: paste the token into the UI’s “API token” field (stored in session storage and sent as `Authorization: Bearer <token>`).
- Web UI (dev): set `TRADER_API_TOKEN` in `haskell/web/.env.local` to have the Vite `/api/*` proxy attach it automatically.

The UI also includes a “Live bot” panel to start/stop the continuous loop and visualize each buy/sell operation on the chart (long-flat only).

Troubleshooting: “No live operations yet”
- The live bot only records an operation when it switches position (BUY/SELL). If the latest signal is `HOLD`/neutral, the operations list stays empty.
- A signal is neutral when the predicted next price is within the `openThreshold` deadband: it must be `> currentPrice*(1+openThreshold)` for UP or `< currentPrice*(1-openThreshold)` for DOWN.
- With `positioning=long-flat` (required by `/bot/start`), a DOWN signal while already flat does nothing; you’ll only see a SELL after you previously bought.
- If you want it to trade more often, lower `openThreshold`/`closeThreshold` (or run “Optimize thresholds/operations”) and/or use a higher timeframe.

Assumptions and limitations
---------------------------
- The strategy is intentionally simple (default long or flat; optional long-short for backtests and futures trade requests) and does not include sizing, risk limits, or robust transaction cost modeling.
- Live order placement attempts to fetch/apply symbol filters (minQty/step size/minNotional), but is not exhaustive and may still be rejected by the exchange.
- This code is for experimentation and education only; it is **not** production-ready nor financial advice.
