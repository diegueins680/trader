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
  --threshold 0.001 \
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
  --bars 500 \
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
  --bars 500 \
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

- Bars & lookback (defaults: `--interval 5m`, `--lookback-window 24h` → 288 bars)
  - `--interval 5m` (alias `--binance-interval`) bar interval / Binance kline interval
  - `--bars 500` (alias `--binance-limit`) number of bars/klines to use (Binance max 1000)
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
  - `--threshold 0.001` direction threshold (fractional deadband)
  - `--method 11` choose `11` (Kalman+LSTM direction-agreement), `10` (Kalman only), `01` (LSTM only)
    - When using `--method 10`, the LSTM is disabled (not trained).
    - When using `--method 01`, the Kalman/predictors are disabled (not trained).
  - `--optimize-operations` optimize `--method` and `--threshold` on the backtest split (uses best combo for the latest signal)
  - `--sweep-threshold` sweep thresholds on the backtest split and pick the best by final equity
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

Endpoints:
- `GET /health`
- `GET /metrics`
- `POST /signal` → returns the latest signal (no orders)
- `POST /trade` → returns the latest signal + attempts an order
- `POST /backtest` → runs a backtest and returns summary metrics
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
  -d '{"binanceSymbol":"BTCUSDT","interval":"1h","bars":200,"method":"10","threshold":0.003838}'
```

Optimize `method` and `threshold` on the backtest split (no orders):
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
  -d '{"binanceSymbol":"BTCUSDT","interval":"1h","bars":200,"method":"10","threshold":0.003838,"orderQuote":20,"binanceLive":false}'
```

Start the live bot (paper mode; no orders):
```
curl -s -X POST http://127.0.0.1:8080/bot/start \
  -H 'Content-Type: application/json' \
  -d '{"binanceSymbol":"BTCUSDT","interval":"5m","bars":500,"method":"11","threshold":0.001,"fee":0.0005,"botOnlineEpochs":1,"botTrade":false}'
```

Auto-optimize after each buy/sell operation:
- Threshold only: add `"sweepThreshold": true`
- Method + threshold: add `"optimizeOperations": true`

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
- `method` is `"11"` (both; direction-agreement gated), `"10"` (Kalman only), or `"01"` (LSTM only).

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

If your backend has `TRADER_API_TOKEN` set, paste the token into the UI’s “API token” field (it sends `Authorization: Bearer <token>`).

The UI also includes a “Live bot” panel to start/stop the continuous loop and visualize each buy/sell operation on the chart.

Assumptions and limitations
---------------------------
- The strategy is intentionally simple (long or flat only) and does not include sizing, risk limits, or robust transaction cost modeling.
- Live order placement does not handle Binance symbol filters (lot size/step size/min notional) and may be rejected by the exchange.
- This code is for experimentation and education only; it is **not** production-ready nor financial advice.
