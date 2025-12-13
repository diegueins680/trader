Haskell Trading Bot (Kalman + LSTM + Binance)
=============================================

This repository contains a small Haskell trading demo that:
- Predicts the next price using a small **LSTM**, and a **multi-sensor Kalman fusion** layer that combines multiple model outputs into a single latent expected return signal.
- Only trades when Kalman and LSTM **agree on direction** (both predict up, or both predict down).
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

Environment variables:
- `BINANCE_API_KEY`
- `BINANCE_API_SECRET`

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
  - `--binance-api-key KEY` (default: none) or env `BINANCE_API_KEY`
  - `--binance-api-secret SECRET` (default: none) or env `BINANCE_API_SECRET`
  - `--binance-trade` (default: off) place a market order for the latest signal
  - `--binance-live` (default: off) send LIVE orders (otherwise uses `/order/test`)
  - `--order-quote AMOUNT` (default: none) quote amount to spend on BUY (`quoteOrderQty`)
  - `--order-quantity QTY` (default: none) base quantity to trade (`quantity`)

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
  - `--threshold 0.001` trade threshold (fractional edge)
  - `--fee 0.0005` fee applied when switching position

- Metrics
  - `--backtest-ratio 0.2` holdout ratio (last portion of series; avoids lookahead)
  - `--periods-per-year N` (default: inferred from `--interval`)

Tests
-----
```
cd haskell
cabal test
```

Assumptions and limitations
---------------------------
- The strategy is intentionally simple (long or flat only) and does not include sizing, risk limits, or robust transaction cost modeling.
- Live order placement does not handle Binance symbol filters (lot size/step size/min notional) and may be rejected by the exchange.
- This code is for experimentation and education only; it is **not** production-ready nor financial advice.
