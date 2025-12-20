# Changelog
All notable changes to this project will be documented in this file.

## Unreleased
- Web UI: when trading is armed, automatically switches Market to Futures when Positioning is set to Long/Short.
- Web UI: falls back to `GET` for async polling when `POST` hits proxy errors (e.g. 502/503).
- Web UI: avoid optimizer combo apply crashes when compute limits are unavailable.
- Backtests: risk halts now record `MAX_DRAWDOWN`/`MAX_DAILY_LOSS` as trade exit reasons.
- Live bot: risk halts now record `MAX_DRAWDOWN`/`MAX_DAILY_LOSS` exit reasons even if a signal exit coincides.
- Kalman market context now honors small `--kalman-market-top-n` values when enough symbols are available.
- Tuning: sweep/optimization validates prediction lengths before scoring to avoid crashes.
- Normalization: `minmax`/`standard` fall back to no-op when the fit window is empty or only contains non-finite values; `log` requires finite, positive values.
- Web UI: improves async job not found handling with a clearer error after the grace period.
- Web UI: fixes a startup crash when optimizer combos apply before API compute limits are available.
- API: rounds `/binance/keys` test order quantities to the symbol step size to avoid precision errors.
- API: enables async job persistence by default (local `.tmp/async`); set `TRADER_API_ASYNC_DIR` to override or disable.
