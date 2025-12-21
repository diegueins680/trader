# Changelog
All notable changes to this project will be documented in this file.

## Unreleased
- Web UI: optimizer combo rows are preview-only with explicit Apply actions plus refresh/apply-top shortcuts.
- Web UI: API panel adds base URL copy + /health open shortcuts; loading a profile clears manual override locks.
- Web UI: add a sticky run bar in the configuration panel showing readiness issues and keeping run actions visible.
- Web UI: when trading is armed, automatically switches Market to Futures when Positioning is set to Long/Short.
- Web UI: falls back to `GET` for async polling when `POST` hits proxy errors (e.g. 502/503).
- Web UI: avoid optimizer combo apply crashes when compute limits are unavailable.
- Backtests: risk halts now record `MAX_DRAWDOWN`/`MAX_DAILY_LOSS` as trade exit reasons.
- Live bot: risk halts now record `MAX_DRAWDOWN`/`MAX_DAILY_LOSS` exit reasons even if a signal exit coincides.
- Kalman market context now honors small `--kalman-market-top-n` values when enough symbols are available.
- Tuning: sweep/optimization validates prediction lengths before scoring to avoid crashes.
- Trading/Tuning: add blend method, min-edge/cost-aware edge gating, max-hold exits, trend/volatility filters, and stress-weighted tune scoring.
- Normalization: `minmax`/`standard` fall back to no-op when the fit window is empty or only contains non-finite values; `log` requires finite, positive values.
- Web UI: improves async job not found handling with a clearer error after the grace period.
- Web UI: fixes a startup crash when optimizer combos apply before API compute limits are available.
- Web UI: show optimizer combo source (API vs static) and last update time.
- Web UI: optimizer combo loads preserve the current positioning unless the combo explicitly specifies one.
- Web UI: hides the Agree overlay for LSTM-only backtests and clarifies bars=0 combo behavior.
- API/Trading: latest signals include `closeDirection`, and live order decisions respect `closeThreshold` for exits.
- Web UI: manual Method/open/close edits are preserved when optimizer combos or optimize/sweep results apply.
- Web UI: optimizer combos now persist/show the operations that produced each top result on hover.
- Web UI: add auto-apply toggle with last-applied marker, manual override lock/unlock hints, and a cross-origin API base warning.
- Optimizer: adds a `--quality` preset plus CSV high/low auto-detection for deeper equity searches.
- Optimizer: adds trade-quality filters (win rate, profit factor, exposure) and samples min-hold/cooldown bars for churn control.
- Deploy: when using CloudFront with a distribution ID, the quick AWS deploy script sets the UI API base to `/api` to avoid CORS.
- Deploy: quick AWS deploy now prints the CloudFront domain and warns when `/api/*` behavior is missing.
- API: rounds `/binance/keys` test order quantities to the symbol step size to avoid precision errors.
- API/UI: `/binance/keys` trade probes report `skipped` when no test order is attempted, and the UI shows `SKIP` instead of `FAIL`.
- API: `/binance/keys` preflight checks minNotional when price data is available and only fetches symbol filters when needed.
- API: enables async job persistence by default (local `.tmp/async`); set `TRADER_API_ASYNC_DIR` to override or disable.
