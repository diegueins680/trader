// Deploy-time UI configuration.
//
// This file is loaded before the app bundle (see `index.html`). Replace it at deploy time.
//
// Example:
// globalThis.__TRADER_CONFIG__ = {
//   // Use "/api" when CloudFront proxies /api/* to your API origin (deploy-aws-quick.sh enforces /api when a
//   // distribution ID is provided). CloudFront is non-sticky, so keep the backend single-instance unless you
//   // configure shared async storage (TRADER_API_ASYNC_DIR or TRADER_STATE_DIR).
//   // Use "https://your-api-host" for direct API calls when you are not proxying via /api.
//   apiBaseUrl: "/api",
//   // Optional: only set apiFallbackUrl when your API host allows CORS (avoid it when using /api proxies).
//   apiFallbackUrl: "",
//   apiToken: "TRADER_API_TOKEN",
//   timeoutsMs: {
//     // Increase these if large backtests/trades time out in the UI.
//     signalMs: 10 * 60_000,
//     backtestMs: 30 * 60_000,
//     tradeMs: 10 * 60_000,
//     botStartMs: 30 * 60_000,
//   },
// };
(() => {
  const existing = globalThis.__TRADER_CONFIG__;
  if (existing && typeof existing === "object") return;
  globalThis.__TRADER_CONFIG__ = { apiBaseUrl: "/api", apiFallbackUrl: "", apiToken: "", timeoutsMs: {} };
})();
