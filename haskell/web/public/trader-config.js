// Deploy-time UI configuration.
//
// This file is loaded before the app bundle (see `index.html`). Replace it at deploy time.
//
// Example:
// globalThis.__TRADER_CONFIG__ = {
//   apiBaseUrl: "https://your-api-host",
//   apiToken: "TRADER_API_TOKEN",
// };
(() => {
  const existing = globalThis.__TRADER_CONFIG__;
  if (existing && typeof existing === "object") return;
  globalThis.__TRADER_CONFIG__ = { apiBaseUrl: "", apiToken: "" };
})();

