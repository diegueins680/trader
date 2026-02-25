// Deploy-time UI configuration.
//
// This file is loaded before the app bundle (see `index.html`). Replace it at deploy time.
//
// Example:
// globalThis.__TRADER_CONFIG__ = {
//   // Use "/api" when Fly's proxy routes /api/* to your API origin.
//   // Fly Proxy is non-sticky, so keep the backend single-instance unless you
//   // configure shared async storage (TRADER_API_ASYNC_DIR or TRADER_STATE_DIR).
//   // Use "https://your-api-host" for direct API calls when you are not proxying via /api.
//   apiBaseUrl: "/api",
//   // Optional: set apiFallbackUrl to "/api" for same-origin fallback when the UI uses direct API calls.
//   // Cross-origin fallbacks are ignored when apiBaseUrl is "/api" (proxy mode) to avoid CORS loops;
//   // use apiBaseUrl=https://<api-host> if you want cross-origin failover.
//   apiFallbackUrl: "",
//   apiToken: "TRADER_API_TOKEN",
//   timeoutsMs: {
//     // Increase these if large backtests/trades time out in the UI.
//     requestMs: 30_000,
//     signalMs: 10 * 60_000,
//     backtestMs: 30 * 60_000,
//     tradeMs: 10 * 60_000,
//     botStartMs: 30 * 60_000,
//     botStatusMs: 60_000,
//   },
// };
(() => {
  const inferFlyApiAppName = (appName) => {
    if (!appName) return "";
    if (appName.endsWith("-web")) {
      return appName.slice(0, -4);
    }

    // "-web-" names are ambiguous (for example, "news-web-api" may be a single app name),
    // so only infer when the backend suffix clearly follows this repo's split naming.
    const marker = "-web-";
    const markerAt = appName.lastIndexOf(marker);
    if (markerAt <= 0) return "";
    const prefix = appName.slice(0, markerAt);
    const suffix = appName.slice(markerAt + marker.length);
    if (!suffix) return "";
    if (!/^hs(?:-[a-z0-9]+)*$/.test(suffix)) return "";
    return `${prefix}-${suffix}`;
  };

  const inferFlyDirectApiBaseUrl = () => {
    if (typeof window === "undefined") return "";
    const host = window.location.hostname.trim().toLowerCase();
    if (!host.endsWith(".fly.dev")) return "";
    const labels = host.split(".");
    const appName = labels[0] ?? "";
    const inferredAppName = inferFlyApiAppName(appName);
    if (!inferredAppName || inferredAppName === appName) return "";
    labels[0] = inferredAppName;
    return `https://${labels.join(".")}`;
  };

  const existing = globalThis.__TRADER_CONFIG__;
  if (existing && typeof existing === "object") return;
  const inferredDirectApiBaseUrl = inferFlyDirectApiBaseUrl();
  const apiBaseUrlInferred = Boolean(inferredDirectApiBaseUrl);
  globalThis.__TRADER_CONFIG__ = {
    // For split Fly apps (for example, trader-web-hs.fly.dev + trader-hs.fly.dev),
    // infer the direct API host and fall back to /api if it is unavailable.
    apiBaseUrl: inferredDirectApiBaseUrl || "/api",
    apiBaseUrlInferred,
    apiFallbackUrl: inferredDirectApiBaseUrl ? "/api" : "",
    apiToken: "",
    timeoutsMs: { botStatusMs: 120000 },
  };
})();
