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
//   // With apiBaseUrl="/api", cross-origin fallbacks are only used in inferred split-host mode
//   // (apiBaseUrlInferred=true) to avoid accidental CORS loops in explicit proxy configs.
//   apiFallbackUrl: "",
//   apiToken: "TRADER_API_TOKEN",
//   timeoutsMs: {
//     // Increase these if slower sync requests (for example Check keys / Open positions)
//     // or large backtests/trades time out in the UI.
//     requestMs: 60_000,
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
    // "-web-" names are ambiguous (for example, "news-web-api" may be a single app name),
    // so only infer when the backend suffix clearly follows this repo's split naming.
    // Plain "*-web" names are treated as ambiguous to avoid rewriting standalone UI apps.
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
    // prefer same-origin /api first and keep the inferred direct API host as fallback.
    apiBaseUrl: "/api",
    apiBaseUrlInferred,
    apiFallbackUrl: inferredDirectApiBaseUrl || "",
    apiToken: "",
    timeoutsMs: { requestMs: 60000, botStatusMs: 120000 },
  };
})();
