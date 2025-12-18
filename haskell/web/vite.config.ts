import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

function parseTimeoutMs(raw: unknown, fallback: number): number {
  const n =
    typeof raw === "number"
      ? raw
      : typeof raw === "string"
        ? Number(raw.trim())
        : Number.NaN;
  if (!Number.isFinite(n) || n < 1000) return fallback;
  return Math.min(Math.round(n), 24 * 60 * 60 * 1000);
}

export default defineConfig(({ mode }) => {
  const env = { ...loadEnv(mode, process.cwd(), ""), ...process.env };
  const apiTarget = env.TRADER_API_TARGET || "http://127.0.0.1:8080";
  const apiToken = (env.TRADER_API_TOKEN || "").trim();
  const proxyTimeoutMs = parseTimeoutMs(env.TRADER_UI_PROXY_TIMEOUT_MS, 30 * 60 * 1000);

  return {
    plugins: [react()],
    define: {
      __TRADER_API_TARGET__: JSON.stringify(apiTarget),
    },
    server: {
      proxy: {
        "/api": {
          target: apiTarget,
          changeOrigin: true,
          timeout: proxyTimeoutMs,
          proxyTimeout: proxyTimeoutMs,
          rewrite: (path) => path.replace(/^\/api/, ""),
          configure: (proxy) => {
            if (!apiToken) return;
            proxy.on("proxyReq", (proxyReq) => {
              const hasAuth = Boolean(proxyReq.getHeader("authorization"));
              const hasApiKey = Boolean(proxyReq.getHeader("x-api-key"));
              if (!hasAuth && !hasApiKey) proxyReq.setHeader("Authorization", `Bearer ${apiToken}`);
            });
          },
        },
      },
    },
  };
});
