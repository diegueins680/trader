import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const env = { ...loadEnv(mode, process.cwd(), ""), ...process.env };
  const apiTarget = env.TRADER_API_TARGET || "http://127.0.0.1:8080";
  const apiToken = (env.TRADER_API_TOKEN || "").trim();
  const proxyTimeoutMs = (() => {
    const raw = env.TRADER_UI_PROXY_TIMEOUT_MS;
    if (!raw) return 30 * 60 * 1000;
    const parsed = Math.trunc(Number(raw));
    if (!Number.isFinite(parsed) || parsed < 1000) return 30 * 60 * 1000;
    return parsed;
  })();

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
