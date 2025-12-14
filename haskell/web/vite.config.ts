import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const env = { ...loadEnv(mode, process.cwd(), ""), ...process.env };
  const apiTarget = env.TRADER_API_TARGET || "http://127.0.0.1:8080";
  const apiToken = (env.TRADER_API_TOKEN || "").trim();

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
