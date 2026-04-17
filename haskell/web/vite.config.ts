import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import { execSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

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

function normalizeCommitHash(raw: string): string {
  const token = raw.trim().split(/\s+/, 1)[0] ?? "";
  if (/^[0-9a-fA-F]{7,64}$/.test(token)) return token.toLowerCase();
  return "";
}

function resolveUiCommit(env: Record<string, string | undefined>): string {
  const fromEnv =
    normalizeCommitHash(env.VITE_GIT_COMMIT ?? "") ||
    normalizeCommitHash(env.TRADER_GIT_COMMIT ?? "") ||
    normalizeCommitHash(env.GIT_COMMIT ?? "") ||
    normalizeCommitHash(env.GITHUB_SHA ?? "");
  if (fromEnv) return fromEnv;

  try {
    return normalizeCommitHash(execSync("git rev-parse HEAD", { stdio: ["ignore", "pipe", "ignore"] }).toString());
  } catch {
    return "";
  }
}

function readPackageVersion(): string {
  try {
    const pkg = JSON.parse(readFileSync(new URL("./package.json", import.meta.url), "utf8")) as { version?: unknown };
    return typeof pkg.version === "string" ? pkg.version.trim() : "";
  } catch {
    return "";
  }
}

function resolveUiVersion(env: Record<string, string | undefined>): string {
  const fromEnv = (env.VITE_UI_VERSION ?? env.TRADER_UI_VERSION ?? env.npm_package_version ?? "").trim();
  return fromEnv || readPackageVersion();
}

export default defineConfig(({ mode }) => {
  const webRoot = dirname(fileURLToPath(import.meta.url));
  const envRoot = resolve(webRoot, "../..");
  const env: Record<string, string | undefined> = {
    ...loadEnv(mode, webRoot, ""),
    ...loadEnv(mode, envRoot, ""),
    ...process.env,
  };
  const apiTarget = env.TRADER_API_TARGET || "http://127.0.0.1:8080";
  const apiToken = (env.TRADER_API_TOKEN || "").trim();
  const proxyTimeoutMs = parseTimeoutMs(env.TRADER_UI_PROXY_TIMEOUT_MS, 30 * 60 * 1000);
  const uiVersion = resolveUiVersion(env);
  const uiCommit = resolveUiCommit(env);

  return {
    plugins: [react()],
    define: {
      __TRADER_API_TARGET__: JSON.stringify(apiTarget),
      __TRADER_UI_VERSION__: JSON.stringify(uiVersion),
      __TRADER_UI_COMMIT__: JSON.stringify(uiCommit),
    },
    build: {
      rollupOptions: {
        external: ["/trader-config.js"],
        output: {
          manualChunks: (id) => {
            if (id.includes("node_modules")) return "vendor";
            return undefined;
          },
        },
      },
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
