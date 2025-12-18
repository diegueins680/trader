export type TraderUiTimeoutsMs = {
  requestMs?: number;
  signalMs?: number;
  backtestMs?: number;
  tradeMs?: number;
  botStartMs?: number;
  botStatusMs?: number;
};

export type TraderUiDeployConfig = {
  apiBaseUrl: string;
  apiToken: string;
  timeoutsMs?: TraderUiTimeoutsMs;
};

function readString(raw: unknown): string {
  return typeof raw === "string" ? raw : "";
}

function readTimeoutMs(raw: unknown): number | undefined {
  if (typeof raw !== "number" || !Number.isFinite(raw)) return;
  const ms = Math.trunc(raw);
  if (ms < 1_000) return;
  return Math.min(ms, 24 * 60 * 60 * 1_000);
}

function readTimeoutsMs(raw: unknown): TraderUiTimeoutsMs | undefined {
  if (!raw || typeof raw !== "object") return;
  const obj = raw as Record<string, unknown>;
  const timeoutsMs: TraderUiTimeoutsMs = {};
  const requestMs = readTimeoutMs(obj.requestMs);
  const signalMs = readTimeoutMs(obj.signalMs);
  const backtestMs = readTimeoutMs(obj.backtestMs);
  const tradeMs = readTimeoutMs(obj.tradeMs);
  const botStartMs = readTimeoutMs(obj.botStartMs);
  const botStatusMs = readTimeoutMs(obj.botStatusMs);
  if (requestMs !== undefined) timeoutsMs.requestMs = requestMs;
  if (signalMs !== undefined) timeoutsMs.signalMs = signalMs;
  if (backtestMs !== undefined) timeoutsMs.backtestMs = backtestMs;
  if (tradeMs !== undefined) timeoutsMs.tradeMs = tradeMs;
  if (botStartMs !== undefined) timeoutsMs.botStartMs = botStartMs;
  if (botStatusMs !== undefined) timeoutsMs.botStatusMs = botStatusMs;
  return Object.keys(timeoutsMs).length ? timeoutsMs : undefined;
}

function readConfigFromGlobal(): TraderUiDeployConfig {
  if (typeof window === "undefined") return { apiBaseUrl: "", apiToken: "" };
  const raw = window.__TRADER_CONFIG__;
  if (!raw || typeof raw !== "object") return { apiBaseUrl: "", apiToken: "" };

  return {
    apiBaseUrl: readString((raw as { apiBaseUrl?: unknown }).apiBaseUrl).trim(),
    apiToken: readString((raw as { apiToken?: unknown }).apiToken).trim(),
    timeoutsMs: readTimeoutsMs((raw as { timeoutsMs?: unknown }).timeoutsMs),
  };
}

export const TRADER_UI_CONFIG: TraderUiDeployConfig = readConfigFromGlobal();
