import type { BotStatusSingle, Market, Method } from "../lib/types";
import { fmtNum } from "../lib/format";
import { DATA_LOG_BAR_SERIES_KEYS } from "./constants";

export function normalizeApiBaseUrlInput(raw: string): string {
  const v = raw.trim();
  if (!v) return "";
  if (v.startsWith("/") || /^https?:\/\//i.test(v)) return v;
  if (v.includes("://")) return v;

  const looksLikeHost = v === "localhost" || v.startsWith("localhost:") || v.includes(".") || v.includes(":");
  if (!looksLikeHost) return `/${v}`;

  const slashIdx = v.indexOf("/");
  const authority = slashIdx === -1 ? v : v.slice(0, slashIdx);
  const rest = slashIdx === -1 ? "" : v.slice(slashIdx);

  const lowerAuthority = authority.toLowerCase();
  const isLocal =
    lowerAuthority === "localhost" ||
    lowerAuthority.startsWith("localhost:") ||
    lowerAuthority === "127.0.0.1" ||
    lowerAuthority.startsWith("127.0.0.1:") ||
    lowerAuthority === "0.0.0.0" ||
    lowerAuthority.startsWith("0.0.0.0:") ||
    lowerAuthority === "::1" ||
    lowerAuthority.startsWith("::1:") ||
    lowerAuthority === "[::1]" ||
    lowerAuthority.startsWith("[::1]:");

  const portFromAuthority = () => {
    if (authority.startsWith("[")) {
      const m = authority.match(/^\[[^\]]+\]:(\d{1,5})$/);
      return m ? m[1] : null;
    }
    // For bare IPv6 (multiple ':'), treat as "no port"; port must be provided via brackets.
    if (authority.split(":").length > 2) return null;
    const m = authority.match(/:([0-9]{1,5})$/);
    return m ? m[1] : null;
  };

  const port = portFromAuthority();
  const normalizeAuthority = () => {
    if (authority.startsWith("[")) return authority;
    const parts = authority.split(":");
    if (parts.length <= 2) return authority;

    // Likely IPv6 without brackets; bracketize. If a port was detected (bracketed form), keep it.
    if (port) {
      const host = parts.slice(0, -1).join(":");
      return `[${host}]:${port}`;
    }
    return `[${authority}]`;
  };

  const scheme = isLocal ? "http" : port && port !== "443" ? "http" : "https";
  return `${scheme}://${normalizeAuthority()}${rest}`;
}

export function normalizeSymbolKey(raw: string): string {
  return raw.trim().toUpperCase();
}

export function normalizePositionSide(raw: string | null | undefined): "LONG" | "SHORT" | "BOTH" | null {
  if (!raw) return null;
  const value = raw.trim().toUpperCase();
  if (value === "LONG" || value === "SHORT" || value === "BOTH") return value;
  return null;
}

export function positionSideFromAmount(amount: number): "LONG" | "SHORT" | null {
  if (!Number.isFinite(amount) || amount === 0) return null;
  return amount > 0 ? "LONG" : "SHORT";
}

export function botPositionSide(status: BotStatusSingle): "LONG" | "SHORT" | null {
  const positions = status.running ? status.positions : status.snapshot?.positions;
  if (!positions || positions.length === 0) return null;
  const last = positions[positions.length - 1];
  if (typeof last !== "number" || !Number.isFinite(last) || Math.abs(last) <= 1e-12) return null;
  return last > 0 ? "LONG" : "SHORT";
}

export function botTradeEnabled(status: BotStatusSingle): boolean | null {
  if (status.running) return status.settings?.tradeEnabled ?? null;
  return status.snapshot?.settings?.tradeEnabled ?? null;
}

export type OrphanedPosition<T> = { pos: T; status: BotStatusSingle | null; reason: string };

type OrphanedPositionsOptions = { market?: Market | null };

function botStatusMarket(status: BotStatusSingle): Market | null {
  if (status.running) return status.market;
  return status.market ?? status.snapshot?.market ?? null;
}

function uniqueSides(sides: Array<"LONG" | "SHORT">): Array<"LONG" | "SHORT"> {
  return Array.from(new Set(sides));
}

export function buildOrphanedPositions<T extends { symbol: string; positionAmt: number; positionSide?: string | null }>(
  positions: T[],
  botEntries: Array<{ symbol: string; status: BotStatusSingle }>,
  opts?: OrphanedPositionsOptions,
): Array<OrphanedPosition<T>> {
  if (positions.length === 0) return [];
  const targetMarket = opts?.market ?? null;
  const statusesBySymbol = new Map<string, BotStatusSingle[]>();
  const otherMarketSymbols = new Set<string>();
  for (const entry of botEntries) {
    const market = botStatusMarket(entry.status);
    const key = normalizeSymbolKey(entry.symbol);
    if (targetMarket && market && market !== targetMarket) otherMarketSymbols.add(key);
    if (targetMarket && market !== targetMarket) continue;
    const list = statusesBySymbol.get(key);
    if (list) list.push(entry.status);
    else statusesBySymbol.set(key, [entry.status]);
  }

  return positions
    .map((pos): OrphanedPosition<T> | null => {
      const statuses = statusesBySymbol.get(normalizeSymbolKey(pos.symbol)) ?? [];
      const activeStatuses = statuses.filter((status) => status.running || status.starting === true);
      const activeTradingStatuses = activeStatuses.filter((status) => botTradeEnabled(status) !== false);
      const sideRaw = normalizePositionSide(pos.positionSide);
      const posSide = sideRaw && sideRaw !== "BOTH" ? sideRaw : positionSideFromAmount(pos.positionAmt);
      const activeSides = uniqueSides(
        activeTradingStatuses
          .map((status) => botPositionSide(status))
          .filter((side): side is "LONG" | "SHORT" => Boolean(side)),
      );
      const adopted = posSide != null && activeSides.some((side) => side === posSide);
      if (adopted) return null;
      let reason = "no bot";
      if (targetMarket && statuses.length === 0 && otherMarketSymbols.has(normalizeSymbolKey(pos.symbol))) reason = "market mismatch";
      else if (statuses.length > 0 && activeStatuses.length === 0) reason = "bot stopped";
      else if (statuses.length > 0 && activeStatuses.length > 0 && activeTradingStatuses.length === 0) reason = "trading disabled";
      else if (statuses.length > 0 && activeStatuses.length > 0 && posSide == null) reason = "position side unknown";
      else if (statuses.length > 0 && activeStatuses.length > 0 && activeSides.length === 0) reason = "bot side unknown";
      else if (activeSides.length > 0) reason = `side mismatch (bot ${activeSides.join("/")})`;
      let status: BotStatusSingle | null = null;
      if (activeStatuses.length > 0) {
        const running = activeStatuses.find((entry) => entry.running);
        if (running) status = running;
        else status = activeStatuses[0] ?? null;
      } else if (statuses.length > 0) {
        status = statuses[0] ?? null;
      }
      return { pos, status, reason };
    })
    .filter((entry): entry is OrphanedPosition<T> => Boolean(entry));
}

export function clamp(n: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, n));
}

export function numFromInput(raw: string, fallback: number): number {
  const trimmed = raw.trim();
  if (trimmed === "") return fallback;
  const normalized = (() => {
    if (!trimmed.includes(",")) return trimmed;
    if (trimmed.includes(".")) return trimmed.replace(/,/g, "");
    const parts = trimmed.split(",");
    if (parts.length === 2) {
      const left = parts[0] ?? "";
      const right = parts[1] ?? "";
      const leftDigits = left.replace(/\D/g, "");
      const rightDigits = right.replace(/\D/g, "");
      if (leftDigits === "0") return `${left}.${right}`;
      if (rightDigits.length === 3) return `${left}${right}`;
      return `${left}.${right}`;
    }
    return trimmed.replace(/,/g, "");
  })();
  const n = Number(normalized);
  return Number.isFinite(n) ? n : fallback;
}

export function escapeSingleQuotes(raw: string): string {
  return raw.replaceAll("'", "'\\''");
}

export function firstReason(...reasons: Array<string | null | undefined>): string | null {
  for (const r of reasons) if (r) return r;
  return null;
}

export type RequestIssueDetail = {
  message: string;
  targetId?: string;
  disabledMessage?: string;
};

export type RequestIssueDetailsInput = {
  rateLimitReason?: string | null;
  apiStatusIssue?: string | null;
  apiBlockedReason?: string | null;
  apiTargetId?: string;
  missingSymbol?: boolean;
  symbolError?: string | null;
  symbolTargetId?: string;
  missingInterval?: boolean;
  intervalTargetId?: string;
  lookbackError?: string | null;
  lookbackTargetId?: string;
  apiLimitsReason?: string | null;
  apiLimitsTargetId?: string;
};

export function buildRequestIssueDetails(input: RequestIssueDetailsInput): RequestIssueDetail[] {
  const issues: RequestIssueDetail[] = [];
  if (input.rateLimitReason) issues.push({ message: input.rateLimitReason });
  if (input.apiStatusIssue) {
    issues.push({
      message: input.apiStatusIssue,
      targetId: input.apiTargetId,
      disabledMessage: input.apiBlockedReason ?? input.apiStatusIssue,
    });
  }
  if (input.missingSymbol) issues.push({ message: "Symbol is required.", targetId: input.symbolTargetId });
  else if (input.symbolError) issues.push({ message: input.symbolError, targetId: input.symbolTargetId });
  if (input.missingInterval) issues.push({ message: "Interval is required.", targetId: input.intervalTargetId });
  if (input.lookbackError) issues.push({ message: input.lookbackError, targetId: input.lookbackTargetId });
  if (input.apiLimitsReason) issues.push({ message: input.apiLimitsReason, targetId: input.apiLimitsTargetId });
  return issues;
}

export function isLocalHostname(hostname: string): boolean {
  return hostname === "localhost" || hostname === "127.0.0.1" || hostname === "::1";
}

export function fmtTimeMs(ms: number): string {
  if (!Number.isFinite(ms)) return "—";
  try {
    return new Date(ms).toLocaleString();
  } catch {
    return String(ms);
  }
}

export function fmtDurationMs(ms: number | null | undefined): string {
  if (typeof ms !== "number" || !Number.isFinite(ms)) return "—";
  const sign = ms < 0 ? "-" : "";
  const abs = Math.abs(ms);
  if (abs < 1000) return `${sign}${Math.round(abs)}ms`;
  if (abs < 60_000) return `${sign}${(abs / 1000).toFixed(abs < 10_000 ? 2 : 1)}s`;
  if (abs < 3_600_000) return `${sign}${(abs / 60_000).toFixed(abs < 600_000 ? 2 : 1)}m`;
  return `${sign}${(abs / 3_600_000).toFixed(abs < 36_000_000 ? 2 : 1)}h`;
}

export function fmtEtaMs(ms: number | null | undefined): string {
  if (typeof ms !== "number" || !Number.isFinite(ms)) return "—";
  if (ms >= 0) return `in ${fmtDurationMs(ms)}`;
  return `${fmtDurationMs(-ms)} ago`;
}

export function fmtProfitFactor(pf: number | null | undefined, grossProfit: number, grossLoss: number): string {
  if (typeof pf === "number" && Number.isFinite(pf)) return fmtNum(pf, 3);
  if (grossLoss === 0 && grossProfit > 0) return "∞";
  if (grossLoss === 0 && grossProfit === 0) return "0";
  return "—";
}

function isJsonPrimitive(v: unknown): v is string | number | boolean | null {
  return v === null || typeof v === "string" || typeof v === "number" || typeof v === "boolean";
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function getBacktestStartIndex(data: unknown): number | null {
  if (!isRecord(data)) return null;
  const split = data.split;
  if (!isRecord(split)) return null;
  const raw = split.backtestStartIndex;
  return typeof raw === "number" && Number.isFinite(raw) ? raw : null;
}

export function indexTopLevelPrimitiveArrays(data: unknown): unknown {
  if (!isRecord(data)) return data;
  const barIndexBase = getBacktestStartIndex(data);
  const out: Record<string, unknown> = { ...data };

  for (const [k, v] of Object.entries(out)) {
    if (!Array.isArray(v) || !v.every(isJsonPrimitive)) continue;

    const isBarSeries = barIndexBase !== null && DATA_LOG_BAR_SERIES_KEYS.has(k);
    out[k] = v.map((item, i) => (isBarSeries ? { i, bar: barIndexBase + i, v: item } : { i, v: item }));
  }

  return out;
}

export function errorName(err: unknown): string {
  if (!err || typeof err !== "object" || !("name" in err)) return "";
  return String((err as { name: unknown }).name);
}

export function isAbortError(err: unknown): boolean {
  const name = errorName(err);
  if (name === "AbortError") return true;
  if (!(err instanceof Error)) return false;
  return err.message.toLowerCase().includes("aborted");
}

export function isTimeoutError(err: unknown): boolean {
  return errorName(err) === "TimeoutError";
}

export function actionBadgeClass(action: string): string {
  const a = action.toUpperCase();
  if (a.includes("LONG")) return "badge badgeStrong badgeLong";
  if (a.includes("FLAT")) return "badge badgeStrong badgeFlat";
  return "badge badgeStrong badgeHold";
}

export function methodLabel(method: Method): string {
  switch (method) {
    case "11":
      return "Both (agreement gated)";
    case "10":
      return "Kalman only";
    case "01":
      return "LSTM only";
    case "blend":
      return "Blend (weighted average)";
    case "router":
      return "Router (adaptive)";
    default:
      return "Unknown";
  }
}

export function marketLabel(m: Market): string {
  switch (m) {
    case "spot":
      return "Spot";
    case "margin":
      return "Margin";
    case "futures":
      return "Futures";
    default:
      return "Unknown";
  }
}

export function generateIdempotencyKey(): string {
  try {
    if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
      return crypto.randomUUID();
    }
  } catch {
    // ignore
  }
  const alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-";
  const bytes = new Uint8Array(24);
  try {
    crypto.getRandomValues(bytes);
  } catch {
    for (let i = 0; i < bytes.length; i += 1) bytes[i] = Math.floor(Math.random() * 256);
  }
  let out = "";
  for (let i = 0; i < bytes.length; i += 1) out += alphabet[bytes[i]! % alphabet.length];
  return out;
}

export function isLikelyOrderError(message: string | null | undefined, sent: boolean | null | undefined, status: string | null | undefined): boolean {
  if (sent === false) return true;
  const s = `${status ?? ""} ${message ?? ""}`.toLowerCase();
  return (
    s.includes("error") ||
    s.includes("fail") ||
    s.includes("rejected") ||
    s.includes("insufficient") ||
    s.includes("no order") ||
    s.includes("halt") ||
    s.includes("denied")
  );
}
