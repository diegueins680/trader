import type {
  ApiError,
  ApiBinanceClosePositionRequest,
  ApiBinancePositionsRequest,
  ApiBinancePositionsResponse,
  ApiRequestProgressStatus,
  ApiBinanceTradesRequest,
  ApiBinanceTradesResponse,
  ApiOrderResult,
  ApiParams,
  ApiTradeResponse,
  BacktestResponse,
  BinanceKeysStatus,
  BinanceListenKeyKeepAliveResponse,
  BinanceListenKeyResponse,
  BotStatus,
  CoinbaseKeysStatus,
  LatestSignal,
  OpsPerformanceResponse,
  OpsResponse,
  OptimizerRunRequest,
  OptimizerRunResponse,
  StateSyncImportResponse,
  StateSyncPayload,
} from "./types";
import { METHOD_IDS } from "../app/contracts";
import { TRADER_UI_CONFIG } from "./deployConfig";
import { readJson, writeJson } from "./storage";

export class HttpError extends Error {
  readonly status: number;
  readonly payload?: unknown;
  readonly retryAfterMs: number | null;

  constructor(status: number, message: string, payload?: unknown, retryAfterMs?: number | null) {
    super(message);
    this.name = "HttpError";
    this.status = status;
    this.payload = payload;
    this.retryAfterMs = typeof retryAfterMs === "number" && Number.isFinite(retryAfterMs) ? retryAfterMs : null;
  }
}

export class AsyncEndpointNotFoundError extends Error {
  readonly httpError: HttpError;

  constructor(httpError: HttpError) {
    super(httpError.message);
    this.name = "AsyncEndpointNotFoundError";
    this.httpError = httpError;
  }
}

export class UnexpectedResponseError extends Error {
  readonly status: number;
  readonly contentType: string;
  readonly bodySnippet: string;

  constructor(status: number, contentType: string, bodySnippet: string) {
    const label = contentType ? contentType.split(";")[0]?.trim() ?? "unknown" : "unknown";
    super(`Unexpected non-JSON response (${label}). Check your API base or /api proxy.`);
    this.name = "UnexpectedResponseError";
    this.status = status;
    this.contentType = contentType;
    this.bodySnippet = bodySnippet;
  }
}

export class InvalidApiResponseError extends Error {
  readonly endpoint: string;
  readonly field: string;

  constructor(endpoint: string, field: string) {
    super(`Invalid ${endpoint} response: ${field}`);
    this.name = "InvalidApiResponseError";
    this.endpoint = endpoint;
    this.field = field;
  }
}

type FetchJsonOptions = {
  signal?: AbortSignal;
  timeoutMs?: number;
  headers?: Record<string, string>;
  allowFallback?: boolean;
  onTransientRetry?: (info: TransientRetryInfo) => void;
};

type TransientRetryInfo = {
  attempt: number;
  maxRetries: number;
  error: unknown;
  delayMs: number;
  retryAfterMs: number | null;
};

type AsyncStartResponse = { jobId: string };
type AsyncPollResponse<T> = { status: "running" | "done" | "error"; result?: T; error?: string };
export type HealthResponse = {
  status: "ok";
  version?: string;
  commit?: string;
  authRequired?: boolean;
  authOk?: boolean;
  computeLimits?: { maxBarsLstm: number; maxEpochs: number; maxHiddenSize: number };
  asyncJobs?: { maxRunning: number; ttlMs: number; persistence: boolean };
  cache?: { enabled: boolean; ttlMs: number; maxEntries: number };
};

export type CacheStatsResponse = {
  enabled: boolean;
  ttlMs: number;
  maxEntries: number;
  signals: { entries: number; hits: number; misses: number };
  backtests: { entries: number; hits: number; misses: number };
  marketData?: Record<string, { entries: number; maxEntries: number | null }>;
  atMs: number;
};

export type CacheClearResponse = { ok: boolean; atMs: number };

type UnknownRecord = Record<string, unknown>;
const MAX_BOT_STATUS_POINTS = 100_000;

type BotStatusDecodeExpectations = {
  tenantKey?: string;
  symbol?: string;
  allowMultiForExpectedSymbol?: boolean;
};

function isUnknownRecord(value: unknown): value is UnknownRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function invalidBotStatus(endpoint: string, field: string): never {
  throw new InvalidApiResponseError(endpoint, field);
}

function requireBotRecord(value: unknown, endpoint: string, field: string): UnknownRecord {
  if (!isUnknownRecord(value)) invalidBotStatus(endpoint, `${field} must be an object`);
  return value;
}

function requireBotBoolean(value: unknown, endpoint: string, field: string): boolean {
  if (typeof value !== "boolean") invalidBotStatus(endpoint, `${field} must be a boolean`);
  return value;
}

function requireBotString(value: unknown, endpoint: string, field: string): string {
  if (typeof value !== "string" || !value.trim()) invalidBotStatus(endpoint, `${field} must be a non-empty string`);
  return value;
}

function requireBotFinite(value: unknown, endpoint: string, field: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    invalidBotStatus(endpoint, `${field} must be a finite number`);
  }
  return value;
}

function requireBotSafeInteger(value: unknown, endpoint: string, field: string): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value)) {
    invalidBotStatus(endpoint, `${field} must be a safe integer`);
  }
  return value;
}

function validateOptionalBotBoolean(record: UnknownRecord, key: string, endpoint: string, field: string): void {
  if (record[key] !== undefined) requireBotBoolean(record[key], endpoint, `${field}.${key}`);
}

function validateOptionalBotFinite(record: UnknownRecord, key: string, endpoint: string, field: string): void {
  if (record[key] !== undefined) requireBotFinite(record[key], endpoint, `${field}.${key}`);
}

function validateOptionalBotSafeInteger(record: UnknownRecord, key: string, endpoint: string, field: string): void {
  if (record[key] !== undefined) requireBotSafeInteger(record[key], endpoint, `${field}.${key}`);
}

function requireBotArray(record: UnknownRecord, key: string, endpoint: string, field: string): unknown[] {
  const value = record[key];
  if (!Array.isArray(value)) invalidBotStatus(endpoint, `${field}.${key} must be an array`);
  return value;
}

function validateFiniteBotArray(values: unknown[], endpoint: string, field: string, nullable = false): void {
  for (let i = 0; i < values.length; i += 1) {
    if (nullable && values[i] === null) continue;
    requireBotFinite(values[i], endpoint, `${field}[${i}]`);
  }
}

function validateSafeIntegerBotArray(values: unknown[], endpoint: string, field: string): void {
  for (let i = 0; i < values.length; i += 1) {
    requireBotSafeInteger(values[i], endpoint, `${field}[${i}]`);
  }
}

function validatePositionBotArray(values: unknown[], endpoint: string, field: string): void {
  for (let i = 0; i < values.length; i += 1) {
    const position = requireBotSafeInteger(values[i], endpoint, `${field}[${i}]`);
    if (position !== -1 && position !== 0 && position !== 1) {
      invalidBotStatus(endpoint, `${field}[${i}] must be -1, 0, or 1`);
    }
  }
}

function validateBoundedBotSeries(values: unknown[], endpoint: string, field: string): void {
  if (values.length > MAX_BOT_STATUS_POINTS) {
    invalidBotStatus(endpoint, `${field} exceeds the supported status limit`);
  }
}

function normalizeExpectedBotIdentity(value: string | undefined): string | undefined {
  const normalized = value?.trim();
  return normalized ? normalized : undefined;
}

function normalizeBotSymbol(value: string): string {
  return value.trim().toUpperCase();
}

function validateBotTenantKey(
  record: UnknownRecord,
  endpoint: string,
  field: string,
  expectedTenantKey: string | undefined,
  required: boolean,
): string | undefined {
  if (record.tenantKey === undefined && !required) return undefined;
  const tenantKey = requireBotString(record.tenantKey, endpoint, `${field}.tenantKey`);
  if (tenantKey !== tenantKey.trim()) invalidBotStatus(endpoint, `${field}.tenantKey must be normalized`);
  if (expectedTenantKey !== undefined && tenantKey !== expectedTenantKey) {
    invalidBotStatus(endpoint, `${field}.tenantKey does not match the requested tenant`);
  }
  return tenantKey;
}

function validateBotSymbol(
  value: unknown,
  endpoint: string,
  field: string,
  expectedSymbol: string | undefined,
): string {
  const symbol = requireBotString(value, endpoint, field);
  if (expectedSymbol !== undefined && normalizeBotSymbol(symbol) !== normalizeBotSymbol(expectedSymbol)) {
    invalidBotStatus(endpoint, `${field} does not match the requested symbol`);
  }
  return symbol;
}

function validateAlignedBotSeries(series: Array<{ key: string; values: unknown[] }>, endpoint: string, field: string): void {
  const [first, ...rest] = series;
  if (!first) invalidBotStatus(endpoint, `${field}.prices must not be empty`);
  const expectedLength = first.values.length;
  if (expectedLength === 0) invalidBotStatus(endpoint, `${field}.${first.key} must not be empty`);
  for (const item of rest) {
    if (item.values.length !== expectedLength) {
      invalidBotStatus(endpoint, `${field}.${item.key} must align with ${field}.${first.key}`);
    }
  }
}

function validateBotRunning(
  record: UnknownRecord,
  endpoint: string,
  field: string,
  expectations: BotStatusDecodeExpectations,
): void {
  validateBotTenantKey(record, endpoint, field, expectations.tenantKey, true);
  requireBotBoolean(record.live, endpoint, `${field}.live`);
  validateBotSymbol(record.symbol, endpoint, `${field}.symbol`, expectations.symbol);
  requireBotString(record.interval, endpoint, `${field}.interval`);
  if (record.market !== "spot" && record.market !== "margin" && record.market !== "futures") {
    invalidBotStatus(endpoint, `${field}.market is unsupported`);
  }
  if (typeof record.method !== "string" || !(METHOD_IDS as readonly string[]).includes(record.method)) {
    invalidBotStatus(endpoint, `${field}.method is unsupported`);
  }
  requireBotFinite(record.threshold, endpoint, `${field}.threshold`);
  validateOptionalBotFinite(record, "openThreshold", endpoint, field);
  validateOptionalBotFinite(record, "closeThreshold", endpoint, field);

  const settings = requireBotRecord(record.settings, endpoint, `${field}.settings`);
  requireBotBoolean(settings.tradeEnabled, endpoint, `${field}.settings.tradeEnabled`);
  const maxPoints = requireBotSafeInteger(settings.maxPoints, endpoint, `${field}.settings.maxPoints`);
  if (maxPoints < 100 || maxPoints > MAX_BOT_STATUS_POINTS) {
    invalidBotStatus(endpoint, `${field}.settings.maxPoints is outside the supported range`);
  }
  validateOptionalBotBoolean(settings, "protectionOrders", endpoint, `${field}.settings`);
  validateOptionalBotBoolean(settings, "adoptExistingPosition", endpoint, `${field}.settings`);

  const halted = requireBotBoolean(record.halted, endpoint, `${field}.halted`);
  requireBotFinite(record.peakEquity, endpoint, `${field}.peakEquity`);
  requireBotFinite(record.dayStartEquity, endpoint, `${field}.dayStartEquity`);
  requireBotSafeInteger(record.consecutiveOrderErrors, endpoint, `${field}.consecutiveOrderErrors`);
  requireBotSafeInteger(record.startIndex, endpoint, `${field}.startIndex`);
  requireBotSafeInteger(record.startedAtMs, endpoint, `${field}.startedAtMs`);
  requireBotSafeInteger(record.updatedAtMs, endpoint, `${field}.updatedAtMs`);
  validateOptionalBotSafeInteger(record, "polledAtMs", endpoint, field);
  validateOptionalBotSafeInteger(record, "haltedAtMs", endpoint, field);

  const prices = requireBotArray(record, "prices", endpoint, field);
  const openTimes = requireBotArray(record, "openTimes", endpoint, field);
  const kalmanPredNext = requireBotArray(record, "kalmanPredNext", endpoint, field);
  const lstmPredNext = requireBotArray(record, "lstmPredNext", endpoint, field);
  const equityCurve = requireBotArray(record, "equityCurve", endpoint, field);
  const positions = requireBotArray(record, "positions", endpoint, field);
  requireBotArray(record, "operations", endpoint, field);
  requireBotArray(record, "orders", endpoint, field);
  requireBotArray(record, "trades", endpoint, field);

  const coreSeries = [
    { key: "prices", values: prices },
    { key: "openTimes", values: openTimes },
    { key: "kalmanPredNext", values: kalmanPredNext },
    { key: "lstmPredNext", values: lstmPredNext },
    { key: "equityCurve", values: equityCurve },
    { key: "positions", values: positions },
  ];
  validateAlignedBotSeries(coreSeries, endpoint, field);

  for (const item of coreSeries) validateBoundedBotSeries(item.values, endpoint, `${field}.${item.key}`);
  validateFiniteBotArray(prices, endpoint, `${field}.prices`);
  validateSafeIntegerBotArray(openTimes, endpoint, `${field}.openTimes`);
  validateFiniteBotArray(kalmanPredNext, endpoint, `${field}.kalmanPredNext`, true);
  validateFiniteBotArray(lstmPredNext, endpoint, `${field}.lstmPredNext`, true);
  validateFiniteBotArray(equityCurve, endpoint, `${field}.equityCurve`);
  validatePositionBotArray(positions, endpoint, `${field}.positions`);

  if (record.latestSignal !== undefined && record.latestSignal !== null && !isUnknownRecord(record.latestSignal)) {
    invalidBotStatus(endpoint, `${field}.latestSignal must be an object when present`);
  }
  if (record.error !== undefined && typeof record.error !== "string") {
    invalidBotStatus(endpoint, `${field}.error must be a string when present`);
  }
  if (halted) {
    requireBotString(record.haltReason, endpoint, `${field}.haltReason`);
  } else if (record.haltReason !== undefined) {
    invalidBotStatus(endpoint, `${field}.haltReason requires halted=true`);
  }
}

function validateBotStopped(
  record: UnknownRecord,
  endpoint: string,
  field: string,
  expectations: BotStatusDecodeExpectations,
): void {
  const starting = record.starting === undefined ? false : requireBotBoolean(record.starting, endpoint, `${field}.starting`);
  const symbol =
    record.symbol === undefined
      ? undefined
      : validateBotSymbol(record.symbol, endpoint, `${field}.symbol`, expectations.symbol);
  const tenantKey = validateBotTenantKey(record, endpoint, field, expectations.tenantKey, starting);
  if (record.interval !== undefined) requireBotString(record.interval, endpoint, `${field}.interval`);
  if (record.market !== undefined && record.market !== "spot" && record.market !== "margin" && record.market !== "futures") {
    invalidBotStatus(endpoint, `${field}.market is unsupported`);
  }
  if (record.method !== undefined && (typeof record.method !== "string" || !(METHOD_IDS as readonly string[]).includes(record.method))) {
    invalidBotStatus(endpoint, `${field}.method is unsupported`);
  }
  validateOptionalBotFinite(record, "threshold", endpoint, field);
  validateOptionalBotFinite(record, "openThreshold", endpoint, field);
  validateOptionalBotFinite(record, "closeThreshold", endpoint, field);
  validateOptionalBotSafeInteger(record, "startedAtMs", endpoint, field);

  if (starting) {
    if (symbol === undefined) invalidBotStatus(endpoint, `${field}.symbol is required while starting`);
    requireBotString(record.interval, endpoint, `${field}.interval`);
    if (record.market !== "spot" && record.market !== "margin" && record.market !== "futures") {
      invalidBotStatus(endpoint, `${field}.market is required while starting`);
    }
    if (typeof record.method !== "string" || !(METHOD_IDS as readonly string[]).includes(record.method)) {
      invalidBotStatus(endpoint, `${field}.method is required while starting`);
    }
    requireBotSafeInteger(record.startedAtMs, endpoint, `${field}.startedAtMs`);
  }

  if (record.snapshot !== undefined) {
    const snapshot = requireBotRecord(record.snapshot, endpoint, `${field}.snapshot`);
    if ("multi" in snapshot || "bots" in snapshot) {
      invalidBotStatus(endpoint, `${field}.snapshot must be a single-bot status`);
    }
    if (requireBotBoolean(snapshot.running, endpoint, `${field}.snapshot.running`) !== true) {
      invalidBotStatus(endpoint, `${field}.snapshot.running must be true`);
    }
    validateBotRunning(snapshot, endpoint, `${field}.snapshot`, {
      tenantKey: expectations.tenantKey ?? tenantKey,
      symbol: expectations.symbol ?? symbol,
    });
    requireBotSafeInteger(record.snapshotAtMs, endpoint, `${field}.snapshotAtMs`);
  } else if (record.snapshotAtMs !== undefined) {
    invalidBotStatus(endpoint, `${field}.snapshotAtMs requires snapshot`);
  }
  if (record.error !== undefined && typeof record.error !== "string") {
    invalidBotStatus(endpoint, `${field}.error must be a string when present`);
  }
}

function validateBotSingle(
  value: unknown,
  endpoint: string,
  field: string,
  expectations: BotStatusDecodeExpectations,
): UnknownRecord {
  const record = requireBotRecord(value, endpoint, field);
  if ("multi" in record || "bots" in record) {
    invalidBotStatus(endpoint, `${field} must not contain multi-bot fields`);
  }
  const running = requireBotBoolean(record.running, endpoint, `${field}.running`);
  if (running) validateBotRunning(record, endpoint, field, expectations);
  else validateBotStopped(record, endpoint, field, expectations);
  return record;
}

function validateBotMessageArray(record: UnknownRecord, key: "errors" | "queued", endpoint: string): number {
  if (record[key] === undefined) return 0;
  const values = requireBotArray(record, key, endpoint, "status");
  const messageKey = key === "errors" ? "error" : "message";
  for (let i = 0; i < values.length; i += 1) {
    const item = requireBotRecord(values[i], endpoint, `status.${key}[${i}]`);
    requireBotString(item.symbol, endpoint, `status.${key}[${i}].symbol`);
    requireBotString(item[messageKey], endpoint, `status.${key}[${i}].${messageKey}`);
  }
  return values.length;
}

/**
 * Decodes the safety-critical BotStatus envelope before it can update UI state.
 * Lower-risk telemetry fields remain forward-compatible, while running/live,
 * arming, halt, identity, and position evidence fail closed on malformed input.
 */
export function decodeBotStatus(
  payload: unknown,
  endpoint = "/bot/status",
  expectations: BotStatusDecodeExpectations = {},
): BotStatus {
  const normalizedExpectations = {
    ...expectations,
    tenantKey: normalizeExpectedBotIdentity(expectations.tenantKey),
    symbol: normalizeExpectedBotIdentity(expectations.symbol),
  };
  const record = requireBotRecord(payload, endpoint, "status");
  if (!("multi" in record) && !("bots" in record)) {
    return validateBotSingle(record, endpoint, "status", normalizedExpectations) as BotStatus;
  }

  if (normalizedExpectations.symbol !== undefined && !normalizedExpectations.allowMultiForExpectedSymbol) {
    invalidBotStatus(endpoint, "status must be single-bot for the requested symbol");
  }
  if (record.multi !== true) invalidBotStatus(endpoint, "status.multi must be true");
  const running = requireBotBoolean(record.running, endpoint, "status.running");
  const botsRaw = requireBotArray(record, "bots", endpoint, "status");
  const bots = botsRaw.map((bot, index) =>
    validateBotSingle(bot, endpoint, `status.bots[${index}]`, { tenantKey: normalizedExpectations.tenantKey }),
  );
  validateBotMessageArray(record, "errors", endpoint);
  const queuedCount = validateBotMessageArray(record, "queued", endpoint);
  const derivedRunning = bots.some((bot) => bot.running === true);
  const derivedStarting = queuedCount > 0 || bots.some((bot) => bot.running === false && bot.starting === true);
  if (running !== derivedRunning) invalidBotStatus(endpoint, "status.running disagrees with status.bots");
  if (record.starting !== undefined) {
    const starting = requireBotBoolean(record.starting, endpoint, "status.starting");
    if (starting !== derivedStarting) invalidBotStatus(endpoint, "status.starting disagrees with status.bots");
  } else if (derivedStarting) {
    invalidBotStatus(endpoint, "status.starting is required when a bot is starting");
  }
  validateOptionalBotSafeInteger(record, "snapshotAtMs", endpoint, "status");
  return record as BotStatus;
}

type AsyncJobOptions = FetchJsonOptions & {
  onJobId?: (jobId: string) => void;
  retryStart?: boolean;
  maxStartRetries?: number;
};

type ResolvedPath = {
  pathname: string;
  searchParams: URLSearchParams;
  hash: string;
};

function parseResolvedPath(path: string): ResolvedPath {
  const raw = path.startsWith("/") ? path : `/${path}`;
  const hashIndex = raw.indexOf("#");
  const rawNoHash = hashIndex >= 0 ? raw.slice(0, hashIndex) : raw;
  const hash = hashIndex >= 0 ? raw.slice(hashIndex) : "";
  const queryIndex = rawNoHash.indexOf("?");
  const pathname = queryIndex >= 0 ? rawNoHash.slice(0, queryIndex) : rawNoHash;
  const search = queryIndex >= 0 ? rawNoHash.slice(queryIndex + 1) : "";
  return { pathname, searchParams: new URLSearchParams(search), hash };
}

function mergeSearchParams(baseSearch: string, pathSearchParams: URLSearchParams): string {
  const merged = new URLSearchParams(baseSearch);
  const overriddenKeys = new Set<string>();
  for (const key of pathSearchParams.keys()) overriddenKeys.add(key);
  for (const key of overriddenKeys) merged.delete(key);
  for (const [key, value] of pathSearchParams.entries()) merged.append(key, value);
  const search = merged.toString();
  return search ? `?${search}` : "";
}

function resolveUrl(baseUrl: string, path: string): string {
  const base = baseUrl.trim();
  const { pathname, searchParams, hash } = parseResolvedPath(path);
  const search = searchParams.toString();

  if (!base || base === "/") {
    return `${pathname}${search ? `?${search}` : ""}${hash}`;
  }

  const normalizedBase = /^https?:\/\//.test(base) || base.startsWith("/") ? base : `/${base}`;
  const url = new URL(normalizedBase, "https://trader.invalid");
  const basePath = url.pathname.replace(/\/+$/, "");
  const resolvedPath = `${basePath}${pathname}`.replace(/\/{2,}/g, "/") || "/";
  const mergedSearch = mergeSearchParams(url.search, searchParams);

  if (/^https?:\/\//.test(normalizedBase)) {
    url.pathname = resolvedPath;
    url.search = mergedSearch;
    url.hash = hash;
    return url.toString();
  }

  return `${resolvedPath}${mergedSearch}${hash}`;
}

function normalizeBaseUrl(raw: string): string {
  const trimmed = raw.trim();
  if (!trimmed) return "";
  const withoutTrailingSlashes = trimmed.replace(/\/+$/, "");
  // Preserve same-origin root-path identity so fallback policy can still
  // distinguish "/" from a direct-host base.
  return withoutTrailingSlashes || (trimmed.startsWith("/") ? "/" : "");
}

const TENANT_HEADER = "X-Tenant-Key";
export const REQUEST_PROGRESS_HEADER = "X-Trader-Request-Id";

function normalizeExactIntegerQueryParam(raw: unknown): number | null {
  if (typeof raw !== "number" || !Number.isSafeInteger(raw)) return null;
  return Object.is(raw, -0) ? 0 : raw;
}

function normalizeTenantKeyValue(raw: unknown): string | null {
  if (typeof raw !== "string") return null;
  const trimmed = raw.trim();
  return trimmed ? trimmed : null;
}

function tenantKeyFromPath(path: string): string | null {
  const queryIndex = path.indexOf("?");
  if (queryIndex < 0) return null;
  const hashIndex = path.indexOf("#", queryIndex);
  const rawQuery = hashIndex >= 0 ? path.slice(queryIndex + 1, hashIndex) : path.slice(queryIndex + 1);
  if (!rawQuery) return null;
  try {
    const query = new URLSearchParams(rawQuery);
    return normalizeTenantKeyValue(query.get("tenantKey"));
  } catch {
    return null;
  }
}

function pathWithoutTenantKey(path: string): string {
  const hashIndex = path.indexOf("#");
  const beforeHash = hashIndex >= 0 ? path.slice(0, hashIndex) : path;
  const hash = hashIndex >= 0 ? path.slice(hashIndex) : "";
  const queryIndex = beforeHash.indexOf("?");
  if (queryIndex < 0) return path;
  const pathname = beforeHash.slice(0, queryIndex);
  const query = new URLSearchParams(beforeHash.slice(queryIndex + 1));
  query.delete("tenantKey");
  const search = query.toString();
  return `${pathname}${search ? `?${search}` : ""}${hash}`;
}

export function tenantKeyFromBody(body: BodyInit | null | undefined): string | null {
  if (!body) return null;
  if (typeof body === "string") {
    const trimmed = body.trim();
    if (!trimmed) return null;
    try {
      const parsed: unknown = JSON.parse(trimmed);
      if (!parsed || typeof parsed !== "object") return null;
      return normalizeTenantKeyValue((parsed as { tenantKey?: unknown }).tenantKey);
    } catch {
      return null;
    }
  }
  if (typeof URLSearchParams !== "undefined" && body instanceof URLSearchParams) {
    return normalizeTenantKeyValue(body.get("tenantKey"));
  }
  if (typeof FormData !== "undefined" && body instanceof FormData) {
    const tenantKey = body.get("tenantKey");
    return typeof tenantKey === "string" ? normalizeTenantKeyValue(tenantKey) : null;
  }
  return null;
}

export function withTenantHeader(
  headers: Headers,
  path: string,
  body: BodyInit | null | undefined,
  allowTenantHeader = true,
): Headers {
  if (!allowTenantHeader) return headers;
  if (headers.has(TENANT_HEADER)) return headers;
  const tenantKey = tenantKeyFromPath(path) ?? tenantKeyFromBody(body);
  if (tenantKey) headers.set(TENANT_HEADER, tenantKey);
  return headers;
}

function requestHasAuthLikeContext(path: string, body: BodyInit | null | undefined, headersInit: HeadersInit | undefined): boolean {
  const headers = new Headers(headersInit);
  if (headers.has("Authorization") || headers.has("X-API-Key") || headers.has(TENANT_HEADER)) return true;
  return Boolean(tenantKeyFromPath(path) ?? tenantKeyFromBody(body));
}

function shouldAttachTenantHeader(requestUrl: string, method: string): boolean {
  const requestMethod = method.trim().toUpperCase() || "GET";
  if (typeof window === "undefined") return true;
  try {
    const resolved = new URL(requestUrl, window.location.origin);
    if (resolved.origin === window.location.origin) return true;
    // Keep public cross-origin GET/HEAD requests header-free. Calls with
    // tenant context override this and send X-Tenant-Key instead of a URL query.
    return requestMethod !== "GET" && requestMethod !== "HEAD";
  } catch {
    return true;
  }
}

// v4 drops legacy persisted fallback preferences so older auth-driven entries
// cannot override the current explicit-host fallback rules after upgrades.
const FALLBACK_STORAGE_KEY = "trader_api_fallback_v4";
const FALLBACK_STORAGE_TTL_MS = 12 * 60 * 60 * 1000;

type FallbackStorage = {
  savedAtMs: number;
  blocked: string[];
  preferred: Record<string, string>;
};

function emptyFallbackStorage(): FallbackStorage {
  return { savedAtMs: 0, blocked: [], preferred: {} };
}

function loadFallbackStorage(): FallbackStorage {
  const raw = readJson<FallbackStorage>(FALLBACK_STORAGE_KEY);
  if (!raw || typeof raw !== "object") return emptyFallbackStorage();
  const savedAtMs = typeof raw.savedAtMs === "number" && Number.isFinite(raw.savedAtMs) ? raw.savedAtMs : 0;
  const nowMs = Date.now();
  if (!savedAtMs || savedAtMs > nowMs || nowMs - savedAtMs > FALLBACK_STORAGE_TTL_MS) return emptyFallbackStorage();
  const blocked = Array.isArray(raw.blocked)
    ? raw.blocked
        .filter((entry): entry is string => typeof entry === "string")
        .map((entry) => normalizeBaseUrl(entry))
        .filter(Boolean)
    : [];
  const preferredEntries =
    raw.preferred && typeof raw.preferred === "object" ? Object.entries(raw.preferred as Record<string, unknown>) : [];
  const preferred: Record<string, string> = {};
  for (const [primary, fallback] of preferredEntries) {
    if (typeof fallback !== "string") continue;
    const primaryNorm = normalizeBaseUrl(primary);
    const fallbackNorm = normalizeBaseUrl(fallback);
    if (!primaryNorm || !fallbackNorm || primaryNorm === fallbackNorm) continue;
    preferred[primaryNorm] = fallbackNorm;
  }
  return { savedAtMs, blocked: Array.from(new Set(blocked)), preferred };
}

const fallbackStorage = loadFallbackStorage();
const blockedFallbackBases = new Set<string>(fallbackStorage.blocked);
const preferredFallbackBases = new Map<string, string>(Object.entries(fallbackStorage.preferred));

function persistFallbackStorage() {
  writeJson(FALLBACK_STORAGE_KEY, {
    savedAtMs: Date.now(),
    blocked: Array.from(blockedFallbackBases),
    preferred: Object.fromEntries(preferredFallbackBases),
  } satisfies FallbackStorage);
}

function rememberPreferredFallback(primary: string, fallback: string) {
  if (primary.startsWith("/")) return;
  if (!primary || !fallback || primary === fallback) return;
  if (blockedFallbackBases.has(fallback)) return;
  if (preferredFallbackBases.get(primary) === fallback) return;
  preferredFallbackBases.set(primary, fallback);
  persistFallbackStorage();
}

function clearPreferredFallback(primary: string) {
  if (!preferredFallbackBases.delete(primary)) return;
  persistFallbackStorage();
}

function blockFallbackBase(fallback: string) {
  if (!fallback || blockedFallbackBases.has(fallback)) return;
  blockedFallbackBases.add(fallback);
  for (const [primary, preferred] of preferredFallbackBases.entries()) {
    if (preferred === fallback) preferredFallbackBases.delete(primary);
  }
  persistFallbackStorage();
}

function isCrossOriginBase(baseUrl: string): boolean {
  if (typeof window === "undefined") return false;
  if (!/^https?:\/\//.test(baseUrl)) return false;
  try {
    return new URL(baseUrl).origin !== window.location.origin;
  } catch {
    return false;
  }
}

function isJsonContentType(raw: string): boolean {
  const ct = raw.toLowerCase();
  return ct.includes("application/json") || ct.includes("+json");
}

function resolveFallbackBase(primaryBase: string): string | null {
  const fallbackRaw = TRADER_UI_CONFIG.apiFallbackUrl?.trim() ?? "";
  if (!fallbackRaw) return null;
  const primary = normalizeBaseUrl(primaryBase);
  const fallback = normalizeBaseUrl(fallbackRaw);
  if (!fallback || fallback === primary) return null;
  if (primary.startsWith("/") && isCrossOriginBase(fallback) && !TRADER_UI_CONFIG.apiBaseUrlInferred) return null;
  if (!primary.startsWith("/") && blockedFallbackBases.has(fallback)) return null;
  return fallback;
}

function resolvePreferredFallback(primaryBase: string, fallbackBase: string | null): string | null {
  if (primaryBase.startsWith("/")) return null;
  if (!fallbackBase) return null;
  const preferred = preferredFallbackBases.get(primaryBase) ?? null;
  if (preferred) {
    if (preferred !== fallbackBase) return null;
    if (!primaryBase.startsWith("/") && blockedFallbackBases.has(preferred)) return null;
    return preferred;
  }
  return null;
}

function mergeHeaders(base: HeadersInit | undefined, extra: Record<string, string> | undefined): HeadersInit | undefined {
  if (!extra || Object.keys(extra).length === 0) return base;
  const merged = new Headers(base);
  for (const [key, value] of Object.entries(extra)) merged.set(key, value);
  return merged;
}

const MAX_TIMER_DELAY_MS = 2_147_483_647;

function clampDelayMs(raw: number): number | null {
  if (!Number.isFinite(raw)) return null;
  return Math.max(0, Math.min(MAX_TIMER_DELAY_MS, Math.floor(raw)));
}

function runtimeSetTimeout(callback: () => void, delayMs: number): ReturnType<typeof globalThis.setTimeout> {
  return globalThis.setTimeout(callback, delayMs);
}

function runtimeClearTimeout(timer: ReturnType<typeof globalThis.setTimeout>) {
  globalThis.clearTimeout(timer);
}

function parseRetryAfterMs(raw: string | null): number | null {
  if (!raw) return null;
  const trimmed = raw.trim();
  if (!trimmed) return null;
  if (/^\d+$/.test(trimmed)) return clampDelayMs(Number(trimmed) * 1000);
  const parsed = Date.parse(trimmed);
  if (!Number.isNaN(parsed)) return clampDelayMs(parsed - Date.now());
  return null;
}

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  const clampedMs = clampDelayMs(ms) ?? 0;
  if (clampedMs <= 0) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const onAbort = () => {
      cleanup();
      reject((signal as AbortSignal & { reason?: unknown }).reason ?? new DOMException("Aborted", "AbortError"));
    };

    const timer = runtimeSetTimeout(() => {
      cleanup();
      resolve();
    }, clampedMs);

    const cleanup = () => {
      runtimeClearTimeout(timer);
      signal?.removeEventListener("abort", onAbort);
    };

    if (signal) {
      if (signal.aborted) return onAbort();
      signal.addEventListener("abort", onAbort, { once: true });
    }
  });
}

function withTimeout(externalSignal: AbortSignal | undefined, timeoutMs: number) {
  const controller = new AbortController();
  let onAbort: (() => void) | null = null;

  if (externalSignal) {
    if (externalSignal.aborted) controller.abort(externalSignal.reason);
    else {
      onAbort = () => controller.abort(externalSignal.reason);
      externalSignal.addEventListener("abort", onAbort, { once: true });
    }
  }

  const timerDelayMs = clampDelayMs(timeoutMs) ?? 1;
  const timer = runtimeSetTimeout(() => controller.abort(new DOMException("Timeout", "TimeoutError")), timerDelayMs);
  return {
    signal: controller.signal,
    cleanup: () => {
      runtimeClearTimeout(timer);
      if (externalSignal && onAbort) externalSignal.removeEventListener("abort", onAbort);
    },
  };
}

async function readJsonOrText(res: Response, contentType: string): Promise<unknown> {
  const bodyText = await res.text();
  if (isJsonContentType(contentType)) {
    const trimmed = bodyText.trim();
    if (!trimmed) return null;
    try {
      return JSON.parse(trimmed);
    } catch {
      throw new UnexpectedResponseError(res.status, contentType, summarizePayload(bodyText));
    }
  }
  return bodyText;
}

function summarizePayload(payload: unknown): string {
  if (payload == null) return "";
  if (typeof payload === "string") {
    const trimmed = payload.trim();
    return trimmed.length > 320 ? `${trimmed.slice(0, 320)}...` : trimmed;
  }
  try {
    const json = JSON.stringify(payload);
    return json.length > 320 ? `${json.slice(0, 320)}...` : json;
  } catch {
    return "";
  }
}

async function fetchJsonOnce<T>(baseUrl: string, path: string, init: RequestInit, opts?: FetchJsonOptions): Promise<T> {
  const timeoutMs = opts?.timeoutMs ?? TRADER_UI_CONFIG.timeoutsMs?.requestMs ?? 60_000;
  const { signal, cleanup } = withTimeout(opts?.signal, timeoutMs);
  try {
    const resolvedUrl = resolveUrl(baseUrl, path);
    const tenantKey = tenantKeyFromPath(path) ?? tenantKeyFromPath(resolvedUrl);
    const url = pathWithoutTenantKey(resolvedUrl);
    const method = String(init.method ?? "GET").toUpperCase();
    const headers = withTenantHeader(
      new Headers(mergeHeaders(init.headers, opts?.headers)),
      path,
      init.body,
      Boolean(tenantKey) || shouldAttachTenantHeader(url, method),
    );
    if (tenantKey && !headers.has(TENANT_HEADER)) headers.set(TENANT_HEADER, tenantKey);
    const res = await fetch(url, {
      ...init,
      cache: init.cache ?? "no-store",
      headers,
      signal,
    });
    const contentType = res.headers.get("content-type") || "";
    const retryAfterMs = parseRetryAfterMs(res.headers.get("retry-after"));
    const payload = await readJsonOrText(res, contentType);
    if (res.ok && !isJsonContentType(contentType)) {
      throw new UnexpectedResponseError(res.status, contentType, summarizePayload(payload));
    }
    if (!res.ok) {
      const baseMessage =
        typeof payload === "object" && payload && "error" in payload
          ? String((payload as ApiError).error)
          : typeof payload === "string" && payload.trim()
            ? payload.trim()
            : `${res.status} ${res.statusText}`;
      const hint =
        typeof payload === "object" && payload && "hint" in payload && (payload as ApiError).hint
          ? String((payload as ApiError).hint)
          : "";
      const message = hint ? `${baseMessage}\nHint: ${hint}` : baseMessage;
      throw new HttpError(res.status, message, payload, retryAfterMs);
    }
    return payload as T;
  } catch (err) {
    if (signal.aborted) {
      const reason = (signal as AbortSignal & { reason?: unknown }).reason;
      if (reason instanceof DOMException && reason.name === "TimeoutError") throw reason;
    }
    throw err;
  } finally {
    cleanup();
  }
}

async function fetchJson<T>(baseUrl: string, path: string, init: RequestInit, opts?: FetchJsonOptions): Promise<T> {
  const primaryBase = normalizeBaseUrl(baseUrl);
  const fallbackBase = resolveFallbackBase(primaryBase);
  const method = String(init.method ?? "GET").toUpperCase();
  const mergedHeaders = mergeHeaders(init.headers, opts?.headers);
  const proxyToDirectCrossOrigin = Boolean(primaryBase.startsWith("/") && fallbackBase && isCrossOriginBase(fallbackBase));
  // Keep inferred /api -> direct-host failover for reads, and only allow
  // cross-origin writes when the request already carries auth-like context
  // that the backend's implicit CORS policy accepts.
  const allowCrossOriginProxyFallbackForMethod =
    !proxyToDirectCrossOrigin ||
    method === "GET" ||
    method === "HEAD" ||
    requestHasAuthLikeContext(path, init.body, mergedHeaders);
  const allowAuthStatusFallback = Boolean(
    TRADER_UI_CONFIG.apiBaseUrlInferred &&
      fallbackBase &&
      fallbackBase.startsWith("/") &&
      isCrossOriginBase(primaryBase),
  );
  const allowTimeoutFallback = Boolean(
    fallbackBase &&
      (allowAuthStatusFallback ||
        (TRADER_UI_CONFIG.apiBaseUrlInferred && primaryBase.startsWith("/") && isCrossOriginBase(fallbackBase))),
  );
  const allowFallback = opts?.allowFallback !== false && allowCrossOriginProxyFallbackForMethod;
  const preferredBase = allowFallback ? resolvePreferredFallback(primaryBase, fallbackBase) : null;

  if (preferredBase) {
    try {
      return await fetchJsonOnce<T>(preferredBase, path, init, opts);
    } catch (err) {
      clearPreferredFallback(primaryBase);
      if (fallbackBase && preferredBase === fallbackBase && isNetworkError(err) && isCrossOriginBase(fallbackBase)) {
        blockFallbackBase(fallbackBase);
      }
    }
  }
  try {
    return await fetchJsonOnce<T>(primaryBase, path, init, opts);
  } catch (err) {
    if (fallbackBase && allowFallback && shouldFallbackToApiBase(err, allowAuthStatusFallback, allowTimeoutFallback)) {
      try {
        const out = await fetchJsonOnce<T>(fallbackBase, path, init, opts);
        rememberPreferredFallback(primaryBase, fallbackBase);
        return out;
      } catch (fallbackErr) {
        if (isNetworkError(fallbackErr) && isCrossOriginBase(fallbackBase)) {
          blockFallbackBase(fallbackBase);
          throw err;
        }
        throw fallbackErr;
      }
    }
    throw err;
  }
}

function timeoutError(): DOMException {
  return new DOMException("Timeout", "TimeoutError");
}

function isTimeoutError(err: unknown): boolean {
  return err instanceof DOMException && err.name === "TimeoutError";
}

function isAbortError(err: unknown): boolean {
  return err instanceof DOMException && err.name === "AbortError";
}

function isNetworkError(err: unknown): boolean {
  return err instanceof TypeError;
}

function shouldFallbackToApiBase(err: unknown, allowAuthStatusFallback: boolean, allowTimeoutFallback: boolean): boolean {
  if (err instanceof UnexpectedResponseError) return true;
  if (isAbortError(err)) return false;
  if (isTimeoutError(err)) return allowTimeoutFallback;
  if (err instanceof HttpError) {
    if (err.status === 401 || err.status === 403) return allowAuthStatusFallback;
    return err.status === 404 || err.status === 502 || err.status === 503 || err.status === 504;
  }
  return isNetworkError(err);
}

function shouldFallbackToGet(err: unknown): boolean {
  if (!(err instanceof HttpError)) return false;
  return err.status === 403 || err.status === 405 || err.status === 501 || err.status === 502 || err.status === 503 || err.status === 504;
}

function asyncJobNotFoundMessage(): string {
  return "Async job not found (server restarted or behind a non-sticky load balancer). Please retry; for multi-instance deployments, enable shared async job storage (TRADER_API_ASYNC_DIR or TRADER_STATE_DIR) or run single-instance.";
}

function describeAsyncTimeout(baseUrl: string, overallTimeoutMs: number, lastError: unknown): string {
  const seconds = Math.max(1, Math.round(overallTimeoutMs / 1000));
  const last =
    lastError instanceof HttpError
      ? `${lastError.status} ${lastError.message}`
      : lastError instanceof Error
        ? lastError.message
        : String(lastError);
  const hint = baseUrl.startsWith("/api")
    ? " Check your CloudFront `/api/*` proxy (or set apiBaseUrl in trader-config.js to your API host)."
    : " Check API connectivity and try again.";
  return `Async request timed out after ${seconds}s while retrying after errors (last error: ${last}).${hint}`;
}

function isBacktestQueueBusy(err: unknown): err is HttpError {
  if (!(err instanceof HttpError)) return false;
  if (err.status !== 429) return false;
  return err.message.toLowerCase().includes("backtest queue is busy");
}

function shouldRetryAsyncStart(err: unknown): boolean {
  if (isTimeoutError(err) || isNetworkError(err)) return true;
  if (err instanceof UnexpectedResponseError) return true;
  return err instanceof HttpError && (err.status === 502 || err.status === 503 || err.status === 504);
}

type TransientRetryOptions = {
  maxRetries?: number;
  initialBackoffMs?: number;
  maxBackoffMs?: number;
};

async function fetchJsonWithTransientRetry<T>(
  baseUrl: string,
  path: string,
  init: RequestInit,
  opts?: FetchJsonOptions,
  retryOpts?: TransientRetryOptions,
): Promise<T> {
  const maxRetries = retryOpts?.maxRetries ?? 2;
  const maxBackoffMs = retryOpts?.maxBackoffMs ?? 5_000;
  let retries = 0;
  let backoffMs = retryOpts?.initialBackoffMs ?? 750;

  for (;;) {
    try {
      return await fetchJson<T>(baseUrl, path, init, opts);
    } catch (err) {
      if (retries >= maxRetries || !shouldRetryAsyncStart(err)) throw err;
      retries += 1;
      const retryAfterMs =
        err instanceof HttpError && typeof err.retryAfterMs === "number" && Number.isFinite(err.retryAfterMs)
          ? Math.max(0, err.retryAfterMs)
          : null;
      const delayMs = retryAfterMs == null ? backoffMs : Math.max(backoffMs, retryAfterMs);
      opts?.onTransientRetry?.({ attempt: retries, maxRetries, error: err, delayMs, retryAfterMs });
      await sleep(delayMs, opts?.signal);
      if (retryAfterMs == null) backoffMs = Math.min(maxBackoffMs, Math.round(backoffMs * 1.4));
    }
  }
}

async function runSyncBacktestWithRetry(
  baseUrl: string,
  params: ApiParams,
  opts?: AsyncJobOptions,
): Promise<BacktestResponse> {
  const startedAt = Date.now();
  const overallTimeoutMs = opts?.timeoutMs ?? 60_000;
  let backoffMs = 750;
  let sawBusy = false;

  for (;;) {
    const elapsed = Date.now() - startedAt;
    const remaining = overallTimeoutMs - elapsed;
    if (remaining <= 0) {
      if (sawBusy) {
        throw new Error("Backtest queue stayed busy. Try again shortly or increase TRADER_API_MAX_BACKTEST_RUNNING.");
      }
      throw timeoutError();
    }

    const requestOpts: FetchJsonOptions = {
      signal: opts?.signal,
      headers: opts?.headers,
      timeoutMs: Math.max(1, remaining),
    };

    try {
      return await fetchJson<BacktestResponse>(
        baseUrl,
        "/backtest",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(params),
        },
        requestOpts,
      );
    } catch (err) {
      if (!isBacktestQueueBusy(err)) throw err;
      sawBusy = true;
      const retryAfterMs =
        typeof err.retryAfterMs === "number" && Number.isFinite(err.retryAfterMs) ? Math.max(0, err.retryAfterMs) : backoffMs;
      const delayMs = Math.min(retryAfterMs, remaining);
      await sleep(delayMs, opts?.signal);
      backoffMs = Math.min(5_000, Math.round(backoffMs * 1.4));
    }
  }
}

async function runAsyncJob<T>(
  baseUrl: string,
  startPath: string,
  pollPath: string,
  params: ApiParams,
  opts?: AsyncJobOptions,
): Promise<T> {
  const startedAt = Date.now();
  const overallTimeoutMs = opts?.timeoutMs ?? 60_000;
  const perRequestTimeoutMs = Math.min(55_000, overallTimeoutMs);
  const notFoundGraceMs = Math.min(2 * 60_000, Math.max(10_000, Math.round(overallTimeoutMs * 0.5)));
  let lastTransientError: unknown = null;
  let notFoundSinceMs: number | null = null;
  const retryStart = opts?.retryStart ?? false;
  const maxStartRetries = opts?.maxStartRetries ?? 2;
  let startRetries = 0;

  let startBackoffMs = 750;
  let start: AsyncStartResponse;
  for (;;) {
    const elapsed = Date.now() - startedAt;
    const remaining = overallTimeoutMs - elapsed;
    if (remaining <= 0) {
      if (lastTransientError) throw new Error(describeAsyncTimeout(baseUrl, overallTimeoutMs, lastTransientError));
      throw timeoutError();
    }

    try {
      start = await fetchJson<AsyncStartResponse>(
        baseUrl,
        startPath,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(params),
        },
        { signal: opts?.signal, headers: opts?.headers, timeoutMs: Math.min(remaining, perRequestTimeoutMs) },
      );
      break;
    } catch (err) {
      if (err instanceof HttpError && err.status === 404) throw new AsyncEndpointNotFoundError(err);
      // 429 is safe to retry: the server didn't start the async job.
      if (err instanceof HttpError && err.status === 429) {
        lastTransientError = err;
        const retryAfterMs = typeof err.retryAfterMs === "number" && Number.isFinite(err.retryAfterMs) ? Math.max(0, err.retryAfterMs) : null;
        const delayMs = retryAfterMs == null ? startBackoffMs : Math.max(startBackoffMs, retryAfterMs);
        await sleep(Math.min(delayMs, remaining), opts?.signal);
        if (retryAfterMs == null) startBackoffMs = Math.min(5_000, Math.round(startBackoffMs * 1.4));
        continue;
      }
      if (retryStart && shouldRetryAsyncStart(err)) {
        lastTransientError = err;
        if (startRetries >= maxStartRetries) throw err;
        startRetries += 1;
        await sleep(Math.min(startBackoffMs, remaining), opts?.signal);
        startBackoffMs = Math.min(5_000, Math.round(startBackoffMs * 1.4));
        continue;
      }
      throw err;
    }
  }
  if (!start || typeof start !== "object" || !("jobId" in start) || typeof (start as { jobId?: unknown }).jobId !== "string") {
    throw new Error("Invalid async start response");
  }
  opts?.onJobId?.(start.jobId);

  let cancelSent = false;
  const cancel = async () => {
    if (cancelSent) return;
    cancelSent = true;
    const cancelUrl = `${pollPath}/${encodeURIComponent(start.jobId)}/cancel`;
    try {
      await fetchJson<{ status?: string }>(baseUrl, cancelUrl, { method: "POST" }, { headers: opts?.headers, timeoutMs: 10_000 });
    } catch {
      // Best-effort; ignore if the API doesn't support cancel or we can't reach it.
    }
  };

  let pollMethod: "POST" | "GET" = "POST";
  let backoffMs = 750;
  try {
    for (;;) {
      const elapsed = Date.now() - startedAt;
      const remaining = overallTimeoutMs - elapsed;
      if (remaining <= 0) {
        await cancel();
        if (lastTransientError) throw new Error(describeAsyncTimeout(baseUrl, overallTimeoutMs, lastTransientError));
        throw timeoutError();
      }

      let status: AsyncPollResponse<T>;
      try {
        const pollUrl = `${pollPath}/${encodeURIComponent(start.jobId)}`;
        try {
          status = await fetchJson<AsyncPollResponse<T>>(
            baseUrl,
            pollUrl,
            { method: pollMethod },
            { signal: opts?.signal, headers: opts?.headers, timeoutMs: Math.min(remaining, perRequestTimeoutMs) },
          );
        } catch (err) {
          if (pollMethod === "POST" && shouldFallbackToGet(err)) {
            status = await fetchJson<AsyncPollResponse<T>>(
              baseUrl,
              pollUrl,
              { method: "GET" },
              { signal: opts?.signal, headers: opts?.headers, timeoutMs: Math.min(remaining, perRequestTimeoutMs) },
            );
            pollMethod = "GET";
          } else {
            throw err;
          }
        }
      } catch (err) {
        if (err instanceof HttpError && (err.status === 401 || err.status === 403)) throw err;
        if (err instanceof HttpError && err.status === 404) {
          lastTransientError = err;
          if (notFoundSinceMs == null) notFoundSinceMs = Date.now();
          if (Date.now() - notFoundSinceMs > notFoundGraceMs) {
            throw new Error(asyncJobNotFoundMessage());
          }
          await sleep(Math.min(backoffMs, remaining), opts?.signal);
          backoffMs = Math.min(5_000, Math.round(backoffMs * 1.4));
          continue;
        }
        if (err instanceof HttpError && err.status === 429) {
          lastTransientError = err;
          const retryAfterMs = typeof err.retryAfterMs === "number" && Number.isFinite(err.retryAfterMs) ? Math.max(0, err.retryAfterMs) : 0;
          const delayMs = Math.min(Math.max(backoffMs, retryAfterMs), remaining);
          await sleep(delayMs, opts?.signal);
          backoffMs = Math.min(5_000, Math.round(backoffMs * 1.4));
          continue;
        }
        if (isTimeoutError(err)) {
          lastTransientError = err;
          await sleep(Math.min(backoffMs, remaining), opts?.signal);
          backoffMs = Math.min(5_000, Math.round(backoffMs * 1.4));
          continue;
        }
        if (err instanceof HttpError && err.status >= 500) {
          lastTransientError = err;
          await sleep(Math.min(backoffMs, remaining), opts?.signal);
          backoffMs = Math.min(5_000, Math.round(backoffMs * 1.4));
          continue;
        }
        if (isNetworkError(err)) {
          lastTransientError = err;
          await sleep(Math.min(backoffMs, remaining), opts?.signal);
          backoffMs = Math.min(5_000, Math.round(backoffMs * 1.4));
          continue;
        }
        throw err;
      }

      lastTransientError = null;
      if (!status || typeof status !== "object" || !("status" in status) || typeof (status as { status?: unknown }).status !== "string") {
        throw new Error("Invalid async poll response");
      }
      if (status.status === "done") return status.result as T;
      if (status.status === "error") {
        const msg = status.error || "Async job failed";
        if (msg.trim().toLowerCase() === "not found") {
          lastTransientError = new Error("Async job not found");
          if (notFoundSinceMs == null) notFoundSinceMs = Date.now();
          if (Date.now() - notFoundSinceMs > notFoundGraceMs) {
            throw new Error(asyncJobNotFoundMessage());
          }
          await sleep(Math.min(backoffMs, remaining), opts?.signal);
          backoffMs = Math.min(5_000, Math.round(backoffMs * 1.4));
          continue;
        }
        throw new Error(msg);
      }
      if (status.status !== "running") throw new Error(`Unexpected async status: ${String(status.status)}`);

      notFoundSinceMs = null;
      await sleep(Math.min(backoffMs, remaining), opts?.signal);
      backoffMs = Math.min(5_000, Math.round(backoffMs * 1.4));
    }
  } catch (err) {
    if (isAbortError(err) || isTimeoutError(err)) await cancel();
    throw err;
  }
}

export async function health(baseUrl: string, opts?: FetchJsonOptions): Promise<HealthResponse> {
  const out = await fetchJson<{
    status: string;
    version?: string;
    commit?: string;
    authRequired?: boolean;
    authOk?: boolean;
    computeLimits?: { maxBarsLstm: number; maxEpochs: number; maxHiddenSize: number };
    asyncJobs?: { maxRunning: number; ttlMs: number; persistence: boolean };
    cache?: { enabled: boolean; ttlMs: number; maxEntries: number };
  }>(baseUrl, "/health", { method: "GET" }, opts);
  if (out.status !== "ok") throw new Error("Unexpected /health response");
  return {
    status: "ok",
    version: out.version,
    commit: out.commit,
    authRequired: out.authRequired,
    authOk: out.authOk,
    computeLimits: out.computeLimits,
    asyncJobs: out.asyncJobs,
    cache: out.cache,
  };
}

export async function cacheStats(baseUrl: string, opts?: FetchJsonOptions): Promise<CacheStatsResponse> {
  return fetchJson<CacheStatsResponse>(baseUrl, "/cache", { method: "GET" }, opts);
}

export async function cacheClear(baseUrl: string, opts?: FetchJsonOptions): Promise<CacheClearResponse> {
  return fetchJson<CacheClearResponse>(baseUrl, "/cache/clear", { method: "POST" }, opts);
}

export async function signal(baseUrl: string, params: ApiParams, opts?: AsyncJobOptions): Promise<LatestSignal> {
  const asyncOpts = opts ? { ...opts, retryStart: true } : { retryStart: true };
  try {
    return await runAsyncJob<LatestSignal>(baseUrl, "/signal/async", "/signal/async", params, asyncOpts);
  } catch (err) {
    if (err instanceof AsyncEndpointNotFoundError) {
      return fetchJson<LatestSignal>(
        baseUrl,
        "/signal",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(params),
        },
        opts,
      );
    }
    throw err;
  }
}

export async function backtest(baseUrl: string, params: ApiParams, opts?: AsyncJobOptions): Promise<BacktestResponse> {
  const asyncOpts = opts ? { ...opts, retryStart: true } : { retryStart: true };
  try {
    return await runAsyncJob<BacktestResponse>(baseUrl, "/backtest/async", "/backtest/async", params, asyncOpts);
  } catch (err) {
    if (err instanceof AsyncEndpointNotFoundError) {
      return runSyncBacktestWithRetry(baseUrl, params, opts);
    }
    throw err;
  }
}

export async function trade(baseUrl: string, params: ApiParams, opts?: AsyncJobOptions): Promise<ApiTradeResponse> {
  try {
    return await runAsyncJob<ApiTradeResponse>(baseUrl, "/trade/async", "/trade/async", params, opts);
  } catch (err) {
    if (err instanceof AsyncEndpointNotFoundError) {
      return fetchJson<ApiTradeResponse>(
        baseUrl,
        "/trade",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(params),
        },
        opts,
      );
    }
    throw err;
  }
}

export async function binanceKeysStatus(
  baseUrl: string,
  params: ApiParams,
  opts?: FetchJsonOptions,
): Promise<BinanceKeysStatus> {
  return fetchJson<BinanceKeysStatus>(
    baseUrl,
    "/binance/keys",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    },
    opts,
  );
}

export async function coinbaseKeysStatus(
  baseUrl: string,
  params: ApiParams,
  opts?: FetchJsonOptions,
): Promise<CoinbaseKeysStatus> {
  return fetchJson<CoinbaseKeysStatus>(
    baseUrl,
    "/coinbase/keys",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    },
    opts,
  );
}

export async function requestProgressStatus(
  baseUrl: string,
  requestId: string,
  opts?: FetchJsonOptions,
): Promise<ApiRequestProgressStatus> {
  return fetchJson<ApiRequestProgressStatus>(baseUrl, `/request-progress/${encodeURIComponent(requestId)}`, { method: "GET" }, opts);
}

type BinanceListenKeyStartParams = Pick<ApiParams, "market" | "binanceTestnet" | "binanceApiKey" | "binanceApiSecret" | "tenantKey">;
type BinanceListenKeyActionParams = BinanceListenKeyStartParams & { listenKey: string };

export async function binanceListenKey(baseUrl: string, params: BinanceListenKeyStartParams, opts?: FetchJsonOptions): Promise<BinanceListenKeyResponse> {
  return fetchJson<BinanceListenKeyResponse>(
    baseUrl,
    "/binance/listenKey",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    },
    opts,
  );
}

export async function binanceListenKeyKeepAlive(
  baseUrl: string,
  params: BinanceListenKeyActionParams,
  opts?: FetchJsonOptions,
): Promise<BinanceListenKeyKeepAliveResponse> {
  return fetchJson<BinanceListenKeyKeepAliveResponse>(
    baseUrl,
    "/binance/listenKey/keepAlive",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    },
    opts,
  );
}

export async function binanceListenKeyClose(
  baseUrl: string,
  params: BinanceListenKeyActionParams,
  opts?: FetchJsonOptions,
): Promise<BinanceListenKeyKeepAliveResponse> {
  return fetchJson<BinanceListenKeyKeepAliveResponse>(
    baseUrl,
    "/binance/listenKey/close",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    },
    opts,
  );
}

export async function binancePositions(
  baseUrl: string,
  params: ApiBinancePositionsRequest,
  opts?: FetchJsonOptions,
): Promise<ApiBinancePositionsResponse> {
  return fetchJson<ApiBinancePositionsResponse>(
    baseUrl,
    "/binance/positions",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    },
    opts,
  );
}

export async function binancePositionsAll(
  baseUrl: string,
  opts?: FetchJsonOptions,
): Promise<ApiBinancePositionsResponse> {
  return fetchJson<ApiBinancePositionsResponse>(
    baseUrl,
    "/binance/positions",
    {
      method: "GET",
    },
    opts,
  );
}

export async function binanceClosePosition(
  baseUrl: string,
  params: ApiBinanceClosePositionRequest,
  opts?: FetchJsonOptions,
): Promise<ApiOrderResult> {
  return fetchJson<ApiOrderResult>(
    baseUrl,
    "/binance/positions/close",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    },
    opts,
  );
}

export async function binanceTrades(
  baseUrl: string,
  params: ApiBinanceTradesRequest,
  opts?: FetchJsonOptions,
): Promise<ApiBinanceTradesResponse> {
  return fetchJson<ApiBinanceTradesResponse>(
    baseUrl,
    "/binance/trades",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    },
    opts,
  );
}

function requestedSingleBotSymbol(params: ApiParams): string | undefined {
  const symbols = params.botSymbols
    ?.map((symbol) => normalizeExpectedBotIdentity(symbol))
    .filter((symbol): symbol is string => symbol !== undefined)
    .map(normalizeBotSymbol)
    .filter((symbol, index, all) => all.indexOf(symbol) === index);
  return symbols?.length === 1 ? symbols[0] : undefined;
}

export async function botStart(baseUrl: string, params: ApiParams, opts?: FetchJsonOptions): Promise<BotStatus> {
  const payload = await fetchJsonWithTransientRetry<unknown>(
    baseUrl,
    "/bot/start",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    },
    opts,
    { maxRetries: 2 },
  );
  return decodeBotStatus(payload, "/bot/start", {
    tenantKey: params.tenantKey,
    symbol: requestedSingleBotSymbol(params),
    allowMultiForExpectedSymbol: true,
  });
}

export async function botStop(baseUrl: string, opts?: FetchJsonOptions, symbol?: string, tenantKey?: string): Promise<BotStatus> {
  const query = new URLSearchParams();
  if (symbol) query.set("symbol", symbol);
  if (tenantKey) query.set("tenantKey", tenantKey);
  const path = query.size > 0 ? `/bot/stop?${query.toString()}` : "/bot/stop";
  const payload = await fetchJson<unknown>(baseUrl, path, { method: "POST" }, opts);
  return decodeBotStatus(payload, "/bot/stop", { tenantKey, symbol });
}

export async function botStatus(
  baseUrl: string,
  opts?: FetchJsonOptions,
  tail?: number,
  symbol?: string,
  tenantKey?: string,
): Promise<BotStatus> {
  const tailSafe = normalizeExactIntegerQueryParam(tail) ?? 0;
  const query = new URLSearchParams();
  if (tailSafe > 0) query.set("tail", String(tailSafe));
  if (symbol) query.set("symbol", symbol);
  if (tenantKey) query.set("tenantKey", tenantKey);
  const path = query.size > 0 ? `/bot/status?${query.toString()}` : "/bot/status";
  const payload = await fetchJson<unknown>(baseUrl, path, { method: "GET" }, opts);
  return decodeBotStatus(payload, "/bot/status", { tenantKey, symbol });
}

export async function ops(
  baseUrl: string,
  params?: {
    kind?: string;
    limit?: number;
    since?: number;
    symbol?: string;
    fromMs?: number;
    toMs?: number;
    bot?: boolean;
    tenantKey?: string;
  },
  opts?: FetchJsonOptions,
): Promise<OpsResponse> {
  const query = new URLSearchParams();
  if (params?.kind) query.set("kind", params.kind);
  const limit = normalizeExactIntegerQueryParam(params?.limit);
  if (limit != null) query.set("limit", String(limit));
  const since = normalizeExactIntegerQueryParam(params?.since);
  if (since != null) query.set("since", String(since));
  if (params?.symbol) query.set("symbol", params.symbol);
  const fromMs = normalizeExactIntegerQueryParam(params?.fromMs);
  if (fromMs != null) query.set("fromMs", String(fromMs));
  const toMs = normalizeExactIntegerQueryParam(params?.toMs);
  if (toMs != null) query.set("toMs", String(toMs));
  if (typeof params?.bot === "boolean") query.set("bot", params.bot ? "1" : "0");
  if (params?.tenantKey) query.set("tenantKey", params.tenantKey);
  const path = query.size > 0 ? `/ops?${query.toString()}` : "/ops";
  return fetchJson<OpsResponse>(baseUrl, path, { method: "GET" }, opts);
}

export async function opsPerformance(
  baseUrl: string,
  params?: { commitLimit?: number; comboLimit?: number; comboScope?: string; comboOrder?: string; tenantKey?: string },
  opts?: FetchJsonOptions,
): Promise<OpsPerformanceResponse> {
  const query = new URLSearchParams();
  const commitLimit = normalizeExactIntegerQueryParam(params?.commitLimit);
  if (commitLimit != null) query.set("commitLimit", String(commitLimit));
  const comboLimit = normalizeExactIntegerQueryParam(params?.comboLimit);
  if (comboLimit != null) query.set("comboLimit", String(comboLimit));
  if (params?.comboScope) query.set("comboScope", params.comboScope);
  if (params?.comboOrder) query.set("comboOrder", params.comboOrder);
  if (params?.tenantKey) query.set("tenantKey", params.tenantKey);
  const path = query.size > 0 ? `/ops/performance?${query.toString()}` : "/ops/performance";
  return fetchJson<OpsPerformanceResponse>(baseUrl, path, { method: "GET" }, opts);
}

export async function optimizerCombos(baseUrl: string, opts?: FetchJsonOptions): Promise<unknown> {
  return fetchJson<unknown>(baseUrl, "/optimizer/combos", { method: "GET" }, opts);
}

export async function optimizerRun(
  baseUrl: string,
  params: OptimizerRunRequest,
  opts?: FetchJsonOptions,
): Promise<OptimizerRunResponse> {
  return fetchJson<OptimizerRunResponse>(
    baseUrl,
    "/optimizer/run",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    },
    opts,
  );
}

export async function stateSyncExport(baseUrl: string, opts?: FetchJsonOptions & { tenantKey?: string }): Promise<StateSyncPayload> {
  const mergedOpts = { ...opts, allowFallback: false };
  const query = new URLSearchParams();
  if (opts?.tenantKey) query.set("tenantKey", opts.tenantKey);
  const path = query.size > 0 ? `/state/sync?${query.toString()}` : "/state/sync";
  return fetchJson<StateSyncPayload>(baseUrl, path, { method: "GET" }, mergedOpts);
}

export async function stateSyncImport(
  baseUrl: string,
  payload: StateSyncPayload,
  opts?: FetchJsonOptions & { tenantKey?: string },
): Promise<StateSyncImportResponse> {
  const mergedOpts = { ...opts, allowFallback: false };
  const query = new URLSearchParams();
  if (opts?.tenantKey) query.set("tenantKey", opts.tenantKey);
  const path = query.size > 0 ? `/state/sync?${query.toString()}` : "/state/sync";
  return fetchJson<StateSyncImportResponse>(
    baseUrl,
    path,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    },
    mergedOpts,
  );
}
