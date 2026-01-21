import type {
  ApiTradeResponse,
  BacktestResponse,
  BinanceKeysStatus,
  BinanceTrade,
  BotStatus,
  BotStatusMulti,
  BotStatusSingle,
  CoinbaseKeysStatus,
  LatestSignal,
  Market,
  Method,
  Normalization,
  OpsOperation,
  OptimizerRunRequest,
  OptimizerSource,
  Platform,
} from "../lib/types";
import type { OptimizationCombo, OptimizationComboOperation } from "../components/TopCombosChart";
import type { FormState } from "./formState";
import type { health } from "../lib/api";
import { clamp, normalizeSymbolKey, numFromInput } from "./utils";

export type RequestKind = "signal" | "backtest" | "trade";

export type RunOptions = {
  silent?: boolean;
};

export type ActiveAsyncJob = {
  kind: RequestKind;
  jobId: string | null;
  startedAtMs: number;
};

export type RateLimitState = {
  untilMs: number;
  reason: string;
  lastHitAtMs: number;
};

export type PanelPrefs = Record<string, boolean>;

export type KeysStatus = BinanceKeysStatus | CoinbaseKeysStatus;

export type OpsUiState = {
  loading: boolean;
  error: string | null;
  enabled: boolean;
  hint: string | null;
  ops: OpsOperation[];
  limit: number;
  lastFetchedAtMs: number | null;
};

export type BotStatusOp = {
  atMs: number;
  running: boolean;
  live: boolean;
  symbol: string | null;
};
export function isCoinbaseKeysStatus(status: KeysStatus): status is CoinbaseKeysStatus {
  return "hasApiPassphrase" in status;
}

export function isBinanceKeysStatus(status: KeysStatus): status is BinanceKeysStatus {
  return "market" in status;
}

export function isBotStatusMulti(status: BotStatus): status is BotStatusMulti {
  return "multi" in status && status.multi === true;
}

export function botStatusSymbol(status: BotStatusSingle): string | null {
  if (status.running) return status.symbol;
  if (status.symbol) return status.symbol;
  if (status.snapshot?.symbol) return status.snapshot.symbol;
  return null;
}

export function botStatusKey(status: { market: Market; symbol: string; interval: string }): string {
  return `${status.market}:${normalizeSymbolKey(status.symbol)}:${status.interval}`;
}

export function botStatusKeyFromSingle(status: BotStatusSingle): string | null {
  const symbol = botStatusSymbol(status);
  if (!symbol) return null;
  const market = status.running ? status.market : status.market ?? status.snapshot?.market;
  const interval = status.running ? status.interval : status.interval ?? status.snapshot?.interval;
  if (!market || !interval) return null;
  return botStatusKey({ market, symbol, interval });
}

export function formatDatetimeLocal(ms: number): string {
  if (!Number.isFinite(ms)) return "";
  const d = new Date(ms);
  const pad = (v: number) => String(v).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

export function parseDatetimeLocal(raw: string): number | null {
  if (!raw.trim()) return null;
  const parsed = Date.parse(raw);
  return Number.isNaN(parsed) ? null : parsed;
}

export function parseBotStatusOp(op: OpsOperation): BotStatusOp | null {
  if (!op || op.kind !== "bot.status") return null;
  if (typeof op.atMs !== "number" || !Number.isFinite(op.atMs)) return null;
  const rec = (op.result as Record<string, unknown> | null | undefined) ?? {};
  const running = typeof rec.running === "boolean" ? rec.running : null;
  if (running == null) return null;
  const live = typeof rec.live === "boolean" ? rec.live : false;
  const symbol = typeof rec.symbol === "string" ? rec.symbol : null;
  return { atMs: op.atMs, running, live, symbol };
}

export function parseSymbolsInput(raw: string): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const part of raw.split(/[,\s]+/)) {
    const sym = part.trim().toUpperCase();
    if (!sym) continue;
    if (seen.has(sym)) continue;
    seen.add(sym);
    out.push(sym);
  }
  return out;
}

export function symbolFormatPattern(platform: Platform): RegExp {
  switch (platform) {
    case "coinbase":
      return /^[A-Z0-9]+-[A-Z0-9]+$/;
    case "poloniex":
      return /^[A-Z0-9]+_[A-Z0-9]+$/;
    case "binance":
    case "kraken":
    default:
      return /^[A-Z0-9]{3,30}$/;
  }
}

export const COMMON_QUOTES = ["USDT", "USDC", "FDUSD", "TUSD", "BUSD", "BTC", "ETH", "BNB"];
export const BINANCE_SYMBOL_PATTERN = /^[A-Z0-9]{3,30}$/;
export const EQUITY_TIPS = {
  preset: [
    'Use "Preset: Equity focus", then bump Trials/Timeout to widen the search.',
    'To bias toward raw equity, reduce or clear drawdown/turnover penalties while keeping Objective/Tune objective on "annualized-equity".',
    "For short windows (e.g., 48h), shorter intervals (15m/30m/1h) increase sample size; keep Backtest + Tune ratios < 1.",
  ],
  trials: ["Higher Trials/Timeout expands the search (slower) and can lift annualized equity."],
  objective: ['Keep Objective and Tune objective on "annualized-equity" for equity-focused ranking.'],
  penalties: ["Reduce or clear drawdown/turnover penalties to favor raw equity."],
  intervals: ["For short windows (e.g., 48h), shorter intervals (15m/30m/1h) increase sample size."],
  ratios: ["Keep Backtest ratio + Tune ratio < 1 to leave enough training data."],
};
export const COMPLEX_TIPS = {
  method: [
    "11 requires Kalman + LSTM agreement beyond the open threshold; fewer trades, higher confidence.",
    "blend averages predictions; blend weight sets the Kalman vs LSTM mix.",
    "router picks the best recent model using router lookback; min score gates to HOLD.",
  ],
  thresholds: [
    "Open threshold is the entry deadband; below break-even can churn after costs.",
    "Close threshold is often <= open threshold to reduce whipsaw.",
  ],
  edge: [
    "Min edge is the minimum predicted return to trade; cost-aware edge adds break-even + buffer.",
    "Edge buffer adds extra margin above break-even when cost-aware edge is on.",
  ],
  snr: ["Signal/vol (SNR) filters trades when predicted edge is small versus recent volatility."],
  blend: ["0 = LSTM only, 1 = Kalman only. Only used with method=blend."],
  router: ["Lookback controls how much recent history the router uses; longer is smoother but slower to adapt.", "Min score gates low-confidence periods to HOLD."],
  split: ["Backtest ratio is the held-out tail; tune ratio is only used for optimization/sweeps.", "Backtest + tune must be < 1 to leave training data."],
  lstm: ["Normalization affects scaling for LSTM only; keep consistent with training.", "Epochs/hidden size trade off fit vs runtime and overfitting."],
  optimization: ["Sweep thresholds searches open/close gates only.", "Optimize operations also tries methods and thresholds; router disables both."],
  tuneObjective: ["Tune objective defines the score used during fit/tune; it can differ from backtest objective."],
  walkForward: ["Walk-forward folds split data into sequential folds to estimate stability."],
};

export function trimBinanceComboSuffix(value: string): string | null {
  const compact = value.replace(/[^A-Z0-9]/g, "");
  if (!compact) return null;
  let best: string | null = null;
  for (const quote of COMMON_QUOTES) {
    let idx = compact.indexOf(quote);
    while (idx >= 0) {
      const end = idx + quote.length;
      if (end < compact.length) {
        const suffix = compact.slice(end);
        if (/\d/.test(suffix)) {
          const candidate = compact.slice(0, end);
          if (BINANCE_SYMBOL_PATTERN.test(candidate) && !COMMON_QUOTES.includes(candidate)) {
            if (!best || candidate.length > best.length) best = candidate;
          }
        }
      }
      idx = compact.indexOf(quote, idx + 1);
    }
  }
  return best;
}

export function normalizeComboSymbol(raw: string, platform: Platform | null): string {
  const value = raw.trim().toUpperCase();
  if (!value) return value;
  const resolvedPlatform: Platform = platform ?? "binance";
  const pattern = symbolFormatPattern(resolvedPlatform);
  const isBinanceLike = resolvedPlatform === "binance" || resolvedPlatform === "kraken";

  if (isBinanceLike) {
    const trimmed = trimBinanceComboSuffix(value);
    if (trimmed) return trimmed;
  }

  if (pattern.test(value)) return value;

  if (resolvedPlatform === "coinbase") {
    const parts = value.split("-");
    if (parts.length >= 2) {
      const candidate = `${parts[0]}-${parts[1]}`;
      if (pattern.test(candidate)) return candidate;
    }
    return value;
  }

  if (resolvedPlatform === "poloniex") {
    const parts = value.split("_");
    if (parts.length >= 2) {
      const candidate = `${parts[0]}_${parts[1]}`;
      if (pattern.test(candidate)) return candidate;
    }
    return value;
  }

  if (isBinanceLike) {
    const tokens = value.split(/[^A-Z0-9]+/).filter(Boolean);
    if (tokens.length >= 2) {
      const joined = `${tokens[0]}${tokens[1]}`;
      if (tokens.length === 2 && pattern.test(joined)) return joined;
      if (tokens.length >= 3 && /^[0-9]+[A-Z]$/.test(tokens[2] ?? "") && pattern.test(joined)) return joined;
    }
    if (tokens.length >= 1 && pattern.test(tokens[0] ?? "")) return tokens[0] ?? value;
  }
  return value;
}

export function symbolFormatExample(platform: Platform): string {
  switch (platform) {
    case "coinbase":
      return "BTC-USD";
    case "poloniex":
      return "BTC_USDT";
    case "kraken":
      return "XBTUSD";
    case "binance":
    default:
      return "BTCUSDT";
  }
}

export function invalidSymbolsForPlatform(platform: Platform, symbols: string[]): string[] {
  const pattern = symbolFormatPattern(platform);
  return symbols.filter((sym) => !pattern.test(sym));
}

export function parseMaybeInt(raw: string): number | null {
  const trimmed = raw.trim();
  if (!trimmed) return null;
  const n = Number(trimmed);
  if (!Number.isFinite(n)) return null;
  const rounded = Math.trunc(n);
  return rounded < 0 ? null : rounded;
}

export function normalizeIsoInput(raw: string): string | null {
  const trimmed = raw.trim();
  if (/^\d{4}-\d{2}-\d{2}$/.test(trimmed)) return `${trimmed}T00:00:00Z`;
  if (
    !/^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d{1,3})?)?(?:Z|[+-]\d{2}:\d{2})?$/.test(
      trimmed,
    )
  ) {
    return null;
  }
  return trimmed.replace(" ", "T");
}

export function parseTimeInputMs(raw: string): number | null {
  const trimmed = raw.trim();
  if (!trimmed) return null;
  if (/^\d+$/.test(trimmed)) {
    const n = Number(trimmed);
    return Number.isFinite(n) ? n : null;
  }
  const iso = normalizeIsoInput(trimmed);
  if (!iso) return null;
  const parsed = Date.parse(iso);
  return Number.isNaN(parsed) ? null : parsed;
}

export function sanitizeFilenameSegment(raw: string, fallback: string): string {
  const trimmed = raw.trim();
  if (!trimmed) return fallback;
  const cleaned = trimmed.replace(/[^A-Za-z0-9]+/g, "-").replace(/^-+/, "").replace(/-+$/, "");
  return cleaned || fallback;
}

export function csvEscape(value: unknown): string {
  if (value == null) return "";
  const text = String(value);
  if (text === "") return "";
  return /[",\n]/.test(text) ? `"${text.replace(/"/g, "\"\"")}"` : text;
}

export const TRADE_PNL_EPS = 1e-9;
export const TRADE_PNL_TOP_N = 5;

export type TradePnlRow = {
  idx: number;
  entryIndex: number;
  exitIndex: number;
  entryEquity: number;
  exitEquity: number;
  return: number;
  holdingPeriods: number;
  exitReason: string | null;
  phase: string;
  pnl: number;
  entryTime: number | null;
  exitTime: number | null;
};

export type TradePnlAnalysis = {
  count: number;
  wins: number;
  losses: number;
  breakeven: number;
  winRate: number | null;
  avgWin: number | null;
  avgLoss: number | null;
  avgReturn: number | null;
  maxWin: number | null;
  maxLoss: number | null;
  totalWin: number;
  totalLoss: number;
  profitFactor: number | null;
  payoffRatio: number | null;
  avgHoldWin: number | null;
  avgHoldLoss: number | null;
  topWins: TradePnlRow[];
  topLosses: TradePnlRow[];
};

export type CommissionTotal = {
  asset: string;
  total: number;
  count: number;
};

export type BinancePnlRow = {
  idx: number;
  tradeId: number;
  orderId: number | null;
  time: number;
  symbol: string;
  side: string;
  price: number;
  qty: number;
  quoteQty: number;
  positionSide: string | null;
  realizedPnl: number;
  commission: number | null;
  commissionAsset: string | null;
};

export type BinancePnlAnalysis = {
  count: number;
  wins: number;
  losses: number;
  breakeven: number;
  winRate: number | null;
  avgWin: number | null;
  avgLoss: number | null;
  avgPnl: number | null;
  maxWin: number | null;
  maxLoss: number | null;
  totalWin: number;
  totalLoss: number;
  totalPnl: number;
  totalQty: number;
  totalQuoteQty: number;
  profitFactor: number | null;
  payoffRatio: number | null;
  commissionTotals: CommissionTotal[];
  topWins: BinancePnlRow[];
  topLosses: BinancePnlRow[];
};

export function pnlBadgeClass(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "badge";
  if (value > TRADE_PNL_EPS) return "badge badgeStrong badgeLong";
  if (value < -TRADE_PNL_EPS) return "badge badgeStrong badgeFlat";
  return "badge badgeHold";
}

export function binanceTradeSideLabel(trade: BinanceTrade): "BUY" | "SELL" | "—" {
  const raw = trade.side?.toUpperCase();
  if (raw === "BUY" || raw === "SELL") return raw;
  if (trade.isBuyer === true) return "BUY";
  if (trade.isBuyer === false) return "SELL";
  return "—";
}

export function backtestTradePhase(split: BacktestResponse["split"], entryIndex: number): string {
  if (entryIndex >= split.backtestStartIndex) return "backtest";
  if (split.tune > 0 && entryIndex >= split.tuneStartIndex) return "tune";
  return split.tune > 0 ? "fit" : "train";
}

export function buildBacktestOpsCsv(backtest: BacktestResponse): string {
  const header = [
    "tradeIndex",
    "phase",
    "entryIndex",
    "exitIndex",
    "entryPrice",
    "exitPrice",
    "entryEquity",
    "exitEquity",
    "return",
    "holdingPeriods",
    "exitReason",
  ].join(",");
  const prices = backtest.prices ?? [];
  const rows = backtest.trades.map((trade, idx) => {
    const entryPrice = prices[trade.entryIndex];
    const exitPrice = prices[trade.exitIndex];
    const phase = backtestTradePhase(backtest.split, trade.entryIndex);
    return [
      idx + 1,
      phase,
      trade.entryIndex,
      trade.exitIndex,
      Number.isFinite(entryPrice) ? entryPrice : "",
      Number.isFinite(exitPrice) ? exitPrice : "",
      trade.entryEquity,
      trade.exitEquity,
      trade.return,
      trade.holdingPeriods,
      trade.exitReason ?? "",
    ]
      .map(csvEscape)
      .join(",");
  });
  return [header, ...rows].join("\n");
}

export function buildBacktestTradePnlAnalysis(backtest: BacktestResponse): TradePnlAnalysis | null {
  const trades = backtest.trades ?? [];
  if (trades.length === 0) return null;
  const openTimes = Array.isArray(backtest.openTimes) ? backtest.openTimes : null;
  const rows: TradePnlRow[] = trades
    .map((trade, idx) => {
      if (!Number.isFinite(trade.return)) return null;
      const entryTimeRaw = openTimes?.[trade.entryIndex];
      const exitTimeRaw = openTimes?.[trade.exitIndex];
      const entryTime = typeof entryTimeRaw === "number" && Number.isFinite(entryTimeRaw) ? entryTimeRaw : null;
      const exitTime = typeof exitTimeRaw === "number" && Number.isFinite(exitTimeRaw) ? exitTimeRaw : null;
      return {
        idx: idx + 1,
        entryIndex: trade.entryIndex,
        exitIndex: trade.exitIndex,
        entryEquity: trade.entryEquity,
        exitEquity: trade.exitEquity,
        return: trade.return,
        holdingPeriods: trade.holdingPeriods,
        exitReason: trade.exitReason ?? null,
        phase: backtestTradePhase(backtest.split, trade.entryIndex),
        pnl: trade.exitEquity - trade.entryEquity,
        entryTime,
        exitTime,
      };
    })
    .filter((row): row is TradePnlRow => Boolean(row));
  if (rows.length === 0) return null;

  let wins = 0;
  let losses = 0;
  let breakeven = 0;
  let sumWin = 0;
  let sumLoss = 0;
  let sumReturn = 0;
  let maxWin: number | null = null;
  let maxLoss: number | null = null;
  let holdWin = 0;
  let holdLoss = 0;

  for (const row of rows) {
    const r = row.return;
    sumReturn += r;
    if (r > TRADE_PNL_EPS) {
      wins += 1;
      sumWin += r;
      holdWin += row.holdingPeriods;
      maxWin = maxWin === null ? r : Math.max(maxWin, r);
    } else if (r < -TRADE_PNL_EPS) {
      losses += 1;
      sumLoss += r;
      holdLoss += row.holdingPeriods;
      maxLoss = maxLoss === null ? r : Math.min(maxLoss, r);
    } else {
      breakeven += 1;
    }
  }

  const count = rows.length;
  const avgWin = wins > 0 ? sumWin / wins : null;
  const avgLoss = losses > 0 ? sumLoss / losses : null;
  const avgReturn = count > 0 ? sumReturn / count : null;
  const winRate = count > 0 ? wins / count : null;
  const profitFactor = sumLoss < 0 ? sumWin / Math.abs(sumLoss) : sumWin > 0 ? Infinity : null;
  const payoffRatio = avgWin !== null && avgLoss !== null && avgLoss !== 0 ? avgWin / Math.abs(avgLoss) : null;
  const avgHoldWin = wins > 0 ? holdWin / wins : null;
  const avgHoldLoss = losses > 0 ? holdLoss / losses : null;
  const topWins = rows
    .filter((row) => row.return > TRADE_PNL_EPS)
    .sort((a, b) => b.return - a.return)
    .slice(0, TRADE_PNL_TOP_N);
  const topLosses = rows
    .filter((row) => row.return < -TRADE_PNL_EPS)
    .sort((a, b) => a.return - b.return)
    .slice(0, TRADE_PNL_TOP_N);

  return {
    count,
    wins,
    losses,
    breakeven,
    winRate,
    avgWin,
    avgLoss,
    avgReturn,
    maxWin,
    maxLoss,
    totalWin: sumWin,
    totalLoss: sumLoss,
    profitFactor,
    payoffRatio,
    avgHoldWin,
    avgHoldLoss,
    topWins,
    topLosses,
  };
}

export function buildBinanceTradePnlAnalysis(trades: BinanceTrade[]): BinancePnlAnalysis | null {
  const rows: BinancePnlRow[] = [];
  const commissionByAsset = new Map<string, CommissionTotal>();
  for (let i = 0; i < trades.length; i += 1) {
    const trade = trades[i];
    if (!trade) continue;
    const pnl = trade.realizedPnl;
    if (typeof pnl !== "number" || !Number.isFinite(pnl)) continue;
    const side = binanceTradeSideLabel(trade);
    const commission = trade.commission;
    const commissionAsset = trade.commissionAsset ?? null;
    if (typeof commission === "number" && Number.isFinite(commission)) {
      const assetKey = commissionAsset ?? "unknown";
      const existing = commissionByAsset.get(assetKey);
      if (existing) {
        existing.total += commission;
        existing.count += 1;
      } else {
        commissionByAsset.set(assetKey, { asset: assetKey, total: commission, count: 1 });
      }
    }
    rows.push({
      idx: i + 1,
      tradeId: trade.tradeId,
      orderId: trade.orderId ?? null,
      time: trade.time,
      symbol: trade.symbol,
      side,
      price: trade.price,
      qty: trade.qty,
      quoteQty: trade.quoteQty,
      positionSide: trade.positionSide ?? null,
      realizedPnl: pnl,
      commission: commission ?? null,
      commissionAsset,
    });
  }

  if (rows.length === 0) return null;

  let wins = 0;
  let losses = 0;
  let breakeven = 0;
  let sumWin = 0;
  let sumLoss = 0;
  let sumPnl = 0;
  let sumQty = 0;
  let sumQuoteQty = 0;
  let maxWin: number | null = null;
  let maxLoss: number | null = null;

  for (const row of rows) {
    const qty = row.qty;
    if (Number.isFinite(qty)) sumQty += qty;
    const quoteQty = row.quoteQty;
    if (Number.isFinite(quoteQty)) sumQuoteQty += quoteQty;
    const pnl = row.realizedPnl;
    sumPnl += pnl;
    if (pnl > TRADE_PNL_EPS) {
      wins += 1;
      sumWin += pnl;
      maxWin = maxWin === null ? pnl : Math.max(maxWin, pnl);
    } else if (pnl < -TRADE_PNL_EPS) {
      losses += 1;
      sumLoss += pnl;
      maxLoss = maxLoss === null ? pnl : Math.min(maxLoss, pnl);
    } else {
      breakeven += 1;
    }
  }

  const count = rows.length;
  const avgWin = wins > 0 ? sumWin / wins : null;
  const avgLoss = losses > 0 ? sumLoss / losses : null;
  const avgPnl = count > 0 ? sumPnl / count : null;
  const winRate = count > 0 ? wins / count : null;
  const profitFactor = sumLoss < 0 ? sumWin / Math.abs(sumLoss) : sumWin > 0 ? Infinity : null;
  const payoffRatio = avgWin !== null && avgLoss !== null && avgLoss !== 0 ? avgWin / Math.abs(avgLoss) : null;
  const topWins = rows
    .filter((row) => row.realizedPnl > TRADE_PNL_EPS)
    .sort((a, b) => b.realizedPnl - a.realizedPnl)
    .slice(0, TRADE_PNL_TOP_N);
  const topLosses = rows
    .filter((row) => row.realizedPnl < -TRADE_PNL_EPS)
    .sort((a, b) => a.realizedPnl - b.realizedPnl)
    .slice(0, TRADE_PNL_TOP_N);
  const commissionTotals = Array.from(commissionByAsset.values()).sort((a, b) => a.asset.localeCompare(b.asset));

  return {
    count,
    wins,
    losses,
    breakeven,
    winRate,
    avgWin,
    avgLoss,
    avgPnl,
    maxWin,
    maxLoss,
    totalWin: sumWin,
    totalLoss: sumLoss,
    totalPnl: sumPnl,
    totalQty: sumQty,
    totalQuoteQty: sumQuoteQty,
    profitFactor,
    payoffRatio,
    commissionTotals,
    topWins,
    topLosses,
  };
}

export function downloadTextFile(filename: string, contents: string, contentType = "text/plain"): void {
  if (typeof window === "undefined") return;
  const blob = new Blob([contents], { type: contentType });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
}

export function buildPositionSeries(prices: number[], side: number): number[] {
  if (prices.length === 0) return [];
  if (!Number.isFinite(side) || side === 0) return Array.from({ length: prices.length }, () => 0);
  const dir = side > 0 ? 1 : -1;
  return Array.from({ length: prices.length }, () => dir);
}

export function buildEquityCurve(prices: number[], side: number): number[] {
  if (prices.length === 0) return [];
  const dir = side > 0 ? 1 : side < 0 ? -1 : 0;
  const out = [1];
  for (let i = 1; i < prices.length; i += 1) {
    const prev = prices[i - 1] ?? 0;
    const cur = prices[i] ?? prev;
    const last = out[out.length - 1] ?? 1;
    if (!Number.isFinite(prev) || !Number.isFinite(cur) || prev === 0 || cur === 0 || dir === 0) {
      out.push(last);
      continue;
    }
    const ratio = dir > 0 ? cur / prev : prev / cur;
    out.push(last * ratio);
  }
  return out;
}

export function positionSideInfo(positionAmt: number, positionSide?: string | null): { dir: number; label: string; key: string } {
  const raw = positionSide?.trim().toUpperCase();
  const side = raw && raw !== "BOTH" ? raw : null;
  const dir = side === "SHORT" ? -1 : side === "LONG" ? 1 : positionAmt > 0 ? 1 : positionAmt < 0 ? -1 : 0;
  const label = side ?? (dir > 0 ? "LONG" : dir < 0 ? "SHORT" : "FLAT");
  const key = side ?? (dir > 0 ? "LONG" : dir < 0 ? "SHORT" : "FLAT");
  return { dir, label, key };
}

export type ListenKeyStreamStatus = "disconnected" | "connecting" | "connected" | "stopped";

export type ListenKeyStreamStatusPayload = { status?: string; message?: string; atMs?: number };
export type ListenKeyStreamKeepAlivePayload = { atMs?: number };
export type ListenKeyStreamErrorPayload = { message?: string; atMs?: number };

export function normalizeListenKeyStreamStatus(raw: string): ListenKeyStreamStatus {
  switch (raw) {
    case "connecting":
    case "connected":
    case "disconnected":
    case "stopped":
      return raw;
    default:
      return "disconnected";
  }
}

export function safeJsonParse<T = unknown>(raw: string): T | null {
  try {
    return JSON.parse(raw) as T;
  } catch {
    return null;
  }
}

// Minimal SSE parser for fetch streams that handles chunk boundaries.
export function createSseParser(onEvent: (event: string, data: string) => void): (chunk: string) => void {
  let buffer = "";
  return (chunk: string) => {
    buffer += chunk.replace(/\r/g, "");
    while (true) {
      const boundary = buffer.indexOf("\n\n");
      if (boundary === -1) return;
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      if (!block.trim()) continue;
      let eventName = "message";
      const dataLines: string[] = [];
      for (const line of block.split("\n")) {
        if (!line || line.startsWith(":")) continue;
        if (line.startsWith("event:")) {
          eventName = line.slice(6).trim();
          continue;
        }
        if (line.startsWith("data:")) {
          dataLines.push(line.slice(5).trimStart());
        }
      }
      onEvent(eventName, dataLines.join("\n"));
    }
  };
}

export function parseOptionalNumber(raw: string): number | undefined {
  const trimmed = raw.trim();
  if (!trimmed) return undefined;
  const parsed = numFromInput(trimmed, Number.NaN);
  return Number.isFinite(parsed) ? parsed : undefined;
}

export function parseOptionalInt(raw: string): number | undefined {
  const parsed = parseOptionalNumber(raw);
  if (parsed == null) return undefined;
  return Math.trunc(parsed);
}

export function parseOptionalString(raw: string): string | undefined {
  const trimmed = raw.trim();
  return trimmed ? trimmed : undefined;
}

export type UiState = {
  loading: boolean;
  error: string | null;
  lastKind: RequestKind | null;
  latestSignal: LatestSignal | null;
  backtest: BacktestResponse | null;
  trade: ApiTradeResponse | null;
};

export type ErrorFix =
  | {
      label: string;
      action: "tuneRatio";
      value: number;
      targetId?: string;
      toast: string;
    }
  | {
      label: string;
      action: "backtestRatio";
      value: number;
      targetId?: string;
      toast: string;
    }
  | {
      label: string;
      action: "bars";
      value: number;
      targetId?: string;
      toast: string;
    }
  | {
      label: string;
      action: "lookbackBars";
      value: number;
      targetId?: string;
      toast: string;
    }
  | {
      label: string;
      action: "lookbackWindow";
      value: string;
      targetId?: string;
      toast: string;
    };

export type BotUiState = {
  loading: boolean;
  error: string | null;
  status: BotStatus;
};

export type BotRtEvent = {
  atMs: number;
  message: string;
};

export type BotTelemetryPoint = {
  atMs: number;
  pollLatencyMs: number | null;
  driftBps: number | null;
};

export type BotRtUiState = {
  lastFetchAtMs: number | null;
  lastFetchDurationMs: number | null;
  lastNewCandles: number;
  lastNewCandlesAtMs: number | null;
  lastKlineUpdates: number;
  lastKlineUpdatesAtMs: number | null;
  telemetry: BotTelemetryPoint[];
  feed: BotRtEvent[];
};

export type BotRtTracker = {
  lastOpenTimeMs: number | null;
  lastError: string | null;
  lastHalted: boolean | null;
  lastFetchedOpenTimeMs: number | null;
  lastFetchedClose: number | null;
  lastMethod: Method | null;
  lastOpenThreshold: number | null;
  lastCloseThreshold: number | null;
  lastTradeEnabled: boolean | null;
  lastTelemetryPolledAtMs: number | null;
};

export const emptyBotRtState = (): BotRtUiState => ({
  lastFetchAtMs: null,
  lastFetchDurationMs: null,
  lastNewCandles: 0,
  lastNewCandlesAtMs: null,
  lastKlineUpdates: 0,
  lastKlineUpdatesAtMs: null,
  telemetry: [],
  feed: [],
});

export const emptyBotRtTracker = (): BotRtTracker => ({
  lastOpenTimeMs: null,
  lastError: null,
  lastHalted: null,
  lastFetchedOpenTimeMs: null,
  lastFetchedClose: null,
  lastMethod: null,
  lastOpenThreshold: null,
  lastCloseThreshold: null,
  lastTradeEnabled: null,
  lastTelemetryPolledAtMs: null,
});

export type KeysUiState = {
  loading: boolean;
  error: string | null;
  status: KeysStatus | null;
  platform: Platform | null;
  checkedAtMs: number | null;
};

export type CacheUiState = {
  loading: boolean;
  error: string | null;
  stats: Awaited<ReturnType<typeof cacheStats>> | null;
};

export type ListenKeyUiState = {
  loading: boolean;
  error: string | null;
  info: BinanceListenKeyResponse | null;
  wsStatus: ListenKeyStreamStatus;
  wsError: string | null;
  lastEventAtMs: number | null;
  lastEvent: string | null;
  keepAliveAtMs: number | null;
  keepAliveError: string | null;
};

export type BinanceTradesUiState = {
  loading: boolean;
  error: string | null;
  response: ApiBinanceTradesResponse | null;
};

export type BinancePositionsUiState = {
  loading: boolean;
  error: string | null;
  response: ApiBinancePositionsResponse | null;
};

export type OptimizerRunUiState = {
  loading: boolean;
  error: string | null;
  response: OptimizerRunResponse | null;
  lastRunAtMs: number | null;
};

export type OptimizerRunForm = {
  source: OptimizerSource;
  symbol: string;
  dataPath: string;
  priceColumn: string;
  highColumn: string;
  lowColumn: string;
  platforms: string;
  intervals: string;
  lookbackWindow: string;
  barsMin: string;
  barsMax: string;
  barsAutoProb: string;
  barsDistribution: "" | "uniform" | "log";
  trials: string;
  timeoutSec: string;
  seed: string;
  seedTrials: string;
  seedRatio: string;
  survivorFraction: string;
  perturbScaleDouble: string;
  perturbScaleInt: string;
  earlyStopNoImprove: string;
  objective: string;
  tuneObjective: string;
  backtestRatio: string;
  tuneRatio: string;
  penaltyMaxDrawdown: string;
  penaltyTurnover: string;
  normalizations: string;
  epochsMin: string;
  epochsMax: string;
  hiddenSizeMin: string;
  hiddenSizeMax: string;
  lrMin: string;
  lrMax: string;
  patienceMax: string;
  gradClipMin: string;
  gradClipMax: string;
  pDisableGradClip: string;
  slippageMax: string;
  spreadMax: string;
  minRoundTrips: string;
  minWinRate: string;
  minSharpe: string;
  minAnnualizedReturn: string;
  minCalmar: string;
  minProfitFactor: string;
  maxTurnover: string;
  minExposure: string;
  minWalkForwardSharpeMean: string;
  maxWalkForwardSharpeStd: string;
  walkForwardFoldsMin: string;
  walkForwardFoldsMax: string;
  minHoldBarsMin: string;
  minHoldBarsMax: string;
  cooldownBarsMin: string;
  cooldownBarsMax: string;
  maxHoldBarsMin: string;
  maxHoldBarsMax: string;
  minEdgeMin: string;
  minEdgeMax: string;
  minSignalToNoiseMin: string;
  minSignalToNoiseMax: string;
  edgeBufferMin: string;
  edgeBufferMax: string;
  trendLookbackMin: string;
  trendLookbackMax: string;
  pCostAwareEdge: string;
  stopMin: string;
  stopMax: string;
  tpMin: string;
  tpMax: string;
  trailMin: string;
  trailMax: string;
  methodWeightBlend: string;
  blendWeightMin: string;
  blendWeightMax: string;
  disableLstmPersistence: boolean;
  noSweepThreshold: boolean;
  extraJson: string;
};

export type TopCombosSource = "api";

export type TopCombosMeta = {
  source: TopCombosSource;
  generatedAtMs: number | null;
  payloadSource: string | null;
  payloadSources: string[] | null;
  fallbackReason: string | null;
  comboCount: number | null;
};

export type ComboOrder = "annualized-equity" | "rank" | "date-desc" | "date-asc";

export type OrderSideFilter = "ALL" | "BUY" | "SELL";

export type OrderLogPrefs = {
  filterText: string;
  sentOnly: boolean;
  side: OrderSideFilter;
  limit: number;
  errorsOnly: boolean;
  showOrderId: boolean;
  showStatus: boolean;
  showClientOrderId: boolean;
};

export type SavedProfiles = Record<string, FormState>;

export type PendingProfileLoad = {
  name: string;
  profile: FormState;
  reasons: string[];
};

export type ComputeLimits = NonNullable<Awaited<ReturnType<typeof health>>["computeLimits"]>;
export type ManualOverrideKey = "method" | "openThreshold" | "closeThreshold";

export function optimizerSourceForPlatform(platform: Platform): OptimizerSource {
  switch (platform) {
    case "coinbase":
      return "coinbase";
    case "kraken":
      return "kraken";
    case "poloniex":
      return "poloniex";
    default:
      return "binance";
  }
}

export function buildDefaultOptimizerRunForm(symbol: string, platform: Platform): OptimizerRunForm {
  return {
    source: optimizerSourceForPlatform(platform),
    symbol: symbol.trim().toUpperCase(),
    dataPath: "",
    priceColumn: "close",
    highColumn: "",
    lowColumn: "",
    platforms: "",
    intervals: "1h,2h,4h,6h,12h,1d",
    lookbackWindow: "7d",
    barsMin: "",
    barsMax: "",
    barsAutoProb: "",
    barsDistribution: "",
    trials: "50",
    timeoutSec: "60",
    seed: "42",
    seedTrials: "",
    seedRatio: "",
    survivorFraction: "",
    perturbScaleDouble: "",
    perturbScaleInt: "",
    earlyStopNoImprove: "",
    objective: "annualized-equity",
    tuneObjective: "annualized-equity",
    backtestRatio: "0.2",
    tuneRatio: "0.25",
    penaltyMaxDrawdown: "",
    penaltyTurnover: "",
    normalizations: "none,minmax,standard,log",
    epochsMin: "",
    epochsMax: "",
    hiddenSizeMin: "",
    hiddenSizeMax: "",
    lrMin: "",
    lrMax: "",
    patienceMax: "",
    gradClipMin: "",
    gradClipMax: "",
    pDisableGradClip: "",
    slippageMax: "",
    spreadMax: "",
    minRoundTrips: "",
    minWinRate: "",
    minSharpe: "",
    minAnnualizedReturn: "",
    minCalmar: "",
    minProfitFactor: "",
    maxTurnover: "",
    minExposure: "",
    minWalkForwardSharpeMean: "",
    maxWalkForwardSharpeStd: "",
    walkForwardFoldsMin: "",
    walkForwardFoldsMax: "",
    minHoldBarsMin: "",
    minHoldBarsMax: "",
    cooldownBarsMin: "",
    cooldownBarsMax: "",
    maxHoldBarsMin: "",
    maxHoldBarsMax: "",
    minEdgeMin: "",
    minEdgeMax: "",
    minSignalToNoiseMin: "",
    minSignalToNoiseMax: "",
    edgeBufferMin: "",
    edgeBufferMax: "",
    trendLookbackMin: "",
    trendLookbackMax: "",
    pCostAwareEdge: "",
    stopMin: "",
    stopMax: "",
    tpMin: "",
    tpMax: "",
    trailMin: "",
    trailMax: "",
    methodWeightBlend: "",
    blendWeightMin: "",
    blendWeightMax: "",
    disableLstmPersistence: false,
    noSweepThreshold: false,
    extraJson: "",
  };
}

export function parseOptimizerExtras(raw: string): { value: Record<string, unknown> | null; error: string | null } {
  const trimmed = raw.trim();
  if (!trimmed) return { value: null, error: null };
  try {
    const parsed = JSON.parse(trimmed);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return { value: null, error: "Extra options must be a JSON object." };
    }
    return { value: parsed as Record<string, unknown>, error: null };
  } catch (err) {
    const msg = err instanceof Error ? err.message : "Invalid JSON.";
    return { value: null, error: `Invalid JSON: ${msg}` };
  }
}

export function buildOptimizerRunRequest(form: OptimizerRunForm, extras: Record<string, unknown> | null): OptimizerRunRequest {
  const req: OptimizerRunRequest = {
    source: form.source,
  };

  const symbol = parseOptionalString(form.symbol);
  const dataPath = parseOptionalString(form.dataPath);
  const priceColumn = parseOptionalString(form.priceColumn);
  const highColumn = parseOptionalString(form.highColumn);
  const lowColumn = parseOptionalString(form.lowColumn);

  if (form.source === "csv") {
    if (dataPath) req.data = dataPath;
    if (priceColumn) req.priceColumn = priceColumn;
    if (highColumn && lowColumn) {
      req.highColumn = highColumn;
      req.lowColumn = lowColumn;
    }
  } else {
    if (symbol) req.binanceSymbol = symbol.toUpperCase();
    const platforms = parseOptionalString(form.platforms);
    if (platforms) req.platforms = platforms;
  }

  const intervals = parseOptionalString(form.intervals);
  if (intervals) req.intervals = intervals;
  const lookbackWindow = parseOptionalString(form.lookbackWindow);
  if (lookbackWindow) req.lookbackWindow = lookbackWindow;

  const barsMin = parseOptionalInt(form.barsMin);
  if (barsMin != null) req.barsMin = barsMin;
  const barsMax = parseOptionalInt(form.barsMax);
  if (barsMax != null) req.barsMax = barsMax;
  const barsAutoProb = parseOptionalNumber(form.barsAutoProb);
  if (barsAutoProb != null) req.barsAutoProb = barsAutoProb;
  if (form.barsDistribution) req.barsDistribution = form.barsDistribution;

  const trials = parseOptionalInt(form.trials);
  if (trials != null) req.trials = trials;
  const timeoutSec = parseOptionalNumber(form.timeoutSec);
  if (timeoutSec != null) req.timeoutSec = timeoutSec;
  const seed = parseOptionalInt(form.seed);
  if (seed != null) req.seed = seed;
  const seedTrials = parseOptionalInt(form.seedTrials);
  if (seedTrials != null) req.seedTrials = seedTrials;
  const seedRatio = parseOptionalNumber(form.seedRatio);
  if (seedRatio != null) req.seedRatio = seedRatio;
  const survivorFraction = parseOptionalNumber(form.survivorFraction);
  if (survivorFraction != null) req.survivorFraction = survivorFraction;
  const perturbScaleDouble = parseOptionalNumber(form.perturbScaleDouble);
  if (perturbScaleDouble != null) req.perturbScaleDouble = perturbScaleDouble;
  const perturbScaleInt = parseOptionalInt(form.perturbScaleInt);
  if (perturbScaleInt != null) req.perturbScaleInt = perturbScaleInt;
  const earlyStopNoImprove = parseOptionalInt(form.earlyStopNoImprove);
  if (earlyStopNoImprove != null) req.earlyStopNoImprove = earlyStopNoImprove;

  const objective = parseOptionalString(form.objective);
  if (objective) req.objective = objective;
  const tuneObjective = parseOptionalString(form.tuneObjective);
  if (tuneObjective) req.tuneObjective = tuneObjective;
  const backtestRatio = parseOptionalNumber(form.backtestRatio);
  if (backtestRatio != null) req.backtestRatio = backtestRatio;
  const tuneRatio = parseOptionalNumber(form.tuneRatio);
  if (tuneRatio != null) req.tuneRatio = tuneRatio;
  const penaltyMaxDrawdown = parseOptionalNumber(form.penaltyMaxDrawdown);
  if (penaltyMaxDrawdown != null) req.penaltyMaxDrawdown = penaltyMaxDrawdown;
  const penaltyTurnover = parseOptionalNumber(form.penaltyTurnover);
  if (penaltyTurnover != null) req.penaltyTurnover = penaltyTurnover;

  const normalizations = parseOptionalString(form.normalizations);
  if (normalizations) req.normalizations = normalizations;
  const epochsMin = parseOptionalInt(form.epochsMin);
  if (epochsMin != null) req.epochsMin = epochsMin;
  const epochsMax = parseOptionalInt(form.epochsMax);
  if (epochsMax != null) req.epochsMax = epochsMax;
  const hiddenSizeMin = parseOptionalInt(form.hiddenSizeMin);
  if (hiddenSizeMin != null) req.hiddenSizeMin = hiddenSizeMin;
  const hiddenSizeMax = parseOptionalInt(form.hiddenSizeMax);
  if (hiddenSizeMax != null) req.hiddenSizeMax = hiddenSizeMax;
  const lrMin = parseOptionalNumber(form.lrMin);
  if (lrMin != null) req.lrMin = lrMin;
  const lrMax = parseOptionalNumber(form.lrMax);
  if (lrMax != null) req.lrMax = lrMax;
  const patienceMax = parseOptionalInt(form.patienceMax);
  if (patienceMax != null) req.patienceMax = patienceMax;
  const gradClipMin = parseOptionalNumber(form.gradClipMin);
  if (gradClipMin != null) req.gradClipMin = gradClipMin;
  const gradClipMax = parseOptionalNumber(form.gradClipMax);
  if (gradClipMax != null) req.gradClipMax = gradClipMax;
  const pDisableGradClip = parseOptionalNumber(form.pDisableGradClip);
  if (pDisableGradClip != null) req.pDisableGradClip = pDisableGradClip;

  const slippageMax = parseOptionalNumber(form.slippageMax);
  if (slippageMax != null) req.slippageMax = slippageMax;
  const spreadMax = parseOptionalNumber(form.spreadMax);
  if (spreadMax != null) req.spreadMax = spreadMax;

  const minRoundTrips = parseOptionalInt(form.minRoundTrips);
  if (minRoundTrips != null) req.minRoundTrips = minRoundTrips;
  const minWinRate = parseOptionalNumber(form.minWinRate);
  if (minWinRate != null) req.minWinRate = minWinRate;
  const minSharpe = parseOptionalNumber(form.minSharpe);
  if (minSharpe != null) req.minSharpe = minSharpe;
  const minAnnualizedReturn = parseOptionalNumber(form.minAnnualizedReturn);
  if (minAnnualizedReturn != null) req.minAnnualizedReturn = minAnnualizedReturn;
  const minCalmar = parseOptionalNumber(form.minCalmar);
  if (minCalmar != null) req.minCalmar = minCalmar;
  const minProfitFactor = parseOptionalNumber(form.minProfitFactor);
  if (minProfitFactor != null) req.minProfitFactor = minProfitFactor;
  const maxTurnover = parseOptionalNumber(form.maxTurnover);
  if (maxTurnover != null) req.maxTurnover = maxTurnover;
  const minExposure = parseOptionalNumber(form.minExposure);
  if (minExposure != null) req.minExposure = minExposure;
  const minWalkForwardSharpeMean = parseOptionalNumber(form.minWalkForwardSharpeMean);
  if (minWalkForwardSharpeMean != null) req.minWalkForwardSharpeMean = minWalkForwardSharpeMean;
  const maxWalkForwardSharpeStd = parseOptionalNumber(form.maxWalkForwardSharpeStd);
  if (maxWalkForwardSharpeStd != null) req.maxWalkForwardSharpeStd = maxWalkForwardSharpeStd;

  const walkForwardFoldsMin = parseOptionalInt(form.walkForwardFoldsMin);
  if (walkForwardFoldsMin != null) req.walkForwardFoldsMin = walkForwardFoldsMin;
  const walkForwardFoldsMax = parseOptionalInt(form.walkForwardFoldsMax);
  if (walkForwardFoldsMax != null) req.walkForwardFoldsMax = walkForwardFoldsMax;

  const minHoldBarsMin = parseOptionalInt(form.minHoldBarsMin);
  if (minHoldBarsMin != null) req.minHoldBarsMin = minHoldBarsMin;
  const minHoldBarsMax = parseOptionalInt(form.minHoldBarsMax);
  if (minHoldBarsMax != null) req.minHoldBarsMax = minHoldBarsMax;
  const cooldownBarsMin = parseOptionalInt(form.cooldownBarsMin);
  if (cooldownBarsMin != null) req.cooldownBarsMin = cooldownBarsMin;
  const cooldownBarsMax = parseOptionalInt(form.cooldownBarsMax);
  if (cooldownBarsMax != null) req.cooldownBarsMax = cooldownBarsMax;
  const maxHoldBarsMin = parseOptionalInt(form.maxHoldBarsMin);
  if (maxHoldBarsMin != null) req.maxHoldBarsMin = maxHoldBarsMin;
  const maxHoldBarsMax = parseOptionalInt(form.maxHoldBarsMax);
  if (maxHoldBarsMax != null) req.maxHoldBarsMax = maxHoldBarsMax;

  const minEdgeMin = parseOptionalNumber(form.minEdgeMin);
  if (minEdgeMin != null) req.minEdgeMin = minEdgeMin;
  const minEdgeMax = parseOptionalNumber(form.minEdgeMax);
  if (minEdgeMax != null) req.minEdgeMax = minEdgeMax;
  const minSignalToNoiseMin = parseOptionalNumber(form.minSignalToNoiseMin);
  if (minSignalToNoiseMin != null) req.minSignalToNoiseMin = minSignalToNoiseMin;
  const minSignalToNoiseMax = parseOptionalNumber(form.minSignalToNoiseMax);
  if (minSignalToNoiseMax != null) req.minSignalToNoiseMax = minSignalToNoiseMax;
  const edgeBufferMin = parseOptionalNumber(form.edgeBufferMin);
  if (edgeBufferMin != null) req.edgeBufferMin = edgeBufferMin;
  const edgeBufferMax = parseOptionalNumber(form.edgeBufferMax);
  if (edgeBufferMax != null) req.edgeBufferMax = edgeBufferMax;
  const trendLookbackMin = parseOptionalInt(form.trendLookbackMin);
  if (trendLookbackMin != null) req.trendLookbackMin = trendLookbackMin;
  const trendLookbackMax = parseOptionalInt(form.trendLookbackMax);
  if (trendLookbackMax != null) req.trendLookbackMax = trendLookbackMax;
  const pCostAwareEdge = parseOptionalNumber(form.pCostAwareEdge);
  if (pCostAwareEdge != null) req.pCostAwareEdge = pCostAwareEdge;

  const stopMin = parseOptionalNumber(form.stopMin);
  if (stopMin != null) req.stopMin = stopMin;
  const stopMax = parseOptionalNumber(form.stopMax);
  if (stopMax != null) req.stopMax = stopMax;
  const tpMin = parseOptionalNumber(form.tpMin);
  if (tpMin != null) req.tpMin = tpMin;
  const tpMax = parseOptionalNumber(form.tpMax);
  if (tpMax != null) req.tpMax = tpMax;
  const trailMin = parseOptionalNumber(form.trailMin);
  if (trailMin != null) req.trailMin = trailMin;
  const trailMax = parseOptionalNumber(form.trailMax);
  if (trailMax != null) req.trailMax = trailMax;

  const methodWeightBlend = parseOptionalNumber(form.methodWeightBlend);
  if (methodWeightBlend != null) req.methodWeightBlend = methodWeightBlend;
  const blendWeightMin = parseOptionalNumber(form.blendWeightMin);
  if (blendWeightMin != null) req.blendWeightMin = blendWeightMin;
  const blendWeightMax = parseOptionalNumber(form.blendWeightMax);
  if (blendWeightMax != null) req.blendWeightMax = blendWeightMax;

  if (form.disableLstmPersistence) req.disableLstmPersistence = true;
  if (form.noSweepThreshold) req.noSweepThreshold = true;

  if (extras) {
    Object.assign(req, extras);
  }

  return req;
}

export const CUSTOM_SYMBOL_VALUE = "__custom__";
export const TOP_COMBOS_POLL_MS = 30_000;
export const TOP_COMBOS_DISPLAY_DEFAULT = 5;
export const TOP_COMBOS_DISPLAY_MIN = 1;
export const TOP_COMBOS_BOT_TARGET = 5;
export const MIN_LOOKBACK_BARS = 2;
export const MIN_BACKTEST_BARS = 2;
export const MIN_BACKTEST_RATIO = 0.01;
export const MAX_BACKTEST_RATIO = 0.99;

export const DURATION_UNITS: Array<{ unit: string; seconds: number }> = [
  { unit: "M", seconds: 30 * 24 * 60 * 60 },
  { unit: "w", seconds: 7 * 24 * 60 * 60 },
  { unit: "d", seconds: 24 * 60 * 60 },
  { unit: "h", seconds: 60 * 60 },
  { unit: "m", seconds: 60 },
  { unit: "s", seconds: 1 },
];

export function formatDurationSeconds(totalSeconds: number): string {
  const sec = Math.max(1, Math.round(totalSeconds));
  for (const { unit, seconds } of DURATION_UNITS) {
    if (sec % seconds === 0) return `${sec / seconds}${unit}`;
  }
  return `${sec}s`;
}

export function sigNumber(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "";
  const rounded = Math.round(value * 1e8) / 1e8;
  return String(rounded);
}

export function coerceNumber(value: number | null | undefined, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

export function clampOptionalRatio(value: number | null | undefined): number {
  const v = coerceNumber(value, 0);
  return v > 0 ? clamp(v, 0, 0.999999) : 0;
}

export function clampOptionalRange(value: number | null | undefined, min: number, max: number): number {
  const v = coerceNumber(value, 0);
  return v > 0 ? clamp(v, min, max) : 0;
}

export function clampOptionalInt(value: number | null | undefined, min: number, max: number): number {
  const v = coerceNumber(value, 0);
  return v > 0 ? clamp(Math.trunc(v), min, max) : 0;
}

export function sigText(value: string | null | undefined): string {
  return typeof value === "string" ? value : "";
}

export function sigBool(value: boolean | null | undefined): string {
  return value ? "1" : "0";
}

export function formatDirectionLabel(value: LatestSignal["closeDirection"]): string {
  if (value === undefined) return "—";
  return value ?? "NEUTRAL";
}

export type DecisionCheckStatus = "ok" | "warn" | "bad" | "skip";

export type DecisionCheck = {
  id: string;
  label: string;
  status: DecisionCheckStatus;
  detail: string;
};

export type DecisionSummary = {
  isHold: boolean;
  reason: string | null;
  checks: DecisionCheck[];
};

export const DIRECTION_HOLD_REASONS = new Set([
  "DIRECTIONS_DISAGREE",
  "BOTH_NEUTRAL",
  "KALMAN_NEUTRAL",
  "LSTM_NEUTRAL",
  "BLEND_NEUTRAL",
  "ROUTER_NEUTRAL",
]);

export function isFiniteNumber(value: number | null | undefined): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

export function formatSignalDirection(value: LatestSignal["chosenDirection"]): string {
  return value ?? "NEUTRAL";
}

export function parseActionReason(action: string): string | null {
  const match = /\(([^)]+)\)/.exec(action);
  if (!match) return null;
  const reason = match[1]?.trim() ?? "";
  return reason ? reason : null;
}

export function normalizeHoldReason(reason: string | null): string | null {
  if (!reason) return null;
  const clean = reason.trim();
  if (!clean) return null;
  return clean.toUpperCase().replace(/\s+/g, "_");
}

export function decisionDotClass(status: DecisionCheckStatus): string {
  switch (status) {
    case "ok":
      return "dot dotOk";
    case "bad":
      return "dot dotBad";
    case "warn":
      return "dot dotWarn";
    default:
      return "dot";
  }
}

export function decisionBadgeClass(status: DecisionCheckStatus): string {
  switch (status) {
    case "ok":
      return "badge badgeOk";
    case "bad":
      return "badge badgeBad";
    case "warn":
      return "badge badgeWarn";
    default:
      return "badge";
  }
}

export function decisionStatusLabel(status: DecisionCheckStatus): string {
  switch (status) {
    case "ok":
      return "pass";
    case "bad":
      return "block";
    case "warn":
      return "needs data";
    default:
      return "off";
  }
}

export const SECONDS_PER_YEAR = 365 * 24 * 60 * 60;

export function inferPeriodsPerYear(platform: Platform, interval: string): number | null {
  const seconds = platformIntervalSeconds(platform, interval);
  if (!seconds || seconds <= 0) return null;
  const out = SECONDS_PER_YEAR / seconds;
  return Number.isFinite(out) && out > 0 ? out : null;
}

export type SplitStats = {
  trainEndRaw: number;
  backtestBars: number;
  tuneBars: number;
  fitBars: number;
  trainOk: boolean;
  backtestOk: boolean;
  tuneOk: boolean;
  fitOk: boolean;
};

export type TuneRatioBounds = {
  trainEndRaw: number;
  minTrainBars: number;
  minTuneBars: number;
  maxTuneBars: number;
  minRatio: number;
  maxRatio: number;
};

export const RATIO_ROUND_DIGITS = 3;
export const RATIO_ROUND_FACTOR = 10 ** RATIO_ROUND_DIGITS;

export function splitStats(
  bars: number,
  backtestRatio: number,
  lookbackBars: number,
  tuneRatio: number,
  tuningEnabled: boolean,
): SplitStats {
  const ratio = clamp(backtestRatio, MIN_BACKTEST_RATIO, MAX_BACKTEST_RATIO);
  const trainEndRaw = Math.floor(bars * (1 - ratio) + 1e-9);
  const backtestBars = Math.max(0, bars - trainEndRaw);
  const minTrainBars = lookbackBars + 1;
  const trainOk = trainEndRaw >= minTrainBars;
  const backtestOk = backtestBars >= MIN_BACKTEST_BARS;
  let tuneBars = 0;
  let fitBars = trainEndRaw;
  let tuneOk = true;
  let fitOk = true;
  if (tuningEnabled) {
    const tuneRatioSafe = clamp(tuneRatio, 0, 0.99);
    tuneBars = Math.max(0, Math.min(trainEndRaw, Math.floor(trainEndRaw * tuneRatioSafe)));
    fitBars = Math.max(0, trainEndRaw - tuneBars);
    tuneOk = tuneBars >= 2;
    fitOk = fitBars >= minTrainBars;
  }
  return { trainEndRaw, backtestBars, tuneBars, fitBars, trainOk, backtestOk, tuneOk, fitOk };
}

export function roundRatioDown(value: number): number {
  return Math.floor(value * RATIO_ROUND_FACTOR) / RATIO_ROUND_FACTOR;
}

export function roundRatioUp(value: number): number {
  return Math.ceil(value * RATIO_ROUND_FACTOR) / RATIO_ROUND_FACTOR;
}

export function tuneRatioBounds(bars: number, backtestRatio: number, lookbackBars: number): TuneRatioBounds | null {
  if (!Number.isFinite(bars) || bars <= 0) return null;
  if (!Number.isFinite(lookbackBars) || lookbackBars < MIN_LOOKBACK_BARS) return null;
  const ratio = clamp(backtestRatio, MIN_BACKTEST_RATIO, MAX_BACKTEST_RATIO);
  const trainEndRaw = Math.floor(bars * (1 - ratio) + 1e-9);
  const minTrainBars = lookbackBars + 1;
  if (trainEndRaw < minTrainBars) return null;
  const minTuneBars = 2;
  const maxTuneBars = trainEndRaw - minTrainBars;
  const minRatio = minTuneBars / trainEndRaw;
  const maxRatio = maxTuneBars / trainEndRaw;
  return {
    trainEndRaw,
    minTrainBars,
    minTuneBars,
    maxTuneBars,
    minRatio: clamp(minRatio, 0, 0.99),
    maxRatio: clamp(maxRatio, 0, 0.99),
  };
}

export function maxBarsForPlatform(platform: Platform, method: Method, apiLimits: ComputeLimits | null): number {
  let maxBars = method === "10" ? (platform === "binance" ? 1000 : Number.POSITIVE_INFINITY) : 1000;
  if (method !== "10") {
    const maxBarsRaw = apiLimits ? Math.trunc(apiLimits.maxBarsLstm) : NaN;
    if (Number.isFinite(maxBarsRaw) && maxBarsRaw > 0) {
      maxBars = maxBarsRaw;
    }
  }
  return maxBars;
}

export function maxLookbackForSplit(bars: number, backtestRatio: number, tuneRatio: number, tuningEnabled: boolean): number | null {
  if (!Number.isFinite(bars) || bars <= 0) return null;
  const ratio = clamp(backtestRatio, MIN_BACKTEST_RATIO, MAX_BACKTEST_RATIO);
  const trainEndRaw = Math.floor(bars * (1 - ratio) + 1e-9);
  if (trainEndRaw <= 0) return null;
  if (!tuningEnabled) return trainEndRaw - 1;
  const tuneRatioSafe = clamp(tuneRatio, 0, 0.99);
  const tuneBars = Math.max(0, Math.min(trainEndRaw, Math.floor(trainEndRaw * tuneRatioSafe)));
  const fitBars = Math.max(0, trainEndRaw - tuneBars);
  return fitBars - 1;
}

export function minTrainEndForTune(minTrainBars: number, tuneRatio: number, tuningEnabled: boolean, maxTrainEnd: number): number {
  if (!tuningEnabled) return minTrainBars;
  const ratio = clamp(tuneRatio, 0, 0.99);
  if (ratio <= 0) return minTrainBars;
  for (let trainEnd = minTrainBars; trainEnd <= maxTrainEnd; trainEnd += 1) {
    const tuneBars = Math.floor(trainEnd * ratio);
    const fitBars = trainEnd - tuneBars;
    if (tuneBars >= 2 && fitBars >= minTrainBars) return trainEnd;
  }
  return minTrainBars;
}

export function ratioForTrainEnd(bars: number, trainEnd: number): number {
  const raw = 1 - (trainEnd + 0.5) / Math.max(1, bars);
  return clamp(raw, MIN_BACKTEST_RATIO, MAX_BACKTEST_RATIO);
}

export function clampComboForLimits(combo: OptimizationCombo, apiLimits: ComputeLimits | null, platform: Platform): {
  bars: number;
  epochs: number;
  hiddenSize: number;
} {
  const lstmEnabled = combo.params.method !== "10";
  let bars = Math.trunc(combo.params.bars);
  if (!Number.isFinite(bars) || bars < 0) bars = 0;
  if (bars > 0) {
    bars = Math.max(MIN_LOOKBACK_BARS, bars);
    const barsCap = maxBarsForPlatform(platform, combo.params.method, apiLimits);
    if (Number.isFinite(barsCap)) {
      bars = Math.min(bars, barsCap);
    }
  }

  let epochs = clamp(Math.trunc(combo.params.epochs), 0, 5000);
  let hiddenSize = clamp(Math.trunc(combo.params.hiddenSize), 1, 512);

  if (lstmEnabled && apiLimits) {
    epochs = Math.min(epochs, apiLimits.maxEpochs);
    hiddenSize = Math.min(hiddenSize, apiLimits.maxHiddenSize);
  }

  return { bars, epochs, hiddenSize };
}

export function applyComboToForm(
  prev: FormState,
  combo: OptimizationCombo,
  apiLimits: ComputeLimits | null,
  manualOverrides?: Set<ManualOverrideKey>,
  allowPositioning = true,
): FormState {
  const nextPlatform = combo.params.platform ?? prev.platform;
  const comboSymbolRaw = combo.params.binanceSymbol?.trim() ?? "";
  const normalizedComboSymbol = comboSymbolRaw ? normalizeComboSymbol(comboSymbolRaw, nextPlatform) : "";
  const comboSymbol =
    normalizedComboSymbol && symbolFormatPattern(nextPlatform).test(normalizedComboSymbol) ? normalizedComboSymbol : "";
  const prevSymbol = prev.binanceSymbol.trim().toUpperCase();
  const prevSymbolValid = symbolFormatPattern(nextPlatform).test(prevSymbol);
  const fallbackSymbol = PLATFORM_DEFAULT_SYMBOL[nextPlatform] ?? prev.binanceSymbol;
  const symbol = comboSymbol || (prevSymbolValid ? prevSymbol : fallbackSymbol);
  const interval = combo.params.interval;
  const method = manualOverrides?.has("method") ? prev.method : combo.params.method;
  const comboPositioning = combo.params.positioning ?? prev.positioning;
  const positioning = allowPositioning ? comboPositioning : prev.positioning;
  const normalization = combo.params.normalization;
  const intrabarFill = combo.params.intrabarFill ?? prev.intrabarFill;
  const confirmConformal = combo.params.confirmConformal ?? prev.confirmConformal;
  const confirmQuantiles = combo.params.confirmQuantiles ?? prev.confirmQuantiles;
  const confidenceSizing = combo.params.confidenceSizing ?? prev.confidenceSizing;

  const comboForLimits =
    method === combo.params.method
      ? combo
      : {
          ...combo,
          params: { ...combo.params, method },
        };
  const { bars, epochs, hiddenSize } = clampComboForLimits(comboForLimits, apiLimits, nextPlatform);
  const openThrRaw = coerceNumber(combo.openThreshold, prev.openThreshold);
  const closeThrRaw =
    combo.closeThreshold == null ? openThrRaw : coerceNumber(combo.closeThreshold, prev.closeThreshold);
  const openThreshold = manualOverrides?.has("openThreshold") ? prev.openThreshold : Math.max(0, openThrRaw);
  const closeThreshold = manualOverrides?.has("closeThreshold") ? prev.closeThreshold : Math.max(0, closeThrRaw);
  const fee = Math.max(0, coerceNumber(combo.params.fee, prev.fee));
  const learningRate = Math.max(1e-9, coerceNumber(combo.params.learningRate, prev.learningRate));
  const valRatio = clamp(coerceNumber(combo.params.valRatio, prev.valRatio), 0, 1);
  const patience = clamp(Math.trunc(coerceNumber(combo.params.patience, prev.patience)), 0, 1000);
  const gradClipRaw = coerceNumber(combo.params.gradClip, 0);
  const gradClip = gradClipRaw > 0 ? clamp(gradClipRaw, 0, 100) : 0;

  const slippage = clampOptionalRatio(combo.params.slippage);
  const spread = clampOptionalRatio(combo.params.spread);
  const stopLoss = clampOptionalRatio(combo.params.stopLoss);
  const takeProfit = clampOptionalRatio(combo.params.takeProfit);
  const trailingStop = clampOptionalRatio(combo.params.trailingStop);
  const stopLossVolMult = Math.max(0, coerceNumber(combo.params.stopLossVolMult ?? prev.stopLossVolMult, prev.stopLossVolMult));
  const takeProfitVolMult = Math.max(0, coerceNumber(combo.params.takeProfitVolMult ?? prev.takeProfitVolMult, prev.takeProfitVolMult));
  const trailingStopVolMult = Math.max(
    0,
    coerceNumber(combo.params.trailingStopVolMult ?? prev.trailingStopVolMult, prev.trailingStopVolMult),
  );
  const minHoldBars = clampOptionalInt(combo.params.minHoldBars ?? prev.minHoldBars, 0, 1_000_000);
  const maxHoldBars = clampOptionalInt(combo.params.maxHoldBars ?? prev.maxHoldBars, 0, 1_000_000);
  const cooldownBars = clampOptionalInt(combo.params.cooldownBars ?? prev.cooldownBars, 0, 1_000_000);
  const maxDrawdown = clampOptionalRatio(combo.params.maxDrawdown);
  const maxDailyLoss = clampOptionalRatio(combo.params.maxDailyLoss);
  const maxOrderErrors = clampOptionalInt(combo.params.maxOrderErrors, 1, 1_000_000);
  const minEdge = Math.max(0, coerceNumber(combo.params.minEdge ?? prev.minEdge, prev.minEdge));
  const minSignalToNoise = Math.max(0, coerceNumber(combo.params.minSignalToNoise ?? prev.minSignalToNoise, prev.minSignalToNoise));
  const costAwareEdge = combo.params.costAwareEdge ?? prev.costAwareEdge;
  const edgeBuffer = Math.max(0, coerceNumber(combo.params.edgeBuffer ?? prev.edgeBuffer, prev.edgeBuffer));
  const trendLookback = clampOptionalInt(combo.params.trendLookback ?? prev.trendLookback, 0, 1_000_000);
  const maxPositionSize = Math.max(0, coerceNumber(combo.params.maxPositionSize ?? prev.maxPositionSize, prev.maxPositionSize));
  const volTarget = Math.max(0, coerceNumber(combo.params.volTarget ?? prev.volTarget, prev.volTarget));
  const volLookback = Math.max(0, Math.trunc(coerceNumber(combo.params.volLookback ?? prev.volLookback, prev.volLookback)));
  const volEwmaAlphaRaw = coerceNumber(combo.params.volEwmaAlpha ?? prev.volEwmaAlpha, prev.volEwmaAlpha);
  const volEwmaAlpha = volEwmaAlphaRaw > 0 && volEwmaAlphaRaw < 1 ? volEwmaAlphaRaw : 0;
  const volFloor = Math.max(0, coerceNumber(combo.params.volFloor ?? prev.volFloor, prev.volFloor));
  const volScaleMax = Math.max(0, coerceNumber(combo.params.volScaleMax ?? prev.volScaleMax, prev.volScaleMax));
  const maxVolatility = Math.max(0, coerceNumber(combo.params.maxVolatility ?? prev.maxVolatility, prev.maxVolatility));
  const rebalanceBars = clampOptionalInt(combo.params.rebalanceBars ?? prev.rebalanceBars, 0, 1_000_000);
  const rebalanceThreshold = Math.max(
    0,
    coerceNumber(combo.params.rebalanceThreshold ?? prev.rebalanceThreshold, prev.rebalanceThreshold),
  );
  const rebalanceGlobal = combo.params.rebalanceGlobal ?? prev.rebalanceGlobal;
  const rebalanceResetOnSignal = combo.params.rebalanceResetOnSignal ?? prev.rebalanceResetOnSignal;
  const fundingRate = coerceNumber(combo.params.fundingRate ?? prev.fundingRate, prev.fundingRate);
  const fundingBySide = combo.params.fundingBySide ?? prev.fundingBySide;
  const fundingOnOpen = combo.params.fundingOnOpen ?? prev.fundingOnOpen;
  const blendWeight = clamp(coerceNumber(combo.params.blendWeight ?? prev.blendWeight, prev.blendWeight), 0, 1);
  const tuneStressVolMult = Math.max(0, coerceNumber(combo.params.tuneStressVolMult ?? prev.tuneStressVolMult, prev.tuneStressVolMult));
  const tuneStressShock = coerceNumber(combo.params.tuneStressShock ?? prev.tuneStressShock, prev.tuneStressShock);
  const tuneStressWeight = Math.max(0, coerceNumber(combo.params.tuneStressWeight ?? prev.tuneStressWeight, prev.tuneStressWeight));
  const walkForwardFolds = clampOptionalInt(combo.params.walkForwardFolds ?? prev.walkForwardFolds, 1, 1000);

  const kalmanZMin = Math.max(0, coerceNumber(combo.params.kalmanZMin, prev.kalmanZMin));
  const kalmanZMax = Math.max(Math.max(0, coerceNumber(combo.params.kalmanZMax, prev.kalmanZMax)), kalmanZMin);
  const maxHighVolProb = clampOptionalRange(combo.params.maxHighVolProb, 0, 1);
  const maxConformalWidthRaw = coerceNumber(combo.params.maxConformalWidth, 0);
  const maxConformalWidth = maxConformalWidthRaw > 0 ? Math.max(0, maxConformalWidthRaw) : 0;
  const maxQuantileWidthRaw = coerceNumber(combo.params.maxQuantileWidth, 0);
  const maxQuantileWidth = maxQuantileWidthRaw > 0 ? Math.max(0, maxQuantileWidthRaw) : 0;
  const minPositionSize = clampOptionalRange(combo.params.minPositionSize, 0, 1);
  const comboOrderQuantity = coerceNumber(combo.params.orderQuantity, 0);
  const comboOrderQuote = coerceNumber(combo.params.orderQuote, 0);
  const comboOrderQuoteFraction = clampOptionalRange(combo.params.orderQuoteFraction, 0, 1);
  const comboMaxOrderQuote = Math.max(0, coerceNumber(combo.params.maxOrderQuote, 0));
  const hasComboSizing =
    combo.params.orderQuantity != null ||
    combo.params.orderQuote != null ||
    combo.params.orderQuoteFraction != null ||
    combo.params.maxOrderQuote != null;

  let orderQuantity = prev.orderQuantity;
  let orderQuote = prev.orderQuote;
  let orderQuoteFraction = prev.orderQuoteFraction;
  let maxOrderQuote = prev.maxOrderQuote;

  if (hasComboSizing) {
    if (comboOrderQuantity > 0) {
      orderQuantity = comboOrderQuantity;
      orderQuote = 0;
      orderQuoteFraction = 0;
      maxOrderQuote = 0;
    } else if (comboOrderQuote > 0) {
      orderQuote = comboOrderQuote;
      orderQuantity = 0;
      orderQuoteFraction = 0;
      maxOrderQuote = 0;
    } else if (comboOrderQuoteFraction > 0) {
      orderQuoteFraction = comboOrderQuoteFraction;
      orderQuantity = 0;
      orderQuote = 0;
      maxOrderQuote = comboMaxOrderQuote > 0 ? comboMaxOrderQuote : 0;
    } else {
      orderQuantity = 0;
      orderQuote = 0;
      orderQuoteFraction = 0;
      maxOrderQuote = 0;
    }
  }

  let lookbackBars = prev.lookbackBars;
  let lookbackWindow = prev.lookbackWindow;
  const intervalChanged = interval !== prev.interval;
  const prevLookbackBars = lookbackBars;

  if (intervalChanged && prevLookbackBars >= MIN_LOOKBACK_BARS) {
    const prevIntervalSec = platformIntervalSeconds(prev.platform, prev.interval);
    lookbackBars = 0;
    if (prevIntervalSec) {
      lookbackWindow = formatDurationSeconds(prevLookbackBars * prevIntervalSec);
    } else {
      const trimmed = lookbackWindow.trim();
      lookbackWindow = trimmed ? trimmed : defaultForm.lookbackWindow;
    }
  }

  if (lookbackBars >= MIN_LOOKBACK_BARS && bars > 0) {
    const maxLookback = Math.max(0, bars - 1);
    if (maxLookback < MIN_LOOKBACK_BARS) {
      lookbackBars = 0;
    } else if (lookbackBars >= maxLookback) {
      lookbackBars = maxLookback;
    }
  }

  if (lookbackBars < MIN_LOOKBACK_BARS) {
    const intervalSec = platformIntervalSeconds(nextPlatform, interval);
    if (intervalSec) {
      const windowSec = parseDurationSeconds(lookbackWindow);
      const minWindowSec = intervalSec * MIN_LOOKBACK_BARS;
      if (!windowSec || windowSec < minWindowSec) {
        lookbackWindow = formatDurationSeconds(minWindowSec);
      }
    }
  }

  return {
    ...prev,
    binanceSymbol: symbol,
    platform: nextPlatform,
    market: nextPlatform === "binance" ? prev.market : "spot",
    interval,
    bars,
    method,
    positioning,
    normalization,
    fee,
    epochs,
    hiddenSize,
    learningRate,
    valRatio,
    patience,
    gradClip,
    slippage,
    spread,
    intrabarFill,
    stopLoss,
    takeProfit,
    trailingStop,
    stopLossVolMult,
    takeProfitVolMult,
    trailingStopVolMult,
    minHoldBars,
    maxHoldBars,
    cooldownBars,
    maxDrawdown,
    maxDailyLoss,
    maxOrderErrors,
    minEdge,
    minSignalToNoise,
    costAwareEdge,
    edgeBuffer,
    trendLookback,
    maxPositionSize,
    volTarget,
    volLookback,
    volEwmaAlpha,
    volFloor,
    volScaleMax,
    maxVolatility,
    rebalanceBars,
    rebalanceThreshold,
    rebalanceGlobal,
    rebalanceResetOnSignal,
    fundingRate,
    fundingBySide,
    fundingOnOpen,
    blendWeight,
    kalmanZMin,
    kalmanZMax,
    maxHighVolProb,
    maxConformalWidth,
    maxQuantileWidth,
    confirmConformal,
    confirmQuantiles,
    confidenceSizing,
    minPositionSize,
    binanceTestnet: nextPlatform === "binance" ? prev.binanceTestnet : false,
    binanceLive: nextPlatform === "binance" ? prev.binanceLive : false,
    tradeArmed: nextPlatform === "binance" ? prev.tradeArmed : false,
    orderQuantity,
    orderQuote,
    orderQuoteFraction,
    maxOrderQuote,
    tuneStressVolMult,
    tuneStressShock,
    tuneStressWeight,
    walkForwardFolds,
    lookbackBars,
    lookbackWindow,
    openThreshold,
    closeThreshold,
  };
}

export function comboApplySignature(
  combo: OptimizationCombo,
  apiLimits: ComputeLimits | null,
  baseForm: FormState,
  manualOverrides?: Set<ManualOverrideKey>,
  allowPositioning = true,
): string {
  return formApplySignature(applyComboToForm(baseForm, combo, apiLimits, manualOverrides, allowPositioning));
}

export function comboAnnualizedEquity(combo: OptimizationCombo): number | null {
  const annReturn = combo.metrics?.annualizedReturn;
  if (typeof annReturn !== "number" || !Number.isFinite(annReturn)) return null;
  const annEq = annReturn + 1;
  return Number.isFinite(annEq) ? annEq : null;
}

export function formApplySignature(form: FormState): string {
  let bars = Math.trunc(form.bars);
  if (!Number.isFinite(bars) || bars < 0) bars = 0;
  if (bars > 0 && bars < MIN_LOOKBACK_BARS) {
    bars = MIN_LOOKBACK_BARS;
  }
  const lookbackBarsRaw = Math.trunc(form.lookbackBars);
  const lookbackOverride = lookbackBarsRaw >= MIN_LOOKBACK_BARS;
  const lookbackBars = lookbackOverride ? lookbackBarsRaw : 0;
  const lookbackWindow = lookbackOverride ? "" : form.lookbackWindow.trim();
  const epochs = Math.max(0, Math.trunc(form.epochs));
  const hiddenSize = Math.max(1, Math.trunc(form.hiddenSize));
  const symbol = form.binanceSymbol.trim().toUpperCase();

  return [
    sigText(symbol),
    sigText(form.platform),
    sigText(form.interval.trim()),
    String(bars),
    lookbackOverride ? String(lookbackBars) : "",
    sigText(lookbackWindow),
    sigText(form.method),
    sigText(form.positioning),
    sigText(form.normalization),
    sigNumber(form.fee),
    String(epochs),
    String(hiddenSize),
    sigNumber(form.learningRate),
    sigNumber(form.valRatio),
    sigNumber(form.patience),
    sigNumber(form.gradClip),
    sigNumber(form.slippage),
    sigNumber(form.spread),
    sigText(form.intrabarFill),
    sigNumber(form.stopLoss),
    sigNumber(form.takeProfit),
    sigNumber(form.trailingStop),
    sigNumber(form.stopLossVolMult),
    sigNumber(form.takeProfitVolMult),
    sigNumber(form.trailingStopVolMult),
    sigNumber(form.minHoldBars),
    sigNumber(form.maxHoldBars),
    sigNumber(form.cooldownBars),
    sigNumber(form.maxDrawdown),
    sigNumber(form.maxDailyLoss),
    sigNumber(form.maxOrderErrors),
    sigNumber(form.orderQuantity),
    sigNumber(form.orderQuote),
    sigNumber(form.orderQuoteFraction),
    sigNumber(form.maxOrderQuote),
    sigNumber(form.minEdge),
    sigNumber(form.minSignalToNoise),
    sigBool(form.costAwareEdge),
    sigNumber(form.edgeBuffer),
    sigNumber(form.trendLookback),
    sigNumber(form.maxPositionSize),
    sigNumber(form.volTarget),
    sigNumber(form.volLookback),
    sigNumber(form.volEwmaAlpha),
    sigNumber(form.volFloor),
    sigNumber(form.volScaleMax),
    sigNumber(form.maxVolatility),
    sigNumber(form.rebalanceBars),
    sigNumber(form.rebalanceThreshold),
    sigBool(form.rebalanceGlobal),
    sigBool(form.rebalanceResetOnSignal),
    sigNumber(form.fundingRate),
    sigBool(form.fundingBySide),
    sigBool(form.fundingOnOpen),
    sigNumber(form.blendWeight),
    sigNumber(form.kalmanZMin),
    sigNumber(form.kalmanZMax),
    sigNumber(form.maxHighVolProb),
    sigNumber(form.maxConformalWidth),
    sigNumber(form.maxQuantileWidth),
    sigBool(form.confirmConformal),
    sigBool(form.confirmQuantiles),
    sigBool(form.confidenceSizing),
    sigNumber(form.minPositionSize),
    sigNumber(form.tuneStressVolMult),
    sigNumber(form.tuneStressShock),
    sigNumber(form.tuneStressWeight),
    sigNumber(form.walkForwardFolds),
    sigNumber(form.openThreshold),
    sigNumber(form.closeThreshold),
  ].join("|");
}
