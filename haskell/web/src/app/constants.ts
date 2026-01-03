import type { Platform } from "../lib/types";
import { TRADER_UI_CONFIG } from "../lib/deployConfig";

export const STORAGE_KEY = "trader.ui.form.v1";
export const STORAGE_PROFILES_KEY = "trader.ui.formProfiles.v1";
export const STORAGE_PERSIST_SECRETS_KEY = "trader.ui.persistSecrets.v1";
export const SESSION_BINANCE_KEY_KEY = "trader.ui.binanceApiKey.v1";
export const SESSION_BINANCE_SECRET_KEY = "trader.ui.binanceApiSecret.v1";
export const SESSION_COINBASE_KEY_KEY = "trader.ui.coinbaseApiKey.v1";
export const SESSION_COINBASE_SECRET_KEY = "trader.ui.coinbaseApiSecret.v1";
export const SESSION_COINBASE_PASSPHRASE_KEY = "trader.ui.coinbaseApiPassphrase.v1";
export const STORAGE_ORDER_LOG_PREFS_KEY = "trader.ui.orderLogPrefs.v1";
export const STORAGE_PANEL_PREFS_KEY = "trader.ui.panelPrefs.v1";
export const STORAGE_CONFIG_PANEL_ORDER_KEY = "trader.ui.configPanelOrder.v1";

const DEFAULT_SIGNAL_TIMEOUT_MS = 10 * 60_000;
const DEFAULT_BACKTEST_TIMEOUT_MS = 20 * 60_000;
const DEFAULT_TRADE_TIMEOUT_MS = 10 * 60_000;
const DEFAULT_BOT_START_TIMEOUT_MS = 20 * 60_000;
const DEFAULT_BOT_STATUS_TIMEOUT_MS = 60_000;

function resolveTimeoutMs(key: "signalMs" | "backtestMs" | "tradeMs" | "botStartMs" | "botStatusMs", fallback: number): number {
  const v = TRADER_UI_CONFIG.timeoutsMs?.[key];
  return typeof v === "number" && Number.isFinite(v) && v >= 1000 ? v : fallback;
}

export const SIGNAL_TIMEOUT_MS = resolveTimeoutMs("signalMs", DEFAULT_SIGNAL_TIMEOUT_MS);
export const BACKTEST_TIMEOUT_MS = resolveTimeoutMs("backtestMs", DEFAULT_BACKTEST_TIMEOUT_MS);
export const TRADE_TIMEOUT_MS = resolveTimeoutMs("tradeMs", DEFAULT_TRADE_TIMEOUT_MS);
export const BOT_START_TIMEOUT_MS = resolveTimeoutMs("botStartMs", DEFAULT_BOT_START_TIMEOUT_MS);
export const BOT_STATUS_TIMEOUT_MS = resolveTimeoutMs("botStatusMs", DEFAULT_BOT_STATUS_TIMEOUT_MS);
export const BOT_STATUS_TAIL_POINTS = 5000;
export const BOT_TELEMETRY_POINTS = 240;
export const BOT_AUTOSTART_RETRY_MS = 15_000;
export const RATE_LIMIT_BASE_MS = 10_000;
export const RATE_LIMIT_MAX_MS = 120_000;
export const RATE_LIMIT_TOAST_MIN_MS = 12_000;

export const PLATFORMS: Platform[] = ["binance", "coinbase", "kraken", "poloniex"];

export const PLATFORM_LABELS: Record<Platform, string> = {
  binance: "Binance",
  coinbase: "Coinbase",
  kraken: "Kraken",
  poloniex: "Poloniex",
};

export const BINANCE_INTERVALS = [
  "1m",
  "3m",
  "5m",
  "15m",
  "30m",
  "1h",
  "2h",
  "4h",
  "6h",
  "8h",
  "12h",
  "1d",
  "3d",
  "1w",
  "1M",
] as const;

export const BINANCE_INTERVAL_SET = new Set<string>(BINANCE_INTERVALS);

export const COINBASE_INTERVALS = ["1m", "5m", "15m", "1h", "6h", "1d"] as const;
export const KRAKEN_INTERVALS = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"] as const;
export const POLONIEX_INTERVALS = ["5m", "15m", "30m", "2h", "4h", "1d"] as const;

export const PLATFORM_INTERVALS: Record<Platform, readonly string[]> = {
  binance: BINANCE_INTERVALS,
  coinbase: COINBASE_INTERVALS,
  kraken: KRAKEN_INTERVALS,
  poloniex: POLONIEX_INTERVALS,
};

export const PLATFORM_INTERVAL_SET: Record<Platform, Set<string>> = {
  binance: BINANCE_INTERVAL_SET,
  coinbase: new Set<string>(COINBASE_INTERVALS),
  kraken: new Set<string>(KRAKEN_INTERVALS),
  poloniex: new Set<string>(POLONIEX_INTERVALS),
};

export const BINANCE_INTERVAL_SECONDS: Record<string, number> = {
  "1m": 60,
  "3m": 3 * 60,
  "5m": 5 * 60,
  "15m": 15 * 60,
  "30m": 30 * 60,
  "1h": 60 * 60,
  "2h": 2 * 60 * 60,
  "4h": 4 * 60 * 60,
  "6h": 6 * 60 * 60,
  "8h": 8 * 60 * 60,
  "12h": 12 * 60 * 60,
  "1d": 24 * 60 * 60,
  "3d": 3 * 24 * 60 * 60,
  "1w": 7 * 24 * 60 * 60,
  "1M": 30 * 24 * 60 * 60,
};

export const BINANCE_SYMBOLS = [
  "BTCUSDT",
  "ETHUSDT",
  "BNBUSDT",
  "SOLUSDT",
  "XRPUSDT",
  "ADAUSDT",
  "DOGEUSDT",
  "MATICUSDT",
  "AVAXUSDT",
  "LINKUSDT",
  "DOTUSDT",
  "LTCUSDT",
  "BCHUSDT",
  "TRXUSDT",
  "ATOMUSDT",
  "ETCUSDT",
  "UNIUSDT",
  "AAVEUSDT",
  "FILUSDT",
  "NEARUSDT",
  "OPUSDT",
  "ARBUSDT",
  "SUIUSDT",
] as const;

export const BINANCE_SYMBOL_SET = new Set<string>(BINANCE_SYMBOLS);

export const COINBASE_SYMBOLS = [
  "BTC-USD",
  "ETH-USD",
  "SOL-USD",
  "XRP-USD",
  "ADA-USD",
  "DOGE-USD",
  "AVAX-USD",
  "LINK-USD",
  "DOT-USD",
  "LTC-USD",
] as const;

export const KRAKEN_SYMBOLS = [
  "XBTUSD",
  "ETHUSD",
  "XBTUSDT",
  "ETHUSDT",
  "SOLUSD",
  "XRPUSD",
  "ADAUSD",
] as const;

export const POLONIEX_SYMBOLS = [
  "BTC_USDT",
  "ETH_USDT",
  "SOL_USDT",
  "XRP_USDT",
  "DOGE_USDT",
  "ADA_USDT",
] as const;

export const PLATFORM_SYMBOLS: Record<Platform, readonly string[]> = {
  binance: BINANCE_SYMBOLS,
  coinbase: COINBASE_SYMBOLS,
  kraken: KRAKEN_SYMBOLS,
  poloniex: POLONIEX_SYMBOLS,
};

export const PLATFORM_SYMBOL_SET: Record<Platform, Set<string>> = {
  binance: BINANCE_SYMBOL_SET,
  coinbase: new Set<string>(COINBASE_SYMBOLS),
  kraken: new Set<string>(KRAKEN_SYMBOLS),
  poloniex: new Set<string>(POLONIEX_SYMBOLS),
};

export const PLATFORM_DEFAULT_SYMBOL: Record<Platform, string> = {
  binance: "BTCUSDT",
  coinbase: "BTC-USD",
  kraken: "XBTUSD",
  poloniex: "BTC_USDT",
};

export const TUNE_OBJECTIVES = [
  "annualized-equity",
  "final-equity",
  "sharpe",
  "calmar",
  "equity-dd",
  "equity-dd-turnover",
] as const;
export const TUNE_OBJECTIVE_SET = new Set<string>(TUNE_OBJECTIVES);

export const DATA_LOG_COLLAPSED_MAX_LINES = 50;
export const DATA_LOG_AUTO_SCROLL_SLOP_PX = 24;
export const DATA_LOG_BAR_SERIES_KEYS = new Set(["prices", "positions", "equityCurve", "agreementOk", "kalmanPredNext", "lstmPredNext"]);
