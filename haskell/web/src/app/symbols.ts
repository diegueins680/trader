import type { Platform } from "../lib/types";

export const COMMON_QUOTES = ["USDT", "USDC", "FDUSD", "TUSD", "BUSD", "BTC", "ETH", "BNB"];
export const BINANCE_SYMBOL_PATTERN = /^[A-Z0-9]{3,30}$/;

function normalizeSymbolText(raw: string): string {
  return raw.trim().toUpperCase();
}

function splitAlphaNumTokens(raw: string): string[] {
  return normalizeSymbolText(raw).split(/[^A-Z0-9]+/).filter(Boolean);
}

function delimitedSymbolPattern(delim: "-" | "_"): RegExp {
  return delim === "-" ? /^[A-Z0-9]+-[A-Z0-9]+$/ : /^[A-Z0-9]+_[A-Z0-9]+$/;
}

function sanitizeDelimitedSymbol(raw: string, delim: "-" | "_"): string | null {
  const value = normalizeSymbolText(raw);
  if (!value) return null;

  const pattern = delimitedSymbolPattern(delim);
  if (pattern.test(value)) return value;

  const replaced = value.replace(/[-_/]/g, delim);
  if (pattern.test(replaced)) return replaced;

  const tokens = splitAlphaNumTokens(replaced);
  if (tokens.length !== 2) return null;
  const candidate = `${tokens[0]}${delim}${tokens[1]}`;
  return pattern.test(candidate) ? candidate : null;
}

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

function trimTokenizedBinanceComboSuffix(tokens: string[]): string | null {
  if (tokens.length < 2 || tokens.length > 3) return null;
  const suffix = tokens[tokens.length - 1] ?? "";
  if (!/^[0-9]+$/.test(suffix)) return null;
  const candidate = tokens.slice(0, -1).join("");
  return BINANCE_SYMBOL_PATTERN.test(candidate) ? candidate : null;
}

function sanitizeBinanceLikeSymbol(raw: string): string | null {
  const value = normalizeSymbolText(raw);
  if (!value) return null;
  if (BINANCE_SYMBOL_PATTERN.test(value)) return value;

  const trimmed = trimBinanceComboSuffix(value);
  if (trimmed) return trimmed;

  const tokens = splitAlphaNumTokens(value);
  const tokenTrimmed = trimTokenizedBinanceComboSuffix(tokens);
  if (tokenTrimmed) return tokenTrimmed;
  if (tokens.length >= 2) {
    const joined = `${tokens[0]}${tokens[1]}`;
    if (tokens.length === 2 && BINANCE_SYMBOL_PATTERN.test(joined)) return joined;
    if (tokens.length >= 3 && /^[0-9]+[A-Z]$/.test(tokens[2] ?? "") && BINANCE_SYMBOL_PATTERN.test(joined)) return joined;
  }
  if (tokens.length >= 1 && BINANCE_SYMBOL_PATTERN.test(tokens[0] ?? "")) return tokens[0] ?? null;
  return null;
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
      return BINANCE_SYMBOL_PATTERN;
  }
}

export function sanitizeSymbolForPlatform(platform: Platform, raw: string): string | null {
  switch (platform) {
    case "coinbase":
      return sanitizeDelimitedSymbol(raw, "-");
    case "poloniex":
      return sanitizeDelimitedSymbol(raw, "_");
    case "binance":
    case "kraken":
    default:
      return sanitizeBinanceLikeSymbol(raw);
  }
}

export function normalizeComboSymbol(raw: string, platform: Platform | null): string {
  const value = normalizeSymbolText(raw);
  if (!value) return value;
  const resolvedPlatform: Platform = platform ?? "binance";
  return sanitizeSymbolForPlatform(resolvedPlatform, value) ?? value;
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
  return symbols.filter((sym) => sanitizeSymbolForPlatform(platform, sym) == null);
}