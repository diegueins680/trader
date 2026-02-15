import type { ConfigPageId, ConfigPanelId } from "./configTypes";

export const CONFIG_PANEL_IDS: ConfigPanelId[] = [
  "config-access",
  "config-market",
  "config-strategy",
  "config-optimization",
  "config-execution",
];

export const CONFIG_PAGE_IDS: ConfigPageId[] = [
  "section-api",
  "section-market",
  "section-lookback",
  "section-thresholds",
  "section-risk",
  "section-optimizer-run",
  "section-optimization",
  "section-livebot",
  "section-trade",
];

export const CONFIG_PAGE_LABELS: Record<ConfigPageId, string> = {
  "section-api": "API",
  "section-market": "Market",
  "section-lookback": "Lookback",
  "section-thresholds": "Thresholds",
  "section-risk": "Risk",
  "section-optimizer-run": "Optimizer run",
  "section-optimization": "Optimization",
  "section-livebot": "Live bot",
  "section-trade": "Trade",
};

export const CONFIG_PAGE_PANEL_MAP: Record<ConfigPageId, ConfigPanelId> = {
  "section-api": "config-access",
  "section-market": "config-market",
  "section-lookback": "config-market",
  "section-thresholds": "config-strategy",
  "section-risk": "config-strategy",
  "section-optimizer-run": "config-optimization",
  "section-optimization": "config-optimization",
  "section-livebot": "config-execution",
  "section-trade": "config-execution",
};

export const CONFIG_PANEL_DEFAULT_PAGE: Record<ConfigPanelId, ConfigPageId> = {
  "config-access": "section-api",
  "config-market": "section-market",
  "config-strategy": "section-thresholds",
  "config-optimization": "section-optimizer-run",
  "config-execution": "section-livebot",
};

export const CONFIG_TARGET_PAGE_MAP: Record<string, ConfigPageId> = {
  platformKeys: "section-api",
  symbol: "section-market",
  platform: "section-market",
  market: "section-market",
  interval: "section-market",
  bars: "section-market",
  lookbackWindow: "section-lookback",
  lookbackBars: "section-lookback",
  positioning: "section-thresholds",
  backtestRatio: "section-thresholds",
  tuneRatio: "section-thresholds",
  epochs: "section-risk",
  hiddenSize: "section-risk",
  botSymbols: "section-livebot",
  orderQuoteFraction: "section-trade",
};

export const CONFIG_PANEL_HEIGHTS: Record<ConfigPanelId, string> = {
  "config-access": "clamp(260px, 32vh, 360px)",
  "config-market": "clamp(280px, 38vh, 420px)",
  "config-strategy": "clamp(320px, 50vh, 600px)",
  "config-optimization": "clamp(320px, 50vh, 600px)",
  "config-execution": "clamp(320px, 50vh, 600px)",
};

export const normalizeConfigPage = (page: unknown): ConfigPageId => {
  if (CONFIG_PAGE_IDS.includes(page as ConfigPageId)) {
    return page as ConfigPageId;
  }
  if (CONFIG_PANEL_IDS.includes(page as ConfigPanelId)) {
    return CONFIG_PANEL_DEFAULT_PAGE[page as ConfigPanelId];
  }
  return "section-api";
};

export const normalizeConfigPanelOrder = (order: unknown): ConfigPanelId[] => {
  const seen = new Set<ConfigPanelId>();
  const out: ConfigPanelId[] = [];
  if (Array.isArray(order)) {
    for (const value of order) {
      if (CONFIG_PANEL_IDS.includes(value as ConfigPanelId)) {
        const id = value as ConfigPanelId;
        if (!seen.has(id)) {
          seen.add(id);
          out.push(id);
        }
      }
    }
  }
  for (const id of CONFIG_PANEL_IDS) {
    if (!seen.has(id)) out.push(id);
  }
  return out;
};

export const resolveConfigPageForTarget = (targetId: string): ConfigPageId | null => {
  if (CONFIG_PAGE_IDS.includes(targetId as ConfigPageId)) {
    return targetId as ConfigPageId;
  }
  return CONFIG_TARGET_PAGE_MAP[targetId] ?? null;
};
