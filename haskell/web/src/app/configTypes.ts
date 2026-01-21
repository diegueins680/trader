export type ConfigPanelId =
  | "config-access"
  | "config-market"
  | "config-strategy"
  | "config-optimization"
  | "config-execution";

export type ConfigPageId =
  | "section-api"
  | "section-market"
  | "section-lookback"
  | "section-thresholds"
  | "section-risk"
  | "section-optimizer-run"
  | "section-optimization"
  | "section-livebot"
  | "section-trade";

export type ConfigPanelDragState = {
  draggingId: ConfigPanelId | null;
  overId: ConfigPanelId | null;
};
