import React from "react";
import type { ApiParams, IntrabarFill, Market, Method, Normalization, OptimizerSource, Platform, Positioning } from "../lib/types";
import type { ConfigPanelDragState, ConfigPanelId, ConfigPageId } from "../app/configTypes";
import type {
  ActiveAsyncJob,
  BotUiState,
  CacheUiState,
  ComputeLimits,
  ManualOverrideKey,
  OptimizerRunForm,
  OptimizerRunUiState,
  PendingProfileLoad,
  RequestKind,
  RunOptions,
  UiState,
} from "../app/appHelpers";
import type { RequestIssueDetail } from "../app/utils";
import type { health } from "../lib/api";
import { copyText } from "../lib/clipboard";
import { fmtPct } from "../lib/format";
import { API_PORT } from "../app/apiTarget";
import { defaultForm, type FormState } from "../app/formState";
import { firstReason, fmtTimeMs, generateIdempotencyKey, numFromInput } from "../app/utils";
import { COMPLEX_TIPS, CUSTOM_SYMBOL_VALUE, EQUITY_TIPS } from "../app/appHelpers";
import { PLATFORM_DEFAULT_SYMBOL, PLATFORM_LABELS, PLATFORM_SYMBOL_SET, PLATFORMS, TUNE_OBJECTIVES } from "../app/constants";
import { CollapsibleCard } from "./CollapsibleCard";
import { InfoList, InfoPopover } from "./InfoPopover";

type PanelToggleHandler = (event: React.SyntheticEvent<HTMLDetailsElement>) => void;

type ConfigPanelHandlers = {
  dragState: ConfigPanelDragState;
  onDragStart: (panelId: ConfigPanelId) => (event: React.DragEvent<HTMLButtonElement>) => void;
  onDragOver: (panelId: ConfigPanelId) => (event: React.DragEvent<HTMLElement>) => void;
  onDrop: (panelId: ConfigPanelId) => (event: React.DragEvent<HTMLElement>) => void;
  onDragEnd: () => void;
};

type CollapsibleSectionProps = {
  panelId: string;
  title: string;
  meta?: string;
  children: React.ReactNode;
  open: boolean;
  onToggle: PanelToggleHandler;
};

type ConfigPanelProps = {
  panelId: ConfigPanelId;
  title: string;
  subtitle?: string;
  draggable?: boolean;
  order: number;
  open: boolean;
  onToggle: PanelToggleHandler;
  maximized: boolean;
  onToggleMaximize: () => void;
  dragState: ConfigPanelDragState;
  onDragStart: (panelId: ConfigPanelId) => (event: React.DragEvent<HTMLButtonElement>) => void;
  onDragOver: (panelId: ConfigPanelId) => (event: React.DragEvent<HTMLElement>) => void;
  onDrop: (panelId: ConfigPanelId) => (event: React.DragEvent<HTMLElement>) => void;
  onDragEnd: () => void;
  style?: React.CSSProperties;
  children: React.ReactNode;
};

type LookbackState = {
  bars: number;
  overrideOn: boolean;
  error: string | null;
  summary: string;
};

type SplitPreview = {
  summary: string;
  warning: boolean;
};

type EstimatedCosts = {
  breakEven: number;
  roundTrip: number;
};

type OrderSizingSummary = {
  conflicts: boolean;
  active: string[];
  effective: string;
  hint: string;
};

type OptimizerRunExtras = {
  error: string | null;
};

type HealthInfo = Awaited<ReturnType<typeof health>>;

type StartBotOptions = { auto?: boolean; forceAdopt?: boolean; silent?: boolean; symbolsOverride?: string[] };

type ApiOkStatus = "unknown" | "ok" | "down" | "auth";

export type ConfigDockProps = {
  configOpen: boolean;
  handlePanelToggle: (panelId: string) => PanelToggleHandler;
  isPanelOpen: (panelId: string, defaultOpen: boolean) => boolean;
  isPanelMaximized: (panelId: string) => boolean;
  togglePanelMaximize: (panelId: string) => void;
  requestIssues: string[];
  primaryIssue: RequestIssueDetail | null;
  handlePrimaryIssueFix: () => void;
  extraIssueCount: number;
  requestDisabled: boolean;
  requestDisabledReason: string | null;
  run: (kind: RequestKind, overrideParams?: ApiParams, opts?: RunOptions) => Promise<void>;
  state: UiState;
  commonParams: ApiParams;
  setForm: React.Dispatch<React.SetStateAction<FormState>>;
  cancelActiveRequest: () => void;
  requestIssueDetails: RequestIssueDetail[];
  scrollToSection: (id?: string) => void;
  rateLimitReason: string | null;
  activeAsyncJob: ActiveAsyncJob | null;
  activeAsyncTickMs: number;
  activeConfigPanel: ConfigPanelId;
  configPanelOrderIndex: Record<ConfigPanelId, number>;
  configPanelStyle: (panelId: ConfigPanelId) => React.CSSProperties;
  configPanelHandlers: ConfigPanelHandlers;
  configPage: ConfigPageId;
  apiBase: string;
  apiToken: string;
  apiBaseAbsolute: string;
  apiBaseError: string | null;
  apiHealthUrl: string | null;
  apiBaseCorsHint: string | null;
  healthInfo: HealthInfo | null;
  cacheUi: CacheUiState;
  apiOk: ApiOkStatus;
  showLocalStartHelp: boolean;
  refreshCacheStats: () => Promise<void>;
  clearCacheUi: () => Promise<void>;
  recheckHealth: () => void;
  showToast: (message: string) => void;
  platformKeyLabel: string;
  platformKeyMode: "coinbase" | "binance" | null;
  revealSecrets: boolean;
  setRevealSecrets: React.Dispatch<React.SetStateAction<boolean>>;
  coinbaseApiKey: string;
  coinbaseApiSecret: string;
  coinbaseApiPassphrase: string;
  setCoinbaseApiKey: (value: string) => void;
  setCoinbaseApiSecret: (value: string) => void;
  setCoinbaseApiPassphrase: (value: string) => void;
  binanceApiKey: string;
  binanceApiSecret: string;
  setBinanceApiKey: (value: string) => void;
  setBinanceApiSecret: (value: string) => void;
  platformKeyHasValues: boolean;
  platformKeyHint: string;
  keysLoading: boolean;
  keysCheckedAtMs: number | null;
  refreshKeys: () => void;
  persistSecrets: boolean;
  setPersistSecrets: React.Dispatch<React.SetStateAction<boolean>>;
  profileSelected: string;
  setProfileSelected: (value: string) => void;
  profileNames: string[];
  profileName: string;
  setProfileName: (value: string) => void;
  saveProfile: () => void;
  requestLoadProfile: () => void;
  deleteProfile: () => void;
  pendingProfileLoad: PendingProfileLoad | null;
  clearManualOverrides: (keys?: ManualOverrideKey[]) => void;
  setPendingProfileLoad: React.Dispatch<React.SetStateAction<PendingProfileLoad | null>>;
  form: FormState;
  setPendingMarket: React.Dispatch<React.SetStateAction<Market | null>>;
  customSymbolByPlatform: Record<Platform, string>;
  setCustomSymbolByPlatform: React.Dispatch<React.SetStateAction<Record<Platform, string>>>;
  missingSymbol: boolean;
  symbolFormatError: string | null;
  symbolSelectValue: string;
  platformSymbols: readonly string[];
  symbolIsCustom: boolean;
  platformLabel: string;
  platform: Platform;
  isCoinbasePlatform: boolean;
  isBinancePlatform: boolean;
  pendingMarket: Market | null;
  setConfirmLive: React.Dispatch<React.SetStateAction<boolean>>;
  missingInterval: boolean;
  platformIntervals: readonly string[];
  barsRangeLabel: string;
  barsExceedsApi: boolean;
  barsMaxLabel: number | null;
  apiComputeLimits: ComputeLimits | null;
  barsAutoHint: string;
  barsRangeHint: string;
  lookbackState: LookbackState;
  splitPreview: SplitPreview;
  markManualOverrides: (keys: ManualOverrideKey[]) => void;
  methodOverride: boolean;
  thresholdOverrideKeys: ManualOverrideKey[];
  estimatedCosts: EstimatedCosts;
  minEdgeEffective: number;
  longShortBotDisabled: boolean;
  apiLstmEnabled: boolean;
  epochsExceedsApi: boolean;
  hiddenSizeExceedsApi: boolean;
  optimizerRunForm: OptimizerRunForm;
  updateOptimizerRunForm: (next: Partial<OptimizerRunForm>) => void;
  optimizerRunValidationError: string | null;
  optimizerRunExtras: OptimizerRunExtras;
  optimizerRunUi: OptimizerRunUiState;
  runOptimizer: () => Promise<void>;
  cancelOptimizerRun: () => void;
  syncOptimizerRunForm: () => void;
  applyEquityPreset: () => void;
  resetOptimizerRunForm: () => void;
  optimizerRunRecordJson: string | null;
  botStartBlocked: boolean;
  botStartBlockedReason: string | null;
  bot: BotUiState;
  botStarting: boolean;
  botAnyRunning: boolean;
  startLiveBot: (opts?: StartBotOptions) => Promise<void>;
  stopLiveBot: (symbol?: string) => Promise<void>;
  botSymbolsActive: string[];
  botSelectedSymbol: string | null;
  apiBlockedReason: string | null;
  refreshBot: (opts?: RunOptions) => Promise<void>;
  botSymbolsFormatError: string | null;
  confirmLive: boolean;
  confirmArm: boolean;
  setConfirmArm: React.Dispatch<React.SetStateAction<boolean>>;
  orderSizing: OrderSizingSummary;
  orderQuoteFractionError: string | null;
  idempotencyKeyError: string | null;
  tradeDisabledReason: string | null;
  tradeDisabledDetail: RequestIssueDetail | null;
};

const CollapsibleSection = ({ panelId, title, meta, children, open, onToggle }: CollapsibleSectionProps) => (
  <details className="sectionPanel" open={open} onToggle={onToggle} data-panel={panelId}>
    <summary className="sectionHeading" id={panelId}>
      <span className="sectionTitle">{title}</span>
      {meta ? <span className="sectionMeta">{meta}</span> : null}
      <span className="sectionToggle" aria-hidden="true">
        <span className="sectionToggleLabel" data-open="Hide" data-closed="Show" />
        <span className="sectionToggleIcon" />
      </span>
    </summary>
    <div className="sectionBody">{children}</div>
  </details>
);

const ConfigPanel = ({
  panelId,
  title,
  subtitle,
  draggable = true,
  order,
  open,
  onToggle,
  maximized,
  onToggleMaximize,
  dragState,
  onDragStart,
  onDragOver,
  onDrop,
  onDragEnd,
  style,
  children,
}: ConfigPanelProps) => {
  const isDragOver = dragState.overId === panelId && dragState.draggingId !== panelId;
  const isDragging = dragState.draggingId === panelId;
  return (
    <details
      id={`config-panel-${panelId}`}
      className={`configPanel${maximized ? " configPanelMaximized" : ""}${isDragOver ? " configPanelDrop" : ""}${
        isDragging ? " configPanelDragging" : ""
      }`}
      open={open}
      onToggle={onToggle}
      style={{ order, ...style }}
      onDragOver={onDragOver(panelId)}
      onDrop={onDrop(panelId)}
      data-panel={panelId}
      aria-labelledby={`config-tab-${panelId}`}
    >
      <summary className="configPanelHeader configPanelSummary">
        <div className="configPanelHeaderText">
          <span className="configPanelTitle">{title}</span>
          {subtitle ? <span className="configPanelSubtitle">{subtitle}</span> : null}
        </div>
        <div className="configPanelControls">
          {draggable ? (
            <button
              className="configPanelHandle"
              type="button"
              draggable={!maximized}
              onDragStart={onDragStart(panelId)}
              onDragEnd={onDragEnd}
              onClick={(event) => {
                event.preventDefault();
                event.stopPropagation();
              }}
              aria-label={`Drag ${title} panel`}
              title="Drag to reorder"
            >
              Drag
            </button>
          ) : null}
          <button
            className="configPanelControl"
            type="button"
            aria-pressed={maximized}
            aria-label={maximized ? "Restore panel size" : "Maximize panel"}
            onClick={(event) => {
              event.preventDefault();
              event.stopPropagation();
              onToggleMaximize();
            }}
          >
            {maximized ? "Restore" : "Maximize"}
          </button>
          <span className="configPanelToggle" aria-hidden="true">
            <span className="configPanelToggleLabel" data-open="Minimize" data-closed="Expand" />
            <span className="configPanelToggleIcon" />
          </span>
        </div>
      </summary>
      <div className="configPanelBody">{children}</div>
    </details>
  );
};

export const ConfigDock = (props: ConfigDockProps) => {
  const {
    activeAsyncJob,
    activeAsyncTickMs,
    activeConfigPanel,
    apiBase,
    apiBaseAbsolute,
    apiBaseCorsHint,
    apiBaseError,
    apiBlockedReason,
    apiComputeLimits,
    apiHealthUrl,
    apiLstmEnabled,
    apiOk,
    apiToken,
    applyEquityPreset,
    barsAutoHint,
    barsExceedsApi,
    barsMaxLabel,
    barsRangeHint,
    barsRangeLabel,
    bot,
    botAnyRunning,
    botSelectedSymbol,
    botStartBlocked,
    botStartBlockedReason,
    botStarting,
    botSymbolsActive,
    botSymbolsFormatError,
    cancelActiveRequest,
    cancelOptimizerRun,
    cacheUi,
    clearCacheUi,
    clearManualOverrides,
    commonParams,
    configOpen,
    configPage,
    configPanelHandlers,
    configPanelOrderIndex,
    configPanelStyle,
    confirmArm,
    confirmLive,
    customSymbolByPlatform,
    deleteProfile,
    epochsExceedsApi,
    estimatedCosts,
    extraIssueCount,
    form,
    handlePanelToggle,
    handlePrimaryIssueFix,
    healthInfo,
    hiddenSizeExceedsApi,
    idempotencyKeyError,
    isBinancePlatform,
    isCoinbasePlatform,
    isPanelMaximized,
    isPanelOpen,
    keysCheckedAtMs,
    keysLoading,
    longShortBotDisabled,
    lookbackState,
    markManualOverrides,
    methodOverride,
    minEdgeEffective,
    missingInterval,
    missingSymbol,
    optimizerRunExtras,
    optimizerRunForm,
    optimizerRunRecordJson,
    optimizerRunUi,
    optimizerRunValidationError,
    orderQuoteFractionError,
    orderSizing,
    pendingMarket,
    pendingProfileLoad,
    persistSecrets,
    platform,
    platformIntervals,
    platformKeyHasValues,
    platformKeyHint,
    platformKeyLabel,
    platformKeyMode,
    platformLabel,
    platformSymbols,
    primaryIssue,
    profileName,
    profileNames,
    profileSelected,
    rateLimitReason,
    refreshBot,
    refreshCacheStats,
    refreshKeys,
    requestDisabled,
    requestDisabledReason,
    requestIssueDetails,
    requestIssues,
    requestLoadProfile,
    resetOptimizerRunForm,
    revealSecrets,
    recheckHealth,
    run,
    runOptimizer,
    saveProfile,
    scrollToSection,
    binanceApiKey,
    binanceApiSecret,
    coinbaseApiKey,
    coinbaseApiPassphrase,
    coinbaseApiSecret,
    setBinanceApiKey,
    setBinanceApiSecret,
    setCoinbaseApiKey,
    setCoinbaseApiPassphrase,
    setCoinbaseApiSecret,
    setConfirmArm,
    setConfirmLive,
    setCustomSymbolByPlatform,
    setForm,
    setPendingMarket,
    setPendingProfileLoad,
    setPersistSecrets,
    setProfileName,
    setProfileSelected,
    setRevealSecrets,
    showLocalStartHelp,
    showToast,
    splitPreview,
    startLiveBot,
    state,
    stopLiveBot,
    symbolFormatError,
    symbolIsCustom,
    symbolSelectValue,
    syncOptimizerRunForm,
    thresholdOverrideKeys,
    togglePanelMaximize,
    tradeDisabledDetail,
    tradeDisabledReason,
    updateOptimizerRunForm,
  } = props;
  const botProtectionNeedsStops =
    form.stopLoss <= 0 && form.takeProfit <= 0 && form.stopLossVolMult <= 0 && form.takeProfitVolMult <= 0;
  const keysSupported = isBinancePlatform || isCoinbasePlatform;
  const keysHint =
    !keysSupported
      ? "Switch Platform to Binance or Coinbase to check keys."
      : keysCheckedAtMs
        ? `Last checked: ${fmtTimeMs(keysCheckedAtMs)}`
        : isBinancePlatform
          ? "Uses Binance signed endpoints + /order/test (no real order)."
          : "Uses Coinbase signed /accounts.";

  return (
    <CollapsibleCard
        panelId="panel-config"
        open={configOpen}
        onToggle={handlePanelToggle("panel-config")}
        maximized={isPanelMaximized("panel-config")}
        onToggleMaximize={() => togglePanelMaximize("panel-config")}
        title="Configuration"
        subtitle="Safe defaults, minimal knobs, and clear outputs."
        className="configCard"
      >
        <div className="stickyActions">
          <div className="pillRow">
            <span className={`pill ${requestIssues.length ? "pillWarn" : "pillOk"}`}>
              {requestIssues.length
                ? `${requestIssues.length} issue${requestIssues.length === 1 ? "" : "s"} to fix`
                : "Ready to run"}
            </span>
            {primaryIssue ? (
              <>
                <span className="pill pillWarn">{primaryIssue.message}</span>
                {primaryIssue.targetId ? (
                  <button className="btnSmall" type="button" onClick={handlePrimaryIssueFix}>
                    Fix
                  </button>
                ) : null}
              </>
            ) : (
              <span className="pill">All required inputs look good.</span>
            )}
            {extraIssueCount > 0 ? <span className="pill">+{extraIssueCount} more</span> : null}
          </div>
          <div className="actions">
            <button
              className="btn btnPrimary"
              disabled={requestDisabled}
              onClick={() => run("signal")}
              title={requestDisabledReason ?? undefined}
            >
              {state.loading && state.lastKind === "signal" ? "Getting signal…" : "Get signal"}
            </button>
            <button className="btn" disabled={requestDisabled} onClick={() => run("backtest")} title={requestDisabledReason ?? undefined}>
              {state.loading && state.lastKind === "backtest" ? "Running backtest…" : "Run backtest"}
            </button>
            <button
              className="btn"
              disabled={requestDisabled}
              title={requestDisabledReason ?? undefined}
              onClick={() => {
                const p = { ...commonParams, sweepThreshold: true, optimizeOperations: false };
                setForm((f) => ({ ...f, sweepThreshold: true, optimizeOperations: false }));
                void run("backtest", p);
              }}
            >
              {state.loading && state.lastKind === "backtest" ? "Optimizing…" : "Optimize thresholds"}
            </button>
            <button
              className="btn"
              disabled={requestDisabled}
              title={requestDisabledReason ?? undefined}
              onClick={() => {
                const p = { ...commonParams, optimizeOperations: true, sweepThreshold: false };
                setForm((f) => ({ ...f, optimizeOperations: true, sweepThreshold: false }));
                void run("backtest", p);
              }}
            >
              {state.loading && state.lastKind === "backtest" ? "Optimizing…" : "Optimize operations"}
            </button>
            <button className="btn" disabled={!state.loading} onClick={cancelActiveRequest}>
              Cancel
            </button>
          </div>
          {requestIssueDetails.length > 1 ? (
            <details className="details">
              <summary>Show all issues</summary>
              <div className="issueList" style={{ marginTop: 10 }}>
                {requestIssueDetails.map((issue, idx) => (
                  <div key={`${issue.message}-${idx}`} className="issueItem">
                    <span>{issue.message}</span>
                    {issue.targetId ? (
                      <button className="btnSmall" type="button" onClick={() => scrollToSection(issue.targetId)}>
                        Jump
                      </button>
                    ) : null}
                  </div>
                ))}
              </div>
            </details>
          ) : null}

          {rateLimitReason ? (
            <div className="hint" style={{ marginTop: 6, color: "var(--warn)" }}>
              {rateLimitReason}
            </div>
          ) : null}

          {state.loading ? (
            <div className="hint" style={{ marginTop: 6 }}>
              {activeAsyncJob?.jobId
                ? `Async job: ${activeAsyncJob.jobId} • ${activeAsyncJob.kind} • ${Math.max(
                    0,
                    Math.floor((activeAsyncTickMs - activeAsyncJob.startedAtMs) / 1000),
                  )}s`
                : "Starting async job…"}
            </div>
          ) : null}
        </div>
        <div className="configPanels">
          {activeConfigPanel === "config-access" ? (
            <ConfigPanel
              panelId="config-access"
              title="Access & Profiles"
              subtitle="API health, keys, and saved setups."
              order={configPanelOrderIndex["config-access"]}
              open={isPanelOpen("config-access", true)}
              onToggle={handlePanelToggle("config-access")}
              maximized={isPanelMaximized("config-access")}
              onToggleMaximize={() => togglePanelMaximize("config-access")}
              style={configPanelStyle("config-access")}
              draggable={false}
              {...configPanelHandlers}
            >
        <div className="row" style={{ gridTemplateColumns: "1fr" }} id="section-api">
          <div className="field" id="platformKeys">
            <div className="label">API</div>
            <div className="kv">
              <div className="k">Base URL</div>
              <div className="v">
                <span className="tdMono">{apiBase}</span>
              </div>
            </div>
            <div className="kv">
              <div className="k">Token</div>
              <div className="v">{apiToken.trim() ? "configured" : "not set"}</div>
            </div>
            <div className="actions" style={{ marginTop: 8 }}>
              <button className="btnSmall" type="button" onClick={() => void copyText(apiBaseAbsolute)} disabled={Boolean(apiBaseError)}>
                Copy base URL
              </button>
              <button
                className="btnSmall"
                type="button"
                onClick={() => {
                  if (!apiHealthUrl) return;
                  if (typeof window !== "undefined") window.open(apiHealthUrl, "_blank", "noopener,noreferrer");
                }}
                disabled={Boolean(apiBaseError) || !apiHealthUrl}
              >
                Open /health
              </button>
            </div>
            <div className="hint" style={{ marginTop: 6 }}>
              Configured at deploy time via <span style={{ fontFamily: "var(--mono)" }}>trader-config.js</span> (apiBaseUrl, apiToken).
            </div>
            {apiBaseError ? (
              <div className="hint" style={{ color: "rgba(239, 68, 68, 0.85)", marginTop: 6 }}>
                {apiBaseError}
              </div>
            ) : null}
            {apiBaseCorsHint ? (
              <div className="hint" style={{ marginTop: 6 }}>
                {apiBaseCorsHint}
              </div>
            ) : null}
            {healthInfo?.computeLimits ? (
              <div className="hint" style={{ marginTop: 6 }}>
                {healthInfo.version ? (
                  <>
                    API build:{" "}
                    <span className="tdMono">
                      {healthInfo.version}
                      {healthInfo.commit ? ` (${healthInfo.commit.slice(0, 12)})` : ""}
                    </span>
                    .{" "}
                  </>
                ) : null}
                {typeof healthInfo.authRequired === "boolean" ? (
                  <>
                    Auth:{" "}
                    {healthInfo.authRequired ? (healthInfo.authOk ? "required (ok)" : "required (failed)") : "not required"}.
                  </>
                ) : null}{" "}
                API limits: max LSTM bars {healthInfo.computeLimits.maxBarsLstm}, epochs {healthInfo.computeLimits.maxEpochs}, hidden{" "}
                {healthInfo.computeLimits.maxHiddenSize}.
                {healthInfo.asyncJobs
                  ? ` Async: max running ${healthInfo.asyncJobs.maxRunning}, TTL ${Math.round(
                      healthInfo.asyncJobs.ttlMs / 60000,
                    )}m, persistence ${healthInfo.asyncJobs.persistence ? "on" : "off"}.`
                  : ""}
                {healthInfo.cache
                  ? ` Cache: ${healthInfo.cache.enabled ? "on" : "off"} (TTL ${Math.round(healthInfo.cache.ttlMs / 1000)}s, max ${
                      healthInfo.cache.maxEntries
                    }).`
                  : ""}
              </div>
            ) : null}
            {healthInfo?.cache ? (
              <div style={{ marginTop: 10 }}>
                <div className="actions" style={{ marginTop: 0 }}>
                  <button className="btn" type="button" onClick={() => void refreshCacheStats()} disabled={cacheUi.loading || apiOk !== "ok"}>
                    {cacheUi.loading ? "Loading…" : "Refresh cache stats"}
                  </button>
                  <button
                    className="btn"
                    type="button"
                    onClick={() => void clearCacheUi()}
                    disabled={cacheUi.loading || apiOk !== "ok" || healthInfo.cache.enabled === false}
                  >
                    Clear cache
                  </button>
                  <span className="hint">Disable via `TRADER_API_CACHE_TTL_MS=0` if you never want cached results.</span>
                </div>
                {cacheUi.error ? (
                  <pre className="code" style={{ borderColor: "rgba(239, 68, 68, 0.35)", marginTop: 8 }}>
                    {cacheUi.error}
                  </pre>
                ) : null}
                {cacheUi.stats ? (
                  <div className="hint" style={{ marginTop: 8 }}>
                    Signals: {cacheUi.stats.signals.entries} entries ({cacheUi.stats.signals.hits} hit / {cacheUi.stats.signals.misses} miss) • Backtests:{" "}
                    {cacheUi.stats.backtests.entries} entries ({cacheUi.stats.backtests.hits} hit / {cacheUi.stats.backtests.misses} miss) • Updated{" "}
                    {fmtTimeMs(cacheUi.stats.atMs)}
                  </div>
                ) : null}
              </div>
            ) : null}
          </div>
        </div>

        {apiOk === "down" || apiOk === "auth" ? (
          <div className="row" style={{ gridTemplateColumns: "1fr" }}>
            <div className="field">
              <div className="label">Connection</div>
              <pre className="code" style={{ borderColor: "rgba(239, 68, 68, 0.35)" }}>
                {apiOk === "down"
                  ? showLocalStartHelp
                    ? `Backend unreachable.\n\nStart it with:\ncd haskell && cabal run -v0 trader-hs -- --serve --port ${API_PORT}`
                    : "Backend unreachable.\n\nConfigure apiBaseUrl in trader-config.js (CORS required for cross-origin) or configure CloudFront to forward `/api/*` to your API origin."
                  : apiToken.trim()
                    ? "API auth failed.\n\nUpdate apiToken in trader-config.js (it must match the backend’s TRADER_API_TOKEN)."
                    : "API auth required.\n\nSet apiToken in trader-config.js (it must match the backend’s TRADER_API_TOKEN)."}
              </pre>
              <div className="actions" style={{ marginTop: 0 }}>
                {apiOk === "down" && showLocalStartHelp ? (
                  <button
                    className="btn"
                    type="button"
                    onClick={() => {
                      void copyText(`cd haskell && cabal run -v0 trader-hs -- --serve --port ${API_PORT}`);
                      showToast("Copied start command");
                    }}
                  >
                    Copy start command
                  </button>
                ) : null}
                <button className="btn" type="button" onClick={() => void recheckHealth()}>
                  Re-check
                </button>
              </div>
            </div>
          </div>
        ) : null}

        <div className="row" style={{ gridTemplateColumns: "1fr" }}>
          <div className="field">
            <div className="label">{platformKeyLabel}</div>
            <div
              className="row"
              style={{
                gridTemplateColumns: platformKeyMode === "coinbase" ? "1fr 1fr 1fr auto auto" : "1fr 1fr auto auto",
                alignItems: "center",
              }}
            >
              <input
                className="input"
                type={revealSecrets ? "text" : "password"}
                value={platformKeyMode === "coinbase" ? coinbaseApiKey : platformKeyMode === "binance" ? binanceApiKey : ""}
                onChange={(e) => {
                  const next = e.target.value;
                  if (platformKeyMode === "coinbase") setCoinbaseApiKey(next);
                  else if (platformKeyMode === "binance") setBinanceApiKey(next);
                }}
                aria-label={platformKeyMode === "coinbase" ? "Coinbase API key" : platformKeyMode === "binance" ? "Binance API key" : "API key"}
                placeholder={
                  platformKeyMode === "coinbase" ? "COINBASE_API_KEY" : platformKeyMode === "binance" ? "BINANCE_API_KEY" : "Select Binance/Coinbase"
                }
                spellCheck={false}
                autoCapitalize="none"
                autoCorrect="off"
                inputMode="text"
                disabled={!platformKeyMode}
              />
              <input
                className="input"
                type={revealSecrets ? "text" : "password"}
                value={platformKeyMode === "coinbase" ? coinbaseApiSecret : platformKeyMode === "binance" ? binanceApiSecret : ""}
                onChange={(e) => {
                  const next = e.target.value;
                  if (platformKeyMode === "coinbase") setCoinbaseApiSecret(next);
                  else if (platformKeyMode === "binance") setBinanceApiSecret(next);
                }}
                aria-label={
                  platformKeyMode === "coinbase" ? "Coinbase API secret" : platformKeyMode === "binance" ? "Binance API secret" : "API secret"
                }
                placeholder={
                  platformKeyMode === "coinbase" ? "COINBASE_API_SECRET" : platformKeyMode === "binance" ? "BINANCE_API_SECRET" : "Select Binance/Coinbase"
                }
                spellCheck={false}
                autoCapitalize="none"
                autoCorrect="off"
                inputMode="text"
                disabled={!platformKeyMode}
              />
              {platformKeyMode === "coinbase" ? (
                <input
                  className="input"
                  type={revealSecrets ? "text" : "password"}
                  value={coinbaseApiPassphrase}
                  onChange={(e) => setCoinbaseApiPassphrase(e.target.value)}
                  aria-label="Coinbase API passphrase"
                  placeholder="COINBASE_API_PASSPHRASE"
                  spellCheck={false}
                  autoCapitalize="none"
                  autoCorrect="off"
                  inputMode="text"
                />
              ) : null}
              <button className="btn" type="button" onClick={() => setRevealSecrets((v) => !v)}>
                {revealSecrets ? "Hide" : "Show"}
              </button>
              <button
                className="btn"
                type="button"
                onClick={() => {
                  if (platformKeyMode === "coinbase") {
                    setCoinbaseApiKey("");
                    setCoinbaseApiSecret("");
                    setCoinbaseApiPassphrase("");
                  } else if (platformKeyMode === "binance") {
                    setBinanceApiKey("");
                    setBinanceApiSecret("");
                  }
                }}
                disabled={!platformKeyHasValues}
              >
                Clear
              </button>
            </div>
            <div className="hint">
              {platformKeyHint} Stored in {persistSecrets ? "local storage" : "session storage"}. The request preview/curl omits it.
            </div>
            {platformKeyMode ? (
              <div className="actions" style={{ marginTop: 8 }}>
                <button
                  className="btn"
                  type="button"
                  onClick={() => refreshKeys()}
                  disabled={!keysSupported || keysLoading || apiOk === "down" || apiOk === "auth"}
                >
                  {keysLoading ? "Checking…" : "Check keys"}
                </button>
                <span className="hint">{keysHint}</span>
              </div>
            ) : null}
            {!platformKeyMode ? (
              <div className="hint" style={{ color: "rgba(245, 158, 11, 0.9)" }}>
                Keys are only editable when Platform is set to Binance or Coinbase.
              </div>
            ) : null}
            <div className="pillRow" style={{ marginTop: 10 }}>
              <label className="pill">
                <input type="checkbox" checked={persistSecrets} onChange={(e) => setPersistSecrets(e.target.checked)} />
                Remember platform keys
              </label>
            </div>
            <div className="hint">
              When enabled, the platform keys are stored in local storage so you can reopen the app later without re-entering them (not recommended on shared
              machines).
            </div>
          </div>
        </div>

        <div className="row" style={{ gridTemplateColumns: "1fr" }}>
          <div className="field">
            <div className="label">Profiles</div>
            <div className="row" style={{ gridTemplateColumns: "1fr 1fr", alignItems: "center" }}>
              <select className="select" value={profileSelected} onChange={(e) => setProfileSelected(e.target.value)} aria-label="Saved profiles">
                <option value="">{profileNames.length ? "Select saved profile…" : "No profiles yet"}</option>
                {profileNames.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
              <input
                className="input"
                value={profileName}
                onChange={(e) => setProfileName(e.target.value)}
                placeholder="New profile name"
                spellCheck={false}
                aria-label="New profile name"
              />
            </div>
            <div className="actions" style={{ marginTop: 10 }}>
              <button className="btn" type="button" onClick={saveProfile} disabled={!profileName.trim()}>
                Save
              </button>
              <button className="btn" type="button" onClick={requestLoadProfile} disabled={!profileSelected.trim()}>
                Load
              </button>
              <button className="btn btnDanger" type="button" onClick={deleteProfile} disabled={!profileSelected.trim()}>
                Delete
              </button>
            </div>
            <div className="hint">Save/load named config presets. Does not include API keys.</div>

            {pendingProfileLoad ? (
              <>
                <pre className="code" style={{ borderColor: "rgba(245, 158, 11, 0.35)", marginTop: 10 }}>
                  Loading “{pendingProfileLoad.name}” will: {pendingProfileLoad.reasons.join(", ")}.
                </pre>
                <div className="actions" style={{ marginTop: 10 }}>
                  <button
                    className="btn btnPrimary"
                    type="button"
                    onClick={() => {
                      clearManualOverrides();
                      setForm(pendingProfileLoad.profile);
                      setPendingProfileLoad(null);
                      showToast(`Profile loaded: ${pendingProfileLoad.name}`);
                    }}
                  >
                    Load profile
                  </button>
                  <button className="btn" type="button" onClick={() => setPendingProfileLoad(null)}>
                    Cancel
                  </button>
                </div>
              </>
            ) : null}
          </div>
        </div>
            </ConfigPanel>
          ) : null}

          {activeConfigPanel === "config-market" ? (
            <ConfigPanel
              panelId="config-market"
              title="Market & Lookback"
              subtitle="Platform, symbol, interval, and window sizing."
              order={configPanelOrderIndex["config-market"]}
              open={isPanelOpen("config-market", true)}
              onToggle={handlePanelToggle("config-market")}
              maximized={isPanelMaximized("config-market")}
              onToggleMaximize={() => togglePanelMaximize("config-market")}
              style={configPanelStyle("config-market")}
              draggable={false}
              {...configPanelHandlers}
            >
      {configPage === "section-market" ? (
        <CollapsibleSection
          panelId="section-market"
          open={isPanelOpen("section-market", true)}
          onToggle={handlePanelToggle("section-market")}
          title="Market"
          meta="Pair, market type, interval, bars."
        >
      <div className="row rowSingle">
        <div className="field">
          <label className="label" htmlFor="platform">
            Platform
          </label>
          <select
            id="platform"
            className="select"
            value={form.platform}
            onChange={(e) => {
              const next = e.target.value as Platform;
              setPendingMarket(null);
              setPendingProfileLoad(null);
              setForm((f) => {
                const symbolSet = PLATFORM_SYMBOL_SET[next];
                const fallback = customSymbolByPlatform[next] || PLATFORM_DEFAULT_SYMBOL[next];
                const normalized = f.binanceSymbol.trim().toUpperCase();
                const nextSymbol = symbolSet.has(normalized) ? normalized : fallback;
                if (next === "binance") return { ...f, platform: next, binanceSymbol: nextSymbol };
                return {
                  ...f,
                  platform: next,
                  binanceSymbol: nextSymbol,
                  market: "spot",
                  binanceTestnet: false,
                  binanceLive: false,
                  tradeArmed: false,
                };
              });
            }}
          >
            {PLATFORMS.map((entry) => (
              <option key={entry} value={entry}>
                {PLATFORM_LABELS[entry]}
              </option>
            ))}
          </select>
          <div className="hint">
            Exchange platform for price data. Trading supports Binance + Coinbase (spot only); live bot is Binance-only. API keys can be stored per platform.
          </div>
        </div>
      </div>
      <div className="row">
        <div className="field">
          <label className="label" htmlFor="symbol">
            Trading pair
          </label>
          <select
            id="symbol"
            className={missingSymbol || Boolean(symbolFormatError) ? "select selectError" : "select"}
            value={symbolSelectValue}
            onChange={(e) => {
              const next = e.target.value;
              if (next === CUSTOM_SYMBOL_VALUE) {
                const fallback = customSymbolByPlatform[platform] || "";
                setForm((f) => ({ ...f, binanceSymbol: fallback }));
                return;
              }
              setForm((f) => ({ ...f, binanceSymbol: next }));
            }}
          >
            {platformSymbols.map((symbol) => (
              <option key={symbol} value={symbol}>
                {symbol}
              </option>
            ))}
            <option value={CUSTOM_SYMBOL_VALUE}>Custom...</option>
          </select>
          {symbolIsCustom ? (
            <input
              id="symbolCustom"
              className={missingSymbol || Boolean(symbolFormatError) ? "input inputError" : "input"}
              value={form.binanceSymbol}
              onChange={(e) => {
                const next = e.target.value.toUpperCase();
                setCustomSymbolByPlatform((prev) => ({ ...prev, [platform]: next }));
                setForm((f) => ({ ...f, binanceSymbol: next }));
              }}
              placeholder={PLATFORM_DEFAULT_SYMBOL[platform]}
              spellCheck={false}
              aria-label="Custom trading pair"
            />
          ) : null}
          <div className="hint" style={missingSymbol || symbolFormatError ? { color: "rgba(239, 68, 68, 0.85)" } : undefined}>
            {missingSymbol
              ? "Required."
              : symbolFormatError
                ? symbolFormatError
                : symbolIsCustom
                  ? isCoinbasePlatform
                    ? "Type any Coinbase product ID (e.g., BTC-USD)."
                    : `Type any ${platformLabel} symbol.`
                  : `Pick a common ${platformLabel} pair or choose Custom to type another symbol.`}
          </div>
        </div>

        <div className="field">
          <label className="label" htmlFor="market">
            Market
          </label>
          <select
            id="market"
            className="select"
            value={form.market}
            disabled={!isBinancePlatform}
            onChange={(e) => {
              const market = e.target.value as Market;
              setPendingMarket(null);
              setPendingProfileLoad(null);
              if (market === "margin" && !form.binanceLive) {
                setPendingMarket(market);
                return;
              }
              setForm((f) => ({ ...f, market, binanceTestnet: market === "margin" ? false : f.binanceTestnet }));
            }}
          >
            <option value="spot">Spot</option>
            <option value="margin">Margin</option>
            <option value="futures">Futures (USDT-M)</option>
          </select>
          <div className="hint">
            {isBinancePlatform
              ? "Margin orders require live mode. Futures can close positions via reduce-only."
              : "Market selection applies to Binance only."}
          </div>
          {pendingMarket === "margin" ? (
            <>
              <pre className="code" style={{ borderColor: "rgba(245, 158, 11, 0.35)", marginTop: 10 }}>
                Switching to Margin requires enabling Live orders (Binance has no margin test endpoint). This will place real orders once you arm trading and
                trade.
              </pre>
              <div className="actions" style={{ marginTop: 10 }}>
                <button
                  className="btn btnPrimary"
                  type="button"
                  onClick={() => {
                    setForm((f) => ({ ...f, market: "margin", binanceTestnet: false, binanceLive: true }));
                    setPendingMarket(null);
                    setConfirmLive(false);
                    showToast("Live orders enabled (required for margin)");
                  }}
                >
                  Enable live + switch
                </button>
                <button className="btn" type="button" onClick={() => setPendingMarket(null)}>
                  Cancel
                </button>
              </div>
            </>
          ) : null}
        </div>
      </div>

      <div className="row" style={{ marginTop: 12 }}>
        <div className="field">
          <label className="label" htmlFor="interval">
            Interval
          </label>
          <select
            id="interval"
            className={missingInterval ? "select selectError" : "select"}
            value={form.interval}
            onChange={(e) => setForm((f) => ({ ...f, interval: e.target.value }))}
          >
            {platformIntervals.map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
          </select>
          <div className="hint" style={missingInterval ? { color: "rgba(239, 68, 68, 0.85)" } : undefined}>
            {missingInterval ? "Required." : `${platformLabel} intervals: ${platformIntervals.join(", ")}.`}
          </div>
        </div>
        <div className="field">
          <label className="label" htmlFor="bars">
            Bars (0=auto, {barsRangeLabel})
          </label>
          <input
            id="bars"
            className={barsExceedsApi ? "input inputError" : "input"}
            type="number"
            min={0}
            max={barsMaxLabel ?? undefined}
            value={form.bars}
            onChange={(e) => setForm((f) => ({ ...f, bars: numFromInput(e.target.value, f.bars) }))}
          />
          <div className="hint" style={barsExceedsApi ? { color: "rgba(239, 68, 68, 0.85)" } : undefined}>
            {barsExceedsApi
              ? `API limit: max ${apiComputeLimits?.maxBarsLstm ?? "?"} bars for LSTM methods. Reduce bars or use method=10 (Kalman-only).`
              : `${barsAutoHint} ${barsRangeHint}`
            }
          </div>
        </div>
      </div>

      </CollapsibleSection>
      ) : null}

      {configPage === "section-lookback" ? (
        <CollapsibleSection
          panelId="section-lookback"
          open={isPanelOpen("section-lookback", true)}
          onToggle={handlePanelToggle("section-lookback")}
          title="Lookback"
          meta="Window length and bar overrides."
        >
        <div className="row">
          <div className="field">
            <label className="label" htmlFor="lookbackWindow">
              Lookback window
            </label>
            <input
              id="lookbackWindow"
              className={lookbackState.error && !lookbackState.overrideOn ? "input inputError" : "input"}
              value={form.lookbackWindow}
              disabled={form.lookbackBars >= 2}
              onChange={(e) => setForm((f) => ({ ...f, lookbackWindow: e.target.value }))}
              placeholder="24h"
              spellCheck={false}
            />
            <div className="hint" style={lookbackState.error && !lookbackState.overrideOn ? { color: "rgba(239, 68, 68, 0.85)" } : undefined}>
              {form.lookbackBars >= 2 ? "Ignored while Lookback bars override is set." : lookbackState.error ?? lookbackState.summary}
            </div>
          </div>
          <div className="field">
            <label className="label" htmlFor="lookbackBars">
              Lookback bars override (optional)
            </label>
            <input
              id="lookbackBars"
              className={lookbackState.error && lookbackState.overrideOn ? "input inputError" : "input"}
              type="number"
              min={0}
              value={form.lookbackBars}
              onChange={(e) => setForm((f) => ({ ...f, lookbackBars: numFromInput(e.target.value, f.lookbackBars) }))}
              placeholder="0 (auto)"
            />
            <div className="hint" style={lookbackState.error && lookbackState.overrideOn ? { color: "rgba(239, 68, 68, 0.85)" } : undefined}>
              {lookbackState.overrideOn ? lookbackState.error ?? lookbackState.summary : "0 = use lookbackWindow. Set ≥2 to override."}
            </div>
            {form.lookbackBars > 0 || lookbackState.error ? (
              <div className="actions" style={{ marginTop: 8 }}>
                <button
                  className="btn"
                  type="button"
                  disabled={lookbackState.bars < 3}
                  onClick={() => setForm((f) => ({ ...f, lookbackBars: Math.max(2, lookbackState.bars - 1) }))}
                >
                  Fit lookback to bars ({Math.max(0, lookbackState.bars - 1)})
                </button>
                <button
                  className="btn"
                  type="button"
                  disabled={form.lookbackBars <= 0}
                  onClick={() => setForm((f) => ({ ...f, lookbackBars: 0 }))}
                >
                  Clear override
                </button>
              </div>
            ) : null}
        </div>
      </div>

      </CollapsibleSection>
      ) : null}
            </ConfigPanel>
          ) : null}
          {activeConfigPanel === "config-strategy" ? (
          <ConfigPanel
            panelId="config-strategy"
            title="Strategy & Risk"
            subtitle="Thresholds, exits, sizing, and safeguards."
            order={configPanelOrderIndex["config-strategy"]}
            open={isPanelOpen("config-strategy", true)}
            onToggle={handlePanelToggle("config-strategy")}
            maximized={isPanelMaximized("config-strategy")}
            onToggleMaximize={() => togglePanelMaximize("config-strategy")}
            style={configPanelStyle("config-strategy")}
            draggable={false}
            {...configPanelHandlers}
          >
      {configPage === "section-thresholds" ? (
        <CollapsibleSection
          panelId="section-thresholds"
          open={isPanelOpen("section-thresholds", true)}
          onToggle={handlePanelToggle("section-thresholds")}
          title="Thresholds"
          meta="Method, positioning, entry/exit gates."
        >
        <div className="row" style={{ gridTemplateColumns: "1fr 1fr 1fr 1fr" }}>
          <div className="field">
            <div className="labelRow">
              <label className="label" htmlFor="method">
                Method
              </label>
              <InfoPopover label="Method details">
                <InfoList items={COMPLEX_TIPS.method} />
              </InfoPopover>
            </div>
            <select
              id="method"
              className="select"
              value={form.method}
              onChange={(e) => {
                const nextMethod = e.target.value as Method;
                markManualOverrides(["method"]);
                setForm((f) => ({
                  ...f,
                  method: nextMethod,
                  ...(nextMethod === "router" ? { optimizeOperations: false, sweepThreshold: false } : {}),
                }));
              }}
            >
              <option value="11">11 — Both (agreement gated)</option>
              <option value="blend">blend — Weighted average</option>
              <option value="router">router — Adaptive router</option>
              <option value="10">10 — Kalman only</option>
              <option value="01">01 — LSTM only</option>
            </select>
            <div className="hint">
              “11” only trades when both models agree on direction (up/down) outside the open threshold. “blend” averages the two predictions. “router” picks the best recent model.
            </div>
            {methodOverride ? (
              <div className="pillRow" style={{ marginTop: 6 }}>
                <span className="pill" style={{ color: "rgba(245, 158, 11, 0.9)", borderColor: "rgba(245, 158, 11, 0.35)" }}>
                  Manual override active
                </span>
                <button className="btnSmall" type="button" onClick={() => clearManualOverrides(["method"])}>
                  Unlock method
                </button>
              </div>
            ) : null}
          </div>
          <div className="field">
            <label className="label" htmlFor="positioning">
              Positioning
            </label>
            <select
              id="positioning"
              className="select"
              value={form.positioning}
              onChange={(e) => setForm((f) => ({ ...f, positioning: e.target.value as Positioning }))}
            >
              <option value="long-flat">Long / Flat</option>
              <option value="long-short" disabled={longShortBotDisabled}>
                Long / Short (futures)
              </option>
            </select>
            <div
              className="hint"
              style={
                form.positioning === "long-short" && form.market !== "futures"
                  ? { color: "rgba(245, 158, 11, 0.9)" }
                  : undefined
              }
            >
              {form.positioning === "long-short"
                ? `${form.market !== "futures" ? "Long/Short trading requires Futures market when trading is armed. " : ""}Live bot supports Long/Short on futures.`
                : "Down signals go FLAT (long/flat) or SHORT (long/short)."}
            </div>
          </div>
          <div className="field">
            <div className="labelRow">
              <label className="label" htmlFor="openThreshold">
                Open threshold (fraction)
              </label>
              <InfoPopover label="Threshold details">
                <InfoList items={COMPLEX_TIPS.thresholds} />
              </InfoPopover>
            </div>
            <input
              id="openThreshold"
              className={estimatedCosts.breakEven > 0 && form.openThreshold < estimatedCosts.breakEven ? "input inputError" : "input"}
              type="number"
              step="0.0001"
              min={0}
              value={form.openThreshold}
              onChange={(e) => {
                markManualOverrides(["openThreshold"]);
                setForm((f) => ({ ...f, openThreshold: numFromInput(e.target.value, f.openThreshold) }));
              }}
            />
            <div className="hint">
              Entry deadband. Default 0.001 = 0.1%. Break-even ≈ {fmtPct(estimatedCosts.breakEven, 3)} (round-trip cost ≈ {fmtPct(estimatedCosts.roundTrip, 3)}).
              {estimatedCosts.breakEven > 0 && form.openThreshold < estimatedCosts.breakEven
                ? " Consider increasing open threshold to avoid churn after costs."
                : null}
            </div>
            {thresholdOverrideKeys.length > 0 ? (
              <div className="pillRow" style={{ marginTop: 8 }}>
                <span className="pill" style={{ color: "rgba(245, 158, 11, 0.9)", borderColor: "rgba(245, 158, 11, 0.35)" }}>
                  Manual override: {thresholdOverrideKeys.includes("openThreshold") ? "open" : ""}
                  {thresholdOverrideKeys.includes("openThreshold") && thresholdOverrideKeys.includes("closeThreshold") ? " + " : ""}
                  {thresholdOverrideKeys.includes("closeThreshold") ? "close" : ""} threshold
                  {thresholdOverrideKeys.length > 1 ? "s" : ""}
                </span>
                <button className="btnSmall" type="button" onClick={() => clearManualOverrides(thresholdOverrideKeys)}>
                  Unlock thresholds
                </button>
              </div>
            ) : null}
            <div className="pillRow" style={{ marginTop: 10 }}>
              <button
                className="btnSmall"
                type="button"
                disabled={!(estimatedCosts.breakEven > 0)}
                onClick={() => {
                  const be = estimatedCosts.breakEven;
                  const open = Number((be * 2).toFixed(6));
                  const close = Number(be.toFixed(6));
                  markManualOverrides(["openThreshold", "closeThreshold"]);
                  setForm((f) => ({ ...f, openThreshold: open, closeThreshold: close }));
                  showToast("Set thresholds to conservative (2× break-even)");
                }}
              >
                Conservative (2× BE)
              </button>
              <button
                className="btnSmall"
                type="button"
                disabled={!(estimatedCosts.breakEven > 0)}
                onClick={() => {
                  const v = Number(estimatedCosts.breakEven.toFixed(6));
                  markManualOverrides(["openThreshold", "closeThreshold"]);
                  setForm((f) => ({ ...f, openThreshold: v, closeThreshold: v }));
                  showToast("Set thresholds to break-even");
                }}
              >
                Set open/close to break-even
              </button>
              <button
                className="btnSmall"
                type="button"
                onClick={() => {
                  markManualOverrides(["openThreshold", "closeThreshold"]);
                  setForm((f) => ({ ...f, openThreshold: defaultForm.openThreshold, closeThreshold: defaultForm.closeThreshold }));
                  showToast("Reset thresholds to defaults");
                }}
              >
                Reset thresholds
              </button>
            </div>
          </div>
          <div className="field">
            <div className="labelRow">
              <label className="label" htmlFor="closeThreshold">
                Close threshold (fraction)
              </label>
              <InfoPopover label="Threshold details">
                <InfoList items={COMPLEX_TIPS.thresholds} />
              </InfoPopover>
            </div>
            <input
              id="closeThreshold"
              className={
                estimatedCosts.breakEven > 0 && form.closeThreshold < estimatedCosts.breakEven
                  ? "input inputError"
                  : form.closeThreshold > form.openThreshold
                    ? "input inputWarn"
                    : "input"
              }
              type="number"
              step="0.0001"
              min={0}
              value={form.closeThreshold}
              onChange={(e) => {
                markManualOverrides(["closeThreshold"]);
                setForm((f) => ({ ...f, closeThreshold: numFromInput(e.target.value, f.closeThreshold) }));
              }}
            />
            <div className="hint">
              Exit deadband. Often smaller than open threshold to reduce churn.
              {estimatedCosts.breakEven > 0 && form.closeThreshold < estimatedCosts.breakEven ? " Below break-even (may churn after costs)." : null}
            </div>
            {form.closeThreshold > form.openThreshold ? (
              <div className="hint" style={{ color: "rgba(245, 158, 11, 0.9)" }}>
                Close threshold is above open threshold (inverted hysteresis). Usually close ≤ open.
              </div>
            ) : null}
          </div>
        </div>

        <div className="row" style={{ marginTop: 12, gridTemplateColumns: "1fr 1fr 1fr 1fr" }}>
          <div className="field">
            <div className="labelRow">
              <label className="label" htmlFor="minEdge">
                Min edge (fraction)
              </label>
              <InfoPopover label="Edge filters">
                <InfoList items={COMPLEX_TIPS.edge} />
              </InfoPopover>
            </div>
            <input
              id="minEdge"
              className="input"
              type="number"
              step="0.0001"
              min={0}
              value={form.minEdge}
              onChange={(e) => setForm((f) => ({ ...f, minEdge: numFromInput(e.target.value, f.minEdge) }))}
            />
            <div className="hint">
              {form.costAwareEdge
                ? `Cost-aware min edge ≈ ${fmtPct(minEdgeEffective, 3)} (break-even ${fmtPct(estimatedCosts.breakEven, 3)} + buffer).`
                : "Minimum predicted return to enter. 0 disables."}
            </div>
            <div className="pillRow" style={{ marginTop: 8 }}>
              <label className="pill">
                <input
                  type="checkbox"
                  checked={form.costAwareEdge}
                  onChange={(e) => setForm((f) => ({ ...f, costAwareEdge: e.target.checked }))}
                />
                Cost-aware edge
              </label>
            </div>
          </div>
          <div className="field">
            <div className="labelRow">
              <label className="label" htmlFor="edgeBuffer">
                Edge buffer (fraction)
              </label>
              <InfoPopover label="Edge filters">
                <InfoList items={COMPLEX_TIPS.edge} />
              </InfoPopover>
            </div>
            <input
              id="edgeBuffer"
              className="input"
              type="number"
              step="0.0001"
              min={0}
              value={form.edgeBuffer}
              onChange={(e) => setForm((f) => ({ ...f, edgeBuffer: numFromInput(e.target.value, f.edgeBuffer) }))}
            />
            <div className="hint">{form.costAwareEdge ? "Extra buffer above break-even." : "Used when cost-aware edge is enabled."}</div>
          </div>
          <div className="field">
            <div className="labelRow">
              <label className="label" htmlFor="minSignalToNoise">
                Min signal/vol (x)
              </label>
              <InfoPopover label="Signal-to-noise filter">
                <InfoList items={COMPLEX_TIPS.snr} />
              </InfoPopover>
            </div>
            <input
              id="minSignalToNoise"
              className="input"
              type="number"
              step="0.05"
              min={0}
              value={form.minSignalToNoise}
              onChange={(e) => setForm((f) => ({ ...f, minSignalToNoise: numFromInput(e.target.value, f.minSignalToNoise) }))}
              placeholder="0"
            />
            <div className="hint">
              {form.minSignalToNoise > 0 ? `${form.minSignalToNoise.toFixed(2)}x sigma` : "0 disables"} - edge / per-bar vol filter.
            </div>
          </div>
          <div className="field">
            <div className="labelRow">
              <label className="label" htmlFor="blendWeight">
                Blend weight (Kalman)
              </label>
              <InfoPopover label="Blend weight">
                <InfoList items={COMPLEX_TIPS.blend} />
              </InfoPopover>
            </div>
            <input
              id="blendWeight"
              className="input"
              type="number"
              step="0.05"
              min={0}
              max={1}
              value={form.blendWeight}
              onChange={(e) => setForm((f) => ({ ...f, blendWeight: numFromInput(e.target.value, f.blendWeight) }))}
              disabled={form.method !== "blend"}
            />
            <div className="hint">0 = LSTM only, 1 = Kalman only. Used with method=blend.</div>
          </div>
        </div>

        <div className="row" style={{ marginTop: 12, gridTemplateColumns: "1fr 1fr" }}>
          <div className="field">
            <div className="labelRow">
              <label className="label" htmlFor="routerLookback">
                Router lookback (bars)
              </label>
              <InfoPopover label="Router settings">
                <InfoList items={COMPLEX_TIPS.router} />
              </InfoPopover>
            </div>
            <input
              id="routerLookback"
              className="input"
              type="number"
              step="1"
              min={2}
              value={form.routerLookback}
              onChange={(e) => setForm((f) => ({ ...f, routerLookback: numFromInput(e.target.value, f.routerLookback) }))}
              disabled={form.method !== "router"}
            />
            <div className="hint">Used with method=router; evaluates recent signal accuracy.</div>
          </div>
          <div className="field">
            <div className="labelRow">
              <label className="label" htmlFor="routerMinScore">
                Router min score
              </label>
              <InfoPopover label="Router settings">
                <InfoList items={COMPLEX_TIPS.router} />
              </InfoPopover>
            </div>
            <input
              id="routerMinScore"
              className="input"
              type="number"
              step="0.05"
              min={0}
              max={1}
              value={form.routerMinScore}
              onChange={(e) => setForm((f) => ({ ...f, routerMinScore: numFromInput(e.target.value, f.routerMinScore) }))}
              disabled={form.method !== "router"}
            />
            <div className="hint">Accuracy × coverage threshold; below = hold.</div>
          </div>
        </div>

        <div className="row" style={{ marginTop: 12, gridTemplateColumns: "1fr 1fr 1fr" }}>
          <div className="field">
            <div className="labelRow">
              <label className="label" htmlFor="backtestRatio">
                Backtest ratio
              </label>
              <InfoPopover label="Backtest and tune split">
                <InfoList items={COMPLEX_TIPS.split} />
              </InfoPopover>
            </div>
            <input
              id="backtestRatio"
              className="input"
              type="number"
              step="0.01"
              min={0.01}
              max={0.99}
              value={form.backtestRatio}
              onChange={(e) => setForm((f) => ({ ...f, backtestRatio: numFromInput(e.target.value, f.backtestRatio) }))}
            />
            <div className="hint">Time-split holdout (last portion). Train and backtest are different.</div>
          </div>
          <div className="field">
            <div className="labelRow">
              <label className="label" htmlFor="tuneRatio">
                Tune ratio
              </label>
              <InfoPopover label="Backtest and tune split">
                <InfoList items={COMPLEX_TIPS.split} />
              </InfoPopover>
            </div>
            <input
              id="tuneRatio"
              className="input"
              type="number"
              step="0.01"
              min={0}
              max={0.99}
              value={form.tuneRatio}
              onChange={(e) => setForm((f) => ({ ...f, tuneRatio: numFromInput(e.target.value, f.tuneRatio) }))}
            />
            <div className="hint">Used only when optimizing/sweeping: tunes thresholds/method on the last part of the train split.</div>
          </div>
          <div className="field">
            <label className="label" htmlFor="fee">
              Fee (fraction)
            </label>
            <input
              id="fee"
              className="input"
              type="number"
              step="0.0001"
              min={0}
              value={form.fee}
              onChange={(e) => setForm((f) => ({ ...f, fee: numFromInput(e.target.value, f.fee) }))}
            />
            <div className="hint">
              Per-side ≈ {fmtPct(Math.max(0, form.fee) + Math.max(0, form.slippage) + Math.max(0, form.spread) / 2, 3)} (fee + slippage + spread/2).
            </div>
          </div>
        </div>
        <div
          className="hint"
          style={{
            marginTop: 6,
            color: splitPreview.warning ? "rgba(239, 68, 68, 0.85)" : undefined,
          }}
        >
          {splitPreview.summary}
        </div>

        <div className="row" style={{ marginTop: 12, gridTemplateColumns: "1fr 1fr" }}>
          <div className="field">
            <label className="label" htmlFor="slippage">
              Slippage (fraction per side)
            </label>
            <input
              id="slippage"
              className="input"
              type="number"
              step="0.0001"
              min={0}
              value={form.slippage}
              onChange={(e) => setForm((f) => ({ ...f, slippage: numFromInput(e.target.value, f.slippage) }))}
            />
            <div className="hint">Approx market impact on entry/exit. 0 disables.</div>
          </div>
          <div className="field">
            <label className="label" htmlFor="spread">
              Spread (fraction total)
            </label>
            <input
              id="spread"
              className="input"
              type="number"
              step="0.0001"
              min={0}
              value={form.spread}
              onChange={(e) => setForm((f) => ({ ...f, spread: numFromInput(e.target.value, f.spread) }))}
            />
            <div className="hint">Half is applied per side. 0 disables.</div>
          </div>
        </div>

      </CollapsibleSection>
      ) : null}

      {configPage === "section-risk" ? (
        <CollapsibleSection
          panelId="section-risk"
          open={isPanelOpen("section-risk", true)}
          onToggle={handlePanelToggle("section-risk")}
          title="Risk"
          meta="Stops, pacing, sizing, and kill-switches."
        >
        <div className="row" style={{ gridTemplateColumns: "1fr" }}>
          <div className="field">
          <div className="label">Bracket exits (fractions)</div>
            <div className="row" style={{ gridTemplateColumns: "1fr 1fr 1fr" }}>
              <div className="field">
                <label className="label" htmlFor="stopLoss">
                  Stop-loss
                </label>
                <input
                  id="stopLoss"
                  className="input"
                  type="number"
                  step="0.001"
                  min={0}
                  max={0.999}
                  value={form.stopLoss}
                  onChange={(e) => setForm((f) => ({ ...f, stopLoss: numFromInput(e.target.value, f.stopLoss) }))}
                  placeholder="0.02 (2%)"
                />
                <div className="hint">{form.stopLoss > 0 ? fmtPct(form.stopLoss, 2) : "0 disables"}</div>
              </div>
              <div className="field">
                <label className="label" htmlFor="takeProfit">
                  Take-profit
                </label>
                <input
                  id="takeProfit"
                  className="input"
                  type="number"
                  step="0.001"
                  min={0}
                  max={0.999}
                  value={form.takeProfit}
                  onChange={(e) => setForm((f) => ({ ...f, takeProfit: numFromInput(e.target.value, f.takeProfit) }))}
                  placeholder="0.03 (3%)"
                />
                <div className="hint">{form.takeProfit > 0 ? fmtPct(form.takeProfit, 2) : "0 disables"}</div>
              </div>
              <div className="field">
                <label className="label" htmlFor="trailingStop">
                  Trailing stop
                </label>
                <input
                  id="trailingStop"
                  className="input"
                  type="number"
                  step="0.001"
                  min={0}
                  max={0.999}
                  value={form.trailingStop}
                  onChange={(e) => setForm((f) => ({ ...f, trailingStop: numFromInput(e.target.value, f.trailingStop) }))}
                  placeholder="0.01 (1%)"
                />
                <div className="hint">{form.trailingStop > 0 ? fmtPct(form.trailingStop, 2) : "0 disables"}</div>
              </div>
            </div>
            <div className="row" style={{ gridTemplateColumns: "1fr 1fr 1fr", marginTop: 10 }}>
              <div className="field">
                <label className="label" htmlFor="stopLossVolMult">
                  Stop-loss vol mult
                </label>
                <input
                  id="stopLossVolMult"
                  className="input"
                  type="number"
                  step="0.05"
                  min={0}
                  value={form.stopLossVolMult}
                  onChange={(e) => setForm((f) => ({ ...f, stopLossVolMult: numFromInput(e.target.value, f.stopLossVolMult) }))}
                  placeholder="0"
                />
                <div className="hint">{form.stopLossVolMult > 0 ? `${form.stopLossVolMult.toFixed(2)}x sigma` : "0 disables"}</div>
              </div>
              <div className="field">
                <label className="label" htmlFor="takeProfitVolMult">
                  Take-profit vol mult
                </label>
                <input
                  id="takeProfitVolMult"
                  className="input"
                  type="number"
                  step="0.05"
                  min={0}
                  value={form.takeProfitVolMult}
                  onChange={(e) => setForm((f) => ({ ...f, takeProfitVolMult: numFromInput(e.target.value, f.takeProfitVolMult) }))}
                  placeholder="0"
                />
                <div className="hint">{form.takeProfitVolMult > 0 ? `${form.takeProfitVolMult.toFixed(2)}x sigma` : "0 disables"}</div>
              </div>
              <div className="field">
                <label className="label" htmlFor="trailingStopVolMult">
                  Trailing vol mult
                </label>
                <input
                  id="trailingStopVolMult"
                  className="input"
                  type="number"
                  step="0.05"
                  min={0}
                  value={form.trailingStopVolMult}
                  onChange={(e) => setForm((f) => ({ ...f, trailingStopVolMult: numFromInput(e.target.value, f.trailingStopVolMult) }))}
                  placeholder="0"
                />
                <div className="hint">
                  {form.trailingStopVolMult > 0 ? `${form.trailingStopVolMult.toFixed(2)}x sigma` : "0 disables"}
                </div>
              </div>
            </div>
            <div className="row" style={{ gridTemplateColumns: "1fr" }}>
              <div className="field">
                <label className="label" htmlFor="intrabarFill">
                  Intrabar fill
                </label>
                <select
                  id="intrabarFill"
                  className="select"
                  value={form.intrabarFill}
                  onChange={(e) => setForm((f) => ({ ...f, intrabarFill: e.target.value as IntrabarFill }))}
                >
                  <option value="stop-first">Stop-first (conservative)</option>
                  <option value="take-profit-first">Take-profit-first (optimistic)</option>
                </select>
                <div className="hint">If take-profit and stop are both hit within a bar.</div>
              </div>
            </div>
            <div className="hint">
              Optional bracket exits (uses OHLC high/low when available; otherwise close-only). Vol multiples use per-bar sigma and override fixed fractions when available.
            </div>
    	              </div>
    	            </div>

        <div className="row" style={{ marginTop: 12, gridTemplateColumns: "1fr" }}>
          <div className="field">
          <div className="label">Trade pacing (bars)</div>
            <div className="row" style={{ gridTemplateColumns: "1fr 1fr 1fr" }}>
              <div className="field">
                <label className="label" htmlFor="minHoldBars">
                  Min hold
                </label>
                <input
                  id="minHoldBars"
                  className="input"
                  type="number"
                  step={1}
                  min={0}
                  value={form.minHoldBars}
                  onChange={(e) => setForm((f) => ({ ...f, minHoldBars: numFromInput(e.target.value, f.minHoldBars) }))}
                  placeholder="0"
                />
                <div className="hint">
                  {form.minHoldBars > 0 ? `${Math.trunc(Math.max(0, form.minHoldBars))} bars` : "0 disables"} • blocks signal exits, not bracket exits
                </div>
              </div>
              <div className="field">
                <label className="label" htmlFor="cooldownBars">
                  Cooldown
                </label>
                <input
                  id="cooldownBars"
                  className="input"
                  type="number"
                  step={1}
                  min={0}
                  value={form.cooldownBars}
                  onChange={(e) => setForm((f) => ({ ...f, cooldownBars: numFromInput(e.target.value, f.cooldownBars) }))}
                  placeholder="0"
                />
                <div className="hint">{form.cooldownBars > 0 ? `${Math.trunc(Math.max(0, form.cooldownBars))} bars` : "0 disables"} • after exiting</div>
              </div>
              <div className="field">
                <label className="label" htmlFor="maxHoldBars">
                  Max hold
                </label>
                <input
                  id="maxHoldBars"
                  className="input"
                  type="number"
                  step={1}
                  min={0}
                  value={form.maxHoldBars}
                  onChange={(e) => setForm((f) => ({ ...f, maxHoldBars: numFromInput(e.target.value, f.maxHoldBars) }))}
                  placeholder="0"
                />
                <div className="hint">
                  {form.maxHoldBars > 0 ? `${Math.trunc(Math.max(0, form.maxHoldBars))} bars` : "0 disables"} • forces exit (MAX_HOLD)
                </div>
              </div>
            </div>
            <div className="hint">
              Helps reduce churn in noisy markets (applies to backtests + live bot; stateless signals/trades ignore state).
            </div>
          </div>
        </div>

        <div className="row" style={{ marginTop: 12, gridTemplateColumns: "1fr" }}>
          <div className="field">
          <div className="label">Sizing + filters</div>
            <div className="row" style={{ gridTemplateColumns: "1fr 1fr 1fr" }}>
              <div className="field">
                <label className="label" htmlFor="trendLookback">
                  Trend lookback (SMA bars)
                </label>
                <input
                  id="trendLookback"
                  className="input"
                  type="number"
                  step={1}
                  min={0}
                  value={form.trendLookback}
                  onChange={(e) => setForm((f) => ({ ...f, trendLookback: numFromInput(e.target.value, f.trendLookback) }))}
                  placeholder="0"
                />
                <div className="hint">
                  {form.trendLookback > 0 ? `${Math.trunc(Math.max(0, form.trendLookback))} bars` : "0 disables"} • filters entries to SMA trend
                </div>
              </div>
              <div className="field">
                <label className="label" htmlFor="maxPositionSize">
                  Max position size
                </label>
                <input
                  id="maxPositionSize"
                  className="input"
                  type="number"
                  step="0.1"
                  min={0}
                  value={form.maxPositionSize}
                  onChange={(e) => setForm((f) => ({ ...f, maxPositionSize: numFromInput(e.target.value, f.maxPositionSize) }))}
                  placeholder="1"
                />
                <div className="hint">Caps size/leverage (1 = full size).</div>
              </div>
              <div className="field">
                <label className="label" htmlFor="maxVolatility">
                  Max volatility (annualized)
                </label>
                <input
                  id="maxVolatility"
                  className="input"
                  type="number"
                  step="0.01"
                  min={0}
                  value={form.maxVolatility}
                  onChange={(e) => setForm((f) => ({ ...f, maxVolatility: numFromInput(e.target.value, f.maxVolatility) }))}
                  placeholder="0"
                />
                <div className="hint">
                  {form.maxVolatility > 0 ? fmtPct(form.maxVolatility, 2) : "0 disables"} • blocks entries when vol is too high
                </div>
              </div>
            </div>
            <div className="row" style={{ gridTemplateColumns: "1fr 1fr 1fr 1fr 1fr", marginTop: 10 }}>
              <div className="field">
                <label className="label" htmlFor="volTarget">
                  Vol target
                </label>
                <input
                  id="volTarget"
                  className="input"
                  type="number"
                  step="0.01"
                  min={0}
                  value={form.volTarget}
                  onChange={(e) => setForm((f) => ({ ...f, volTarget: numFromInput(e.target.value, f.volTarget) }))}
                  placeholder="0"
                />
                <div className="hint">{form.volTarget > 0 ? fmtPct(form.volTarget, 2) : "0 disables"} • annualized</div>
              </div>
              <div className="field">
                <label className="label" htmlFor="volLookback">
                  Vol lookback
                </label>
                <input
                  id="volLookback"
                  className="input"
                  type="number"
                  step={1}
                  min={0}
                  value={form.volLookback}
                  onChange={(e) => setForm((f) => ({ ...f, volLookback: numFromInput(e.target.value, f.volLookback) }))}
                  placeholder="20"
                />
                <div
                  className="hint"
                  style={
                    form.volTarget > 0 && !(form.volEwmaAlpha > 0 && form.volEwmaAlpha < 1) && form.volLookback < 2
                      ? { color: "rgba(239, 68, 68, 0.85)" }
                      : undefined
                  }
                >
                  {form.volTarget > 0 && !(form.volEwmaAlpha > 0 && form.volEwmaAlpha < 1) && form.volLookback < 2
                    ? "Must be >=2 when vol target is set (unless EWMA alpha is provided)."
                    : "Realized-vol lookback window (bars)."}
                </div>
              </div>
              <div className="field">
                <label className="label" htmlFor="volEwmaAlpha">
                  Vol EWMA alpha
                </label>
                <input
                  id="volEwmaAlpha"
                  className="input"
                  type="number"
                  step="0.01"
                  min={0}
                  max={0.999}
                  value={form.volEwmaAlpha}
                  onChange={(e) => setForm((f) => ({ ...f, volEwmaAlpha: numFromInput(e.target.value, f.volEwmaAlpha) }))}
                  placeholder="0"
                />
                <div className="hint">Optional; overrides lookback when set (0 disables).</div>
              </div>
              <div className="field">
                <label className="label" htmlFor="volFloor">
                  Vol floor
                </label>
                <input
                  id="volFloor"
                  className="input"
                  type="number"
                  step="0.01"
                  min={0}
                  value={form.volFloor}
                  onChange={(e) => setForm((f) => ({ ...f, volFloor: numFromInput(e.target.value, f.volFloor) }))}
                  placeholder="0"
                />
                <div className="hint">Annualized floor for sizing (0 disables).</div>
              </div>
              <div className="field">
                <label className="label" htmlFor="volScaleMax">
                  Vol scale max
                </label>
                <input
                  id="volScaleMax"
                  className="input"
                  type="number"
                  step="0.1"
                  min={0}
                  value={form.volScaleMax}
                  onChange={(e) => setForm((f) => ({ ...f, volScaleMax: numFromInput(e.target.value, f.volScaleMax) }))}
                  placeholder="1"
                />
                <div className="hint">Caps volatility-based scaling.</div>
              </div>
            </div>
            <div className="row" style={{ gridTemplateColumns: "1fr 1fr", marginTop: 10 }}>
              <div className="field">
                <label className="label" htmlFor="rebalanceBars">
                  Rebalance bars
                </label>
                <input
                  id="rebalanceBars"
                  className="input"
                  type="number"
                  step={1}
                  min={0}
                  value={form.rebalanceBars}
                  onChange={(e) => setForm((f) => ({ ...f, rebalanceBars: numFromInput(e.target.value, f.rebalanceBars) }))}
                  placeholder="0"
                />
                <div className="hint">
                  {form.rebalanceBars > 0 ? `${Math.trunc(Math.max(0, form.rebalanceBars))} bars` : "0 disables"} • resize toward target
                </div>
              </div>
              <div className="field">
                <label className="label" htmlFor="rebalanceThreshold">
                  Rebalance threshold
                </label>
                <input
                  id="rebalanceThreshold"
                  className="input"
                  type="number"
                  step="0.01"
                  min={0}
                  value={form.rebalanceThreshold}
                  onChange={(e) =>
                    setForm((f) => ({ ...f, rebalanceThreshold: numFromInput(e.target.value, f.rebalanceThreshold) }))
                  }
                  placeholder="0"
                />
                <div className="hint">
                  {form.rebalanceThreshold > 0 ? form.rebalanceThreshold.toFixed(2) : "0 disables"} • min abs size delta
                </div>
              </div>
            </div>
            <div className="row" style={{ gridTemplateColumns: "1fr 1fr", marginTop: 10 }}>
              <div className="field">
                <label className="label" htmlFor="rebalanceCostMult">
                  Rebalance cost mult
                </label>
                <input
                  id="rebalanceCostMult"
                  className="input"
                  type="number"
                  step="0.1"
                  min={0}
                  value={form.rebalanceCostMult}
                  onChange={(e) => setForm((f) => ({ ...f, rebalanceCostMult: numFromInput(e.target.value, f.rebalanceCostMult) }))}
                  placeholder="0"
                />
                <div className="hint">
                  {form.rebalanceCostMult > 0 ? form.rebalanceCostMult.toFixed(2) : "0 disables"} • size delta &ge; cost x mult
                </div>
              </div>
              <div className="field">
                <label className="label" htmlFor="fundingRate">
                  Funding rate (annualized)
                </label>
                <input
                  id="fundingRate"
                  className="input"
                  type="number"
                  step="0.01"
                  value={form.fundingRate}
                  onChange={(e) => setForm((f) => ({ ...f, fundingRate: numFromInput(e.target.value, f.fundingRate) }))}
                  placeholder="0"
                />
                <div className="hint">{form.fundingRate !== 0 ? fmtPct(form.fundingRate, 2) : "0 disables"} • backtests only</div>
              </div>
            </div>
            <div className="pillRow" style={{ marginTop: 8 }}>
              <label className="pill">
                <input
                  type="checkbox"
                  checked={form.rebalanceGlobal}
                  onChange={(e) => setForm((f) => ({ ...f, rebalanceGlobal: e.target.checked }))}
                />
                Rebalance global cadence
              </label>
              <label className="pill">
                <input
                  type="checkbox"
                  checked={form.fundingBySide}
                  onChange={(e) => setForm((f) => ({ ...f, fundingBySide: e.target.checked }))}
                />
                Funding by side
              </label>
            </div>
            <div className="pillRow" style={{ marginTop: 6 }}>
              <label className="pill">
                <input
                  type="checkbox"
                  checked={form.rebalanceResetOnSignal}
                  onChange={(e) => setForm((f) => ({ ...f, rebalanceResetOnSignal: e.target.checked }))}
                />
                Reset rebalance on signal
              </label>
              <label className="pill">
                <input
                  type="checkbox"
                  checked={form.fundingOnOpen}
                  onChange={(e) => setForm((f) => ({ ...f, fundingOnOpen: e.target.checked }))}
                />
                Funding on open bar
              </label>
            </div>
            <div className="hint">Defaults: rebalancing anchors to entry age; funding is side-agnostic and only charged if the position survives the bar.</div>
            <div className="hint">Vol sizing scales position by target/realized volatility when vol target is set.</div>
          </div>
        </div>

        <div className="row" style={{ marginTop: 12, gridTemplateColumns: "1fr" }}>
          <div className="field">
          <div className="label">Risk kill-switches</div>
            <div className="row" style={{ gridTemplateColumns: "1fr 1fr 1fr" }}>
              <div className="field">
                <label className="label" htmlFor="maxDrawdown">
                  Max drawdown
                </label>
                <input
                  id="maxDrawdown"
                  className="input"
                  type="number"
                  step="0.01"
                  min={0}
                  max={0.999}
                  value={form.maxDrawdown}
                  onChange={(e) => setForm((f) => ({ ...f, maxDrawdown: numFromInput(e.target.value, f.maxDrawdown) }))}
                  placeholder="0.20 (20%)"
                />
                <div className="hint">{form.maxDrawdown > 0 ? fmtPct(form.maxDrawdown, 2) : "0 disables"}</div>
              </div>
              <div className="field">
                <label className="label" htmlFor="maxDailyLoss">
                  Max daily loss
                </label>
                <input
                  id="maxDailyLoss"
                  className="input"
                  type="number"
                  step="0.01"
                  min={0}
                  max={0.999}
                  value={form.maxDailyLoss}
                  onChange={(e) => setForm((f) => ({ ...f, maxDailyLoss: numFromInput(e.target.value, f.maxDailyLoss) }))}
                  placeholder="0.10 (10%)"
                />
                <div className="hint">{form.maxDailyLoss > 0 ? fmtPct(form.maxDailyLoss, 2) : "0 disables"}</div>
              </div>
              <div className="field">
                <label className="label" htmlFor="maxOrderErrors">
                  Max order errors
                </label>
                <input
                  id="maxOrderErrors"
                  className="input"
                  type="number"
                  step="1"
                  min={0}
                  value={form.maxOrderErrors}
                  onChange={(e) => setForm((f) => ({ ...f, maxOrderErrors: numFromInput(e.target.value, f.maxOrderErrors) }))}
                  placeholder="3"
                />
                <div className="hint">{form.maxOrderErrors >= 1 ? `${Math.trunc(form.maxOrderErrors)} errors` : "0 disables"}</div>
              </div>
            </div>
            <div className="hint">When set, the live bot halts (and forces exit) on max drawdown, max daily loss, or consecutive order failures.</div>
          </div>
        </div>

        <div className="row" style={{ marginTop: 12 }}>
          <div className="field">
            <div className="labelRow">
              <label className="label" htmlFor="norm">
                Normalization
              </label>
              <InfoPopover label="LSTM normalization">
                <InfoList items={COMPLEX_TIPS.lstm} />
              </InfoPopover>
            </div>
            <select
              id="norm"
              className="select"
              value={form.normalization}
              onChange={(e) => setForm((f) => ({ ...f, normalization: e.target.value as Normalization }))}
            >
              <option value="standard">standard</option>
              <option value="minmax">minmax</option>
              <option value="log">log</option>
              <option value="none">none</option>
            </select>
            <div className="hint">Used for the LSTM pipeline.</div>
          </div>
          <div className="field">
            <div className="labelRow">
              <label className="label" htmlFor="epochs">
                Epochs / Hidden size
              </label>
              <InfoPopover label="LSTM epochs and hidden size">
                <InfoList items={COMPLEX_TIPS.lstm} />
              </InfoPopover>
            </div>
            <div className="row" style={{ gridTemplateColumns: "1fr 1fr" }}>
              <input
                id="epochs"
                className={epochsExceedsApi ? "input inputError" : "input"}
                type="number"
                min={0}
                value={form.epochs}
                onChange={(e) => setForm((f) => ({ ...f, epochs: numFromInput(e.target.value, f.epochs) }))}
              />
              <input
                id="hiddenSize"
                aria-label="Hidden size"
                className={hiddenSizeExceedsApi ? "input inputError" : "input"}
                type="number"
                min={1}
                value={form.hiddenSize}
                onChange={(e) => setForm((f) => ({ ...f, hiddenSize: numFromInput(e.target.value, f.hiddenSize) }))}
              />
            </div>
            <div className="hint" style={epochsExceedsApi || hiddenSizeExceedsApi ? { color: "rgba(239, 68, 68, 0.85)" } : undefined}>
              {!apiLstmEnabled
                ? "Ignored for Kalman-only."
                : epochsExceedsApi || hiddenSizeExceedsApi
                  ? `API limits: epochs ≤ ${apiComputeLimits?.maxEpochs ?? "?"}, hidden ≤ ${apiComputeLimits?.maxHiddenSize ?? "?"}.`
                  : "Higher = slower. For quick iteration, reduce epochs."}
            </div>
          </div>
        </div>

      </CollapsibleSection>
      ) : null}
            </ConfigPanel>
          ) : null}
          {activeConfigPanel === "config-optimization" ? (
          <ConfigPanel
            panelId="config-optimization"
            title="Optimization & Runs"
            subtitle="Tune sweeps and launch optimizer runs."
            order={configPanelOrderIndex["config-optimization"]}
            open={isPanelOpen("config-optimization", true)}
            onToggle={handlePanelToggle("config-optimization")}
            maximized={isPanelMaximized("config-optimization")}
            onToggleMaximize={() => togglePanelMaximize("config-optimization")}
            style={configPanelStyle("config-optimization")}
            draggable={false}
            {...configPanelHandlers}
          >
      {configPage === "section-optimizer-run" ? (
        <CollapsibleSection
          panelId="section-optimizer-run"
          open={isPanelOpen("section-optimizer-run", true)}
          onToggle={handlePanelToggle("section-optimizer-run")}
          title="Optimizer run"
          meta="Kick off /optimizer/run with the current config or a CSV source."
        >
        <div className="row">
          <div className="field">
            <div className="label">Request</div>
            <div className="row" style={{ gridTemplateColumns: "1fr 1fr 1fr" }}>
              <div className="field">
                <label className="label" htmlFor="optimizerRunSource">
                  Source
                </label>
                <select
                  id="optimizerRunSource"
                  className="select"
                  value={optimizerRunForm.source}
                  onChange={(e) => updateOptimizerRunForm({ source: e.target.value as OptimizerSource })}
                >
                  <option value="binance">Binance</option>
                  <option value="coinbase">Coinbase</option>
                  <option value="kraken">Kraken</option>
                  <option value="poloniex">Poloniex</option>
                  <option value="csv">CSV (file path)</option>
                </select>
                <div className="hint">Exchange determines symbol format; CSV bypasses the API.</div>
              </div>
              {optimizerRunForm.source === "csv" ? (
                <>
                  <div className="field">
                    <label className="label" htmlFor="optimizerRunDataPath">
                      CSV path
                    </label>
                    <input
                      id="optimizerRunDataPath"
                      className={optimizerRunValidationError ? "input inputError" : "input"}
                      value={optimizerRunForm.dataPath}
                      onChange={(e) => updateOptimizerRunForm({ dataPath: e.target.value })}
                      placeholder="data/my-prices.csv"
                    />
                    <div className="hint">Required for CSV source.</div>
                  </div>
                  <div className="field">
                    <label className="label" htmlFor="optimizerRunPriceColumn">
                      Price / High / Low
                    </label>
                    <input
                      id="optimizerRunPriceColumn"
                      className="input"
                      value={optimizerRunForm.priceColumn}
                      onChange={(e) => updateOptimizerRunForm({ priceColumn: e.target.value })}
                      placeholder="close"
                    />
                    <div className="row" style={{ gridTemplateColumns: "1fr 1fr", marginTop: 6 }}>
                      <input
                        className={optimizerRunValidationError ? "input inputError" : "input"}
                        value={optimizerRunForm.highColumn}
                        onChange={(e) => updateOptimizerRunForm({ highColumn: e.target.value })}
                        placeholder="high (optional)"
                      />
                      <input
                        className={optimizerRunValidationError ? "input inputError" : "input"}
                        value={optimizerRunForm.lowColumn}
                        onChange={(e) => updateOptimizerRunForm({ lowColumn: e.target.value })}
                        placeholder="low (optional)"
                      />
                    </div>
                    <div className="hint">Provide both High/Low or leave both blank.</div>
                  </div>
                </>
              ) : (
                <div className="field" style={{ gridColumn: "span 2" }}>
                  <label className="label" htmlFor="optimizerRunSymbol">
                    Symbol
                  </label>
                  <input
                    id="optimizerRunSymbol"
                    className={optimizerRunValidationError ? "input inputError" : "input"}
                    value={optimizerRunForm.symbol}
                    onChange={(e) => updateOptimizerRunForm({ symbol: e.target.value.toUpperCase() })}
                    placeholder="BTCUSDT"
                  />
                  <div className="hint">Defaults to the current symbol/platform.</div>
                </div>
              )}
            </div>

            <div className="row" style={{ marginTop: 10, gridTemplateColumns: "1fr 1fr 1fr" }}>
              <div className="field">
                <div className="labelRow">
                  <label className="label" htmlFor="optimizerRunIntervals">
                    Intervals
                  </label>
                  <InfoPopover label="Equity tip: intervals">
                    <InfoList items={EQUITY_TIPS.intervals} />
                  </InfoPopover>
                </div>
                <input
                  id="optimizerRunIntervals"
                  className="input"
                  value={optimizerRunForm.intervals}
                  onChange={(e) => updateOptimizerRunForm({ intervals: e.target.value })}
                  placeholder="1h,4h,1d"
                />
                <div className="hint">Comma-separated list.</div>
              </div>
              <div className="field">
                <label className="label" htmlFor="optimizerRunLookback">
                  Lookback window
                </label>
                <input
                  id="optimizerRunLookback"
                  className="input"
                  value={optimizerRunForm.lookbackWindow}
                  onChange={(e) => updateOptimizerRunForm({ lookbackWindow: e.target.value })}
                  placeholder="7d"
                />
                <div className="hint">Same format as main form (e.g., 7d, 30d).</div>
              </div>
              <div className="field">
                <div className="labelRow">
                  <label className="label" htmlFor="optimizerRunTrials">
                    Trials / Timeout / Seed
                  </label>
                  <InfoPopover label="Equity tip: trials and timeout">
                    <InfoList items={EQUITY_TIPS.trials} />
                  </InfoPopover>
                </div>
                <div className="row" style={{ gridTemplateColumns: "1fr 1fr 1fr" }}>
                  <input
                    id="optimizerRunTrials"
                    className="input"
                    type="number"
                    min={1}
                    value={optimizerRunForm.trials}
                    onChange={(e) => updateOptimizerRunForm({ trials: e.target.value })}
                    placeholder="50"
                  />
                  <input
                    className="input"
                    type="number"
                    min={5}
                    value={optimizerRunForm.timeoutSec}
                    onChange={(e) => updateOptimizerRunForm({ timeoutSec: e.target.value })}
                    placeholder="60"
                  />
                  <input
                    className="input"
                    type="number"
                    value={optimizerRunForm.seed}
                    onChange={(e) => updateOptimizerRunForm({ seed: e.target.value })}
                    placeholder="42"
                  />
                </div>
                <div className="hint">Numbers are optional; blanks are omitted.</div>
              </div>
            </div>

            <div className="row" style={{ marginTop: 10, gridTemplateColumns: "1fr 1fr 1fr" }}>
              <div className="field">
                <div className="labelRow">
                  <label className="label" htmlFor="optimizerRunObjective">
                    Objective
                  </label>
                  <InfoPopover label="Equity tip: objective">
                    <InfoList items={EQUITY_TIPS.objective} />
                  </InfoPopover>
                </div>
                <input
                  id="optimizerRunObjective"
                  className="input"
                  value={optimizerRunForm.objective}
                  onChange={(e) => updateOptimizerRunForm({ objective: e.target.value })}
                  placeholder="annualized-equity"
                />
                <div className="hint">Matches backend objective names.</div>
              </div>
              <div className="field">
                <div className="labelRow">
                  <label className="label" htmlFor="optimizerRunBacktestRatio">
                    Backtest / Tune ratio
                  </label>
                  <InfoPopover label="Equity tip: backtest and tune ratios">
                    <InfoList items={EQUITY_TIPS.ratios} />
                  </InfoPopover>
                </div>
                <div className="row" style={{ gridTemplateColumns: "1fr 1fr" }}>
                  <input
                    id="optimizerRunBacktestRatio"
                    className="input"
                    type="number"
                    min={0}
                    max={0.9}
                    step="0.01"
                    value={optimizerRunForm.backtestRatio}
                    onChange={(e) => updateOptimizerRunForm({ backtestRatio: e.target.value })}
                    placeholder="0.2"
                  />
                  <input
                    id="optimizerRunTuneRatio"
                    className="input"
                    type="number"
                    min={0}
                    max={0.9}
                    step="0.01"
                    value={optimizerRunForm.tuneRatio}
                    onChange={(e) => updateOptimizerRunForm({ tuneRatio: e.target.value })}
                    placeholder="0.25"
                  />
                </div>
                <div className="hint">Leave blank to use defaults.</div>
              </div>
              <div className="field">
                <label className="label" htmlFor="optimizerRunExtra">
                  Extra JSON (advanced)
                </label>
                <textarea
                  id="optimizerRunExtra"
                  className={optimizerRunExtras.error ? "input inputError" : "input"}
                  value={optimizerRunForm.extraJson}
                  onChange={(e) => updateOptimizerRunForm({ extraJson: e.target.value })}
                  placeholder='{"minSharpe":1.0,"minRoundTrips":5}'
                  rows={4}
                  spellCheck={false}
                />
                <div className="hint">{optimizerRunExtras.error ?? "Merged into the payload verbatim."}</div>
              </div>
            </div>

            <div className="actions" style={{ marginTop: 10 }}>
              <button
                className="btn btnPrimary"
                type="button"
                disabled={optimizerRunUi.loading || Boolean(optimizerRunValidationError)}
                onClick={runOptimizer}
              >
                {optimizerRunUi.loading ? "Running optimizer…" : "Run optimizer"}
              </button>
              <button className="btn" type="button" disabled={!optimizerRunUi.loading} onClick={cancelOptimizerRun}>
                Cancel run
              </button>
              <button
                className="btn"
                type="button"
                onClick={() => {
                  syncOptimizerRunForm();
                }}
              >
                Sync from config
              </button>
              <button className="btn" type="button" onClick={applyEquityPreset}>
                Preset: Equity focus
              </button>
              <InfoPopover label="Equity options" align="left">
                <InfoList items={EQUITY_TIPS.preset} />
              </InfoPopover>
              <button className="btn" type="button" onClick={resetOptimizerRunForm}>
                Reset
              </button>
              <span className="hint">
                {optimizerRunValidationError
                  ? optimizerRunValidationError
                  : optimizerRunUi.loading
                    ? "Submitting to /optimizer/run…"
                    : "Uses the same auth/proxy settings as other requests."}
              </span>
            </div>
          </div>

          <div className="field">
            <div className="label">Result</div>
            {optimizerRunUi.error ? (
              <pre className="code" style={{ borderColor: "rgba(239, 68, 68, 0.35)" }}>
                {optimizerRunUi.error}
              </pre>
            ) : null}
            {optimizerRunUi.response ? (
              <div className="hint" style={{ marginBottom: 8 }}>
                Last run: {optimizerRunUi.lastRunAtMs ? fmtTimeMs(optimizerRunUi.lastRunAtMs) : "just now"}
              </div>
            ) : null}
            {optimizerRunUi.response?.stdout ? (
              <details className="details" open>
                <summary>Stdout</summary>
                <pre className="code">{optimizerRunUi.response.stdout}</pre>
              </details>
            ) : null}
            {optimizerRunUi.response?.stderr ? (
              <details className="details">
                <summary>Stderr</summary>
                <pre className="code">{optimizerRunUi.response.stderr}</pre>
              </details>
            ) : null}
            {optimizerRunRecordJson ? (
              <details className="details" open>
                <summary>Last record</summary>
                <pre className="code">{optimizerRunRecordJson}</pre>
              </details>
            ) : (
              <div className="hint">No optimizer run yet.</div>
            )}
          </div>
        </div>
      </CollapsibleSection>
      ) : null}

      {configPage === "section-optimization" ? (
        <CollapsibleSection
          panelId="section-optimization"
          open={isPanelOpen("section-optimization", true)}
          onToggle={handlePanelToggle("section-optimization")}
          title="Optimization"
          meta="Tuning sweeps, presets, and constraints."
        >
        <div className="row">
          <div className="field">
          <div className="labelRow">
            <div className="label">Optimization</div>
            <InfoPopover label="Optimization modes">
              <InfoList items={COMPLEX_TIPS.optimization} />
            </InfoPopover>
          </div>
            <div className="pillRow">
              <label className="pill">
                <input
                  type="checkbox"
                  checked={form.sweepThreshold}
                  disabled={form.method === "router"}
                  onChange={(e) => setForm((f) => ({ ...f, sweepThreshold: e.target.checked, optimizeOperations: false }))}
                />
                Sweep thresholds
              </label>
              <label className="pill">
                <input
                  type="checkbox"
                  checked={form.optimizeOperations}
                  disabled={form.method === "router"}
                  onChange={(e) => setForm((f) => ({ ...f, optimizeOperations: e.target.checked, sweepThreshold: false }))}
                />
                Optimize operations (method + thresholds)
              </label>
            </div>
            <div className="hint">
              Tunes on the last part of the train split (fit/tune), then evaluates on the held-out backtest.
              {form.method === "router" ? " Router mode disables optimize/sweep." : ""}
            </div>
            <div className="pillRow" style={{ marginTop: 10 }}>
              <button
                className="btnSmall"
                type="button"
                disabled={form.method === "router"}
                onClick={() => {
                  setForm((f) => ({
                    ...f,
                    optimizeOperations: true,
                    sweepThreshold: false,
                    minRoundTrips: Math.max(5, Math.trunc(f.minRoundTrips)),
                    walkForwardFolds: Math.max(5, Math.trunc(f.walkForwardFolds)),
                    walkForwardEmbargoBars: Math.max(1, Math.trunc(f.walkForwardEmbargoBars)),
                  }));
                  showToast("Preset: safe optimize (min round trips + folds)");
                }}
              >
                Preset: Safe optimize
              </button>
              <button
                className="btnSmall"
                type="button"
                disabled={form.method === "router"}
                onClick={() => {
                  setForm((f) => ({
                    ...f,
                    sweepThreshold: true,
                    optimizeOperations: false,
                    minRoundTrips: Math.max(3, Math.trunc(f.minRoundTrips)),
                    walkForwardFolds: Math.max(3, Math.trunc(f.walkForwardFolds)),
                    walkForwardEmbargoBars: Math.max(1, Math.trunc(f.walkForwardEmbargoBars)),
                  }));
                  showToast("Preset: fast sweep (min round trips + folds)");
                }}
              >
                Preset: Fast sweep
              </button>
            </div>
            <div className="row" style={{ marginTop: 10, gridTemplateColumns: "1fr 1fr 1fr 1fr 1fr" }}>
              <div className="field">
                <div className="labelRow">
                  <label className="label" htmlFor="tuneObjective">
                    Tune objective
                  </label>
                  <InfoPopover label="Tune objective">
                    <InfoList items={COMPLEX_TIPS.tuneObjective} />
                  </InfoPopover>
                </div>
                <select
                  id="tuneObjective"
                  className="select"
                  value={form.tuneObjective}
                  onChange={(e) => setForm((f) => ({ ...f, tuneObjective: e.target.value }))}
                >
                  {TUNE_OBJECTIVES.map((o) => (
                    <option key={o} value={o}>
                      {o}
                    </option>
                  ))}
                </select>
                <div className="hint">Used by “Optimize thresholds/operations”.</div>
              </div>
              <div className="field">
                <label className="label" htmlFor="tunePenaltyMaxDrawdown">
                  DD penalty
                </label>
                <input
                  id="tunePenaltyMaxDrawdown"
                  className="input"
                  type="number"
                  step="0.1"
                  min={0}
                  value={form.tunePenaltyMaxDrawdown}
                  onChange={(e) => setForm((f) => ({ ...f, tunePenaltyMaxDrawdown: numFromInput(e.target.value, f.tunePenaltyMaxDrawdown) }))}
                />
                <div className="hint">Applied when objective includes drawdown.</div>
              </div>
              <div className="field">
                <label className="label" htmlFor="tunePenaltyTurnover">
                  Turnover penalty
                </label>
                <input
                  id="tunePenaltyTurnover"
                  className="input"
                  type="number"
                  step="0.01"
                  min={0}
                  value={form.tunePenaltyTurnover}
                  onChange={(e) => setForm((f) => ({ ...f, tunePenaltyTurnover: numFromInput(e.target.value, f.tunePenaltyTurnover) }))}
                />
                <div className="hint">Applied when objective includes turnover.</div>
              </div>
              <div className="field">
                <label className="label" htmlFor="minRoundTrips">
                  Min round trips
                </label>
                <input
                  id="minRoundTrips"
                  className="input"
                  type="number"
                  step="1"
                  min={0}
                  value={form.minRoundTrips}
                  onChange={(e) => setForm((f) => ({ ...f, minRoundTrips: numFromInput(e.target.value, f.minRoundTrips) }))}
                />
                <div className="hint">Only used when optimizing/sweeping. 0 disables.</div>
              </div>
              <div className="field">
                <div className="labelRow">
                  <label className="label" htmlFor="walkForwardFolds">
                    Walk-forward folds
                  </label>
                  <InfoPopover label="Walk-forward folds">
                    <InfoList items={COMPLEX_TIPS.walkForward} />
                  </InfoPopover>
                </div>
                <input
                  id="walkForwardFolds"
                  className="input"
                  type="number"
                  step="1"
                  min={1}
                  value={form.walkForwardFolds}
                  onChange={(e) => setForm((f) => ({ ...f, walkForwardFolds: numFromInput(e.target.value, f.walkForwardFolds) }))}
                />
                <div className="hint">Used for tune scoring + backtest variability.</div>
              </div>
            </div>
            <div className="row" style={{ marginTop: 10, gridTemplateColumns: "1fr 1fr 1fr 1fr" }}>
              <div className="field">
                <label className="label" htmlFor="walkForwardEmbargoBars">
                  Walk-forward embargo bars
                </label>
                <input
                  id="walkForwardEmbargoBars"
                  className="input"
                  type="number"
                  step="1"
                  min={0}
                  value={form.walkForwardEmbargoBars}
                  onChange={(e) =>
                    setForm((f) => ({ ...f, walkForwardEmbargoBars: numFromInput(e.target.value, f.walkForwardEmbargoBars) }))
                  }
                />
                <div className="hint">Drop bars around fold edges (0 disables).</div>
              </div>
              <div className="field">
                <label className="label" htmlFor="tuneStressVolMult">
                  Stress vol mult
                </label>
                <input
                  id="tuneStressVolMult"
                  className="input"
                  type="number"
                  step="0.1"
                  min={0}
                  value={form.tuneStressVolMult}
                  onChange={(e) => setForm((f) => ({ ...f, tuneStressVolMult: numFromInput(e.target.value, f.tuneStressVolMult) }))}
                />
                <div className="hint">1 disables. &gt;1 increases stress volatility.</div>
              </div>
              <div className="field">
                <label className="label" htmlFor="tuneStressShock">
                  Stress shock
                </label>
                <input
                  id="tuneStressShock"
                  className="input"
                  type="number"
                  step="0.001"
                  value={form.tuneStressShock}
                  onChange={(e) => setForm((f) => ({ ...f, tuneStressShock: numFromInput(e.target.value, f.tuneStressShock) }))}
                  placeholder="0"
                />
                <div className="hint">Additive return shock (e.g., -0.01).</div>
              </div>
              <div className="field">
                <label className="label" htmlFor="tuneStressWeight">
                  Stress weight
                </label>
                <input
                  id="tuneStressWeight"
                  className="input"
                  type="number"
                  step="0.01"
                  min={0}
                  value={form.tuneStressWeight}
                  onChange={(e) => setForm((f) => ({ ...f, tuneStressWeight: numFromInput(e.target.value, f.tuneStressWeight) }))}
                  placeholder="0"
                />
                <div className="hint">Penalty weight (0 disables).</div>
              </div>
            </div>
          </div>
          <div className="field">
          <div className="label">Options</div>
            <div className="pillRow">
              <label className="pill">
                <input
                  type="checkbox"
                  checked={form.binanceTestnet}
                  disabled={form.market === "margin"}
                  onChange={(e) =>
                    setForm((f) => ({
                      ...f,
                      binanceTestnet: f.market === "margin" ? false : e.target.checked,
                    }))
                  }
                />
                Testnet (spot/futures)
              </label>
              <label className="pill">
                <input
                  type="checkbox"
                  checked={form.autoRefresh}
                  onChange={(e) => setForm((f) => ({ ...f, autoRefresh: e.target.checked }))}
                />
                Auto-refresh
              </label>
              <label className="pill">
                <input
                  type="checkbox"
                  checked={form.bypassCache}
                  onChange={(e) => setForm((f) => ({ ...f, bypassCache: e.target.checked }))}
                />
                Bypass cache
              </label>
            </div>
            <div className="hint">
              Auto-refresh every{" "}
              <input
                className="input"
                style={{ height: 32, width: 84, padding: "0 10px", margin: "0 8px" }}
                type="number"
                min={5}
                max={600}
                value={form.autoRefreshSec}
                onChange={(e) => setForm((f) => ({ ...f, autoRefreshSec: numFromInput(e.target.value, f.autoRefreshSec) }))}
                disabled={!form.autoRefresh}
                aria-label="Auto-refresh interval in seconds"
              />{" "}
              seconds.{!form.autoRefresh ? " Enable Auto-refresh to use this interval." : ""}{" "}
              {form.bypassCache ? "Bypass cache adds Cache-Control: no-cache." : ""}
            </div>
            <div className="hint" style={{ marginTop: 6 }}>
              Positions open-time cache{" "}
              <input
                className="input"
                style={{ height: 32, width: 84, padding: "0 10px", margin: "0 8px" }}
                type="number"
                min={0}
                max={86_400}
                value={form.positionsOpenTimeCacheSec}
                onChange={(e) =>
                  setForm((f) => ({ ...f, positionsOpenTimeCacheSec: numFromInput(e.target.value, f.positionsOpenTimeCacheSec) }))
                }
                aria-label="Positions open-time cache in seconds"
              />{" "}
              seconds. 0 disables caching.
            </div>
          </div>
        </div>

      </CollapsibleSection>
      ) : null}
            </ConfigPanel>
          ) : null}
          {activeConfigPanel === "config-execution" ? (
          <ConfigPanel
            panelId="config-execution"
            title="Live Bot & Trade"
            subtitle="Arm trading, run bots, and size orders."
            order={configPanelOrderIndex["config-execution"]}
            open={isPanelOpen("config-execution", true)}
            onToggle={handlePanelToggle("config-execution")}
            maximized={isPanelMaximized("config-execution")}
            onToggleMaximize={() => togglePanelMaximize("config-execution")}
            style={configPanelStyle("config-execution")}
            draggable={false}
            {...configPanelHandlers}
          >
      {configPage === "section-livebot" ? (
        <CollapsibleSection
          panelId="section-livebot"
          open={isPanelOpen("section-livebot", true)}
          onToggle={handlePanelToggle("section-livebot")}
          title="Live bot"
          meta="Start, stop, and tune the continuous loop."
        >
          <div className="row" style={{ gridTemplateColumns: "1fr" }}>
            <div className="field">
              <div className="label">Live bot</div>
              <div className="actions" style={{ marginTop: 0 }}>
                <button
                  className="btn btnPrimary"
                  disabled={botStartBlocked}
                  onClick={() => void startLiveBot()}
                  title={
                    firstReason(
                      botStartBlockedReason,
                      form.tradeArmed ? "Trading armed (will send orders)" : "Paper mode (no orders)",
                    ) ?? undefined
                  }
                >
                  {bot.loading || botStarting ? "Starting…" : botAnyRunning ? "Running" : "Start live bot"}
                </button>
                <button
                  className="btn btnDanger"
                  disabled={bot.loading || (!botAnyRunning && !botStarting)}
                  onClick={() => void stopLiveBot()}
                >
                  {botSymbolsActive.length > 1 ? "Stop all" : "Stop bot"}
                </button>
                {botSymbolsActive.length > 1 ? (
                  <button
                    className="btn"
                    disabled={bot.loading || !botSelectedSymbol || botStarting}
                    onClick={() => botSelectedSymbol && stopLiveBot(botSelectedSymbol)}
                  >
                    Stop selected
                  </button>
                ) : null}
                <button className="btn" disabled={bot.loading || Boolean(apiBlockedReason)} onClick={() => refreshBot()} title={apiBlockedReason ?? undefined}>
                  Refresh
                </button>
              </div>
              <div className="row" style={{ marginTop: 10 }}>
                <div className="field" style={{ flex: "1 1 360px" }}>
                  <label className="label" htmlFor="botSymbols">
                    Bot symbols (optional)
                  </label>
                  <input
                    id="botSymbols"
                    className="input"
                    value={form.botSymbols}
                    onChange={(e) => setForm((f) => ({ ...f, botSymbols: e.target.value }))}
                    placeholder="BTCUSDT, ETHUSDT, SOLUSDT"
                  />
                  <div className="hint">
                    Comma-separated list for multi-symbol bots. Leave blank to use the Symbol from Market settings.
                  </div>
                  {botSymbolsFormatError ? (
                    <div className="hint" style={{ color: "rgba(239, 68, 68, 0.9)" }}>
                      {botSymbolsFormatError}
                    </div>
                  ) : null}
                </div>
              </div>
              <div className="hint">
                Continuously ingests new bars, fine-tunes on each bar, and switches position based on the latest signal. Enable “Arm trading” to actually place
                Binance orders; otherwise it runs in paper mode. If “Sweep thresholds” or “Optimize operations” is enabled, the bot re-optimizes after each
                buy/sell operation.
              </div>
              <div className="row" style={{ marginTop: 10 }}>
                <div className="field">
                  <div className="label">Protection orders</div>
                  <div className="pillRow">
                    <label className="pill">
                      <input
                        type="checkbox"
                        checked={form.botProtectionOrders}
                        onChange={(e) => setForm((f) => ({ ...f, botProtectionOrders: e.target.checked }))}
                      />
                      Exchange SL/TP (futures, algo)
                    </label>
                  </div>
                  <div className="hint">
                    Places reduce-only STOP_MARKET / TAKE_PROFIT_MARKET orders on Binance futures when live orders + trading are armed. Uses the Algo Order API
                    when required. Requires stop-loss or take-profit (or vol-mult). Trailing stops remain internal.
                  </div>
                  {!isBinancePlatform ? (
                    <div className="hint" style={{ color: "rgba(245, 158, 11, 0.9)" }}>
                      Live bots are Binance-only. Protection orders are ignored on other platforms.
                    </div>
                  ) : form.market !== "futures" ? (
                    <div className="hint" style={{ color: "rgba(245, 158, 11, 0.9)" }}>
                      Requires Binance futures (market=futures).
                    </div>
                  ) : null}
                  {form.botProtectionOrders && !form.binanceLive ? (
                    <div className="hint" style={{ color: "rgba(245, 158, 11, 0.9)" }}>
                      Enable Live orders in Trade settings to place exchange protection orders.
                    </div>
                  ) : null}
                  {form.botProtectionOrders && botProtectionNeedsStops ? (
                    <div className="hint" style={{ color: "rgba(245, 158, 11, 0.9)" }}>
                      Configure stop-loss / take-profit (or vol-mult) to place protection orders.
                    </div>
                  ) : null}
                </div>
              </div>
              {botStartBlockedReason && !botAnyRunning && !botStarting ? (
                <div className="hint" style={{ color: "rgba(245, 158, 11, 0.9)" }}>
                  Start live bot is disabled: {botStartBlockedReason}
                </div>
              ) : null}
              {bot.error ? <div className="hint" style={{ color: "rgba(239, 68, 68, 0.9)", whiteSpace: "pre-wrap" }}>{bot.error}</div> : null}

              <details className="details" style={{ marginTop: 10 }}>
                <summary>Advanced live bot</summary>
                <div className="row" style={{ marginTop: 10 }}>
                  <div className="field">
                    <label className="label" htmlFor="botPollSeconds">
                      Poll seconds (0 = auto)
                    </label>
                    <input
                      id="botPollSeconds"
                      className="input"
                      type="number"
                      min={0}
                      max={3600}
                      value={form.botPollSeconds}
                      onChange={(e) => setForm((f) => ({ ...f, botPollSeconds: numFromInput(e.target.value, f.botPollSeconds) }))}
                    />
                    <div className="hint">How often the bot checks for a new bar (server-side).</div>
                  </div>
                  <div className="field">
                    <label className="label" htmlFor="botOnlineEpochs">
                      Online epochs
                    </label>
                    <input
                      id="botOnlineEpochs"
                      className="input"
                      type="number"
                      min={0}
                      max={50}
                      value={form.botOnlineEpochs}
                      onChange={(e) => setForm((f) => ({ ...f, botOnlineEpochs: numFromInput(e.target.value, f.botOnlineEpochs) }))}
                    />
                    <div className="hint">0 disables per-bar fine-tuning (faster, less adaptive).</div>
                  </div>
                </div>
                <div className="row" style={{ marginTop: 12 }}>
                  <div className="field">
                    <div className="label">Startup position</div>
                    <div className="pillRow">
                      <label className="pill">
                        <input
                          type="checkbox"
                          checked={true}
                          disabled
                        />
                        Always adopt existing positions
                      </label>
                    </div>
                    <div className="hint">Existing positions are adopted automatically when the live bot starts.</div>
                  </div>
                </div>
                <div className="row" style={{ marginTop: 12 }}>
                  <div className="field">
                    <label className="label" htmlFor="botTrainBars">
                      Train bars (rolling)
                    </label>
                    <input
                      id="botTrainBars"
                      className="input"
                      type="number"
                      min={10}
                      value={form.botTrainBars}
                      onChange={(e) => setForm((f) => ({ ...f, botTrainBars: numFromInput(e.target.value, f.botTrainBars) }))}
                    />
                    <div className="hint">Bars used for online fine-tuning and optimization windows.</div>
                  </div>
                  <div className="field">
                    <label className="label" htmlFor="botMaxPoints">
                      Max points (history)
                    </label>
                    <input
                      id="botMaxPoints"
                      className="input"
                      type="number"
                      min={100}
                      max={100000}
                      value={form.botMaxPoints}
                      onChange={(e) => setForm((f) => ({ ...f, botMaxPoints: numFromInput(e.target.value, f.botMaxPoints) }))}
                    />
                    <div className="hint">Caps in-memory chart/history. Larger uses more memory.</div>
                  </div>
                </div>
                <div className="actions" style={{ marginTop: 10 }}>
                  <button
                    className="btn"
                    type="button"
                    onClick={() =>
                      setForm((f) => ({
                        ...f,
                        botPollSeconds: defaultForm.botPollSeconds,
                        botOnlineEpochs: defaultForm.botOnlineEpochs,
                        botTrainBars: defaultForm.botTrainBars,
                        botMaxPoints: defaultForm.botMaxPoints,
                        botProtectionOrders: defaultForm.botProtectionOrders,
                        botAdoptExistingPosition: defaultForm.botAdoptExistingPosition,
                      }))
                    }
                  >
                    Reset defaults
                  </button>
                  <span className="hint">Changes apply the next time you start the bot.</span>
                </div>
              </details>
            </div>
          </div>
      </CollapsibleSection>
      ) : null}

      {configPage === "section-trade" ? (
        <CollapsibleSection
          panelId="section-trade"
          open={isPanelOpen("section-trade", true)}
          onToggle={handlePanelToggle("section-trade")}
          title="Trade"
          meta="Arm trading, size orders, and run /trade."
        >
          <div className="row">
            <div className="field">
              <div className="label">Trade controls</div>
              <div className="pillRow">
                <label className="pill">
                  <input
                    type="checkbox"
                    checked={form.binanceLive}
                    disabled={form.market === "margin"}
                    onChange={(e) => {
                      setPendingProfileLoad(null);
                      if (!e.target.checked) {
                        setConfirmLive(false);
                        setForm((f) => ({ ...f, binanceLive: false }));
                        return;
                      }
                      setConfirmArm(false);
                      setConfirmLive(true);
                    }}
                  />
                  Live orders
                </label>
                <label className="pill">
                  <input
                    type="checkbox"
                    checked={form.tradeArmed}
                    onChange={(e) => {
                      setPendingProfileLoad(null);
                      if (!e.target.checked) {
                        setConfirmArm(false);
                        setForm((f) => ({ ...f, tradeArmed: false }));
                        return;
                      }
                      setConfirmLive(false);
                      setConfirmArm(true);
                    }}
                  />
                  Arm trading
                </label>
              </div>
              <div className="hint">Trading is disabled by default. Only arm it when you’re ready.</div>
              {form.market === "margin" ? <div className="hint">Live orders are required for margin (forced on).</div> : null}

              {confirmLive ? (
                <>
                  <pre className="code" style={{ borderColor: "rgba(245, 158, 11, 0.35)", marginTop: 10 }}>
                    Enable Live orders? This can place real orders on Binance or Coinbase when you trade (live bot orders remain Binance-only).
                  </pre>
                  <div className="actions" style={{ marginTop: 10 }}>
                    <button
                      className="btn btnPrimary"
                      type="button"
                      onClick={() => {
                        setForm((f) => ({ ...f, binanceLive: true }));
                        setConfirmLive(false);
                        showToast("Live orders enabled");
                      }}
                    >
                      Enable live orders
                    </button>
                    <button className="btn" type="button" onClick={() => setConfirmLive(false)}>
                      Cancel
                    </button>
                  </div>
                </>
              ) : null}

              {confirmArm ? (
                <>
                  <pre className="code" style={{ borderColor: "rgba(245, 158, 11, 0.35)", marginTop: 10 }}>
                    Arm trading? This unlocks calling /trade and allows the live bot to send orders (paper mode when unarmed).
                  </pre>
                  <div className="actions" style={{ marginTop: 10 }}>
                    <button
                      className="btn btnPrimary"
                      type="button"
                      onClick={() => {
                        setForm((f) => ({ ...f, tradeArmed: true }));
                        setConfirmArm(false);
                        showToast("Trading armed");
                      }}
                    >
                      Arm trading
                    </button>
                    <button className="btn" type="button" onClick={() => setConfirmArm(false)}>
                      Cancel
                    </button>
                  </div>
                </>
              ) : null}
            </div>
            <div className="field">
              <div className="label">Order sizing</div>
              <div className="hint" style={orderSizing.conflicts ? { color: "rgba(239, 68, 68, 0.9)" } : undefined}>
                {orderSizing.conflicts ? `Multiple sizing inputs are set (${orderSizing.active.join(", ")}). ` : ""}
                {orderSizing.hint}
              </div>

              {orderSizing.conflicts ? (
                <div className="actions" style={{ marginTop: 10 }}>
                  <button
                    className="btn"
                    type="button"
                    onClick={() =>
                      setForm((f) => {
                        if (orderSizing.effective === "orderQuantity") return { ...f, orderQuote: 0, orderQuoteFraction: 0 };
                        if (orderSizing.effective === "orderQuote") return { ...f, orderQuantity: 0, orderQuoteFraction: 0 };
                        if (orderSizing.effective === "orderQuoteFraction") return { ...f, orderQuantity: 0, orderQuote: 0 };
                        return f;
                      })
                    }
                  >
                    Keep {orderSizing.effective} and clear others
                  </button>
                </div>
              ) : null}

              <div className="row" style={{ gridTemplateColumns: "1fr 1fr", marginTop: 8 }}>
                <div className="field">
                  <label className="label" htmlFor="orderQuote">
                    Order quote (e.g., USDT)
                  </label>
                  <input
                    id="orderQuote"
                    className="input"
                    type="number"
                    min={0}
                    value={form.orderQuote}
                    onChange={(e) =>
                      setForm((f) => {
                        const v = numFromInput(e.target.value, f.orderQuote);
                        return v > 0 ? { ...f, orderQuote: v, orderQuantity: 0, orderQuoteFraction: 0 } : { ...f, orderQuote: v };
                      })
                    }
                    placeholder="20"
                  />
                </div>
                <div className="field">
                  <label className="label" htmlFor="orderQuantity">
                    Order quantity (base units)
                  </label>
                  <input
                    id="orderQuantity"
                    className="input"
                    type="number"
                    min={0}
                    value={form.orderQuantity}
                    onChange={(e) =>
                      setForm((f) => {
                        const v = numFromInput(e.target.value, f.orderQuantity);
                        return v > 0 ? { ...f, orderQuantity: v, orderQuote: 0, orderQuoteFraction: 0 } : { ...f, orderQuantity: v };
                      })
                    }
                    placeholder="0.001"
                  />
                </div>
              </div>

              <div className="row" style={{ gridTemplateColumns: "1fr 1fr", marginTop: 10 }}>
                <div className="field">
                  <label className="label" htmlFor="orderQuoteFraction">
                    Order quote fraction (0 &lt; F ≤ 1; 0 disables)
                  </label>
                  <input
                    id="orderQuoteFraction"
                    className={orderQuoteFractionError ? "input inputError" : "input"}
                    type="number"
                    step="0.01"
                    min={0}
                    max={1}
                    value={form.orderQuoteFraction}
                    onChange={(e) =>
                      setForm((f) => {
                        const v = numFromInput(e.target.value, f.orderQuoteFraction);
                        return v > 0 ? { ...f, orderQuoteFraction: v, orderQuote: 0, orderQuantity: 0 } : { ...f, orderQuoteFraction: v };
                      })
                    }
                    placeholder="0.10 (10%)"
                  />
                  <div className="hint" style={orderQuoteFractionError ? { color: "rgba(239, 68, 68, 0.9)" } : undefined}>
                    {orderQuoteFractionError ?? "Applies to BUYs: uses a fraction of your available quote balance."}
                  </div>
                </div>
                <div className="field">
                  <label className="label" htmlFor="maxOrderQuote">
                    Max quote cap (optional)
                  </label>
                  <input
                    id="maxOrderQuote"
                    className="input"
                    type="number"
                    step="1"
                    min={0}
                    disabled={form.orderQuoteFraction <= 0}
                    value={form.maxOrderQuote}
                    onChange={(e) => setForm((f) => ({ ...f, maxOrderQuote: numFromInput(e.target.value, f.maxOrderQuote) }))}
                    placeholder="0 (no cap)"
                  />
                  <div className="hint">
                    {form.orderQuoteFraction > 0 ? "Optional cap when using orderQuoteFraction." : "Enable orderQuoteFraction to use this cap."}
                  </div>
                </div>
              </div>

              <label className="label" htmlFor="idempotencyKey" style={{ marginTop: 10 }}>
                Idempotency key (optional)
              </label>
              <div className="row" style={{ gridTemplateColumns: "1fr auto auto", marginTop: 8, alignItems: "center" }}>
                <input
                  id="idempotencyKey"
                  className={idempotencyKeyError ? "input inputError" : "input"}
                  value={form.idempotencyKey}
                  onChange={(e) => setForm((f) => ({ ...f, idempotencyKey: e.target.value }))}
                  placeholder="e.g. 1f6a2c7a-…"
                  spellCheck={false}
                  autoCapitalize="none"
                  autoCorrect="off"
                  inputMode="text"
                />
                <button
                  className="btn"
                  type="button"
                  onClick={() => setForm((f) => ({ ...f, idempotencyKey: generateIdempotencyKey() }))}
                >
                  Generate
                </button>
                <button
                  className="btn"
                  type="button"
                  disabled={!form.idempotencyKey.trim()}
                  onClick={() => setForm((f) => ({ ...f, idempotencyKey: "" }))}
                >
                  Clear
                </button>
              </div>
              <div className="hint" style={idempotencyKeyError ? { color: "rgba(239, 68, 68, 0.9)" } : undefined}>
                {idempotencyKeyError
                  ? `${idempotencyKeyError} (not sent to the API).`
                  : "Use for manual /trade retries. Leave blank for the live bot unless you know what you’re doing."}
              </div>
            </div>
          </div>

          <div className="actions" style={{ marginTop: 10 }}>
            <button
              className="btn btnDanger"
              disabled={state.loading || !form.tradeArmed || Boolean(tradeDisabledReason)}
              onClick={() => run("trade")}
              title={tradeDisabledReason ?? (form.binanceLive ? "LIVE order mode enabled" : "Test order mode (default)")}
            >
              {state.loading && state.lastKind === "trade" ? "Trading…" : "Trade (uses latest signal)"}
            </button>
            <button
              className="btn"
              disabled={!state.loading}
              onClick={cancelActiveRequest}
            >
              Cancel
            </button>
          </div>
          {tradeDisabledDetail ? (
            <div className="issueItem" style={{ marginTop: 10 }}>
              <span>Trade disabled: {tradeDisabledDetail.message}</span>
              {tradeDisabledDetail.targetId ? (
                <button className="btnSmall" type="button" onClick={() => scrollToSection(tradeDisabledDetail.targetId)}>
                  Fix
                </button>
              ) : null}
            </div>
          ) : null}

      </CollapsibleSection>
      ) : null}
            </ConfigPanel>
          ) : null}
        </div>

      <p className="footerNote">
          Backend: start with{" "}
          <span style={{ fontFamily: "var(--mono)" }}>
            cd haskell && cabal run -v0 trader-hs -- --serve --port {API_PORT}
          </span>
          .{" "}
          {import.meta.env.DEV ? (
            <>
              The UI uses a same-origin dev proxy (<span style={{ fontFamily: "var(--mono)" }}>/api</span>) to avoid CORS and reduce local attack surface.
            </>
          ) : (
            <>
              When hosting the UI separately (CloudFront/S3), configure <span style={{ fontFamily: "var(--mono)" }}>trader-config.js</span> (apiBaseUrl,
              apiToken, and optional apiFallbackUrl for CORS-enabled failover) and/or route <span style={{ fontFamily: "var(--mono)" }}>/api/*</span> to your backend.
            </>
          )}
        </p>
    </CollapsibleCard>
  );
};
