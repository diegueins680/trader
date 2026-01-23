import React from "react";
import { EQUITY_TIPS, TOP_COMBOS_DISPLAY_MIN, type ComboOrder, type OptimizerRunForm, type OptimizerRunUiState, type TopCombosMeta } from "../app/appHelpers";
import { TUNE_OBJECTIVES } from "../app/constants";
import { comboMarketLabel, type ComboMarketFilter, type ComboMarketValue } from "../app/comboMarket";
import { clamp, fmtDurationMs, fmtTimeMs, methodLabel, numFromInput } from "../app/utils";
import { fmtRatio } from "../lib/format";
import type { Method, OptimizerSource } from "../lib/types";
import { InfoList, InfoPopover } from "./InfoPopover";
import type { OptimizationCombo } from "./TopCombosChart";
import { TopCombosChart } from "./TopCombosChart";

type ComboFilterOptions = {
  symbols: string[];
  markets: ComboMarketValue[];
  intervals: string[];
  methods: Method[];
};

type OptimizerRunExtras = { value: Record<string, unknown> | null; error: string | null };

export type OptimizerCombosPanelProps = {
  apiOk: "unknown" | "ok" | "down" | "auth";
  autoAppliedCombo: { id: number; atMs: number } | null;
  autoAppliedAge: string | null;
  manualOverrideLabels: string[];
  clearManualOverrides: () => void;
  topCombosMeta: TopCombosMeta;
  topCombos: OptimizationCombo[];
  topCombosFiltered: OptimizationCombo[];
  topCombosAll: OptimizationCombo[];
  deferredTopCombos: OptimizationCombo[];
  topCombosLoading: boolean;
  topCombosError: string | null;
  selectedComboId: number | null;
  selectedCombo: OptimizationCombo | null;
  topComboDisplay: OptimizationCombo | null;
  selectedComboStartLabel: string;
  comboStartBlocked: boolean;
  comboStartBlockedReason: string | null;
  comboStartPending: boolean;
  comboMinEquity: number | null;
  comboSymbolFilters: string[];
  comboMarketFilter: ComboMarketFilter;
  comboIntervalFilter: string;
  comboMethodFilter: Method | "all";
  comboOrder: ComboOrder;
  comboHasFilters: boolean;
  comboFilterOptions: ComboFilterOptions;
  comboMinEquityInput: string;
  comboSymbolFilterInput: string;
  topCombosDisplayCount: number;
  topCombosDisplayMax: number;
  setTopCombosDisplayCount: (value: number) => void;
  setComboOrder: (value: ComboOrder) => void;
  setComboMinEquityInput: (value: string) => void;
  setComboSymbolFilterInput: (value: string) => void;
  setComboMarketFilter: (value: ComboMarketFilter) => void;
  setComboIntervalFilter: (value: string) => void;
  setComboMethodFilter: (value: Method | "all") => void;
  refreshTopCombos: () => void;
  handleComboApply: (combo: OptimizationCombo) => void;
  handleComboPreview: (combo: OptimizationCombo) => void;
  handleComboStart: (combo: OptimizationCombo) => void;
  optimizerRunForm: OptimizerRunForm;
  setOptimizerRunForm: React.Dispatch<React.SetStateAction<OptimizerRunForm>>;
  setOptimizerRunDirty: React.Dispatch<React.SetStateAction<boolean>>;
  optimizerRunExtras: OptimizerRunExtras;
  optimizerRunUi: OptimizerRunUiState;
  optimizerRunValidationError: string | null;
  optimizerRunRecordJson: string | null;
  runOptimizer: () => Promise<void>;
  cancelOptimizerRun: () => void;
  syncOptimizerRunSymbolInterval: () => void;
  applyEquityPreset: () => void;
  resetOptimizerRunForm: () => void;
};

export function OptimizerCombosPanel(props: OptimizerCombosPanelProps) {
  const {
    apiOk,
    autoAppliedCombo,
    autoAppliedAge,
    manualOverrideLabels,
    clearManualOverrides,
    topCombosMeta,
    topCombos,
    topCombosFiltered,
    topCombosAll,
    deferredTopCombos,
    topCombosLoading,
    topCombosError,
    selectedComboId,
    selectedCombo,
    topComboDisplay,
    selectedComboStartLabel,
    comboStartBlocked,
    comboStartBlockedReason,
    comboStartPending,
    comboMinEquity,
    comboSymbolFilters,
    comboMarketFilter,
    comboIntervalFilter,
    comboMethodFilter,
    comboOrder,
    comboHasFilters,
    comboFilterOptions,
    comboMinEquityInput,
    comboSymbolFilterInput,
    topCombosDisplayCount,
    topCombosDisplayMax,
    setTopCombosDisplayCount,
    setComboOrder,
    setComboMinEquityInput,
    setComboSymbolFilterInput,
    setComboMarketFilter,
    setComboIntervalFilter,
    setComboMethodFilter,
    refreshTopCombos,
    handleComboApply,
    handleComboPreview,
    handleComboStart,
    optimizerRunForm,
    setOptimizerRunForm,
    setOptimizerRunDirty,
    optimizerRunExtras,
    optimizerRunUi,
    optimizerRunValidationError,
    optimizerRunRecordJson,
    runOptimizer,
    cancelOptimizerRun,
    syncOptimizerRunSymbolInterval,
    applyEquityPreset,
    resetOptimizerRunForm,
  } = props;

  return (
  <div className="row" style={{ gridTemplateColumns: "1fr" }}>
    <div className="field">
      <div className="label">Optimizer combos</div>
      {(() => {
        const updatedAtMs = topCombosMeta.generatedAtMs;
        const updatedLabel = updatedAtMs ? fmtTimeMs(updatedAtMs) : "—";
        const ageLabel = updatedAtMs ? fmtDurationMs(Math.max(0, Date.now() - updatedAtMs)) : null;
        const sourceLabel = "Source: API";
        const payloadSources = topCombosMeta.payloadSources;
        const payloadSource = topCombosMeta.payloadSource;
        const payloadLabel =
          payloadSources && payloadSources.length > 0
            ? ` • payload ${payloadSources.join(" + ")}`
            : payloadSource
              ? ` • payload ${payloadSource}`
              : "";
        const displayCount = topCombos.length;
        const filteredCount = topCombosFiltered.length;
        const totalCount = topCombosMeta.comboCount ?? topCombosAll.length;
        const activeFilters: string[] = [];
        if (comboMinEquity != null) {
          activeFilters.push(`min final equity > ${fmtRatio(comboMinEquity, 4)}`);
        }
        if (comboSymbolFilters.length > 0) {
          activeFilters.push(`symbol ${comboSymbolFilters.join(", ")}`);
        }
        if (comboMarketFilter !== "all") {
          activeFilters.push(`market ${comboMarketLabel(comboMarketFilter)}`);
        }
        if (comboIntervalFilter !== "all") {
          activeFilters.push(`interval ${comboIntervalFilter}`);
        }
        if (comboMethodFilter !== "all") {
          activeFilters.push(`method ${methodLabel(comboMethodFilter)}`);
        }
        const filterLabel = activeFilters.length > 0 ? ` (filters: ${activeFilters.join(", ")})` : "";
        const countLabel =
          activeFilters.length > 0
            ? `Showing ${displayCount} of ${filteredCount} combos${filterLabel}`
            : totalCount > displayCount
              ? `Showing ${displayCount} of ${totalCount} combos`
              : `Showing ${displayCount} combo${displayCount === 1 ? "" : "s"}`;
        const totalLabel =
          activeFilters.length > 0 && totalCount > filteredCount ? ` • ${totalCount} total` : "";
        return (
          <div style={{ marginBottom: 8 }}>
            <div className="hint">
              {sourceLabel}
              {payloadLabel}
            </div>
            <div className="hint">
              Last updated {updatedLabel}
              {ageLabel ? ` (${ageLabel} ago)` : ""}
              {" • "}
              {countLabel}
              {totalLabel}
            </div>
          </div>
        );
      })()}
      <div className="pillRow" style={{ marginBottom: 8 }}>
        {autoAppliedCombo ? (
          <span className="pill">
            Auto-applied #{autoAppliedCombo.id}
            {autoAppliedAge ? ` (${autoAppliedAge} ago)` : ""}
          </span>
        ) : null}
        {manualOverrideLabels.length > 0 ? (
          <>
            <span
              className="pill"
              style={{ color: "rgba(245, 158, 11, 0.9)", borderColor: "rgba(245, 158, 11, 0.35)" }}
            >
              Manual overrides: {manualOverrideLabels.join(", ")}
            </span>
            <button className="btnSmall" type="button" onClick={() => clearManualOverrides()}>
              Unlock overrides
            </button>
          </>
        ) : null}
      </div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 10, alignItems: "center", marginBottom: 8 }}>
        <label className="label" htmlFor="comboDisplayCount">
          Combos to show
        </label>
        <input
          id="comboDisplayCount"
          className="input"
          type="number"
          min={TOP_COMBOS_DISPLAY_MIN}
          max={topCombosDisplayMax}
          step={1}
          value={topCombosDisplayCount}
          onChange={(e) => {
            const rawValue = numFromInput(e.target.value, topCombosDisplayCount);
            const next = clamp(Math.trunc(rawValue), TOP_COMBOS_DISPLAY_MIN, topCombosDisplayMax);
            setTopCombosDisplayCount(next);
          }}
          style={{ width: 120 }}
        />
        <label className="label" htmlFor="comboOrder">
          Order by
        </label>
        <select
          id="comboOrder"
          className="select"
          value={comboOrder}
          onChange={(e) => setComboOrder(e.target.value as ComboOrder)}
          style={{ minWidth: 180 }}
        >
          <option value="annualized-equity">Annualized equity</option>
          <option value="rank">Rank (score/final equity)</option>
          <option value="date-desc">Date (newest)</option>
          <option value="date-asc">Date (oldest)</option>
        </select>
        <label className="label" htmlFor="comboMinEquity">
          Min final equity
        </label>
        <input
          id="comboMinEquity"
          className="input"
          type="number"
          step="0.0001"
          value={comboMinEquityInput}
          onChange={(e) => setComboMinEquityInput(e.target.value)}
          placeholder="e.g. 1.5"
          style={{ width: 140 }}
        />
      </div>
      <div
        className="row"
        style={{ marginBottom: 8, gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))" }}
      >
        <div className="field">
          <label className="label" htmlFor="comboSymbolFilter">
            Symbol filter
          </label>
          <input
            id="comboSymbolFilter"
            className="input"
            value={comboSymbolFilterInput}
            onChange={(e) => setComboSymbolFilterInput(e.target.value)}
            placeholder="BTCUSDT, ETHUSDT"
            spellCheck={false}
            list="comboSymbolFilterList"
          />
          <datalist id="comboSymbolFilterList">
            {comboFilterOptions.symbols.map((symbol) => (
              <option key={symbol} value={symbol} />
            ))}
          </datalist>
        </div>
        <div className="field">
          <label className="label" htmlFor="comboMarketFilter">
            Market
          </label>
          <select
            id="comboMarketFilter"
            className="select"
            value={comboMarketFilter}
            onChange={(e) => setComboMarketFilter(e.target.value as ComboMarketFilter)}
          >
            <option value="all">All markets</option>
            {comboFilterOptions.markets.map((market) => (
              <option key={market} value={market}>
                {comboMarketLabel(market)}
              </option>
            ))}
          </select>
        </div>
        <div className="field">
          <label className="label" htmlFor="comboIntervalFilter">
            Interval
          </label>
          <select
            id="comboIntervalFilter"
            className="select"
            value={comboIntervalFilter}
            onChange={(e) => setComboIntervalFilter(e.target.value)}
          >
            <option value="all">All intervals</option>
            {comboFilterOptions.intervals.map((interval) => (
              <option key={interval} value={interval}>
                {interval}
              </option>
            ))}
          </select>
        </div>
        <div className="field">
          <label className="label" htmlFor="comboMethodFilter">
            Method
          </label>
          <select
            id="comboMethodFilter"
            className="select"
            value={comboMethodFilter}
            onChange={(e) => setComboMethodFilter(e.target.value as Method | "all")}
          >
            <option value="all">All methods</option>
            {comboFilterOptions.methods.map((method) => (
              <option key={method} value={method}>
                {methodLabel(method)} ({method})
              </option>
            ))}
          </select>
        </div>
      </div>
      <div className="actions" style={{ marginBottom: 8 }}>
        <button
          className="btnSmall"
          type="button"
          onClick={() => {
            setComboMinEquityInput("");
            setComboSymbolFilterInput("");
            setComboMarketFilter("all");
            setComboIntervalFilter("all");
            setComboMethodFilter("all");
          }}
          disabled={!comboHasFilters}
        >
          Clear filters
        </button>
      </div>
      {comboHasFilters && topCombosFiltered.length === 0 ? (
        <div className="hint" style={{ marginBottom: 8 }}>
          No combos match the current filters.
        </div>
      ) : null}
      <div className="actions" style={{ marginBottom: 8 }}>
        <button className="btnSmall" type="button" onClick={refreshTopCombos} disabled={topCombosLoading}>
          {topCombosLoading ? "Refreshing…" : "Refresh combos now"}
        </button>
        <button
          className="btnSmall"
          type="button"
          onClick={() => {
            if (topComboDisplay) handleComboApply(topComboDisplay);
          }}
          disabled={!topComboDisplay}
        >
          Apply top combo now
        </button>
        {selectedCombo ? (
          <button
            className="btnSmall btnPrimary"
            type="button"
            onClick={() => handleComboStart(selectedCombo)}
            disabled={comboStartBlocked}
            title={comboStartBlockedReason ?? undefined}
          >
            {comboStartPending ? "Starting…" : selectedComboStartLabel}
          </button>
        ) : null}
      </div>
      {selectedCombo && comboStartBlockedReason ? (
        <div className="hint" style={{ marginBottom: 8, color: "rgba(239, 68, 68, 0.85)" }}>
          Start bot with selected combo is disabled: {comboStartBlockedReason}
        </div>
      ) : null}
      <div className="combosScroll">
        <details className="details">
          <summary>Run optimizer (create combos)</summary>
          <div onChange={() => setOptimizerRunDirty((prev) => (prev ? prev : true))}>
            <div className="row" style={{ marginTop: 10 }}>
          <div className="field">
            <label className="label" htmlFor="optimizerSource">
              Source
            </label>
            <select
              id="optimizerSource"
              className="select"
              value={optimizerRunForm.source}
              onChange={(e) =>
                setOptimizerRunForm((prev) => ({
                  ...prev,
                  source: e.target.value as OptimizerSource,
                }))
              }
            >
              <option value="binance">Binance</option>
              <option value="coinbase">Coinbase</option>
              <option value="kraken">Kraken</option>
              <option value="poloniex">Poloniex</option>
              <option value="csv">CSV</option>
            </select>
            <div className="hint">Choose the source used for optimizer data (CSV requires a path below).</div>
          </div>
          <div className="field">
            <label className="label" htmlFor="optimizerSymbol">
              Symbol
            </label>
            <input
              id="optimizerSymbol"
              className="input"
              value={optimizerRunForm.symbol}
              onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, symbol: e.target.value }))}
              placeholder="BTCUSDT"
              spellCheck={false}
              disabled={optimizerRunForm.source === "csv"}
            />
            <div className="hint">Required for exchange sources; ignored for CSV.</div>
          </div>
        </div>
        {optimizerRunForm.source === "csv" ? (
          <>
            <div className="row" style={{ marginTop: 10 }}>
              <div className="field">
                <label className="label" htmlFor="optimizerDataPath">
                  CSV path
                </label>
                <input
                  id="optimizerDataPath"
                  className="input"
                  value={optimizerRunForm.dataPath}
                  onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, dataPath: e.target.value }))}
                  placeholder="../data/sample_prices.csv"
                  spellCheck={false}
                />
                <div className="hint">Path is resolved on the API host.</div>
              </div>
              <div className="field">
                <label className="label" htmlFor="optimizerPriceColumn">
                  Price column
                </label>
                <input
                  id="optimizerPriceColumn"
                  className="input"
                  value={optimizerRunForm.priceColumn}
                  onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, priceColumn: e.target.value }))}
                  placeholder="close"
                  spellCheck={false}
                />
                <div className="hint">Defaults to close when omitted.</div>
              </div>
            </div>
            <div className="row" style={{ marginTop: 10, gridTemplateColumns: "1fr 1fr" }}>
              <div className="field">
                <label className="label" htmlFor="optimizerHighColumn">
                  High column (optional)
                </label>
                <input
                  id="optimizerHighColumn"
                  className="input"
                  value={optimizerRunForm.highColumn}
                  onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, highColumn: e.target.value }))}
                  placeholder="high"
                  spellCheck={false}
                />
              </div>
              <div className="field">
                <label className="label" htmlFor="optimizerLowColumn">
                  Low column (optional)
                </label>
                <input
                  id="optimizerLowColumn"
                  className="input"
                  value={optimizerRunForm.lowColumn}
                  onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, lowColumn: e.target.value }))}
                  placeholder="low"
                  spellCheck={false}
                />
              </div>
            </div>
          </>
        ) : null}
        <div className="row" style={{ marginTop: 10 }}>
          <div className="field">
            <div className="labelRow">
              <label className="label" htmlFor="optimizerIntervals">
                Intervals
              </label>
              <InfoPopover label="Equity tip: intervals">
                <InfoList items={EQUITY_TIPS.intervals} />
              </InfoPopover>
            </div>
            <input
              id="optimizerIntervals"
              className="input"
              value={optimizerRunForm.intervals}
              onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, intervals: e.target.value }))}
              placeholder="1h,2h,4h,6h,12h,1d"
              spellCheck={false}
            />
            <div className="hint">Comma-separated list; leave blank for API defaults.</div>
          </div>
          <div className="field">
            <label className="label" htmlFor="optimizerLookbackWindow">
              Lookback window
            </label>
            <input
              id="optimizerLookbackWindow"
              className="input"
              value={optimizerRunForm.lookbackWindow}
              onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, lookbackWindow: e.target.value }))}
              placeholder="7d"
              spellCheck={false}
            />
            <div className="hint">Duration string like 48h, 7d, 30d.</div>
          </div>
        </div>
        <div className="row" style={{ marginTop: 10, gridTemplateColumns: "1.2fr 0.8fr 0.8fr" }}>
          <div className="field">
            <label className="label" htmlFor="optimizerBarsMin">
              Bars range
            </label>
            <div className="row" style={{ gridTemplateColumns: "1fr 1fr" }}>
              <input
                id="optimizerBarsMin"
                className="input"
                type="number"
                value={optimizerRunForm.barsMin}
                onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, barsMin: e.target.value }))}
                placeholder="min"
              />
              <input
                aria-label="Bars max"
                className="input"
                type="number"
                value={optimizerRunForm.barsMax}
                onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, barsMax: e.target.value }))}
                placeholder="max"
              />
            </div>
            <div className="hint">0 or blank lets the optimizer choose.</div>
          </div>
          <div className="field">
            <label className="label" htmlFor="optimizerBarsAutoProb">
              Bars auto prob
            </label>
            <input
              id="optimizerBarsAutoProb"
              className="input"
              type="number"
              step="0.01"
              min={0}
              max={1}
              value={optimizerRunForm.barsAutoProb}
              onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, barsAutoProb: e.target.value }))}
              placeholder="0.25"
            />
            <div className="hint">Chance to use bars=0 (exchange default).</div>
          </div>
          <div className="field">
            <label className="label" htmlFor="optimizerBarsDistribution">
              Bars distribution
            </label>
            <select
              id="optimizerBarsDistribution"
              className="select"
              value={optimizerRunForm.barsDistribution}
              onChange={(e) =>
                setOptimizerRunForm((prev) => ({
                  ...prev,
                  barsDistribution: e.target.value as OptimizerRunForm["barsDistribution"],
                }))
              }
            >
              <option value="">Default (uniform)</option>
              <option value="uniform">uniform</option>
              <option value="log">log</option>
            </select>
          </div>
        </div>
        <div className="row" style={{ marginTop: 10, gridTemplateColumns: "1fr 1fr 1fr" }}>
          <div className="field">
            <div className="labelRow">
              <label className="label" htmlFor="optimizerTrials">
                Trials
              </label>
              <InfoPopover label="Equity tip: trials and timeout">
                <InfoList items={EQUITY_TIPS.trials} />
              </InfoPopover>
            </div>
            <input
              id="optimizerTrials"
              className="input"
              type="number"
              min={1}
              value={optimizerRunForm.trials}
              onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, trials: e.target.value }))}
              placeholder="50"
            />
          </div>
          <div className="field">
            <div className="labelRow">
              <label className="label" htmlFor="optimizerTimeoutSec">
                Timeout (sec)
              </label>
              <InfoPopover label="Equity tip: trials and timeout">
                <InfoList items={EQUITY_TIPS.trials} />
              </InfoPopover>
            </div>
            <input
              id="optimizerTimeoutSec"
              className="input"
              type="number"
              min={1}
              step="1"
              value={optimizerRunForm.timeoutSec}
              onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, timeoutSec: e.target.value }))}
              placeholder="60"
            />
          </div>
          <div className="field">
            <label className="label" htmlFor="optimizerSeed">
              Seed
            </label>
            <input
              id="optimizerSeed"
              className="input"
              type="number"
              min={0}
              step="1"
              value={optimizerRunForm.seed}
              onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, seed: e.target.value }))}
              placeholder="42"
            />
          </div>
        </div>
        <div className="row" style={{ marginTop: 10, gridTemplateColumns: "1fr 1fr" }}>
          <div className="field">
            <div className="labelRow">
              <label className="label" htmlFor="optimizerObjective">
                Objective
              </label>
              <InfoPopover label="Equity tip: objective">
                <InfoList items={EQUITY_TIPS.objective} />
              </InfoPopover>
            </div>
            <select
              id="optimizerObjective"
              className="select"
              value={optimizerRunForm.objective}
              onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, objective: e.target.value }))}
            >
              <option value="">Default</option>
              {TUNE_OBJECTIVES.map((o) => (
                <option key={o} value={o}>
                  {o}
                </option>
              ))}
            </select>
            <div className="hint">Controls which combos survive.</div>
          </div>
          <div className="field">
            <div className="labelRow">
              <label className="label" htmlFor="optimizerTuneObjective">
                Tune objective
              </label>
              <InfoPopover label="Equity tip: objective">
                <InfoList items={EQUITY_TIPS.objective} />
              </InfoPopover>
            </div>
            <select
              id="optimizerTuneObjective"
              className="select"
              value={optimizerRunForm.tuneObjective}
              onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, tuneObjective: e.target.value }))}
            >
              <option value="">Default</option>
              {TUNE_OBJECTIVES.map((o) => (
                <option key={o} value={o}>
                  {o}
                </option>
              ))}
            </select>
            <div className="hint">Used during fit/tune scoring.</div>
          </div>
        </div>
        <div className="row" style={{ marginTop: 10, gridTemplateColumns: "1fr 1fr" }}>
          <div className="field">
            <div className="labelRow">
              <label className="label" htmlFor="optimizerBacktestRatio">
                Backtest ratio
              </label>
              <InfoPopover label="Equity tip: backtest and tune ratios">
                <InfoList items={EQUITY_TIPS.ratios} />
              </InfoPopover>
            </div>
            <input
              id="optimizerBacktestRatio"
              className="input"
              type="number"
              step="0.01"
              min={0}
              max={0.99}
              value={optimizerRunForm.backtestRatio}
              onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, backtestRatio: e.target.value }))}
              placeholder="0.2"
            />
          </div>
          <div className="field">
            <div className="labelRow">
              <label className="label" htmlFor="optimizerTuneRatio">
                Tune ratio
              </label>
              <InfoPopover label="Equity tip: backtest and tune ratios">
                <InfoList items={EQUITY_TIPS.ratios} />
              </InfoPopover>
            </div>
            <input
              id="optimizerTuneRatio"
              className="input"
              type="number"
              step="0.01"
              min={0}
              max={0.99}
              value={optimizerRunForm.tuneRatio}
              onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, tuneRatio: e.target.value }))}
              placeholder="0.25"
            />
          </div>
        </div>
        <div className="row" style={{ marginTop: 10, gridTemplateColumns: "1fr 1fr" }}>
          <div className="field">
            <div className="labelRow">
              <label className="label" htmlFor="optimizerPenaltyMaxDrawdown">
                DD penalty
              </label>
              <InfoPopover label="Equity tip: penalties">
                <InfoList items={EQUITY_TIPS.penalties} />
              </InfoPopover>
            </div>
            <input
              id="optimizerPenaltyMaxDrawdown"
              className="input"
              type="number"
              step="0.1"
              min={0}
              value={optimizerRunForm.penaltyMaxDrawdown}
              onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, penaltyMaxDrawdown: e.target.value }))}
              placeholder="1.5"
            />
          </div>
          <div className="field">
            <div className="labelRow">
              <label className="label" htmlFor="optimizerPenaltyTurnover">
                Turnover penalty
              </label>
              <InfoPopover label="Equity tip: penalties">
                <InfoList items={EQUITY_TIPS.penalties} />
              </InfoPopover>
            </div>
            <input
              id="optimizerPenaltyTurnover"
              className="input"
              type="number"
              step="0.1"
              min={0}
              value={optimizerRunForm.penaltyTurnover}
              onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, penaltyTurnover: e.target.value }))}
              placeholder="0.2"
            />
          </div>
        </div>
        <div className="row" style={{ marginTop: 10, gridTemplateColumns: "1fr 1fr" }}>
          <div className="field">
            <label className="label" htmlFor="optimizerSlippageMax">
              Slippage max
            </label>
            <input
              id="optimizerSlippageMax"
              className="input"
              type="number"
              step="0.0001"
              min={0}
              value={optimizerRunForm.slippageMax}
              onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, slippageMax: e.target.value }))}
              placeholder="0.0005"
            />
          </div>
          <div className="field">
            <label className="label" htmlFor="optimizerSpreadMax">
              Spread max
            </label>
            <input
              id="optimizerSpreadMax"
              className="input"
              type="number"
              step="0.0001"
              min={0}
              value={optimizerRunForm.spreadMax}
              onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, spreadMax: e.target.value }))}
              placeholder="0.0005"
            />
          </div>
        </div>
        <div className="actions" style={{ marginTop: 12 }}>
          <button
            className="btn btnPrimary"
            type="button"
            onClick={() => void runOptimizer()}
            disabled={optimizerRunUi.loading || Boolean(optimizerRunValidationError) || apiOk === "down" || apiOk === "auth"}
            title={optimizerRunValidationError ?? undefined}
          >
            {optimizerRunUi.loading ? "Running…" : "Run optimizer"}
          </button>
          <button className="btn" type="button" onClick={cancelOptimizerRun} disabled={!optimizerRunUi.loading}>
            Cancel
          </button>
          <button className="btn" type="button" onClick={syncOptimizerRunSymbolInterval}>
            Use current symbol/interval
          </button>
          <button className="btn" type="button" onClick={applyEquityPreset}>
            Preset: Equity focus
          </button>
          <InfoPopover label="Equity options" align="left">
            <InfoList items={EQUITY_TIPS.preset} />
          </InfoPopover>
          <button className="btn" type="button" onClick={resetOptimizerRunForm}>
            Reset defaults
          </button>
        </div>
        <div className="hint" style={{ marginTop: 8 }}>
          Runs <code>/optimizer/run</code> to generate new combos and refreshes the list above. For annualized equity, keep objective/tune objective on
          <code>annualized-equity</code> and increase trials/timeout.
        </div>
        {optimizerRunValidationError ? (
          <div className="hint" style={{ marginTop: 8, color: "rgba(239, 68, 68, 0.85)" }}>
            {optimizerRunValidationError}
          </div>
        ) : null}
        <details className="details" style={{ marginTop: 12 }}>
          <summary>Sampling + model ranges</summary>
          <div className="row" style={{ marginTop: 10, gridTemplateColumns: "1fr 1fr" }}>
            <div className="field">
              <label className="label" htmlFor="optimizerPlatforms">
                Platforms (optional)
              </label>
              <input
                id="optimizerPlatforms"
                className="input"
                value={optimizerRunForm.platforms}
                onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, platforms: e.target.value }))}
                placeholder="binance,coinbase"
                spellCheck={false}
                disabled={optimizerRunForm.source === "csv"}
              />
              <div className="hint">Overrides the source platform for multi-exchange runs.</div>
            </div>
            <div className="field">
              <label className="label" htmlFor="optimizerNormalizations">
                Normalizations
              </label>
              <input
                id="optimizerNormalizations"
                className="input"
                value={optimizerRunForm.normalizations}
                onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, normalizations: e.target.value }))}
                placeholder="none,minmax,standard,log"
                spellCheck={false}
              />
              <div className="hint">Comma-separated list for LSTM runs.</div>
            </div>
          </div>
          <div className="row" style={{ marginTop: 10, gridTemplateColumns: "1fr 1fr" }}>
            <div className="field">
              <label className="label" htmlFor="optimizerEpochsMin">
                Epochs range
              </label>
              <div className="row" style={{ gridTemplateColumns: "1fr 1fr" }}>
                <input
                  id="optimizerEpochsMin"
                  className="input"
                  type="number"
                  min={0}
                  value={optimizerRunForm.epochsMin}
                  onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, epochsMin: e.target.value }))}
                  placeholder="0"
                />
                <input
                  aria-label="Epochs max"
                  className="input"
                  type="number"
                  min={0}
                  value={optimizerRunForm.epochsMax}
                  onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, epochsMax: e.target.value }))}
                  placeholder="10"
                />
              </div>
            </div>
            <div className="field">
              <label className="label" htmlFor="optimizerHiddenMin">
                Hidden size range
              </label>
              <div className="row" style={{ gridTemplateColumns: "1fr 1fr" }}>
                <input
                  id="optimizerHiddenMin"
                  className="input"
                  type="number"
                  min={1}
                  value={optimizerRunForm.hiddenSizeMin}
                  onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, hiddenSizeMin: e.target.value }))}
                  placeholder="8"
                />
                <input
                  aria-label="Hidden size max"
                  className="input"
                  type="number"
                  min={1}
                  value={optimizerRunForm.hiddenSizeMax}
                  onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, hiddenSizeMax: e.target.value }))}
                  placeholder="64"
                />
              </div>
            </div>
          </div>
          <div className="row" style={{ marginTop: 10, gridTemplateColumns: "1fr 1fr" }}>
            <div className="field">
              <label className="label" htmlFor="optimizerLrMin">
                Learning rate range
              </label>
              <div className="row" style={{ gridTemplateColumns: "1fr 1fr" }}>
                <input
                  id="optimizerLrMin"
                  className="input"
                  type="number"
                  step="0.0001"
                  min={0}
                  value={optimizerRunForm.lrMin}
                  onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, lrMin: e.target.value }))}
                  placeholder="0.0001"
                />
                <input
                  aria-label="Learning rate max"
                  className="input"
                  type="number"
                  step="0.0001"
                  min={0}
                  value={optimizerRunForm.lrMax}
                  onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, lrMax: e.target.value }))}
                  placeholder="0.01"
                />
              </div>
            </div>
            <div className="field">
              <label className="label" htmlFor="optimizerPatienceMax">
                Patience max
              </label>
              <input
                id="optimizerPatienceMax"
                className="input"
                type="number"
                min={0}
                value={optimizerRunForm.patienceMax}
                onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, patienceMax: e.target.value }))}
                placeholder="20"
              />
            </div>
          </div>
          <div className="row" style={{ marginTop: 10, gridTemplateColumns: "1fr 1fr" }}>
            <div className="field">
              <label className="label" htmlFor="optimizerGradClipMin">
                Grad clip range
              </label>
              <div className="row" style={{ gridTemplateColumns: "1fr 1fr" }}>
                <input
                  id="optimizerGradClipMin"
                  className="input"
                  type="number"
                  step="0.0001"
                  min={0}
                  value={optimizerRunForm.gradClipMin}
                  onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, gradClipMin: e.target.value }))}
                  placeholder="0.001"
                />
                <input
                  aria-label="Grad clip max"
                  className="input"
                  type="number"
                  step="0.0001"
                  min={0}
                  value={optimizerRunForm.gradClipMax}
                  onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, gradClipMax: e.target.value }))}
                  placeholder="1.0"
                />
              </div>
            </div>
            <div className="field">
              <label className="label" htmlFor="optimizerDisableGradClipProb">
                Disable grad clip prob
              </label>
              <input
                id="optimizerDisableGradClipProb"
                className="input"
                type="number"
                step="0.01"
                min={0}
                max={1}
                value={optimizerRunForm.pDisableGradClip}
                onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, pDisableGradClip: e.target.value }))}
                placeholder="0.7"
              />
            </div>
          </div>
          <div className="row" style={{ marginTop: 10, gridTemplateColumns: "1fr 1fr 1fr" }}>
            <div className="field">
              <label className="label" htmlFor="optimizerSeedTrials">
                Seed trials
              </label>
              <input
                id="optimizerSeedTrials"
                className="input"
                type="number"
                min={0}
                value={optimizerRunForm.seedTrials}
                onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, seedTrials: e.target.value }))}
                placeholder="0"
              />
            </div>
            <div className="field">
              <label className="label" htmlFor="optimizerSeedRatio">
                Seed ratio
              </label>
              <input
                id="optimizerSeedRatio"
                className="input"
                type="number"
                min={0}
                max={1}
                step="0.01"
                value={optimizerRunForm.seedRatio}
                onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, seedRatio: e.target.value }))}
                placeholder="0.0"
              />
            </div>
            <div className="field">
              <label className="label" htmlFor="optimizerSurvivorFraction">
                Survivor fraction
              </label>
              <input
                id="optimizerSurvivorFraction"
                className="input"
                type="number"
                min={0}
                max={1}
                step="0.01"
                value={optimizerRunForm.survivorFraction}
                onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, survivorFraction: e.target.value }))}
                placeholder="0.5"
              />
            </div>
          </div>
          <div className="row" style={{ marginTop: 10, gridTemplateColumns: "1fr 1fr 1fr" }}>
            <div className="field">
              <label className="label" htmlFor="optimizerPerturbScaleDouble">
                Perturb scale (float)
              </label>
              <input
                id="optimizerPerturbScaleDouble"
                className="input"
                type="number"
                step="0.01"
                min={0}
                value={optimizerRunForm.perturbScaleDouble}
                onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, perturbScaleDouble: e.target.value }))}
                placeholder="0.1"
              />
            </div>
            <div className="field">
              <label className="label" htmlFor="optimizerPerturbScaleInt">
                Perturb scale (int)
              </label>
              <input
                id="optimizerPerturbScaleInt"
                className="input"
                type="number"
                step="1"
                min={0}
                value={optimizerRunForm.perturbScaleInt}
                onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, perturbScaleInt: e.target.value }))}
                placeholder="2"
              />
            </div>
            <div className="field">
              <label className="label" htmlFor="optimizerEarlyStop">
                Early stop (no improve)
              </label>
              <input
                id="optimizerEarlyStop"
                className="input"
                type="number"
                step="1"
                min={0}
                value={optimizerRunForm.earlyStopNoImprove}
                onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, earlyStopNoImprove: e.target.value }))}
                placeholder="0"
              />
            </div>
          </div>
          <div className="pillRow" style={{ marginTop: 10 }}>
            <label className="pill">
              <input
                type="checkbox"
                checked={optimizerRunForm.disableLstmPersistence}
                onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, disableLstmPersistence: e.target.checked }))}
              />
              Disable LSTM persistence
            </label>
            <label className="pill">
              <input
                type="checkbox"
                checked={optimizerRunForm.noSweepThreshold}
                onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, noSweepThreshold: e.target.checked }))}
              />
              No threshold sweep
            </label>
          </div>
        </details>
        <details className="details" style={{ marginTop: 12 }}>
          <summary>Quality filters + constraints</summary>
          <div className="row" style={{ marginTop: 10, gridTemplateColumns: "1fr 1fr 1fr" }}>
            <div className="field">
              <label className="label" htmlFor="optimizerMinRoundTrips">
                Min round trips
              </label>
              <input
                id="optimizerMinRoundTrips"
                className="input"
                type="number"
                min={0}
                value={optimizerRunForm.minRoundTrips}
                onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, minRoundTrips: e.target.value }))}
                placeholder="0"
              />
            </div>
            <div className="field">
              <label className="label" htmlFor="optimizerMinWinRate">
                Min win rate
              </label>
              <input
                id="optimizerMinWinRate"
                className="input"
                type="number"
                step="0.01"
                min={0}
                max={1}
                value={optimizerRunForm.minWinRate}
                onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, minWinRate: e.target.value }))}
                placeholder="0.0"
              />
            </div>
            <div className="field">
              <label className="label" htmlFor="optimizerMinSharpe">
                Min Sharpe
              </label>
              <input
                id="optimizerMinSharpe"
                className="input"
                type="number"
                step="0.1"
                min={0}
                value={optimizerRunForm.minSharpe}
                onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, minSharpe: e.target.value }))}
                placeholder="0.0"
              />
            </div>
          </div>
          <div className="row" style={{ marginTop: 10, gridTemplateColumns: "1fr 1fr 1fr" }}>
            <div className="field">
              <label className="label" htmlFor="optimizerMinAnnualizedReturn">
                Min annualized return
              </label>
              <input
                id="optimizerMinAnnualizedReturn"
                className="input"
                type="number"
                step="0.01"
                min={0}
                value={optimizerRunForm.minAnnualizedReturn}
                onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, minAnnualizedReturn: e.target.value }))}
                placeholder="0.0"
              />
            </div>
            <div className="field">
              <label className="label" htmlFor="optimizerMinCalmar">
                Min Calmar
              </label>
              <input
                id="optimizerMinCalmar"
                className="input"
                type="number"
                step="0.1"
                min={0}
                value={optimizerRunForm.minCalmar}
                onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, minCalmar: e.target.value }))}
                placeholder="0.0"
              />
            </div>
            <div className="field">
              <label className="label" htmlFor="optimizerMinProfitFactor">
                Min profit factor
              </label>
              <input
                id="optimizerMinProfitFactor"
                className="input"
                type="number"
                step="0.1"
                min={0}
                value={optimizerRunForm.minProfitFactor}
                onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, minProfitFactor: e.target.value }))}
                placeholder="0.0"
              />
            </div>
          </div>
          <div className="row" style={{ marginTop: 10, gridTemplateColumns: "1fr 1fr 1fr" }}>
            <div className="field">
              <label className="label" htmlFor="optimizerMaxTurnover">
                Max turnover
              </label>
              <input
                id="optimizerMaxTurnover"
                className="input"
                type="number"
                step="0.01"
                min={0}
                value={optimizerRunForm.maxTurnover}
                onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, maxTurnover: e.target.value }))}
                placeholder="0.0"
              />
            </div>
            <div className="field">
              <label className="label" htmlFor="optimizerMinExposure">
                Min exposure
              </label>
              <input
                id="optimizerMinExposure"
                className="input"
                type="number"
                step="0.01"
                min={0}
                max={1}
                value={optimizerRunForm.minExposure}
                onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, minExposure: e.target.value }))}
                placeholder="0.0"
              />
            </div>
            <div className="field">
              <label className="label" htmlFor="optimizerMinWfSharpeMean">
                Min WF Sharpe mean
              </label>
              <input
                id="optimizerMinWfSharpeMean"
                className="input"
                type="number"
                step="0.1"
                min={0}
                value={optimizerRunForm.minWalkForwardSharpeMean}
                onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, minWalkForwardSharpeMean: e.target.value }))}
                placeholder="0.0"
              />
            </div>
          </div>
          <div className="row" style={{ marginTop: 10, gridTemplateColumns: "1fr 1fr 1fr" }}>
            <div className="field">
              <label className="label" htmlFor="optimizerMaxWfSharpeStd">
                Max WF Sharpe std
              </label>
              <input
                id="optimizerMaxWfSharpeStd"
                className="input"
                type="number"
                step="0.1"
                min={0}
                value={optimizerRunForm.maxWalkForwardSharpeStd}
                onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, maxWalkForwardSharpeStd: e.target.value }))}
                placeholder="0.0"
              />
            </div>
            <div className="field">
              <label className="label" htmlFor="optimizerWalkForwardMin">
                Walk-forward folds min
              </label>
              <input
                id="optimizerWalkForwardMin"
                className="input"
                type="number"
                min={1}
                value={optimizerRunForm.walkForwardFoldsMin}
                onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, walkForwardFoldsMin: e.target.value }))}
                placeholder="7"
              />
            </div>
            <div className="field">
              <label className="label" htmlFor="optimizerWalkForwardMax">
                Walk-forward folds max
              </label>
              <input
                id="optimizerWalkForwardMax"
                className="input"
                type="number"
                min={1}
                value={optimizerRunForm.walkForwardFoldsMax}
                onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, walkForwardFoldsMax: e.target.value }))}
                placeholder="7"
              />
            </div>
          </div>
          <div className="row" style={{ marginTop: 10, gridTemplateColumns: "1fr 1fr" }}>
            <div className="field">
              <label className="label" htmlFor="optimizerWalkForwardEmbargoMin">
                Walk-forward embargo min
              </label>
              <input
                id="optimizerWalkForwardEmbargoMin"
                className="input"
                type="number"
                min={0}
                value={optimizerRunForm.walkForwardEmbargoBarsMin}
                onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, walkForwardEmbargoBarsMin: e.target.value }))}
                placeholder="0"
              />
            </div>
            <div className="field">
              <label className="label" htmlFor="optimizerWalkForwardEmbargoMax">
                Walk-forward embargo max
              </label>
              <input
                id="optimizerWalkForwardEmbargoMax"
                className="input"
                type="number"
                min={0}
                value={optimizerRunForm.walkForwardEmbargoBarsMax}
                onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, walkForwardEmbargoBarsMax: e.target.value }))}
                placeholder="0"
              />
            </div>
          </div>
          <div className="row" style={{ marginTop: 10, gridTemplateColumns: "1fr 1fr" }}>
            <div className="field">
              <label className="label" htmlFor="optimizerRebalanceCostMultMin">
                Rebalance cost mult min
              </label>
              <input
                id="optimizerRebalanceCostMultMin"
                className="input"
                type="number"
                step="0.1"
                min={0}
                value={optimizerRunForm.rebalanceCostMultMin}
                onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, rebalanceCostMultMin: e.target.value }))}
                placeholder="0"
              />
            </div>
            <div className="field">
              <label className="label" htmlFor="optimizerRebalanceCostMultMax">
                Rebalance cost mult max
              </label>
              <input
                id="optimizerRebalanceCostMultMax"
                className="input"
                type="number"
                step="0.1"
                min={0}
                value={optimizerRunForm.rebalanceCostMultMax}
                onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, rebalanceCostMultMax: e.target.value }))}
                placeholder="0"
              />
            </div>
          </div>
          <div className="row" style={{ marginTop: 10, gridTemplateColumns: "1fr 1fr" }}>
            <div className="field">
              <label className="label" htmlFor="optimizerMinEdgeMin">
                Min edge range
              </label>
              <div className="row" style={{ gridTemplateColumns: "1fr 1fr" }}>
                <input
                  id="optimizerMinEdgeMin"
                  className="input"
                  type="number"
                  step="0.0001"
                  min={0}
                  value={optimizerRunForm.minEdgeMin}
                  onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, minEdgeMin: e.target.value }))}
                  placeholder="min"
                />
                <input
                  aria-label="Min edge max"
                  className="input"
                  type="number"
                  step="0.0001"
                  min={0}
                  value={optimizerRunForm.minEdgeMax}
                  onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, minEdgeMax: e.target.value }))}
                  placeholder="max"
                />
              </div>
            </div>
            <div className="field">
              <label className="label" htmlFor="optimizerMinSnrMin">
                Min SNR range
              </label>
              <div className="row" style={{ gridTemplateColumns: "1fr 1fr" }}>
                <input
                  id="optimizerMinSnrMin"
                  className="input"
                  type="number"
                  step="0.0001"
                  min={0}
                  value={optimizerRunForm.minSignalToNoiseMin}
                  onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, minSignalToNoiseMin: e.target.value }))}
                  placeholder="min"
                />
                <input
                  aria-label="Min SNR max"
                  className="input"
                  type="number"
                  step="0.0001"
                  min={0}
                  value={optimizerRunForm.minSignalToNoiseMax}
                  onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, minSignalToNoiseMax: e.target.value }))}
                  placeholder="max"
                />
              </div>
            </div>
          </div>
          <div className="row" style={{ marginTop: 10, gridTemplateColumns: "1fr 1fr" }}>
            <div className="field">
              <label className="label" htmlFor="optimizerEdgeBufferMin">
                Edge buffer range
              </label>
              <div className="row" style={{ gridTemplateColumns: "1fr 1fr" }}>
                <input
                  id="optimizerEdgeBufferMin"
                  className="input"
                  type="number"
                  step="0.0001"
                  min={0}
                  value={optimizerRunForm.edgeBufferMin}
                  onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, edgeBufferMin: e.target.value }))}
                  placeholder="min"
                />
                <input
                  aria-label="Edge buffer max"
                  className="input"
                  type="number"
                  step="0.0001"
                  min={0}
                  value={optimizerRunForm.edgeBufferMax}
                  onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, edgeBufferMax: e.target.value }))}
                  placeholder="max"
                />
              </div>
            </div>
            <div className="field">
              <label className="label" htmlFor="optimizerTrendLookbackMin">
                Trend lookback range
              </label>
              <div className="row" style={{ gridTemplateColumns: "1fr 1fr" }}>
                <input
                  id="optimizerTrendLookbackMin"
                  className="input"
                  type="number"
                  min={0}
                  value={optimizerRunForm.trendLookbackMin}
                  onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, trendLookbackMin: e.target.value }))}
                  placeholder="min"
                />
                <input
                  aria-label="Trend lookback max"
                  className="input"
                  type="number"
                  min={0}
                  value={optimizerRunForm.trendLookbackMax}
                  onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, trendLookbackMax: e.target.value }))}
                  placeholder="max"
                />
              </div>
            </div>
          </div>
          <div className="row" style={{ marginTop: 10 }}>
            <div className="field">
              <label className="label" htmlFor="optimizerCostAwareEdge">
                Cost-aware edge prob
              </label>
              <input
                id="optimizerCostAwareEdge"
                className="input"
                type="number"
                step="0.01"
                min={0}
                max={1}
                value={optimizerRunForm.pCostAwareEdge}
                onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, pCostAwareEdge: e.target.value }))}
                placeholder="0.0"
              />
              <div className="hint">Probability of enabling cost-aware edge (0 disables).</div>
            </div>
          </div>
          <div className="row" style={{ marginTop: 10, gridTemplateColumns: "1fr 1fr 1fr" }}>
            <div className="field">
              <label className="label" htmlFor="optimizerMinHoldBarsMin">
                Min hold bars range
              </label>
              <div className="row" style={{ gridTemplateColumns: "1fr 1fr" }}>
                <input
                  id="optimizerMinHoldBarsMin"
                  className="input"
                  type="number"
                  min={0}
                  value={optimizerRunForm.minHoldBarsMin}
                  onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, minHoldBarsMin: e.target.value }))}
                  placeholder="min"
                />
                <input
                  aria-label="Min hold bars max"
                  className="input"
                  type="number"
                  min={0}
                  value={optimizerRunForm.minHoldBarsMax}
                  onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, minHoldBarsMax: e.target.value }))}
                  placeholder="max"
                />
              </div>
            </div>
            <div className="field">
              <label className="label" htmlFor="optimizerCooldownBarsMin">
                Cooldown bars range
              </label>
              <div className="row" style={{ gridTemplateColumns: "1fr 1fr" }}>
                <input
                  id="optimizerCooldownBarsMin"
                  className="input"
                  type="number"
                  min={0}
                  value={optimizerRunForm.cooldownBarsMin}
                  onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, cooldownBarsMin: e.target.value }))}
                  placeholder="min"
                />
                <input
                  aria-label="Cooldown bars max"
                  className="input"
                  type="number"
                  min={0}
                  value={optimizerRunForm.cooldownBarsMax}
                  onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, cooldownBarsMax: e.target.value }))}
                  placeholder="max"
                />
              </div>
            </div>
            <div className="field">
              <label className="label" htmlFor="optimizerMaxHoldBarsMin">
                Max hold bars range
              </label>
              <div className="row" style={{ gridTemplateColumns: "1fr 1fr" }}>
                <input
                  id="optimizerMaxHoldBarsMin"
                  className="input"
                  type="number"
                  min={0}
                  value={optimizerRunForm.maxHoldBarsMin}
                  onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, maxHoldBarsMin: e.target.value }))}
                  placeholder="min"
                />
                <input
                  aria-label="Max hold bars max"
                  className="input"
                  type="number"
                  min={0}
                  value={optimizerRunForm.maxHoldBarsMax}
                  onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, maxHoldBarsMax: e.target.value }))}
                  placeholder="max"
                />
              </div>
            </div>
          </div>
          <div className="row" style={{ marginTop: 10, gridTemplateColumns: "1fr 1fr 1fr" }}>
            <div className="field">
              <label className="label" htmlFor="optimizerStopMin">
                Stop loss range
              </label>
              <div className="row" style={{ gridTemplateColumns: "1fr 1fr" }}>
                <input
                  id="optimizerStopMin"
                  className="input"
                  type="number"
                  step="0.0001"
                  min={0}
                  value={optimizerRunForm.stopMin}
                  onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, stopMin: e.target.value }))}
                  placeholder="min"
                />
                <input
                  aria-label="Stop loss max"
                  className="input"
                  type="number"
                  step="0.0001"
                  min={0}
                  value={optimizerRunForm.stopMax}
                  onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, stopMax: e.target.value }))}
                  placeholder="max"
                />
              </div>
            </div>
            <div className="field">
              <label className="label" htmlFor="optimizerTpMin">
                Take profit range
              </label>
              <div className="row" style={{ gridTemplateColumns: "1fr 1fr" }}>
                <input
                  id="optimizerTpMin"
                  className="input"
                  type="number"
                  step="0.0001"
                  min={0}
                  value={optimizerRunForm.tpMin}
                  onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, tpMin: e.target.value }))}
                  placeholder="min"
                />
                <input
                  aria-label="Take profit max"
                  className="input"
                  type="number"
                  step="0.0001"
                  min={0}
                  value={optimizerRunForm.tpMax}
                  onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, tpMax: e.target.value }))}
                  placeholder="max"
                />
              </div>
            </div>
            <div className="field">
              <label className="label" htmlFor="optimizerTrailMin">
                Trailing stop range
              </label>
              <div className="row" style={{ gridTemplateColumns: "1fr 1fr" }}>
                <input
                  id="optimizerTrailMin"
                  className="input"
                  type="number"
                  step="0.0001"
                  min={0}
                  value={optimizerRunForm.trailMin}
                  onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, trailMin: e.target.value }))}
                  placeholder="min"
                />
                <input
                  aria-label="Trailing stop max"
                  className="input"
                  type="number"
                  step="0.0001"
                  min={0}
                  value={optimizerRunForm.trailMax}
                  onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, trailMax: e.target.value }))}
                  placeholder="max"
                />
              </div>
            </div>
          </div>
          <div className="row" style={{ marginTop: 10, gridTemplateColumns: "1fr 1fr" }}>
            <div className="field">
              <label className="label" htmlFor="optimizerMethodWeightBlend">
                Blend method weight
              </label>
              <input
                id="optimizerMethodWeightBlend"
                className="input"
                type="number"
                step="0.1"
                min={0}
                value={optimizerRunForm.methodWeightBlend}
                onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, methodWeightBlend: e.target.value }))}
                placeholder="0.0"
              />
            </div>
            <div className="field">
              <label className="label" htmlFor="optimizerBlendWeightMin">
                Blend weight range
              </label>
              <div className="row" style={{ gridTemplateColumns: "1fr 1fr" }}>
                <input
                  id="optimizerBlendWeightMin"
                  className="input"
                  type="number"
                  step="0.01"
                  min={0}
                  max={1}
                  value={optimizerRunForm.blendWeightMin}
                  onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, blendWeightMin: e.target.value }))}
                  placeholder="min"
                />
                <input
                  aria-label="Blend weight max"
                  className="input"
                  type="number"
                  step="0.01"
                  min={0}
                  max={1}
                  value={optimizerRunForm.blendWeightMax}
                  onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, blendWeightMax: e.target.value }))}
                  placeholder="max"
                />
              </div>
            </div>
          </div>
        </details>
        <details className="details" style={{ marginTop: 12 }}>
          <summary>Advanced JSON overrides</summary>
          <div className="row" style={{ marginTop: 10 }}>
            <div className="field">
              <label className="label" htmlFor="optimizerExtraJson">
                Extra JSON fields
              </label>
              <textarea
                id="optimizerExtraJson"
                className="input"
                value={optimizerRunForm.extraJson}
                onChange={(e) => setOptimizerRunForm((prev) => ({ ...prev, extraJson: e.target.value }))}
                placeholder='{"minPositionSizeMin": 0.1, "pConfirmQuantiles": 0.2}'
                rows={4}
                spellCheck={false}
              />
              <div className="hint">Merged into the request (overrides form inputs when keys overlap).</div>
              {optimizerRunExtras.error ? (
                <div className="hint" style={{ color: "rgba(239, 68, 68, 0.85)" }}>
                  {optimizerRunExtras.error}
                </div>
              ) : null}
            </div>
          </div>
        </details>
        {optimizerRunUi.lastRunAtMs ? (
          <div className="hint" style={{ marginTop: 12 }}>
            Last optimizer run {fmtTimeMs(optimizerRunUi.lastRunAtMs)}.
          </div>
        ) : null}
        {optimizerRunUi.response ? (
          <div style={{ marginTop: 10 }}>
            <div className="label">Last optimizer output</div>
            {optimizerRunRecordJson ? <pre className="code">{optimizerRunRecordJson}</pre> : null}
            {optimizerRunUi.response.stdout ? <pre className="code">{optimizerRunUi.response.stdout}</pre> : null}
            {optimizerRunUi.response.stderr ? (
              <pre className="code" style={{ borderColor: "rgba(239, 68, 68, 0.35)" }}>
                {optimizerRunUi.response.stderr}
              </pre>
            ) : null}
          </div>
        ) : null}
        {optimizerRunUi.error && !optimizerRunValidationError ? (
          <div className="hint" style={{ marginTop: 8, color: "rgba(239, 68, 68, 0.85)" }}>
            {optimizerRunUi.error}
          </div>
        ) : null}
          </div>
        </details>
        <div className="combosList">
          <TopCombosChart
            combos={deferredTopCombos}
            loading={topCombosLoading}
            error={topCombosError}
            selectedId={selectedComboId}
            onSelect={handleComboPreview}
            onApply={handleComboApply}
          />
        </div>
        <div className="hint">
          Select a combo to preview. Click Apply to load params into the form and auto-start a live bot for that symbol (Binance only). bars=0 uses all CSV data or the exchange default (500).
        </div>
        <div className="hint">
          Top combos auto-apply when available (manual overrides respected). If the bot is idle, it will auto-start once the top combo is applied.
        </div>
      </div>
    </div>
  </div>
  );
}
