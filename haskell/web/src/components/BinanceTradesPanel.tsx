import React, { useEffect, useMemo, useState } from "react";
import type { BinancePnlAnalysis, BinanceTradesUiState, CommissionTotal, OrderSideFilter } from "../app/appHelpers";
import { binanceTradeSideLabel, csvEscape, pnlBadgeClass } from "../app/appHelpers";
import { fmtTimeMs, fmtTimeMsWithMs, marketLabel, numFromInput } from "../app/utils";
import { fmtMoney, fmtNum, fmtPct } from "../lib/format";
import type { BinanceTrade } from "../lib/types";

type BinanceTradesFilteredTotals = {
  totalPnl: number;
  totalPnlCount: number;
  commissionTotals: CommissionTotal[];
};

export type BinanceTradesPanelProps = {
  binanceTradesSymbolsInput: string;
  setBinanceTradesSymbolsInput: (value: string) => void;
  binanceTradesLimit: number;
  setBinanceTradesLimit: (value: number) => void;
  binanceTradesStartInput: string;
  setBinanceTradesStartInput: (value: string) => void;
  binanceTradesEndInput: string;
  setBinanceTradesEndInput: (value: string) => void;
  binanceTradesFromIdInput: string;
  setBinanceTradesFromIdInput: (value: string) => void;
  binanceTradesUi: BinanceTradesUiState;
  setBinanceTradesUi: (value: BinanceTradesUiState) => void;
  binanceTradesInputError: string | null;
  fetchBinanceTrades: () => Promise<void>;
  copyText: (value: string) => Promise<void> | void;
  showToast: (value: string) => void;
  binanceTradesCopyText: string;
  binanceTradesJson: string | null;
  binanceTradesTotalCount: number;
  binanceTradesFilterActive: boolean;
  binanceTradesFilteredCount: number;
  binanceTradesFilterSymbolsInput: string;
  setBinanceTradesFilterSymbolsInput: (value: string) => void;
  binanceTradesFilterSide: OrderSideFilter;
  setBinanceTradesFilterSide: (value: OrderSideFilter) => void;
  binanceTradesFilterStartInput: string;
  setBinanceTradesFilterStartInput: (value: string) => void;
  binanceTradesFilterEndInput: string;
  setBinanceTradesFilterEndInput: (value: string) => void;
  binanceTradesFilterError: string | null;
  binanceTradesFilteredTotals: BinanceTradesFilteredTotals;
  binanceTradesFilteredPnlLabel: string;
  binanceTradesFilteredCommissionLabel: string;
  binanceTradesFiltered: BinanceTrade[];
  binanceTradesAnalysis: BinancePnlAnalysis | null;
};

export function BinanceTradesPanel({
  binanceTradesSymbolsInput,
  setBinanceTradesSymbolsInput,
  binanceTradesLimit,
  setBinanceTradesLimit,
  binanceTradesStartInput,
  setBinanceTradesStartInput,
  binanceTradesEndInput,
  setBinanceTradesEndInput,
  binanceTradesFromIdInput,
  setBinanceTradesFromIdInput,
  binanceTradesUi,
  setBinanceTradesUi,
  binanceTradesInputError,
  fetchBinanceTrades,
  copyText,
  showToast,
  binanceTradesCopyText,
  binanceTradesJson,
  binanceTradesTotalCount,
  binanceTradesFilterActive,
  binanceTradesFilteredCount,
  binanceTradesFilterSymbolsInput,
  setBinanceTradesFilterSymbolsInput,
  binanceTradesFilterSide,
  setBinanceTradesFilterSide,
  binanceTradesFilterStartInput,
  setBinanceTradesFilterStartInput,
  binanceTradesFilterEndInput,
  setBinanceTradesFilterEndInput,
  binanceTradesFilterError,
  binanceTradesFilteredTotals,
  binanceTradesFilteredPnlLabel,
  binanceTradesFilteredCommissionLabel,
  binanceTradesFiltered,
  binanceTradesAnalysis,
}: BinanceTradesPanelProps) {
  const [clearedCache, setClearedCache] = useState<BinanceTradesUiState | null>(null);

  useEffect(() => {
    if (!clearedCache) return;
    if (binanceTradesUi.response || binanceTradesUi.error || binanceTradesUi.loading) {
      setClearedCache(null);
    }
  }, [binanceTradesUi.error, binanceTradesUi.loading, binanceTradesUi.response, clearedCache]);

  const tradesCsv = useMemo(() => {
    if (binanceTradesFiltered.length === 0) return "";
    const header = [
      "timeMs",
      "timeIso",
      "symbol",
      "side",
      "price",
      "qty",
      "quoteQty",
      "realizedPnl",
      "commission",
      "commissionAsset",
      "positionSide",
      "orderId",
      "tradeId",
    ];
    const rows = binanceTradesFiltered.map((trade) => {
      const timeMs = trade.time ?? null;
      const timeIso = Number.isFinite(timeMs) ? new Date(timeMs).toISOString() : "";
      const side = binanceTradeSideLabel(trade);
      return [
        timeMs ?? "",
        timeIso,
        trade.symbol ?? "",
        side,
        trade.price ?? "",
        trade.qty ?? "",
        trade.quoteQty ?? "",
        trade.realizedPnl ?? "",
        trade.commission ?? "",
        trade.commissionAsset ?? "",
        trade.positionSide ?? "",
        trade.orderId ?? "",
        trade.tradeId ?? "",
      ];
    });
    return [header.map(csvEscape).join(","), ...rows.map((row) => row.map(csvEscape).join(","))].join("\n");
  }, [binanceTradesFiltered]);

  return (
    <>
<div className="row">
  <div className="field" style={{ flex: "2 1 360px" }}>
    <label className="label" htmlFor="binanceTradesSymbols">
      Symbols (optional)
    </label>
    <input
      id="binanceTradesSymbols"
      className="input"
      value={binanceTradesSymbolsInput}
      onChange={(e) => setBinanceTradesSymbolsInput(e.target.value)}
      placeholder="BTCUSDT, ETHUSDT"
    />
    <div className="hint">Leave blank for all symbols (futures only). Spot/margin require a symbol.</div>
  </div>
  <div className="field" style={{ flex: "1 1 160px" }}>
    <label className="label" htmlFor="binanceTradesLimit">
      Limit
    </label>
    <input
      id="binanceTradesLimit"
      className="input"
      type="number"
      min={1}
      max={1000}
      value={binanceTradesLimit}
      onChange={(e) => setBinanceTradesLimit(numFromInput(e.target.value, binanceTradesLimit))}
    />
    <div className="hint">Max 1000 per request.</div>
  </div>
</div>
<div className="row" style={{ marginTop: 10 }}>
  <div className="field">
    <label className="label" htmlFor="binanceTradesStart">
      Start time (optional)
    </label>
    <input
      id="binanceTradesStart"
      className="input"
      value={binanceTradesStartInput}
      onChange={(e) => setBinanceTradesStartInput(e.target.value)}
      placeholder="2025-12-23T00:00:00Z or 1700000000000"
    />
  </div>
  <div className="field">
    <label className="label" htmlFor="binanceTradesEnd">
      End time (optional)
    </label>
    <input
      id="binanceTradesEnd"
      className="input"
      value={binanceTradesEndInput}
      onChange={(e) => setBinanceTradesEndInput(e.target.value)}
      placeholder="2025-12-23T23:59:59Z"
    />
  </div>
  <div className="field">
    <label className="label" htmlFor="binanceTradesFromId">
      From ID (optional)
    </label>
    <input
      id="binanceTradesFromId"
      className="input"
      value={binanceTradesFromIdInput}
      onChange={(e) => setBinanceTradesFromIdInput(e.target.value)}
      placeholder="Trade ID"
    />
  </div>
</div>
<div className="actions" style={{ marginTop: 10 }}>
  <button
    className="btn btnPrimary"
    disabled={binanceTradesUi.loading || Boolean(binanceTradesInputError)}
    onClick={() => void fetchBinanceTrades()}
  >
    {binanceTradesUi.loading ? "Loading…" : "Fetch trades"}
  </button>
  <button
    className="btn"
    type="button"
    disabled={!binanceTradesUi.response}
    onClick={() => {
      void copyText(binanceTradesCopyText);
      showToast("Copied trade log");
    }}
  >
    Copy
  </button>
  <button
    className="btn"
    type="button"
    disabled={binanceTradesFiltered.length === 0}
    onClick={() => {
      void copyText(tradesCsv);
      showToast("Copied trades CSV");
    }}
  >
    Copy CSV
  </button>
  <button
    className="btn"
    type="button"
    disabled={!binanceTradesJson}
    onClick={() => {
      void copyText(binanceTradesJson);
      showToast("Copied trade log JSON");
    }}
  >
    Copy JSON
  </button>
  <button
    className="btn"
    type="button"
    disabled={!binanceTradesUi.response && !binanceTradesUi.error}
    onClick={() => {
      if (!binanceTradesUi.response && !binanceTradesUi.error) return;
      setClearedCache(binanceTradesUi);
      setBinanceTradesUi({ loading: false, error: null, response: null });
      showToast("Trades cleared");
    }}
  >
    Clear
  </button>
</div>
{clearedCache ? (
  <div className="undoBar">
    <span>Trades cleared.</span>
    <button
      className="btnSmall"
      type="button"
      onClick={() => {
        setBinanceTradesUi(clearedCache);
        setClearedCache(null);
        showToast("Trades restored");
      }}
    >
      Undo
    </button>
    <button className="btnSmall" type="button" onClick={() => setClearedCache(null)}>
      Dismiss
    </button>
  </div>
) : null}
{binanceTradesInputError ? (
  <div className="hint" style={{ marginTop: 10, color: "rgba(239, 68, 68, 0.9)" }}>
    {binanceTradesInputError}
  </div>
) : null}
{binanceTradesUi.error ? (
  <div className="hint" style={{ marginTop: 10, color: "rgba(239, 68, 68, 0.9)", whiteSpace: "pre-wrap" }}>
    {binanceTradesUi.error}
  </div>
) : null}
{binanceTradesUi.response ? (
  <>
    <div className="pillRow" style={{ marginTop: 12, marginBottom: 10 }}>
      <span className="badge">{marketLabel(binanceTradesUi.response.market)}</span>
      <span className="badge">{binanceTradesUi.response.testnet ? "TESTNET" : "LIVE"}</span>
      <span className="badge">{binanceTradesTotalCount} trades</span>
      {binanceTradesFilterActive ? <span className="badge">filtered {binanceTradesFilteredCount}</span> : null}
      <span className="badge">
        {binanceTradesUi.response.allSymbols
          ? "all symbols"
          : binanceTradesUi.response.symbols.length > 0
            ? binanceTradesUi.response.symbols.join(", ")
            : "symbol"}
      </span>
      <span className="badge">fetched {fmtTimeMs(binanceTradesUi.response.fetchedAtMs)}</span>
    </div>
    <div className="pillRow" style={{ marginBottom: 10 }}>
      <input
        className="input"
        style={{ flex: "1 1 220px" }}
        value={binanceTradesFilterSymbolsInput}
        onChange={(e) => setBinanceTradesFilterSymbolsInput(e.target.value)}
        placeholder="Filter symbols (BTCUSDT, ETHUSDT)"
        spellCheck={false}
        aria-label="Filter trade symbols"
      />
      <select
        className="select"
        style={{ width: 140 }}
        value={binanceTradesFilterSide}
        onChange={(e) => setBinanceTradesFilterSide(e.target.value as OrderSideFilter)}
        aria-label="Filter trade side"
      >
        <option value="ALL">All sides</option>
        <option value="BUY">BUY</option>
        <option value="SELL">SELL</option>
      </select>
      <input
        className="input"
        style={{ flex: "1 1 200px" }}
        value={binanceTradesFilterStartInput}
        onChange={(e) => setBinanceTradesFilterStartInput(e.target.value)}
        placeholder="Filter start date"
        aria-label="Filter start date"
      />
      <input
        className="input"
        style={{ flex: "1 1 200px" }}
        value={binanceTradesFilterEndInput}
        onChange={(e) => setBinanceTradesFilterEndInput(e.target.value)}
        placeholder="Filter end date"
        aria-label="Filter end date"
      />
      <button
        className="btn"
        type="button"
        disabled={!binanceTradesFilterActive && !binanceTradesFilterError}
        onClick={() => {
          setBinanceTradesFilterSymbolsInput("");
          setBinanceTradesFilterSide("ALL");
          setBinanceTradesFilterStartInput("");
          setBinanceTradesFilterEndInput("");
          showToast("Cleared trade filters");
        }}
      >
        Clear filters
      </button>
    </div>
    {binanceTradesFilterError ? (
      <div className="hint" style={{ marginTop: 6, color: "rgba(239, 68, 68, 0.9)" }}>
        {binanceTradesFilterError}
      </div>
    ) : (
      <div className="hint" style={{ marginTop: 6 }}>
        Filter dates accept unix ms timestamps or ISO dates (YYYY-MM-DD or YYYY-MM-DDTHH:MM).
      </div>
    )}
    <div className="pillRow" style={{ marginTop: 8 }}>
      {binanceTradesFilterActive ? (
        <span className="badge">
          showing {binanceTradesFilteredCount} of {binanceTradesTotalCount}
        </span>
      ) : null}
      <span
        className={pnlBadgeClass(
          binanceTradesFilteredTotals.totalPnlCount > 0 ? binanceTradesFilteredTotals.totalPnl : null,
        )}
      >
        Total P&amp;L {binanceTradesFilteredPnlLabel}
      </span>
      <span className="badge">Total commission {binanceTradesFilteredCommissionLabel}</span>
    </div>
    {binanceTradesFilteredCount > 0 ? (
      binanceTradesAnalysis ? (
        <details className="details" style={{ marginTop: 12 }}>
          <summary>Trade P&amp;L analysis</summary>
          <div style={{ marginTop: 10 }}>
            {(() => {
              const stats = binanceTradesAnalysis;
              const response = binanceTradesUi.response;
              if (!response) return null;
              const winRateLabel = stats.winRate != null && Number.isFinite(stats.winRate) ? fmtPct(stats.winRate, 1) : "—";
              const avgWinLabel = stats.avgWin != null && Number.isFinite(stats.avgWin) ? fmtMoney(stats.avgWin, 4) : "—";
              const avgLossLabel = stats.avgLoss != null && Number.isFinite(stats.avgLoss) ? fmtMoney(stats.avgLoss, 4) : "—";
              const avgPnlLabel = stats.avgPnl != null && Number.isFinite(stats.avgPnl) ? fmtMoney(stats.avgPnl, 4) : "—";
              const maxWinLabel = stats.maxWin != null && Number.isFinite(stats.maxWin) ? fmtMoney(stats.maxWin, 4) : "—";
              const maxLossLabel = stats.maxLoss != null && Number.isFinite(stats.maxLoss) ? fmtMoney(stats.maxLoss, 4) : "—";
              const totalWinLabel = Number.isFinite(stats.totalWin) ? fmtMoney(stats.totalWin, 4) : "—";
              const totalLossLabel = Number.isFinite(stats.totalLoss) ? fmtMoney(stats.totalLoss, 4) : "—";
              const totalPnlLabel = Number.isFinite(stats.totalPnl) ? fmtMoney(stats.totalPnl, 4) : "—";
              const totalQtyLabel = Number.isFinite(stats.totalQty) ? fmtNum(stats.totalQty, 8) : "—";
              const totalQuoteLabel = Number.isFinite(stats.totalQuoteQty) ? fmtMoney(stats.totalQuoteQty, 2) : "—";
              const profitFactorLabel =
                stats.profitFactor == null
                  ? "—"
                  : Number.isFinite(stats.profitFactor)
                    ? fmtNum(stats.profitFactor, 3)
                    : "∞";
              const payoffRatioLabel =
                stats.payoffRatio == null
                  ? "—"
                  : Number.isFinite(stats.payoffRatio)
                    ? fmtNum(stats.payoffRatio, 3)
                    : "∞";
              const commissionLabel =
                stats.commissionTotals.length > 0
                  ? stats.commissionTotals.map((c) => `${fmtNum(c.total, 6)} ${c.asset}`).join(" • ")
                  : "—";
              const totalsScope = binanceTradesFilterActive
                ? "Totals across filtered trades."
                : response.allSymbols || response.symbols.length > 1
                  ? response.allSymbols
                    ? "Totals across all symbols."
                    : `Totals across ${response.symbols.length} symbols.`
                  : null;
              return (
                <>
                  <div className="summaryGrid">
                    <div className="summaryItem">
                      <div className="labelRow">
                        <div className="summaryLabel">Outcomes</div>
                        <InfoPopover label="Outcomes">
                          <InfoList items={ACCOUNT_TRADE_PNL_TIPS.outcomes} />
                        </InfoPopover>
                      </div>
                      <div className="summaryValue">
                        <span className="badge badgeStrong badgeLong">{stats.wins} wins</span>
                        <span className="badge badgeStrong badgeFlat">{stats.losses} losses</span>
                        <span className="badge">{stats.breakeven} flat</span>
                        <span className="summaryMeta">Win rate {winRateLabel} • trades {stats.count}</span>
                      </div>
                    </div>
                    <div className="summaryItem">
                      <div className="labelRow">
                        <div className="summaryLabel">Avg win / loss</div>
                        <InfoPopover label="Average win/loss">
                          <InfoList items={ACCOUNT_TRADE_PNL_TIPS.avgWinLoss} />
                        </InfoPopover>
                      </div>
                      <div className="summaryValue">
                        <span className={pnlBadgeClass(stats.avgWin)}>{avgWinLabel}</span>
                        <span className={pnlBadgeClass(stats.avgLoss)}>{avgLossLabel}</span>
                        <span className="summaryMeta">Payoff ratio {payoffRatioLabel}</span>
                      </div>
                    </div>
                    <div className="summaryItem">
                      <div className="labelRow">
                        <div className="summaryLabel">Best / worst trade</div>
                        <InfoPopover label="Best/worst trade">
                          <InfoList items={ACCOUNT_TRADE_PNL_TIPS.bestWorst} />
                        </InfoPopover>
                      </div>
                      <div className="summaryValue">
                        <span className={pnlBadgeClass(stats.maxWin)}>{maxWinLabel}</span>
                        <span className={pnlBadgeClass(stats.maxLoss)}>{maxLossLabel}</span>
                        <span className="summaryMeta">Avg P&amp;L {avgPnlLabel}</span>
                      </div>
                    </div>
                    <div className="summaryItem">
                      <div className="labelRow">
                        <div className="summaryLabel">Total P&amp;L</div>
                        <InfoPopover label="Total P&L" align="left">
                          <InfoList items={ACCOUNT_TRADE_PNL_TIPS.totalPnl} />
                        </InfoPopover>
                      </div>
                      <div className="summaryValue">
                        <span className={pnlBadgeClass(stats.totalPnl)}>{totalPnlLabel}</span>
                        <span className={pnlBadgeClass(stats.totalWin)}>{totalWinLabel}</span>
                        <span className={pnlBadgeClass(stats.totalLoss)}>{totalLossLabel}</span>
                        <span className="summaryMeta">Profit factor {profitFactorLabel} • Fees {commissionLabel}</span>
                      </div>
                    </div>
                    <div className="summaryItem">
                      <div className="labelRow">
                        <div className="summaryLabel">Totals</div>
                        <InfoPopover label="Totals" align="left">
                          <InfoList items={ACCOUNT_TRADE_PNL_TIPS.totals} />
                        </InfoPopover>
                      </div>
                      <div className="summaryValue">
                        <span className="badge">Qty {totalQtyLabel}</span>
                        <span className="badge">Quote {totalQuoteLabel}</span>
                        {totalsScope ? <span className="summaryMeta">{totalsScope}</span> : null}
                      </div>
                    </div>
                  </div>
                  <div className="chartBlock" style={{ marginTop: 12 }}>
                    <div className="hint">Top winners</div>
                    {stats.topWins.length > 0 ? (
                      <div className="tableWrap" role="region" aria-label="Top winning account trades">
                        <table className="table">
                          <thead>
                            <tr>
                              <th>Time</th>
                              <th>Symbol</th>
                              <th>Side</th>
                              <th>Price</th>
                              <th>Qty</th>
                              <th>Pos</th>
                              <th>PNL</th>
                              <th>Commission</th>
                              <th>Order</th>
                            </tr>
                          </thead>
                          <tbody>
                            {stats.topWins.map((row) => {
                              const sideClass =
                                row.side === "BUY"
                                  ? "badge badgeStrong badgeLong"
                                  : row.side === "SELL"
                                    ? "badge badgeStrong badgeFlat"
                                    : "badge";
                              const qtyTxt = Number.isFinite(row.qty) ? fmtNum(row.qty, 8) : "—";
                              const pnlTxt = Number.isFinite(row.realizedPnl) ? fmtMoney(row.realizedPnl, 4) : "—";
                              const commissionTxt =
                                row.commission != null && Number.isFinite(row.commission)
                                  ? `${fmtNum(row.commission, 8)}${row.commissionAsset ? ` ${row.commissionAsset}` : ""}`
                                  : "—";
                              return (
                                <tr key={`binance-win-${row.tradeId}`}>
                                  <td className="tdMono">{fmtTimeMsWithMs(row.time)}</td>
                                  <td className="tdMono">{row.symbol}</td>
                                  <td>
                                    <span className={sideClass}>{row.side}</span>
                                  </td>
                                  <td className="tdMono">{fmtMoney(row.price, 4)}</td>
                                  <td className="tdMono">{qtyTxt}</td>
                                  <td className="tdMono">{row.positionSide ?? "—"}</td>
                                  <td>
                                    <span className={pnlBadgeClass(row.realizedPnl)}>{pnlTxt}</span>
                                  </td>
                                  <td className="tdMono">{commissionTxt}</td>
                                  <td className="tdMono">{row.orderId ?? "—"}</td>
                                </tr>
                              );
                            })}
                          </tbody>
                        </table>
                      </div>
                    ) : (
                      <div className="hint">No winning trades in this sample.</div>
                    )}
                  </div>
                  <div className="chartBlock" style={{ marginTop: 12 }}>
                    <div className="hint">Top losers</div>
                    {stats.topLosses.length > 0 ? (
                      <div className="tableWrap" role="region" aria-label="Top losing account trades">
                        <table className="table">
                          <thead>
                            <tr>
                              <th>Time</th>
                              <th>Symbol</th>
                              <th>Side</th>
                              <th>Price</th>
                              <th>Qty</th>
                              <th>Pos</th>
                              <th>PNL</th>
                              <th>Commission</th>
                              <th>Order</th>
                            </tr>
                          </thead>
                          <tbody>
                            {stats.topLosses.map((row) => {
                              const sideClass =
                                row.side === "BUY"
                                  ? "badge badgeStrong badgeLong"
                                  : row.side === "SELL"
                                    ? "badge badgeStrong badgeFlat"
                                    : "badge";
                              const qtyTxt = Number.isFinite(row.qty) ? fmtNum(row.qty, 8) : "—";
                              const pnlTxt = Number.isFinite(row.realizedPnl) ? fmtMoney(row.realizedPnl, 4) : "—";
                              const commissionTxt =
                                row.commission != null && Number.isFinite(row.commission)
                                  ? `${fmtNum(row.commission, 8)}${row.commissionAsset ? ` ${row.commissionAsset}` : ""}`
                                  : "—";
                              return (
                                <tr key={`binance-loss-${row.tradeId}`}>
                                  <td className="tdMono">{fmtTimeMsWithMs(row.time)}</td>
                                  <td className="tdMono">{row.symbol}</td>
                                  <td>
                                    <span className={sideClass}>{row.side}</span>
                                  </td>
                                  <td className="tdMono">{fmtMoney(row.price, 4)}</td>
                                  <td className="tdMono">{qtyTxt}</td>
                                  <td className="tdMono">{row.positionSide ?? "—"}</td>
                                  <td>
                                    <span className={pnlBadgeClass(row.realizedPnl)}>{pnlTxt}</span>
                                  </td>
                                  <td className="tdMono">{commissionTxt}</td>
                                  <td className="tdMono">{row.orderId ?? "—"}</td>
                                </tr>
                              );
                            })}
                          </tbody>
                        </table>
                      </div>
                    ) : (
                      <div className="hint">No losing trades in this sample.</div>
                    )}
                  </div>
                  <div className="hint" style={{ marginTop: 8 }}>
                    P&amp;L uses Binance realizedPnl per fill. Spot/margin trades typically omit realizedPnl.
                  </div>
                </>
              );
            })()}
          </div>
        </details>
      ) : (
        <div className="hint" style={{ marginTop: 10 }}>
          Realized P&amp;L is unavailable for these trades. Binance only returns realizedPnl for futures fills.
        </div>
      )
    ) : null}
    {binanceTradesFilteredCount === 0 ? (
      <div className="hint">
        {binanceTradesTotalCount > 0 ? "No trades match the filters." : "No trades returned."}
      </div>
    ) : (
      <div className="tableWrap" role="region" aria-label="Binance account trades">
        <table className="table">
          <thead>
            <tr>
              <th>Time</th>
              <th>Symbol</th>
              <th>Side</th>
              <th>Price</th>
              <th>Qty</th>
              <th>Quote</th>
              <th>Pos</th>
              <th>Commission</th>
              <th>PNL</th>
              <th>Order</th>
            </tr>
          </thead>
          <tbody>
            {binanceTradesFiltered.map((trade) => {
              const side = binanceTradeSideLabel(trade);
              const qtyTxt = Number.isFinite(trade.qty) ? fmtNum(trade.qty, 8) : "—";
              const quoteTxt = Number.isFinite(trade.quoteQty) ? fmtMoney(trade.quoteQty, 2) : "—";
              const commissionTxt =
                trade.commission != null && Number.isFinite(trade.commission)
                  ? `${fmtNum(trade.commission, 8)}${trade.commissionAsset ? ` ${trade.commissionAsset}` : ""}`
                  : "—";
              const pnlTxt =
                trade.realizedPnl != null && Number.isFinite(trade.realizedPnl) ? fmtMoney(trade.realizedPnl, 4) : "—";
              return (
                <tr key={`${trade.symbol}-${trade.tradeId}`}>
                  <td className="tdMono">{fmtTimeMsWithMs(trade.time)}</td>
                  <td className="tdMono">{trade.symbol}</td>
                  <td>
                    <span className={side === "BUY" ? "badge badgeStrong badgeLong" : side === "SELL" ? "badge badgeStrong badgeFlat" : "badge"}>
                      {side}
                    </span>
                  </td>
                  <td className="tdMono">{fmtMoney(trade.price, 4)}</td>
                  <td className="tdMono">{qtyTxt}</td>
                  <td className="tdMono">{quoteTxt}</td>
                  <td className="tdMono">{trade.positionSide ?? "—"}</td>
                  <td className="tdMono">{commissionTxt}</td>
                  <td className="tdMono">{pnlTxt}</td>
                  <td className="tdMono">{trade.orderId ?? "—"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    )}
    <div className="hint" style={{ marginTop: 8 }}>
      Binance returns up to 1000 trades per request. Use start/end time or fromId to page deeper history.
    </div>
  </>
) : (
  <div className="hint" style={{ marginTop: 10 }}>
    No trades loaded yet.
  </div>
)}
    </>
  );
}
