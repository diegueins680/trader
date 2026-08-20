import { useState } from "react";
import type { BinanceRevenueUiState } from "../app/appHelpers";
import { downloadTextFile, pnlBadgeClass } from "../app/appHelpers";
import {
  buildRevenueDailyCsv,
  buildRevenueMarkdown,
  buildRevenueReportJson,
  buildRevenueSymbolsCsv,
  revenueReportBaseName,
  type RevenueReportMetadata,
} from "../app/revenueReport";
import { fmtTimeMs, marketLabel, numFromInput } from "../app/utils";
import { fmtMoney, fmtPct } from "../lib/format";

export type BinanceRevenuePanelProps = {
  asset: string;
  setAsset: (value: string) => void;
  startInput: string;
  setStartInput: (value: string) => void;
  endInput: string;
  setEndInput: (value: string) => void;
  infrastructureCost: number;
  setInfrastructureCost: (value: number) => void;
  includeUnrealized: boolean;
  setIncludeUnrealized: (value: boolean) => void;
  state: BinanceRevenueUiState;
  inputError: string | null;
  fetchRevenue: () => Promise<void>;
  showToast: (value: string) => void;
};

function money(value: number, asset: string): string {
  return `${fmtMoney(value, 4)} ${asset}`;
}

export function BinanceRevenuePanel({
  asset,
  setAsset,
  startInput,
  setStartInput,
  endInput,
  setEndInput,
  infrastructureCost,
  setInfrastructureCost,
  includeUnrealized,
  setIncludeUnrealized,
  state,
  inputError,
  fetchRevenue,
  showToast,
}: BinanceRevenuePanelProps) {
  const [clientName, setClientName] = useState("");
  const [strategyName, setStrategyName] = useState("");
  const [deploymentName, setDeploymentName] = useState("");
  const [reviewedCommit, setReviewedCommit] = useState("");
  const response = state.response;
  const ledger = response?.ledger;
  const currency = (ledger?.asset ?? asset.trim().toUpperCase()) || "USDT";
  const warningParts = ledger
    ? [
        ledger.incomeMayBeTruncated ? "Income response reached its limit; shorten the range for a complete ledger." : null,
        ledger.tradesMayBeTruncated ? "Trade response reached its limit; shorten the range for complete execution metrics." : null,
        ledger.unclassifiedIncomeTypes.length > 0
          ? `Unclassified income excluded from net revenue: ${ledger.unclassifiedIncomeTypes.join(", ")}.`
          : null,
      ].filter((item): item is string => Boolean(item))
    : [];
  const reportMetadata: RevenueReportMetadata = {
    clientName,
    strategyName,
    deploymentName,
    reviewedCommit,
  };
  const downloadReport = (format: "markdown" | "json" | "dailyCsv" | "symbolsCsv") => {
    if (!response) return;
    const generatedAtMs = Date.now();
    const baseName = revenueReportBaseName(response, reportMetadata);
    switch (format) {
      case "markdown":
        downloadTextFile(
          `${baseName}.md`,
          buildRevenueMarkdown(response, reportMetadata, generatedAtMs),
          "text/markdown;charset=utf-8",
        );
        showToast("Revenue assurance report downloaded");
        return;
      case "json":
        downloadTextFile(
          `${baseName}.json`,
          buildRevenueReportJson(response, reportMetadata, generatedAtMs),
          "application/json;charset=utf-8",
        );
        showToast("Revenue evidence JSON downloaded");
        return;
      case "dailyCsv":
        downloadTextFile(`${baseName}-daily.csv`, buildRevenueDailyCsv(response), "text/csv;charset=utf-8");
        showToast("Daily revenue CSV downloaded");
        return;
      case "symbolsCsv":
        downloadTextFile(`${baseName}-symbols.csv`, buildRevenueSymbolsCsv(response), "text/csv;charset=utf-8");
        showToast("Symbol revenue CSV downloaded");
    }
  };

  return (
    <>
      <div className="row">
        <div className="field">
          <label className="label" htmlFor="binanceRevenueAsset">Settlement asset</label>
          <input
            id="binanceRevenueAsset"
            className="input"
            value={asset}
            onChange={(event) => setAsset(event.target.value.toUpperCase())}
            placeholder="USDT"
          />
        </div>
        <div className="field">
          <label className="label" htmlFor="binanceRevenueStart">Start date (optional)</label>
          <input
            id="binanceRevenueStart"
            className="input"
            type="date"
            value={startInput}
            onChange={(event) => setStartInput(event.target.value)}
            max={endInput || undefined}
          />
        </div>
        <div className="field">
          <label className="label" htmlFor="binanceRevenueEnd">End date (optional)</label>
          <input
            id="binanceRevenueEnd"
            className="input"
            type="date"
            value={endInput}
            onChange={(event) => setEndInput(event.target.value)}
            min={startInput || undefined}
          />
        </div>
        <div className="field">
          <label className="label" htmlFor="binanceRevenueInfrastructure">Infrastructure cost ({currency})</label>
          <input
            id="binanceRevenueInfrastructure"
            className="input"
            type="number"
            min={0}
            step="0.01"
            value={infrastructureCost}
            onChange={(event) => setInfrastructureCost(numFromInput(event.target.value, infrastructureCost))}
          />
        </div>
      </div>
      <div className="actions" style={{ marginTop: 10 }}>
        <button
          className="btn btnPrimary"
          disabled={state.loading || Boolean(inputError)}
          onClick={() => void fetchRevenue()}
        >
          {state.loading ? "Reconciling…" : "Reconcile revenue"}
        </button>
        <label className="checkRow">
          <input
            type="checkbox"
            checked={includeUnrealized}
            onChange={(event) => setIncludeUnrealized(event.target.checked)}
          />
          Include current unrealized P&amp;L
        </label>
      </div>
      <div className="hint" style={{ marginTop: 8 }}>
        Defaults to the latest seven days. Exchange income is authoritative; transfers are disclosed but excluded from revenue.
      </div>
      {inputError ? <div className="hint" style={{ marginTop: 8, color: "rgba(239, 68, 68, 0.9)" }}>{inputError}</div> : null}
      {state.error ? <div className="hint" style={{ marginTop: 8, color: "rgba(239, 68, 68, 0.9)" }}>{state.error}</div> : null}
      {ledger && response ? (
        <>
          <div className="pillRow" style={{ marginTop: 12, marginBottom: 10 }}>
            <span className="badge">{marketLabel(response.market)}</span>
            <span className="badge">{response.testnet ? "TESTNET" : "LIVE"}</span>
            <span className="badge">{ledger.incomeRecords} income records</span>
            <span className="badge">{ledger.tradeRecords} fills</span>
            <span className="badge">
              {new Date(ledger.startAtMs).toISOString().slice(0, 10)} → {new Date(ledger.endAtMs).toISOString().slice(0, 10)}
            </span>
            <span className="badge">fetched {fmtTimeMs(response.fetchedAtMs)}</span>
          </div>
          {warningParts.length > 0 ? (
            <div className="hint" style={{ marginBottom: 10, color: "rgba(245, 158, 11, 0.95)" }}>
              {warningParts.join(" ")}
            </div>
          ) : null}
          <div className="summaryGrid">
            <div className="summaryItem">
              <div className="summaryLabel">Net revenue</div>
              <div className="summaryValue"><span className={pnlBadgeClass(ledger.netRevenue)}>{money(ledger.netRevenue, currency)}</span></div>
              <div className="summaryMeta">Exchange net + optional unrealized − infrastructure</div>
            </div>
            <div className="summaryItem">
              <div className="summaryLabel">Realized P&amp;L</div>
              <div className="summaryValue"><span className={pnlBadgeClass(ledger.breakdown.realizedPnl)}>{money(ledger.breakdown.realizedPnl, currency)}</span></div>
              <div className="summaryMeta">Exchange income ledger, not reconstructed from fills</div>
            </div>
            <div className="summaryItem">
              <div className="summaryLabel">Funding / fees</div>
              <div className="summaryValue">
                <span className={pnlBadgeClass(ledger.breakdown.funding)}>{money(ledger.breakdown.funding, currency)}</span>
                <span className={pnlBadgeClass(ledger.breakdown.commission)}>{money(ledger.breakdown.commission, currency)}</span>
              </div>
              <div className="summaryMeta">Funding followed by signed commission</div>
            </div>
            <div className="summaryItem">
              <div className="summaryLabel">Rebates / other</div>
              <div className="summaryValue">
                <span className={pnlBadgeClass(ledger.breakdown.rebates)}>{money(ledger.breakdown.rebates, currency)}</span>
                <span className={pnlBadgeClass(ledger.breakdown.otherOperating)}>{money(ledger.breakdown.otherOperating, currency)}</span>
              </div>
            </div>
            <div className="summaryItem">
              <div className="summaryLabel">Execution</div>
              <div className="summaryValue">
                <span className="badge">{ledger.execution.makerTrades} maker</span>
                <span className="badge">{ledger.execution.takerTrades} taker</span>
              </div>
              <div className="summaryMeta">Maker rate {ledger.execution.makerRate == null ? "—" : fmtPct(ledger.execution.makerRate)} • notional {money(ledger.execution.quoteNotional, currency)}</div>
            </div>
            <div className="summaryItem">
              <div className="summaryLabel">Adjustments</div>
              <div className="summaryValue">
                <span className={pnlBadgeClass(ledger.unrealizedPnl)}>{money(ledger.unrealizedPnl, currency)} unrealized</span>
                <span className="badge">{money(ledger.infrastructureCost, currency)} infra</span>
              </div>
              <div className="summaryMeta">Excluded transfers {money(ledger.breakdown.excludedNonOperating, currency)}</div>
            </div>
          </div>
          <div style={{ borderTop: "1px solid var(--stroke)", margin: "16px 0 12px" }} />
          <div className="hint" style={{ marginBottom: 10 }}>
            Optional report identity. These values stay in the downloaded files and are not sent to the API.
          </div>
          <div className="row">
            <div className="field">
              <label className="label" htmlFor="binanceRevenueClient">Client</label>
              <input
                id="binanceRevenueClient"
                className="input"
                value={clientName}
                onChange={(event) => setClientName(event.target.value)}
                placeholder="Client or account owner"
              />
            </div>
            <div className="field">
              <label className="label" htmlFor="binanceRevenueStrategy">Strategy</label>
              <input
                id="binanceRevenueStrategy"
                className="input"
                value={strategyName}
                onChange={(event) => setStrategyName(event.target.value)}
                placeholder="Reviewed strategy"
              />
            </div>
            <div className="field">
              <label className="label" htmlFor="binanceRevenueDeployment">Deployment</label>
              <input
                id="binanceRevenueDeployment"
                className="input"
                value={deploymentName}
                onChange={(event) => setDeploymentName(event.target.value)}
                placeholder="Production deployment"
              />
            </div>
            <div className="field">
              <label className="label" htmlFor="binanceRevenueCommit">Reviewed commit</label>
              <input
                id="binanceRevenueCommit"
                className="input"
                value={reviewedCommit}
                onChange={(event) => setReviewedCommit(event.target.value)}
                placeholder="Git commit"
              />
            </div>
          </div>
          <div className="actions" style={{ marginTop: 10 }}>
            <button className="btn btnPrimary" type="button" onClick={() => downloadReport("markdown")}>
              Download assurance report
            </button>
            <button className="btn" type="button" onClick={() => downloadReport("json")}>
              Evidence JSON
            </button>
            <button className="btn" type="button" onClick={() => downloadReport("dailyCsv")}>
              Daily CSV
            </button>
            <button className="btn" type="button" onClick={() => downloadReport("symbolsCsv")}>
              Symbol CSV
            </button>
          </div>
          <div className="tableWrap" role="region" aria-label="Revenue by day" style={{ marginTop: 12 }}>
            <table className="table">
              <thead><tr><th>UTC day</th><th>Realized</th><th>Funding</th><th>Fees</th><th>Rebates</th><th>Exchange net</th></tr></thead>
              <tbody>
                {ledger.daily.length > 0 ? ledger.daily.map((row) => (
                  <tr key={row.startAtMs}>
                    <td>{new Date(row.startAtMs).toISOString().slice(0, 10)}</td>
                    <td>{money(row.breakdown.realizedPnl, currency)}</td>
                    <td>{money(row.breakdown.funding, currency)}</td>
                    <td>{money(row.breakdown.commission, currency)}</td>
                    <td>{money(row.breakdown.rebates, currency)}</td>
                    <td><span className={pnlBadgeClass(row.breakdown.exchangeNet)}>{money(row.breakdown.exchangeNet, currency)}</span></td>
                  </tr>
                )) : <tr><td colSpan={6}>No operating income in this period.</td></tr>}
              </tbody>
            </table>
          </div>
          <div className="tableWrap" role="region" aria-label="Revenue by symbol" style={{ marginTop: 12 }}>
            <table className="table">
              <thead><tr><th>Symbol</th><th>Exchange net</th><th>Realized</th><th>Funding</th><th>Fees</th><th>Maker rate</th><th>Fills</th></tr></thead>
              <tbody>
                {ledger.symbols.length > 0 ? ledger.symbols.map((row) => (
                  <tr key={row.symbol}>
                    <td>{row.symbol || "Account"}</td>
                    <td><span className={pnlBadgeClass(row.breakdown.exchangeNet)}>{money(row.breakdown.exchangeNet, currency)}</span></td>
                    <td>{money(row.breakdown.realizedPnl, currency)}</td>
                    <td>{money(row.breakdown.funding, currency)}</td>
                    <td>{money(row.breakdown.commission, currency)}</td>
                    <td>{row.execution.makerRate == null ? "—" : fmtPct(row.execution.makerRate)}</td>
                    <td>{row.execution.trades}</td>
                  </tr>
                )) : <tr><td colSpan={7}>No symbol-level activity in this period.</td></tr>}
              </tbody>
            </table>
          </div>
        </>
      ) : null}
    </>
  );
}
