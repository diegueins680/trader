import type { ApiBinanceRevenueResponse, RevenueBreakdown } from "../lib/types";

export type RevenueReportMetadata = {
  clientName?: string;
  strategyName?: string;
  deploymentName?: string;
  reviewedCommit?: string;
};

export type RevenueReportPackage = {
  schemaVersion: 1;
  reportType: "strategy-assurance-revenue";
  generatedAtMs: number;
  metadata: RevenueReportMetadata;
  response: ApiBinanceRevenueResponse;
};

function finite(value: number): number {
  return Number.isFinite(value) ? value : 0;
}

function decimal(value: number): string {
  const rendered = finite(value).toFixed(8);
  return rendered.replace(/\.0+$/, "").replace(/(\.\d*?)0+$/, "$1");
}

function utcDate(timestampMs: number): string {
  const date = new Date(timestampMs);
  return Number.isFinite(date.getTime()) ? date.toISOString().slice(0, 10) : "unknown-date";
}

function utcTimestamp(timestampMs: number): string {
  const date = new Date(timestampMs);
  return Number.isFinite(date.getTime()) ? date.toISOString() : "unknown-time";
}

function markdownCell(value: unknown): string {
  return String(value ?? "")
    .replace(/\\/g, "\\\\")
    .replace(/\|/g, "\\|")
    .replace(/[\r\n]+/g, " ");
}

function csvCell(value: unknown): string {
  const text = String(value ?? "");
  return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

function metadataValue(value: string | undefined): string {
  const trimmed = value?.trim();
  return trimmed || "Not supplied";
}

function breakdownColumns(breakdown: RevenueBreakdown): string[] {
  return [
    decimal(breakdown.realizedPnl),
    decimal(breakdown.funding),
    decimal(breakdown.commission),
    decimal(breakdown.rebates),
    decimal(breakdown.otherOperating),
    decimal(breakdown.exchangeNet),
    decimal(breakdown.excludedNonOperating),
    decimal(breakdown.unclassified),
  ];
}

export function revenueReportBaseName(
  response: ApiBinanceRevenueResponse,
  metadata: RevenueReportMetadata,
): string {
  const safe = (value: string, fallback: string) => {
    const normalized = value.trim().replace(/[^A-Za-z0-9]+/g, "-").replace(/^-+|-+$/g, "").toLowerCase();
    return normalized || fallback;
  };
  const ledger = response.ledger;
  return [
    "strategy-assurance-revenue",
    safe(metadata.clientName ?? "", "client"),
    safe(ledger.asset, "asset"),
    utcDate(ledger.startAtMs),
    utcDate(ledger.endAtMs),
  ].join("-");
}

export function buildRevenueReportJson(
  response: ApiBinanceRevenueResponse,
  metadata: RevenueReportMetadata,
  generatedAtMs: number,
): string {
  const report: RevenueReportPackage = {
    schemaVersion: 1,
    reportType: "strategy-assurance-revenue",
    generatedAtMs,
    metadata: {
      clientName: metadata.clientName?.trim() || undefined,
      strategyName: metadata.strategyName?.trim() || undefined,
      deploymentName: metadata.deploymentName?.trim() || undefined,
      reviewedCommit: metadata.reviewedCommit?.trim() || undefined,
    },
    response,
  };
  return `${JSON.stringify(report, null, 2)}\n`;
}

export function buildRevenueDailyCsv(response: ApiBinanceRevenueResponse): string {
  const header = [
    "utcDay",
    "asset",
    "realizedPnl",
    "funding",
    "commission",
    "rebates",
    "otherOperating",
    "exchangeNet",
    "excludedNonOperating",
    "unclassified",
  ];
  const rows = response.ledger.daily.map((row) => [
    utcDate(row.startAtMs),
    response.ledger.asset,
    ...breakdownColumns(row.breakdown),
  ]);
  return `${[header, ...rows].map((row) => row.map(csvCell).join(",")).join("\n")}\n`;
}

export function buildRevenueSymbolsCsv(response: ApiBinanceRevenueResponse): string {
  const header = [
    "symbol",
    "asset",
    "realizedPnl",
    "funding",
    "commission",
    "rebates",
    "otherOperating",
    "exchangeNet",
    "excludedNonOperating",
    "unclassified",
    "trades",
    "makerTrades",
    "takerTrades",
    "unknownLiquidityTrades",
    "makerRate",
    "quoteNotional",
  ];
  const rows = response.ledger.symbols.map((row) => [
    row.symbol || "ACCOUNT",
    response.ledger.asset,
    ...breakdownColumns(row.breakdown),
    row.execution.trades,
    row.execution.makerTrades,
    row.execution.takerTrades,
    row.execution.unknownLiquidityTrades,
    row.execution.makerRate == null ? "" : decimal(row.execution.makerRate),
    decimal(row.execution.quoteNotional),
  ]);
  return `${[header, ...rows].map((row) => row.map(csvCell).join(",")).join("\n")}\n`;
}

export function buildRevenueMarkdown(
  response: ApiBinanceRevenueResponse,
  metadata: RevenueReportMetadata,
  generatedAtMs: number,
): string {
  const ledger = response.ledger;
  const breakdown = ledger.breakdown;
  const warnings = [
    ledger.incomeMayBeTruncated ? "Income history reached the request limit; this period may be incomplete." : null,
    ledger.tradesMayBeTruncated ? "Trade history reached a per-window request limit; execution metrics may be incomplete." : null,
    ledger.unclassifiedIncomeTypes.length > 0
      ? `Unclassified income types were excluded from net revenue: ${ledger.unclassifiedIncomeTypes.join(", ")}.`
      : null,
  ].filter((warning): warning is string => Boolean(warning));
  const status = warnings.length === 0 ? "COMPLETE" : "REVIEW REQUIRED";
  const makerRate = ledger.execution.makerRate == null ? "Not available" : `${decimal(ledger.execution.makerRate * 100)}%`;
  const summaryRows = [
    ["Realized P&L", decimal(breakdown.realizedPnl)],
    ["Funding", decimal(breakdown.funding)],
    ["Signed commission", decimal(breakdown.commission)],
    ["Rebates", decimal(breakdown.rebates)],
    ["Other operating", decimal(breakdown.otherOperating)],
    ["Exchange net", decimal(breakdown.exchangeNet)],
    ["Current unrealized P&L", decimal(ledger.unrealizedPnl)],
    ["Infrastructure cost", decimal(ledger.infrastructureCost)],
    ["Net revenue", decimal(ledger.netRevenue)],
    ["Excluded non-operating", decimal(breakdown.excludedNonOperating)],
    ["Unclassified, excluded", decimal(breakdown.unclassified)],
  ];
  const lines = [
    "# Strategy Assurance Revenue Snapshot",
    "",
    `Generated: ${utcTimestamp(generatedAtMs)}`,
    `Status: **${status}**`,
    "",
    "## Review identity",
    "",
    `- Client: ${markdownCell(metadataValue(metadata.clientName))}`,
    `- Strategy: ${markdownCell(metadataValue(metadata.strategyName))}`,
    `- Deployment: ${markdownCell(metadataValue(metadata.deploymentName))}`,
    `- Reviewed commit: ${markdownCell(metadataValue(metadata.reviewedCommit))}`,
    `- Market: ${markdownCell(response.market)} (${response.testnet ? "testnet" : "live"})`,
    `- Settlement asset: ${markdownCell(ledger.asset)}`,
    `- Period: ${utcTimestamp(ledger.startAtMs)} through ${utcTimestamp(ledger.endAtMs)}`,
    `- Exchange fetch completed: ${utcTimestamp(response.fetchedAtMs)}`,
    "",
    "## Accounting policy",
    "",
    "Exchange income history is the realized accounting authority. Fill-level realized P&L is not added again. Transfers, bonuses, and unclassified income are disclosed but excluded from net revenue. Current unrealized P&L is reported separately and is zero when omitted or when no qualifying position is open.",
    "",
    "## Revenue summary",
    "",
    `| Measure | Amount (${markdownCell(ledger.asset)}) |`,
    "| --- | ---: |",
    ...summaryRows.map(([label, value]) => `| ${markdownCell(label)} | ${markdownCell(value)} |`),
    "",
    "## Execution summary",
    "",
    `- Fills: ${ledger.execution.trades}`,
    `- Maker / taker / unknown: ${ledger.execution.makerTrades} / ${ledger.execution.takerTrades} / ${ledger.execution.unknownLiquidityTrades}`,
    `- Maker rate: ${makerRate}`,
    `- Quote notional: ${decimal(ledger.execution.quoteNotional)} ${markdownCell(ledger.asset)}`,
    `- Income records: ${ledger.incomeRecords}`,
    "",
    "## Completeness findings",
    "",
    ...(warnings.length > 0 ? warnings.map((warning) => `- ${markdownCell(warning)}`) : ["- No truncation or unclassified-income warning was reported."]),
    "",
    "## Daily revenue",
    "",
    "| UTC day | Realized | Funding | Commission | Rebates | Exchange net |",
    "| --- | ---: | ---: | ---: | ---: | ---: |",
    ...(ledger.daily.length > 0
      ? ledger.daily.map(
          (row) =>
            `| ${utcDate(row.startAtMs)} | ${decimal(row.breakdown.realizedPnl)} | ${decimal(row.breakdown.funding)} | ${decimal(row.breakdown.commission)} | ${decimal(row.breakdown.rebates)} | ${decimal(row.breakdown.exchangeNet)} |`,
        )
      : ["| No operating income | 0 | 0 | 0 | 0 | 0 |"]),
    "",
    "## Symbol attribution",
    "",
    "| Symbol | Exchange net | Realized | Funding | Commission | Maker rate | Fills |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ...(ledger.symbols.length > 0
      ? ledger.symbols.map((row) => {
          const symbolMakerRate = row.execution.makerRate == null ? "" : `${decimal(row.execution.makerRate * 100)}%`;
          return `| ${markdownCell(row.symbol || "ACCOUNT")} | ${decimal(row.breakdown.exchangeNet)} | ${decimal(row.breakdown.realizedPnl)} | ${decimal(row.breakdown.funding)} | ${decimal(row.breakdown.commission)} | ${symbolMakerRate} | ${row.execution.trades} |`;
        })
      : ["| No attributed activity | 0 | 0 | 0 |  |  | 0 |"]),
    "",
    "## Limitations",
    "",
    "This snapshot is engineering assurance, not an audit opinion, tax advice, investment advice, or a guarantee of future performance. Validate any accounting treatment with a qualified professional.",
    "",
  ];
  return lines.join("\n");
}
