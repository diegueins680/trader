import assert from "node:assert/strict";
import { test } from "node:test";

const reportBundleUrl = new URL("../.tmp/web-tests/revenueReport.js", import.meta.url);

async function loadReportModule() {
  const url = new URL(reportBundleUrl);
  url.searchParams.set("cachebust", `${Date.now()}-${Math.random()}`);
  return import(url.href);
}

function breakdown(overrides = {}) {
  return {
    realizedPnl: 100,
    funding: -2,
    commission: -4,
    rebates: 1,
    otherOperating: 0,
    exchangeNet: 95,
    excludedNonOperating: 500,
    unclassified: 0,
    ...overrides,
  };
}

function response(overrides = {}) {
  return {
    market: "futures",
    testnet: false,
    fetchedAtMs: 1_704_153_600_000,
    ledger: {
      asset: "USDT",
      startAtMs: 1_704_067_200_000,
      endAtMs: 1_704_153_599_999,
      incomeRecords: 5,
      tradeRecords: 2,
      incomeMayBeTruncated: false,
      tradesMayBeTruncated: false,
      breakdown: breakdown(),
      unrealizedPnl: 5,
      infrastructureCost: 3,
      netRevenue: 97,
      execution: {
        trades: 2,
        makerTrades: 1,
        takerTrades: 1,
        unknownLiquidityTrades: 0,
        makerRate: 0.5,
        quoteNotional: 1_000,
      },
      daily: [{ startAtMs: 1_704_067_200_000, breakdown: breakdown() }],
      symbols: [
        {
          symbol: "BTCUSDT",
          breakdown: breakdown(),
          execution: {
            trades: 2,
            makerTrades: 1,
            takerTrades: 1,
            unknownLiquidityTrades: 0,
            makerRate: 0.5,
            quoteNotional: 1_000,
          },
        },
      ],
      unclassifiedIncomeTypes: [],
      ...overrides,
    },
  };
}

test("revenue Markdown is client-ready and preserves accounting authority", async () => {
  const report = await loadReportModule();
  const markdown = report.buildRevenueMarkdown(
    response(),
    {
      clientName: "Example | Capital",
      strategyName: "Momentum",
      deploymentName: "Primary",
      reviewedCommit: "abc123",
    },
    1_704_153_600_000,
  );
  assert.match(markdown, /Status: \*\*COMPLETE\*\*/);
  assert.match(markdown, /Client: Example \\| Capital/);
  assert.match(markdown, /Exchange income history is the realized accounting authority/);
  assert.match(markdown, /\| Net revenue \| 97 \|/);
  assert.match(markdown, /Maker rate: 50%/);
  assert.doesNotMatch(markdown, /binanceApiSecret|binanceApiKey/);
});

test("revenue report flags incomplete and unclassified evidence", async () => {
  const report = await loadReportModule();
  const input = response({
    incomeMayBeTruncated: true,
    tradesMayBeTruncated: true,
    breakdown: breakdown({ unclassified: 20 }),
    unclassifiedIncomeTypes: ["NEW_REWARD"],
  });
  const markdown = report.buildRevenueMarkdown(input, {}, 1_704_153_600_000);
  assert.match(markdown, /Status: \*\*REVIEW REQUIRED\*\*/);
  assert.match(markdown, /Income history reached the request limit/);
  assert.match(markdown, /Unclassified income types were excluded from net revenue: NEW_REWARD/);
});

test("revenue exports are deterministic, typed, and spreadsheet-safe", async () => {
  const report = await loadReportModule();
  const input = response();
  input.ledger.symbols[0].symbol = 'BTC,"USDT';
  const metadata = { clientName: "Example Capital" };
  const json = JSON.parse(report.buildRevenueReportJson(input, metadata, 123));
  const dailyCsv = report.buildRevenueDailyCsv(input);
  const symbolsCsv = report.buildRevenueSymbolsCsv(input);
  assert.equal(json.schemaVersion, 1);
  assert.equal(json.reportType, "strategy-assurance-revenue");
  assert.equal(json.generatedAtMs, 123);
  assert.equal(json.metadata.clientName, "Example Capital");
  assert.match(dailyCsv, /^utcDay,asset,realizedPnl/);
  assert.match(symbolsCsv, /^symbol,asset,realizedPnl/);
  assert.match(symbolsCsv, /^"BTC,""USDT",USDT/m);
  assert.equal(
    report.revenueReportBaseName(input, metadata),
    "strategy-assurance-revenue-example-capital-usdt-2024-01-01-2024-01-01",
  );
});
