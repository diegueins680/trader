import assert from "node:assert/strict";
import { test } from "node:test";

import {
  binanceTradeKey,
  buildBinanceTradeIpMap,
  findOptionalWholeNumberFieldError,
  parseMaybeInt,
  parseOptionalInt,
  parseTimeInputMs,
} from "../.tmp/web-tests/appHelpers.js";

function mkTrade({
  symbol = "FILUSDT",
  tradeId,
  orderId,
  side,
  positionSide = "BOTH",
  qty,
  time,
  originIp,
  executorIp,
  originInstance,
  realizedPnl = 0,
}) {
  return {
    symbol,
    tradeId,
    orderId,
    price: 1,
    qty,
    quoteQty: qty,
    time,
    side,
    positionSide,
    realizedPnl,
    originIp,
    executorIp,
    originInstance,
  };
}

test("buildBinanceTradeIpMap propagates close meta back onto the opening trade", () => {
  const t0 = 1_000;
  const t1 = 2_000;
  const trades = [
    mkTrade({ tradeId: 1, orderId: 101, side: "BUY", qty: 10, time: t0, executorIp: "1.2.3.4", originInstance: "laptop" }),
    mkTrade({ tradeId: 2, orderId: 102, side: "SELL", qty: 10, time: t1, executorIp: "5.6.7.8", originInstance: "fly", realizedPnl: 1 }),
  ];

  const meta = buildBinanceTradeIpMap(trades);
  const openKey = binanceTradeKey(trades[0]);
  const closeKey = binanceTradeKey(trades[1]);

  assert.deepEqual(meta.get(openKey), {
    entryIp: "1.2.3.4",
    exitIp: "5.6.7.8",
    entryInstance: "laptop",
    exitInstance: "fly",
    entryTime: t0,
    exitTime: t1,
  });
  assert.deepEqual(meta.get(closeKey), {
    entryIp: "1.2.3.4",
    exitIp: "5.6.7.8",
    entryInstance: "laptop",
    exitInstance: "fly",
    entryTime: t0,
    exitTime: t1,
  });
});

test("buildBinanceTradeIpMap waits until a lot is fully closed before backfilling exit meta", () => {
  const t0 = 1_000;
  const t1 = 2_000;
  const t2 = 3_000;
  const trades = [
    mkTrade({ tradeId: 1, orderId: 101, side: "BUY", qty: 10, time: t0, executorIp: "1.1.1.1" }),
    mkTrade({ tradeId: 2, orderId: 102, side: "SELL", qty: 6, time: t1, executorIp: "2.2.2.2", realizedPnl: 1 }),
    mkTrade({ tradeId: 3, orderId: 103, side: "SELL", qty: 4, time: t2, executorIp: "3.3.3.3", realizedPnl: 1 }),
  ];

  const meta = buildBinanceTradeIpMap(trades);
  const openKey = binanceTradeKey(trades[0]);

  assert.deepEqual(meta.get(openKey), {
    entryIp: "1.1.1.1",
    exitIp: "3.3.3.3",
    entryInstance: null,
    exitInstance: null,
    entryTime: t0,
    exitTime: t2,
  });
});

test("buildBinanceTradeIpMap aggregates entry IPs when multiple opening lots are closed together", () => {
  const t0 = 1_000;
  const t1 = 1_500;
  const t2 = 2_000;
  const trades = [
    mkTrade({ tradeId: 1, orderId: 101, side: "BUY", qty: 5, time: t0, executorIp: "1.1.1.1", originInstance: "fly" }),
    mkTrade({ tradeId: 2, orderId: 102, side: "BUY", qty: 5, time: t1, executorIp: "4.4.4.4", originInstance: "hetzner" }),
    mkTrade({ tradeId: 3, orderId: 103, side: "SELL", qty: 10, time: t2, executorIp: "9.9.9.9", originInstance: "laptop", realizedPnl: 1 }),
  ];

  const meta = buildBinanceTradeIpMap(trades);
  const openKey1 = binanceTradeKey(trades[0]);
  const openKey2 = binanceTradeKey(trades[1]);
  const closeKey = binanceTradeKey(trades[2]);

  assert.deepEqual(meta.get(openKey1), {
    entryIp: "1.1.1.1",
    exitIp: "9.9.9.9",
    entryInstance: "fly",
    exitInstance: "laptop",
    entryTime: t0,
    exitTime: t2,
  });
  assert.deepEqual(meta.get(openKey2), {
    entryIp: "4.4.4.4",
    exitIp: "9.9.9.9",
    entryInstance: "hetzner",
    exitInstance: "laptop",
    entryTime: t1,
    exitTime: t2,
  });
  assert.deepEqual(meta.get(closeKey), {
    entryIp: "1.1.1.1 • 4.4.4.4",
    exitIp: "9.9.9.9",
    entryInstance: "fly • hetzner",
    exitInstance: "laptop",
    entryTime: t0,
    exitTime: t2,
  });
});

test("buildBinanceTradeIpMap does not let an unpaired one-way close create a phantom lot", () => {
  const historicalClose = mkTrade({
    tradeId: 1,
    orderId: 101,
    side: "SELL",
    qty: 2,
    time: 1_000,
    executorIp: "1.1.1.1",
    originInstance: "fly",
    realizedPnl: 6,
  });
  const visibleOpen = mkTrade({
    tradeId: 2,
    orderId: 102,
    side: "BUY",
    qty: 1,
    time: 2_000,
    executorIp: "2.2.2.2",
    originInstance: "laptop",
  });
  const visibleClose = mkTrade({
    tradeId: 3,
    orderId: 103,
    side: "SELL",
    qty: 1,
    time: 3_000,
    executorIp: "3.3.3.3",
    originInstance: "hetzner",
    realizedPnl: 10,
  });

  const meta = buildBinanceTradeIpMap([historicalClose, visibleOpen, visibleClose]);

  assert.deepEqual(meta.get(binanceTradeKey(historicalClose)), {
    entryIp: null,
    exitIp: "1.1.1.1",
    entryInstance: null,
    exitInstance: "fly",
    entryTime: null,
    exitTime: 1_000,
  });
  assert.deepEqual(meta.get(binanceTradeKey(visibleOpen)), {
    entryIp: "2.2.2.2",
    exitIp: "3.3.3.3",
    entryInstance: "laptop",
    exitInstance: "hetzner",
    entryTime: 2_000,
    exitTime: 3_000,
  });
  assert.deepEqual(meta.get(binanceTradeKey(visibleClose)), {
    entryIp: "2.2.2.2",
    exitIp: "3.3.3.3",
    entryInstance: "laptop",
    exitInstance: "hetzner",
    entryTime: 2_000,
    exitTime: 3_000,
  });
});

test("buildBinanceTradeIpMap ignores requester origin IP when executor IP is absent", () => {
  const trades = [
    mkTrade({ tradeId: 1, orderId: 101, side: "BUY", qty: 1, time: 1_000, originIp: "192.0.2.10", originInstance: "hetzner" }),
    mkTrade({ tradeId: 2, orderId: 102, side: "SELL", qty: 1, time: 2_000, originIp: "192.0.2.10", originInstance: "hetzner", realizedPnl: -1 }),
  ];

  const meta = buildBinanceTradeIpMap(trades);

  assert.deepEqual(meta.get(binanceTradeKey(trades[0])), {
    entryIp: null,
    exitIp: null,
    entryInstance: "hetzner",
    exitInstance: "hetzner",
    entryTime: 1_000,
    exitTime: 2_000,
  });
});

test("parseOptionalInt accepts whole numbers and rejects fractional values", () => {
  assert.equal(parseOptionalInt("123"), 123);
  assert.equal(parseOptionalInt("1,234"), undefined);
  assert.equal(parseOptionalInt("12.5"), undefined);
  assert.equal(parseOptionalInt("0,5"), undefined);
});

test("parseMaybeInt preserves the non-negative whole-number contract", () => {
  assert.equal(parseMaybeInt("123"), 123);
  assert.equal(parseMaybeInt("12.0"), 12);
  assert.equal(parseMaybeInt("12.5"), null);
  assert.equal(parseMaybeInt("-0.5"), null);
  assert.equal(parseMaybeInt("-1"), null);
});

test("whole-number parsers reject unsafe integers instead of rounding them", () => {
  const unsafe = (BigInt(Number.MAX_SAFE_INTEGER) + 1n).toString();
  assert.equal(parseOptionalInt(unsafe), undefined);
  assert.equal(parseMaybeInt(unsafe), null);
  assert.equal(parseTimeInputMs(unsafe), null);
});

test("parseTimeInputMs rejects impossible ISO calendar dates instead of rolling them forward", () => {
  assert.equal(parseTimeInputMs("2024-02-29"), Date.parse("2024-02-29T00:00:00Z"));
  assert.equal(parseTimeInputMs("2025-02-30"), null);
  assert.equal(parseTimeInputMs("2025-02-30T00:00:00Z"), null);
  assert.equal(parseTimeInputMs("2025-01-01T24:01"), null);
  assert.equal(parseTimeInputMs("2025-01-01T23:59:59+02:30"), Date.parse("2025-01-01T23:59:59+02:30"));
});

test("findOptionalWholeNumberFieldError reports invalid form and override values", () => {
  assert.equal(
    findOptionalWholeNumberFieldError([{ label: "Trials", raw: "12.5" }]),
    "Trials must be a whole number.",
  );
  assert.equal(
    findOptionalWholeNumberFieldError([
      { label: "Seed", raw: (BigInt(Number.MAX_SAFE_INTEGER) + 1n).toString() },
    ]),
    "Seed must be a whole number.",
  );
  assert.equal(
    findOptionalWholeNumberFieldError([{ label: "Bars min", raw: "", override: 10.25 }]),
    "Bars min must be a whole number.",
  );
  assert.equal(
    findOptionalWholeNumberFieldError([{ label: "Seed", raw: "100" }, { label: "Bars max", raw: "", override: 500 }]),
    null,
  );
});
