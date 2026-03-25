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
  };
}

test("buildBinanceTradeIpMap propagates close meta back onto the opening trade", () => {
  const t0 = 1_000;
  const t1 = 2_000;
  const trades = [
    mkTrade({ tradeId: 1, orderId: 101, side: "BUY", qty: 10, time: t0, originIp: "1.2.3.4" }),
    mkTrade({ tradeId: 2, orderId: 102, side: "SELL", qty: 10, time: t1, originIp: "5.6.7.8", realizedPnl: 1 }),
  ];

  const meta = buildBinanceTradeIpMap(trades);
  const openKey = binanceTradeKey(trades[0]);
  const closeKey = binanceTradeKey(trades[1]);

  assert.deepEqual(meta.get(openKey), { entryIp: "1.2.3.4", exitIp: "5.6.7.8", entryTime: t0, exitTime: t1 });
  assert.deepEqual(meta.get(closeKey), { entryIp: "1.2.3.4", exitIp: "5.6.7.8", entryTime: t0, exitTime: t1 });
});

test("buildBinanceTradeIpMap waits until a lot is fully closed before backfilling exit meta", () => {
  const t0 = 1_000;
  const t1 = 2_000;
  const t2 = 3_000;
  const trades = [
    mkTrade({ tradeId: 1, orderId: 101, side: "BUY", qty: 10, time: t0, originIp: "1.1.1.1" }),
    mkTrade({ tradeId: 2, orderId: 102, side: "SELL", qty: 6, time: t1, originIp: "2.2.2.2", realizedPnl: 1 }),
    mkTrade({ tradeId: 3, orderId: 103, side: "SELL", qty: 4, time: t2, originIp: "3.3.3.3", realizedPnl: 1 }),
  ];

  const meta = buildBinanceTradeIpMap(trades);
  const openKey = binanceTradeKey(trades[0]);

  assert.deepEqual(meta.get(openKey), { entryIp: "1.1.1.1", exitIp: "3.3.3.3", entryTime: t0, exitTime: t2 });
});

test("buildBinanceTradeIpMap aggregates entry IPs when multiple opening lots are closed together", () => {
  const t0 = 1_000;
  const t1 = 1_500;
  const t2 = 2_000;
  const trades = [
    mkTrade({ tradeId: 1, orderId: 101, side: "BUY", qty: 5, time: t0, originIp: "1.1.1.1" }),
    mkTrade({ tradeId: 2, orderId: 102, side: "BUY", qty: 5, time: t1, originIp: "4.4.4.4" }),
    mkTrade({ tradeId: 3, orderId: 103, side: "SELL", qty: 10, time: t2, originIp: "9.9.9.9", realizedPnl: 1 }),
  ];

  const meta = buildBinanceTradeIpMap(trades);
  const openKey1 = binanceTradeKey(trades[0]);
  const openKey2 = binanceTradeKey(trades[1]);
  const closeKey = binanceTradeKey(trades[2]);

  assert.deepEqual(meta.get(openKey1), { entryIp: "1.1.1.1", exitIp: "9.9.9.9", entryTime: t0, exitTime: t2 });
  assert.deepEqual(meta.get(openKey2), { entryIp: "4.4.4.4", exitIp: "9.9.9.9", entryTime: t1, exitTime: t2 });
  assert.deepEqual(meta.get(closeKey), { entryIp: "1.1.1.1 • 4.4.4.4", exitIp: "9.9.9.9", entryTime: t0, exitTime: t2 });
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
    findOptionalWholeNumberFieldError([{ label: "Bars min", raw: "", override: 10.25 }]),
    "Bars min must be a whole number.",
  );
  assert.equal(
    findOptionalWholeNumberFieldError([{ label: "Seed", raw: "100" }, { label: "Bars max", raw: "", override: 500 }]),
    null,
  );
});
