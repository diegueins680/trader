import assert from "node:assert/strict";
import { test } from "node:test";
import {
  buildDefaultOptimizerRunForm,
  buildOptimizerRunRequest,
  findOptionalWholeNumberFieldError,
  formatDatetimeLocal,
  parseDatetimeLocal,
  positionSideInfo,
  readExactSafeInteger,
  readNonNegativeExactSafeInteger,
  sanitizeOptimizationComboOperation,
} from "../.tmp/web-tests/appHelpers.js";

function pad2(value) {
  return String(value).padStart(2, "0");
}

function isLeapYear(year) {
  return (year % 4 === 0 && year % 100 !== 0) || year % 400 === 0;
}

function daysInMonth(year, month) {
  switch (month) {
    case 2:
      return isLeapYear(year) ? 29 : 28;
    case 4:
    case 6:
    case 9:
    case 11:
      return 30;
    default:
      return 31;
  }
}

test("parseDatetimeLocal accepts exactly the valid month-end local timestamps in a bounded matrix", () => {
  const years = [2023, 2024];
  const days = [28, 29, 30, 31];

  for (const year of years) {
    for (let month = 1; month <= 12; month += 1) {
      for (const day of days) {
        const raw = `${year}-${pad2(month)}-${pad2(day)}T12:34`;
        const parsed = parseDatetimeLocal(raw);
        const valid = day <= daysInMonth(year, month);
        assert.equal(parsed !== null, valid, `${raw}: expected calendar validity ${valid}`);
        if (valid) {
          assert.equal(
            formatDatetimeLocal(parsed),
            raw,
            `${raw}: expected parse/format to preserve valid local timestamps`,
          );
        }
      }
    }
  }
});

test("parseDatetimeLocal rejects impossible local timestamps that Date.parse would otherwise normalize", () => {
  assert.equal(parseDatetimeLocal("2024-02-31T12:34"), null);
  assert.equal(parseDatetimeLocal("2023-04-31T00:00"), null);
  assert.equal(parseDatetimeLocal("2024-02-29 12:34"), parseDatetimeLocal("2024-02-29T12:34"));
});

test("formatDatetimeLocal suppresses finite out-of-range timestamps instead of rendering NaN fragments", () => {
  assert.equal(formatDatetimeLocal(1e20), "");
  assert.equal(formatDatetimeLocal(-1e20), "");
});

test("readExactSafeInteger accepts only exact safe integers from combo payloads", () => {
  const unsafe = Number.MAX_SAFE_INTEGER + 1;
  const cases = [
    { raw: 0, expected: 0 },
    { raw: -7, expected: -7 },
    { raw: 12.5, expected: null },
    { raw: Number.NaN, expected: null },
    { raw: Number.POSITIVE_INFINITY, expected: null },
    { raw: unsafe, expected: null },
    { raw: "9", expected: null },
  ];

  for (const { raw, expected } of cases) {
    assert.equal(
      readExactSafeInteger(raw),
      expected,
      `expected ${String(raw)} to normalize to ${String(expected)}`,
    );
  }
});

test("readNonNegativeExactSafeInteger preserves zero and rejects negative or inexact discrete payloads", () => {
  const unsafe = Number.MAX_SAFE_INTEGER + 1;
  const cases = [
    { raw: 0, expected: 0 },
    { raw: 7, expected: 7 },
    { raw: -1, expected: null },
    { raw: 2.5, expected: null },
    { raw: Number.NaN, expected: null },
    { raw: unsafe, expected: null },
  ];

  for (const { raw, expected } of cases) {
    assert.equal(
      readNonNegativeExactSafeInteger(raw),
      expected,
      `expected ${String(raw)} to normalize to ${String(expected)}`,
    );
  }
});

test("findOptionalWholeNumberFieldError accepts exact string overrides and rejects fractional string overrides", () => {
  assert.equal(
    findOptionalWholeNumberFieldError([{ label: "Trials", raw: "", override: " 12 " }]),
    null,
  );
  assert.equal(
    findOptionalWholeNumberFieldError([{ label: "Trials", raw: "", override: "12.5" }]),
    "Trials must be a whole number.",
  );
});

test("buildOptimizerRunRequest normalizes known integer extra overrides before merging", () => {
  const form = {
    ...buildDefaultOptimizerRunForm("BTCUSDT", "binance"),
    trials: "",
    barsMin: "",
  };
  const request = buildOptimizerRunRequest(form, {
    trials: "12",
    barsMin: "250",
    objective: "roi",
  });

  assert.equal(request.trials, 12);
  assert.equal(request.barsMin, 250);
  assert.equal(request.objective, "roi");
  assert.equal(typeof request.trials, "number");
  assert.equal(typeof request.barsMin, "number");
});

test("buildOptimizerRunRequest drops invalid known integer extra overrides instead of emitting stringly payloads", () => {
  const unsafe = (BigInt(Number.MAX_SAFE_INTEGER) + 1n).toString();
  const form = {
    ...buildDefaultOptimizerRunForm("BTCUSDT", "binance"),
    trials: "",
    trendLookbackMin: "",
  };
  const request = buildOptimizerRunRequest(form, {
    trials: "12.5",
    trendLookbackMin: unsafe,
  });

  assert.equal("trials" in request, false);
  assert.equal("trendLookbackMin" in request, false);
});

test("sanitizeOptimizationComboOperation preserves only exact non-negative discrete operation coordinates", () => {
  assert.deepEqual(
    sanitizeOptimizationComboOperation({
      entryIndex: 3,
      exitIndex: 8,
      entryEquity: 1000,
      exitEquity: 1025,
      return: 0.025,
      holdingPeriods: 5,
      exitReason: " target ",
    }),
    {
      entryIndex: 3,
      exitIndex: 8,
      entryEquity: 1000,
      exitEquity: 1025,
      return: 0.025,
      holdingPeriods: 5,
      exitReason: "target",
    },
  );
  assert.equal(sanitizeOptimizationComboOperation({ entryIndex: 1.5, exitIndex: 4 }), null);
  assert.equal(sanitizeOptimizationComboOperation({ entryIndex: 4, exitIndex: 3 }), null);
  assert.deepEqual(
    sanitizeOptimizationComboOperation({ entryIndex: 1, exitIndex: 4, holdingPeriods: 2.5 }),
    {
      entryIndex: 1,
      exitIndex: 4,
      entryEquity: null,
      exitEquity: null,
      return: null,
      holdingPeriods: null,
      exitReason: null,
    },
  );
});

test("positionSideInfo treats zero/dust amounts as flat before stale side metadata", () => {
  assert.deepEqual(positionSideInfo(0, "LONG"), { dir: 0, label: "FLAT", key: "FLAT" });
  assert.deepEqual(positionSideInfo(1e-13, "SHORT"), { dir: 0, label: "FLAT", key: "FLAT" });
  assert.deepEqual(positionSideInfo(2, "SHORT"), { dir: -1, label: "SHORT", key: "SHORT" });
});

test("positionSideInfo suppresses non-finite amounts before stale side metadata", () => {
  assert.deepEqual(positionSideInfo(Number.NaN, "LONG"), { dir: 0, label: "FLAT", key: "FLAT" });
  assert.deepEqual(positionSideInfo(Number.POSITIVE_INFINITY, "SHORT"), { dir: 0, label: "FLAT", key: "FLAT" });
});
