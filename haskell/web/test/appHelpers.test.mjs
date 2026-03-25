import assert from "node:assert/strict";
import { test } from "node:test";
import { formatDatetimeLocal, parseDatetimeLocal } from "../.tmp/web-tests/appHelpers.js";

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
