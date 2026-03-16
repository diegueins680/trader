import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const cabalConfig = readFileSync(new URL("../../../.cabal/config", import.meta.url), "utf8");
const frontendConstantsSource = readFileSync(new URL("../src/app/constants.ts", import.meta.url), "utf8");
const backendBinanceIntervalsSource = readFileSync(new URL("../../app/Trader/BinanceIntervals.hs", import.meta.url), "utf8");

function parseStringLiterals(source) {
  return Array.from(source.matchAll(/"([^"]+)"/g), ([, value]) => value);
}

function parseTsConstStringArray(source, constName) {
  const match = source.match(new RegExp(`export const ${constName} = \\[(.*?)\\] as const;`, "s"));
  assert.ok(match, `expected ${constName} const array in frontend constants`);
  return parseStringLiterals(match[1]);
}

function parseTsConstObjectKeys(source, constName) {
  const match = source.match(new RegExp(`export const ${constName}[^=]*= \\{(.*?)\\};`, "s"));
  assert.ok(match, `expected ${constName} const object in frontend constants`);
  return Array.from(match[1].matchAll(/"([^"]+)":/g), ([, value]) => value);
}

function parseHsStringList(source, bindingName) {
  const match = source.match(
    new RegExp(`${bindingName}\\s*=\\s*\\n(.*?)(?=\\n[a-zA-Z_][\\w']*\\s*::|\\n[a-zA-Z_][\\w']*\\s*=|$)`, "s"),
  );
  assert.ok(match, `expected ${bindingName} binding in backend BinanceIntervals module`);
  return parseStringLiterals(match[1]);
}

function assertUnique(values, label) {
  assert.equal(new Set(values).size, values.length, `${label} must stay duplicate-free`);
}

test("tracked cabal config leaves machine-specific path overrides disabled", () => {
  const activeLines = cabalConfig
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line && !line.startsWith("--"));

  for (const key of ["remote-repo-cache", "build-summary", "installdir"]) {
    assert.equal(activeLines.some((line) => line.startsWith(`${key}:`)), false);
  }
});

test("repo contract keeps frontend Binance intervals aligned with backend validation", () => {
  const frontendBinanceIntervals = parseTsConstStringArray(frontendConstantsSource, "BINANCE_INTERVALS");
  const frontendBinanceIntervalSecondsKeys = parseTsConstObjectKeys(frontendConstantsSource, "BINANCE_INTERVAL_SECONDS");
  const backendBinanceIntervals = parseHsStringList(backendBinanceIntervalsSource, "binanceIntervals");

  assertUnique(frontendBinanceIntervals, "frontend BINANCE_INTERVALS");
  assertUnique(backendBinanceIntervals, "backend binanceIntervals");
  assert.deepEqual(frontendBinanceIntervals, backendBinanceIntervals);
  assert.deepEqual(frontendBinanceIntervalSecondsKeys, frontendBinanceIntervals);
});