import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const cabalConfig = readFileSync(new URL("../../../.cabal/config", import.meta.url), "utf8");

test("tracked cabal config leaves machine-specific path overrides disabled", () => {
  const activeLines = cabalConfig
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line && !line.startsWith("--"));

  for (const key of ["remote-repo-cache", "build-summary", "installdir"]) {
    assert.equal(activeLines.some((line) => line.startsWith(`${key}:`)), false);
  }
});
