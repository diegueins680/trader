import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import test from "node:test";

test("causal OPE composition: frozen-free imports, exact estimates, bounded sampling-law gap, no dropped runs and all-or-nothing publication", () => {
  const result = spawnSync("python3", ["test/causal_ope_v8_test.py"], {
    encoding: "utf8",
    timeout: 120_000,
  });
  assert.equal(result.status, 0, `${result.stdout}\n${result.stderr}`);
});
