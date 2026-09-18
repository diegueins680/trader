import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import test from "node:test";

test("offline sequential contracts: causality, accounting, gradients, OPE and non-authorizing safety", () => {
  const result = spawnSync("python3", ["test/sequential_screen_test.py"], {
    encoding: "utf8",
    timeout: 120_000,
  });
  assert.equal(result.status, 0, `${result.stdout}\n${result.stderr}`);
});
