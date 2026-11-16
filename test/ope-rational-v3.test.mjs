import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import test from "node:test";

test("exact OPE v3: predecessor-equal arithmetic, full registered capacity, extreme values and all-or-nothing publication", () => {
  const result = spawnSync("python3", ["test/ope_rational_v3_test.py"], {
    encoding: "utf8",
    timeout: 120_000,
  });
  assert.equal(result.status, 0, `${result.stdout}\n${result.stderr}`);
});
