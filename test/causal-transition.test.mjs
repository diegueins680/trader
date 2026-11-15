import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import test from "node:test";

test("causal transition adapter: exact receipts, support, termination and immutable rejection", () => {
  const result = spawnSync("python3", ["test/causal_transition_test.py"], {
    encoding: "utf8",
    timeout: 120_000,
  });
  assert.equal(result.status, 0, `${result.stdout}\n${result.stderr}`);
});
