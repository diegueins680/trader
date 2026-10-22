import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import test from "node:test";

test("causal PPO composition: nine inputs, checked conversion, immutable updates and replay evidence", () => {
  const result = spawnSync("python3", ["test/causal_ppo_test.py"], {
    encoding: "utf8",
    timeout: 120_000,
  });
  assert.equal(result.status, 0, `${result.stdout}\n${result.stderr}`);
});
