import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import test from "node:test";

test("exposure RL path: shield, causality, exact ledger, learner mechanics, toy problems and artifacts", () => {
  const result = spawnSync("python3", ["test/exposure_rl_test.py"], {
    encoding: "utf8",
    timeout: 900_000,
  });
  assert.equal(result.status, 0, `${result.stdout}\n${result.stderr}`);
});
