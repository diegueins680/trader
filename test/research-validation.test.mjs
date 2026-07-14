import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const RESEARCH_DIR = path.join(ROOT, "scripts", "research");
const HAS_RESEARCH_PYTHON =
  spawnSync("python3", ["-c", "import numpy, pandas"], { encoding: "utf8" }).status === 0;

function runPython(program) {
  const source = `
import json
import sys

sys.path.insert(0, sys.argv[1])
import harness as H

${program}
`;
  const run = spawnSync("python3", ["-c", source, RESEARCH_DIR], {
    cwd: ROOT,
    encoding: "utf8",
  });
  assert.equal(run.status, 0, run.stderr || run.stdout);
  return JSON.parse(run.stdout);
}

test(
  "rolling-origin splits expand training, derive the embargo, and retain the final partial fold",
  { skip: !HAS_RESEARCH_PYTHON },
  () => {
    const splits = runPython(`
from dataclasses import asdict

splits = H.rolling_origin_splits(
    n_obs=18,
    initial_train_size=5,
    test_size=4,
    label_horizon=2,
)
print(json.dumps([asdict(split) for split in splits]))
`);

    assert.deepEqual(splits, [
      {
        fold: 0,
        train_start: 0,
        train_stop: 5,
        embargo_start: 5,
        embargo_stop: 7,
        test_start: 7,
        test_stop: 11,
      },
      {
        fold: 1,
        train_start: 0,
        train_stop: 9,
        embargo_start: 9,
        embargo_stop: 11,
        test_start: 11,
        test_stop: 15,
      },
      {
        fold: 2,
        train_start: 0,
        train_stop: 13,
        embargo_start: 13,
        embargo_stop: 15,
        test_start: 15,
        test_stop: 18,
      },
    ]);

    const coveredRows = splits.flatMap(({ test_start: start, test_stop: stop }) =>
      Array.from({ length: stop - start }, (_, offset) => start + offset),
    );
    assert.deepEqual(coveredRows, Array.from({ length: 11 }, (_, offset) => offset + 7));
  },
);

test(
  "nested validation selects on inner rows and evaluates the frozen winner once out of sample",
  { skip: !HAS_RESEARCH_PYTHON },
  () => {
    const output = runPython(`
import pandas as pd

frame = pd.DataFrame({
    "openTime": range(100, 118),
    "slow": [0.0] * 18,
    "fast": [0.0] * 18,
})
# Inner validation prefers slow. The untouched outer test strongly prefers fast,
# which must not affect the selected candidate.
frame.loc[6:11, "slow"] = 1.0
frame.loc[14:17, "slow"] = -10.0
frame.loc[14:17, "fast"] = 10.0

calls = []

def fit_candidate(candidate, train):
    calls.append({
        "kind": "fit",
        "candidate": candidate,
        "start": int(train.index.min()),
        "stop": int(train.index.max()) + 1,
    })
    return candidate

def evaluate_candidate(candidate, evaluation):
    calls.append({
        "kind": "evaluate",
        "candidate": candidate,
        "start": int(evaluation.index.min()),
        "stop": int(evaluation.index.max()) + 1,
    })
    return pd.DataFrame({"net": evaluation[candidate].to_numpy()})

result = H.nested_rolling_origin(
    frame,
    {"slow": "slow", "fast": "fast"},
    fit_candidate,
    evaluate_candidate,
    lambda validation: validation["net"].mean(),
    initial_train_size=12,
    outer_test_size=5,
    inner_initial_train_size=4,
    inner_test_size=3,
    label_horizon=2,
)
print(json.dumps({
    "oos": result.oos.to_dict("records"),
    "outer": result.outer_folds.to_dict("records"),
    "inner": result.inner_scores.to_dict("records"),
    "calls": calls,
}))
`);

    assert.equal(output.outer.length, 1);
    assert.deepEqual(output.outer[0], {
      outer_fold: 0,
      train_start: 0,
      train_stop: 12,
      embargo_start: 12,
      embargo_stop: 14,
      test_start: 14,
      test_stop: 18,
      selected_candidate: "slow",
      selection_score: 1,
      inner_folds: 2,
    });
    assert.deepEqual(
      output.inner.map(({ candidate, score, inner_folds, validation_rows }) => ({
        candidate,
        score,
        inner_folds,
        validation_rows,
      })),
      [
        { candidate: "slow", score: 1, inner_folds: 2, validation_rows: 6 },
        { candidate: "fast", score: 0, inner_folds: 2, validation_rows: 6 },
      ],
    );
    assert.deepEqual(
      output.oos.map(({ row_position, openTime, selected_candidate, net }) => ({
        row_position,
        openTime,
        selected_candidate,
        net,
      })),
      Array.from({ length: 4 }, (_, offset) => ({
        row_position: offset + 14,
        openTime: offset + 114,
        selected_candidate: "slow",
        net: -10,
      })),
    );

    const outerFits = output.calls.filter(
      ({ kind, start, stop }) => kind === "fit" && start === 0 && stop === 12,
    );
    const outerEvaluations = output.calls.filter(
      ({ kind, start, stop }) => kind === "evaluate" && start === 14 && stop === 18,
    );
    assert.deepEqual(outerFits, [
      { kind: "fit", candidate: "slow", start: 0, stop: 12 },
    ]);
    assert.deepEqual(outerEvaluations, [
      { kind: "evaluate", candidate: "slow", start: 14, stop: 18 },
    ]);

    const evaluationCalls = output.calls.filter(({ kind }) => kind === "evaluate");
    for (const evaluation of evaluationCalls) {
      const matchingFit = output.calls.find(
        ({ kind, candidate, stop }) =>
          kind === "fit" && candidate === evaluation.candidate && stop === evaluation.start - 2,
      );
      assert.ok(matchingFit, `missing two-row embargo before ${JSON.stringify(evaluation)}`);
    }
  },
);
