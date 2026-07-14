import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import test from "node:test";
import { fileURLToPath } from "node:url";

const RESEARCH_DIR = fileURLToPath(new URL("../scripts/research/", import.meta.url));
const PYTHON_RESEARCH_DEPS =
  spawnSync("python3", ["-c", "import numpy, pandas"], { encoding: "utf8" }).status === 0;

function runResearchPython(program) {
  const run = spawnSync("python3", ["-c", program, RESEARCH_DIR], {
    encoding: "utf8",
  });
  assert.equal(run.status, 0, run.stderr);
  return JSON.parse(run.stdout);
}

test(
  "formal DSR matches a fixed cross-trial benchmark and non-normality calculation",
  { skip: !PYTHON_RESEARCH_DEPS },
  () => {
    const result = runResearchPython(String.raw`
import json
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, sys.argv[1])
import diagnostics

base = np.array([
    -0.020, -0.010, 0.000, 0.010, 0.020, -0.015,
     0.005,  0.012, -0.008, 0.018, -0.004, 0.009,
])
matrix = pd.DataFrame(
    {
        "selected": base + 0.008,
        "flat": base,
        "weak": base + 0.002,
        "inverse": -base,
    },
    index=pd.Index(range(len(base)), name="timestamp"),
)
formal = diagnostics.deflated_sharpe_ratio(
    matrix,
    selected_trial="selected",
    periods_per_year=365,
)
one_trial_null = diagnostics.deflated_sharpe_ratio(
    matrix,
    selected_trial="selected",
    independent_trials=1,
)
print(json.dumps({
    "formal": formal.to_dict(),
    "oneTrialProbability": one_trial_null.probability,
}))
`);

    assert.equal(result.formal.method, "deflated_sharpe_ratio");
    assert.equal(result.formal.selectedTrial, "selected");
    assert.equal(result.formal.observations, 12);
    assert.equal(result.formal.trials, 4);
    assert.ok(Math.abs(result.formal.selectedSharpePerPeriod - 0.7251558223238601) < 1e-10);
    assert.ok(Math.abs(result.formal.benchmarkSharpePerPeriod - 0.3718395168109235) < 1e-10);
    assert.ok(Math.abs(result.formal.selectedReturnSkewness - -0.17209883347493737) < 1e-10);
    assert.ok(
      Math.abs(result.formal.selectedReturnPearsonKurtosis - 1.8763104841710696) < 1e-10,
    );
    assert.ok(Math.abs(result.formal.probability - 0.8536743108994668) < 1e-10);
    assert.ok(result.formal.probability < result.oneTrialProbability);
  },
);

test(
  "CSCV PBO separates a stable winner from block-specific in-sample winners",
  { skip: !PYTHON_RESEARCH_DEPS },
  () => {
    const result = runResearchPython(String.raw`
import json
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, sys.argv[1])
import diagnostics

index = pd.Index(range(8), name="timestamp")
stable = pd.DataFrame({
    "edge": np.tile([0.02, 0.01], 4),
    "loser": np.tile([-0.01, -0.02], 4),
}, index=index)
block_specific = np.array([0.03, 0.01, 0.03, 0.01, -0.01, -0.03, -0.01, -0.03])
overfit = pd.DataFrame({
    "first_half": block_specific,
    "second_half": -block_specific,
}, index=index)

stable_result = diagnostics.cscv_pbo(stable, n_slices=4)
overfit_result = diagnostics.cscv_pbo(overfit, n_slices=4)
print(json.dumps({
    "stable": stable_result.to_dict(),
    "overfit": overfit_result.to_dict(),
}))
`);

    assert.equal(result.stable.method, "cscv_probability_of_backtest_overfitting");
    assert.equal(result.stable.splits, 6);
    assert.equal(result.stable.sliceSize, 2);
    assert.equal(result.stable.probability, 0);
    assert.ok(result.stable.logits.every((value) => value > 0));
    assert.ok(
      result.stable.selectedInSampleSharpes.every(
        (value, index) => value > result.stable.selectedOutOfSampleSharpes[index] - 1e-12,
      ),
    );
    assert.equal(result.overfit.probability, 1);
    assert.ok(result.overfit.logits.every((value) => value <= 0));
  },
);

test(
  "return compounding keeps the latest full blocks and records exact source boundaries",
  { skip: !PYTHON_RESEARCH_DEPS },
  () => {
    const result = runResearchPython(String.raw`
import json
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, sys.argv[1])
import diagnostics

values = np.arange(1.0, 12.0) / 100.0
source = pd.DataFrame(
    {"a": values, "b": -values / 2.0},
    index=pd.Index(np.arange(100, 111), name="openTime"),
)
compounded = diagnostics.compound_return_matrix(source, 4)

changed_leading_rows = source.copy()
changed_leading_rows.iloc[:3] = 10.0
changed = diagnostics.compound_return_matrix(changed_leading_rows, 4)

errors = []
for invalid_size in (True, 0, 12):
    try:
        diagnostics.compound_return_matrix(source, invalid_size)
    except (TypeError, ValueError) as error:
        errors.append(str(error))

incomplete = source.copy()
incomplete.iloc[5, 0] = np.nan
try:
    diagnostics.compound_return_matrix(incomplete, 4)
except ValueError as error:
    errors.append(str(error))

print(json.dumps({
    "index": compounded.matrix.index.tolist(),
    "columns": compounded.matrix.columns.tolist(),
    "values": compounded.matrix.to_numpy().tolist(),
    "unchangedByDroppedRows": bool(np.array_equal(
        compounded.matrix.to_numpy(), changed.matrix.to_numpy()
    )),
    "audit": compounded.to_dict(),
    "errors": errors,
}))
`);

    const compound = (values) => values.reduce((equity, value) => equity * (1 + value), 1) - 1;
    assert.deepEqual(result.index, [106, 110]);
    assert.deepEqual(result.columns, ["a", "b"]);
    assert.ok(Math.abs(result.values[0][0] - compound([0.04, 0.05, 0.06, 0.07])) < 1e-12);
    assert.ok(Math.abs(result.values[1][0] - compound([0.08, 0.09, 0.1, 0.11])) < 1e-12);
    assert.ok(
      Math.abs(result.values[0][1] - compound([-0.02, -0.025, -0.03, -0.035])) < 1e-12,
    );
    assert.equal(result.unchangedByDroppedRows, true);
    assert.deepEqual(result.audit, {
      method: "non_overlapping_compounded_returns",
      originalObservations: 11,
      retainedObservations: 8,
      compoundedObservations: 2,
      blockSize: 4,
      droppedLeadingObservations: 3,
      sourceFirstKey: 100,
      sourceLastKey: 110,
      retainedFirstKey: 103,
      droppedLeadingKeys: [100, 101, 102],
      blockBoundaries: [
        {
          block: 0,
          startPosition: 3,
          stopPosition: 7,
          startKey: 103,
          endKey: 106,
          observations: 4,
        },
        {
          block: 1,
          startPosition: 7,
          stopPosition: 11,
          startKey: 107,
          endKey: 110,
          observations: 4,
        },
      ],
    });
    assert.equal(result.errors.length, 4);
    assert.match(result.errors[0], /positive integer/);
    assert.match(result.errors[1], /positive integer/);
    assert.match(result.errors[2], /one complete block/);
    assert.match(result.errors[3], /complete and finite/);
  },
);

test(
  "return-matrix construction fails closed on incomplete or ambiguous trial ledgers",
  { skip: !PYTHON_RESEARCH_DEPS },
  () => {
    const result = runResearchPython(String.raw`
import json
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, sys.argv[1])
import diagnostics

complete = pd.DataFrame([
    {"timestamp": t, "trial_id": trial, "net_return": (t + 1) * scale}
    for t in range(4)
    for trial, scale in (("a", 0.001), ("b", -0.001))
])
wide = diagnostics.return_matrix_from_long(complete)

errors = []
cases = [
    complete.drop(index=0),
    pd.concat([complete, complete.iloc[[0]]], ignore_index=True),
]
for case in cases:
    try:
        diagnostics.return_matrix_from_long(case)
    except ValueError as error:
        errors.append(str(error))

nan_matrix = wide.copy()
nan_matrix.iloc[0, 0] = np.nan
try:
    diagnostics.validate_return_matrix(nan_matrix)
except ValueError as error:
    errors.append(str(error))

try:
    diagnostics.cscv_pbo(
        pd.concat([wide, wide.iloc[[0]].set_axis([4])]),
        n_slices=4,
    )
except ValueError as error:
    errors.append(str(error))

constant_trial = wide.copy()
constant_trial["a"] = 0.0
try:
    diagnostics.deflated_sharpe_ratio(constant_trial)
except ValueError as error:
    errors.append(str(error))

print(json.dumps({
    "shape": list(wide.shape),
    "errors": errors,
}))
`);

    assert.deepEqual(result.shape, [4, 2]);
    assert.equal(result.errors.length, 5);
    assert.match(result.errors[0], /complete and finite/);
    assert.match(result.errors[1], /duplicate timestamp\/trial/);
    assert.match(result.errors[2], /complete and finite/);
    assert.match(result.errors[3], /observations must divide n_slices/);
    assert.match(result.errors[4], /non-constant finite returns/);
  },
);

test(
  "legacy harness labels its scalar adjustment as a proxy rather than formal DSR",
  { skip: !PYTHON_RESEARCH_DEPS },
  () => {
    const result = runResearchPython(String.raw`
import json
import sys

sys.path.insert(0, sys.argv[1])
import harness

proxy = harness.multiple_testing_sharpe_proxy(1.2, 2000, 365, 12)
compatibility = harness.deflated_sharpe_prob(1.2, 2000, 365, 12)
print(json.dumps({
    "equal": proxy == compatibility,
    "rejectsFormalClaim": (
        "not the formal Deflated Sharpe Ratio"
        in harness.multiple_testing_sharpe_proxy.__doc__
    ),
    "proxyClaim": "proxy" in harness.multiple_testing_sharpe_proxy.__doc__,
}))
`);

    assert.equal(result.equal, true);
    assert.equal(result.rejectsFormalClaim, true);
    assert.equal(result.proxyClaim, true);
  },
);
