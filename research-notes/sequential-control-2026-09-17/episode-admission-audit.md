# Episode admission audit — 2026-09-17

Baseline: merged main `0c8aa6e85cfffd939b00597feee1abbc861ab73a`.
This audit uses synthetic engineering fixtures only. It does not rerun a financial
experiment, open protected data or change the existing no-adoption decision.

## Reproduced defects

- `Replay(..., start=30, stop=33.5, horizon=1)` admitted a fractional end. In a
  bounded eight-step probe, time stopped at 33 with `done=False`; later calls made
  no progress because the terminal equality could never hold. An initial unbounded
  synthetic probe was stopped after reproducing the nontermination.
- A boolean horizon was accepted as cadence 1. `collect(..., count=1.5)` collected
  two transitions, exceeding its supplied budget.
- Complex and masked price arrays produced features because conversion discarded
  their representation. Boolean execution coefficients were accepted as numbers,
  including a false cost multiplier interpreted as zero costs.

The initial four regression methods brought the suite to 46 tests and reproduced
59 failures and 30 errors. After repair, all 46 methods pass, including additional
behavior-probability and replaced-funding regressions. These tests are not counted
as financial trials or independent confirmation.

## Admission and causality

Replay indices, horizon, collection seed and budget must be Python or NumPy
integers, excluding booleans. Admitted NumPy integers normalize to Python integers
before arithmetic. Existing bounds remain: horizon 1/3/6, warmup of 24 bars,
nonnegative seed, positive transition budget and endpoints within the supplied
series. Valid episodes progress to `stop-1` and terminate.

Prices and funding must be aligned one-dimensional real unmasked arrays. Collection
also requires matching nonempty string-symbol maps and at least 121 bars per
symbol, the minimum needed by its existing 97-bar episode sampler. All of these
checks precede RNG sampling. Behavior probabilities must have a real unmasked
vector representation and satisfy the existing probability checks. Execution
coefficients require finite real nonboolean scalars in their unchanged ranges.

Admission inspects shape, dtype, lengths and indices, not future values. Features
still read only their trailing prefix; actual price/funding scalars are checked
when each transition consumes them. Future NaNs leave the current observation
unchanged, then fail the path when reached without inventing a fill. Replacing
funding with non-real/masked/non-scalar values also fails when consumed. This
does not establish external point-in-time provenance or a realistic fill model.

## Valid-input parity

All 108 scenarios from the [proposal audit recipe](proposal-types-audit.md#reproduction)
retain the same sorted-JSON hash:
`22276e1b07f503e97c2bb57315aac1ef3079ed1c25dcbb0a07f6cb7e10f3325a`.
They cover 3 horizons, 9 execution assumptions and 4 action sequences, comparing
observations, rewards, economic rows, inventory and failure/intervention state.

Another 18 seeded collection scenarios match their baseline bytes exactly,
including observations, actions, rewards, next states, terminal indicators,
behavior probabilities and episode summaries. Their hash is
`81f811bae0802aba61cfba6674be0f90afb65ad828158cda281492a0ba9d1926`.
These are same-host engineering comparisons, not cross-platform guarantees,
historical-market reruns or economic superiority tests. Generated JSON stays
outside Git; no model artifact or market dataset is added.

## Reproduction and verification

From the repository root:

```bash
python3 -m unittest discover -s test -p sequential_screen_test.py
bash scripts/verify.sh automation
bash scripts/verify.sh full
```

The associated PR records the exact final source revision, actual full-wrapper
result, log hash and CI evidence. To reproduce the collection comparison, execute
this recipe in isolated baseline and candidate checkouts on the same host:

```python
import hashlib, json, sys
import numpy as np
sys.path.insert(0, "scripts/research")
from sequential_env import Scale, Execution, collect
p = 100*np.exp(np.sin(np.arange(256)/9)*.01)
f = np.zeros(256); f[::7] = .01
scale = Scale.fit([p[:160]])
rows = []
for horizon in (1, 3, 6):
    for seed in (11, 23, 47):
        for cost in (1., 2.):
            data = collect({"x": p[:160]}, {"x": f[:160]}, scale,
                horizon, seed, 64, execution=Execution(cost_multiplier=cost))
            rows.append(dict(horizon=horizon, seed=seed, cost=cost,
                data={k: v.tolist() if isinstance(v, np.ndarray) else v
                      for k, v in data.items()}))
print(hashlib.sha256(json.dumps(rows, sort_keys=True,
                               allow_nan=False).encode()).hexdigest())
```

## Limits and decision

`A-SEQUENTIAL-RESEARCH-R7` and four executable witnesses cover these admission
rules. Typed Haskell and Markdown risk mitigations remain synchronized;
`RL-OFFLINE-001` stays HIGH/OPEN. Valid v1 reward, execution, policy and artifact
semantics remain unchanged. There is no new dependency or configuration flag.

This does not add preemptive inference timeouts, prove joint state-action support,
validate simulator fills, supply matched-champion evidence or satisfy statistical
promotion gates. No live authorization, risk setting, fleet identity, exposure,
deployment, protected holdout or original financial result changes. Preserve the
champion; no candidate passed.
