# Training-control admission audit — 2026-09-17

Baseline: merged main `1fada736413f4a482e47a103776ad09c032f3d9a`.
This is a synthetic engineering audit. No market data, protected holdout or
financial experiment is used.

## Reproduced defects

- PPO with step budgets 0, -1 or -256 returned an initialized policy and metadata
  with zero updates, without ever reaching the collector's budget validation.
- Q training with `offline="false"` selected uniform offline behavior because
  the string was truthy.
- Training initialized models before invalid controls reached downstream checks.
- Fixed-width NumPy integer arithmetic could wrap seed offsets or batch-count
  arithmetic before the collector normalized the values. With a maximum int64
  budget, the old PPO loop could execute no batches.

Four regression methods brought the suite to 62 tests. Before repair they
reproduced 89 failures and one error; all 62 methods pass after repair. Large
integer boundary tests stop at the first collector call and never allocate a
large replay buffer or execute a large-budget training run.

## Repair

Both training entry points require Python or NumPy integer horizons, seeds and
positive budgets, excluding booleans. Horizons remain 1/3/6 and seeds nonnegative.
Admitted values convert to Python integers before model initialization, seed
offsets or batch arithmetic. Q mode requires an exact Python boolean; its existing
execution/risk validation now also precedes initialization.

Invalid controls raise before model construction or RNG use. Valid learning
formulas, batch sizes, default 4,096-step budget, reward terms, seeds, replay
construction and checkpoint-selection rules remain unchanged. Returned step-budget
metadata uses a Python integer even when the caller supplied a NumPy integer.

There is no new upper compute limit: callers still need an explicit registered
budget and appropriate resources. Dataset admission and model initialization are
separate responsibilities; this change concerns training controls, not every
possible input failure. Episode-summary timing and coverage are unchanged.

## Valid-training parity

Twelve deterministic synthetic fits cover four family/ablation configurations,
three seeds (11/23/47), horizon 3 and 257 steps, crossing the 256-step batch
boundary. Each candidate result matches its baseline bytes for policy parameters,
Adam moments, update counts and all returned training metadata. Sorted fixture
SHA-256:
`11339b9f60b617de50cabb5eecfb82b2b556e12b35c39ae49d1b3da87834c10a`.

These are same-host engineering comparisons, not new financial trials, economic
performance evidence or a general cross-platform numerical guarantee. No generated
policy or replay buffer is committed. Reproduce in isolated baseline/candidate
checkouts with the same Python/NumPy environment:

```python
import hashlib, json, sys
import numpy as np
sys.path.insert(0, "scripts/research")
from sequential_env import Scale
from sequential_learning import train_ppo, train_q
p = 100*np.exp(np.sin(np.arange(160)/9)*.01)
f = np.zeros(160); scale = Scale.fit([p]); rows = []
for algorithm in ("ppo", "double_dqn", "cql", "cql_no_inventory_penalty"):
    for seed in (11, 23, 47):
        args = ({"x":p}, {"x":f}, scale, 3, seed)
        net, info = (train_ppo(*args, steps=257) if algorithm == "ppo" else
            train_q(*args, offline=algorithm != "double_dqn",
                risk_penalty=0. if algorithm == "cql_no_inventory_penalty" else .01,
                steps=257))
        rows.append(dict(algorithm=algorithm, seed=seed, info=info, steps=net.steps,
            p={k:v.tolist() for k,v in net.p.items()},
            m={k:v.tolist() for k,v in net.m.items()},
            v={k:v.tolist() for k,v in net.v.items()}))
print(hashlib.sha256(json.dumps(rows, sort_keys=True, allow_nan=False).encode()).hexdigest())
```

Tests also verify that a valid 257-step run has eight PPO policy updates or 257
Q updates, and that Q buffers and action counts contain exactly 257 transitions.
Small Python-integer and NumPy-integer runs have identical weights and metadata.
Invalid-control fixtures instrument both model construction and RNG entry points.

## Verification and decision

From the repository root:

```bash
python3 test/sequential_screen_test.py
bash scripts/verify.sh automation
bash scripts/verify.sh full
```

The PR records final source, actual full-wrapper result and log hash, and
final-head GitHub CI. `A-SEQUENTIAL-RESEARCH-R11` links the four executable
witnesses; Haskell and Markdown risk mitigations remain synchronized.
`RL-OFFLINE-001` remains HIGH/OPEN.

No historical result or artifact is rewritten, and no assertion is made that
the registered historical run used invalid budgets or modes. Original costs,
stress results, all seeds, invalid OPE and no-adoption conclusions remain intact.
No dependency, configuration, production caller, order interface, live exploration,
authorization, risk setting, champion, deployment or protected holdout changes.
No candidate passed; preserve the champion and continue offline research.
