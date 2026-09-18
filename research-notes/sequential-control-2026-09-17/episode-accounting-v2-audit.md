# Episode accounting v2 audit — 2026-09-18

Baseline: merged main `66f8b366219c63c5020a6805a2d68936745bc8ba`.
This is a synthetic engineering audit. No market data, protected holdout or
financial experiment is used.

## Reproduced defect

The collector appended an episode summary only when it began the next episode.
An episode ending on the last requested sample was therefore omitted from
training episode-return and failure summaries, even though its terminal
transition was present in the learning batch.

On 121 constant prices of 100 with zero funding and a no-trade policy, budgets
96/32/16 at horizons 1/3/6 each produced one terminal decision but zero episode
summaries. With horizon 3, a one-decision long policy and synthetic funding 64 at
index 26, an accounted drawdown stop returning -16.05% was likewise absent from
the summaries. This extreme funding fixture tests accounting, not market realism.

## Repair and versioned contract

Collection now records each admitted terminal immediately. New successful fits
carry this optional object alongside the existing episode records:

```json
{"episodeAccountingV2": {"collections": 1, "started": 3,
  "completed": 2, "truncated": 1, "decisions": 193}}
```

- `collections`: actual collection calls that returned admitted batches.
- `started`: episodes initialized in those calls.
- `completed`: fully accounted terminal episodes, including risk stops.
- `truncated`: episodes still live at a collection budget cutoff.
- `decisions`: admitted transitions collected, equal to the fit's step budget.

Started equals completed plus truncated. Each collection has at most one budget
truncation. A completed episode is not necessarily successful performance. A
truncated episode receives neither a fabricated terminal liquidation nor an
invented episode return; its existing nonterminal successor still participates
in learning. PPO and Double DQN sum counts across their 256-decision collections;
CQL counts its one offline collection. No learning formula, seed, reward, batch,
checkpoint or policy-artifact semantics change.

The exporter requires exact nonnegative integer count fields (excluding booleans),
the algorithm's expected collection count, the positive step budget, reconciled
started/completed/truncated counts and one valid return/reason record per terminal.
It rejects inconsistent or unsupported accounting versions before creating output.
These checks establish internal consistency, not the truth of supplied evidence.
A producer omitting v2 is still a legacy producer with unknown coverage; hashes and
source provenance remain necessary to identify that producer.

Legacy archives without v2 retain their original report bytes and potentially
incomplete coverage. They are not retroactively relabeled as fully accounted.
Aborted collection/training calls still return no successful coverage record;
existing failed-fit registry entries preserve those trials, but v2 does not
recover episode detail from inside an aborted call. The field is a report extension,
not a change to the v1 environment, observation, action, reward or policy artifact.

## Executable evidence

Four regression/integration methods bring the suite to 66 tests. Final fixtures
against the three baseline collector/trainer/registry modules produced 23 failures
and 14 errors before the final collection-count mutation was added. All 66 methods
pass with the repair, including that added mutation. An early test assumption was
corrected: flat prices do not imply exclusively normal endpoints, because turnover
limits and transaction costs can still stop a changing-position policy. Trainer
aggregation is checked against observed transition terminal flags instead.

Coverage includes exact endpoint budgets and neighboring budgets at all three
horizons, the final-sample risk loss, a live-inventory budget truncation, eight
algorithm/horizon trainer combinations crossing the 256-decision boundary,
malformed/inconsistent archive mutations, and exact legacy-report compatibility.

Twelve independent deterministic synthetic fits cover four family/ablation
configurations and all three seeds, horizon 3 and 257 decisions. After excluding
only the intentionally changed `episodes` and new `episodeAccountingV2` report
fields, every remaining training field, policy weight, Adam moment and update
count matches the baseline bytes. Projected fixture SHA-256:
`6eccaf30c49b3c0717709e50ec38052edf8c0bdd235880b54f74f075488cf407`.

| Algorithm | Seed | Legacy summaries | Collections | Started | Completed | Truncated | Decisions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ppo | 11 | 32 | 2 | 34 | 32 | 2 | 257 |
| ppo | 23 | 32 | 2 | 34 | 32 | 2 | 257 |
| ppo | 47 | 24 | 2 | 26 | 24 | 2 | 257 |
| double_dqn | 11 | 17 | 2 | 19 | 17 | 2 | 257 |
| double_dqn | 23 | 21 | 2 | 23 | 21 | 2 | 257 |
| double_dqn | 47 | 24 | 2 | 26 | 24 | 2 | 257 |
| cql | 11 | 37 | 1 | 38 | 37 | 1 | 257 |
| cql | 23 | 37 | 1 | 38 | 37 | 1 | 257 |
| cql | 47 | 27 | 1 | 28 | 27 | 1 | 257 |
| cql_no_inventory_penalty | 11 | 37 | 1 | 38 | 37 | 1 | 257 |
| cql_no_inventory_penalty | 23 | 37 | 1 | 38 | 37 | 1 | 257 |
| cql_no_inventory_penalty | 47 | 27 | 1 | 28 | 27 | 1 | 257 |

These 257-decision fixtures happen not to end on a terminal in their final
collection samples, so their legacy and v2 completed counts agree. The separate
exact-boundary fixtures demonstrate the omitted-terminal repair. The new truncation
counts expose the coverage boundary even when the old completed count was correct.

All seven compact reports from the deterministic legacy synthetic archive retain
exact content. Sorted filename/content fixture SHA-256:
`d9cdec2fa64d52f3f12d4d41ec1a8d9a3c3c7aca902dddc9b30f021d5c89661b`.
This test does not reopen or rewrite the historical financial archive.

Reproduce projected training parity in isolated baseline/candidate checkouts with
the same Python/NumPy environment:

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
        info.pop("episodes", None)
        info.pop("episodeAccountingV2", None)
        rows.append(dict(algorithm=algorithm, seed=seed, info=info, steps=net.steps,
            p={k:v.tolist() for k,v in net.p.items()},
            m={k:v.tolist() for k,v in net.m.items()},
            v={k:v.tolist() for k,v in net.v.items()}))
print(hashlib.sha256(json.dumps(rows, sort_keys=True, allow_nan=False).encode()).hexdigest())
```

These are same-host engineering comparisons, not cross-platform numerical
guarantees or financial trials. No trained policy, replay buffer or large generated
artifact is committed.

## Verification and decision

From the repository root:

```bash
python3 test/sequential_screen_test.py
bash scripts/verify.sh automation
bash scripts/verify.sh full
```

The PR records frozen source, actual full-wrapper result and log hash, and
final-head GitHub CI. `A-SEQUENTIAL-RESEARCH-R12` links the four executable witnesses.
The Haskell and Markdown risk mitigations remain synchronized;
`RL-OFFLINE-001` remains HIGH/OPEN.

No historical return, seed, cost stress, invalid OPE finding or no-adoption decision
changes. No new data, holdout access, statistical confirmation, dependency,
configuration, production caller, order interface, live exploration, authorization,
risk setting, champion or deployment is introduced. No candidate passed;
preserve the champion and continue offline research.
