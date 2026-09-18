# Research proposal type audit — 2026-09-17

Baseline: merged main `32f461f0eebfe32f6ceca7afde3d77f859bff4a3`.
This is an engineering audit using synthetic inputs, not a new financial trial.
The original negative results, source hashes, policy artifacts and protected
holdouts remain unchanged. No candidate passed and no integration is recommended.

## Reproduced defect

The Python shield accepted `enabled="false"`, `valid="invalid"` and `ownership=1`
as affirmative evidence. Boolean elapsed time was accepted as zero, a missing
elapsed value raised `TypeError`, and a NumPy complex action could be converted
to a real proposal. A constant-price synthetic replay with these invalid gates
acquired 0.0025 units and appended an outcome row. These were simulated units,
not an exchange order or live-money experiment.

The inference helper had the same enablement truthiness problem and did not
consistently reject boolean, complex, object or malformed observation/output
vectors. The new regression suite reproduced 37 failures and 14 errors before
repair (42 test methods). All 42 methods pass after repair, including added
masked-observation and exact elapsed-boundary checks.

## Contract and implementation

- Only exact Python `True` admits enabled, valid-observation and ownership gates.
  Numeric flags, strings, containers and NumPy booleans are not authorization.
- Action and elapsed-time inputs must be finite real numeric scalars. Booleans,
  complex values, arrays and unrepresentable integers are rejected without
  coercion or directional defaults. Ordinary Python and NumPy real scalars remain
  supported within their existing bounds.
- Replay checks the shield before observation reads or pending-fill processing.
  Rejection terminates the path with no new transition, fill, reward or invented
  liquidation. Existing units/pending state remain historical failed-path state;
  stepping a terminated episode remains prohibited.
- Inference accepts only finite, real, unmasked vectors with widths 12 and 3 for
  observations and outputs. Invalid enablement/observations do not call the
  network. A model exception yields no proposal. Normal evaluation is unchanged.
- Elapsed time must lie in `[0,20]` ms. This remains a post-call check, not a
  preemptive timeout or proof that a hung model cannot stall research.

No model identifier, artifact schema, production caller or environment variable
changes. Haskell's typed boolean evidence and always-false order authorization
remain unchanged. `A-SEQUENTIAL-RESEARCH-I2` and three executable regression
witnesses describe the strengthened Python boundary. Typed and Markdown
`RL-OFFLINE-001` mitigations remain synchronized; canonical status stays HIGH/OPEN.

## Valid-input parity and latency observation

On the same host, all 108 deterministic synthetic scenarios produced identical
observations, rewards, rows, failures, equity, units and intervention counts
before and after repair. The scenarios cover horizons 1/3/6, nine execution
assumptions and four action sequences. Their sorted JSON SHA-256 was
`22276e1b07f503e97c2bb57315aac1ef3079ed1c25dcbb0a07f6cb7e10f3325a`.
This demonstrates fixture parity, not historical-market or cross-platform parity.

A single CPU observation using Python 3.13.3 and NumPy 2.3.5 measured 1,000
inference calls after 20 warmups, with an untrained `Network(11)` and zero-valued
12-element synthetic observations: median 0.019154 ms, p99 0.052767 ms, maximum
0.144717 ms, zero absent proposals. Timing covers the whole `infer` call. This
uncontrolled local measurement is not a production budget, speed comparison or
evidence of trading efficacy; concurrent workloads and hung calls are unproven.

## Reproduction

Run from the repository root:

```bash
python3 -m unittest discover -s test -p sequential_screen_test.py
bash scripts/verify.sh automation
bash scripts/verify.sh full
```

The associated PR records final source identity, actual wrapper results, log
hash and CI. The regression tests use no market input, credentials or network.
No original financial experiment, training search or holdout evaluation is rerun.

To reproduce the 108-scenario comparison, execute the following in both an
isolated checkout of the baseline commit and this revision, on the same host:

```python
import hashlib, json, sys
import numpy as np
sys.path.insert(0, "scripts/research")
from sequential_env import Replay, Scale, Execution
p = 100*np.exp(np.sin(np.arange(256)/9)*.01)
f = np.zeros(256); f[::7] = .01
scale = Scale.fit([p[:160]])
configs = [Execution(), Execution(cost_multiplier=1.5),
    Execution(cost_multiplier=2), Execution(cost_multiplier=2.5),
    Execution(extra_delay=1), Execution(fill_fraction=.5),
    Execution(miss_every=10), Execution(impact_bps=10),
    Execution(funding_multiplier=2)]
rows = []
for horizon in (1, 3, 6):
    for cfg_id, cfg in enumerate(configs):
        for seq in ([.25], [-.25], [0.], [.25, 0., -.25, 0.]):
            env = Replay(p, f, 30, 91, horizon, scale, cfg, enabled=True)
            steps = []
            while not env.done:
                obs, reward, done = env.step(seq[len(steps) % len(seq)])
                steps.append([None if obs is None else obs.tolist(), reward, done])
            rows.append(dict(horizon=horizon, cfg=cfg_id, sequence=seq,
                steps=steps, rows=env.rows, failure=env.failure,
                equity=env.equity, units=env.units, rejections=env.rejections,
                modifications=env.modifications))
print(hashlib.sha256(json.dumps(rows, sort_keys=True,
                               allow_nan=False).encode()).hexdigest())
```

The generated replay JSON remains outside Git. This audit adds no economic
confirmation, market-support evidence or production authorization. Preserve the
champion and keep the research policies rejected and non-authorizing.
