# Evidence-publication boundary audit — 2026-09-18

Baseline: merged main `487d33adba9cc363a2001f82ee3fd434a5d9ab52`.
This audit uses synthetic fixtures only, with no market data, protected holdout,
new training experiment or production action.

## Reproduced defect

The runner included successful-fit terminal-event publication inside the training
exception handler. It also appended replay results, wrote return rows and emitted
terminal events inside the replay exception handler. An evidence-write failure
could therefore be classified as a computational trial failure after the outcome
had already been appended, or after a complete terminal line had been written.

Seven one-shot failure scenarios reproduced the defect on the baseline:

- Training and replay terminal-event writes failing before writing the line.
- Training and replay terminal-event flushes failing after writing the line.
- Return-path writes failing at data row 1, 2 or 20.

The two fault-injection test methods failed in all seven cases because the runner
swallowed the injected `OSError` and continued. The existing exporter already
rejects the resulting duplicate/inconsistent records; this is a producer failure
boundary repair, not evidence that malformed archives were accepted for promotion.

## Repair

Successful-fit bookkeeping and terminal-event publication now occur after the
training/artifact-admission exception boundary. OPE retains its separate failure
handler. Replay computation alone is inside the replay handler; one result is
appended afterward, followed by return-path publication and one terminal event.

A terminal-ledger write/flush/serialization failure or return-path publication
failure now propagates before final summary and evidence-index creation. There is
no retry that could publish a contradictory replacement outcome. A flush failure
may leave a complete-looking terminal line; that line alone does not establish
archive completion. Previously written rows, policies and event lines are preserved
as partial evidence for investigation, without automatic cleanup or resume.

Genuine training, artifact-admission, OPE and replay-computation exceptions retain
their existing failed-trial behavior. Artifact-save failures are still failed fits,
not successful fits. Failures during final JSON writes or index writing are already
outside these handlers; this change does not make those writes atomic, promise
`fsync` durability, or implement crash recovery. Do not manually create an index to
turn an interrupted directory into completed evidence. A repeat requires a new
output directory and the original authorization/data boundaries.

## Tests and compatibility

Three new integration methods bring the suite to 69 passing tests. The seven
injected publication failures now propagate, with no replacement terminal attempt,
final summary or index. Return-row tests verify the exact persisted partial prefix.
A successful synthetic runner-to-export fixture produces one terminal per planned
trial and all seven compact reports. Existing tests retain failed trials across
training, policy-save, replay and OPE exceptions. Legacy report-byte compatibility
also remains covered by the existing suite.

A same-host compatibility comparison uses the same successful fixture before and
after the repair. It contains one initialized PPO-shaped policy artifact used only
for plumbing, a fixed neutral proposal, one horizon/seed and nine stress replays.
There is no learning or financial claim. Eight projected archive members match
exactly: the policy artifact, training, evaluation, OPE, planned registry, terminal
events, returns CSV and summary. Projection removes only elapsed seconds and peak
memory; the changing source manifest and its index are excluded. SHA-256:
`70f068be78d6f1094e272c791ee09d1cb7641ebb1a9080f4b44e7b6417a73d65`.

The fixture helper is committed in `test/sequential_screen_test.py`. To compare
baseline and repaired runners, use that same helper with each runner version in
isolated checkouts and the same Python/NumPy environment. Run this recipe from the
repository root with an output pathname as its first argument, then compare the
two output files:

```python
import hashlib, json, sys, tempfile
from pathlib import Path
sys.path.insert(0, 'test')
from sequential_screen_test import SequentialContracts, runner
case = SequentialContracts(); case.setUp()
def project(value):
    if isinstance(value, dict):
        return {k:project(v) for k,v in value.items() if k not in ('seconds','processPeakRssPlatformUnits')}
    if isinstance(value, list): return [project(v) for v in value]
    return value
with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    with case.publication_fixture(root):
        runner.run(root/'unused-panel',root/'unused-funding',root/'run')
    result = {}
    for p in sorted((root/'run').rglob('*')):
        if not p.is_file() or p.name in ('manifest.json','evidence-index.json'): continue
        text = p.read_text()
        value = ([json.loads(line) for line in text.splitlines()] if p.suffix == '.jsonl' else
                 json.loads(text) if p.suffix == '.json' else text)
        result[str(p.relative_to(root/'run'))] = project(value)
encoded = json.dumps(result,sort_keys=True,allow_nan=False).encode()
Path(sys.argv[1]).write_bytes(encoded)
print(len(result), 'projected archive members',hashlib.sha256(encoded).hexdigest())
```

The source-commit and development-loader stubs exist only inside the synthetic
test fixture. No production source/data admission rule changes. This comparison
is not a cross-platform numerical guarantee. No generated policy or large output
is committed.

## Verification and decision

```bash
python3 test/sequential_screen_test.py
bash scripts/verify.sh automation
bash scripts/verify.sh full
```

The PR records the frozen source, actual full-wrapper result/log hash and final-head
GitHub CI. Formal clause `A-SEQUENTIAL-RESEARCH-R13` links the three witnesses.
Haskell and Markdown risk mitigations remain synchronized; `RL-OFFLINE-001` remains
HIGH/OPEN. README, CHANGELOG and reproduction instructions describe the boundary.

No historical result, cost stress, OPE/support finding, statistical diagnostic or
no-adoption decision changes. No holdout is accessed, no champion is replaced, no
model is integrated and no live authorization, exploration, order, risk setting,
fleet setting or deployment is introduced. No candidate passed; preserve the
champion and continue offline research.
