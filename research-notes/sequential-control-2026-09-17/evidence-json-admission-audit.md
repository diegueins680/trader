# Evidence JSON admission audit — 2026-09-18

Baseline: merged main `f454216b91612a33f8b9023554597d8bdc5aba7d`.
Synthetic engineering fixtures only; no market data, training experiment,
protected holdout or production action is used.

## Reproduced defects

The exporter used default JSON decoding after verifying archive hashes. Duplicate
object keys silently retained only the last value, and non-standard numeric
constants or overflowing exponents could become non-finite Python floats. Later
schema/reconciliation checks did not see discarded duplicate values. Report
rendering could also omit non-finite values before its finite JSON serialization
check, for example an interior training loss.

Eleven duplicate-key fixtures were accepted on the baseline:

- One mutation for each input: index, manifest, summary, evaluation, training,
  planned registry, JSONL events and OPE.
- An identical duplicate authorization value.
- An escaped spelling colliding with an authorization key.
- A nested escaped-key collision in training metadata.

Examples include an earlier `promotionAllowed: true` hidden by a later false
value and an earlier incorrect index hash hidden by the correct hash. These are
ambiguous evidence interpretations, not proof that an order was authorized or
that malformed evidence was used in the historical run.

Twenty additional fixtures were accepted: `NaN`, `Infinity`, `-Infinity`, `1e309`
and `-1e309`, each placed in an omitted field of evaluation, training, planned
registry or JSONL events. Training fixtures put the value between finite first
and last losses, which the compact report alone retained. Each mutation was
rehash-bound to its synthetic index, isolating decoding from hash-tamper rejection.

The two baseline regression methods reported 11 and 20 assertion failures,
respectively. Hash verification establishes byte identity; it does not establish
unambiguous or numerically admissible JSON.

## Repair

One small standard-library decoder handles admitted index bytes, all seven
verified report-input snapshots and each JSONL event. It rejects:

- Duplicate object keys at any depth, including identical values and collisions
  after JSON string escape decoding, before silently losing a value.
- Non-finite constants through the JSON constant hook.
- Floating-point tokens that overflow to non-finite values through the float hook.

The checks apply even to fields the compact reports omit. They run before output
creation and do not reopen verified paths. Reconciliation, report rendering,
streamed hashing of large return files/policies and admitted index provenance are
unchanged. Strings such as `"NaN"` remain text; valid finite floats, signed zero,
large integers, booleans, null and escaped Unicode retain normal JSON semantics.

Existing policy loading and market-context decoding already reject duplicate keys
in their separate contracts. Their helpers are scoped to those artifacts/campaigns;
this change keeps the exporter independent and adds explicit float-overflow
rejection without importing another campaign's data/network machinery. No new
module or dependency is introduced.

This is not a canonical-encoding requirement, proof of source truth or exhaustive
schema validation. Finite unknown fields retain their existing behavior. It does
not add input-size/depth limits or promise bounded memory for arbitrarily large
report JSON. Large non-report files remain streamed and are not parsed by this
decoder. Ambiguous or invalid historical archives require investigation, not an
automatic rewrite that chooses one interpretation.

## Verification and compatibility

Three new methods bring the suite to 72 passing tests. All 31 malformed fixtures
now fail before report-directory creation. A valid-value fixture explicitly checks
Unicode escapes, a large integer, finite exponent notation, text resembling invalid
numbers and preservation of negative zero. Existing verified-snapshot replacement,
registry reconciliation, v2 episode-accounting and publication-failure tests pass.

The existing exact legacy-report fixture still produces all seven files with the
same sorted filename/content SHA-256:
`d9cdec2fa64d52f3f12d4d41ec1a8d9a3c3c7aca902dddc9b30f021d5c89661b`.
This is synthetic compatibility evidence, not a rerun or reinterpretation of the
historical financial archive. No generated artifact or dataset is committed.

From the repository root:

```bash
python3 test/sequential_screen_test.py
python3 -m unittest discover -s test -p sequential_screen_test.py -k export_rejects_duplicate_json
python3 -m unittest discover -s test -p sequential_screen_test.py -k export_rejects_nonfinite_numbers
bash scripts/verify.sh automation
bash scripts/verify.sh full
```

To reproduce the baseline failures, use the new test fixtures with the exporter
from the baseline commit in an isolated checkout. The PR records final source,
actual full-wrapper result and log hash, and final-head GitHub CI.
`A-SEQUENTIAL-RESEARCH-R14` links the three executable witnesses. Haskell and
Markdown risk mitigations stay synchronized; `RL-OFFLINE-001` remains HIGH/OPEN.

## Decision

No historical return, seed, cost stress, OPE/support finding, statistical inference
or no-adoption conclusion changes. No candidate is trained, selected, integrated
or promoted. The current champion, protected holdouts, production settings and
existing deployment are preserved. No live exploration or order authorization is
introduced. No candidate passed; continue offline research without adoption.
