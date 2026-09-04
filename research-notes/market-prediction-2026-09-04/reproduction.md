# Reproduction and verification

Run from the repository root with the pinned toolchain in `.tool-versions` and `.nvmrc`.

## Validate the committed research packet

```bash
jq empty \
  research-notes/market-prediction-2026-09-04/experiment-manifest.json \
  research-notes/market-prediction-2026-09-04/experiment-registry.json \
  research-notes/market-prediction-2026-09-04/data-source-license-manifest.json \
  research-notes/registrations/har-rv-risk-gate-v1.json \
  research-notes/registrations/depth-normalized-ofi-v1.json \
  research-notes/registrations/missingness-aware-calibrated-shallow-v1.json

python3 -c 'import csv; rows=list(csv.DictReader(open("research-notes/market-prediction-2026-09-04/paper-matrix.csv", encoding="utf-8"))); assert len(rows)==50; assert all(len(row)==27 and None not in row for row in rows)'

node --test test/market-prediction-research.test.mjs
node scripts/verify-formal-specs.mjs
```

## Verify code and compatibility

```bash
bash scripts/verify.sh haskell
bash scripts/verify.sh automation
bash scripts/verify.sh full
```

The full wrapper includes the web surface even though this change does not redesign the UI.

## Verify an alternative-data panel artifact

Build every prospective alternative-data panel with `--manifest`, then verify the exact cache, bar grid, panel bytes, ordered schema, and coverage semantics before registering or fitting anything:

```bash
python3 scripts/research/alternative_data.py verify-panel \
  --manifest data/research/BTCUSDT_1h-alternative.json
```

If artifacts were lawfully moved without changing their bytes, pass their new locations with `--cache`, `--bars`, and `--panel`. Do not edit hashes to make changed inputs appear compatible. The verifier checks panel semantics, recomputes the declared populations, and reconstructs exact panel bytes after digest validation, so rehashing a changed panel still fails closed.

## Reproduce existing negative results safely

The immutable result notes contain the exact registrations, implementation/data hashes, evidence paths, and outcomes:

- `research-notes/2026-07-15-historical-funding-development-result.md`
- `research-notes/2026-07-15-residual-reversal-development-result.md`
- `research-notes/2026-07-15-risk-controlled-residual-reversal-development-result.md`
- `research-notes/2026-07-15-risk-controlled-residual-reversal-v2-development-result.md`

Use the corresponding no-flag runner only to validate an existing immutable evidence directory. Do **not** pass `--open-final-holdout`, do not change an output directory to evade the shared registry, and do not alter the registered costs/gates. The historical final holdout was not opened by this audit.

## Prospective carry boundary

`research-notes/registrations/cross-sectional-funding-carry-v1.json` is authoritative. Before `2027-01-20T13:00:00Z`, inspect acquisition metadata only. Do not calculate a return, rank, weight, PnL, risk measure, or performance statistic. After that timestamp, follow its two-step acquisition receipt and one-shot registry protocol exactly.

## Future candidate procedure

The three new registration files define start time `2027-01-21T00:00:00Z`. Do not move it earlier. Before a first development run:

1. Implement the exact registered interpretation and all fail-closed tests.
2. Freeze and hash source manifests, event/availability schema, exact cache, bar grid, panel, code, costs, baselines, artifact schema, and split manifest; run `verify-panel` before fitting.
3. Count all configurations in the registered budget; a change requires a successor registration.
4. Run development nested walk-forward only.
5. Freeze code/hyperparameters/features/costs/gates in a committed receipt.
6. Open the final holdout once. An interruption consumes it.
7. A pass permits only a disabled challenger; it never changes live authorization or auto-promotes.

## Data handling

Do not commit downloaded market archives or trained artifacts. Store them outside Git with source/terms, retrieval command, schema, point-in-time rules, content hashes, split manifest, and deterministic expected hashes where possible. Never commit credentials or paper PDFs.
