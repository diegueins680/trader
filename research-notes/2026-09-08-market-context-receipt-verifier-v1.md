# Market-context receipt verifier v1

Date: 2026-09-08
Status: implemented and tested with synthetic fixtures; no real acquisition or research admission

## Decision

Add a separate offline, read-only replay verifier for persisted `binance_usdm_market_context_bundle_receipt_v1` evidence. The bundle verifier proves integrity when it runs, but a saved receipt also needs a deterministic path back to the exact frozen external archive. Without replay, later review could not distinguish a still-valid receipt from one whose archive, verifier version, or authority fields had changed.

This closes an evidence-lifecycle gap, not a model gap. No predictor is integrated, no outcome is evaluated, and the current champion remains unchanged.

## Contract

Run:

```bash
python3 scripts/research/verify_market_context_receipt.py \
  --receipt <saved-bundle-receipt.json> \
  --archive <frozen-bundle>
```

The bundle verifier first refuses to produce a receipt unless both its executing file and the imported source verifier exactly match their versions at the collection commit. The archive must then contain exactly `collection-status.json`, `source-manifest.json`, their registered raw files, and the required directory structure. Receipts and derived panels remain outside that frozen acquisition archive. The replay verifier:

- strictly rejects duplicate keys, non-finite constants, extra fields, type confusion, malformed hashes, unsafe symlinks, and any non-zero downstream authority;
- requires the receipt's bundle-verifier hash to match `scripts/research/verify_market_context_bundle.py` at the collection commit and the currently executing bundle-verifier bytes;
- hashes every archive file and validates the exact inventory before replay;
- delegates complete status, Git provenance, raw response, population, causal-clock, and panel reconstruction to the bundle and source verifiers;
- compares the complete recomputed receipt with type-sensitive JSON equality and independently confirms the derived-panel hash; and
- re-hashes the exact archive after replay so a same-size content mutation cannot pass as stable evidence.

Verifier evolution is fail closed. If the current bundle-verifier bytes do not match the receipt, replay requires checking out the recorded collection commit rather than silently interpreting the old receipt under new code.

## Output semantics

Success emits one deterministic JSON summary with receipt, status, manifest, panel, archive-file, and archive-byte identities. It sets `researchAdmission`, `experimentUse`, `holdoutUse`, `modelUse`, `promotionUse`, `deploymentUse`, `orderUse`, and `liveAuthorizationUse` to false.

Replay proves frozen acquisition integrity only. It does not admit the data, modify an experiment manifest, open a holdout, fit a model, calculate performance, promote a challenger, deploy code, or authorize an order.

## Evidence and remaining blockers

Synthetic tests cover exact replay, determinism, producer-side verifier-version drift, raw/status/receipt tampering, extra archive files, forged verifier and panel hashes, boolean/integer confusion, authority escalation, unknown fields, duplicate/non-finite JSON, symlinked receipt/archive paths, and receipts incorrectly placed inside the archive.

No public endpoint was contacted. No market data, return, forecast, PnL, cost, statistical diagnostic, model, artifact, experiment, or holdout was accessed. Collection remains prohibited before 2027-01-21. Sufficient prospective history, explicit data admission, remaining timestamp-preserving inputs, and the versioned fold-local fitted-artifact boundary remain open. `.env.example` is unchanged because replay adds no configuration.
