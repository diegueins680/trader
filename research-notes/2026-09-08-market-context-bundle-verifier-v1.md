# Market-context bundle verifier v1

Date: 2026-09-08
Status: implemented and tested with synthetic fixtures; no real acquisition or research admission

## Decision

Add a separate offline verifier for collector-produced USDⓈ-M market-context bundles. The earlier source verifier proves that registered raw public responses reconstruct a deterministic panel, but it can be called directly on a manifest and therefore does not prove that the collector published a complete success status. The new wrapper closes that integrity gap without changing the research decision: no candidate passed, the current champion is unchanged, and no candidate gains model, promotion, deployment, order, or live authority.

## Contract

Run:

```bash
python3 scripts/research/verify_market_context_bundle.py verify \
  --status <frozen-bundle>/collection-status.json \
  --panel-output <output>/market-context.csv \
  --receipt-output <output>/bundle-receipt.json
```

The command is offline and accepts no URL, credential, outcome, model, or trading input. It requires:

- the exact schema-1 `complete_unverified` status with `sourceManifestPublished: true`, clean tracked provenance, and every collector authority field false;
- strict duplicate-key and non-finite rejection for status, manifest, and delegated source evidence;
- a regular non-symlink status named `collection-status.json` and a safe sibling source manifest;
- matching status/manifest hashes, code commit, causal clocks, eligible-population count, and exact ordered raw-artifact inventory;
- a 40-hex Git object whose type is `commit`;
- historical collector and source-verifier blobs whose SHA-256 values match the status, plus exactly one public read-only Binance source/license record at that commit; and
- successful independent raw-response and panel reconstruction by `market_context_source.py`, with its returned manifest commit and digest matching the status-bound bytes.

`collecting`, `partial_failure`, `cleanup_pending`, `cleanup_failure`, absent, malformed, changed, authorizing, or ambiguous evidence is rejected. Requested outputs cannot overwrite the status, manifest, or raw directory.

## Receipt semantics

The deterministic `binance_usdm_market_context_bundle_receipt_v1` records hashes for the status, manifest, historical provenance blobs, embedded source-verification receipt, derived panel, and current bundle verifier. It sets `researchAdmission`, `experimentUse`, `holdoutUse`, `modelUse`, `promotionUse`, `deploymentUse`, `orderUse`, and `liveAuthorizationUse` to false.

This receipt proves only that the frozen collector and source evidence passed the implemented integrity contract at verification time. It is not a data-admission record. Future research still requires the registered 2027-01-21-or-later collection window, an explicit experiment-manifest update, separate admission review, sufficient prospective history, and the versioned fitted-artifact boundary. The untouched final holdout remains sealed.

## Verification evidence

The focused regression suite uses only synthetic fixture bytes and temporary directories. It proves deterministic success, status and manifest binding, historical Git provenance, independent panel reconstruction, downstream-authority rejection, all collector failure-state rejection, duplicate-key and non-finite rejection, symlink rejection, integer population semantics, incomplete inventory rejection, and frozen-input overwrite protection.

No request to Binance or any other market-data provider was made. No outcome, return, model, experiment, or holdout was evaluated. `.env.example` is unchanged because the offline verifier adds no configuration.
