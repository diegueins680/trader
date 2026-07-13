# Formal Methods: Gate Telemetry System

## Module
`Trader.GateTelemetry`

## Purpose
Provide structured, measurable observability for every signal gate rejection.

Current conformance scope: this contract verifies the standalone accumulator. The accumulator is not yet threaded through every `Main`/`Trading`/`SignalGates` production path, so repository-wide rejection capture remains an explicit integration gap.

## Invariants

### I1: Observability Does Not Affect Logic
**Statement**: For all gate checks `g`, `gate_result(g) = gate_result(g_without_telemetry)`.
**Proof**: The telemetry accumulator is passed as a separate argument and is never read during gate logic evaluation. The `recordRejection` function is only called AFTER the gate decision is made.

### I2: Telemetry Accumulation and Cardinality are Bounded
**Statement**: Recording a rejection takes `O(log k + maxRecent)`, where `k` is the finite set of canonical gate/reason pairs and `maxRecent` is capped at `1000`.
**Proof**: `recordRejectionWithContext` performs:
- strict `Map.insertWith` operations in `O(log k)`;
- a prepend plus `take maxRecent` in `O(maxRecent)`;
- canonicalization of every `ReasonUnknown` value to one `UNKNOWN` bucket, preventing attacker-controlled reason cardinality;
- normalization of the recent-event bound into `[0,1000]`.

### I3: Fail-Closed by Default
**Statement**: `emptyTelemetry` produces a telemetry object with zero counts and no binding gate.
**Proof**: `emptyTelemetry` initializes all counters to 0 and all Maps to `Map.empty`. `bindingGate` on an empty Map returns `Nothing`.

### I4: Binding Gate Identifies the Bottleneck
**Statement**: The binding gate is the gate with the highest rejection count.
**Proof**: `bindingGate` sorts gates by count descending and returns the first (maximal) element. This is the gate that most frequently rejected candidates.

## Failure Modes

### F1: Telemetry Memory Leak
**Condition**: `maxRecent` is negative or very large, or untrusted unknown-reason text has unbounded cardinality.
**Mitigation**: Bounds normalize into `[0,1000]`, and unknown reasons collapse to one stable bucket before histogram/recent storage.

### F2: Concurrent Access Corruption
**Condition**: Multiple threads update the same `GateTelemetry` without synchronization.
**Mitigation**: Use `MVar GateTelemetry` or `TVar GateTelemetry` in concurrent contexts. The pure functions are thread-safe; the accumulator must be serialized.

## Metrics
- `gate_rejection_histogram`: Count per (gate, reason)
- `binding_gate`: Most frequent rejector
- `rejection_rate`: rejections / candidates
- `diagnosis`: NO_CANDIDATES | ALL_REJECTED | MOSTLY_REJECTED | NORMAL

## Validation
See `test/TestMain.hs`:
- `testGateTelemetryEmptyInvariant`
- `testGateTelemetryAccumulationInvariant`
- `testGateTelemetryBindingGateIdentification`
- `testGateTelemetryHistogramSorting`
- `Trader.Test.FormalVerification.formalVerificationSuite` covers negative/huge recent-history bounds and canonical unknown-reason cardinality.
