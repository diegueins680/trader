# Formal Methods: Gate Telemetry System

## Module
`Trader.GateTelemetry`

## Purpose
Provide structured, measurable observability for every signal gate rejection.

## Invariants

### I1: Observability Does Not Affect Logic
**Statement**: For all gate checks `g`, `gate_result(g) = gate_result(g_without_telemetry)`.
**Proof**: The telemetry accumulator is passed as a separate argument and is never read during gate logic evaluation. The `recordRejection` function is only called AFTER the gate decision is made.

### I2: Telemetry Accumulation is O(1)
**Statement**: Recording a rejection takes constant time regardless of history size.
**Proof**: `recordRejectionWithContext` performs:
- One `Map.insertWith` on a strict Map (O(log n) where n = number of distinct gate/reason pairs, bounded by ~50)
- One list prepend for recent rejections (O(1))
- The recent rejection list is bounded by `maxRecent` (default 100), so memory is O(1)

### I3: Fail-Closed by Default
**Statement**: `emptyTelemetry` produces a telemetry object with zero counts and no binding gate.
**Proof**: `emptyTelemetry` initializes all counters to 0 and all Maps to `Map.empty`. `bindingGate` on an empty Map returns `Nothing`.

### I4: Binding Gate Identifies the Bottleneck
**Statement**: The binding gate is the gate with the highest rejection count.
**Proof**: `bindingGate` sorts gates by count descending and returns the first (maximal) element. This is the gate that most frequently rejected candidates.

## Failure Modes

### F1: Telemetry Memory Leak
**Condition**: `maxRecent` is set to a very large value.
**Mitigation**: Default is 100. Production should cap at 1000.

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
