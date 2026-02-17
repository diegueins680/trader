# Distributed Architecture Plan for `trader`

This plan turns the current API + stateful bot process into a multi-instance system where instances coordinate safely across tenants/symbols.

## 1. Current constraints to fix

- `/bot/*` and async jobs currently assume local/shared filesystem state (`TRADER_STATE_DIR` / async dir).
- Stateful bot loops can conflict when multiple API instances run behind a load balancer.
- Trade/order side effects need stronger idempotency across retries/restarts.

## 2. Target runtime components

1. `api-gateway`
- Keep existing HTTP surface (`/signal`, `/trade`, `/backtest`, `/bot/*`, `/ops`, `/state/sync`).
- Validate/auth/normalize requests, then publish command events.
- Return `jobId` for async flows.

2. `job-orchestrator`
- Consumes async job requests and dispatches to workers.
- Tracks job lifecycle in Postgres (`queued/running/succeeded/failed`).

3. `market-data-ingestor`
- Owns exchange WS/REST ingestion.
- Publishes normalized candles/ticks.

4. `feature-predictor-worker`
- Builds features and predictor outputs (Kalman/LSTM/other predictors).
- Publishes `signal.generated` events.

5. `bot-supervisor`
- Runs continuous bot loops by partition ownership.
- Emits status snapshots and heartbeats.

6. `execution-worker`
- Applies risk checks, position checks, and sends exchange orders.
- Uses strict idempotency + outbox pattern.

7. `state-sync-worker`
- Handles `/state/sync` import/export via object store + DB metadata.

## 3. Data/control planes

- Control plane: HTTP/gRPC command-response (`api-gateway` to internal services).
- Data plane: Kafka event streams for candles/signals/orders/status.

Use:
- Kafka (durable event log, consumer groups, replay).
- Postgres (source of truth: tenants, ops, jobs, order intents, bot leases, position origins).
- Redis (optional): hot cache, rate limits, short-lived locks.
- S3 (or compatible): large snapshots/artifacts (bot snapshots, top-combos history, model artifacts).

## 4. Partitioning and ownership

Partition key:
- `ownerKey = tenantKey + ":" + platform + ":" + symbol + ":" + market + ":" + interval`

Rules:
- All stateful processing for an `ownerKey` is serialized through one Kafka partition.
- Consumer groups distribute different owner keys across instances.
- `bot-supervisor` acquires a Postgres lease per `ownerKey` (`SELECT ... FOR UPDATE SKIP LOCKED` or advisory lock + TTL).

Lease table:
- `bot_leases(owner_key PK, holder_id, lease_until_ms, version)`
- Renew every 5-10s; treat expired lease as transferable.

## 5. Event topics (v1)

1. `trader.v1.jobs.requested`
- key: `jobId`
- retention: 7d

2. `trader.v1.jobs.progress`
- key: `jobId`
- retention: 7d

3. `trader.v1.market.candles`
- key: `ownerKey`
- compaction+retention: compact, 3d

4. `trader.v1.signals.generated`
- key: `ownerKey`
- retention: 3d

5. `trader.v1.orders.intents`
- key: `ownerKey`
- retention: 14d

6. `trader.v1.orders.submitted`
- key: `ownerKey`
- retention: 14d

7. `trader.v1.orders.fills`
- key: `ownerKey`
- retention: 30d

8. `trader.v1.bot.status`
- key: `ownerKey`
- compaction+retention: compact, 7d

9. `trader.v1.dlq`
- key: original topic key
- retention: 30d

## 6. Message contract (all events)

```json
{
  "schemaVersion": 1,
  "eventType": "orders.intent.created",
  "eventId": "uuid-v7",
  "traceId": "uuid-v7",
  "idempotencyKey": "tenant:symbol:barTs:action",
  "producedAtMs": 1760000000000,
  "producer": "execution-worker",
  "ownerKey": "tenant:binance:BTCUSDT:spot:1m",
  "payload": {}
}
```

Contract rules:
- `eventId` globally unique.
- `idempotencyKey` stable across retries.
- Schema evolution is backward-compatible (`schemaVersion` + additive fields only for minor changes).

## 7. Core schemas

`jobs.requested.payload`
```json
{
  "jobId": "uuid-v7",
  "jobType": "signal|trade|backtest|optimizer",
  "requestedBy": "tenantKey",
  "request": { "...": "original API request" }
}
```

`signals.generated.payload`
```json
{
  "symbol": "BTCUSDT",
  "interval": "1m",
  "priceTsMs": 1760000000000,
  "method": "bandit_router",
  "direction": "BUY|SELL|HOLD",
  "confidence": 0.73,
  "openThreshold": 0.002,
  "closeThreshold": 0.001,
  "predictedNext": 65321.42
}
```

`orders.intents.payload`
```json
{
  "intentId": "uuid-v7",
  "symbol": "BTCUSDT",
  "side": "BUY|SELL",
  "market": "spot|futures|margin",
  "size": { "type": "quote|base", "value": 100.0 },
  "riskChecks": {
    "maxNotionalOk": true,
    "positionLimitOk": true,
    "minSignalToNoiseOk": true
  },
  "exchange": "binance",
  "mode": "test|live"
}
```

`orders.submitted.payload`
```json
{
  "intentId": "uuid-v7",
  "exchangeOrderId": "123456789",
  "clientOrderId": "idempotency-key",
  "submittedAtMs": 1760000000500,
  "requestHash": "sha256:..."
}
```

`orders.fills.payload`
```json
{
  "exchangeOrderId": "123456789",
  "fillTsMs": 1760000001200,
  "avgPrice": 65325.1,
  "filledQty": 0.0015,
  "fee": 0.06,
  "feeAsset": "USDT",
  "status": "PARTIALLY_FILLED|FILLED|CANCELED|REJECTED"
}
```

## 8. Reliability patterns (required)

1. Idempotency
- DB unique constraints on `(tenant_key, idempotency_key)` for order intents.
- Store exchange `clientOrderId` and map retries to the same intent.

2. Transactional outbox
- Write business row + outbox row in one Postgres transaction.
- Publisher relays outbox to Kafka; retries are safe.

3. Retry strategy
- Retry transient errors with exponential backoff + jitter.
- Non-retryable errors go straight to DLQ with error metadata.

4. Timeouts/circuit breakers
- Per-exchange timeout budget (connect/read/write).
- Open circuit on repeated 5xx/timeouts; half-open probes.

5. Graceful shutdown
- Stop accepting new work.
- Commit processed offsets.
- Release leases.

## 9. Collaboration between instances

- Same service instances collaborate via Kafka consumer groups.
- Different services collaborate via versioned events + trace IDs.
- Each instance sends heartbeat (`trader.v1.bot.status` + metrics labels: `instance_id`, `consumer_group`, `partition`).
- Failover occurs by lease expiry + partition rebalance; no manual intervention.

## 10. Observability SLOs

Track:
- Kafka lag per topic/group/partition.
- Job queue age p95.
- Signal-to-order latency p95.
- Order reject rate and retry rate.
- Lease steal/rebalance frequency.
- Bot loop staleness (`now - last_bar_ts`).

Tracing:
- Propagate `traceId` from HTTP request to all produced events and exchange calls.

## 11. Security model

- Keep tenant isolation as first-class key in every table/event.
- mTLS between internal services.
- Secrets only from env/secret manager, never events/logs.
- Encrypt API key material at rest (KMS envelope encryption).

## 12. Rollout plan (phased)

1. Phase 0: hardening current single service
- Move async job state from filesystem-only to Postgres table.
- Add idempotency table + unique constraints for `/trade` and `/trade/async`.
- Add structured `traceId` logging.

2. Phase 1: introduce Kafka + outbox
- Add outbox tables and publisher worker inside current binary.
- Emit events for signal/trade/bot status while keeping existing synchronous behavior.

3. Phase 2: split execution path
- Extract `execution-worker` consuming `orders.intents`.
- API now submits intent and waits/polls result.

4. Phase 3: distributed bot supervisor
- Replace in-process per-instance bot loops with lease-based `bot-supervisor` workers.
- `/bot/start|stop|status` becomes command/query over DB + event stream.

5. Phase 4: full multi-instance deployment
- Run N replicas each for `api-gateway`, `bot-supervisor`, `execution-worker`, `feature-predictor-worker`.
- Enable autoscaling on lag + CPU + latency.

6. Phase 5: deprecate legacy local state
- Remove dependency on shared filesystem for async/bot persistence.
- Keep object store for snapshots/artifacts only.

## 13. Minimal first implementation checklist

- [ ] Add tables: `async_jobs`, `order_intents`, `outbox_events`, `bot_leases`.
- [ ] Add idempotency key generation/validation in `/trade` and bot order paths.
- [ ] Add outbox writer helpers around existing DB writes.
- [ ] Add a small background publisher process.
- [ ] Add consumer skeleton for `orders.intents` in a separate executable.
- [ ] Add metrics for lag, lease health, and idempotency hit rate.

