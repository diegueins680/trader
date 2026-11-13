"""Deterministic exposure shield (exposure_shield_v1), integer arithmetic only.

Exposure is a signed integer in units of 1e-4 of account equity. The learned
policy is untrusted: its raw output is decoded, validated and projected here
before any simulated intent exists. This module mirrors
``haskell/app/Trader/Research/ExposureShieldV1.hs``; the two are compared
case-by-case by ``scripts/formal/exposure_shield.py``.

Guarantees are action-admission invariants over a validated snapshot. They say
nothing about future drawdown, solvency or realised loss.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

VERSION = "exposure_shield_v1"

SCALE = 10_000  # exposure units per unit of equity
CAP = 1_666  # maximum admitted |target exposure|
MIN_TRADE = 25  # smallest admitted |target - current|
DD_LIMIT = 1_000  # drawdown (1e-4 of peak equity) at which only reductions pass
SNAPSHOT_BOUND = 100_000  # |current exposure| beyond this is not a valid snapshot

# Outcome kinds.
PERMITTED = 0  # a research intent to move exposure to ``target``
ABSTAIN = 1  # valid proposal, no intent (change below MIN_TRADE)
REJECTED = 2  # invalid proposal; state-dependent fallback applied
RECOVERY_REQUIRED = 3  # invalid snapshot; no intent may be formed

# Intervention reason bits (logged, never used for control decisions).
R_CAP = 1
R_HEADROOM = 2
R_RISK_OFF = 4
R_MIN_TRADE = 16
R_INVALID_PROPOSAL = 32
R_INVALID_SNAPSHOT = 64


class Snapshot(NamedTuple):
    exposure: int  # current signed exposure, rounded away from zero
    drawdown: int  # current drawdown from peak, rounded up
    fresh: bool  # account state is within its freshness bound
    headroom: int  # additional |exposure| this account may reserve


class Outcome(NamedTuple):
    kind: int
    target: int  # exposure after the decision; equals snapshot.exposure if no intent
    reasons: int


def _clamp(value: int, low: int, high: int) -> int:
    return low if value < low else high if value > high else value


def valid_snapshot(snapshot: Snapshot) -> bool:
    return (
        type(snapshot.exposure) is int
        and type(snapshot.drawdown) is int
        and type(snapshot.headroom) is int
        and type(snapshot.fresh) is bool
        and -SNAPSHOT_BOUND <= snapshot.exposure <= SNAPSHOT_BOUND
        and 0 <= snapshot.drawdown <= SCALE
        and 0 <= snapshot.headroom <= SCALE
    )


def decode(raw: object) -> int | None:
    """Decode a raw policy output in [-1, 1] to units, truncating toward zero."""
    if type(raw) is not float or not math.isfinite(raw) or not -1.0 <= raw <= 1.0:
        return None
    return int(raw * CAP)


def project(snapshot: Snapshot, proposal: int) -> tuple[int, int]:
    """Project an integer proposal onto the admitted set. Total on valid snapshots."""
    current = snapshot.exposure
    reasons = 0
    target = _clamp(proposal, -CAP, CAP)
    if target != proposal:
        reasons |= R_CAP
    limit = min(CAP, min(abs(current), CAP) + snapshot.headroom)
    limited = _clamp(target, -limit, limit)
    if limited != target:
        reasons |= R_HEADROOM
    target = limited
    if (not snapshot.fresh) or snapshot.drawdown >= DD_LIMIT:
        # Reduce-only: never increase |exposure| and never cross through zero.
        if current > 0:
            reduced = _clamp(target, 0, current)
        elif current < 0:
            reduced = _clamp(target, current, 0)
        else:
            reduced = 0
        if reduced != target:
            reasons |= R_RISK_OFF
        target = reduced
    return target, reasons


def shield(snapshot: Snapshot, raw: object) -> Outcome:
    """shield(validated snapshot, proposed action) -> outcome. Total."""
    if not valid_snapshot(snapshot):
        exposure = snapshot.exposure if type(snapshot.exposure) is int else 0
        return Outcome(RECOVERY_REQUIRED, exposure, R_INVALID_SNAPSHOT)
    current = snapshot.exposure
    proposal = decode(raw)
    invalid = proposal is None
    if invalid:
        # State-dependent fallback: propose holding the current exposure. The
        # projection still forces a reduction to the cap when over it.
        proposal = current
    target, reasons = project(snapshot, proposal)
    if invalid:
        reasons |= R_INVALID_PROPOSAL
    if abs(target - current) < MIN_TRADE:
        # A change below the minimum tradable size is never rounded up.
        if target != current:
            reasons |= R_MIN_TRADE
        return Outcome(REJECTED if invalid else ABSTAIN, current, reasons)
    return Outcome(REJECTED if invalid else PERMITTED, target, reasons)


def reserve(budget: int, held: tuple[int, ...], version: int, seen_version: int, index: int, wanted: int):
    """Atomic compare-and-swap reservation of |exposure| against a shared budget.

    Returns ``(held, version, granted)``. A stale ``seen_version`` or a request
    that would exceed the budget leaves the state unchanged.
    """
    if seen_version != version or wanted < 0 or not 0 <= index < len(held):
        return held, version, False
    total = sum(held) - held[index] + wanted
    if total > budget and wanted > held[index]:
        return held, version, False
    updated = held[:index] + (wanted,) + held[index + 1 :]
    return updated, version + 1, True


def shield_batch(exposure, drawdown, fresh, headroom, raw):
    """Vectorised ``shield`` for valid snapshots. Returns (kind, target, reasons)."""
    exposure = np.asarray(exposure, dtype=np.int64)
    drawdown = np.asarray(drawdown, dtype=np.int64)
    headroom = np.asarray(headroom, dtype=np.int64)
    fresh = np.asarray(fresh, dtype=bool)
    raw = np.asarray(raw, dtype=np.float64)
    bad_snapshot = (np.abs(exposure) > SNAPSHOT_BOUND) | (drawdown < 0) | (drawdown > SCALE) | (headroom < 0) | (headroom > SCALE)
    with np.errstate(invalid="ignore"):
        invalid = ~(np.isfinite(raw) & (raw >= -1.0) & (raw <= 1.0))
    safe_raw = np.where(invalid, 0.0, raw)
    proposal = np.where(invalid, exposure, np.trunc(safe_raw * CAP).astype(np.int64))
    reasons = np.where(invalid, R_INVALID_PROPOSAL, 0).astype(np.int64)
    target = np.clip(proposal, -CAP, CAP)
    reasons |= np.where(target != proposal, R_CAP, 0)
    limit = np.minimum(CAP, np.minimum(np.abs(exposure), CAP) + headroom)
    limited = np.clip(target, -limit, limit)
    reasons |= np.where(limited != target, R_HEADROOM, 0)
    target = limited
    risk_off = (~fresh) | (drawdown >= DD_LIMIT)
    reduced = np.where(
        exposure > 0,
        np.clip(target, 0, np.maximum(exposure, 0)),
        np.where(exposure < 0, np.clip(target, np.minimum(exposure, 0), 0), 0),
    )
    reasons |= np.where(risk_off & (reduced != target), R_RISK_OFF, 0)
    target = np.where(risk_off, reduced, target)
    small = np.abs(target - exposure) < MIN_TRADE
    reasons |= np.where(small & (target != exposure), R_MIN_TRADE, 0)
    kind = np.where(invalid, REJECTED, np.where(small, ABSTAIN, PERMITTED))
    target = np.where(small, exposure, target)
    kind = np.where(bad_snapshot, RECOVERY_REQUIRED, kind)
    target = np.where(bad_snapshot, exposure, target)
    reasons = np.where(bad_snapshot, R_INVALID_SNAPSHOT, reasons)
    return kind.astype(np.int64), target.astype(np.int64), reasons.astype(np.int64)
