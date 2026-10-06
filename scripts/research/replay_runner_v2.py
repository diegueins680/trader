"""Disabled exact episode runner over a fixed target schedule. No policy, data or order interface."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as F

import funding_events_v2 as funding
import replay_accounting_v2 as accounting

VERSION = "replay-runner-v2"
MAX_STEPS = 4096
MAX_BARS = 8192
__all__ = ["VERSION", "Episode", "run_v2"]


@dataclass(frozen=True, slots=True)
class Episode:
    version: str
    left: int
    receipts: tuple
    final: accounting.State


def _admissible(prices: object, buckets: object, left: object, targets: object) -> bool:
    if (type(buckets) is not funding.Buckets or buckets.version != funding.VERSION or
        type(prices) is not tuple or not 1 <= len(prices) <= MAX_BARS or
        len(prices) != len(buckets.closes) or len(buckets.events) != len(prices)):
        return False
    if (type(left) is not int or type(targets) is not tuple or
        not 1 <= len(targets) <= MAX_STEPS or not 0 <= left or
        left + len(targets) >= len(prices)):
        return False
    return all(type(p) is F and p > 0 for p in prices)


def _run(prices: tuple, buckets: funding.Buckets, left: int, targets: tuple,
         fill: object, multiplier: object, impact: object) -> Episode | None:
    state = accounting.initial_v2(prices[left], enabled=True)
    if state is None:
        return None
    receipts = []
    for k in range(1, len(targets) + 1):
        receipt = accounting.advance_v2(state, prices[left + k], buckets.events[left + k],
                                        targets[k - 1], terminal=k == len(targets), fill=fill,
                                        multiplier=multiplier, impact=impact, enabled=True)
        if receipt is None:
            return None
        receipts.append(receipt)
        state = receipt.after
        if state.terminal:
            break
    if not state.terminal:
        return None
    return Episode(VERSION, left, tuple(receipts), state)


def run_v2(prices: object, buckets: object, left: object, targets: object, *,
           fill: object = F(1), multiplier: object = F(1), impact: object = F(0),
           enabled: object = False, version: object = VERSION) -> Episode | None:
    if enabled is not True or type(version) is not str or version != VERSION:
        return None
    try:
        if not _admissible(prices, buckets, left, targets):
            return None
        return _run(prices, buckets, left, targets, fill, multiplier, impact)
    except (ArithmeticError, ValueError, TypeError, MemoryError):
        return None
