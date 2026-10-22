"""Disabled exact transition adapter; no learner, callback or order interface."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as F
import causal_replay_v3 as replay
import replay_accounting_v2 as accounting

VERSION = "causal-transition-v4"
REWARD = "net-equity-delta-v4"
TARGETS = (F(-1, 4), F(0), F(1, 4))
__all__ = ["VERSION", "REWARD", "Transition", "transition_v4"]


@dataclass(frozen=True, slots=True)
class Transition:
    version: str
    reward_version: str
    observation: replay.Observation
    action: int
    replay_target: F
    receipt: accounting.Receipt
    reward: F
    next_observation: replay.Observation | None
    done: bool


def transition_v4(session: object, bar: object, events: object, action: object, *,
                  terminal: object = False, fill: object = F(1),
                  multiplier: object = F(1), impact: object = F(0),
                  enabled: object = False, version: object = VERSION
                  ) -> tuple[replay.Session, Transition] | None:
    if (enabled is not True or type(version) is not str or version != VERSION or
        type(action) is not int or not 0 <= action < len(TARGETS)):
        return None
    try:
        observation = replay.observe_v3(session, enabled=True)
        if observation is None:
            return None
        requested = TARGETS[action]
        result = replay.step_v3(session, bar, events, requested, terminal=terminal,
                                fill=fill, multiplier=multiplier, impact=impact,
                                enabled=True)
        if result is None:
            return None
        following, receipt = result
        done = receipt.after.terminal
        next_observation = None if done else replay.observe_v3(following, enabled=True)
        if not done and next_observation is None:
            return None
        reward = accounting._add(receipt.after.equity, -receipt.before.equity)
        sample = Transition(VERSION, REWARD, observation, action,
                            requested if observation.supported else F(0),
                            receipt, reward, next_observation, done)
        return following, sample
    except (ArithmeticError, ValueError, TypeError, MemoryError):
        return None
