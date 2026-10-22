"""Disabled offline causal PPO composition; no persistence or production caller."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as F
from math import isfinite
import sys
import numpy as np
import causal_replay_v3 as replay
import replay_accounting_v2 as accounting
from causal_transition_v4 import REWARD, Transition, transition_v4
from causal_optimizer_v5 import Snapshot, create_v5, forward_v5, update_v5
from gae_targets_v2 import batch_v2
from sequential_learning import ppo_gradient, softmax

VERSION = "causal-ppo-v5"
__all__ = ["VERSION", "Outcome", "TrainingResult", "encode_v5", "train_causal_ppo_v5"]


@dataclass(frozen=True, slots=True)
class Outcome:
    bar: replay.Bar
    events: tuple
    terminal: bool = False
    fill: F = F(1)
    multiplier: F = F(1)
    impact: F = F(0)


@dataclass(frozen=True, slots=True)
class TrainingResult:
    version: str
    observation_version: str
    reward_version: str
    seed: int
    requested_steps: int
    steps: int
    unused_outcomes: int
    stop_reason: str
    session: replay.Session
    transitions: tuple[Transition, ...]
    behavior_probabilities: tuple[float, ...]
    actor: Snapshot
    critic: Snapshot
    losses: tuple[float, ...]


def _number(value: object) -> float:
    if not accounting._valid(value):
        raise ValueError("invalid exact value")
    converted = float(value)
    if (not isfinite(converted) or
        (value != 0 and (abs(converted) < sys.float_info.min or
         abs(F.from_float(converted) - value) > abs(value) * F(1, 2**52)))):
        raise ValueError("unrepresentable exact value")
    return converted


def encode_v5(observation: object, *, enabled: object = False,
              version: object = VERSION) -> tuple[float, ...] | None:
    if enabled is not True or type(version) is not str or version != VERSION:
        return None
    try:
        if (type(observation) is not replay.Observation or
            type(observation.version) is not str or observation.version != replay.OBSERVATION or
            type(observation.decision) is not int or not 0 <= observation.decision < 2**63 or
            type(observation.supported) is not bool or type(observation.values) is not tuple or
            len(observation.values) != 9):
            return None
        return tuple(_number(value) for value in observation.values)
    except (ArithmeticError, ValueError, TypeError, MemoryError):
        return None


def _vector(observation: replay.Observation) -> np.ndarray:
    encoded = encode_v5(observation, enabled=True)
    if encoded is None:
        raise ValueError("rejected observation encoding")
    return np.asarray(encoded, dtype="float64")


def _forward(snapshot: Snapshot, inputs: np.ndarray) -> np.ndarray:
    result = forward_v5(snapshot, inputs, enabled=True)
    if result is None:
        raise ValueError("rejected network evaluation")
    return result


def _finite(value: object, shape: tuple) -> bool:
    return (type(value) is np.ndarray and value.dtype == np.dtype("float64") and
            value.shape == shape and bool(np.isfinite(value).all()))


def _collect(session: replay.Session, outcomes: tuple, offset: int, actor: Snapshot,
             rng: np.random.Generator) -> tuple:
    rows, probabilities, inputs, rewards = [], [], [], []
    for outcome in outcomes[offset:offset + 256]:
        observation = replay.observe_v3(session, enabled=True)
        x = _vector(observation)
        probs = softmax(_forward(actor, x[None, :]))[0]
        if not _finite(probs, (3,)) or np.any(probs < sys.float_info.min) or np.any(probs > 1):
            raise ValueError("invalid behavior probabilities")
        action = int(rng.choice(3, p=probs))
        # No fields of the next outcome are consumed before the action draw.
        if type(outcome) is not Outcome:
            raise ValueError("invalid outcome")
        advanced = transition_v4(session, outcome.bar, outcome.events, action,
                                 terminal=outcome.terminal, fill=outcome.fill,
                                 multiplier=outcome.multiplier, impact=outcome.impact,
                                 enabled=True)
        if advanced is None:
            raise ValueError("rejected causal transition")
        session, row = advanced
        rows.append(row)
        probabilities.append(float(probs[action]))
        inputs.append(x)
        rewards.append(_number(row.reward))
        if row.done:
            break
    return (session, tuple(rows), tuple(probabilities),
            np.asarray(inputs, dtype="float64"), np.asarray(rewards, dtype="float64"))


def _targets(rows: tuple, inputs: np.ndarray, rewards: np.ndarray,
             critic: Snapshot) -> tuple[np.ndarray, np.ndarray]:
    values = _forward(critic, inputs).ravel()
    # No dummy observation is evaluated for terminal rows.
    following = np.zeros(len(rows), dtype="float64")
    active = [i for i, row in enumerate(rows) if not row.done]
    if active:
        following[active] = _forward(critic, np.asarray(
            [_vector(rows[i].next_observation) for i in active], dtype="float64")).ravel()
    pairs = batch_v2(tuple((float(r), float(v), float(n), row.done)
                          for r, v, n, row in zip(rewards, values, following, rows)),
                     0.99, enabled=True)
    if pairs is None:
        raise ValueError("rejected GAE targets")
    advantages, targets = np.asarray(pairs, dtype="float64").T
    advantages = (advantages - advantages.mean()) / max(float(advantages.std()), 1e-8)
    if not _finite(advantages, (len(rows),)) or not _finite(targets, (len(rows),)):
        raise ValueError("non-finite normalized targets")
    return advantages, targets


def _update(actor: Snapshot, critic: Snapshot, inputs: np.ndarray, rows: tuple,
            probabilities: tuple, advantages: np.ndarray, targets: np.ndarray) -> tuple:
    count = len(rows)
    actions = np.asarray([row.action for row in rows], dtype="int64")
    loss, gradient = ppo_gradient(_forward(actor, inputs), actions,
                                  np.asarray(probabilities, dtype="float64"), advantages)
    if type(loss) is not float or not isfinite(loss) or not _finite(gradient, (count, 3)):
        raise ValueError("non-finite PPO objective")
    candidate_actor = update_v5(actor, inputs, gradient, enabled=True)
    residual = (_forward(critic, inputs).ravel() - targets) / count
    if not _finite(residual, (count,)):
        raise ValueError("non-finite critic residual")
    candidate_critic = update_v5(critic, inputs, residual[:, None], enabled=True)
    if candidate_actor is None or candidate_critic is None:
        raise ValueError("rejected optimizer update")
    return candidate_actor, candidate_critic, loss


def train_causal_ppo_v5(initial_bars: object, outcomes: object, seed: object, *,
                        enabled: object = False, version: object = VERSION) -> TrainingResult | None:
    if (enabled is not True or type(version) is not str or version != VERSION or
        type(seed) is not int or not 0 <= seed < 2**32 - 1 or
        type(outcomes) is not tuple or not 1 <= len(outcomes) <= 4096):
        return None
    try:
        with np.errstate(all="raise"):
            session = replay.start_v3(initial_bars, enabled=True)
            actor, critic = create_v5(seed, 3, enabled=True), create_v5(seed + 1, 1, enabled=True)
            if session is None or actor is None or critic is None:
                return None
            rng = np.random.default_rng(seed)
            transitions, behavior, losses = [], [], []
            while len(transitions) < len(outcomes):
                session, rows, probabilities, inputs, rewards = _collect(
                    session, outcomes, len(transitions), actor, rng)
                advantages, targets = _targets(rows, inputs, rewards, critic)
                for _ in range(4):
                    actor, critic, loss = _update(actor, critic, inputs, rows, probabilities, advantages, targets)
                    losses.append(loss)
                transitions.extend(rows)
                behavior.extend(probabilities)
                if rows[-1].done:
                    break
            reason = "input_exhausted"
            if session.state.terminal:
                reason = transitions[-1].receipt.reason or ("tick_limit" if session.state.tick == 4096 else "terminal")
            count = len(transitions)
            return TrainingResult(VERSION, replay.OBSERVATION, REWARD, seed, len(outcomes), count,
                                  len(outcomes) - count, reason, session, tuple(transitions),
                                  tuple(behavior), actor, critic, tuple(losses))
    except (ArithmeticError, ValueError, TypeError, MemoryError):
        return None
