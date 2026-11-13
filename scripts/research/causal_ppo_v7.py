"""Disabled offline causal PPO composition; no frozen learner, persistence or production caller."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as F
import math
import numpy as np
import causal_replay_v3 as replay
import replay_accounting_v2 as accounting
from causal_transition_v4 import REWARD, Transition, transition_v4
from causal_optimizer_v5 import VERSION as OPTIMIZER, Snapshot, create_v5, forward_v5, update_v5
from gae_targets_v2 import VERSION as TARGETS, batch_v2
from ppo_objective_v2 import VERSION as OBJECTIVE, objective_v2

VERSION = "causal-ppo-v7"
BOUND = 2.0 ** 99
TINY = 2.0 ** -1022
MIN_PROB = 2.0 ** -100
GAMMA = 0.99
FLOOR = 1e-08
ROLLOUT = 256
UPDATES = 4
__all__ = ["VERSION", "Outcome", "TrainingResult", "train_causal_ppo_v7"]


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
    objective_version: str
    target_version: str
    optimizer_version: str
    seed: int
    requested_steps: int
    steps: int
    unused_outcomes: int
    stop_reason: str
    session: replay.Session
    transitions: tuple[Transition, ...]
    behavior_probabilities: tuple[float, ...]
    targets: tuple[float, ...]
    advantages: tuple[float, ...]
    losses: tuple[float, ...]
    underflow: int
    actor: Snapshot
    critic: Snapshot


def _number(value: object) -> float:
    if not accounting._valid(value):
        raise ValueError("invalid exact value")
    converted = float(value)
    if (not math.isfinite(converted) or
        (value != 0 and (abs(converted) < TINY or
         abs(F.from_float(converted) - value) > abs(value) * F(1, 2**52)))):
        raise ValueError("unrepresentable exact value")
    return converted


def _bounded(value: object) -> float:
    if type(value) is not float or not math.isfinite(value) or abs(value) > BOUND:
        raise ValueError("value outside the admitted magnitude")
    return value


def _vector(observation: object) -> np.ndarray:
    if (type(observation) is not replay.Observation or type(observation.version) is not str or
        observation.version != replay.OBSERVATION or type(observation.values) is not tuple or
        len(observation.values) != 9):
        raise ValueError("rejected observation")
    return np.asarray([_number(value) for value in observation.values], dtype="float64")


def _values(snapshot: Snapshot, inputs: np.ndarray, width: int) -> tuple:
    result = forward_v5(snapshot, inputs, enabled=True)
    if result is None or result.shape != (len(inputs), width):
        raise ValueError("rejected network evaluation")
    return tuple(tuple(_bounded(float(x)) for x in row) for row in result)


def _policy(row: tuple) -> tuple:
    # Differences only; every action keeps a probability inside the objective's domain.
    m = max(row)
    terms = [math.exp(x - m) for x in row]
    total = math.fsum(terms)
    probs = tuple(t / total for t in terms)
    if not all(MIN_PROB <= p <= 1.0 for p in probs):
        raise ValueError("behavior probability outside the admitted domain")
    return probs


def _collect(session: replay.Session, outcomes: tuple, offset: int, actor: Snapshot,
             rng: np.random.Generator) -> tuple:
    rows, behavior, inputs, rewards = [], [], [], []
    for outcome in outcomes[offset:offset + ROLLOUT]:
        x = _vector(replay.observe_v3(session, enabled=True))
        probs = _policy(_values(actor, x[None, :], 3)[0])
        # One draw per step against the cumulative probabilities.
        u = float(rng.random())
        action = 0 if u < probs[0] else 1 if u < probs[0] + probs[1] else 2
        # No field of this step's outcome is read before the action draw.
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
        behavior.append(probs[action])
        inputs.append(x)
        rewards.append(_bounded(_number(row.reward)))
        if row.done:
            break
    return session, tuple(rows), tuple(behavior), np.asarray(inputs, dtype="float64"), tuple(rewards)


def _targets(rows: tuple, inputs: np.ndarray, rewards: tuple, critic: Snapshot) -> tuple:
    values = _values(critic, inputs, 1)
    # No dummy observation is evaluated for terminal rows.
    active = [i for i, row in enumerate(rows) if not row.done]
    following = {}
    if active:
        batch = np.asarray([_vector(rows[i].next_observation) for i in active], dtype="float64")
        following = dict(zip(active, _values(critic, batch, 1)))
    pairs = batch_v2(tuple((reward, value[0], following[i][0] if i in following else 0.0, row.done)
                           for i, (row, reward, value) in enumerate(zip(rows, rewards, values))),
                     GAMMA, enabled=True)
    if pairs is None:
        raise ValueError("rejected GAE targets")
    raw = [_bounded(pair[0]) for pair in pairs]
    targets = tuple(_bounded(pair[1]) for pair in pairs)
    n = len(raw)
    mean = math.fsum(raw) / n
    centered = [a - mean for a in raw]
    squares = [c * c for c in centered]
    underflow = sum(1 for c, s in zip(centered, squares) if c != 0.0 and s < TINY)
    scale = max(math.sqrt(math.fsum(squares) / n), FLOOR)
    return targets, tuple(_bounded(c / scale) for c in centered), underflow


def _update(actor: Snapshot, critic: Snapshot, inputs: np.ndarray, actions: tuple, behavior: tuple,
            advantages: tuple, targets: tuple) -> tuple:
    objective = objective_v2(_values(actor, inputs, 3), actions, behavior, advantages, enabled=True)
    if objective is None:
        raise ValueError("rejected PPO objective")
    n = len(targets)
    residual = [[_bounded((value[0] - target) / n)] for value, target in zip(_values(critic, inputs, 1), targets)]
    candidate_actor = update_v5(actor, inputs, np.asarray(objective.grad, dtype="float64"), enabled=True)
    candidate_critic = update_v5(critic, inputs, np.asarray(residual, dtype="float64"), enabled=True)
    # Both candidates or neither: a failed stage never leaves the pair half updated.
    if candidate_actor is None or candidate_critic is None:
        raise ValueError("rejected optimizer update")
    return candidate_actor, candidate_critic, objective.loss, objective.underflow


def train_causal_ppo_v7(initial_bars: object, outcomes: object, seed: object, *,
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
            rng = np.random.default_rng([seed, 7])
            transitions, behavior, published, normalized, losses, underflow = [], [], [], [], [], 0
            while len(transitions) < len(outcomes):
                session, rows, probabilities, inputs, rewards = _collect(
                    session, outcomes, len(transitions), actor, rng)
                targets, advantages, lost = _targets(rows, inputs, rewards, critic)
                underflow += lost
                actions = tuple(row.action for row in rows)
                for _ in range(UPDATES):
                    actor, critic, loss, lost = _update(actor, critic, inputs, actions, probabilities,
                                                        advantages, targets)
                    losses.append(loss)
                    underflow += lost
                transitions.extend(rows)
                behavior.extend(probabilities)
                published.extend(targets)
                normalized.extend(advantages)
                if rows[-1].done:
                    break
            reason = "input_exhausted"
            if session.state.terminal:
                reason = transitions[-1].receipt.reason or ("tick_limit" if session.state.tick == 4096 else "terminal")
            count = len(transitions)
            return TrainingResult(VERSION, replay.OBSERVATION, REWARD, OBJECTIVE, TARGETS, OPTIMIZER, seed,
                                  len(outcomes), count, len(outcomes) - count, reason, session,
                                  tuple(transitions), tuple(behavior), tuple(published), tuple(normalized),
                                  tuple(losses), underflow, actor, critic)
    except (ArithmeticError, ValueError, TypeError, MemoryError):
        return None
