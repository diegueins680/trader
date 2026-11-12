"""Disabled offline causal Double-DQN composition; no frozen learner, persistence or production caller."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as F
import math
import numpy as np
import causal_replay_v3 as replay
import replay_accounting_v2 as accounting
from causal_transition_v4 import REWARD, Transition, transition_v4
from causal_optimizer_v5 import VERSION as OPTIMIZER, Snapshot, create_v5, forward_v5, update_v5
from value_objective_v2 import VERSION as OBJECTIVE, objective_v2

VERSION = "causal-value-v6"
BOUND = 2.0 ** 99
TINY = 2.0 ** -1022
GAMMA = 0.99
ALPHA = 0.1
EXPLORE = F(1, 8)
OTHER = EXPLORE / 3
GREEDY = 1 - EXPLORE + OTHER
ROLLOUT = 256
UPDATES = 4
__all__ = ["VERSION", "Outcome", "TrainingResult", "train_causal_value_v6"]


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
    optimizer_version: str
    seed: int
    requested_steps: int
    steps: int
    unused_outcomes: int
    stop_reason: str
    session: replay.Session
    transitions: tuple[Transition, ...]
    behavior_probabilities: tuple[F, ...]
    targets: tuple[float, ...]
    losses: tuple[float, ...]
    underflow: int
    online: Snapshot
    target: Snapshot


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


def _values(snapshot: Snapshot, inputs: np.ndarray) -> tuple:
    result = forward_v5(snapshot, inputs, enabled=True)
    if result is None or result.shape != (len(inputs), 3):
        raise ValueError("rejected network evaluation")
    return tuple(tuple(_bounded(float(x)) for x in row) for row in result)


def _collect(session: replay.Session, outcomes: tuple, offset: int, online: Snapshot,
             rng: np.random.Generator) -> tuple:
    rows, behavior, inputs, rewards = [], [], [], []
    for outcome in outcomes[offset:offset + ROLLOUT]:
        x = _vector(replay.observe_v3(session, enabled=True))
        q = _values(online, x[None, :])[0]
        greedy = q.index(max(q))
        # Two draws per step whether or not it explores: the stream never depends on data.
        explore = float(rng.random()) < EXPLORE
        index = int(rng.integers(3))
        action = index if explore else greedy
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
        behavior.append(GREEDY if action == greedy else OTHER)
        inputs.append(x)
        rewards.append(_bounded(_number(row.reward)))
        if row.done:
            break
    return session, tuple(rows), tuple(behavior), np.asarray(inputs, dtype="float64"), tuple(rewards)


def _targets(rows: tuple, rewards: tuple, online: Snapshot, target: Snapshot) -> tuple[tuple, int]:
    # No dummy observation is evaluated for terminal rows.
    active = [i for i, row in enumerate(rows) if not row.done]
    choose, evaluate = {}, {}
    if active:
        following = np.asarray([_vector(rows[i].next_observation) for i in active], dtype="float64")
        choose = dict(zip(active, _values(online, following)))
        evaluate = dict(zip(active, _values(target, following)))
    targets, underflow = [], 0
    for i, (row, reward) in enumerate(zip(rows, rewards)):
        if row.done:
            targets.append(reward)
            continue
        selected = evaluate[i][choose[i].index(max(choose[i]))]
        bootstrap = GAMMA * selected
        if selected != 0.0 and abs(bootstrap) < TINY:
            underflow += 1
        # |reward|, |selected| <= 2^99 keep the sum inside the objective's 2^100 domain.
        targets.append(reward + bootstrap)
    return tuple(targets), underflow


def _update(online: Snapshot, inputs: np.ndarray, actions: tuple, targets: tuple) -> tuple:
    objective = objective_v2(_values(online, inputs), actions, targets, ALPHA, enabled=True)
    if objective is None:
        raise ValueError("rejected value objective")
    candidate = update_v5(online, inputs, np.asarray(objective.grad, dtype="float64"), enabled=True)
    if candidate is None:
        raise ValueError("rejected optimizer update")
    return candidate, objective.loss, objective.underflow


def train_causal_value_v6(initial_bars: object, outcomes: object, seed: object, *,
                          enabled: object = False, version: object = VERSION) -> TrainingResult | None:
    if (enabled is not True or type(version) is not str or version != VERSION or
        type(seed) is not int or not 0 <= seed < 2**32 or
        type(outcomes) is not tuple or not 1 <= len(outcomes) <= 4096):
        return None
    try:
        with np.errstate(all="raise"):
            session = replay.start_v3(initial_bars, enabled=True)
            online = create_v5(seed, 3, enabled=True)
            if session is None or online is None:
                return None
            target = online
            rng = np.random.default_rng([seed, 6])
            transitions, behavior, published, losses, underflow = [], [], [], [], 0
            while len(transitions) < len(outcomes):
                session, rows, probabilities, inputs, rewards = _collect(
                    session, outcomes, len(transitions), online, rng)
                targets, lost = _targets(rows, rewards, online, target)
                underflow += lost
                actions = tuple(row.action for row in rows)
                for _ in range(UPDATES):
                    online, loss, lost = _update(online, inputs, actions, targets)
                    losses.append(loss)
                    underflow += lost
                # Snapshots are immutable values: replacing the target is a rebinding.
                target = online
                transitions.extend(rows)
                behavior.extend(probabilities)
                published.extend(targets)
                if rows[-1].done:
                    break
            reason = "input_exhausted"
            if session.state.terminal:
                reason = transitions[-1].receipt.reason or ("tick_limit" if session.state.tick == 4096 else "terminal")
            count = len(transitions)
            return TrainingResult(VERSION, replay.OBSERVATION, REWARD, OBJECTIVE, OPTIMIZER, seed,
                                  len(outcomes), count, len(outcomes) - count, reason, session,
                                  tuple(transitions), tuple(behavior), tuple(published),
                                  tuple(losses), underflow, online, target)
    except (ArithmeticError, ValueError, TypeError, MemoryError):
        return None
