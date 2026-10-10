"""Isolated, opt-in causal RL composition. No network, order or production API.

Proposal actions are the learner's action domain, including shield interventions.
The exact kernel owns the economic ledger; binary64 is used only for learning.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as F
import hashlib
import math
import numpy as np
import causal_replay_v3 as causal
from gae_targets_v2 import batch_v2
from optimizer_snapshot_v2 import Snapshot, create_v2, forward_v2, update_v2
from sequential_learning import ppo_gradient, softmax
from value_objective_v2 import objective_v2

VERSION = "rl-causal-composition-v1"
OBSERVATION = "causal-inventory-pending-finite-v1"
ACTIONS = (F(-1, 4), F(0), F(1, 4))
STRESSES = ("base", "cost1_5x", "cost2x", "severe50bp", "delay1bar", "partial50pct", "funding2x")


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


class Episode:
    """Single-owner simulation, a pending slot and immutable causal sessions."""
    def __init__(self, bars: tuple, events: tuple, start: int, stop: int,
                 stress: str = "base", *, enabled: bool = False):
        require(enabled is True, "research disabled")
        require(type(start) is int and type(stop) is int and 96 <= start < stop - 1 <= len(bars) - 1,
                "invalid episode window")
        require(type(bars) is tuple and type(events) is tuple and len(events) == len(bars), "invalid panel")
        require(stress in STRESSES and stop - start <= 129, "unsupported episode")
        self.session = causal.start_v3(bars[start-96:start+1], enabled=True)
        require(self.session is not None, "burn-in rejected")
        self.bars, self.events, self.start, self.stop = bars, events, start, stop
        self.index, self.stress, self.pending = start, stress, None
        self.rows = []
        self.done = False
        self.failure = None
        self._owned = self.session
        self._observation = causal._observe(self.session)

    def observation(self) -> np.ndarray:
        require(not self.done, "terminal observation")
        require(self.session is self._owned, "position-state inconsistency")
        obs = self._observation
        pending = F(0) if self.pending is None else self.pending
        remaining = F(self.stop - 1 - self.index, self.stop - 1 - self.start)
        values = obs.values + (F(int(obs.supported)), remaining, pending)
        out = np.array([float(v) for v in values], dtype="float64")
        require(out.shape == (12,) and bool(np.isfinite(out).all()), "invalid observation projection")
        return out

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        require(not self.done and type(action) is int and 0 <= action < 3, "invalid proposal")
        require(self.session is self._owned, "position-state inconsistency")
        obs = self._observation
        proposal = ACTIONS[action]
        permitted = proposal if obs.supported else F(0)
        submitted = None
        applied = permitted
        reason = "permitted" if obs.supported else "unsupported_neutralized"
        pending_after = None
        if self.stress == "delay1bar":
            if self.pending is None:
                # Delay keeps existing units, not a free rebalance to the same ratio.
                submitted, pending_after = permitted, permitted
                applied = None
                reason += ":reserved"
            else:
                applied = self.pending if obs.supported else F(0)
                reason += ":pending_consumed_new_proposal_ignored"
        else:
            submitted = permitted
        j = self.index + 1
        terminal = j == self.stop - 1
        funding = self.events[j]
        if self.stress == "funding2x":
            funding = tuple((mark, rate * 2) for mark, rate in funding)
        multiplier = {"cost1_5x": F(3, 2), "cost2x": F(2), "severe50bp": F(5)}.get(self.stress, F(1))
        fill = F(1, 2) if self.stress == "partial50pct" else F(1)
        if applied is None and not terminal:
            # Exact no-fill transition: use the same kernel, with zero target and
            # a separate exact carry adapter (kernel only accepts positive fill).
            pair = carry(self.session, self.bars[j], funding, multiplier)
        else:
            pair = advance_owned(self.session, self.bars[j], funding,
                                  F(0) if applied is None else applied, terminal=terminal,
                                  multiplier=multiplier, fill=fill, enabled=True)
        require(pair is not None, "replay rejected (including rational resource limit)")
        following, receipt = pair
        costs = receipt.costs
        debit = costs.fee + costs.spread + costs.slippage + costs.impact
        delta = receipt.after.equity - receipt.before.equity
        require(delta == receipt.gross + receipt.funding - debit, "exact ledger reconciliation")
        reward = float(100 * delta)
        require(math.isfinite(reward), "nonfinite reward")
        row = {"index": j, "decisionTime": obs.decision, "fillClose": self.bars[j].close,
               "proposal": float(proposal), "permitted": float(permitted),
               "submitted": None if submitted is None else float(submitted),
               "applied": None if applied is None else float(applied),
               "filledUnits": float(receipt.after.units - receipt.before.units),
               "intervention": reason != "permitted", "reason": reason,
               "equityBefore": float(receipt.before.equity), "equity": float(receipt.after.equity),
               "increment": float(delta), "reward": reward, "gross": float(receipt.gross),
               "funding": float(receipt.funding), "fee": float(costs.fee),
               "spread": float(costs.spread), "slippage": float(costs.slippage),
               "turnover": float(costs.turnover),
               "exposure": float(receipt.after.units * receipt.after.price / receipt.after.equity)
                    if receipt.after.equity else None,
               "terminal": receipt.after.terminal, "failure": receipt.reason,
               "liquidation": receipt.liquidation, "supported": obs.supported}
        self.session, self.index = following, j
        self._owned = following
        self._observation = None if receipt.after.terminal else causal._observe(following)
        self.pending = None if receipt.after.terminal else pending_after
        self.done, self.failure = receipt.after.terminal, receipt.reason
        self.rows.append(row)
        nxt = np.zeros(12, dtype="float64") if self.done else self.observation()
        return nxt, reward, self.done


def advance_owned(session, bar, events, target, *, terminal, multiplier, fill, enabled):
    """Validated construction path only; never accepts an external Session.

Episode owns immutable sessions, fitted scale and cached observations. Repeat
scale fitting is avoided, with differential tests against public step_v3.
"""
    if (enabled is not True or not causal._bar(bar) or
            bar.symbol != session.history[-1].symbol or
            session.history[-1].available >= bar.close or len(session.history) >= causal.MAX_HISTORY):
        return None
    observation = causal._observe(session)
    chosen = target if observation.supported else F(0)
    receipt = causal.a.advance_v2(session.state, bar.price, events, chosen,
        terminal=terminal, multiplier=multiplier, fill=fill, enabled=True)
    if receipt is None:
        return None
    following = causal.Session(causal.VERSION, session.history + (bar,), session.burn, session.scale, receipt.after)
    return following, receipt


def carry(session, bar, events, multiplier):
    """No-fill delay adapter. Kernel extension is explicit and independently tested.

Use the kernel's exact mark/funding/risk/liquidation formulas with fill=0.
The public positive-fill admission is not weakened. This private reviewed call
is confined to this simulated reservation transition; no live effect exists.
"""
    a = causal.a
    if (not causal._bar(bar) or bar.symbol != session.history[-1].symbol or
            session.history[-1].available >= bar.close or len(session.history) >= causal.MAX_HISTORY or
            not a._admissible(session.state, bar.price, events, F(0), False, F(1), multiplier, F(0))):
        return None
    try:
        receipt = a._advance(session.state, bar.price, events, F(0), False, F(0), multiplier, F(0))
        following = causal.Session(causal.VERSION, session.history + (bar,), session.burn, session.scale, receipt.after)
        return following, receipt
    except (ArithmeticError, TypeError, ValueError, MemoryError):
        return None


def forward(net, x: np.ndarray) -> np.ndarray:
    if type(net) is Snapshot:
        require(net.version == "optimizer-snapshot-v2" and net.outputs in (1, 3) and
                type(net.p) is tuple and len(net.p) == 4 and
                all(type(b) is bytes and len(b) == size * 8 for b, size in
                    zip(net.p, (192, 16, 16 * net.outputs, net.outputs))), "invalid frozen snapshot")
        require(type(x) is np.ndarray and x.dtype == np.dtype("float64") and x.ndim == 2 and
                1 <= len(x) <= 256 and x.shape[1] == 12 and np.isfinite(x).all(), "invalid inference input")
        shapes = ((12,16),(16,),(16,net.outputs),(net.outputs,))
        w1,b1,w2,b2 = (np.frombuffer(b,dtype="<f8").reshape(shape) for b,shape in zip(net.p,shapes))
        require(all(np.isfinite(a).all() for a in (w1,b1,w2,b2)), "nonfinite artifact")
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            hidden = x @ w1 + b1
            require(np.isfinite(hidden).all(), "hidden overflow")
            out = np.tanh(hidden) @ w2 + b2
            require(np.isfinite(out).all(), "output overflow")
        return out
    out = forward_v2(net, x, enabled=True)
    require(out is not None, "network evaluation rejected")
    return out


def update(net, x, gradient, rate):
    require(update_v2(net, x, gradient, rate, enabled=True) is not None, "optimizer rejected")


class Collector:
    """Preserves episodes across rollout cuts. Resets only at economic terminal."""
    def __init__(self, factory, seed):
        self.factory, self.rng, self.episode = factory, np.random.default_rng(seed), None
        self.transitions = 0
        self.failures = []

    def collect(self, policy, count):
        rows = []
        for _ in range(count):
            if self.episode is None or self.episode.done:
                self.episode = self.factory(self.rng)
            x = self.episode.observation()
            probs = np.asarray(policy(x), dtype="float64")
            require(probs.shape == (3,) and np.isfinite(probs).all() and np.all(probs >= 0)
                    and abs(probs.sum() - 1) < 1e-12, "invalid proposal probabilities")
            action = int(self.rng.choice(3, p=probs))
            nxt, reward, done = self.episode.step(action)
            rows.append((x, action, reward, nxt, done, float(probs[action])))
            self.transitions += 1
            if done and self.episode.failure:
                self.failures.append(self.episode.failure)
        columns = zip(*rows)
        return {key: np.asarray(v, dtype=dtype) for key, dtype, v in
                zip(("s", "a", "r", "next", "done", "prob"),
                    ("float64", "int64", "float64", "float64", "bool", "float64"), columns)}


def buffer_hash(data):
    digest = hashlib.sha256()
    for key in ("s", "a", "r", "next", "done", "prob"):
        a = data[key]
        digest.update(key.encode()); digest.update(str(a.shape).encode()); digest.update(a.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class Fit:
    algorithm: str
    seed: int
    actor: object
    metadata: dict


def train(algorithm, factory, seed, steps=512, frozen=None, *, enabled=False):
    require(enabled is True, "research disabled")
    require(algorithm in ("ppo", "double_dqn", "cql", "cql_alpha0", "behavior_cloning") and
            type(seed) is int and 0 <= seed < 2**31 and type(steps) is int and
            128 <= steps <= 4096 and steps % 128 == 0, "invalid training configuration")
    net = create_v2(seed, 3, enabled=True)
    require(net is not None, "unsupported optimizer runtime")
    rng = np.random.default_rng(seed + 2)
    collector = Collector(factory, seed + 1000)
    losses, buffer = [], None
    before = None if frozen is None else buffer_hash(frozen)
    if algorithm == "ppo":
        critic = create_v2(seed + 1, 1, enabled=True)
        require(critic is not None, "critic rejected")
        for _ in range(steps // 128):
            data = collector.collect(lambda x: softmax(forward(net, x[None, :]))[0], 128)
            v = forward(critic, data["s"]).ravel()
            nxt = forward(critic, data["next"]).ravel()
            rows = tuple((float(r), float(a), float(b), bool(d)) for r, a, b, d in zip(data["r"], v, nxt, data["done"]))
            pairs = batch_v2(rows, 1.0, enabled=True)
            require(pairs is not None, "GAE rejected")
            advantage, targets = np.asarray(pairs).T
            advantage = (advantage - advantage.mean()) / max(advantage.std(), 1e-8)
            for _ in range(4):
                loss, grad = ppo_gradient(forward(net, data["s"]), data["a"], data["prob"], advantage)
                require(math.isfinite(loss) and np.isfinite(grad).all(), "PPO objective rejected")
                update(net, data["s"], grad, .0003)
                update(critic, data["s"], (forward(critic, data["s"]).ravel() - targets)[:, None] / len(v), .0003)
                losses.append(loss)
    else:
        offline = algorithm != "double_dqn"
        if offline:
            require(frozen is not None, "fixed dataset required")
            buffer = frozen
        target = net.snapshot()
        for batch in range(steps // 128):
            if not offline:
                def behavior(x):
                    probs = np.full(3, .2 / 3)
                    probs[int(np.argmax(forward(net, x[None, :]))) ] += .8
                    return probs
                new = collector.collect(behavior, 128)
                buffer = new if buffer is None else {k: np.concatenate((buffer[k], new[k])) for k in buffer}
            for _ in range(128):
                idx = rng.integers(len(buffer["a"]), size=64)
                x, actions = buffer["s"][idx], buffer["a"][idx]
                if algorithm == "behavior_cloning":
                    grad = softmax(forward(net, x))
                    loss = -float(np.log(grad[np.arange(64), actions]).mean())
                    grad[np.arange(64), actions] -= 1
                    grad /= 64
                else:
                    nxt = buffer["next"][idx]
                    greedy = np.argmax(forward(net, nxt), axis=1)
                    boot = forward(target, nxt)[np.arange(64), greedy]
                    targets = np.where(buffer["done"][idx], buffer["r"][idx], buffer["r"][idx] + boot)
                    objective = objective_v2(tuple(map(tuple, forward(net, x).tolist())),
                        tuple(map(int, actions)), tuple(map(float, targets)),
                        .1 if algorithm == "cql" else 0., enabled=True)
                    require(objective is not None, "value objective rejected")
                    loss, grad = objective.loss, np.asarray(objective.grad)
                update(net, x, grad, .001)
                if net.snapshot().step % 100 == 0:
                    target = net.snapshot()
                losses.append(loss)
    require(frozen is None or before == buffer_hash(frozen), "offline dataset mutated")
    return Fit(algorithm, seed, net.snapshot(), {"environment": VERSION, "observation": OBSERVATION,
        "steps": steps, "updates": net.snapshot().step, "collectedTransitions": collector.transitions,
        "trainingFailures": collector.failures, "lossFirst": losses[0], "lossLast": losses[-1],
        "frozenDatasetSha256": before, "regime": "on_policy_simulator" if algorithm == "ppo" else
        "off_policy_simulator" if algorithm == "double_dqn" else "supervised_simulated" if algorithm == "behavior_cloning" else "fixed_simulated_dataset"})


def policy(fit, x):
    return int(np.argmax(forward(fit.actor, x[None, :])))


def baseline(name, episode):
    if name == "cash": return 1
    if name == "constant_long": return 2
    raw = causal._features(episode.session.history)[1]
    require(name in ("momentum3", "reversal3"), "unknown baseline")
    sign = (raw > 0) - (raw < 0)
    return 1 + sign * (1 if name == "momentum3" else -1)
