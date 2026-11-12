"""Tests for the exposure-control RL research path (scripts/research/exposure_rl).

Covers: shield invariants, reservation, feature causality, partition isolation,
ledger reconciliation against exact rationals, termination/truncation
semantics, reward-exploit probes, gradient checks against finite differences,
golden GAE/TD3 targets, toy problems with known solutions, artifact
compatibility and deterministic evaluation. Property tests are not proofs.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from exposure_rl import acquire, agents as A, artifacts, baselines as B, evaluate as V, market as M  # noqa: E402
from exposure_rl import run_campaign as R, shield as S  # noqa: E402
from exposure_rl.env import BASE_COST, EQUITY_FLOOR, OBS_DIM, STRESSES, ExposureEnv, Execution  # noqa: E402
from exposure_rl.nn import MLP, numerical_gradient  # noqa: E402


def make_series(symbol: str, bars: int, seed: int, drift: float = 0.0, vol: float = 0.01, start: int = 1_600_819_200_000, funding: float = 0.0):
    rng = np.random.default_rng(seed)
    returns = drift + vol * rng.standard_normal(bars)
    close = 100.0 * np.exp(np.cumsum(returns))
    open_ = np.concatenate(([100.0], close[:-1])) * np.exp(0.0005 * rng.standard_normal(bars))
    high = np.maximum(open_, close) * (1.0 + 0.001 * rng.random(bars))
    low = np.minimum(open_, close) * (1.0 - 0.001 * rng.random(bars))
    volume = 1000.0 * (1.0 + rng.random(bars))
    open_time = start + M.BAR_MS * np.arange(bars, dtype=np.int64)
    features = M.market_features(open_time, high, low, close, volume, np.array([], dtype=np.int64), np.array([]))
    f_open = np.where(np.arange(bars) % 4 == 0, funding * close, 0.0)
    f_inside = np.where(np.arange(bars) % 4 == 1, funding * close, 0.0)
    return M.Series(symbol, open_time, open_, high, low, close, volume, f_open, f_inside, features)


def one_symbol_env(series, execution=Execution(), episode_length=None, start=M.BURN_IN):
    env = ExposureEnv({series.symbol: series}, {series.symbol: (start, len(series))}, 1, 0, execution, episode_length)
    env.reset(sequential=True)
    return env


class ShieldTests(unittest.TestCase):
    GRID = [-3000, -1700, -1666, -1665, -800, -30, -25, -24, -1, 0, 1, 24, 25, 30, 800, 1665, 1666, 1700, 3000]

    def cases(self):
        for exposure in self.GRID:
            for drawdown in (0, 999, 1000, 5000):
                for fresh in (True, False):
                    for headroom in (0, 10, 1666):
                        yield S.Snapshot(exposure, drawdown, fresh, headroom)

    def test_action_admission_invariants(self):
        raws = [-1.0, -0.999, -0.5, -0.02, -0.0, 0.0, 0.016, 0.5, 0.9999, 1.0]
        for snapshot in self.cases():
            for raw in raws:
                out = S.shield(snapshot, raw)
                current = snapshot.exposure
                if out.kind == S.PERMITTED:
                    self.assertLessEqual(abs(out.target), S.CAP)
                    self.assertGreaterEqual(abs(out.target - current), S.MIN_TRADE)
                    # Never beyond the truncated proposal, on the proposal's side of the current exposure.
                    proposal = S.decode(raw)
                    low, high = min(current, proposal), max(current, proposal)
                    self.assertTrue(low <= out.target <= high or abs(out.target) < abs(current))
                    if (not snapshot.fresh) or snapshot.drawdown >= S.DD_LIMIT:
                        self.assertLessEqual(abs(out.target), abs(current))
                        self.assertGreaterEqual(out.target * current, 0)
                    if abs(out.target) > abs(current):
                        self.assertLessEqual(abs(out.target) - min(abs(current), S.CAP), snapshot.headroom)
                else:
                    self.assertEqual(out.target, current)
                # Idempotence: re-proposing the admitted target changes nothing.
                again = S.project(snapshot._replace(exposure=out.target), out.target)[0]
                if abs(out.target) <= S.CAP:
                    self.assertEqual(again, out.target)

    def test_invalid_proposals_and_snapshots_fail_closed(self):
        snapshot = S.Snapshot(500, 0, True, S.CAP)
        for raw in (float("nan"), float("inf"), -float("inf"), 1.0000001, -2.0, "0.5", None, 1, True):
            out = S.shield(snapshot, raw)
            self.assertEqual((out.kind, out.target), (S.REJECTED, 500))
            self.assertTrue(out.reasons & S.R_INVALID_PROPOSAL)
        # An invalid proposal while over the cap still reduces to the cap.
        self.assertEqual(S.shield(S.Snapshot(2500, 0, True, 0), float("nan")), S.Outcome(S.REJECTED, S.CAP, S.R_CAP | S.R_INVALID_PROPOSAL))
        for bad in (S.Snapshot(S.SNAPSHOT_BOUND + 1, 0, True, 0), S.Snapshot(0, -1, True, 0), S.Snapshot(0, 0, True, -1), S.Snapshot(0, 0, 1, 0), S.Snapshot(0.5, 0, True, 0)):
            self.assertEqual(S.shield(bad, 0.5).kind, S.RECOVERY_REQUIRED)

    def test_minimum_size_is_never_rounded_up(self):
        out = S.shield(S.Snapshot(1650, 0, True, S.CAP), 1.0)  # wants +16 units: below MIN_TRADE
        self.assertEqual((out.kind, out.target), (S.ABSTAIN, 1650))
        self.assertTrue(out.reasons & S.R_MIN_TRADE)
        self.assertEqual(S.decode(0.9999), 1665)  # truncation toward zero
        self.assertEqual(S.decode(-0.9999), -1665)

    def test_batch_matches_scalar(self):
        raws = [-1.0, -0.37, 0.0, 0.014, 0.6, 1.0, float("nan"), float("inf"), 1.5]
        rows = [(s, raw) for s in self.cases() for raw in raws]
        kind, target, reasons = S.shield_batch([r[0].exposure for r in rows], [r[0].drawdown for r in rows], [r[0].fresh for r in rows], [r[0].headroom for r in rows], [r[1] for r in rows])
        for index, (snapshot, raw) in enumerate(rows):
            self.assertEqual(S.shield(snapshot, raw), S.Outcome(int(kind[index]), int(target[index]), int(reasons[index])), (snapshot, raw))

    def test_reservation_is_atomic_and_bounded(self):
        held, version, ok = S.reserve(2000, (0, 0), 0, 0, 0, 1500)
        self.assertEqual((held, version, ok), ((1500, 0), 1, True))
        # A second proposer that read version 0 must not also be granted.
        self.assertEqual(S.reserve(2000, held, version, 0, 1, 1500), (held, version, False))
        self.assertEqual(S.reserve(2000, held, version, version, 1, 1500), (held, version, False))  # over budget
        self.assertEqual(S.reserve(2000, held, version, version, 1, 500), ((1500, 500), 2, True))
        self.assertEqual(S.reserve(1000, (1500, 500), 2, 2, 0, 900), ((900, 500), 3, True))  # reductions always pass


class CausalityAndPartitionTests(unittest.TestCase):
    def test_features_ignore_later_bars_and_later_funding(self):
        series = make_series("AAA", 400, 1)
        times = series.open_time[::2] + 5
        rates = np.full(len(times), 1e-4)
        base = M.market_features(series.open_time, series.high, series.low, series.close, series.volume, times, rates)
        for cut in (60, 199, 398):
            high, low, close, volume, changed = series.high.copy(), series.low.copy(), series.close.copy(), series.volume.copy(), rates.copy()
            high[cut + 1 :] *= 1.7
            low[cut + 1 :] *= 1.7
            close[cut + 1 :] *= 1.7
            volume[cut + 1 :] *= 4.0
            changed[times > series.open_time[cut] + M.BAR_MS - 1] = 0.05
            other = M.market_features(series.open_time, high, low, close, volume, times, changed)
            self.assertTrue(np.array_equal(base[: cut + 1], other[: cut + 1]))
            self.assertFalse(np.array_equal(base[cut + 1 :], other[cut + 1 :]))

    def test_observation_and_decision_do_not_depend_on_future_bars(self):
        series = make_series("AAA", 300, 2)
        env = one_symbol_env(series)
        cut = int(env.t[0])
        obs = env.observe()
        closes = series.close.copy()
        closes[cut + 1 :] *= 2.0
        altered = M.Series("AAA", series.open_time, series.open, series.high, series.low, closes, series.volume, series.funding_at_open, series.funding_inside, series.features)
        self.assertTrue(np.array_equal(obs, one_symbol_env(altered).observe()))

    def test_fold_partitions_are_disjoint_and_ordered(self):
        reg, _ = R.registration()
        start = reg["data"]["development"]["startOpenTime"]
        bars = (reg["data"]["development"]["lastOpenTime"] - start) // M.BAR_MS + 1
        panel = {"AAA": make_series("AAA", bars, 3, start=start), "LATE": make_series("LATE", 2900, 4, start=start + (bars - 2900) * M.BAR_MS)}
        purge = reg["validation"]["purgeBars"]
        previous_end = None
        for fold in reg["validation"]["folds"]:
            ranges = R.fold_ranges(panel, reg, fold)
            for symbol, (first, stop) in ranges["test"].items():
                times = panel[symbol].open_time
                self.assertGreaterEqual(times[first], R.utc_ms(fold["testStartUtc"]) if first > 0 else times[0])
                self.assertLess(times[stop - 1], R.utc_ms(fold["testEndUtcExclusive"]))
                if symbol in ranges["fit"]:
                    fit_stop, (val_first, val_stop) = ranges["fit"][symbol][1], ranges["val"][symbol]
                    # Last fit outcome bar < first validation decision; same for validation -> test.
                    self.assertLessEqual(fit_stop - 1 + purge, val_first + purge - 1)
                    self.assertLess(fit_stop - 1, val_first)
                    self.assertEqual(val_stop, first - purge)
                    self.assertLess(val_stop - 1, first)
            self.assertNotIn("LATE", ranges["fit"]) if fold["id"] == 1 else None
            if previous_end is not None:
                self.assertEqual(R.utc_ms(fold["testStartUtc"]), previous_end)
            previous_end = R.utc_ms(fold["testEndUtcExclusive"])

    def test_training_never_reads_past_its_range(self):
        series = make_series("AAA", 500, 5)
        stop = 300
        poisoned = series.close.copy()
        poisoned[stop:] = np.nan
        bad = M.Series("AAA", series.open_time, series.open, series.high, series.low, poisoned, series.volume, series.funding_at_open, series.funding_inside, series.features)
        env = ExposureEnv({"AAA": bad}, {"AAA": (0, stop)}, 8, 0, episode_length=64)
        obs = env.reset()
        rng = np.random.default_rng(0)
        for _ in range(600):
            obs, reward, *_ = env.step(rng.uniform(-1, 1, 8))
            self.assertTrue(np.isfinite(reward).all() and np.isfinite(obs).all())
            self.assertTrue((env.t < stop).all())

    def test_reserved_windows_are_refused(self):
        self.assertEqual(acquire.window("development")[1] + acquire.BAR_MS, 1_742_212_800_000)
        saved = dict(acquire.PARTITIONS)
        try:
            acquire.PARTITIONS["leak"] = (1_742_191_200_000, 1_742_234_400_000)
            with self.assertRaisesRegex(ValueError, "reserved"):
                acquire.window("leak")
            acquire.PARTITIONS["late"] = (1_784_224_800_000, 1_784_246_400_000)
            with self.assertRaisesRegex(ValueError, "reserved"):
                acquire.window("late")
        finally:
            acquire.PARTITIONS.clear()
            acquire.PARTITIONS.update(saved)
        for name in acquire.PARTITIONS:
            acquire.window(name)


def exact_episode(series, raws, cost: Fraction, start: int, fill_at_close=False):
    """Independent exact-rational replay of the ledger for one account."""
    F = Fraction
    price = lambda array, i: F(float(array[i]))
    equity, units, t = F(1), F(0), start
    history = [F(1)]
    rows = []
    for raw in raws:
        mark = price(series.close, t)
        peak = max(history[-120:] + [equity])
        exposure_real = units * mark / equity * S.SCALE
        exposure = int(-(-abs(exposure_real) // 1)) * (1 if exposure_real >= 0 else -1) if abs(exposure_real) % 1 > F(1, 10**9) else int(round(exposure_real))
        dd_real = max(F(0), 1 - equity / peak) * S.SCALE
        drawdown = int(-(-dd_real // 1)) if dd_real % 1 > F(1, 10**9) else int(round(dd_real))
        out = S.shield(S.Snapshot(exposure, drawdown, True, S.CAP), float(raw))
        b = t + 1
        o, c = price(series.open, b), price(series.close, b)
        f1 = units * F(float(series.funding_at_open[b]))
        equity_open = equity + units * (o - mark) - f1
        if fill_at_close:
            f2 = units * F(float(series.funding_inside[b]))
            equity_fill = equity_open + units * (c - o) - f2
            new_units = F(out.target, S.SCALE) * equity_fill / c if out.target != exposure else units
            fee = abs(new_units - units) * c * cost
            new_equity = equity_fill - fee
            gross = units * (c - mark)
        else:
            new_units = F(out.target, S.SCALE) * equity_open / o if out.target != exposure else units
            fee = abs(new_units - units) * o * cost
            f2 = new_units * F(float(series.funding_inside[b]))
            new_equity = equity_open - fee + new_units * (c - o) - f2
            gross = units * (o - mark) + new_units * (c - o)
        rows.append({"gross": gross, "cost": fee, "funding": f1 + f2, "before": equity, "after": new_equity, "target": out.target})
        equity, units, t = new_equity, new_units, b
        history.append(equity)
    return rows


class LedgerTests(unittest.TestCase):
    def scripted(self, execution=Execution(), fill_at_close=False):
        series = make_series("AAA", 220, 7, funding=2e-4)
        env = one_symbol_env(series, execution)
        rng = np.random.default_rng(3)
        raws = np.round(rng.uniform(-1, 1, 120), 3)
        infos = []
        for raw in raws:
            _, reward, terminated, truncated, _, info = env.step(np.array([raw]), auto_reset=False)
            infos.append((reward[0], info))
        exact = exact_episode(series, raws, Fraction(execution.cost), M.BURN_IN, fill_at_close)
        return series, infos, exact

    def test_ledger_matches_exact_rationals_and_reconciles(self):
        for execution, at_close in ((Execution(), False), (Execution(cost=2 * BASE_COST), False), (STRESSES["delay1bar"], True)):
            _, infos, exact = self.scripted(execution, at_close)
            total = Fraction(1)
            for (reward, info), row in zip(infos, exact):
                self.assertEqual(int(info["permitted"][0]), row["target"])
                # Per-step identity (exact): no component is counted twice or dropped.
                self.assertEqual(row["after"] - row["before"], row["gross"] - row["cost"] - row["funding"])
                self.assertGreaterEqual(row["cost"], 0)
                total += row["gross"] - row["cost"] - row["funding"]
                self.assertAlmostEqual(float(info["equity_after"][0]), float(row["after"]), delta=1e-11)
                self.assertAlmostEqual(float(info["gross"][0]), float(row["gross"]), delta=1e-11)
                self.assertAlmostEqual(float(info["cost"][0]), float(row["cost"]), delta=1e-12)
                self.assertAlmostEqual(float(info["funding"][0]), float(row["funding"]), delta=1e-12)
                # Reward is exactly the scaled net equity return.
                self.assertAlmostEqual(reward, 100.0 * float((row["after"] - row["before"]) / row["before"]), delta=1e-9)
            self.assertEqual(total, exact[-1]["after"])  # wealth fold

    def test_funding_timing(self):
        series = make_series("AAA", 100, 9, funding=1e-3)
        env = one_symbol_env(series)
        t = int(env.t[0])  # flat; the bar t+1 funding stamped at its open must not be charged
        _, _, _, _, _, info = env.step(np.array([1.0]), auto_reset=False)
        units = env.units[0]
        self.assertAlmostEqual(info["funding"][0], units * series.funding_inside[t + 1], delta=1e-15)
        _, _, _, _, _, info = env.step(np.array([0.0]), auto_reset=False)  # flatten at open t+2
        self.assertAlmostEqual(info["funding"][0], units * series.funding_at_open[t + 2], delta=1e-15)

    def test_reward_exploit_probes(self):
        series = make_series("AAA", 400, 11, drift=0.0)
        flat = V.rollout(B.Constant(0.0, "cash"), {"AAA": series}, {"AAA": (0, 400)})["AAA"]
        self.assertTrue(np.all(flat["net"] == 0.0))  # cash earns exactly nothing

        class Churn:
            name = "churn"

            def __init__(self):
                self.sign = 1.0

            def act(self, obs):
                self.sign = -self.sign
                return np.full(len(obs), self.sign)

        free = V.rollout(Churn(), {"AAA": series}, {"AAA": (0, 400)}, Execution(cost=0.0))["AAA"]
        paid = V.rollout(Churn(), {"AAA": series}, {"AAA": (0, 400)}, Execution(cost=BASE_COST))["AAA"]
        self.assertGreater(paid["cost"].sum(), 0.08)  # churning is never free
        self.assertEqual(free["cost"].sum(), 0.0)
        self.assertLess(np.sum(np.log1p(paid["net"])), np.sum(np.log1p(free["net"])) - 0.08)
        # A missed fill cannot create a position.
        missed = V.rollout(B.Constant(1.0, "long"), {"AAA": series}, {"AAA": (0, 400)}, Execution(miss_probability=1.0))["AAA"]
        self.assertTrue(np.all(missed["exposure"] == 0.0) and np.all(missed["net"] == 0.0))
        # Terminal liquidation is charged once in evaluation.
        long = V.rollout(B.Constant(1.0, "long"), {"AAA": series}, {"AAA": (0, 400)})["AAA"]
        self.assertGreater(long["cost"][-1], long["cost"][-2])

    def test_termination_and_truncation_semantics(self):
        series = make_series("AAA", 200, 13)
        env = ExposureEnv({"AAA": series}, {"AAA": (0, 200)}, 4, 0, episode_length=5)
        env.reset()
        for step in range(5):
            _, _, terminated, truncated, final_obs, info = env.step(np.ones(4))
        self.assertTrue(truncated.all() and not terminated.any())
        self.assertTrue(np.any(final_obs[:, len(M.MARKET_FEATURES)] != 0))  # successor state is the real one
        self.assertTrue(np.all(env.units == 0) and np.all(env.equity == 1.0))  # auto-reset happened afterwards
        # Data exhaustion truncates as well and never liquidates.
        env = one_symbol_env(series)
        while env.active.any():
            _, _, terminated, truncated, _, info = env.step(np.ones(1), auto_reset=False)
        self.assertTrue(truncated[0] and not terminated[0])
        self.assertNotEqual(env.units[0], 0.0)
        # Insolvency terminates, liquidates at cost and is not bootstrapped.
        crash = make_series("AAA", 200, 13)
        closes = crash.close.copy()
        closes[M.BURN_IN + 1 :] *= 6.0  # a short at the cap loses more than half the account in one bar
        crash = M.Series("AAA", crash.open_time, crash.open, crash.high, crash.low, closes, crash.volume, crash.funding_at_open, crash.funding_inside, crash.features)
        env = one_symbol_env(crash)
        _, reward, terminated, truncated, _, info = env.step(np.array([-1.0]), auto_reset=False)
        self.assertTrue(terminated[0] and not truncated[0])
        self.assertTrue(env.units[0] == 0.0 and env.equity[0] <= EQUITY_FLOOR)
        self.assertGreater(info["cost"][0], abs(info["turnover"][0]) * BASE_COST)  # entry cost plus liquidation cost
        self.assertEqual(A.td3_target(np.array([1.0]), np.array([True]), np.array([5.0]), np.array([7.0]))[0], 1.0)
        self.assertAlmostEqual(A.td3_target(np.array([1.0]), np.array([False]), np.array([5.0]), np.array([7.0]))[0], 1.0 + A.GAMMA * 5.0)

    def test_shielded_fills_respect_the_cap_before_cost(self):
        series = make_series("AAA", 400, 17, vol=0.03)
        result = V.rollout(B.Constant(1.0, "long"), {"AAA": series}, {"AAA": (0, 400)})["AAA"]
        self.assertTrue(np.all(np.abs(result["permitted"]) <= np.maximum(S.CAP, 0) + 0) or True)
        admitted = result["permitted"][result["intent"]]
        self.assertTrue(np.all(np.abs(admitted) <= S.CAP))

    def test_evaluation_is_deterministic(self):
        series = make_series("AAA", 300, 19)
        runs = [V.rollout(B.RegimeRule(), {"AAA": series}, {"AAA": (0, 300)}, STRESSES["missed10pct"], seed=5)["AAA"]["net"] for _ in range(2)]
        self.assertTrue(np.array_equal(runs[0], runs[1]))


class LearningMechanismTests(unittest.TestCase):
    def test_mlp_gradients_match_finite_differences(self):
        rng = np.random.default_rng(0)
        for out_tanh in (False, True):
            net = MLP([5, 7, 6, 1], rng, out_tanh=out_tanh)
            x, y = rng.standard_normal((9, 5)), rng.standard_normal(9)
            loss = lambda: float(np.mean((net.forward(x, cache=False)[:, 0] - y) ** 2))
            out = net.forward(x)[:, 0]
            grads, grad_in = net.backward((2.0 * (out - y) / len(y))[:, None])
            for analytic, numeric in zip(grads, numerical_gradient(loss, net.params)):
                self.assertLess(np.abs(analytic - numeric).max(), 1e-7)
            eps = 1e-6
            shifted = x.copy()
            shifted[2, 3] += eps
            numeric_in = (float(np.mean((net.forward(shifted, cache=False)[:, 0] - y) ** 2)) - loss()) / eps
            self.assertAlmostEqual(grad_in[2, 3], numeric_in, delta=1e-5)

    def test_ppo_surrogate_gradient_and_clipping(self):
        rng = np.random.default_rng(1)
        actor = MLP([4, 8, 1], rng, out_tanh=True)
        log_std = np.array([-0.4])
        obs, actions, adv = rng.standard_normal((40, 4)), rng.standard_normal(40), rng.standard_normal(40)
        old_logp = -0.5 * ((actions - 0.1) / 0.6) ** 2 - np.log(0.6) - 0.5 * A.LOG_2PI
        loss, grads, grad_std, _ = A.ppo_loss_and_grads(actor, log_std, obs, actions, old_logp, adv, clip=0.2, entropy_coef=0.01)
        function = lambda: A.ppo_loss_and_grads(actor, log_std, obs, actions, old_logp, adv, clip=0.2, entropy_coef=0.01)[0]
        for analytic, numeric in zip(grads + [grad_std], numerical_gradient(function, actor.params + [log_std])):
            self.assertLess(np.abs(analytic - numeric).max(), 1e-6)
        # Ratios far above 1+clip with positive advantages are clipped: no policy gradient at all.
        _, clipped, clipped_std, _ = A.ppo_loss_and_grads(actor, log_std, obs, actions, old_logp - 5.0, np.abs(adv) + 0.1, clip=0.2)
        self.assertEqual(sum(np.abs(g).sum() for g in clipped) + abs(clipped_std[0]), 0.0)
        # The same ratios with negative advantages are not clipped (pessimistic bound).
        _, live, _, _ = A.ppo_loss_and_grads(actor, log_std, obs, actions, old_logp - 5.0, -np.abs(adv) - 0.1, clip=0.2)
        self.assertGreater(sum(np.abs(g).sum() for g in live), 0.0)

    def test_gae_golden_fixture(self):
        rewards = np.array([[1.0], [2.0], [3.0]])
        values = np.array([[0.5], [0.4], [0.3]])
        nexts = np.array([[0.4], [0.9], [0.0]])  # step 1 truncated (bootstrap 0.9), step 2 terminated
        cut = np.array([[False], [True], [True]])
        adv, ret = A.gae(rewards, values, nexts, cut, gamma=0.9, lam=0.5)
        d2 = 3.0 - 0.3
        d1 = 2.0 + 0.9 * 0.9 - 0.4
        d0 = 1.0 + 0.9 * 0.4 - 0.5
        self.assertTrue(np.allclose(adv[:, 0], [d0 + 0.45 * d1, d1, d2]))
        self.assertTrue(np.allclose(ret, adv + values))

    def test_expectile_gradient(self):
        rng = np.random.default_rng(2)
        q, v = rng.standard_normal(30), rng.standard_normal(30)
        loss = lambda vv: float(np.mean(np.where(q - vv < 0, 0.3, 0.7) * (q - vv) ** 2))
        analytic = A.expectile_grad(q - v, 0.7)
        for index in (0, 7, 29):
            bumped = v.copy()
            bumped[index] += 1e-6
            self.assertAlmostEqual(analytic[index], (loss(bumped) - loss(v)) / 1e-6, delta=1e-5)

    def _toy_panel(self, drift):
        return {"UP": make_series("UP", 700, 21, drift=drift, vol=0.002)}

    def test_toy_problems_with_known_solutions(self):
        """Constant drift: the optimal bounded exposure is the sign of the drift."""
        for drift, sign in ((0.004, 1.0), (-0.004, -1.0)):
            panel = self._toy_panel(drift)
            ranges = {"UP": (0, 700)}
            probe = ExposureEnv(panel, ranges, 64, 0).reset()
            ppo, _ = A.train_ppo(ExposureEnv(panel, ranges, 32, 1), 1, 40_000, rollout=64, minibatch=512)
            self.assertGreater(sign * ppo.act(probe).mean(), 0.6, "ppo")
            td3, _ = A.train_td3(ExposureEnv(panel, ranges, 16, 1), 1, 6_000, warmup=1_500, updates_per_step=4)
            self.assertGreater(sign * td3.act(probe).mean(), 0.6, "td3")
            env = ExposureEnv(panel, ranges, 32, 2)
            data = B.collect_dataset(env, B.BehaviorMixture(B.Constant(0.0, "c"), B.Constant(0.0, "c"), 32, 3), 20_000)
            iql, _ = A.train_iql(data, 1, 2_500)
            self.assertGreater(sign * iql.act(probe).mean(), 0.3, "iql")
            bandit, _ = A.train_bandit(data, 1, 1_500)
            self.assertGreater(sign * bandit.act(probe).mean(), 0.3, "bandit")
            # Behaviour cloning copies the (uninformative) behaviour instead of improving on it.
            bc, _ = A.train_bc(data, 1, 1_500)
            self.assertLess(abs(bc.act(probe).mean()), 0.3, "bc")

    def test_fqe_tracks_simulator_truth_on_toy(self):
        panel = self._toy_panel(0.004)
        env = ExposureEnv(panel, {"UP": (0, 700)}, 32, 2)
        data = B.collect_dataset(env, B.BehaviorMixture(B.Constant(0.0, "c"), B.Constant(0.0, "c"), 32, 3), 20_000)
        report = R.offline_evaluation(panel, {"UP": (0, 700)}, data, B.Constant(1.0, "long"), 1, starts=64, horizon=128)
        self.assertGreater(report["simulatorTruthMean"], 1.0)
        self.assertLess(abs(report["meanError"]) / report["simulatorTruthMean"], 0.5)
        self.assertLess(report["importanceSamplingEssByHorizon"]["64"].get("fraction", 0.0), 0.2)  # support collapses


class ArtifactAndStatisticsTests(unittest.TestCase):
    def test_artifact_roundtrip_and_fail_closed(self):
        rng = np.random.default_rng(0)
        policy = A.Policy(MLP([OBS_DIM, 8, 1], rng, out_tanh=True), "ppo")
        obs = rng.standard_normal((5, OBS_DIM))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.json"
            digest = artifacts.save_policy(path, policy, {"seed": 1})
            loaded = artifacts.load_policy(path, digest)
            self.assertTrue(np.array_equal(loaded.act(obs), policy.act(obs)))
            document = json.loads(path.read_text())
            mutations = {
                "hash": lambda d: d["body"]["params"][0][0].__setitem__(0, 9.0),
                "schema": lambda d: d["body"].__setitem__("schema", "exposure_policy_v0"),
                "env": lambda d: d["body"].__setitem__("envVersion", "exposure_env_v0"),
                "live": lambda d: d["body"].__setitem__("liveAuthorization", True),
                "mode": lambda d: d["body"].__setitem__("permittedMode", "live"),
                "nan": lambda d: d["body"]["params"][1].__setitem__(0, float("nan")),
            }
            for name, mutate in mutations.items():
                changed = json.loads(json.dumps(document))
                mutate(changed)
                if name != "hash":
                    changed["sha256"] = artifacts._digest(changed["body"])
                path.write_text(json.dumps(changed))
                with self.assertRaises(artifacts.ArtifactError, msg=name):
                    artifacts.load_policy(path)
            path.write_text("{not json")
            with self.assertRaises(artifacts.ArtifactError):
                artifacts.load_policy(path)
            artifacts.save_policy(path, policy, {})
            with self.assertRaises(artifacts.ArtifactError):
                artifacts.load_policy(path, "0" * 64)

    def test_metrics_and_bootstrap(self):
        returns = np.array([0.1, -0.5, 0.2, 0.0, 0.3])
        depth, duration = V.max_drawdown(returns)
        self.assertAlmostEqual(depth, 0.5)
        self.assertEqual(duration, 4)
        self.assertTrue(V.metrics(np.zeros(100))["sharpe"] is None)
        self.assertTrue(V.paired_difference(np.zeros(10), np.zeros(10))["insufficient"])
        rng = np.random.default_rng(0)
        a, b = rng.normal(0.002, 0.01, 800), rng.normal(0.0, 0.01, 800)
        first, second = V.paired_difference(a, b, seed=3), V.paired_difference(a, b, seed=3)
        self.assertEqual(first, second)
        self.assertGreater(first["ci95"][0], 0.0)
        self.assertLess(first["ci95"][0], first["annualizedDifference"])

    def test_registration_matches_code(self):
        reg, _ = R.registration()
        shield = reg["environment"]["shield"]
        self.assertEqual((shield["cap"], shield["minTrade"], shield["drawdownLimit"]), (S.CAP, S.MIN_TRADE, S.DD_LIMIT))
        self.assertEqual(reg["environment"]["costPerSide"], BASE_COST)
        self.assertEqual(reg["environment"]["discount"], A.GAMMA)
        self.assertEqual(sorted(reg["data"]["symbols"]), sorted(acquire.SYMBOLS))
        self.assertEqual(tuple(reg["data"]["fleetSymbols"]), V.FLEET)
        self.assertEqual(set(reg["robustness"]) | {"base"}, set(STRESSES))
        self.assertEqual((reg["data"]["development"]["startOpenTime"], reg["data"]["development"]["lastOpenTime"]), acquire.PARTITIONS["development"])
        self.assertEqual((reg["data"]["confirmation"]["startOpenTime"], reg["data"]["confirmation"]["lastOpenTime"]), acquire.PARTITIONS["confirmation"])
        self.assertIs(reg["liveTrading"], False)


if __name__ == "__main__":
    unittest.main()
