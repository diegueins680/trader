"""Deterministic engineering fixtures; no protected market data or network."""
import hashlib
from contextlib import contextmanager
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/research"))
from sequential_env import ACTIONS, Execution, Replay, Scale, collect, market_features, shield
from sequential_learning import Network, advantages, bellman_gradient, infer, load_policy, ppo_gradient, save_policy, train_ppo, train_q
from sequential_evaluation import Baselines, economic, ope_estimates, replay_policy, short_ope
from run_sequential_screen import REGISTRATION, load_development
import run_sequential_screen as runner
from summarize_sequential_screen import export
import summarize_sequential_screen as exporter
from sequential_registry import reconcile_groups


class SequentialContracts(unittest.TestCase):
    def setUp(self):
        self.p = 100 * np.exp(np.sin(np.arange(256) / 9) * 0.01)
        self.f = np.zeros(256)
        self.scale = Scale.fit([self.p[:160]])

    def env(self, **kw):
        return Replay(self.p, self.f, 30, 61, 1, self.scale, **kw)

    def test_future_changes_do_not_change_observation(self):
        a = self.env(enabled=True)
        b = self.p.copy(); b[31:] *= 19
        other = Replay(b, self.f, 30, 61, 1, self.scale, enabled=True)
        np.testing.assert_array_equal(a.observation(), other.observation())
        for t in range(24, 80):
            b = self.p.copy(); b[t + 1:] = np.nan
            np.testing.assert_array_equal(market_features(self.p, t), market_features(b, t))

    def test_scaler_training_prefix_is_immutable(self):
        p = self.p.copy(); p[160:] *= 100
        other = Scale.fit([p[:160]])
        np.testing.assert_array_equal(self.scale.mean, other.mean)
        with self.assertRaises(ValueError):
            self.scale.mean[0] = 1

    def test_scale_rejects_invalid_parameters(self):
        valid = dict(mean=np.zeros(6), std=np.ones(6), low=np.full(6, -1.), high=np.ones(6))
        for field in valid:
            for bad in (None, [0.]*6, np.zeros(5), np.zeros((6, 1)), np.zeros(6, dtype=complex),
                        np.zeros(6, dtype=bool), np.zeros(6, dtype=object), np.full(6, np.nan),
                        np.full(6, np.inf), np.ma.array(np.zeros(6), mask=True)):
                with self.subTest(field=field, value=repr(bad)), self.assertRaises(ValueError):
                    Scale(**{**valid, field: bad})
        for std in (np.zeros(6), np.full(6, -1.)):
            with self.assertRaises(ValueError):
                Scale(**{**valid, "std": std})
        with self.assertRaises(ValueError):
            Scale(**{**valid, "low": np.full(6, 2.)})

    def test_scale_snapshots_cannot_be_mutated_through_arrays(self):
        arrays = dict(mean=np.zeros(6), std=np.ones(6), low=np.full(6, -1.), high=np.ones(6))
        direct = Scale(**arrays)
        for name, original in arrays.items():
            before = getattr(direct, name).copy()
            original[:] = 19.
            np.testing.assert_array_equal(getattr(direct, name), before)
        for scale in (direct, self.scale):
            for name in arrays:
                with self.subTest(name=name), self.assertRaises(ValueError):
                    getattr(scale, name).setflags(write=True)

    def test_scale_queries_require_finite_real_feature_vectors(self):
        scale = Scale(np.zeros(6), np.ones(6), np.full(6, -1.), np.ones(6))
        for bad in (None, 0., np.zeros(1), np.zeros((1, 6)), [0.]*6, np.zeros(6, dtype=bool),
                    np.zeros(6, dtype=complex), np.zeros(6, dtype=object), np.full(6, np.nan),
                    np.ma.array(np.zeros(6), mask=True)):
            with self.subTest(value=repr(bad)):
                self.assertFalse(scale.supported(bad))
                with self.assertRaises(ValueError):
                    scale.transform(bad)
        np.testing.assert_array_equal(scale.transform(np.full(6, .5)), np.full(6, .5))
        self.assertTrue(scale.supported(np.full(6, .5)))
        self.assertFalse(scale.supported(np.full(6, 2.)))
        bound = np.iinfo(np.int64).max
        integer = Scale(np.full(6, bound, dtype=np.int64), np.ones(6),
                        np.full(6, -bound, dtype=np.int64), np.full(6, bound, dtype=np.int64))
        np.testing.assert_array_equal(integer.transform(np.full(6, -bound, dtype=np.int64)),
                                      np.full(6, -2.*bound))
        tiny = Scale(np.zeros(6), np.full(6, 1e-308), np.full(6, -1.), np.ones(6))
        with np.errstate(over="ignore"), self.assertRaises(ValueError):
            tiny.transform(np.full(6, 100.))
        # Overflowing normalization cannot produce a pending fill in replay.
        unstable = Scale(np.zeros(6), np.full(6, np.nextafter(0., 1.)), np.full(6, -1.), np.ones(6))
        env = Replay(self.p, self.f, 30, 61, 1, unstable, enabled=True)
        with np.errstate(over="ignore"):
            self.assertIsNone(env.observation())
            env.step(.25)
        self.assertEqual(env.failure, "invalid_observation_or_position")
        self.assertEqual(env.rows, []); self.assertEqual(env.units, 0)

    def test_scale_fit_rejects_incomplete_prefixes(self):
        for prefixes in (None, [], [self.p, self.p[:0]], [self.p, self.p[:24]],
                         [self.p, self.p[:24].astype(complex)], [self.p, None]):
            with self.subTest(prefixes=repr(prefixes)), self.assertRaises(ValueError):
                Scale.fit(prefixes)
        single = Scale.fit([self.p[:25]])
        np.testing.assert_array_equal(single.mean, market_features(self.p, 24))
        np.testing.assert_array_equal(single.std, np.full(6, 1e-8))

    def test_invalid_inputs_are_absent(self):
        for value in [np.nan, np.inf, -np.inf, 0, -1]:
            p = self.p.copy(); p[30] = value
            self.assertIsNone(market_features(p, 30))
        self.assertIsNone(market_features(self.p, 23))

    def test_shield_exhaustive_bounds_and_gates(self):
        for enabled in [False, True]:
            for valid in [False, True]:
                for ownership in [False, True]:
                    for elapsed in [-1, 0, 20, 21, np.nan, np.inf]:
                        for action in [-1, -.25, 0, .25, 1, np.nan, np.inf, True, None]:
                            a, _ = shield(action, enabled=enabled, valid=valid, ownership=ownership, elapsed_ms=elapsed)
                            if a is not None:
                                self.assertTrue(enabled and valid and ownership)
                                self.assertTrue(0 <= elapsed <= 20)
                                self.assertIn(a, ACTIONS)

    def test_disabled_policy_has_no_transition(self):
        e = self.env()
        _, reward, done = e.step(.25)
        self.assertTrue(done); self.assertEqual(reward, 0)
        self.assertEqual(e.equity, 1); self.assertEqual(e.units, 0)
        self.assertEqual(e.rows, [])
        self.assertIsNone(infer(Network(11), np.zeros(12))[0])

    def test_proposal_requires_boolean_evidence_and_real_scalars(self):
        for field in ("enabled", "valid", "ownership"):
            for value in ("false", "true", 0, 1, None, np.bool_(True), [True], np.array([True, False])):
                with self.subTest(field=field, value=repr(value)):
                    kwargs = dict(enabled=True, valid=True, ownership=True)
                    kwargs[field] = value
                    self.assertIsNone(shield(.25, **kwargs)[0])
        for value in (True, None, "0", 0j, np.complex128(0), np.array([0, 1]), 10**1000):
            with self.subTest(elapsed=repr(value)):
                self.assertIsNone(shield(.25, enabled=True, valid=True, elapsed_ms=value)[0])
        for value in (np.complex128(.25), .25+0j, np.bool_(False), ".25", np.array([.25])):
            with self.subTest(action=repr(value)):
                self.assertIsNone(shield(value, enabled=True, valid=True)[0])
        for action in (np.float32(-.25), np.float64(.25), np.int64(0), -.25, 0, .25):
            self.assertEqual(shield(action, enabled=True, valid=True, elapsed_ms=np.float64(20))[0], float(action))

    def test_invalid_replay_gates_cannot_fill_pending_inventory(self):
        for field in ("enabled", "valid", "ownership", "elapsed_ms"):
            for value in ("false", 1 if field != "elapsed_ms" else True, np.array([True, False])):
                with self.subTest(field=field, value=repr(value)):
                    e = self.env(enabled=True, execution=Execution(extra_delay=1))
                    e.step(.25)
                    before = (e.t, e.equity, e.units, len(e.rows), e.pending)
                    kwargs = {field: value}
                    if field == "enabled":
                        e.enabled = value
                        kwargs = {}
                    with patch.object(e, "observation", side_effect=AssertionError("invalid gate read observation")):
                        obs, reward, done = e.step(.25, **kwargs)
                    self.assertIsNone(obs); self.assertEqual(reward, 0); self.assertTrue(done)
                    self.assertEqual((e.t, e.equity, e.units, len(e.rows), e.pending), before)
                    self.assertIsNotNone(e.failure)
        e = self.env(enabled=True)
        e.step(.25)
        units, equity, tick = e.units, e.equity, e.t
        self.assertGreater(units, 0)
        e.step(0., ownership=False)
        self.assertEqual((e.units, e.equity, e.t), (units, equity, tick))

    def test_inference_rejects_invalid_types_and_model_failures(self):
        net = Network(11)
        for enabled in ("false", 1, np.bool_(True), np.array([True, False])):
            with self.subTest(enabled=repr(enabled)), patch.object(net, "forward") as forward:
                self.assertIsNone(infer(net, np.zeros(12), enabled=enabled)[0])
                forward.assert_not_called()
        for obs in (None, [0.]*12, np.zeros(12, dtype=bool), np.zeros(12, dtype=complex),
                    np.zeros(12, dtype=object), np.zeros(11), np.full(12, np.nan),
                    np.ma.array(np.zeros(12), mask=[True]+[False]*11)):
            with self.subTest(observation=repr(obs)), patch.object(net, "forward") as forward:
                self.assertIsNone(infer(net, obs, enabled=True)[0])
                forward.assert_not_called()
        for out in ([0., 1., 2.], np.ones(3, dtype=bool), np.ones(3, dtype=complex),
                    np.ones(3, dtype=object), np.ones(2), np.full(3, np.inf),
                    np.ma.array(np.ones(3), mask=[True, False, False])):
            with self.subTest(output=repr(out)), patch.object(net, "forward", return_value=out):
                self.assertIsNone(infer(net, np.zeros(12), enabled=True)[0])
        with patch.object(net, "forward", side_effect=RuntimeError("fixture")):
            proposal, elapsed = infer(net, np.zeros(12), enabled=True)
            self.assertIsNone(proposal); self.assertTrue(np.isfinite(elapsed) and elapsed >= 0)
        with patch.object(net, "forward", return_value=np.array([-1., 0., 1.])):
            self.assertEqual(infer(net, np.zeros(12), enabled=True)[0], .25)
            for elapsed_ns in (-1, 20_000_000, 20_000_001):
                with patch("sequential_learning.time.perf_counter_ns", side_effect=[0, elapsed_ns]):
                    proposal, _ = infer(net, np.zeros(12), enabled=True)
                    self.assertEqual(proposal, .25 if elapsed_ns == 20_000_000 else None)

    def baseline_fixture(self):
        controls = Baselines.__new__(Baselines)
        controls.scale = Scale(np.zeros(6), np.ones(6), np.full(6, -10.), np.full(6, 10.))
        controls.horizon = 1
        controls.mean = .125
        controls.ridge = np.arange(7, dtype=float) / 16
        controls.logistic = np.arange(7, dtype=float) / 8
        controls.bandit = np.arange(39, dtype=float).reshape(13, 3) / 16
        controls.clone_action = -.25
        return controls

    def test_baselines_reject_invalid_observations_before_computation(self):
        controls = self.baseline_fixture()
        invalid = (None, [0.]*12, np.zeros(11), np.zeros((1, 12)),
                   np.zeros(12, dtype=bool), np.zeros(12, dtype=complex),
                   np.zeros(12, dtype=object), np.full(12, np.nan), np.full(12, np.inf),
                   np.ma.array(np.zeros(12), mask=[True]+[False]*11))
        for obs in invalid:
            for name in Baselines.names:
                with self.subTest(name=name, observation=repr(obs)):
                    rng = np.random.default_rng(11)
                    state = json.dumps(rng.bit_generator.state, sort_keys=True)
                    self.assertIsNone(controls.action(name, obs, rng))
                    self.assertEqual(json.dumps(rng.bit_generator.state, sort_keys=True), state)
            for name in ("historical_mean", "last_return", "momentum", "reversal", "ridge_optimizer"):
                with self.subTest(forecast=name, observation=repr(obs)):
                    self.assertIsNone(controls.forecast(name, obs))

    def test_baselines_reject_unknown_names(self):
        controls = self.baseline_fixture()
        for name in ("ridge_optmizer", "", None, 1):
            with self.subTest(name=name):
                rng = np.random.default_rng(11)
                state = json.dumps(rng.bit_generator.state, sort_keys=True)
                self.assertIsNone(controls.action(name, np.zeros(12), rng))
                self.assertIsNone(controls.forecast(name, np.zeros(12)))
                self.assertEqual(json.dumps(rng.bit_generator.state, sort_keys=True), state)
        for name in set(Baselines.names) - {"historical_mean", "last_return", "momentum", "reversal", "ridge_optimizer"}:
            self.assertIsNone(controls.forecast(name, np.zeros(12)))

    def test_baseline_absence_is_rejected_by_replay_shield(self):
        controls = self.baseline_fixture()
        def choose(obs):
            return controls.action("ridge_optmizer", obs, np.random.default_rng(11)), 0.
        env, result = replay_policy(self.p, self.f, 30, 61, 1, self.scale, choose)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["reason"], "invalid_action")
        self.assertEqual(result["observations"], 0)
        self.assertEqual(env.fills, 0)
        self.assertEqual(env.actions, [])
        self.assertEqual(env.equity, 1.)
        self.assertEqual(env.rejections, 1)

    def test_valid_baselines_retain_exact_actions_and_forecasts(self):
        controls = self.baseline_fixture()
        rng = np.random.default_rng(11)
        rows = []
        for obs in (np.zeros(12), np.ones(12), -np.ones(12), np.arange(12, dtype=float) / 16):
            rows.append({"actions": [controls.action(name, obs, rng) for name in Baselines.names],
                         "forecasts": [controls.forecast(name, obs) for name in
                                       ("historical_mean", "last_return", "momentum", "reversal", "ridge_optimizer")]})
        actual = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
        self.assertEqual(actual, "1866180923c03a7716705d8dc021ccbac977c77992586f63acf697f5ffc2dc35")

    def test_baselines_reject_invalid_fitted_parameters(self):
        cases = []
        for name, field, shape in (("ridge_optimizer", "ridge", (7,)),
                                   ("logistic", "logistic", (7,)),
                                   ("contextual_bandit", "bandit", (13, 3))):
            values = (None, np.ones(shape).tolist(), np.ones(shape, dtype=bool),
                      np.ones(shape, dtype=complex), np.ones(shape, dtype=object),
                      np.ones((1,)), np.full(shape, np.nan), np.full(shape, np.inf),
                      np.ma.array(np.ones(shape), mask=True))
            cases.extend((name, field, value) for value in values)
        cases.extend(("historical_mean", "mean", value) for value in
                     (None, True, "0.1", np.array(.1), np.nan, np.inf))
        cases.extend(("behavior_clone", "clone_action", value) for value in
                     (None, True, "0.25", np.array(.25), np.nan, np.inf, .5))
        for name, field, value in cases:
            with self.subTest(name=name, field=field, value=repr(value)):
                controls = self.baseline_fixture(); setattr(controls, field, value)
                rng = np.random.default_rng(11)
                before = json.dumps(rng.bit_generator.state, sort_keys=True)
                with np.errstate(all="ignore"):
                    self.assertIsNone(controls.action(name, np.ones(12), rng))
                    if name in ("historical_mean", "ridge_optimizer"):
                        self.assertIsNone(controls.forecast(name, np.ones(12)))
                self.assertEqual(json.dumps(rng.bit_generator.state, sort_keys=True), before)

    def test_baselines_reject_nonfinite_intermediate_arithmetic(self):
        for name in ("ridge_optimizer", "logistic", "contextual_bandit"):
            with self.subTest(name=name):
                controls = self.baseline_fixture()
                field = "ridge" if name == "ridge_optimizer" else "bandit" if name == "contextual_bandit" else name
                setattr(controls, field, np.full(getattr(controls, field).shape, 1e308))
                with np.errstate(all="ignore"):
                    self.assertIsNone(controls.action(name, np.ones(12), np.random.default_rng(11)))
                    if name == "ridge_optimizer":
                        self.assertIsNone(controls.forecast(name, np.ones(12)))
        controls = self.baseline_fixture()
        controls.scale = Scale(np.full(6, 1e308), np.full(6, 1e308), np.zeros(6), np.ones(6))
        for name in ("last_return", "momentum", "reversal"):
            with self.subTest(name=name), np.errstate(all="ignore"):
                self.assertIsNone(controls.forecast(name, np.ones(12)))
                self.assertIsNone(controls.action(name, np.ones(12), np.random.default_rng(11)))

    def test_baseline_numerical_failures_reach_shield_as_absence(self):
        controls = self.baseline_fixture(); controls.logistic[:] = np.inf
        def choose(obs):
            return controls.action("logistic", obs, np.random.default_rng(11)), 0.
        with np.errstate(all="ignore"):
            env, result = replay_policy(self.p, self.f, 30, 61, 1, self.scale, choose)
        self.assertEqual(result["reason"], "invalid_action")
        self.assertEqual(result["observations"], 0)
        self.assertEqual(env.fills, 0)
        self.assertEqual(env.equity, 1.)

    def test_baseline_numeric_admission_preserves_independent_rules(self):
        controls = self.baseline_fixture()
        controls.clone_action = None
        controls.ridge[:] = np.nan
        controls.logistic[:] = np.nan
        controls.bandit[:] = np.nan
        controls.mean = np.nan
        for name, expected in (("cash", 0.), ("constant_long", .25), ("constant_short", -.25)):
            self.assertEqual(controls.action(name, np.zeros(12), np.random.default_rng(11)), expected)
        rng, reference = np.random.default_rng(11), np.random.default_rng(11)
        self.assertEqual(controls.action("behavior_uniform", np.zeros(12), rng), float(reference.choice(ACTIONS)))
        self.assertEqual(rng.bit_generator.state, reference.bit_generator.state)
        controls.logistic[:] = 1e300  # Large but finite score keeps existing clipping semantics.
        self.assertEqual(controls.action("logistic", np.ones(12), rng), .25)
        controls.clone_action = np.float64(-.25)
        self.assertEqual(controls.action("behavior_clone", np.zeros(12), rng), -.25)
        controls.mean = np.float64(.125)
        self.assertEqual(controls.forecast("historical_mean", np.zeros(12)), .125)

    def test_delay_prevents_same_bar_profit(self):
        p = np.full(65, 100.0); p[31:] = 200
        e = Replay(p, np.zeros(65), 30, 34, 1, Scale.fit([np.full(65, 100.0)]), enabled=True)
        e.step(.25)
        self.assertAlmostEqual(e.rows[0]["gross"], 0)
        self.assertAlmostEqual(e.equity, .99975)
        self.assertAlmostEqual(e.units, .25 / 200)

    def test_exact_fee_spread_slippage_and_terminal_accounting(self):
        p = np.full(65, 100.0)
        e = Replay(p, np.zeros(65), 30, 33, 1, Scale.fit([p]), enabled=True)
        e.step(.25); e.step(0)
        self.assertTrue(e.done); self.assertIsNone(e.failure)
        self.assertEqual(e.units, 0)
        result = economic(e)
        self.assertAlmostEqual(result["netReturn"], -.0005)
        self.assertAlmostEqual(result["costsOverInitialEquity"]["fee"], .00025)
        self.assertAlmostEqual(result["costsOverInitialEquity"]["spread"], .000025)
        self.assertAlmostEqual(result["costsOverInitialEquity"]["slippage"], .000225)
        with self.assertRaises(ValueError):
            e.step(0)

    def test_funding_is_charged_to_old_units_before_fill(self):
        p = np.full(65, 100.0); f = np.zeros(65); f[31:33] = .01
        e = Replay(p, f, 30, 33, 1, Scale.fit([p]), enabled=True)
        e.step(.25); e.step(0)
        self.assertAlmostEqual(e.rows[0]["funding"], 0)
        self.assertAlmostEqual(e.rows[1]["funding"], -.25 / 100 * .01)
        economic(e)

    def test_missing_invalid_timeout_and_ownership_fail_closed(self):
        for kw in [{"valid": False}, {"ownership": False}, {"elapsed_ms": 21}]:
            e = self.env(enabled=True)
            e.step(.25, **kw)
            self.assertTrue(e.done); self.assertEqual(e.units, 0)
        for action in [np.nan, np.inf, -.5, .5]:
            e = self.env(enabled=True); e.step(action)
            self.assertEqual(e.failure, "invalid_action")

    def test_partial_missed_and_delayed_fills(self):
        for cfg in [Execution(fill_fraction=.5), Execution(miss_every=1), Execution(extra_delay=1)]:
            e = self.env(execution=cfg, enabled=True)
            e.step(.25)
            if cfg.fill_fraction == .5:
                self.assertAlmostEqual(e.units * self.p[31], .125)
            else:
                self.assertEqual(e.units, 0)

    def test_risk_breach_stops_without_cash_padding(self):
        p = np.full(80, 100.0); p[32:] = 1
        e = Replay(p, np.zeros(80), 30, 70, 1, Scale.fit([np.full(80, 100.0)]), enabled=True)
        e.step(.25); e.step(.25)
        self.assertTrue(e.done); self.assertIsNotNone(e.failure)
        self.assertEqual(len(e.rows), 2)
        self.assertEqual(e.units, 0)
        self.assertEqual(economic(e)["status"], "failed")

    def test_replay_is_exactly_deterministic(self):
        for h in [1, 3, 6]:
            a = Replay(self.p, self.f, 30, 61, h, self.scale, enabled=True)
            b = Replay(self.p, self.f, 30, 61, h, self.scale, enabled=True)
            while not a.done:
                a.step(.25); b.step(.25)
            self.assertEqual(a.rows, b.rows)

    def test_buffer_cannot_cross_training_prefix(self):
        a = collect({"x": self.p[:160]}, {"x": self.f[:160]}, self.scale, 3, 11, 64)
        altered = self.p.copy(); altered[160:] *= 100
        b = collect({"x": altered[:160]}, {"x": self.f[:160]}, self.scale, 3, 11, 64)
        for key in ["s", "a", "r", "next", "done", "prob"]:
            np.testing.assert_array_equal(a[key], b[key])
        self.assertEqual(set(a["a"]), {0, 1, 2})
        np.testing.assert_array_equal(a["prob"], np.full(64, 1/3))

    def test_episode_admission_requires_integral_bounds_and_progress(self):
        for field in ("start", "stop", "horizon"):
            for bad in (True, np.bool_(True), 30.0, 33.5, None, "31", np.array([30, 31])):
                with self.subTest(field=field, value=repr(bad)), self.assertRaises(ValueError):
                    args = dict(start=30, stop=61, horizon=1)
                    args[field] = bad
                    Replay(self.p, self.f, scale=self.scale, **args)
        for h in (1, 3, 6):
            for length in (2, 7, 19):
                e = Replay(self.p, self.f, np.int64(30), np.int64(30+length), np.int64(h), self.scale, enabled=True)
                for _ in range((length-2)//h+1):
                    old = e.t
                    e.step(0.)
                    self.assertGreater(e.t, old)
                self.assertTrue(e.done)
                self.assertEqual(e.t, 30+length-1)
                self.assertEqual([r["t"] for r in e.rows], list(range(31, 30+length)))

    def test_series_admission_is_real_unmasked_and_causal(self):
        bad_series = (None, self.p.tolist(), self.p[:, None], self.p.astype(complex)+1j,
                      self.p.astype(object), self.p.astype(str), np.ones(256, dtype=bool),
                      np.ma.array(self.p, mask=np.arange(256) == 30))
        for bad in bad_series:
            with self.subTest(series=type(bad).__name__):
                self.assertIsNone(market_features(bad, 30))
                with self.assertRaises(ValueError):
                    Replay(bad, self.f, 30, 61, 1, self.scale)
                with self.assertRaises(ValueError):
                    Replay(self.p, bad, 30, 61, 1, self.scale)
        for bad in (30.0, True, None, "30", np.array([30, 31])):
            with self.subTest(index=repr(bad)):
                self.assertIsNone(market_features(self.p, bad))
        prices, funding = self.p.copy(), self.f.copy()
        prices[31:] = np.nan; funding[31:] = np.nan
        e = Replay(prices, funding, 30, 61, 1, self.scale, enabled=True)
        np.testing.assert_array_equal(e.observation(), self.env(enabled=True).observation())
        e.step(.25)
        self.assertTrue(e.done); self.assertEqual(e.failure, "invalid_market_transition")
        self.assertEqual(e.rows, []); self.assertEqual(e.units, 0)
        for funding in (self.f[:, None], self.f.astype(complex), np.zeros(256, dtype=bool),
                        np.ma.array(self.f, mask=np.arange(256) == 31)):
            with self.subTest(replaced_funding=repr(funding)):
                e = self.env(enabled=True)
                e.funding = funding
                e.step(.25)
                self.assertEqual(e.failure, "invalid_market_transition")
                self.assertEqual(e.rows, []); self.assertEqual(e.units, 0)

    def test_collection_rejects_invalid_budget_and_panel_before_sampling(self):
        invalid = [{field: value} for field in ("count", "seed", "horizon")
                   for value in (True, np.bool_(True), 1.5, None, "1", np.array([1, 2]))]
        invalid += [{"count": 0}, {"count": -1}, {"seed": -1}, {"horizon": 2},
                    {"prices": {}}, {"funding": {}}, {"prices": {"x": self.p[:120]}},
                    {"funding": {"x": self.f[:-1]}}, {"prices": {"x": self.p[:, None]}},
                    {"funding": {"x": self.f.astype(complex)}},
                    {"prices": {1: self.p}, "funding": {1: self.f}}]
        for change in invalid:
            args = dict(prices={"x": self.p}, funding={"x": self.f}, scale=self.scale, horizon=1, seed=11, count=2)
            args.update(change)
            with self.subTest(change=repr(change)), patch("sequential_env.np.random.default_rng", side_effect=AssertionError("sampled before admission")):
                with self.assertRaises(ValueError):
                    collect(**args)
        result = collect({"x": self.p[:121]}, {"x": self.f[:121]}, Scale.fit([self.p[:121]]),
                         np.int64(3), np.int64(11), np.int64(2))
        self.assertEqual(len(result["s"]), 2)
        for bad in (None, [1/3]*3, np.ones(3, dtype=bool), np.ones(3, dtype=complex),
                    np.ma.array([0., .5, .5], mask=[True, False, False])):
            with self.subTest(probabilities=repr(bad)), self.assertRaises(ValueError):
                collect({"x": self.p}, {"x": self.f}, self.scale, 1, 11, 1, policy=lambda _: bad)

    def test_execution_assumptions_require_real_scalars(self):
        for field in ("cost_multiplier", "fill_fraction", "impact_bps", "funding_multiplier", "risk_penalty"):
            for bad in (False, np.bool_(True), 1+0j, np.complex128(1), None, "1", np.array([1]), 10**1000):
                with self.subTest(field=field, value=repr(bad)), self.assertRaises(ValueError):
                    Execution(**{field: bad})
        self.assertEqual(Execution(cost_multiplier=np.float64(1)).cost_multiplier, 1.)

    def test_collection_rejects_incomplete_market_steps(self):
        p = np.full(121, 100.)
        scale = Scale.fit([p])
        for horizon in (1, 3, 6):
            for offset in (1, horizon):
                for bad in (np.nan, np.inf):
                    f = np.zeros(121); f[24+offset] = bad
                    with self.subTest(horizon=horizon, offset=offset, value=bad):
                        with self.assertRaisesRegex(ValueError, "incomplete training transition"):
                            collect({"x": p}, {"x": f}, scale, horizon, 11, 1,
                                    policy=lambda _: np.array([0., 0., 1.]))
        # Insolvency cannot be treated as a fully liquidated terminal sample.
        f = np.zeros(121); f[26] = 1000.
        with self.assertRaisesRegex(ValueError, "incomplete training transition"):
            collect({"x": p}, {"x": f}, scale, 3, 11, 1,
                    policy=lambda _: np.array([0., 0., 1.]))
        # A later invalid transition aborts the whole batch, not just its last row.
        f = np.zeros(121); f[26] = np.nan
        with self.assertRaisesRegex(ValueError, "incomplete training transition"):
            collect({"x": p}, {"x": f}, scale, 1, 11, 2)

    def test_collection_rejects_missing_or_malformed_next_observations(self):
        observation = Replay.observation
        for bad in (None, np.zeros(11), np.full(12, np.nan), np.zeros(12, dtype=complex),
                    np.ma.array(np.zeros(12), mask=True)):
            def corrupt_next(env):
                return observation(env) if env.t == env.start else bad
            with self.subTest(next=repr(bad)), patch.object(Replay, "observation", corrupt_next):
                with self.assertRaisesRegex(ValueError, "incomplete training transition"):
                    collect({"x": self.p[:121]}, {"x": self.f[:121]}, self.scale, 1, 11, 1)

    def test_training_cannot_update_from_incomplete_collection(self):
        p = np.full(121, 100.); f = np.zeros(121); f[25:] = np.nan
        scale = Scale.fit([p])
        for seed in (11, 23, 47):
            for algorithm in ("ppo", "double_dqn", "cql"):
                args = ({"x": p}, {"x": f}, scale, 1, seed)
                with self.subTest(seed=seed, algorithm=algorithm):
                    with patch.object(Network, "update", side_effect=AssertionError("updated from invalid data")):
                        with self.assertRaisesRegex(ValueError, "incomplete training transition"):
                            if algorithm == "ppo":
                                train_ppo(*args, steps=4)
                            else:
                                train_q(*args, offline=algorithm == "cql", steps=4)

    def test_collection_keeps_accounted_risk_terminals(self):
        p = np.full(121, 100.); f = np.zeros(121); f[26] = 64.
        scale = Scale.fit([p])
        data = collect({"x": p}, {"x": f}, scale, 3, 11, 2,
                       policy=lambda _: np.array([0., 0., 1.]))
        self.assertTrue(data["done"].all())
        self.assertTrue((data["r"] < -15).all())
        np.testing.assert_array_equal(data["next"], np.zeros((2, 12)))
        self.assertEqual(data["episodes"][0]["failure"], "drawdown_limit")
        # Fully observed risk loss includes both entry and terminal exit costs.
        self.assertAlmostEqual(data["r"][0], -16.05)
        normal = collect({"x": p}, {"x": np.zeros(121)}, scale, 1, 11, 96,
                         policy=lambda _: np.array([0., 1., 0.]))
        self.assertEqual(np.flatnonzero(normal["done"]).tolist(), [95])
        self.assertTrue(np.isfinite(normal["next"]).all())
        self.assertEqual(normal["next"][0, 8], 1.)  # Live successor equity, not padding.
        np.testing.assert_array_equal(normal["next"][-1], np.zeros(12))

    def test_neural_gradients_match_finite_difference(self):
        n = Network(11)
        x = np.random.default_rng(4).normal(size=(4, 12))
        dz = np.random.default_rng(5).normal(size=(4, 3))
        g = n.gradients(x, dz)
        for name, row in n.p.items():
            for index in [tuple(0 for _ in row.shape), tuple(s-1 for s in row.shape)]:
                old = row[index]; eps = 1e-6
                row[index] = old + eps; hi = np.sum(n.forward(x) * dz)
                row[index] = old - eps; lo = np.sum(n.forward(x) * dz)
                row[index] = old
                self.assertAlmostEqual(g[name][index], (hi - lo)/(2*eps), places=7)

    def test_ppo_clipping_gradient_and_cql_gradient(self):
        q = np.array([[.1, -.2, .3], [.8, -.3, -.1]])
        actions = np.array([0, 2])
        functions = [lambda z: ppo_gradient(z, actions, np.array([.1, .8]), np.array([1., -1.])),
                     lambda z: bellman_gradient(z, actions, np.array([.5, -.2]), .1)]
        for fn in functions:
            _, grad = fn(q)
            for index in np.ndindex(q.shape):
                a = q.copy(); b = q.copy(); a[index] += 1e-6; b[index] -= 1e-6
                self.assertAlmostEqual(grad[index], (fn(a)[0]-fn(b)[0])/2e-6, places=6)

    def test_terminal_value_cannot_bootstrap(self):
        n = Network(1, 1); n.p["b2"][:] = 100
        d = {"s": np.zeros((2, 12)), "next": np.ones((2, 12)), "r": np.array([1., 2.]), "done": np.array([True, True])}
        _, targets = advantages(d, n, .99)
        np.testing.assert_allclose(targets, d["r"])

    def test_episode_accounting_includes_final_terminal_and_risk_loss(self):
        p = np.full(121, 100.); f = np.zeros(121); scale = Scale.fit([p])
        for h in (1, 3, 6):
            length = 96//h
            for count in (length-1, length, length+1, 2*length):
                data = collect({"x":p}, {"x":f}, scale, h, 11, count,
                               policy=lambda _:np.array([0., 1., 0.]))
                with self.subTest(horizon=h, count=count):
                    self.assertEqual(len(data["episodes"]), int(data["done"].sum()))
                    self.assertEqual(data["episodeAccountingV2"], dict(collections=1,
                        started=(count+length-1)//length, completed=count//length,
                        truncated=int(count%length != 0), decisions=count))
        f[26] = 64.
        data = collect({"x":p}, {"x":f}, scale, 3, 11, 1,
                       policy=lambda _:np.array([0., 0., 1.]))
        self.assertEqual(len(data["episodes"]), 1)
        self.assertEqual(data["episodes"][0]["failure"], "drawdown_limit")
        self.assertAlmostEqual(data["episodes"][0]["return"], -.1605)
        self.assertEqual(data["episodeAccountingV2"]["truncated"], 0)
        partial = collect({"x":p}, {"x":np.zeros(121)}, scale, 1, 11, 1,
                          policy=lambda _:np.array([0., 0., 1.]))
        self.assertEqual(partial["episodes"], [])
        self.assertEqual(partial["episodeAccountingV2"]["truncated"], 1)
        self.assertFalse(partial["done"][0])
        self.assertGreater(partial["next"][0, 6], 0)  # No fabricated terminal liquidation.

    def test_training_aggregates_episode_accounting_across_rollouts(self):
        p = np.full(121, 100.); f = np.zeros(121); scale = Scale.fit([p])
        for algorithm in ("ppo", "double_dqn", "cql", "cql_no_inventory_penalty"):
            for h in (1, 3):
                args = ({"x":p}, {"x":f}, scale, h, 11)
                observed = []
                def observe_collection(*args, **kwargs):
                    data = collect(*args, **kwargs)
                    completed = int(data["done"].sum())
                    truncated = int(not data["done"][-1])
                    observed.append(dict(collections=1, completed=completed, truncated=truncated,
                                         started=completed+truncated, decisions=len(data["done"])))
                    return data
                with patch("sequential_learning.collect", side_effect=observe_collection):
                    _, info = (train_ppo(*args, steps=257) if algorithm=="ppo" else
                        train_q(*args, offline=algorithm!="double_dqn", steps=257,
                                risk_penalty=0. if algorithm=="cql_no_inventory_penalty" else .01))
                offline = algorithm.startswith("cql")
                expected = {key:sum(row[key] for row in observed) for key in observed[0]}
                with self.subTest(algorithm=algorithm, horizon=h):
                    self.assertEqual(expected["collections"], 1 if offline else 2)
                    self.assertEqual(expected["decisions"], 257)
                    self.assertEqual(len(info["episodes"]), expected["completed"])
                    self.assertEqual(info["episodeAccountingV2"], expected)

    def test_export_reconciles_v2_episode_accounting_before_output(self):
        mutations = [None,
            lambda f:f.update(episodeAccountingV2=None),
            lambda f:f["episodeAccountingV2"].update(completed=True),
            lambda f:f["episodeAccountingV2"].update(completed=-1),
            lambda f:f["episodeAccountingV2"].update(completed=1),
            lambda f:f["episodeAccountingV2"].update(started=2),
            lambda f:f["episodeAccountingV2"].update(truncated=2, started=4),
            lambda f:f["episodeAccountingV2"].update(collections=0),
            lambda f:f["episodeAccountingV2"].update(collections=2),
            lambda f:f["episodeAccountingV2"].update(decisions=192),
            lambda f:f["episodeAccountingV2"].pop("started"),
            lambda f:f["episodeAccountingV2"].update(extra=1),
            lambda f:f.update(steps=True),
            lambda f:f.update(episodes=[]),
            lambda f:f["episodes"][0].update(failure="invalid_market_transition"),
            lambda f:f["episodes"][0].update(**{"return":-1.}),
            lambda f:f["episodes"][0].update(**{"return":True}),
            lambda f:f.update(episodeAccountingV3=f.pop("episodeAccountingV2"))]
        with tempfile.TemporaryDirectory() as td:
            for i, mutate in enumerate(mutations):
                root = Path(td)/str(i); _, values = self.export_fixture(root, evaluated_rl=True)
                fit = values["training.json"][0]
                fit.update(steps=193, episodes=[{"return":-.1605, "failure":"drawdown_limit"},
                                               {"return":0., "failure":None}],
                           episodeAccountingV2=dict(collections=1, started=3, completed=2, truncated=1, decisions=193))
                if mutate is not None: mutate(fit)
                (root/"training.json").write_text(json.dumps(values["training.json"])+"\n")
                index = json.loads((root/"evidence-index.json").read_text())
                index["training.json"] = runner.digest(root/"training.json")
                (root/"evidence-index.json").write_text(json.dumps(index)+"\n")
                output = Path(td)/f"review-{i}"
                kwargs = dict(rss_unit="bytes", platform_label="fixture",
                              expected_index_sha256=runner.digest(root/"evidence-index.json"))
                with self.subTest(mutation=i):
                    if mutate is not None:
                        with self.assertRaises(ValueError): export(root, output, **kwargs)
                        self.assertFalse(output.exists())
                    else:
                        export(root, output, **kwargs)
                        report = json.loads((output/"multi-seed-training.json").read_text())[0]
                        self.assertEqual(report["episodeAccountingV2"], fit["episodeAccountingV2"])
                        self.assertEqual(report["trainingEpisodes"], 2)
                        self.assertEqual(report["trainingFailures"], {"drawdown_limit":1, "complete":1})
                        self.assertAlmostEqual(report["trainingEpisodeReturns"]["mean"], -.08025)

    def test_legacy_episode_reports_preserve_exact_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); sha, _ = self.export_fixture(root/"input", evaluated_rl=True)
            export(root/"input", root/"review", rss_unit="bytes", platform_label="fixture", expected_index_sha256=sha)
            rows = {p.name:p.read_text() for p in sorted((root/"review").iterdir())}
        self.assertEqual(len(rows), 7)
        self.assertEqual(hashlib.sha256(json.dumps(rows, sort_keys=True, allow_nan=False).encode()).hexdigest(),
                         "d9cdec2fa64d52f3f12d4d41ec1a8d9a3c3c7aca902dddc9b30f021d5c89661b")

    def test_training_indices_are_admitted_before_initialization(self):
        changes = [{field: bad} for field in ("horizon", "seed", "steps")
                   for bad in (True, np.bool_(True), None, "1", 1.5, np.array([1, 2]))]
        changes += [{"horizon":2}, {"seed":-1}, {"steps":0}, {"steps":-1}, {"steps":-256}]
        for algorithm in ("ppo", "double_dqn", "cql"):
            for change in changes:
                kwargs = dict(horizon=3, seed=11, steps=4); kwargs.update(change)
                with self.subTest(algorithm=algorithm, change=repr(change)):
                    with patch("sequential_learning.Network", side_effect=AssertionError("initialized before admission")), \
                         patch("sequential_learning.np.random.default_rng", side_effect=AssertionError("sampled before admission")):
                        with self.assertRaises(ValueError):
                            if algorithm == "ppo":
                                train_ppo({"x":self.p}, {"x":self.f}, self.scale, **kwargs)
                            else:
                                train_q({"x":self.p}, {"x":self.f}, self.scale, offline=algorithm=="cql", **kwargs)

    def test_q_mode_and_risk_are_admitted_before_initialization(self):
        changes = [{"offline": bad} for bad in ("false", "true", 0, 1, None, np.bool_(True), np.array([True, False]))]
        changes += [{"risk_penalty": bad} for bad in (False, None, "0.01", -.01, np.nan, np.inf, 1j, np.array([.01]))]
        for change in changes:
            kwargs = dict(offline=False, risk_penalty=.01); kwargs.update(change)
            with self.subTest(change=repr(change)):
                with patch("sequential_learning.Network", side_effect=AssertionError("initialized before admission")), \
                     patch("sequential_learning.np.random.default_rng", side_effect=AssertionError("sampled before admission")):
                    with self.assertRaises(ValueError):
                        train_q({"x":self.p}, {"x":self.f}, self.scale, 3, 11, steps=4, **kwargs)

    def test_training_integer_offsets_cannot_wrap_before_collection(self):
        class StopBeforeCollection(Exception):
            pass
        seed = np.uint64(np.iinfo(np.uint64).max)
        budget = np.int64(np.iinfo(np.int64).max)
        for algorithm, offset in (("ppo",1000), ("double_dqn",3000), ("cql",2000)):
            def inspect_collection(prices, funding, scale, horizon, next_seed, count, **kwargs):
                self.assertIs(type(horizon), int); self.assertEqual(horizon, 3)
                self.assertIs(type(next_seed), int); self.assertEqual(next_seed, int(seed)+offset)
                self.assertIs(type(count), int)
                self.assertEqual(count, int(budget) if algorithm=="cql" else 256)
                raise StopBeforeCollection()
            with self.subTest(algorithm=algorithm), patch("sequential_learning.collect", side_effect=inspect_collection):
                with np.errstate(over="ignore"), self.assertRaises(StopBeforeCollection):
                    args = ({"x":self.p}, {"x":self.f}, self.scale, np.int64(3), seed)
                    if algorithm == "ppo": train_ppo(*args, steps=budget)
                    else: train_q(*args, offline=algorithm=="cql", steps=budget)

    def test_training_preserves_exact_positive_budget_and_numpy_semantics(self):
        for algorithm in ("ppo", "double_dqn", "cql"):
            def fit(horizon, seed, steps):
                args = ({"x":self.p[:160]}, {"x":self.f[:160]}, self.scale, horizon, seed)
                return train_ppo(*args, steps=steps) if algorithm=="ppo" else train_q(*args, offline=algorithm=="cql", steps=steps)
            with self.subTest(algorithm=algorithm):
                net, info = fit(3, 11, 257)
                self.assertEqual(info["steps"], 257)
                self.assertEqual(net.steps, 8 if algorithm=="ppo" else 257)
                if algorithm != "ppo":
                    self.assertEqual(info["bufferTransitions"], 257)
                    self.assertEqual(sum(info["behaviorActionCounts"]), 257)
                plain, plain_info = fit(3, 11, 4)
                numpy, numpy_info = fit(np.int64(3), np.int64(11), np.int64(4))
                self.assertIs(type(numpy_info["steps"]), int)
                self.assertEqual(plain_info, numpy_info)
                for key in plain.p:
                    np.testing.assert_array_equal(plain.p[key], numpy.p[key])

    def test_multiple_training_seeds_are_reproducible(self):
        for seed in [11, 23, 47]:
            for alg in ["ppo", "double_dqn", "cql"]:
                def fit():
                    args = ({"x": self.p[:160]}, {"x": self.f[:160]}, self.scale, 3, seed)
                    return train_ppo(*args, steps=64)[0] if alg == "ppo" else train_q(*args, offline=alg == "cql", steps=64)[0]
                a, b = fit(), fit()
                for k in a.p:
                    np.testing.assert_array_equal(a.p[k], b.p[k])

    def test_policy_save_rejects_invalid_parameters_before_file_creation(self):
        meta = dict(codeCommit="a"*40, registrationSha256="b"*64, dataSha256="c"*64,
                    seed=11, horizon=1, algorithm="ppo", fold=0)
        with tempfile.TemporaryDirectory() as td:
            for key, template in Network(11).p.items():
                bad_values = (None, template.tolist(), np.zeros(1), np.zeros(template.shape, dtype=bool),
                              np.zeros(template.shape, dtype=complex), np.zeros(template.shape, dtype=object),
                              np.full(template.shape, np.nan), np.full(template.shape, np.inf),
                              np.ma.array(template, mask=True))
                for index, bad in enumerate(bad_values):
                    target = Path(td)/f"{key}-{index}.json"
                    net = Network(11); net.p[key] = bad
                    with self.subTest(key=key, value=repr(bad)):
                        with self.assertRaises(ValueError):
                            save_policy(target, net, meta)
                        self.assertFalse(target.exists())
            for index, parameters in enumerate(({}, {**Network(11).p, "extra": np.ones(1)}, None)):
                target = Path(td)/f"fields-{index}.json"
                net = Network(11); net.p = parameters
                with self.subTest(parameters=repr(parameters)), self.assertRaises(ValueError):
                    save_policy(target, net, meta)

    def test_policy_save_retains_valid_v1_bytes_and_loader_parity(self):
        rows = []
        with tempfile.TemporaryDirectory() as td:
            for algorithm in ("ppo", "double_dqn", "cql", "cql_no_inventory_penalty"):
                for seed in (11, 23, 47):
                    for variant in ("float64", "float32", "integer_bias"):
                        net = Network(seed)
                        if variant == "float32": net.p = {k:v.astype(np.float32) for k,v in net.p.items()}
                        if variant == "integer_bias": net.p["b2"] = np.array([-1, 0, 1], dtype=np.int64)
                        meta = dict(codeCommit="a"*40, registrationSha256="b"*64, dataSha256="c"*64,
                                    seed=seed, horizon=3, algorithm=algorithm, fold=0)
                        path = Path(td)/f"{algorithm}-{seed}-{variant}.json"
                        sha = save_policy(path, net, meta)
                        loaded = load_policy(path, sha, meta)
                        np.testing.assert_array_equal(loaded.forward(np.ones(12)), net.forward(np.ones(12)))
                        rows.append(dict(algorithm=algorithm, seed=seed, variant=variant, sha256=sha,
                                         bytes=path.stat().st_size, artifact=path.read_text()))
        raw = json.dumps(rows, sort_keys=True, allow_nan=False).encode()
        self.assertEqual(hashlib.sha256(raw).hexdigest(),
                         "020dbfdf167008b238a7d1f414493eb9677686bf2c8d9113fd87ffd7f76474ee")

    def test_policy_save_never_overwrites_existing_artifact(self):
        meta = dict(codeCommit="a"*40, registrationSha256="b"*64, dataSha256="c"*64,
                    seed=11, horizon=1, algorithm="ppo", fold=0)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/"policy.json"
            save_policy(path, Network(11), meta); original = path.read_bytes()
            with self.assertRaises(FileExistsError): save_policy(path, Network(23), meta)
            self.assertEqual(path.read_bytes(), original)
            with self.assertRaises(ValueError): save_policy(path, Network(11, outputs=1), meta)
            self.assertEqual(path.read_bytes(), original)

    def test_runner_records_policy_save_failure_without_success_artifact(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); source = root/"input.csv"; source.write_text("fixture-only")
            registration = json.loads(REGISTRATION.read_text())
            registration["data"]["decisionHorizonBars"] = [1]
            registration["seeds"] = [11]
            registration["validation"]["outerFolds"] = [{"trainStop":160,"testStart":166,"testStop":240}]
            reg = root/"registration.json"; reg.write_text(json.dumps(registration))
            invalid = Network(11, outputs=1)
            with patch.object(runner, "REGISTRATION", reg), patch.object(runner, "source_commit", return_value="a"*40), \
                 patch.object(runner, "load_development", return_value=({"x":self.p}, {"x":self.f}, np.array([0,1]))), \
                 patch.object(runner, "ALGORITHMS", ("ppo",)), patch.object(runner, "Baselines") as controls, \
                 patch.object(runner, "train_ppo", return_value=(invalid, {})):
                controls.names = ()
                runner.run(source, source, root/"run")
            self.assertEqual(list((root/"run/policies").glob("*.json")), [])
            training = json.loads((root/"run/training.json").read_text())
            self.assertEqual(training[0]["status"], "failed")
            self.assertIn("parameter", training[0]["reason"])
            self.assertNotIn("artifactSha256", training[0])
            planned = json.loads((root/"run/planned-registry.json").read_text())
            events = [json.loads(line) for line in (root/"run/events.jsonl").read_text().splitlines()]
            terminal = [e for e in events if e["status"] in ("complete", "failed")]
            self.assertEqual({e["id"] for e in terminal}, {e["id"] for e in planned})
            self.assertTrue(all(e["status"] == "failed" for e in terminal))
            index_sha = hashlib.sha256((root/"run/evidence-index.json").read_bytes()).hexdigest()
            export(root/"run", root/"review", rss_unit="bytes", platform_label="fixture", expected_index_sha256=index_sha)
            self.assertFalse(json.loads((root/"review/evaluation-summary.json").read_text())["promotionAllowed"])

    def test_artifact_roundtrip_hash_version_and_nonfinite(self):
        meta = {"codeCommit": "a"*40, "registrationSha256": "b"*64, "dataSha256": "c"*64,
                "seed": 11, "horizon": 1, "algorithm": "ppo", "fold": 0}
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/"policy.json"; n = Network(11)
            sha = save_policy(p, n, meta)
            other = load_policy(p, sha, meta)
            np.testing.assert_array_equal(n.forward(np.ones(12)), other.forward(np.ones(12)))
            with self.assertRaises(ValueError): load_policy(p, "0"*64, meta)
            a = json.loads(p.read_text()); a["enabled"] = True
            p.write_text(json.dumps(a))
            with self.assertRaises(ValueError): load_policy(p, hashlib.sha256(p.read_bytes()).hexdigest(), meta)
            a["enabled"] = False; a["parameters"]["b2"][0] = float("nan")
            p.write_text(json.dumps(a))
            with self.assertRaises(ValueError): load_policy(p, hashlib.sha256(p.read_bytes()).hexdigest(), meta)

    def test_ope_known_policy_and_no_support(self):
        r = np.array([[1., 2.], [3., 4.]])
        probs = np.ones_like(r)
        result = ope_estimates(r, np.zeros_like(r, dtype=int), probs, probs, np.zeros_like(r), np.zeros((2,3)), 1)
        for k in ["ordinaryIS", "weightedIS", "perDecisionIS", "doublyRobust"]:
            self.assertEqual(result[k], 5)
        self.assertEqual(result["effectiveSampleSize"], 2)
        unsupported = ope_estimates(r, np.zeros_like(r, dtype=int), probs, np.zeros_like(r), np.zeros_like(r), np.zeros((2,3)), 1)
        self.assertIsNone(unsupported["weightedIS"])
        self.assertEqual(unsupported["effectiveSampleSize"], 0)
        self.assertFalse(unsupported["reliable"])

    def test_ope_rejects_invalid_domains_and_overflow(self):
        valid = dict(rewards=np.ones((2, 2)), actions=np.zeros((2, 2), dtype=int),
                     behavior_prob=np.ones((2, 2)), target_prob=np.ones((2, 2)),
                     q=np.zeros((2, 2)), v=np.zeros((2, 3)), gamma=1.)
        invalid = [("gamma", x) for x in [np.nan, np.inf, -1., 1.01, True, "0.99"]]
        invalid += [("actions", x) for x in [np.zeros((2, 1), dtype=int),
                    np.zeros((2, 2)), np.full((2, 2), -1), np.full((2, 2), 3),
                    np.zeros((2, 2), dtype=bool)]]
        invalid += [("rewards", np.full((2, 2), "1")), ("rewards", np.full((2, 2), np.inf)),
                    ("behavior_prob", np.zeros((2, 2))), ("target_prob", np.full((2, 2), 1.1)),
                    ("v", np.ones((2, 3))),
                    ("behavior_prob", np.full((2, 2), 1e-300)),
                    ("behavior_prob", np.full((2, 2), 1e-80)),
                    ("rewards", np.full((2, 2), 1e308))]
        for key, value in invalid:
            with self.subTest(key=key, value=value):
                with self.assertRaises(ValueError):
                    ope_estimates(**{**valid, key:value})
        for shape in [(0, 2), (2, 0)]:
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                ope_estimates(np.zeros(shape), np.zeros(shape, dtype=int), np.ones(shape),
                              np.ones(shape), np.zeros(shape), np.zeros((shape[0], shape[1]+1)), 1.)

    def ope_fixture(self):
        return dict(rewards=np.array([[1., 2.], [3., 4.]]),
                    actions=np.zeros((2, 2), dtype=int),
                    behavior_prob=np.ones((2, 2)), target_prob=np.ones((2, 2)),
                    q=np.zeros((2, 2)), v=np.zeros((2, 3)), gamma=1.)

    def test_ope_rejects_masks_before_array_coercion(self):
        valid = self.ope_fixture()
        for key in ("rewards", "actions", "behavior_prob", "target_prob", "q", "v"):
            for kind in ("partial", "all", "none"):
                mask = np.zeros(valid[key].shape, dtype=bool)
                if kind == "partial": mask.flat[0] = True
                if kind == "all": mask[:] = True
                value = np.ma.array(valid[key], mask=mask)
                with self.subTest(key=key, kind=kind), patch("sequential_evaluation.np.asarray", wraps=np.asarray) as convert:
                    with self.assertRaisesRegex(ValueError, "masked OPE input"):
                        ope_estimates(**{**valid, key:value})
                    convert.assert_not_called()
                np.testing.assert_array_equal(value.data, valid[key])
                np.testing.assert_array_equal(value.mask, mask)

    def test_ope_unmasked_arraylike_results_remain_exact(self):
        valid = self.ope_fixture()
        expected = ope_estimates(**valid)
        for name in ("ordinaryIS", "perDecisionIS", "weightedIS", "doublyRobust"):
            self.assertEqual(expected[name], 5.)
        self.assertEqual(expected["effectiveSampleSize"], 2.)
        self.assertFalse(expected["reliable"])
        for representation in ("lists", "tuples", "readonly"):
            values = {}
            for key, value in valid.items():
                if key == "gamma": values[key] = value; continue
                if representation == "lists": values[key] = value.tolist()
                elif representation == "tuples": values[key] = tuple(map(tuple, value.tolist()))
                else:
                    values[key] = value.copy()
                    values[key].flags.writeable = False
            with self.subTest(representation=representation):
                self.assertEqual(ope_estimates(**values), expected)

    def test_runner_records_masked_ope_failure_without_losing_fit(self):
        inputs = self.ope_fixture()
        inputs["rewards"] = np.ma.array(inputs["rewards"], mask=True)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with self.publication_fixture(root), patch.object(runner, "short_ope", side_effect=lambda *args: ope_estimates(**inputs)):
                runner.run(root/"unused-panel", root/"unused-funding", root/"run")
            ope = json.loads((root/"run/ope.json").read_bytes())
            self.assertEqual(len(ope), 1)
            self.assertEqual(ope[0]["result"], {"status":"failed", "reason":"ValueError: masked OPE input"})
            events = [json.loads(line) for line in (root/"run/events.jsonl").read_text().splitlines()]
            terminal = [e for e in events if e["id"] == ope[0]["id"] and e["status"] != "started"]
            self.assertEqual(len(terminal), 1)
            self.assertEqual(terminal[0]["status"], "complete")
            export(root/"run", root/"review", rss_unit="bytes", platform_label="fixture",
                   expected_index_sha256=runner.digest(root/"run/evidence-index.json"))
            self.assertEqual(json.loads((root/"review/ope-report.json").read_bytes()), ope)
            self.assertFalse(json.loads((root/"review/evaluation-summary.json").read_bytes())["promotionAllowed"])

    def short_ope_fixture(self, net, **overrides):
        p = np.full(128, 100.)
        args = dict(prices={"x":p}, funding={"x":np.zeros_like(p)}, scale=Scale.fit([p[:60]]),
                    horizon=1, start=30, stop=80, net=net, seed=0, episodes=1)
        return short_ope(**{**args, **overrides})

    def test_short_ope_rejects_invalid_outputs_before_transition(self):
        invalid = (None, [0., 1., 0.], np.ones(3, dtype=bool), np.ones(3, dtype=complex),
                   np.ones(3, dtype=object), np.ones(2), np.ones((1, 3)),
                   np.full(3, np.nan), np.full(3, np.inf),
                   np.ma.array(np.ones(3), mask=[True, False, False]))
        for phase in ("logged", "direct"):
            for value in invalid:
                net = Network(11)
                calls = 0
                def forward(obs):
                    nonlocal calls
                    calls += 1
                    return value if phase == "logged" or calls > 6 else np.array([0., 1., 0.])
                with self.subTest(phase=phase, value=repr(value)), patch.object(net, "forward", side_effect=forward), \
                     patch.object(Replay, "step", autospec=True, side_effect=Replay.step) as step:
                    with self.assertRaisesRegex(ValueError, "invalid OPE policy output"):
                        self.short_ope_fixture(net)
                    self.assertEqual(step.call_count, 0 if phase == "logged" else 6)

    def test_short_ope_rejects_invalid_observation_before_forward(self):
        net = Network(11)
        for obs in (None, np.full(12, np.nan), np.ones(11), np.ones(12, dtype=bool),
                    np.ma.array(np.ones(12), mask=True)):
            with self.subTest(obs=repr(obs)), patch.object(Replay, "observation", return_value=obs), \
                 patch.object(net, "forward", return_value=np.array([0., 1., 0.])) as forward, \
                 patch.object(Replay, "step") as step:
                with self.assertRaisesRegex(ValueError, "invalid OPE policy observation"):
                    self.short_ope_fixture(net)
                forward.assert_not_called()
                step.assert_not_called()

    def test_short_ope_valid_policy_retains_action_trace(self):
        net = Network(11)
        with patch.object(net, "forward", return_value=np.array([0., 1., 0.])), \
             patch.object(Replay, "step", autospec=True, side_effect=Replay.step) as step:
            result = self.short_ope_fixture(net)
        trace = [(call.args[0].start, float(call.args[1])) for call in step.call_args_list]
        actual = hashlib.sha256(json.dumps(trace).encode()).hexdigest()
        self.assertEqual(actual, "f104f3cd18e1919ded7ca5bbb242e70b046491d5b7e59d3a24b4e0af64db4e4a")
        self.assertEqual(len(trace), 12)
        self.assertEqual([action for _, action in trace[6:]], [0.]*6)
        self.assertEqual(result["directSimulatorValue"], 0.)
        self.assertFalse(result["reliable"])
        self.assertEqual(result["liveStateActionSupport"], "unavailable")

    def test_short_ope_retains_failed_episode_accounting(self):
        net = Network(11)
        with patch.object(net, "forward", return_value=np.array([0., 1., 0.])):
            result = self.short_ope_fixture(net, seed=11)
        self.assertEqual(result, {"status":"invalid", "failedEpisodes":1,
                                 "reason":"No failed episodes may be silently excluded from OPE."})

    def test_short_ope_forward_failure_is_explicit(self):
        net = Network(11)
        with patch.object(net, "forward", side_effect=RuntimeError("fixture inference failure")):
            with self.assertRaisesRegex(RuntimeError, "fixture inference failure"):
                self.short_ope_fixture(net)

    def test_ope_matches_enumerated_behavior_tree(self):
        # Enumerate all length-two trajectories of a uniform two-action policy.
        # The deterministic target selects action zero twice and earns 1 + .5*3.
        actions = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        rewards = np.array([[1., 3.], [1., 4.], [2., 3.], [2., 4.]])
        target = (actions == 0).astype(float)
        q = np.zeros_like(rewards)
        v = np.zeros((4, 3))
        result = ope_estimates(rewards, actions, np.full((4, 2), .5), target, q, v, .5)
        for name in ["ordinaryIS", "perDecisionIS", "weightedIS", "doublyRobust"]:
            self.assertAlmostEqual(result[name], 2.5)
        self.assertEqual(result["effectiveSampleSize"], 1.)
        self.assertEqual(result["maxTrajectoryWeight"], 4.)
        # With exact Q/V, every DR trajectory equals the target value, even
        # trajectories disagreeing with its first action (zero cumulative weight).
        q[:, 0] = rewards[:, 0] + 1.5
        q[:, 1] = rewards[:, 1]
        v[:, 0] = 2.5; v[:, 1] = 3.
        exact = ope_estimates(rewards, actions, np.full((4, 2), .5), target, q, v, .5)
        self.assertEqual(exact["conditionalBootstrap95"]["DR"], [2.5, 2.5])
        self.assertFalse(exact["reliable"])

    def test_artifact_types_and_size_fail_closed(self):
        meta = {"codeCommit": "a"*40, "registrationSha256": "b"*64, "dataSha256": "c"*64,
                "seed": 11, "horizon": 1, "algorithm": "ppo", "fold": 0}
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/"policy.json"
            save_policy(p, Network(11), meta)
            original = json.loads(p.read_text())
            for bad in ["0", True, None, {"value": 0}]:
                a = json.loads(json.dumps(original)); a["parameters"]["b2"][0] = bad
                p.write_text(json.dumps(a))
                with self.assertRaises(ValueError): load_policy(p, hashlib.sha256(p.read_bytes()).hexdigest(), meta)
            for bad in [[], None, "schema", {"schema": "offline_policy_v1"}]:
                p.write_text(json.dumps(bad))
                with self.assertRaises(ValueError): load_policy(p, hashlib.sha256(p.read_bytes()).hexdigest(), meta)
            p.write_bytes(b" " * 65537)
            with self.assertRaises(ValueError): load_policy(p, hashlib.sha256(p.read_bytes()).hexdigest(), meta)

    def test_data_admission_rejects_unregistered_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/"x.csv"; p.write_text("not registered data")
            with self.assertRaises(ValueError):
                load_development(p, p, json.loads(REGISTRATION.read_text()))

    def test_development_parser_uses_the_verified_byte_snapshot(self):
        with tempfile.TemporaryDirectory() as td:
            panel, settlements = Path(td)/"panel.csv", Path(td)/"settlements.csv"
            panel.write_text("symbol,openTime,closeTime,close\n"
                             "BTCUSDT,0,999,100\nBTCUSDT,1000,1999,101\nBTCUSDT,2000,2999,102\n")
            settlements.write_text("symbol,fundingTime,fundingRate,resolvedMarkPrice\n"
                                   "BTCUSDT,1999,0.001,100\n")
            spec = {"data": {"panelSha256": runner.digest(panel),
                    "settlementsSha256": runner.digest(settlements), "symbols": ["BTCUSDT"],
                    "startOpenTime": 0, "endOpenTime": 2000,
                    "intervalMilliseconds": 1000, "rowsPerSymbol": 3}}
            original_read = runner.pd.read_csv
            def replace_before_parse(source, *args, **kwargs):
                panel.write_text(panel.read_text().replace(",100\n", ",900\n"))
                settlements.write_text(settlements.read_text().replace(",0.001,", ",0.9,"))
                return original_read(source, *args, **kwargs)
            with patch.object(runner.pd, "read_csv", side_effect=replace_before_parse):
                prices, funding, _ = load_development(panel, settlements, spec)
            np.testing.assert_array_equal(prices["BTCUSDT"], [100., 101., 102.])
            np.testing.assert_allclose(funding["BTCUSDT"], [0., 0.1, 0.])
            self.assertFalse(prices["BTCUSDT"].flags.writeable)
            self.assertFalse(funding["BTCUSDT"].flags.writeable)
            # A subsequent invocation must reject the now-replaced bytes before parsing.
            with patch.object(runner.pd, "read_csv") as read:
                with self.assertRaises(ValueError):
                    load_development(panel, settlements, spec)
                read.assert_not_called()

    def test_artifact_rejects_invalid_provenance_even_with_matching_hash(self):
        valid = {"codeCommit": "a"*40, "registrationSha256": "b"*64, "dataSha256": "c"*64,
                 "seed": 11, "horizon": 1, "algorithm": "ppo", "fold": 0}
        invalid = [{}, *[{**valid, k:v} for k,v in [
            ("codeCommit","not-a-commit"), ("registrationSha256","z"*64),
            ("dataSha256",None), ("seed",True), ("seed",-1), ("seed",1.0),
            ("horizon",True), ("horizon",2), ("algorithm","unknown"),
            ("fold",False), ("fold",-1), ("fold",1.0),
            ("fundingSha256","unbound"), ("extra",float("inf"))]]]
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"policy.json"; save_policy(p,Network(11),valid)
            artifact=json.loads(p.read_text())
            for metadata in invalid:
                with self.subTest(metadata=metadata):
                    artifact["provenance"]=metadata
                    p.write_text(json.dumps(artifact))
                    sha=hashlib.sha256(p.read_bytes()).hexdigest()
                    with self.assertRaises(ValueError): load_policy(p,sha,metadata)
                    target=Path(td)/"invalid-output.json"
                    with self.assertRaises(ValueError): save_policy(target,Network(11),metadata)
                    self.assertFalse(target.exists())

    def test_training_failure_retains_every_planned_replay_and_export(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "input.csv"; source.write_text("fixture-only")
            registration = json.loads(REGISTRATION.read_text())
            registration["data"]["decisionHorizonBars"] = [1]
            registration["seeds"] = [11]
            registration["validation"]["outerFolds"] = [{"trainStop":160,"testStart":166,"testStop":240}]
            reg = root / "registration.json"; reg.write_text(json.dumps(registration))
            with patch.object(runner,"REGISTRATION",reg), patch.object(runner,"source_commit",return_value="a"*40), \
                 patch.object(runner,"load_development",return_value=({"x":self.p},{"x":self.f},np.array([0,1]))), \
                 patch.object(runner,"ALGORITHMS",("ppo",)), patch.object(runner,"Baselines") as controls, \
                 patch.object(runner,"train_ppo",side_effect=ValueError("fixture training failure")):
                controls.names = ()
                runner.run(source,source,root/"run")
            planned=json.loads((root/"run/planned-registry.json").read_text())
            events=[json.loads(line) for line in (root/"run/events.jsonl").read_text().splitlines()]
            terminal=[e for e in events if e["status"] in ("complete","failed")]
            self.assertEqual({e["id"] for e in terminal},{p["id"] for p in planned})
            self.assertEqual(len(terminal),10)
            self.assertTrue(all(e["status"]=="failed" for e in terminal))
            index_sha = hashlib.sha256((root/"run/evidence-index.json").read_bytes()).hexdigest()
            export(root/"run",root/"review",rss_unit="bytes",platform_label="fixture",expected_index_sha256=index_sha)
            result=json.loads((root/"review/evaluation-summary.json").read_text())
            self.assertIsNone(result["rlAllPathsMaxDrawdown"])
            self.assertFalse(result["promotionAllowed"])
            (root/"run/evaluation.json").write_text("[]")
            with self.assertRaises(ValueError):
                export(root/"run",root/"tampered",rss_unit="bytes",platform_label="fixture",expected_index_sha256=index_sha)
            self.assertFalse((root/"tampered").exists())

    def export_fixture(self, root, *, evaluated_rl=False):
        """Synthetic archive; optional untrained policy artifact, never market data."""
        trial = "ppo/h1/f0/s11"
        failed = {"id": trial+"/base/x", "algorithm": "ppo", "horizon": 1,
                  "fold": 0, "seed": 11, "stress": "base", "symbol": "x",
                  "result": {"status": "failed", "reason": "training_failed", "observations": 0}}
        cash = {"id": "cash/h1/f0/s20260917/base/x", "algorithm": "cash", "horizon": 1,
                "fold": 0, "seed": 20260917, "stress": "base", "symbol": "x",
                "result": {"status": "complete", "reason": None, "observations": 1,
                           "netReturn": 0., "maxDrawdown": 0., "expectedShortfall95": 0.,
                           "sharpe": None, "costsOverInitialEquity": {"fee": 0.},
                           "fundingPnlOverInitialEquity": 0.}}
        planned = [{"id": trial, "kind": "training", "status": "planned"},
                   *[{"id": r["id"], "kind": "replay", "status": "planned"} for r in (failed, cash)]]
        events = [{"id": trial, "status": "started"},
                  {"id": trial, "status": "failed", "reason": "fixture"},
                  {"id": failed["id"], **failed["result"]},
                  {"id": cash["id"], "status": "started"},
                  {"id": cash["id"], **{k: cash["result"][k] for k in ("status", "reason", "observations")}}]
        values = {
            "manifest.json": {"promotionAllowed": False, "holdoutOpened": False,
                              "liveAuthorization": False, "evidenceClass": "contaminated_development_only"},
            "summary.json": {"groups": runner.summary([failed, cash]),
                             "promotionAllowed": False, "holdoutOpened": False,
                             "decision": "no_candidate_passed", "evidenceClass": "contaminated_development_only",
                             "statistics": {"DSR": None, "PBO": None, "SPA": None, "pairedConfidence": None},
                             "trainingFits": 1, "replayPaths": 2, "plannedEntries": 3,
                             "processPeakRssPlatformUnits": 0},
            "evaluation.json": [failed, cash],
            "training.json": [{"id": trial, "algorithm": "ppo", "horizon": 1,
                               "fold": 0, "seed": 11, "status": "failed", "reason": "fixture"}],
            "planned-registry.json": planned,
            "events.jsonl": events,
            "ope.json": [],
        }
        root.mkdir()
        policy_index = {}
        if evaluated_rl:
            artifact = root/"policies/ppo_h1_f0_s11.json"
            artifact.parent.mkdir()
            sha = save_policy(artifact, Network(11), {"codeCommit": "a"*40,
                "registrationSha256": "b"*64, "dataSha256": "c"*64,
                "algorithm": "ppo", "seed": 11, "horizon": 1, "fold": 0})
            policy_index[str(artifact.relative_to(root))] = sha
            fit = values["training.json"][0]
            fit.pop("status")
            fit.pop("reason")
            fit.update(artifactSha256=sha, artifactBytes=artifact.stat().st_size)
            events[1] = {"id": trial, "status": "complete", "artifactSha256": sha}
            failed["result"] = {**cash["result"], "latencyP99Ms": .1, "oodObservationRate": 0.}
            events[2] = {"id": failed["id"], "status": "complete", "reason": None, "observations": 1}
            events.insert(2, {"id": failed["id"], "status": "started"})
            values["ope.json"] = [{"id": trial, "result": {"status": "invalid", "failedEpisodes": 1, "reason": "fixture"}}]
            values["summary.json"]["groups"] = runner.summary([failed, cash])
        for name, value in values.items():
            (root/name).write_text("".join(json.dumps(e)+"\n" for e in value)
                                   if name == "events.jsonl" else json.dumps(value)+"\n")
        # Last index entry deliberately exercises the streaming-only path.
        (root/"returns.csv").write_text("trial,symbol,outcomeIndex,netReturn\n")
        index = {name: runner.digest(root/name) for name in (*values, "returns.csv")}
        index.update(policy_index)
        (root/"evidence-index.json").write_text(json.dumps(index)+"\n")
        return runner.digest(root/"evidence-index.json"), values

    def rewrite_export_member(self, root, name, raw):
        (root/name).write_bytes(raw)
        index_path = root/"evidence-index.json"
        if name != "evidence-index.json":
            index = json.loads(index_path.read_bytes())
            index[name] = runner.digest(root/name)
            index_path.write_text(json.dumps(index)+"\n")
        return runner.digest(index_path)

    def test_export_rejects_invalid_resource_metadata(self):
        cases = [("training.json", "seconds", x) for x in (-1, True, None, "1")]
        cases += [("events.jsonl", "seconds", x) for x in (-1, True, None, "1")]
        cases += [("training.json", "artifactBytes", x) for x in (-1, 0, True, 1.5, "12", None)]
        cases += [("events.jsonl", "seconds", 2)]  # Conflicts with the fit's one second.
        for name, field, value in cases:
            for evaluated in (False, True):
                with self.subTest(name=name, field=field, value=value, evaluated=evaluated), tempfile.TemporaryDirectory() as td:
                    root = Path(td); source = root/"archive"
                    _, values = self.export_fixture(source, evaluated_rl=evaluated)
                    values["training.json"][0]["seconds"] = 1
                    values["events.jsonl"][1]["seconds"] = 1
                    values[name][1 if name == "events.jsonl" else 0][field] = value
                    for member in ("training.json", "events.jsonl"):
                        raw = ("".join(json.dumps(e)+"\n" for e in values[member])
                               if member == "events.jsonl" else json.dumps(values[member]))
                        sha = self.rewrite_export_member(source, member, raw.encode())
                    with self.assertRaises(ValueError):
                        export(source, root/"review", rss_unit="bytes", platform_label="fixture",
                               expected_index_sha256=sha)
                    self.assertFalse((root/"review").exists())

    def test_export_rejects_unknown_rss_units_before_reads(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for unit in ("byte", "KiB", "", None, True, 1024):
                with self.subTest(unit=unit), patch.object(Path, "read_bytes") as reads:
                    with self.assertRaises(ValueError):
                        export(root/"absent", root/"review", rss_unit=unit,
                               platform_label="fixture", expected_index_sha256="0"*64)
                    reads.assert_not_called()
                    self.assertFalse((root/"review").exists())

    def test_export_preserves_valid_resource_measurements(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); source = root/"archive"
            _, values = self.export_fixture(source, evaluated_rl=True)
            values["summary.json"]["processPeakRssPlatformUnits"] = 1048576
            for seconds in (0, 1.25):
                values["training.json"][0]["seconds"] = seconds
                values["events.jsonl"][1]["seconds"] = seconds
                for member in ("summary.json", "training.json", "events.jsonl"):
                    raw = ("".join(json.dumps(e)+"\n" for e in values[member])
                           if member == "events.jsonl" else json.dumps(values[member]))
                    sha = self.rewrite_export_member(source, member, raw.encode())
                for unit, expected in (("bytes", 1), ("kib", 1024)):
                    output = root/f"review-{seconds}-{unit}"
                    export(source, output, rss_unit=unit, platform_label="fixture",
                           expected_index_sha256=sha)
                    report = json.loads((output/"evaluation-summary.json").read_bytes())
                    self.assertEqual(report["trainingSecondsSum"], seconds)
                    self.assertEqual(report["peakResidentMemoryMiB"], expected)
                    size = values["training.json"][0]["artifactBytes"]
                    self.assertEqual(report["artifactByteRange"], [size, size])

    def test_export_rejects_duplicate_json_keys_in_every_input(self):
        cases = [
            ("evidence-index.json", '"manifest.json":"' + "0"*64 + '",'),
            ("manifest.json", '"promotionAllowed":true,'),
            ("summary.json", '"decision":"integrate",'),
            ("evaluation.json", '"symbol":"wrong",'),
            ("training.json", '"algorithm":"wrong",'),
            ("planned-registry.json", '"kind":"replay",'),
            ("events.jsonl", '"status":"complete",'),
            ("ope.json", '"id":"wrong",'),
            ("manifest.json", '"promotionAllowed":false,'),
            ("manifest.json", r'"promotion\u0041llowed":true,'),
            ("training.json", r'"audit":{"key":1,"\u006bey":2},')]
        with tempfile.TemporaryDirectory() as td:
            for i, (name, prefix) in enumerate(cases):
                root = Path(td)/str(i); self.export_fixture(root, evaluated_rl=True)
                raw = (root/name).read_bytes().replace(b"{", b"{"+prefix.encode(), 1)
                sha = self.rewrite_export_member(root, name, raw)
                output = Path(td)/f"review-{i}"
                with self.subTest(name=name, prefix=prefix):
                    with self.assertRaises(ValueError):
                        export(root, output, rss_unit="bytes", platform_label="fixture", expected_index_sha256=sha)
                    self.assertFalse(output.exists())

    def test_export_rejects_nonfinite_numbers_even_in_omitted_fields(self):
        with tempfile.TemporaryDirectory() as td:
            for i, token in enumerate(("NaN", "Infinity", "-Infinity", "1e309", "-1e309")):
                for name in ("evaluation.json", "training.json", "planned-registry.json", "events.jsonl"):
                    root = Path(td)/f"{i}-{name}"; self.export_fixture(root, evaluated_rl=True)
                    prefix = ('"losses":[0,'+token+',0],') if name == "training.json" else ('"ignored":{"values":['+token+']},')
                    raw = (root/name).read_bytes().replace(b"{", b"{"+prefix.encode(), 1)
                    sha = self.rewrite_export_member(root, name, raw)
                    output = Path(td)/f"review-{i}-{name}"
                    with self.subTest(name=name, token=token):
                        with self.assertRaises(ValueError):
                            export(root, output, rss_unit="bytes", platform_label="fixture", expected_index_sha256=sha)
                        self.assertFalse(output.exists())

    def test_evidence_json_preserves_unambiguous_values(self):
        raw = b'{"text":"NaN Infinity 1e309", "flag":true, "missing":null, "rows":[-0.0,1.25e-2,1e308], "large":340282366920938463463374607431768211456, "key\\u00e9": "caf\\u00e9"}'
        value = exporter.decode_evidence(raw)
        self.assertEqual(value, json.loads(raw))
        self.assertTrue(np.signbit(value["rows"][0]))
        self.assertIs(type(value["large"]), int)
        self.assertEqual(value["keyé"], "café")

    def test_export_rejects_inconsistent_hash_valid_registry(self):
        def set_outcome(values, status, reason):
            values["evaluation.json"][1]["result"].update(status=status, reason=reason)
            values["events.jsonl"][-1].update(status=status, reason=reason)
            values["summary.json"]["groups"] = runner.summary(values["evaluation.json"])

        mutations = {
            "missing RSS": lambda v: v["summary.json"].pop("processPeakRssPlatformUnits"),
            "nonnumeric RSS": lambda v: v["summary.json"].update(processPeakRssPlatformUnits="bad"),
            "nonfinite RSS": lambda v: v["summary.json"].update(processPeakRssPlatformUnits=float("nan")),
            "negative RSS": lambda v: v["summary.json"].update(processPeakRssPlatformUnits=-1),
            "malformed training episode": lambda v: v["training.json"][0].update(episodes=[{}]),
            "malformed training seconds": lambda v: v["training.json"][0].update(seconds="bad"),
            "failed with null reason": lambda v: set_outcome(v, "failed", None),
            "failed with empty reason": lambda v: set_outcome(v, "failed", ""),
            "failed with blank reason": lambda v: set_outcome(v, "failed", " "),
            "failed with reserved reason": lambda v: set_outcome(v, "failed", "complete"),
            "complete with failure reason": lambda v: set_outcome(v, "complete", "drawdown"),
            "summary promotion": lambda v: v["summary.json"].update(promotionAllowed=True),
            "summary holdout": lambda v: v["summary.json"].update(holdoutOpened=True),
            "summary live authorization": lambda v: v["summary.json"].update(liveAuthorization=True),
            "summary decision": lambda v: v["summary.json"].update(decision="candidate_passed"),
            "summary evidence class": lambda v: v["summary.json"].update(evidenceClass="confirmation"),
            "summary statistical inference": lambda v: v["summary.json"]["statistics"].update(DSR=.99),
            "summary extra statistical claim": lambda v: v["summary.json"]["statistics"].update(passed=True),
            "manifest nonboolean authorization": lambda v: v["manifest.json"].update(liveAuthorization=0),
            "duplicate planned ID": lambda v: v["planned-registry.json"].append(v["planned-registry.json"][0]),
            "missing failed replay": lambda v: v["evaluation.json"].pop(0),
            "duplicate replay": lambda v: v["evaluation.json"].append(v["evaluation.json"][0]),
            "missing training": lambda v: v["training.json"].clear(),
            "missing training reason": lambda v: v["training.json"][0].pop("reason"),
            "contradictory training reason": lambda v: v["training.json"][0].update(reason="other"),
            "blank training reason": lambda v: v["training.json"][0].update(reason=""),
            "duplicate training": lambda v: v["training.json"].append(v["training.json"][0]),
            "wrong seed": lambda v: v["evaluation.json"][0].update(seed=23),
            "nested identity override": lambda v: v["evaluation.json"][0]["result"].update(seed=23),
            "contradictory status": lambda v: v["evaluation.json"][0]["result"].update(status="complete"),
            "contradictory reason": lambda v: v["evaluation.json"][0]["result"].update(reason="other_failure"),
            "contradictory observations": lambda v: v["events.jsonl"][2].update(observations=1),
            "non-text terminal reason": lambda v: v["events.jsonl"][1].update(reason=True),
            "missing training start": lambda v: v["events.jsonl"].pop(0),
            "missing evaluated replay start": lambda v: v["events.jsonl"].pop(3),
            "event after terminal": lambda v: v["events.jsonl"].append({"id": "ppo/h1/f0/s11", "status": "started"}),
            "missing group": lambda v: v["summary.json"]["groups"].pop(0),
            "duplicate group": lambda v: v["summary.json"]["groups"].append(v["summary.json"]["groups"][0]),
            "wrong summary count": lambda v: v["summary.json"].update(replayPaths=1),
            "false group failures": lambda v: v["summary.json"]["groups"][0].update(failedPaths=0),
            "false group return": lambda v: v["summary.json"]["groups"][1].update(meanTerminalOrStoppedReturn=.5),
            "OPE on failed fit": lambda v: v["ope.json"].append({"id": "ppo/h1/f0/s11", "result": {"status": "invalid"}}),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                _, values = self.export_fixture(root/"archive")
                mutate(values)
                for name, value in values.items():
                    (root/"archive"/name).write_text("".join(json.dumps(e)+"\n" for e in value)
                        if name == "events.jsonl" else json.dumps(value)+"\n")
                index = {name: runner.digest(root/"archive"/name) for name in (*values, "returns.csv")}
                (root/"archive/evidence-index.json").write_text(json.dumps(index)+"\n")
                sha = runner.digest(root/"archive/evidence-index.json")
                with self.assertRaises(ValueError):
                    export(root/"archive", root/"review", rss_unit="bytes",
                           platform_label="fixture", expected_index_sha256=sha)
                self.assertFalse((root/"review").exists())

    def test_export_rejects_missing_evaluated_policy_report_metrics(self):
        for field in ("latencyP99Ms", "oodObservationRate", "completedFitReason"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                sha, values = self.export_fixture(root/"archive", evaluated_rl=True)
                export(root/"archive", root/"control", rss_unit="bytes",
                       platform_label="fixture", expected_index_sha256=sha)
                name = "training.json" if field == "completedFitReason" else "evaluation.json"
                if field == "completedFitReason":
                    values[name][0]["reason"] = "unexpected_failure"
                else:
                    values[name][0]["result"].pop(field)
                (root/"archive"/name).write_text(json.dumps(values[name]))
                index_path = root/"archive/evidence-index.json"
                index = json.loads(index_path.read_text())
                index[name] = runner.digest(root/"archive"/name)
                index_path.write_text(json.dumps(index))
                with self.assertRaises(ValueError):
                    export(root/"archive", root/"rejected", rss_unit="bytes",
                           platform_label="fixture", expected_index_sha256=runner.digest(index_path))
                self.assertFalse((root/"rejected").exists())

    def test_ope_payload_contract(self):
        from sequential_registry import reconcile_ope_payload
        estimate = ope_estimates(np.zeros((2, 6)), np.zeros((2, 6), dtype=int),
            np.full((2, 6), 1/3), np.ones((2, 6)), np.zeros((2, 6)), np.zeros((2, 7)), .99)
        estimate.update(directSimulatorValue=0., valueModel="fixture control variate",
            uncertaintyScope="fixture conditional interval", liveStateActionSupport="unavailable",
            simulatedActionSupport=[1/3, 1/3, 1/3])
        for valid in [estimate, {"status": "invalid", "failedEpisodes": 1, "reason": "fixture"},
                      {"status": "failed", "reason": "MemoryError"}]:
            reconcile_ope_payload(valid)
        for bad in [None, {}, {"status": "complete"}, {"status": "invalid", "reason": "fixture"},
                    {"status": "failed", "reason": ""}, {**estimate, "reliable": True},
                    {**estimate, "effectiveSampleSize": float("nan")},
                    {**estimate, "nonzeroTrajectories": 3},
                    {**estimate, "conditionalBootstrap95": {}}]:
            with self.subTest(payload=bad), self.assertRaises(ValueError):
                reconcile_ope_payload(bad)
        for payload in ({"id": "ppo/h1/f0/s11"}, {"id": "ppo/h1/f0/s11", "result": {}}):
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                _, values = self.export_fixture(root/"archive", evaluated_rl=True)
                (root/"archive/ope.json").write_text(json.dumps([payload]))
                index_path = root/"archive/evidence-index.json"
                index = json.loads(index_path.read_text())
                index["ope.json"] = runner.digest(root/"archive/ope.json")
                index_path.write_text(json.dumps(index))
                with self.assertRaises(ValueError):
                    export(root/"archive", root/"review", rss_unit="bytes", platform_label="fixture",
                           expected_index_sha256=runner.digest(index_path))
                self.assertFalse((root/"review").exists())

    def test_group_reconciliation_matches_hand_calculated_outcomes(self):
        identity = {"algorithm": "cash", "horizon": 1, "seed": 11, "stress": "base"}
        outcomes = [(-.2, .2, .2, None, .01, -.02, "failed"),
                    (.1, 0., -.1, 1., .02, .01, "complete"),
                    (.4, 0., -.4, 3., .03, .04, "complete")]
        records = [{**identity, "result": {"netReturn": r, "maxDrawdown": dd,
                    "expectedShortfall95": es, "sharpe": sr,
                    "costsOverInitialEquity": {"fee": fee}, "fundingPnlOverInitialEquity": funding,
                    "status": status}} for r, dd, es, sr, fee, funding, status in outcomes]
        group = {**identity, "paths": 3, "completePaths": 2, "failedPaths": 1,
                 "failureRate": 1/3, "meanTerminalOrStoppedReturn": .1,
                 "worstTerminalOrStoppedReturn": -.2, "worstDrawdown": .2, "worstES95": .2,
                 "medianPathSharpe": 2., "meanFees": .02, "meanFunding": .01}
        summary = {"trainingFits": 0, "replayPaths": 3, "plannedEntries": 3, "groups": [group]}
        reconcile_groups(summary, records, 0, 3)
        for bad in [None, float("nan"), True, .1001]:
            group["meanTerminalOrStoppedReturn"] = bad
            with self.assertRaises(ValueError):
                reconcile_groups(summary, records, 0, 3)

    def test_export_rejects_destinations_inside_source_archive(self):
        for mode in ("same", "child", "nested", "source_alias", "output_alias", "both_aliases", "dotdot"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as td:
                root = Path(td); source = root/"archive"
                sha, _ = self.export_fixture(source, evaluated_rl=True)
                before = {str(p.relative_to(source)):p.read_bytes() for p in source.rglob("*") if p.is_file()}
                before_paths = {str(p.relative_to(source)) for p in source.rglob("*")}
                (root/"source-alias").symlink_to(source, target_is_directory=True)
                (root/"output-alias").symlink_to(source, target_is_directory=True)
                (root/"outside").mkdir()
                output = {"same":source, "child":source/"review", "nested":source/"new/deep/review",
                    "source_alias":source/"review", "output_alias":root/"output-alias/review",
                    "both_aliases":root/"output-alias/review", "dotdot":root/"outside/../archive/review"}[mode]
                argument = root/"source-alias" if mode in ("source_alias", "both_aliases") else source
                with patch.object(Path, "read_bytes", autospec=True, side_effect=Path.read_bytes) as reads:
                    with self.assertRaisesRegex(ValueError, "outside.*source archive"):
                        export(argument, output, rss_unit="bytes", platform_label="fixture", expected_index_sha256=sha)
                    reads.assert_not_called()
                self.assertEqual({str(p.relative_to(source)) for p in source.rglob("*")}, before_paths)
                self.assertEqual({str(p.relative_to(source)):p.read_bytes() for p in source.rglob("*") if p.is_file()}, before)

    def test_export_pins_admitted_destination_alias(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); source = root/"archive"
            sha, _ = self.export_fixture(source, evaluated_rl=True)
            before = {str(p.relative_to(source)):p.read_bytes() for p in source.rglob("*") if p.is_file()}
            destination = root/"destination"; destination.mkdir()
            alias = root/"destination-alias"; alias.symlink_to(destination, target_is_directory=True)
            original_digest = exporter.digest; changed = []
            def retarget_after_verification(path):
                result = original_digest(path)
                if path.name == "returns.csv":
                    alias.unlink(); alias.symlink_to(source, target_is_directory=True); changed.append(True)
                return result
            with patch.object(exporter, "digest", side_effect=retarget_after_verification):
                export(source, alias/"review", rss_unit="bytes", platform_label="fixture", expected_index_sha256=sha)
            self.assertEqual(changed, [True])
            self.assertEqual(len(list((destination/"review").iterdir())), 7)
            self.assertFalse((source/"review").exists())
            self.assertEqual({str(p.relative_to(source)):p.read_bytes() for p in source.rglob("*") if p.is_file()}, before)

    def test_export_isolated_destinations_preserve_reports_and_existing_files(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); source = root/"archive"
            sha, _ = self.export_fixture(source, evaluated_rl=True)
            before = {str(p.relative_to(source)):p.read_bytes() for p in source.rglob("*") if p.is_file()}
            (root/"source-alias").symlink_to(source, target_is_directory=True)
            (root/"destination").mkdir()
            (root/"destination-alias").symlink_to(root/"destination", target_is_directory=True)
            paths = [root/"archive-review", source/"../normalized-review", root/"destination-alias/review"]
            outputs = []
            for output in paths:
                export(root/"source-alias", output, rss_unit="bytes", platform_label="fixture", expected_index_sha256=sha)
                reports = {p.name:p.read_bytes() for p in output.iterdir()}
                self.assertEqual(len(reports), 7)
                outputs.append(reports)
                with self.assertRaises(FileExistsError):
                    export(source, output, rss_unit="bytes", platform_label="fixture", expected_index_sha256=sha)
                self.assertEqual({p.name:p.read_bytes() for p in output.iterdir()}, reports)
            self.assertTrue(all(value == outputs[0] for value in outputs))
            self.assertEqual({str(p.relative_to(source)):p.read_bytes() for p in source.rglob("*") if p.is_file()}, before)

    def test_export_parses_verified_snapshots_after_archive_replacement(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sha, values = self.export_fixture(root/"archive")
            kwargs = dict(rss_unit="bytes", platform_label="fixture", expected_index_sha256=sha)
            export(root/"archive", root/"control", **kwargs)
            original_digest = exporter.digest
            replaced = []
            def verify_then_replace(path):
                result = original_digest(path)
                if path.name == "returns.csv":
                    for name in values:
                        (root/"archive"/name).write_text("unverified replacement")
                    replaced.append(True)
                return result
            with patch.object(exporter, "digest", side_effect=verify_then_replace):
                export(root/"archive", root/"review", **kwargs)
            self.assertEqual(replaced, [True])
            for path in (root/"control").iterdir():
                self.assertEqual(path.read_bytes(), (root/"review"/path.name).read_bytes())
            with self.assertRaises(ValueError):
                export(root/"archive", root/"rejected", **kwargs)
            self.assertFalse((root/"rejected").exists())

    def test_export_retains_verified_index_identity_after_replacement(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sha, _ = self.export_fixture(root/"archive")
            original_loads = json.loads
            replaced = []
            def parse_then_replace(raw, *args, **kwargs):
                value = original_loads(raw, *args, **kwargs)
                if not replaced:
                    (root/"archive/evidence-index.json").write_text("unverified index replacement")
                    replaced.append(True)
                return value
            with patch.object(exporter.json, "loads", side_effect=parse_then_replace):
                export(root/"archive", root/"review", rss_unit="bytes",
                       platform_label="fixture", expected_index_sha256=sha)
            manifest = json.loads((root/"review/experiment-manifest.json").read_text())
            self.assertEqual(replaced, [True])
            self.assertEqual(manifest["externalEvidenceIndexSha256"], sha)
            self.assertNotEqual(runner.digest(root/"archive/evidence-index.json"), sha)

    def test_export_rejects_tampered_streamed_returns_before_output(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sha, _ = self.export_fixture(root/"archive")
            (root/"archive/returns.csv").write_text("unverified returns")
            with self.assertRaisesRegex(ValueError, "hash/path mismatch"):
                export(root/"archive", root/"review", rss_unit="bytes",
                       platform_label="fixture", expected_index_sha256=sha)
            self.assertFalse((root/"review").exists())

    def test_prepared_reports_use_utf8_bytes_without_newline_translation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sha, _ = self.export_fixture(root/"archive")
            with patch.object(Path, "write_text", side_effect=AssertionError("text-mode newline translation")):
                export(root/"archive", root/"review", rss_unit="bytes", platform_label="fixture-caf\u00e9",
                       expected_index_sha256=sha)
            files = list((root/"review").iterdir())
            self.assertEqual(len(files), 7)
            for path in files:
                raw = path.read_bytes()
                self.assertNotIn(b"\r\n", raw)
                self.assertTrue(raw.endswith(b"\n"))
                self.assertEqual(raw.decode("utf-8").encode("utf-8"), raw)
            self.assertEqual(json.loads((root/"review/evaluation-summary.json").read_bytes())["platform"], "fixture-caf\u00e9")
            with patch.object(exporter, "render_reports", return_value={"invalid.csv": "\ud800"}):
                with self.assertRaises(UnicodeEncodeError):
                    export(root/"archive", root/"bad-encoding", rss_unit="bytes", platform_label="fixture",
                           expected_index_sha256=sha)
            self.assertFalse((root/"bad-encoding").exists())

    def test_run_provenance_keeps_admitted_hashes_after_input_replacement(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            panel, settlements = root/"panel.csv", root/"funding.csv"
            with panel.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["symbol", "openTime", "closeTime", "close"])
                writer.writerows(["x", i*1000, i*1000+999, p] for i, p in enumerate(self.p))
            settlements.write_text("symbol,fundingTime,fundingRate,resolvedMarkPrice\n"
                                   "x,1999,0.001,100\n")
            registration = json.loads(REGISTRATION.read_text())
            registration["data"].update(panelSha256=runner.digest(panel),
                settlementsSha256=runner.digest(settlements), symbols=["x"],
                startOpenTime=0, endOpenTime=255000, intervalMilliseconds=1000,
                rowsPerSymbol=256, decisionHorizonBars=[1])
            registration["seeds"] = [11]
            registration["validation"]["outerFolds"] = [{"trainStop":160,"testStart":166,"testStop":240}]
            reg = root/"registration.json"; reg.write_text(json.dumps(registration))
            def admit_then_replace(*args):
                data = load_development(*args)
                panel.write_text("replaced after verified admission")
                settlements.write_text("replaced after verified admission")
                return data
            with patch.object(runner, "REGISTRATION", reg), \
                 patch.object(runner, "source_commit", return_value="a"*40), \
                 patch.object(runner, "load_development", side_effect=admit_then_replace), \
                 patch.object(runner, "ALGORITHMS", ("ppo",)), \
                 patch.object(runner, "Baselines") as controls, \
                 patch.object(runner, "train_ppo", return_value=(Network(11), {})), \
                 patch.object(runner, "short_ope", return_value={"status":"invalid","failedEpisodes":1,"reason":"fixture"}):
                controls.names = ()
                runner.run(panel, settlements, root/"run")
            manifest = json.loads((root/"run/manifest.json").read_text())
            provenance = json.loads((root/"run/policies/ppo_h1_f0_s11.json").read_text())["provenance"]
            for source_key, policy_key, path in [("panelSha256", "dataSha256", panel),
                                                ("settlementsSha256", "fundingSha256", settlements)]:
                expected = registration["data"][source_key]
                self.assertEqual(manifest[source_key], expected)
                self.assertEqual(provenance[policy_key], expected)
                self.assertNotEqual(runner.digest(path), expected)
            self.assertFalse(manifest["holdoutOpened"])
            self.assertFalse(manifest["promotionAllowed"])
            index_path = root/"run/evidence-index.json"
            export(root/"run", root/"review", rss_unit="bytes", platform_label="fixture",
                   expected_index_sha256=runner.digest(index_path))
            (root/"run/ope.json").write_text("[]")
            index = json.loads(index_path.read_text())
            index["ope.json"] = runner.digest(root/"run/ope.json")
            index_path.write_text(json.dumps(index))
            with self.assertRaisesRegex(ValueError, "OPE rows differ"):
                export(root/"run", root/"missing-ope", rss_unit="bytes", platform_label="fixture",
                       expected_index_sha256=runner.digest(index_path))
            self.assertFalse((root/"missing-ope").exists())

    def test_exception_failures_remain_exportable_without_losing_trials(self):
        for error in (MemoryError(), RuntimeError(" "), RuntimeError("complete")):
            for training_failure in (False, True):
                with self.subTest(error=repr(error), training_failure=training_failure), tempfile.TemporaryDirectory() as td:
                    root = Path(td)
                    registration = json.loads(REGISTRATION.read_text())
                    registration["data"].update(symbols=["x"], decisionHorizonBars=[1])
                    registration["seeds"] = [11]
                    registration["validation"]["outerFolds"] = [{"trainStop":160,"testStart":166,"testStop":240}]
                    reg = root/"registration.json"
                    reg.write_text(json.dumps(registration))
                    with patch.object(runner, "REGISTRATION", reg), \
                         patch.object(runner, "source_commit", return_value="a"*40), \
                         patch.object(runner, "load_development", return_value=({"x": self.p}, {"x": self.f}, np.arange(256)*1000)), \
                         patch.object(runner, "ALGORITHMS", ("ppo",)), \
                         patch.object(runner, "Baselines") as controls, \
                         patch.object(runner, "train_ppo", side_effect=error if training_failure else None,
                                      return_value=(Network(11), {})), \
                         patch.object(runner, "short_ope", side_effect=error), \
                         patch.object(runner, "replay_policy", side_effect=error):
                        controls.names = ()
                        runner.run(root/"unused-panel", root/"unused-funding", root/"run")
                    export(root/"run", root/"review", rss_unit="bytes", platform_label="fixture",
                           expected_index_sha256=runner.digest(root/"run/evidence-index.json"))
                    report = json.loads((root/"review/evaluation-summary.json").read_text())
                    self.assertEqual(report["replayPaths"], len(runner.STRESSES))
                    self.assertNotIn("complete", report["totalFailuresByReason"])
                    self.assertEqual(sum(report["totalFailuresByReason"].values()), len(runner.STRESSES))
                    self.assertFalse(report["promotionAllowed"])
                    for name in ("training.json", "ope.json"):
                        for row in json.loads((root/"run"/name).read_text()):
                            outcome = row.get("result", row)
                            if outcome.get("status") == "failed":
                                self.assertTrue(outcome["reason"].strip())
                                self.assertNotEqual(outcome["reason"], "complete")

    @contextmanager
    def publication_fixture(self, root):
        registration = json.loads(REGISTRATION.read_text())
        registration["data"].update(symbols=["x"], decisionHorizonBars=[1])
        registration["seeds"] = [11]
        registration["validation"]["outerFolds"] = [{"trainStop":160,"testStart":166,"testStop":240}]
        reg = root/"registration.json"; reg.write_text(json.dumps(registration))
        with patch.object(runner, "REGISTRATION", reg), \
             patch.object(runner, "source_commit", return_value="a"*40), \
             patch.object(runner, "load_development", return_value=({"x":self.p}, {"x":self.f}, np.arange(256)*1000)), \
             patch.object(runner, "ALGORITHMS", ("ppo",)), \
             patch.object(runner, "Baselines") as controls, \
             patch.object(runner, "train_ppo", return_value=(Network(11), {})), \
             patch.object(runner, "infer", return_value=(0., 0.)), \
             patch.object(runner, "short_ope", return_value={"status":"invalid", "failedEpisodes":1, "reason":"fixture"}):
            controls.names = ()
            yield

    def test_terminal_ledger_write_failure_aborts_without_reclassification(self):
        for target in ("training", "replay"):
            for operation in ("write", "flush"):
                with self.subTest(target=target, operation=operation), tempfile.TemporaryDirectory() as td:
                    root = Path(td); attempts = []; injected = []
                    original_open = Path.open
                    class Ledger:
                        def __init__(self, stream):
                            self.stream, self.fail_flush = stream, False
                        def __enter__(self): return self
                        def __exit__(self, *args): return self.stream.__exit__(*args)
                        def write(self, text):
                            event = json.loads(text)
                            if event["status"] in ("complete", "failed"):
                                attempts.append(event)
                                training = event["id"].count("/") == 3
                                if not injected and training == (target == "training"):
                                    injected.append(True)
                                    if operation == "write": raise OSError("fixture ledger write")
                                    self.fail_flush = True
                            return self.stream.write(text)
                        def flush(self):
                            if self.fail_flush:
                                self.fail_flush = False
                                raise OSError("fixture ledger flush")
                            return self.stream.flush()
                    def open_stream(path, *args, **kwargs):
                        stream = original_open(path, *args, **kwargs)
                        return Ledger(stream) if path == root/"run/events.jsonl" and args == ("x",) else stream
                    with self.publication_fixture(root), patch.object(Path, "open", open_stream), \
                         self.assertRaisesRegex(OSError, "fixture ledger"):
                        runner.run(root/"unused-panel", root/"unused-funding", root/"run")
                    self.assertEqual(len(injected), 1)
                    self.assertEqual(len(attempts), 1 if target == "training" else 2)
                    self.assertTrue(all(e["status"] == "complete" for e in attempts))
                    self.assertFalse((root/"run/evidence-index.json").exists())
                    self.assertFalse((root/"run/summary.json").exists())

    def test_return_path_write_failure_aborts_before_replay_terminal(self):
        for fail_row in (1, 2, 20):
            with self.subTest(fail_row=fail_row), tempfile.TemporaryDirectory() as td:
                root = Path(td); writes = []; original_writer = csv.writer
                class ReturnsWriter:
                    def __init__(self, stream, *args, **kwargs):
                        self.writer = original_writer(stream, *args, **kwargs)
                    def writerow(self, row):
                        writes.append(row)
                        if len(writes) == fail_row+1: raise OSError("fixture return write")
                        return self.writer.writerow(row)
                with self.publication_fixture(root), patch.object(runner.csv, "writer", ReturnsWriter), \
                     self.assertRaisesRegex(OSError, "fixture return write"):
                    runner.run(root/"unused-panel", root/"unused-funding", root/"run")
                self.assertEqual(len(writes), fail_row+1)
                events = [json.loads(line) for line in (root/"run/events.jsonl").read_text().splitlines()]
                self.assertEqual([e["status"] for e in events], ["started", "complete", "started"])
                self.assertFalse((root/"run/evidence-index.json").exists())
                self.assertFalse((root/"run/summary.json").exists())
                with (root/"run/returns.csv").open() as stream:
                    self.assertEqual(len(list(csv.DictReader(stream))), fail_row-1)

    def test_successful_publication_has_one_terminal_per_trial(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with self.publication_fixture(root):
                runner.run(root/"unused-panel", root/"unused-funding", root/"run")
            events = [json.loads(line) for line in (root/"run/events.jsonl").read_text().splitlines()]
            planned = json.loads((root/"run/planned-registry.json").read_text())
            terminals = [e for e in events if e["status"] in ("complete", "failed")]
            self.assertEqual(len(terminals), len(planned))
            self.assertEqual({e["id"] for e in terminals}, {e["id"] for e in planned})
            self.assertTrue(all(e["status"] == "complete" for e in terminals))
            export(root/"run", root/"review", rss_unit="bytes", platform_label="fixture",
                   expected_index_sha256=runner.digest(root/"run/evidence-index.json"))
            summary = json.loads((root/"review/evaluation-summary.json").read_text())
            self.assertEqual(summary["replayPaths"], len(runner.STRESSES))
            self.assertEqual(summary["totalFailuresByReason"], {"complete":len(runner.STRESSES)})
            self.assertFalse(summary["promotionAllowed"])
            self.assertEqual(len(list((root/"review").iterdir())), 7)

    def test_registered_separation_and_no_final_holdout(self):
        r = json.loads(REGISTRATION.read_text())
        for split in r["validation"]["outerFolds"]:
            self.assertGreaterEqual(split["testStart"]-split["trainStop"], 6)
            self.assertLess(split["testStop"], r["data"]["rowsPerSymbol"])
        self.assertFalse(r["promotionGates"]["screenPromotionPermitted"])
        self.assertEqual(r["seeds"], [11,23,47])

    def test_committed_evidence_retains_all_trials_and_seeds(self):
        root = Path(__file__).resolve().parents[1]
        evidence = root / "research-notes/sequential-control-2026-09-17"
        with (evidence/"experiment-registry.csv").open() as stream:
            registry = list(csv.DictReader(stream))
        self.assertEqual(len(registry),19548)
        self.assertEqual(len({r["id"] for r in registry}),19548)
        self.assertTrue(all(r["status"] in ("complete","failed") for r in registry))
        fits=[r for r in registry if r["kind"]=="training"]
        self.assertEqual(len(fits),108)
        for algorithm in ["ppo","double_dqn","cql","cql_no_inventory_penalty"]:
            for horizon in [1,3,6]:
                for fold in range(3):
                    for seed in [11,23,47]:
                        self.assertIn(f"{algorithm}/h{horizon}/f{fold}/s{seed}",{r["id"] for r in fits})
        result=json.loads((evidence/"evaluation-summary.json").read_text())
        self.assertFalse(result["holdoutOpened"])
        self.assertFalse(result["promotionAllowed"])
        self.assertEqual(result["rlStressFailures"]["base"],{"failed":686,"total":1080})

    def test_haskell_proposal_boundary_has_no_production_caller(self):
        app=Path(__file__).resolve().parents[1]/"haskell/app"
        for p in app.rglob("*.hs"):
            if p.name != "PolicyProposalV1.hs":
                self.assertNotIn("import Trader.Research.PolicyProposalV1",p.read_text(),str(p))


if __name__ == "__main__":
    unittest.main()
