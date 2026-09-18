"""Deterministic engineering fixtures; no protected market data or network."""
import hashlib
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
from sequential_evaluation import economic, ope_estimates
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

    def test_multiple_training_seeds_are_reproducible(self):
        for seed in [11, 23, 47]:
            for alg in ["ppo", "double_dqn", "cql"]:
                def fit():
                    args = ({"x": self.p[:160]}, {"x": self.f[:160]}, self.scale, 3, seed)
                    return train_ppo(*args, steps=64)[0] if alg == "ppo" else train_q(*args, offline=alg == "cql", steps=64)[0]
                a, b = fit(), fit()
                for k in a.p:
                    np.testing.assert_array_equal(a.p[k], b.p[k])

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
