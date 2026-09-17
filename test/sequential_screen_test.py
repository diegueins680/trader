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
        result = ope_estimates(r, r.astype(int), probs, probs, np.zeros_like(r), np.zeros((2,3)), 1)
        for k in ["ordinaryIS", "weightedIS", "perDecisionIS", "doublyRobust"]:
            self.assertEqual(result[k], 5)
        self.assertEqual(result["effectiveSampleSize"], 2)
        unsupported = ope_estimates(r, r.astype(int), probs, np.zeros_like(r), np.zeros_like(r), np.zeros((2,3)), 1)
        self.assertIsNone(unsupported["weightedIS"])
        self.assertEqual(unsupported["effectiveSampleSize"], 0)
        self.assertFalse(unsupported["reliable"])

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
