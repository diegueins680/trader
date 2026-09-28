"""Regression tests for fail-closed certificate admission and model mutations."""
import ast
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import lifecycle
import causal_footprint
import numpy as np
import gap_risk
import gap_conformance
import z3
from verify import ROOT, read_json, validate_ledger


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.ledger = read_json(ROOT / 'formal/research/proof-ledger.json')

    def test_valid_ledger_keeps_critical_blockers(self):
        self.assertGreater(validate_ledger(self.ledger, ROOT), 0)

    def test_mutated_ledgers_fail(self):
        changes = [lambda x: x['entries'].append(copy.deepcopy(x['entries'][0])),
                   lambda x: x['entries'][0].update(status='fully_verified'),
                   lambda x: x['entries'][0].update(formalStatement='a different theorem'),
                   lambda x: x['entries'][11].update(relatedRequirements=[]),
                   lambda x: x['criticalFiles'].remove('scripts/formal/proofs.py'),
                   lambda x: x['entries'][9].update(status='proved'),
                   lambda x: x['missionObligations'][0].update(status='proved'),
                   lambda x: x['entries'][0].update(implementationFiles=[]),
                   lambda x: x['entries'][0].update(assumptions=['unknown']),
                   lambda x: x['entries'][0].update(artifact='missing-result.json'),
                   lambda x: x['missionObligations'].pop(),
                   lambda x: x['criticalFiles'].append('unmapped-critical.py')]
        for change in changes:
            with self.subTest(change=change):
                data = copy.deepcopy(self.ledger)
                change(data)
                with self.assertRaises(ValueError):
                    validate_ledger(data, ROOT)

    def test_duplicate_and_nonfinite_json_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.json'
            for source in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}', '{"a":1e999}'):
                path.write_text(source)
                with self.assertRaises(ValueError):
                    read_json(path)

    def test_live_authority_mutation_detected(self):
        with patch.object(lifecycle, 'authority', lambda state: state[0]):
            with self.assertRaisesRegex(RuntimeError, 'unsafe state'):
                lifecycle.check_model()

    def test_start_during_drain_mutation_detected(self):
        original = lifecycle.successors
        def mutated(state):
            yield from original(state)
            if state[1]:
                yield 'start0:1', (False, True, 1, state[3])
        with patch.object(lifecycle, 'successors', mutated):
            with self.assertRaisesRegex(RuntimeError, 'start while draining'):
                lifecycle.check_model()

    def test_disable_does_not_revoke_immutable_proposal(self):
        state = lifecycle.INITIAL
        for event in read_json(ROOT / 'formal/research/counterexamples.json')['entries'][0]['trace']:
            state = dict(lifecycle.successors(state))[event]
        self.assertFalse(state[0])
        self.assertIn(3, state[2:])
        self.assertFalse(lifecycle.authority(state))

    def test_gap_witnesses_replay_and_accounting_grid(self):
        document = read_json(ROOT / 'formal/research/gap-counterexamples.json')
        self.assertEqual(len(gap_risk.check_counterexamples(document)), 2)
        self.assertEqual(gap_conformance.check_replay(document)['boundedAccountingTraces'], 180)

    def test_invalid_gap_witness_fails(self):
        document = read_json(ROOT / 'formal/research/gap-counterexamples.json')
        document['entries'][0]['endingEquity'] = '1'
        with self.assertRaisesRegex(ValueError, 'invalid loss-floor witness'):
            gap_risk.check_counterexamples(document)

    def test_gap_smt_cannot_accept_false_or_vacuous_claims(self):
        for premise, conclusion in ((z3.BoolVal(True), z3.BoolVal(False)),
                                    (z3.BoolVal(False), z3.BoolVal(True))):
            with self.subTest(premise=premise), patch.object(
                    gap_risk, 'obligations', lambda: {'mutant': (premise, conclusion)}):
                with self.assertRaises(RuntimeError):
                    gap_risk.check_obligations()

    def test_gap_conformance_detects_dropped_costs(self):
        original = gap_conformance.Replay._trade
        def mutated(replay, target, terminal=False):
            before = replay.equity
            result = original(replay, target, terminal)
            replay.equity = before
            return result
        with patch.object(gap_conformance.Replay, '_trade', mutated):
            with self.assertRaisesRegex(RuntimeError, 'counterexample mismatch'):
                gap_conformance.check_replay(read_json(ROOT / 'formal/research/gap-counterexamples.json'))


class SourceCausalityTests(unittest.TestCase):
    def setUp(self):
        self.source = (ROOT / causal_footprint.SOURCE).read_text()

    def test_actual_source_and_safe_formula_variant(self):
        for source in (self.source, self.source.replace('np.std(r[-6:])', 'np.std(r[-3:])')):
            result = causal_footprint.check_source(source)
            self.assertEqual(result['reads'], [{'lower': 't - 24', 'upperExclusive': 't + 1', 'result': 'unsat'}])

    def test_future_and_wrong_window_slices_fail_smt(self):
        for replacement in ('prices[t - 24:t + 2]', 'prices[0:t + 1]', 'prices[t - 25:t + 1]'):
            with self.subTest(replacement=replacement), self.assertRaisesRegex(RuntimeError, 'unsafe source read'):
                causal_footprint.check_source(self.source.replace('prices[t - 24:t + 1]', replacement))

    def test_unbounded_and_hidden_inputs_fail_static_analysis(self):
        mutants = [('prices[t - 24:t + 1]', 'prices'),
                   ('prices[t - 24:t + 1]', 'prices[:][-25:]'),
                   ('np.std(r)])', 'np.std(prices)])'),
                   ('np.std(r)])', 'np.std(other_prices)])'),
                   ('np.std(r)])', 'unknown_function(r)])'),
                   ('r = p[1:] / p[:-1] - 1', 't = t + 1\n    r = p[1:] / p[:-1] - 1'),
                   ('r = p[1:] / p[:-1] - 1', 'import os\n    r = p[1:] / p[:-1] - 1')]
        for old, new in mutants:
            self.assertIn(old, self.source)
            with self.subTest(new=new), self.assertRaises(ValueError):
                causal_footprint.check_source(self.source.replace(old, new))

    def test_admission_and_helper_drift_fail(self):
        for old, new in [('24 <= t < len(prices)', '24 <= t <= len(prices)'),
                         ('value.ndim == 1', 'np.isfinite(value).all()'),
                         ('def market_features(', '@unknown_decorator\ndef market_features(')]:
            with self.subTest(new=new), self.assertRaises(ValueError):
                causal_footprint.check_source(self.source.replace(old, new))

    def test_preserved_next_bar_mutant_counterexample(self):
        import sequential_env
        fixture = read_json(ROOT / 'formal/research/causal-counterexamples.json')['entries'][0]
        mutant = self.source.replace(fixture['replace'], fixture['with'])
        with self.assertRaisesRegex(RuntimeError, 'unsafe source read'):
            causal_footprint.check_source(mutant)
        # Validate the prescribed integer witness, not a solver-dependent model.
        reads = causal_footprint.analyze(mutant)
        t = z3.IntVal(fixture['decisionIndex'])
        lo, hi = (causal_footprint.integer(node, t) for node in reads[0])
        forbidden = fixture['firstForbiddenIndex']
        solver = z3.Solver()
        solver.set(timeout=10000)
        solver.add(t >= 24, t < fixture['length'], forbidden > t,
                   lo <= forbidden, forbidden < hi, hi <= fixture['length'])
        self.assertEqual(solver.check(), z3.sat)
        # Compile only the statically admitted feature AST in a private namespace.
        function = next(n for n in ast.parse(mutant).body
                        if isinstance(n, ast.FunctionDef) and n.name == 'market_features')
        namespace = {'np': np, '_real_series': sequential_env._real_series,
                     '_integer': sequential_env._integer}
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<causal-mutant>', 'exec'), namespace)
        prices = np.full(fixture['length'], fixture['prefixPrice'])
        prices[forbidden] = fixture['originalFuturePrice']
        changed = prices.copy()
        changed[forbidden] = fixture['changedFuturePrice']
        decision = fixture['decisionIndex']
        np.testing.assert_array_equal(sequential_env.market_features(prices, decision),
                                      sequential_env.market_features(changed, decision))
        self.assertFalse(np.array_equal(namespace['market_features'](prices, decision),
                                        namespace['market_features'](changed, decision)))

    def test_future_value_property_for_actual_features_and_observations(self):
        # The finite seeded checks connect the static dependency certificate to
        # actual NumPy execution. They are property tests, not a universal proof.
        from sequential_env import market_features, Scale, Replay
        rng = np.random.default_rng(20260928)
        for _ in range(32):
            prices = 100 * np.exp(np.cumsum(rng.normal(0, .002, 128)))
            t = int(rng.integers(24, 126))
            scale = Scale.fit([prices[:t + 1]])
            expected = market_features(prices, t)
            baseline = Replay(prices, np.zeros(128), t, 128, 1, scale).observation()
            for future in (float('nan'), float('inf'), -1., 1e200):
                changed = prices.copy()
                changed[t + 1:] = future
                np.testing.assert_array_equal(expected, market_features(changed, t))
                actual = Replay(changed, np.zeros(128), t, 128, 1, scale).observation()
                np.testing.assert_array_equal(baseline, actual)


if __name__ == '__main__':
    unittest.main()
