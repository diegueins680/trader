"""Regression tests for fail-closed certificate admission and model mutations."""
import ast
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import lifecycle
import causal_footprint
import training_prefix
import artifact_admission
import transition_admission
import terminal_numerics
import target_v2
import ppo_objective
import value_objective
import optimizer_publication
import inference_boundary
import ope_algebra
import ess_v2
import funding_boundary
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


class TrainingPrefixTests(unittest.TestCase):
    def setUp(self):
        self.environment = (ROOT / training_prefix.ENV).read_text()
        self.runner = (ROOT / training_prefix.RUNNER).read_text()
        self.registration = read_json(ROOT / training_prefix.REGISTRATION)

    def check(self, environment=None, runner=None, registration=None):
        return training_prefix.check_training(
            self.registration if registration is None else registration,
            self.environment if environment is None else environment,
            self.runner if runner is None else runner)

    def setup_from_source(self, prices, funding, stop, runner=None):
        from sequential_env import Scale
        source = self.runner if runner is None else runner
        _, statements = training_prefix.extract_runner(source)
        namespace = dict(prices=prices, funding=funding, split={'trainStop': stop}, Scale=Scale)
        exec(compile(ast.Module(body=statements, type_ignores=[]), '<source-fold-setup>', 'exec'), namespace)
        return namespace

    def test_actual_source_and_registered_cases(self):
        result = self.check()
        self.assertEqual(result['solverQueries'], 6)
        self.assertEqual(result['foldHorizonCases'], 9)
        self.assertEqual([f['gapBars'] for f in result['registeredFolds']], [6, 6, 6])

    def test_source_bound_mutants_fail(self):
        for old, new, location in [
            ('p[:split["trainStop"]]', 'p[:split["trainStop"] + 1]', 'runner'),
            ('p[:split["trainStop"]]', 'p[:split["testStop"]]', 'runner'),
            ('Scale.fit(list(train.values()))', 'Scale.fit(list(prices.values()))', 'runner'),
            ('s: p[:split["trainStop"]]', '"OTHER": p[:split["trainStop"]]', 'runner'),
            ('range(24, len(p))', 'range(24, len(p) + 1)', 'environment'),
            ('x.mean(0)', 'outside.mean(0)', 'environment'),
            ('start, start + 97, horizon', 'start, start + 98, horizon', 'environment'),
            ('len(p) - 96', 'len(p) - 95', 'environment'),
            ('len(p) <= 120', 'len(p) <= 119', 'environment')]:
            original = getattr(self, location)
            self.assertIn(old, original)
            with self.subTest(new=new), self.assertRaises((RuntimeError, ValueError)):
                self.check(**{location: original.replace(old, new)})

    def test_training_smt_rejects_false_and_vacuous_claims(self):
        for premise, conclusion in ((z3.BoolVal(True), z3.BoolVal(False)),
                                    (z3.BoolVal(False), z3.BoolVal(True))):
            with self.assertRaises(RuntimeError):
                training_prefix.prove('mutant', premise, conclusion)

    def test_registered_domain_mutants_fail(self):
        changes = [lambda r: r['validation']['outerFolds'][0].update(testStart=1605),
                   lambda r: r['validation']['outerFolds'][0].update(trainStop=True),
                   lambda r: r['validation']['outerFolds'][0].update(testStop=5000),
                   lambda r: r['data'].update(decisionHorizonBars=[True, 3, 6]),
                   lambda r: r['validation'].update(embargoBars=0),
                   lambda r: r['validation']['outerFolds'].pop()]
        for change in changes:
            altered = copy.deepcopy(self.registration)
            change(altered)
            with self.assertRaises(ValueError):
                self.check(registration=altered)

    def test_preserved_training_counterexamples(self):
        import sequential_env
        from sequential_env import Replay, Scale
        for fixture in read_json(ROOT / 'formal/research/training-counterexamples.json')['entries']:
            location = 'runner' if fixture['id'] == 'CE-RL-005' else 'environment'
            original = getattr(self, location)
            mutant = original.replace(fixture['replace'], fixture['with'], fixture['replacementCount'])
            with self.subTest(id=fixture['id']), self.assertRaisesRegex(RuntimeError, 'violating source bound'):
                self.check(**{location: mutant})
            if location == 'runner':
                solver = z3.Solver(); solver.set(timeout=10000)
                T = fixture['trainStop']; forbidden = fixture['firstForbiddenIndex']
                solver.add(z3.BoolVal(25 <= T < fixture['sourceLength']),
                           z3.BoolVal(T <= forbidden < T+1))
                self.assertEqual(solver.check(), z3.sat)
                prices = np.full(fixture['sourceLength'], fixture['prefixPrice'])
                changed = prices.copy()
                changed[fixture['firstForbiddenIndex']] = fixture['changedFuturePrice']
                args = ({'X': np.zeros_like(prices)}, fixture['trainStop'])
                before = self.setup_from_source({'X': prices}, *args, runner=mutant)['scale']
                after = self.setup_from_source({'X': changed}, *args, runner=mutant)['scale']
                self.assertFalse(np.array_equal(before.mean, after.mean))
                correct = self.setup_from_source({'X': changed}, *args)['scale']
                np.testing.assert_array_equal(correct.mean, np.zeros(6))
            else:
                p = np.full(fixture['prefixLength'], 100.)
                with self.assertRaisesRegex(ValueError, 'invalid episode boundaries'):
                    Replay(p, np.zeros_like(p), fixture['sampledStart'], fixture['mutantStop'], 1, Scale.fit([p]))
                function = next(n for n in ast.parse(mutant).body
                                if isinstance(n, ast.FunctionDef) and n.name == 'collect')
                namespace = dict(vars(sequential_env))
                exec(compile(ast.Module(body=[function], type_ignores=[]), '<episode-mutant>', 'exec'), namespace)
                with self.assertRaisesRegex(ValueError, 'invalid episode boundaries'):
                    namespace['collect']({'X': p}, {'X': np.zeros_like(p)}, Scale.fit([p]), 1, 11, 1)
                solver = z3.Solver(); solver.set(timeout=10000)
                start, length = fixture['sampledStart'], fixture['prefixLength']
                solver.add(z3.BoolVal(24 <= start < length-96), z3.BoolVal(start+98 > length))
                self.assertEqual(solver.check(), z3.sat)

    def test_source_setup_normalization_future_corruption(self):
        rng = np.random.default_rng(20260929)
        for stop in (121, 160, 200):
            for _ in range(4):
                prices = {s: 100*np.exp(np.cumsum(rng.normal(0, .002, 256))) for s in ('X', 'Y')}
                funding = {s: np.zeros(256) for s in prices}
                baseline = self.setup_from_source(prices, funding, stop)
                for future in (float('nan'), float('inf'), -1., 1e200):
                    changed = {s: p.copy() for s, p in prices.items()}
                    for p in changed.values():
                        p[stop:] = future
                    actual = self.setup_from_source(changed, funding, stop)
                    self.assertEqual(set(actual['train']), set(prices))
                    self.assertTrue(all(len(p) == stop for p in actual['train'].values()))
                    for field in ('mean', 'std', 'low', 'high'):
                        np.testing.assert_array_equal(getattr(baseline['scale'], field), getattr(actual['scale'], field))

    def test_collection_future_corruption_and_actual_episode_bounds(self):
        import sequential_env as env
        replay = env.Replay
        feature_function = env.market_features
        for stop in (121, 160, 200):
            prices = 100*np.exp(np.sin(np.arange(256)/9)*.001)
            funding = np.zeros(256)
            original = self.setup_from_source({'X': prices}, {'X': funding}, stop)
            changed = prices.copy(); changed[stop:] = np.nan
            changed_funding = funding.copy(); changed_funding[stop:] = np.inf
            altered = self.setup_from_source({'X': changed}, {'X': changed_funding}, stop)
            for horizon in (1, 3, 6):
                for seed in (11, 23, 47):
                    episodes, reads = [], []
                    def record_features(p, t):
                        self.assertLessEqual(len(p), stop)
                        self.assertGreaterEqual(t - 24, 0)
                        self.assertLess(t, stop)
                        reads.append((t - 24, t))
                        return feature_function(p, t)
                    def record(*args, **kwargs):
                        episode = replay(*args, **kwargs)
                        self.assertLessEqual(episode.stop, stop)
                        self.assertEqual(len(episode.prices), stop)
                        episodes.append(episode)
                        return episode
                    with patch.object(env, 'Replay', record), patch.object(env, 'market_features', record_features):
                        a = env.collect(original['train'], original['funds'], original['scale'], horizon, seed, 12)
                        b = env.collect(altered['train'], altered['funds'], altered['scale'], horizon, seed, 12)
                    for key in ('s', 'a', 'r', 'next', 'done', 'prob'):
                        np.testing.assert_array_equal(a[key], b[key])
                    self.assertTrue(episodes)
                    self.assertTrue(reads)
                    self.assertTrue(all(e.t < stop for e in episodes))


class ArtifactAdmissionTests(unittest.TestCase):
    def setUp(self):
        import sequential_learning as learning
        self.learning = learning
        self.source = (ROOT / artifact_admission.SOURCE).read_text()
        self.meta = {'codeCommit': 'a'*40, 'registrationSha256': 'b'*64,
                     'dataSha256': 'c'*64, 'seed': 11, 'horizon': 1,
                     'algorithm': 'ppo', 'fold': 0}

    def test_source_predicates_and_model(self):
        result = artifact_admission.check_artifact(self.source)
        self.assertEqual(result['model']['states'], 27)
        self.assertEqual(result['model']['transitions'], 40)
        self.assertEqual(result['model']['terminationGateBound'], 13)
        self.assertEqual(result['metadata']['queries'], 2)

    def test_source_bypasses_and_rebinding_rejected(self):
        mutants = [('return net', 'return Network(0)'),
                   ('validate_provenance(expected_provenance)', 'pass'),
                   ('snapshots = _parameter_snapshots(parameters)', 'snapshots = parameters'),
                   ('raw = stream.read(65537)', 'raw = stream.read()'),
                   ('net = Network(0)', 'return Network(0)'),
                   ('a["enabled"] is not False', 'a["enabled"] != False')]
        for old, new in mutants:
            self.assertIn(old, self.source)
            with self.subTest(new=new), self.assertRaises(ValueError):
                artifact_admission.check_artifact(self.source.replace(old, new))

    def test_predicates_reject_false_vacuous_and_unknown(self):
        for value in ('False', 'True', 'unknown_check(a)'):
            with self.subTest(value=value), self.assertRaises((ValueError, RuntimeError)):
                artifact_admission.check_artifact(self.source.replace(
                    'hashlib.sha256(raw).hexdigest() != expected_sha256', value))

    def test_model_rejects_skipped_gate_and_failure_escape(self):
        original = artifact_admission.successors
        for mutation in ('skip', 'escape'):
            def successors(state, count):
                yield from original(state, count)
                if mutation == 'skip' and state == (0, 0, False):
                    yield (count, (1 << count) - 1, False)
                if mutation == 'escape' and state[2]:
                    yield (count, (1 << count) - 1, False)
            with self.subTest(mutation=mutation), patch.object(artifact_admission, 'successors', successors):
                with self.assertRaises(RuntimeError):
                    artifact_admission.check_model(artifact_admission.extract(self.source)[1])

    def test_preserved_artifact_bypass_counterexamples(self):
        import hashlib
        import json
        fixtures = read_json(ROOT / 'formal/research/artifact-counterexamples.json')['entries']
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'policy.json'
            self.learning.save_policy(path, self.learning.Network(11), self.meta)
            raw = path.read_bytes()
            for fixture in fixtures:
                with self.subTest(id=fixture['id']):
                    self.assertEqual(self.source.count(fixture['replace']), 1)
                    mutant = self.source.replace(fixture['replace'], fixture['with'])
                    with self.assertRaises(RuntimeError):
                        artifact_admission.check_artifact(mutant)
                    body = json.loads(raw)
                    if fixture['id'] == 'CE-RL-008':
                        body['enabled'] = True
                    data = json.dumps(body, sort_keys=True).encode()
                    path.write_bytes(data)
                    digest = '0'*64 if fixture['id'] == 'CE-RL-007' else hashlib.sha256(data).hexdigest()
                    with self.assertRaises(ValueError):
                        self.learning.load_policy(path, digest, self.meta)
                    function = next(n for n in ast.parse(mutant).body
                                    if isinstance(n, ast.FunctionDef) and n.name == 'load_policy')
                    namespace = dict(vars(self.learning))
                    exec(compile(ast.Module(body=[function], type_ignores=[]), '<artifact-mutant>', 'exec'), namespace)
                    admitted = namespace['load_policy'](path, digest, self.meta)
                    self.assertEqual(set(admitted.p), {'w1', 'b1', 'w2', 'b2'})

    def test_actual_loader_admission_matrix_precedes_construction(self):
        import hashlib
        import json
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'policy.json'
            sha = self.learning.save_policy(path, self.learning.Network(11), self.meta)
            raw = path.read_bytes()
            base = json.loads(raw)
            accepted = self.learning.load_policy(path, sha, self.meta)
            for name, values in accepted.p.items():
                np.testing.assert_array_equal(values, np.asarray(base['parameters'][name]))
            variants = []
            for key, value in [('schema', 'offline_policy_v2'), ('environment', 'other'),
                               ('observation', 'other'), ('promotion', 'paper_eligible'),
                               ('actions', [-1, 0, 1]), ('actions', [-0.25, False, 0.25]),
                               ('actions', []), ('parameters', {}), ('provenance', {})]:
                item = copy.deepcopy(base); item[key] = value
                variants.append(json.dumps(item).encode())
            for enabled in (True, 0, 1, None, 'false', [], {}):
                item = copy.deepcopy(base); item['enabled'] = enabled
                variants.append(json.dumps(item).encode())
            for key, value in [('seed', 12), ('horizon', 3), ('fold', 1), ('algorithm', 'cql')]:
                item = copy.deepcopy(base); item['provenance'][key] = value
                variants.append(json.dumps(item).encode())
            for value in (True, '0', None, float('nan'), float('inf'), 10**400):
                item = copy.deepcopy(base); item['parameters']['b2'][0] = value
                variants.append(json.dumps(item).encode())
            variants += [b'{', b'[]', b'null', raw.replace(b'"enabled": false', b'"enabled": false, "enabled": false'),
                         raw.replace(b'"b2": [', b'"b2": [1e999,'), raw + b' '*(65537-len(raw))]
            self.assertEqual(len(variants), 32)
            for index, data in enumerate(variants):
                with self.subTest(index=index):
                    path.write_bytes(data)
                    with patch.object(self.learning, 'Network', side_effect=AssertionError('constructed rejected policy')):
                        with self.assertRaises((ValueError, OverflowError)):
                            self.learning.load_policy(path, hashlib.sha256(data).hexdigest(), self.meta)
            path.write_bytes(raw)
            with patch.object(self.learning, 'Network', side_effect=AssertionError('constructed wrong-hash policy')):
                with self.assertRaisesRegex(ValueError, 'artifact hash mismatch'):
                    self.learning.load_policy(path, '0'*64, self.meta)
            # Exactly 65536 bytes is legal JSON (trailing whitespace); one more
            # was rejected above. This is a size boundary, not a large artifact.
            boundary = raw + b' '*(65536-len(raw))
            path.write_bytes(boundary)
            self.learning.load_policy(path, hashlib.sha256(boundary).hexdigest(), self.meta)


class TransitionAdmissionTests(unittest.TestCase):
    def setUp(self):
        import sequential_env
        self.env_module = sequential_env
        self.source = (ROOT / transition_admission.SOURCE).read_text()

    def state(self, **changes):
        from types import SimpleNamespace
        fields = dict(t=31, stop=34, horizon=1, equity=1.0, units=0.0,
                      failure=None, pending=None)
        fields.update(changes)
        return SimpleNamespace(**fields)

    def test_actual_source_and_guard_mutations(self):
        self.assertEqual(transition_admission.check_transition(self.source)['queries'], 3)
        for old, new in [('and env.units == 0', 'and True'),
                         ('and env.pending is None', 'and True'),
                         ('_finite_real(reward)', 'True'),
                         ('env.t == left + env.horizon', 'env.t <= left + env.horizon'),
                         ('np.isfinite(nxt).all()', 'True')]:
            with self.subTest(new=new), self.assertRaises(RuntimeError):
                transition_admission.check_transition(self.source.replace(old, new))

    def test_control_flow_and_unknown_predicate_fail_closed(self):
        for old, new in [('_admit_training_transition(env, left, nxt, reward, done)', 'None'),
                         ('if not valid:', 'if False:'),
                         ('if done is True:', 'if unknown(done):')]:
            self.assertIn(old, self.source)
            with self.subTest(new=new), self.assertRaises(ValueError):
                transition_admission.check_transition(self.source.replace(old, new))
        with self.assertRaisesRegex(RuntimeError, 'unsatisfied or unknown premise'):
            transition_admission.prove(z3.BoolVal(False), z3.BoolVal(True))

    def test_preserved_terminal_inventory_counterexample(self):
        from types import SimpleNamespace
        fixture = read_json(ROOT / 'formal/research/transition-counterexamples.json')['entries'][0]
        env = SimpleNamespace(**{key: fixture[key] for key in ('t','stop','horizon','equity','units','failure','pending')})
        with self.assertRaisesRegex(ValueError, 'incomplete training transition'):
            self.env_module._admit_training_transition(env, fixture['left'], None, fixture['reward'], True)
        self.assertEqual(self.source.count(fixture['replace']), 1)
        mutant = self.source.replace(fixture['replace'], fixture['with'])
        with self.assertRaisesRegex(RuntimeError, 'unsafe admission predicate'):
            transition_admission.check_transition(mutant)
        function = next(n for n in ast.parse(mutant).body
                        if isinstance(n, ast.FunctionDef) and n.name == '_admit_training_transition')
        namespace = dict(vars(self.env_module))
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<transition-mutant>', 'exec'), namespace)
        namespace['_admit_training_transition'](env, fixture['left'], None, fixture['reward'], True)

    def test_actual_helper_boundaries(self):
        admit = self.env_module._admit_training_transition
        nxt = np.zeros(12)
        admit(self.state(), 30, nxt, 0.0, False)
        for units in (0.0, -0.0):
            admit(self.state(t=33, horizon=3, units=units), 30, None, 0.0, True)
        for failure in ('capital_floor', 'drawdown_limit', 'endpoint_exposure', 'turnover_limit'):
            admit(self.state(failure=failure), 30, None, 0.0, True)
        rejected = []
        for reward in (float('nan'), float('inf'), -float('inf'), True, None):
            rejected.append((self.state(), nxt, reward, False))
        for equity in (float('nan'), float('inf'), -float('inf'), 0.0, -1.0, True):
            rejected.append((self.state(equity=equity), nxt, 0.0, False))
        for done in (None, 0, 1, np.bool_(True), np.bool_(False)):
            rejected.append((self.state(), nxt, 0.0, done))
        for change in ({'t':30}, {'t':32}, {'stop':32}, {'failure':'invalid_market_transition'},
                       {'failure':'capital_floor'}):
            rejected.append((self.state(**change), nxt, 0.0, False))
        for successor in (None, np.zeros(11), np.full(12,np.nan), np.zeros(12,dtype=complex)):
            rejected.append((self.state(), successor, 0.0, False))
        for change in ({'units':.25}, {'units':float('nan')}, {'pending':(34,.25)}, {'t':31},
                       {'failure':'invalid_observation'}):
            rejected.append((self.state(t=33,horizon=3,**change) if 't' not in change else self.state(horizon=3,**change),None,0.0,True))
        self.assertEqual(len(rejected), 30)
        for i, values in enumerate(rejected):
            with self.subTest(index=i), self.assertRaises(ValueError):
                admit(values[0], 30, values[1], values[2], values[3])

    def test_actual_replay_terminal_conformance(self):
        from itertools import product
        e = self.env_module
        prices, funding = np.full(80,100.0), np.zeros(80)
        scale = e.Scale(np.zeros(6),np.ones(6),-np.ones(6),np.ones(6))
        cases = 0
        for horizon, target, costs, fraction in product((1,3,6),(-.25,0,.25),(1.,2.),(1.,.5)):
            replay = e.Replay(prices,funding,30,45,horizon,scale,
                              e.Execution(cost_multiplier=costs,fill_fraction=fraction),enabled=True)
            for _ in range(15):
                left = replay.t
                nxt, reward, done = replay.step(target)
                e._admit_training_transition(replay,left,nxt,reward,done)
                if done:
                    break
            self.assertTrue(replay.done)
            self.assertIsNone(replay.failure)
            self.assertEqual(replay.units,0.0)
            self.assertIsNone(replay.pending)
            cases += 1
        self.assertEqual(cases,36)

    def test_collector_does_not_publish_unliquidated_transition(self):
        e = self.env_module
        original = e.Replay.step
        rejected = []
        def injected(replay, action):
            result = original(replay, action)
            if result[2]:
                replay.units = 0.25
                rejected.append(replay.t)
            return result
        prices, funding = np.full(160,100.0), np.zeros(160)
        scale = e.Scale.fit([prices])
        with patch.object(e.Replay,'step',injected):
            with self.assertRaisesRegex(ValueError,'incomplete training transition'):
                e.collect({'x':prices},{'x':funding},scale,6,11,20)
        self.assertEqual(len(rejected),1)


class TerminalNumericsTests(unittest.TestCase):
    def setUp(self):
        import sequential_learning
        self.learning = sequential_learning
        self.source = (ROOT / terminal_numerics.SOURCE).read_text()
        self.fixtures = read_json(ROOT / 'formal/research/terminal-counterexamples.json')

    def critic(self, constant):
        net = self.learning.Network(11,1)
        for value in net.p.values():
            value.fill(0)
        net.p['b2'][0] = constant
        return net

    def data(self, rewards):
        rows = len(rewards)
        return {'s':np.zeros((rows,12)), 'next':np.zeros((rows,12)),
                'r':np.asarray(rewards,dtype=np.float64), 'done':np.ones(rows,dtype=bool)}

    def test_scoped_positive_claims_and_prescribed_refutations(self):
        result = terminal_numerics.check_terminal(self.fixtures,self.source)
        self.assertEqual(result['positiveQueries'],3)
        self.assertEqual([x['result'] for x in result['counterexamples']],['sat','sat'])

    def test_missing_masks_and_changed_arithmetic_rejected(self):
        for old,new in [('(~data["done"])','1.0'),
                        ('(not data["done"][i])','1.0'),
                        ('targets = adv + v','targets = adv - v')]:
            self.assertIn(old,self.source)
            with self.subTest(new=new),self.assertRaisesRegex(RuntimeError,'violating terminal claim'):
                terminal_numerics.check_terminal(self.fixtures,self.source.replace(old,new))
        with self.assertRaisesRegex(ValueError,'source skeleton drift'):
            terminal_numerics.check_terminal(self.fixtures,self.source.replace('reversed(range(len(v)))','range(len(v))'))
        with self.assertRaisesRegex(ValueError,'unsupported expression'):
            terminal_numerics.check_terminal(self.fixtures,self.source.replace('targets = adv + v','targets = unknown(adv) + v'))
        with self.assertRaisesRegex(RuntimeError,'unsatisfied/unknown premise'):
            terminal_numerics.prove('test',z3.BoolVal(False),z3.BoolVal(True))

    def test_counterexample_receipt_tampering_rejected(self):
        changed = copy.deepcopy(self.fixtures)
        changed['entries'][0]['expectedTargets'] = [float(1).hex()]
        with self.assertRaisesRegex(RuntimeError,'prescribed witness not SAT'):
            terminal_numerics.check_terminal(changed,self.source)

    def test_current_numpy_counterexamples_and_downstream_rejection(self):
        for fixture in self.fixtures['entries']:
            with self.subTest(id=fixture['id']):
                net = self.critic(float.fromhex(fixture['criticHex']))
                data = self.data([float.fromhex(x) for x in fixture['rewardHex']])
                self.assertTrue(np.isfinite(data['r']).all())
                self.assertTrue(np.isfinite(net.forward(data['s'])).all())
                self.assertTrue(np.isfinite(net.forward(data['next'])).all())
                # Expected non-finite results are asserted below, never accepted
                # as valid training output. Warning policy is local to this fixture.
                with np.errstate(over='ignore',invalid='ignore'):
                    adv,targets = self.learning.advantages(data,net,float.fromhex(fixture['gammaHex']))
                for actual,expected in zip(targets,fixture['expectedTargets']):
                    if expected == 'nan':
                        self.assertTrue(np.isnan(actual))
                    elif expected == 'inf':
                        self.assertTrue(np.isposinf(actual))
                    else:
                        self.assertEqual(float(actual).hex(),expected)
                if fixture['id'] == 'CE-RL-010':
                    self.assertNotEqual(targets[0],data['r'][0])
                else:
                    self.assertTrue(np.isnan(adv).all())
                    actor = self.learning.Network(23)
                    before = {k:v.copy() for k,v in actor.p.items()}
                    with self.assertRaises(ValueError):
                        actor.update(data['s'],np.repeat(adv[:,None],3,axis=1),0.001)
                    self.assertEqual(actor.steps,0)
                    for key in before:
                        np.testing.assert_array_equal(actor.p[key],before[key])

    def test_well_conditioned_terminal_grid(self):
        from itertools import product
        cases = 0
        for value,reward,gamma in product((-8.,0.,8.),(-2.,0.,2.),(0.,.99,1.)):
            data = self.data([reward])
            _,target = self.learning.advantages(data,self.critic(value),gamma)
            self.assertEqual(target[0],reward)
            cases += 1
        self.assertEqual(cases,27)

    def test_terminal_target_reset_excludes_normalized_advantage_claim(self):
        net = self.critic(0.)
        first = self.data([1.,2.,3.])
        later = self.data([1.,2.,1000.])
        a,t = self.learning.advantages(first,net,.99)
        b,u = self.learning.advantages(later,net,.99)
        self.assertEqual(t[0],u[0])
        self.assertEqual(t[0],1.)
        self.assertNotEqual(a[0],b[0])  # Batch normalization intentionally pools rows.


class PPOObjectiveTests(unittest.TestCase):
    def setUp(self):
        import sequential_learning
        self.learning=sequential_learning
        self.source=(ROOT/ppo_objective.SOURCE).read_text()
        self.fixtures=read_json(ROOT/'formal/research/ppo-counterexamples.json')

    def test_scoped_algebra_and_prescribed_counterexamples(self):
        result=ppo_objective.check_ppo(self.fixtures)
        self.assertEqual(result['queries'],4)
        self.assertEqual([x['result'] for x in result['counterexamples']],['sat','sat'])
        self.assertFalse(result['hardTrustRegionVerified'])
        self.assertFalse(result['softmaxImplementationVerified'])

    def test_objective_and_control_mutants_fail(self):
        mutations=[('probs[idx, actions] / old_prob','probs[idx, actions] * old_prob'),
                   ('np.clip(ratio, 0.8, 1.2)','np.clip(ratio, 0.8, 1.3)'),
                   ('loss = -np.minimum','loss = np.minimum'),
                   ('(advantage >= 0) & (ratio <= 1.2)','(advantage >= 0) & (ratio >= 1.2)'),
                   ('active * advantage * ratio / len(actions)','active * advantage * ratio / (2 * len(actions))'),
                   ('gradient[idx, actions] -= 1','gradient[idx, actions] += 1'),
                   ('clipped * advantage).mean()','clipped * advantage).sum()'),
                   ('return float(loss), gradient','return float(loss), gradient / 2')]
        for old,new in mutations:
            self.assertIn(old,self.source)
            with self.subTest(old=old),self.assertRaises((RuntimeError,ValueError)):
                ppo_objective.check_ppo(self.fixtures,self.source.replace(old,new))

    def test_fixture_tampering_rejected(self):
        changed=copy.deepcopy(self.fixtures)
        changed['entries'][0]['lossHex']=float(0).hex()
        with self.assertRaisesRegex(RuntimeError,'prescribed witness not SAT'):
            ppo_objective.check_ppo(changed)
        changed=copy.deepcopy(self.fixtures);changed['entries'].reverse()
        with self.assertRaisesRegex(ValueError,'roster drift'):
            ppo_objective.check_ppo(changed)

    def witness(self,entry):
        logits=np.array([[float.fromhex(x) for x in entry['logitsHex']]])
        old=np.array([float.fromhex(entry['oldProbabilityHex'])])
        adv=np.array([float.fromhex(entry['advantageHex'])])
        self.assertTrue(np.isfinite(logits).all() and np.isfinite(old).all() and np.isfinite(adv).all())
        self.assertGreater(old[0],0);self.assertLessEqual(old[0],1)
        probs=self.learning.softmax(logits)
        self.assertEqual(float(probs[0,entry['action']]).hex(),entry['probabilityHex'])
        # Expected failure is checked explicitly, never accepted as valid training.
        with np.errstate(over='ignore',invalid='ignore'):
            loss,gradient=self.learning.ppo_gradient(logits,np.array([entry['action']]),old,adv)
        self.assertEqual(loss.hex(),entry['lossHex'])
        return gradient

    def test_current_source_finite_loss_and_large_multiplier_witnesses(self):
        for entry in self.fixtures['entries']:
            gradient=self.witness(entry)
            if entry['id']=='CE-RL-012':
                self.assertTrue(np.isnan(gradient).all())
            else:
                self.assertTrue(np.isfinite(gradient).all())
                np.testing.assert_allclose(gradient,[[16/3,-8/3,-8/3]],rtol=0,atol=2e-15)
                self.assertGreater(abs(gradient[0,0]),1.2)

    def test_invalid_gradient_does_not_publish_optimizer_state(self):
        gradient=self.witness(self.fixtures['entries'][0])
        net=self.learning.Network(11)
        before={field:{k:v.copy() for k,v in getattr(net,field).items()} for field in ('p','m','v')}
        steps=net.steps
        with self.assertRaises(ValueError):
            net.update(np.zeros((1,12)),gradient,.0003)
        self.assertEqual(net.steps,steps)
        for field in before:
            for key in before[field]:
                np.testing.assert_array_equal(getattr(net,field)[key],before[field][key])

    def test_registered_rational_grid_and_finite_differences(self):
        from fractions import Fraction as F
        from itertools import product
        cases=0
        lo,hi=F(.8),F(1.2)
        for wanted,a,n in product((0,.5,.8,1,1.2,1.5,8),(-2,0,2),(1,2,256)):
            logits=np.tile([-1000.,0.,0.] if wanted==0 else [0.,0.,0.],(n,1))
            probs=self.learning.softmax(logits)
            old=np.full(n,1/3 if wanted==0 else probs[0,0]/wanted)
            ratio=float(probs[0,0]/old[0]);r=F(ratio)
            loss,gradient=self.learning.ppo_gradient(logits,np.zeros(n,dtype=int),old,np.full(n,float(a)))
            expected_loss=-a*(min(r,hi) if a>=0 else max(r,lo))
            active=(a>=0 and r<=hi) or (a<0 and r>=lo)
            coefficient=F(a)*r/n if active else F(0)
            expected=[float((F(float(p))-(1 if j==0 else 0))*coefficient) for j,p in enumerate(probs[0])]
            self.assertAlmostEqual(loss,float(expected_loss),delta=1e-12)
            np.testing.assert_allclose(gradient,np.tile(expected,(n,1)),rtol=0,atol=1e-12)
            cases+=1
        self.assertEqual(cases,63)
        for ratio,a in product((.5,1.,2.),(-2.,2.)):
            logits=np.array([[.2,-.1,.4]])
            old=np.array([self.learning.softmax(logits)[0,0]/ratio]);adv=np.array([a]);actions=np.array([0])
            loss,gradient=self.learning.ppo_gradient(logits,actions,old,adv)
            for j in range(3):
                up=logits.copy();down=logits.copy();up[0,j]+=1e-6;down[0,j]-=1e-6
                derivative=(self.learning.ppo_gradient(up,actions,old,adv)[0]-self.learning.ppo_gradient(down,actions,old,adv)[0])/2e-6
                self.assertAlmostEqual(gradient[0,j],derivative,delta=1e-8)

    def test_boundary_convention_with_prescribed_probabilities(self):
        # Conditional branch conformance, not verification of softmax itself.
        for center,a in ((.4,-1.),(.6,1.)):
            for p in (np.nextafter(center,0),center,np.nextafter(center,1)):
                probs=np.array([[p,(1-p)/2,(1-p)/2]])
                with patch.object(self.learning,'softmax',return_value=probs):
                    _,gradient=self.learning.ppo_gradient(np.zeros((1,3)),np.array([0]),np.array([.5]),np.array([a]))
                ratio=p/.5
                active=(a>=0 and ratio<=1.2) or (a<0 and ratio>=.8)
                self.assertEqual(bool(np.any(gradient!=0)),active)


class QueryIsolationTests(unittest.TestCase):
    """Conformance of the actual two proof drivers; solver semantics remain trusted."""

    def exercise(self, driver, outcomes):
        premise, conclusion, witness = z3.Bools('qi_p qi_c qi_w')
        records = []

        class RecordingSolver:
            def __init__(self):
                self.index = len(records)
                self.formulas, self.options, self.checks = [], {}, 0
                records.append(self)

            def set(self, **options):
                self.options.update(options)

            def add(self, *formulas):
                self.formulas.extend(formulas)

            def check(self):
                self.checks += 1
                result = outcomes[self.index]
                if isinstance(result, Exception):
                    raise result
                return result

            def model(self):
                return 'injected counterexample'

            def reason_unknown(self):
                return 'injected timeout'

        error = None
        with patch.object(z3, 'Solver', RecordingSolver):
            try:
                if driver == 'prove':
                    terminal_numerics.prove('fixture', premise, conclusion)
                else:
                    target_v2.certify('fixture', premise, conclusion, [witness])
            except RuntimeError as exc:
                error = exc
        first = [premise] if driver == 'prove' else [premise, witness]
        expected = [first, [premise, z3.Not(conclusion)]]
        for index, solver in enumerate(records):
            self.assertEqual(solver.options, {'timeout':10000, 'random_seed':0})
            self.assertEqual(solver.checks, 1)
            self.assertEqual([f.sexpr() for f in solver.formulas],
                             [f.sexpr() for f in expected[index]])
        self.assertEqual(len(records), 2 if outcomes[0] == z3.sat else 1)
        if len(records) == 2:
            self.assertIsNot(records[0], records[1])
        return error

    def test_complete_solver_result_table(self):
        cases = 0
        for driver in ('prove','certify'):
            for first in (z3.sat,z3.unsat,z3.unknown):
                for second in (z3.sat,z3.unsat,z3.unknown):
                    with self.subTest(driver=driver,first=first,second=second):
                        error = self.exercise(driver,(first,second))
                        self.assertEqual(error is None, first == z3.sat and second == z3.unsat)
                        if first == z3.unknown or first == z3.sat and second == z3.unknown:
                            self.assertIn('injected timeout',str(error))
                        if first == z3.sat and second == z3.sat:
                            self.assertIn('injected counterexample',str(error))
                    cases += 1
        self.assertEqual(cases,18)

    def test_exceptions_from_either_query_are_not_accepted(self):
        for driver in ('prove','certify'):
            for index in (0,1):
                failure = RuntimeError('solver failed')
                results = [z3.sat,z3.unsat]
                results[index] = failure
                with self.subTest(driver=driver,index=index):
                    self.assertIs(self.exercise(driver,results),failure)

    def test_actual_solver_checks_entire_domain(self):
        x = z3.Real('qi_entire_domain')
        for driver in ('prove','certify'):
            def check(p,c,w):
                return (terminal_numerics.prove('actual',p,c) if driver == 'prove'
                        else target_v2.certify('actual',p,c,w))
            with self.subTest(driver=driver):
                check(x>=0,x+1>0,[x==0])
                with self.assertRaises(RuntimeError):
                    check(z3.BoolVal(True),x==0,[x==0])
                with self.assertRaises(RuntimeError):
                    check(z3.BoolVal(False),z3.BoolVal(True),[])
                # A witness inconsistent with P must not permit a vacuous success.
                if driver == 'certify':
                    with self.assertRaises(RuntimeError):
                        check(x>0,x>=0,[x==0])


class TargetV2Tests(unittest.TestCase):
    def setUp(self):
        import gae_targets_v2
        self.kernel = gae_targets_v2
        self.source = (ROOT / target_v2.SOURCE).read_text()

    def test_scoped_v2_proofs_and_publication_model(self):
        result = target_v2.check_targets()
        self.assertEqual(result['queries'],4)
        self.assertTrue(result['premiseWitnessRemovedBeforeViolation'])
        self.assertEqual(result['model']['states'],33410)
        self.assertEqual(result['model']['transitions'],66562)
        self.assertEqual(result['model']['progressBound'],258)
        self.assertFalse(result['wholeLearnerCorrection'])

    def test_source_mutations_rejected(self):
        mutations = [
            ('target = reward','target = raw + value'),
            ('gamma * 0.95','gamma * 0.5'),
            ('enabled: object = False','enabled: object = True'),
            ('if not all(isfinite(x) for x in (raw, target)):', 'if False:'),
            ('if pair is None:', 'if False:'),
            ('staged.append(pair)', 'return tuple(staged)'),
            ('from math import isfinite','from math import isfinite\nimport os')]
        for old,new in mutations:
            self.assertIn(old,self.source)
            with self.subTest(old=old),self.assertRaises((RuntimeError,ValueError)):
                target_v2.check_targets(self.source.replace(old,new))

    def test_premise_witness_does_not_restrict_universal_query(self):
        x = z3.Real('witness_scope')
        with self.assertRaisesRegex(RuntimeError,'violating/unknown obligation'):
            target_v2.certify('scope',z3.BoolVal(True),x==0,[x==0])
        with self.assertRaisesRegex(RuntimeError,'premise witness failed'):
            target_v2.certify('vacuous',z3.BoolVal(False),z3.BoolVal(True),[])

    def test_v1_counterexamples_corrected_or_rejected_in_v2_only(self):
        fixtures = read_json(ROOT / 'formal/research/terminal-counterexamples.json')
        for entry in fixtures['entries']:
            r = tuple(float.fromhex(x) for x in entry['rewardHex'])
            v,g = float.fromhex(entry['criticHex']),float.fromhex(entry['gammaHex'])
            rows = tuple((reward,v,v,True) for reward in r)
            result = self.kernel.batch_v2(rows,g,enabled=True)
            if entry['id']=='CE-RL-010':
                self.assertEqual(result[0][1].hex(),r[0].hex())
                self.assertEqual(result[0][1],1.)
            else:
                self.assertIsNone(result)
            self.assertEqual(rows,tuple((reward,v,v,True) for reward in r))
        # Preserve the original defect and artifact/training semantics.
        import sequential_learning
        net = sequential_learning.Network(11,1)
        for value in net.p.values(): value.fill(0)
        net.p['b2'][0] = 1e16
        data = {'s':np.zeros((1,12)),'next':np.zeros((1,12)),
                'r':np.array([1.]),'done':np.array([True])}
        _,targets = sequential_learning.advantages(data,net,.99)
        self.assertEqual(targets[0],0.)

    def test_default_disabled_controls_and_strict_native_inputs(self):
        row = (1.,0.,0.,True)
        self.assertIsNone(self.kernel.batch_v2((row,),.99))
        self.assertIsNone(self.kernel.step_v2(1.,0.,0.,0.,True,.99))
        for enabled in (False,None,0,1,'true',np.bool_(True)):
            self.assertIsNone(self.kernel.batch_v2((row,),.99,enabled=enabled))
        class Hostile:
            def __eq__(self, other): raise AssertionError('coercion')
            def __len__(self): raise AssertionError('inspection')
        for version in (None,1,'gae-targets-v1',Hostile()):
            self.assertIsNone(self.kernel.batch_v2((row,),.99,enabled=True,version=version))
        self.assertIsNone(self.kernel.batch_v2(Hostile(),Hostile()))
        for gamma in (True,1,-.1,1.1,float('nan'),float('inf'),np.float64(.99),Hostile()):
            self.assertIsNone(self.kernel.batch_v2((row,),gamma,enabled=True))
        for rows in ([],(),(row,)*257,[row],((),),((1.,0.,0.),),(Hostile(),)):
            self.assertIsNone(self.kernel.batch_v2(rows,.99,enabled=True))
        for i in range(4):
            for bad in (None,Hostile(),float('nan'),float('inf'),1,np.float64(1.)):
                changed = list(row); changed[i] = bad
                self.assertIsNone(self.kernel.batch_v2((tuple(changed),),.99,enabled=True))

    def test_signed_zero_extremes_and_overflow(self):
        import sys
        for r in (-0.,0.,float.fromhex('0x0.0000000000001p-1022'),sys.float_info.max):
            pair = self.kernel.step_v2(r,0.,0.,0.,True,.99,enabled=True)
            self.assertEqual(pair[1].hex(),r.hex())
        maximum = sys.float_info.max
        for row in ((maximum,-maximum,0.,True),(maximum,0.,maximum,False)):
            self.assertIsNone(self.kernel.batch_v2((row,),1.,enabled=True))
        self.assertIsNone(self.kernel.step_v2(1.,0.,0.,float('inf'),True,.99,enabled=True))

    def test_every_failure_position_and_maximum_batch(self):
        row = (1.,0.,0.,True)
        rows = (row,)*256
        self.assertEqual(self.kernel.batch_v2(rows,.99,enabled=True),((1.,1.),)*256)
        for i in range(256):
            changed = rows[:i]+((float('nan'),0.,0.,True),)+rows[i+1:]
            self.assertIsNone(self.kernel.batch_v2(changed,.99,enabled=True))
            self.assertTrue(np.isnan(changed[i][0]))
        first = self.kernel.batch_v2(rows,.99,enabled=True)
        self.assertIsNone(self.kernel.batch_v2(rows,.99))
        self.assertEqual(self.kernel.batch_v2(rows,.99,enabled=True),first)

    def test_rational_reference_grid_and_deterministic_sequences(self):
        from fractions import Fraction as F
        from itertools import product
        import math
        import random
        count = 0
        for r,v,n,c,gamma,done in product(*([(-2.,0.,2.)]*4), (0.,.5,1.), (False,True)):
            pair = self.kernel.step_v2(r,v,n,c,done,gamma,enabled=True)
            rf,vf,nf,cf,gf = map(F,(r,v,n,c,gamma))
            raw = rf-vf if done else rf+gf*nf-vf+gf*F(.95)*cf
            target = rf if done else raw+vf
            self.assertTrue(math.isclose(pair[0],float(raw),rel_tol=0.,abs_tol=1e-12))
            self.assertTrue(math.isclose(pair[1],float(target),rel_tol=0.,abs_tol=1e-12))
            count+=1
        self.assertEqual(count,486)
        rng = random.Random(20260928)
        for _ in range(64):
            rows = tuple((rng.uniform(-4,4),rng.uniform(-4,4),rng.uniform(-4,4),rng.choice((True,False)))
                         for _ in range(rng.randint(1,256)))
            first = self.kernel.batch_v2(rows,.99,enabled=True)
            self.assertEqual(first,self.kernel.batch_v2(rows,.99,enabled=True))
            self.assertTrue(all(math.isfinite(x) for pair in first for x in pair))
            # An exact terminal branch disconnects earlier raw targets from
            # changes strictly after that terminal, provided both batches admit.
            cut = next((i for i,row in enumerate(rows[:-1]) if row[3]),None)
            if cut is not None:
                changed = rows[:cut+1]+tuple((r+1,v-1,n+2,d) for r,v,n,d in rows[cut+1:])
                second = self.kernel.batch_v2(changed,.99,enabled=True)
                self.assertEqual(first[:cut+1],second[:cut+1])


class ValueObjectiveTests(unittest.TestCase):
    def setUp(self):
        import sys
        sys.path.insert(0,str(ROOT/'scripts/research'))
        import sequential_learning
        self.learning=sequential_learning
        self.fixtures=read_json(ROOT/'formal/research/value-counterexamples.json')
        self.source=(ROOT/value_objective.SOURCE).read_text()

    def test_scoped_value_certificates(self):
        result=value_objective.check_values(self.fixtures)
        self.assertEqual(result['queries'],2)
        self.assertEqual(set(result['smt']),{'F-RL-DOUBLE-TARGET','F-RL-CQL-GRADIENT'})
        self.assertEqual([x['result'] for x in result['counterexamples']],['sat','sat'])
        self.assertFalse(result['transcendentalsVerified'])
        self.assertFalse(result['fullTrainingRefinement'])
        self.assertFalse(result['policyLowerBoundVerified'])

    def test_source_mutants_fail(self):
        mutants=[
            ('greedy = np.argmax(net.forward(nxt), axis=1)','greedy = np.argmax(target.forward(nxt), axis=1)'),
            ('target.forward(nxt)[np.arange(64), greedy]','net.forward(nxt)[np.arange(64), greedy]'),
            ('0.99**horizon * (~buffer["done"][idx])','0.9**horizon * (~buffer["done"][idx])'),
            ('(~buffer["done"][idx])','buffer["done"][idx]'),
            ('residual = q[idx, actions] - targets','residual = q[idx, actions] + targets'),
            ('grad[idx, actions] = residual / n','grad[idx, actions] = -residual / n'),
            ('grad += alpha * penalty / n','grad += alpha * penalty / (2*n)'),
            ('penalty[idx, actions] -= 1','penalty[idx, actions] += 1'),
            ('0.5 * np.mean(residual**2)','0.25 * np.mean(residual**2)'),
            ('alpha * np.mean(logsumexp - q[idx, actions])','alpha * np.mean(logsumexp + q[idx, actions])'),
            ('return float(loss), grad\n','return float(loss), grad / 2\n'),
            ('np.log(np.exp(q - m[:, None]).sum(1))','np.log(np.exp(q).sum(1))')]
        for before,after in mutants:
            with self.subTest(mutation=after):
                self.assertIn(before,self.source)
                with self.assertRaises((ValueError,RuntimeError)):
                    value_objective.check_values(self.fixtures,self.source.replace(before,after,1))

    def test_fixture_drift_fails(self):
        for change in ('loss','roster'):
            fixture=copy.deepcopy(self.fixtures)
            if change=='loss':fixture['entries'][1]['shiftedLossHex']=float(1).hex()
            else:fixture['entries'].reverse()
            with self.assertRaises((ValueError,RuntimeError)):
                value_objective.check_values(fixture)

    def test_target_grid_against_independent_reference(self):
        from itertools import product
        expressions=value_objective.extract(self.source)
        # Compile exactly the audited source slice, including actual NumPy argmax.
        greedy=ast.parse(value_objective.TARGET).body[0]
        target=ast.Assign(targets=[ast.Name(id='targets',ctx=ast.Store())],value=expressions['target'])
        code=compile(ast.fix_missing_locations(ast.Module(body=[greedy,target],type_ignores=[])),'target-slice','exec')
        triples=list(product((-1.,0.,1.),repeat=3));cases=0
        class Forward:
            def __init__(self,values):self.values=values
            def forward(self,_):return self.values
        for horizon in (1,3,6):
            rows=list(product(triples,triples,(-1.,0.,1.),(False,True)))
            for start in range(0,len(rows),64):
                batch=rows[start:start+64];actual_count=len(batch)
                batch+=batch[-1:]*(64-actual_count)
                online=np.array([v[0] for v in batch]);target_q=np.array([v[1] for v in batch])
                rewards=np.array([v[2] for v in batch]);done=np.array([v[3] for v in batch])
                scope={'np':np,'net':Forward(online),'target':Forward(target_q),'nxt':None,
                       'buffer':{'r':rewards,'done':done},'idx':np.arange(64),'horizon':horizon}
                exec(code,scope)
                for i,(o,t,r,d) in enumerate(batch[:actual_count]):
                    chosen=max(range(3),key=lambda j:o[j])
                    expected=r if d else r+(.99**horizon)*t[chosen]
                    self.assertEqual(scope['greedy'][i],chosen)
                    self.assertEqual(scope['targets'][i],expected)
                    cases+=1
        self.assertEqual(cases,13122)

    def test_gradient_grid_and_finite_differences(self):
        from itertools import product
        from math import exp,fsum,log
        cases=0
        for values,a,y,alpha,n in product(((-1.,0.,1.),(0.,0.,0.),(1.,-1.,.5)),range(3),(-1.,0.,1.),(0.,.1,1.),(1,2,64,256)):
            q=np.tile(values,(n,1));actions=np.full(n,a);targets=np.full(n,y)
            loss,gradient=self.learning.bellman_gradient(q,actions,targets,alpha)
            exps=[exp(x-max(values)) for x in values];den=fsum(exps);p=[v/den for v in exps]
            residual=values[a]-y
            expected=[((residual if j==a else 0)+alpha*(p[j]-(j==a)))/n for j in range(3)]
            regularizer=log(den)+(max(values)-values[a])
            self.assertAlmostEqual(loss,.5*residual**2+alpha*regularizer,delta=1e-12)
            np.testing.assert_allclose(gradient,np.tile(expected,(n,1)),rtol=0,atol=1e-12)
            cases+=1
        self.assertEqual(cases,324)
        for a,alpha in product(range(3),(0.,.1,1.)):
            q=np.array([[.2,-.1,.4]]);actions=np.array([a]);targets=np.array([.7])
            _,gradient=self.learning.bellman_gradient(q,actions,targets,alpha)
            for j in range(3):
                up=q.copy();down=q.copy();up[0,j]+=1e-6;down[0,j]-=1e-6
                derivative=(self.learning.bellman_gradient(up,actions,targets,alpha)[0]-self.learning.bellman_gradient(down,actions,targets,alpha)[0])/2e-6
                self.assertAlmostEqual(gradient[0,j],derivative,delta=1e-8)

    def test_current_source_numeric_witnesses(self):
        for entry in self.fixtures['entries']:
            q=np.array([[float.fromhex(v) for v in entry['qHex']]])
            target=np.array([float.fromhex(entry['targetHex'])]);alpha=float.fromhex(entry['alphaHex'])
            self.assertTrue(np.isfinite(q).all() and np.isfinite(target).all())
            with np.errstate(over='ignore',invalid='ignore'):
                intermediate=np.log(np.exp(q-q.max(1)[:,None]).sum(1))[0]
                loss,gradient=self.learning.bellman_gradient(q,np.array([entry['action']]),target,alpha)
            self.assertEqual(float(intermediate).hex(),entry['logIntermediateHex'])
            self.assertTrue(np.isfinite(gradient).all())
            if entry['id']=='CE-RL-014':
                self.assertTrue(np.isnan(loss));np.testing.assert_array_equal(gradient,np.zeros((1,3)))
            else:
                original,original_gradient=self.learning.bellman_gradient(np.zeros((1,3)),np.array([0]),np.array([0.]),alpha)
                self.assertEqual(loss.hex(),entry['shiftedLossHex'])
                self.assertEqual(original.hex(),entry['originalLossHex'])
                self.assertNotEqual(loss,original)
                np.testing.assert_array_equal(gradient,original_gradient)

    def test_nan_loss_is_not_an_optimizer_input(self):
        entry=self.fixtures['entries'][0]
        q=np.array([[float.fromhex(v) for v in entry['qHex']]])
        with np.errstate(over='ignore',invalid='ignore'):
            loss,gradient=self.learning.bellman_gradient(q,np.array([2]),np.array([float.fromhex(entry['targetHex'])]),0.)
        self.assertTrue(np.isnan(loss))
        net=self.learning.Network(11)
        before={k:v.copy() for k,v in net.p.items()}
        net.update(np.zeros((1,12)),gradient,.001)
        self.assertEqual(net.steps,1)
        for k in before:np.testing.assert_array_equal(net.p[k],before[k])


class OptimizerPublicationTests(unittest.TestCase):
    def setUp(self):
        import sys
        sys.path.insert(0,str(ROOT/'scripts/research'))
        import sequential_learning
        self.learning=sequential_learning
        self.source=(ROOT/optimizer_publication.SOURCE).read_text()
        self.fixtures=read_json(ROOT/'formal/research/optimizer-counterexamples.json')

    def test_model_and_clip_certificate(self):
        r=optimizer_publication.check_optimizer(self.fixtures)
        self.assertEqual(r['smt'],{'F-RL-GRADIENT-CLIP':'unsat'})
        self.assertEqual(r['singleWriter']['gates'],32)
        self.assertEqual(r['singleWriter']['states'],69)
        self.assertEqual(r['singleWriter']['maxShortestDepth'],36)
        self.assertEqual(r['observerExtension']['observedMasks'],[0,1,3,7])
        self.assertFalse(r['concurrentAtomicityVerified'])
        self.assertFalse(r['floatingNormPremiseVerified'])

    def test_source_mutants_fail(self):
        changes=[
            ('steps = int(self.steps) + 1','self.steps = int(self.steps) + 1\n                steps = self.steps'),
            ('if not all(np.isfinite(v).all() for state in (params, moments, variances)','if not all(np.isfinite(v).all() for state in (params,)'),
            ('if not np.isfinite(norm):','if norm < 0:'),
            ('grad / max(1.0, norm)','grad * max(1.0, norm)'),
            ('grad / max(1.0, norm)','grad / max(0.5, norm)'),
            ('self.p, self.m, self.v, self.steps = params, moments, variances, steps','self.m, self.p, self.v, self.steps = moments, params, variances, steps'),
            ('h = np.tanh(x @ self.p["w1"] + self.p["b1"])','self.p["w1"] *= 2\n        h = np.tanh(x @ self.p["w1"] + self.p["b1"])'),
            ('moments[k] = 0.9 * self.m[k] + 0.1 * g','self.m[k] = 0.9 * self.m[k] + 0.1 * g\n                    moments[k] = self.m[k]')]
        for before,after in changes:
            with self.subTest(mutation=after):
                self.assertIn(before,self.source)
                with self.assertRaises((ValueError,RuntimeError)):
                    optimizer_publication.check_optimizer(self.fixtures,self.source.replace(before,after,1))

    def test_fixture_and_model_mutants_fail(self):
        for key,value in (('normalMasks',[0,15]),('interruptedMask',0),('trace',[])):
            fixture=copy.deepcopy(self.fixtures);fixture['entries'][0][key]=value
            with self.assertRaises(ValueError):optimizer_publication.check_optimizer(fixture)
        original=optimizer_publication.transitions
        def stalled(state,gates,fields,extended):
            if state==(0,False,-1):yield 'pass:stalled',state
            else:yield from original(state,gates,fields,extended)
        with patch.object(optimizer_publication,'transitions',stalled),self.assertRaisesRegex(RuntimeError,'rank'):
            optimizer_publication.check_optimizer(self.fixtures)

    def snapshot(self,net):
        return ({k:getattr(net,k) for k in ('p','m','v')},
                {k:{name:v.copy() for name,v in getattr(net,k).items()} for k in ('p','m','v')},net.steps)

    def test_gradient_helper_preserves_inputs_and_optimizer_state(self):
        from itertools import product
        for seed,outputs,warm in product((11,23,47),(1,3),(False,True)):
            net=self.learning.Network(seed,outputs)
            x=np.arange(24,dtype=float).reshape(2,12)/24
            dz=np.ones((2,outputs))*.1
            if warm:net.update(x,dz,.001)
            refs,values,steps=self.snapshot(net);before_x=x.copy();before_dz=dz.copy()
            gradients=net.gradients(x,dz)
            self.assertEqual(set(gradients),{'w1','b1','w2','b2'})
            self.assertEqual(net.steps,steps)
            np.testing.assert_array_equal(x,before_x)
            np.testing.assert_array_equal(dz,before_dz)
            for field in refs:
                self.assertIs(getattr(net,field),refs[field])
                for name in values[field]:
                    np.testing.assert_array_equal(getattr(net,field)[name],values[field][name])
                    for gradient in gradients.values():
                        self.assertFalse(np.shares_memory(gradient,getattr(net,field)[name]))

    def test_prepublication_failure_matrix_preserves_references_and_values(self):
        from itertools import product
        count=0
        for seed,outputs,warm,fault in product((11,23,47),(1,3),(False,True),('rate','counter','gradient','norm','candidate')):
            net=self.learning.Network(seed,outputs);x=np.ones((2,12))*.1;dz=np.ones((2,outputs))*.1
            if warm:net.update(x,dz,.001)
            rate=.001
            if fault=='rate':rate=float('nan')
            if fault=='counter':net.steps=-1
            if fault=='candidate':net.v['b2'][:]=-1
            refs,values,steps=self.snapshot(net);errors=np.geterr().copy()
            gradients=net.gradients(x,dz)
            if fault in ('gradient','norm'):
                for a in gradients.values():a[:]=float('nan') if fault=='gradient' else 1e200
            with self.subTest(seed=seed,outputs=outputs,warm=warm,fault=fault):
                with patch.object(net,'gradients',return_value=gradients),np.errstate(all='ignore'):
                    with self.assertRaises(ValueError):net.update(x,dz,rate)
                    self.assertEqual(np.geterr(),dict.fromkeys(errors,'ignore'))
                self.assertEqual(np.geterr(),errors)
                self.assertEqual(net.steps,steps)
                for field in refs:
                    self.assertIs(getattr(net,field),refs[field])
                    for name in values[field]:np.testing.assert_array_equal(getattr(net,field)[name],values[field][name])
                count+=1
        self.assertEqual(count,60)

    def trace_update(self,net,outputs,interrupt):
        import sys,dis
        old=(net.p,net.m,net.v,net.steps);events=[]
        code=self.learning.Network.update.__code__;ops={i.offset:i for i in dis.get_instructions(code)}
        def mask():
            return sum(int((getattr(net,k) is not v) if k!='steps' else net.steps!=v)<<i
                       for i,(k,v) in enumerate(zip(('p','m','v','steps'),old)))
        def trace(frame,event,arg):
            if frame.f_code is code:
                if event=='call':
                    # Install the local hook explicitly before requesting opcodes.
                    frame.f_trace=trace
                    frame.f_trace_opcodes=True
                if event=='opcode':
                    op=ops.get(frame.f_lasti)
                    if op and op.opname=='STORE_ATTR':
                        events.append(mask())
                        if interrupt and op.argval=='m':raise RuntimeError('registered publication interruption')
                if event=='return' and not interrupt:events.append(mask())
                return trace
        prior=sys.gettrace();caller=sys._getframe();old_flag=caller.f_trace_opcodes
        caught=None
        try:
            caller.f_trace_opcodes=True
            sys.settrace(trace)
            net.update(np.ones((2,12))*.1,np.ones((2,outputs))*.1,.001)
        except RuntimeError as exc:
            caught=str(exc)
        finally:
            sys.settrace(prior);caller.f_trace_opcodes=old_flag
        self.assertIs(sys.gettrace(),prior)
        self.assertEqual(caller.f_trace_opcodes,old_flag)
        self.assertEqual(caught,'registered publication interruption' if interrupt else None)
        return events,mask()

    def test_opcode_observation_and_interruption_conformance(self):
        from itertools import product
        entry=self.fixtures['entries'][0];cases=0
        for seed,outputs,warm,interrupt in product((11,23,47),(1,3),(False,True),(False,True)):
            net=self.learning.Network(seed,outputs)
            if warm:net.update(np.ones((2,12))*.1,np.ones((2,outputs))*.1,.001)
            with self.subTest(seed=seed,outputs=outputs,warm=warm,interrupt=interrupt):
                events,mask=self.trace_update(net,outputs,interrupt)
                self.assertEqual(events,[0,1] if interrupt else entry['normalMasks'])
                self.assertEqual(mask,entry['interruptedMask'] if interrupt else 15)
                cases+=1
        self.assertEqual(cases,24)

    def test_clip_grid_and_existing_golden(self):
        from fractions import Fraction as F
        from itertools import product
        clip,_,_=optimizer_publication.extract(self.source)
        code=compile(ast.fix_missing_locations(ast.Expression(body=clip)),'source-clip','eval')
        for grad,extra in product((-2.,-1.,-.5,0.,.5,1.,2.),(0.,1.,10.)):
            norm=abs(grad)+extra;result=eval(code,{'grad':grad,'norm':norm,'max':max})
            reference=F(grad)/max(F(1),F(norm))
            self.assertAlmostEqual(result,float(reference),delta=1e-15)
            self.assertLessEqual(abs(result),1)
            self.assertLessEqual(abs(result),abs(grad))
        net=self.learning.Network(11);x=np.arange(48).reshape(4,12)/48;dz=np.arange(12).reshape(4,3)/12-.5
        for i in range(3):net.update(x,dz*(i+1),np.float64(.0003))
        fixture=read_json(ROOT/'test/fixtures/sequential-adam-v1.json')
        self.assertEqual(net.steps,fixture['steps'])
        for name,values in fixture['state'].items():
            for key,expected in values.items():
                np.testing.assert_allclose(getattr(net,name)[key],expected,rtol=1e-13,atol=1e-15)


class InferenceBoundaryTests(unittest.TestCase):
    def setUp(self):
        import sys
        sys.path.insert(0, str(ROOT / 'scripts/research'))
        import sequential_learning
        self.learning = sequential_learning
        self.source = (ROOT / inference_boundary.SOURCE).read_text()
        self.env = (ROOT / inference_boundary.ENV).read_text()
        self.fixtures = read_json(ROOT / 'formal/research/inference-counterexamples.json')

    def test_certificates_and_pending_lasso(self):
        result = inference_boundary.check_inference(self.fixtures)
        self.assertEqual(len(result['smt']), 2)
        self.assertEqual(result['model']['states'], 9)
        self.assertFalse(result['model']['preemptionVerified'])
        self.assertFalse(result['runtimeRefinement'])
        self.assertEqual(result['model']['deadlineCounterexample'],
                         {'prefix': ['enable', 'admit_observation'], 'cycle': ['pending']})

    def test_source_and_constant_mutants_fail(self):
        changes = [('enabled is not True or not _finite_real_vector(observation, FEATURE_COUNT)',
                    'enabled is not True and not _finite_real_vector(observation, FEATURE_COUNT)'),
                   ('0 <= elapsed <= 20', '0 <= elapsed <= 21'),
                   ('0 <= elapsed <= 20', 'elapsed <= 20'),
                   ('_finite_real_vector(out, 3)', '_finite_real_vector(out, 2)'),
                   ('enabled: bool = False', 'enabled: bool = True'),
                   ('np.argmax(out)', 'np.argmin(out)'),
                   ('except Exception:\n', 'except BaseException:\n'),
                   ('not np.ma.isMaskedArray(value)', 'True'),
                   ('value.dtype.kind in "iuf"', 'value.dtype.kind in "biuf"')]
        for before, after in changes:
            with self.subTest(mutation=after):
                self.assertIn(before, self.source)
                with self.assertRaises((ValueError, RuntimeError)):
                    inference_boundary.check_inference(self.fixtures, self.source.replace(before, after, 1))
        for before, after in [('[-0.25, 0.0, 0.25]', '[-0.5, 0.0, 0.5]'), ('FEATURE_COUNT = 12', 'FEATURE_COUNT = 11')]:
            with self.assertRaises(ValueError):
                inference_boundary.check_inference(self.fixtures, env=self.env.replace(before, after, 1))

    def test_model_and_fixture_mutations_fail(self):
        for key, value in [('prefix', []), ('cycle', []), ('elapsedNanoseconds', 20000000), ('expectedProposal', 0.)]:
            fixture = copy.deepcopy(self.fixtures); fixture['entries'][0][key] = value
            with self.assertRaises((ValueError, RuntimeError)):
                inference_boundary.check_inference(fixture)
        original = inference_boundary.transitions
        def bypass(state):
            if state == ('entry', 0): yield 'bypass', ('proposal', 0)
            else: yield from original(state)
        with patch.object(inference_boundary, 'transitions', bypass), self.assertRaisesRegex(RuntimeError, 'bypass'):
            inference_boundary.check_inference(self.fixtures)
        def no_pending(state):
            for event, nxt in original(state):
                if event != 'pending': yield event, nxt
        with patch.object(inference_boundary, 'transitions', no_pending), self.assertRaisesRegex(RuntimeError, 'lasso'):
            inference_boundary.check_inference(self.fixtures)

    def test_score_timing_grid_matches_first_maximum(self):
        from itertools import product
        registration = read_json(ROOT / 'research-notes/registrations/inference-boundary-audit-engineering.json')
        net = self.learning.Network(11); count = 0
        for scores in product(registration['scoreGrid'], repeat=3):
            expected = (-.25, 0., .25)[scores.index(max(scores))]
            for elapsed in registration['elapsedNanoseconds']:
                with self.subTest(scores=scores, elapsed=elapsed), \
                     patch.object(net, 'forward', return_value=np.array(scores, dtype=float)) as forward, \
                     patch.object(self.learning.time, 'perf_counter_ns', side_effect=[123, 123 + elapsed]):
                    proposal, measured = self.learning.infer(net, np.zeros(12), enabled=True)
                    self.assertEqual(proposal, expected if 0 <= elapsed <= 20000000 else None)
                    self.assertEqual(measured, elapsed / 1e6)
                    forward.assert_called_once()
                    count += 1
        self.assertEqual(count, registration['syntheticScoreTimingCases'])

    def test_invalid_representations_and_disabled_call_suppression(self):
        net = self.learning.Network(11)
        invalid = lambda n: [None, [0.] * n, np.zeros(n, dtype=bool), np.zeros(n, dtype=complex),
                             np.zeros(n, dtype=object), np.zeros(n - 1), np.zeros((1, n)),
                             np.full(n, np.nan), np.full(n, np.inf),
                             np.ma.array(np.zeros(n), mask=False), np.ma.array(np.zeros(n), mask=True)]
        for enabled in (False, None, 1, 'true', np.bool_(True), np.array([True])):
            with patch.object(net, 'forward') as forward, patch.object(self.learning.time, 'perf_counter_ns', side_effect=[0, 1]):
                self.assertIsNone(self.learning.infer(net, np.zeros(12), enabled=enabled)[0])
                forward.assert_not_called()
        for observation in invalid(12):
            with patch.object(net, 'forward') as forward, patch.object(self.learning.time, 'perf_counter_ns', side_effect=[0, 1]):
                self.assertIsNone(self.learning.infer(net, observation, enabled=True)[0])
                forward.assert_not_called()
        for output in invalid(3):
            with patch.object(net, 'forward', return_value=output), patch.object(self.learning.time, 'perf_counter_ns', side_effect=[0, 1]):
                self.assertIsNone(self.learning.infer(net, np.zeros(12), enabled=True)[0])

    def test_late_call_order_and_post_measurement_validation(self):
        events = []; clock = iter((0, self.fixtures['entries'][0]['elapsedNanoseconds']))
        def now():
            events.append('clock'); return next(clock)
        def forward(observation):
            events.append('forward_enter'); events.append('forward_return')
            return np.array([0., 0., 1.])
        original = self.learning._finite_real_vector
        def validate(value, width):
            events.append('observation' if width == 12 else 'output')
            return original(value, width)
        net = self.learning.Network(11)
        with patch.object(net, 'forward', side_effect=forward), \
             patch.object(self.learning.time, 'perf_counter_ns', side_effect=now), \
             patch.object(self.learning, '_finite_real_vector', side_effect=validate):
            proposal, elapsed = self.learning.infer(net, np.zeros(12), enabled=True)
        self.assertIsNone(proposal); self.assertEqual(elapsed, 25.)
        self.assertEqual(events, ['clock', 'observation', 'forward_enter', 'forward_return', 'clock', 'output'])

    def test_exception_scope_and_unchanged_network_parity(self):
        net = self.learning.Network(11)
        with patch.object(net, 'forward', side_effect=ValueError('fixture')), \
             patch.object(self.learning.time, 'perf_counter_ns', side_effect=[0, 1]):
            self.assertIsNone(self.learning.infer(net, np.zeros(12), enabled=True)[0])
        with patch.object(net, 'forward', side_effect=KeyboardInterrupt('fixture')), \
             patch.object(self.learning.time, 'perf_counter_ns', return_value=0), self.assertRaises(KeyboardInterrupt):
            self.learning.infer(net, np.zeros(12), enabled=True)
        for observation in (np.zeros(12), np.ones(12), -np.ones(12)):
            scores = net.forward(observation).tolist()
            expected = (-.25, 0., .25)[scores.index(max(scores))]
            with patch.object(self.learning.time, 'perf_counter_ns', side_effect=[0, 1]):
                self.assertEqual(self.learning.infer(net, observation, enabled=True), (expected, 1e-6))


class OPEAlgebraTests(unittest.TestCase):
    def setUp(self):
        import sys
        sys.path.insert(0, str(ROOT / 'scripts/research'))
        import sequential_evaluation
        self.evaluation = sequential_evaluation
        self.source = (ROOT / ope_algebra.SOURCE).read_text()
        self.fixtures = read_json(ROOT / 'formal/research/ope-counterexamples.json')
        self.registration = read_json(ROOT / ope_algebra.REGISTRATION)

    def test_certificates_and_witness(self):
        result = ope_algebra.check_ope(self.fixtures, self.registration)
        self.assertEqual(len(result['smt']), 3)
        self.assertEqual(result['queries'], 8)
        self.assertEqual(result['premiseChecks'], 8)
        self.assertEqual(result['counterexample']['result'], 'sat')
        self.assertFalse(result['runtimeRefinement'])
        self.assertFalse(result['statisticalReliabilityVerified'])

    def test_source_mutants_fail(self):
        changes = [('np.cumprod(pi / b, axis=1)', 'pi / b'),
                   ('gamma ** np.arange(r.shape[1])', 'np.ones(r.shape[1])'),
                   ('gamma * v[:, 1:] - q', 'v[:, 1:] - q'),
                   ('gamma * v[:, 1:] - q', 'gamma * v[:, 1:]'),
                   ('v[:, 0] + np.sum', 'np.sum'),
                   ('w.sum()**2 / (w @ w)', 'w.sum() / (w @ w)'),
                   ('w.sum()**2 / (w @ w)', 'w.sum()**2 / w.sum()'),
                   ('w @ returns / w.sum()', 'w @ returns / (w @ w)'),
                   ('np.any(v[:, -1] != 0)', 'np.any(v[:, -1] < 0)'),
                   ('"reliable": False', '"reliable": True'),
                   ('"weightClipping": "none"', '"weightClipping": "one"')]
        for old, new in changes:
            with self.subTest(old=old):
                self.assertIn(old, self.source)
                with self.assertRaises((RuntimeError, ValueError)):
                    ope_algebra.check_ope(self.fixtures, self.registration, self.source.replace(old, new))

    def test_fixture_and_registration_mutants_fail(self):
        for key, value in [('expectedESS', 2), ('expectedNonzero', 0), ('horizon', 5),
                           ('targetProbabilityHex', '0x1.0000000000000p-99')]:
            altered = copy.deepcopy(self.fixtures); altered['entries'][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                ope_algebra.check_witness(altered, self.registration)
        altered = copy.deepcopy(self.registration); altered['originalOpeReruns'] = 1
        with self.assertRaises(ValueError):
            ope_algebra.check_witness(self.fixtures, altered)

    def test_unknown_and_vacuous_results_fail(self):
        class Unknown:
            def set(self, **kwargs): pass
            def add(self, *args): pass
            def check(self): return z3.unknown
            def reason_unknown(self): return 'registered synthetic unknown'
        with patch.object(ope_algebra.z, 'Solver', Unknown):
            with self.assertRaises(RuntimeError):
                ope_algebra.check_witness(self.fixtures, self.registration)
            with self.assertRaises(RuntimeError):
                ope_algebra.check_real(ope_algebra.extract(self.source))
        for premise, conclusion in [(z3.BoolVal(False), z3.BoolVal(True)),
                                    (z3.BoolVal(True), z3.BoolVal(False))]:
            with self.assertRaises(RuntimeError):
                ope_algebra.certify('ope-mutant', premise, conclusion, [])

    def test_prescribed_underflow_public_helper(self):
        e = self.fixtures['entries'][0]; shape = (e['episodes'], e['horizon'])
        r = np.full(shape, e['reward'], dtype=float)
        a = np.zeros(shape, dtype=int); b = np.full(shape, e['behaviorProbability'], dtype=float)
        pi = np.full(shape, float.fromhex(e['targetProbabilityHex']))
        q = np.full(shape, e['q'], dtype=float); v = np.full((shape[0], shape[1]+1), e['v'], dtype=float)
        prior = np.geterr()
        with np.errstate(under='ignore'):
            weights = np.cumprod(pi/b, axis=1)[:, -1]
            self.assertEqual([x.hex() for x in weights], [e['expectedFinalWeightHex']]*2)
            result = self.evaluation.ope_estimates(r, a, b, pi, q, v, e['gamma'])
        self.assertEqual(result['effectiveSampleSize'], e['expectedESS'])
        self.assertEqual(result['nonzeroTrajectories'], e['expectedNonzero'])
        self.assertEqual(result['weightedIS'], e['expectedWIS'])
        self.assertFalse(result['reliable'])
        with np.errstate(under='raise'), self.assertRaisesRegex(ValueError, 'non-finite OPE arithmetic'):
            self.evaluation.ope_estimates(r, a, b, pi, q, v, e['gamma'])
        self.assertEqual(np.geterr(), prior)

    def test_two_weight_real_reference_grid(self):
        from fractions import Fraction as F
        from itertools import product
        cases = 0
        for weights in product(self.registration['weightGrid'], repeat=2):
            if not sum(weights): continue
            for returns in product(self.registration['returnGrid'], repeat=2):
                r = np.array(returns, dtype=float).reshape(2, 1)
                b = np.full((2, 1), .25); pi = np.array(weights, dtype=float).reshape(2, 1)*.25
                result = self.evaluation.ope_estimates(r, np.zeros((2, 1), dtype=int), b, pi,
                                                     np.zeros((2, 1)), np.zeros((2, 2)), 1.)
                ess = F(sum(weights)**2, sum(x*x for x in weights))
                wis = F(sum(w*r for w, r in zip(weights, returns)), sum(weights))
                self.assertAlmostEqual(result['effectiveSampleSize'], float(ess), delta=1e-14)
                self.assertAlmostEqual(result['weightedIS'], float(wis), delta=1e-14)
                self.assertGreaterEqual(result['effectiveSampleSize'], 1)
                self.assertLessEqual(result['effectiveSampleSize'], 2)
                cases += 1
        self.assertEqual(cases, 135)

    def test_dr_telescoping_grid_and_assumption(self):
        from fractions import Fraction as F
        cases = 0
        for size in range(1, 7):
            for gamma in self.registration['discountGrid']:
                r = np.array([(-1)**i*(i+1) for i in range(size)], dtype=float).reshape(1, size)
                v = np.array(list(range(1, size+1))+[0], dtype=float).reshape(1, size+1)
                args = (r, np.zeros((1, size), dtype=int), np.full((1, size), .5), np.full((1, size), .5))
                result = self.evaluation.ope_estimates(*args, v[:, :-1], v, gamma)
                reference = sum(F(int(x))*F(gamma)**i for i, x in enumerate(r[0]))
                self.assertAlmostEqual(result['doublyRobust'], float(reference), delta=1e-14)
                # pi=b alone does not force pathwise DR=return when Q differs from V.
                different = self.evaluation.ope_estimates(*args, v[:, :-1]+1, v, gamma)
                self.assertNotEqual(different['doublyRobust'], float(reference))
                cases += 1
        self.assertEqual(cases, 18)

    def test_current_six_step_weight_bounds(self):
        from itertools import product
        patterns = list(product((0., 1.), repeat=6))
        weights = np.cumprod(np.array(patterns)/(1/3), axis=1)[:, -1]
        self.assertEqual(len(patterns), 64)
        self.assertEqual(set(weights), {0., 729.})
        # Extract the actual ESS expression, not a reimplemented numerical formula.
        node = ope_algebra.extract(self.source)['ess']
        code = compile(ast.fix_missing_locations(ast.Expression(body=node)), 'source-ess', 'eval')
        for count in range(201):
            w = np.array([729.]*count+[0.]*(200-count))
            self.assertEqual(eval(code, {'w': w, 'float': float}), float(count))


class ExactESSV2Tests(unittest.TestCase):
    def setUp(self):
        import sys
        sys.path.insert(0, str(ROOT / 'scripts/research'))
        import ess_rational_v2
        self.kernel = ess_rational_v2
        self.source = (ROOT / ess_v2.SOURCE).read_text()
        self.registration = read_json(ROOT / ess_v2.REGISTRATION)

    @staticmethod
    def reference(weights):
        from fractions import Fraction
        pairs = [w.as_integer_ratio() for w in weights]
        denominator = max(d for _, d in pairs)
        integers = [n*(denominator//d) for n, d in pairs]
        squares = sum(n*n for n in integers)
        return Fraction(sum(integers)**2, squares) if squares else Fraction(0)

    def test_certificates_and_bounded_model(self):
        result = ess_v2.check_ess(self.registration)
        self.assertEqual(len(result['smt']), 2)
        self.assertEqual(result['queries'], 3)
        self.assertEqual(result['model']['states'], 66309)
        self.assertEqual(result['model']['transitions'], 99718)
        self.assertEqual(result['model']['maxShortestDepth'], 515)
        self.assertFalse(result['model']['runtimeRefinement'])
        self.assertFalse(result['statisticalReliabilityVerified'])

    def test_source_mutants_fail(self):
        changes = [('total + weight', 'total + weight * 2'),
                   ('squares + weight * weight', 'squares + weight'),
                   ('squares + weight * weight', 'squares + total * weight'),
                   ('total * total / squares', 'Fraction(1)'),
                   ('total * total / squares', 'total / squares'),
                   ('Fraction(0) if squares == 0', 'Fraction(1) if squares == 0'),
                   ('enabled: object = False', 'enabled: object = True'),
                   ('type(weights) is not tuple', 'type(weights) is not list'),
                   ('MAX_ROWS = 256', 'MAX_ROWS = 257'),
                   ('type(value) is float', 'isinstance(value, float)'),
                   ('isfinite(value) and value >= 0', 'value >= 0'),
                   ('Fraction.from_float(value)', 'Fraction(int(value))'),
                   ('return Fraction(0) if squares == 0 else total * total / squares',
                    'return float(Fraction(0) if squares == 0 else total * total / squares)')]
        for old, new in changes:
            with self.subTest(old=old):
                self.assertIn(old, self.source)
                with self.assertRaises((ValueError, RuntimeError)):
                    ess_v2.check_ess(self.registration, self.source.replace(old, new))

    def test_fail_closed_solver_model_registration_and_isolation(self):
        class Unknown:
            def set(self, **kwargs): pass
            def add(self, *args): pass
            def check(self): return z3.unknown
            def reason_unknown(self): return 'registered synthetic unknown'
        with patch.object(ess_v2.z, 'Solver', Unknown), self.assertRaises(RuntimeError):
            ess_v2.check_arithmetic(ess_v2.extract(self.source))
        original = ess_v2.transitions
        def bypass(state, maximum):
            yield from original(state, maximum)
            if state[0] == 'validate' and state[2] == 0:
                yield ('accumulate', state[1], 0)
        with patch.object(ess_v2, 'transitions', bypass), self.assertRaises(RuntimeError):
            ess_v2.check_publication(4)
        def early(state, maximum):
            yield from original(state, maximum)
            if state[0] == 'accumulate' and state[2] == 0:
                yield ('positive', 0, 0)
        with patch.object(ess_v2, 'transitions', early), self.assertRaises(RuntimeError):
            ess_v2.check_publication(4)
        registration = copy.deepcopy(self.registration); registration['maxRows'] = 257
        with self.assertRaises(ValueError): ess_v2.check_ess(registration)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); folder = root/'scripts/research'; folder.mkdir(parents=True)
            (folder/'consumer.py').write_text('from ess_rational_v2 import effective_sample_size_v2\n')
            with patch.object(ess_v2, 'ROOT', root), self.assertRaises(ValueError):
                ess_v2.check_isolation()

    def test_default_admission_and_no_coercion(self):
        from fractions import Fraction
        class Hostile:
            def __float__(self): raise RuntimeError('coercion')
            def __iter__(self): raise RuntimeError('iteration')
            def __len__(self): raise RuntimeError('length')
            def __eq__(self, other): raise RuntimeError('equality')
        class TupleSubclass(tuple): pass
        class FloatSubclass(float): pass
        class StringSubclass(str): pass
        call = self.kernel.effective_sample_size_v2
        for flag in (False, None, 0, 1, 'true', np.bool_(True), Hostile()):
            self.assertIsNone(call(Hostile(), enabled=flag))
        self.assertIsNone(call(Hostile()))
        for version in (None, 2, b'ess-rational-v2', 'bad', StringSubclass(self.kernel.VERSION), Hostile()):
            self.assertIsNone(call(Hostile(), enabled=True, version=version))
        bad = [None, [], [1.], (), (1.,)*257, np.ones(2), iter([1.]), Hostile(),
               TupleSubclass((1.,)), (FloatSubclass(1.),), (True,), (1,), (Fraction(1),),
               (np.float64(1.),), (float('nan'),), (float('inf'),), (-float('inf'),), (-1.,), (Hostile(),)]
        with patch.object(self.kernel.Fraction, 'from_float') as convert:
            for weights in bad:
                with self.subTest(type=type(weights)):
                    self.assertIsNone(call(weights, enabled=True))
            convert.assert_not_called()
        self.assertEqual(call((0., -0.), enabled=True), Fraction(0))

    def test_every_bad_element_position_precedes_arithmetic(self):
        call = self.kernel.effective_sample_size_v2
        with patch.object(self.kernel.Fraction, 'from_float') as convert:
            for position in range(256):
                weights = [1.]*256; weights[position] = float('nan')
                self.assertIsNone(call(tuple(weights), enabled=True))
            convert.assert_not_called()

    def test_extremes_and_preserved_counterexample(self):
        from fractions import Fraction
        call = self.kernel.effective_sample_size_v2
        fixture = read_json(ROOT/'formal/research/ope-counterexamples.json')['entries'][0]
        weight = float.fromhex(fixture['expectedFinalWeightHex'])
        self.assertEqual(call((weight, weight), enabled=True), Fraction(2))
        extremes = [float.fromhex(v) for v in self.registration['extremeHexValues']]
        for n in (1, 2, 200, 256):
            for value in (0., *extremes):
                weights = (value,)*n
                expected = Fraction(n if value else 0)
                self.assertEqual(call(weights, enabled=True), expected)
            weights = tuple(extremes[i % len(extremes)] for i in range(n))
            self.assertEqual(call(weights, enabled=True), self.reference(weights))
        # Exact input arithmetic cannot reconstruct positive values lost upstream.
        self.assertEqual(call((0., 0.), enabled=True), Fraction(0))

    def test_fixed_grid_against_integer_reference(self):
        from itertools import product
        count = 0
        for n in self.registration['gridLengths']:
            for weights in product(map(float, self.registration['grid']), repeat=n):
                result = self.kernel.effective_sample_size_v2(weights, enabled=True)
                self.assertEqual(result, self.reference(weights))
                self.assertEqual(result == 0, all(w == 0 for w in weights))
                if result:
                    self.assertGreaterEqual(result, 1); self.assertLessEqual(result, n)
                count += 1
        self.assertEqual(count, 340)

    def test_seeded_scale_permutation_and_replay_properties(self):
        import math
        import random
        rng = random.Random(self.registration['randomSeed'])
        call = self.kernel.effective_sample_size_v2
        for case in range(self.registration['randomCases']):
            size = rng.randint(*self.registration['randomLengthRange'])
            weights = tuple(math.ldexp(1., rng.randint(*self.registration['randomPowerOfTwoExponentRange'])) for _ in range(size))
            before = tuple(w.hex() for w in weights)
            result = call(weights, enabled=True)
            self.assertEqual(result, self.reference(weights))
            self.assertEqual(call(weights, enabled=True), result)
            shuffled = list(weights); rng.shuffle(shuffled)
            self.assertEqual(call(tuple(shuffled), enabled=True), result)
            for exponent in self.registration['scalePowers']:
                self.assertEqual(call(tuple(math.ldexp(w, exponent) for w in weights), enabled=True), result)
            self.assertEqual(tuple(w.hex() for w in weights), before)
            self.assertIsNone(call(weights))



class FundingBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.reg = funding_boundary.registration()
        self.source = (ROOT/funding_boundary.SOURCE).read_text()
        self.expr, self.loop = funding_boundary.extract(self.source, self.reg['sourceFunctionASTSha256'])
        self.apply = funding_boundary.compiled_loop(self.loop)

    def test_conditional_certificates_and_prescribed_witnesses(self):
        result = funding_boundary.check_funding()
        self.assertEqual(len(result['smt']), 2)
        self.assertEqual(result['grid']['rows'], 4910)
        fixture = read_json(ROOT/'formal/research/funding-counterexamples.json')
        self.assertEqual(result['witnesses'], fixture['outcomes'])

    def test_source_mutants_rejected(self):
        changes = [('side="left"', 'side="right"'), ('if j >= len(f)', 'if j > len(f)'),
                   ('f[j] +=', 'f[j] ='), ('closes = times + spec["intervalMilliseconds"] - 1',
                                           'closes = times + spec["intervalMilliseconds"]'),
                   ('pd.read_csv(BytesIO(panel_bytes))', 'pd.read_csv(panel)'),
                   ('hashlib.sha256(panel_bytes).hexdigest() != spec["panelSha256"]', 'False')]
        for old,new in changes:
            self.assertIn(old,self.source)
            with self.subTest(change=new), self.assertRaises(ValueError):
                funding_boundary.extract(self.source.replace(old,new), self.reg['sourceFunctionASTSha256'])

    def test_grid_formula_and_registration_mutants_fail_smt(self):
        spec = read_json(ROOT/'research-notes/registrations/sequential-control-screen-v1.json')['data']
        changed = copy.deepcopy(self.expr)
        changed['closes'] = ast.parse('times + spec["intervalMilliseconds"]', mode='eval').body
        with self.assertRaises(RuntimeError): funding_boundary.grid_certificate(changed,spec)
        for field,delta in [('rowsPerSymbol',1),('endOpenTime',1),('startOpenTime',-1)]:
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                funding_boundary.grid_certificate(self.expr,dict(spec,**{field:spec[field]+delta}))

    def test_synthetic_linear_reference(self):
        reg = self.reg['syntheticGrid']; closes = [9,19,29]
        count = 0
        for t in range(reg['eventStart'],reg['eventStopInclusive']+1):
            index = next((i for i,c in enumerate(closes) if t <= c),len(closes))
            if index == len(closes):
                with self.assertRaisesRegex(ValueError,'beyond development'):
                    funding_boundary.apply_events(self.apply,closes,[(t,.25,4.)])
            else:
                f = funding_boundary.apply_events(self.apply,closes,[(t,.25,4.)])
                expected = np.zeros(3); expected[index] = 1.
                np.testing.assert_array_equal(f,expected)
            count += 1
        self.assertEqual(count,32)

    def test_all_registered_grid_neighbors(self):
        spec = read_json(ROOT/'research-notes/registrations/sequential-control-screen-v1.json')['data']
        closes = np.arange(spec['startOpenTime'],spec['endOpenTime']+1,
                           spec['intervalMilliseconds'],dtype=np.int64)+spec['intervalMilliseconds']-1
        count = 0
        for i,c in enumerate(closes):
            for delta in self.reg['boundaryOffsets']:
                t = int(c)+delta
                # Independent linear reference, not another searchsorted call.
                expected = next((j for j,end in enumerate(closes) if t <= end),len(closes))
                if expected == len(closes):
                    with self.assertRaises(ValueError):
                        funding_boundary.apply_events(self.apply,closes,[(t,.25,4.)])
                else:
                    f = funding_boundary.apply_events(self.apply,closes,[(t,.25,4.)])
                    self.assertEqual(f[expected],1.)
                    self.assertEqual(np.count_nonzero(f),1)
                count += 1
        self.assertEqual(count,14730)

    def test_signed_and_zero_coefficients_preserve_bucket_accounting(self):
        f = funding_boundary.apply_events(self.apply,[9,19,29],
                                         [(-1,.5,4.),(10,.5,4.),(11,-.25,4.),(19,0.,4.),(20,-.5,4.)])
        np.testing.assert_array_equal(f,[2.,1.,-2.])

    def test_unknown_solver_fails(self):
        class Unknown:
            def set(self, **kw): pass
            def add(self, *args): pass
            def check(self): return z3.unknown
            def reason_unknown(self): return 'synthetic unknown'
        with patch.object(funding_boundary.z,'Solver',Unknown), self.assertRaises(RuntimeError):
            funding_boundary.endpoint_certificate()
        with patch.object(funding_boundary.z,'Solver',Unknown), self.assertRaises(ValueError):
            funding_boundary.overflow_evidence(self.apply,self.reg)

    def test_registration_drift_fails(self):
        with patch.object(funding_boundary,'REGISTRATION_SHA256','0'*64), self.assertRaises(ValueError):
            funding_boundary.registration()

if __name__ == '__main__':
    unittest.main()
