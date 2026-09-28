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


if __name__ == '__main__':
    unittest.main()
