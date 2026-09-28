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


if __name__ == '__main__':
    unittest.main()
