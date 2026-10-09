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
import replay_order
import replay_cutoff
import reward_accounting
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
                   ('expected_identity = _provenance_identity(expected_provenance)', 'expected_identity = None'),
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


class ReplayOrderTests(unittest.TestCase):
    def setUp(self):
        self.reg = replay_order.registration()
        self.source = (ROOT/replay_order.SOURCE).read_text()
        self.nodes = replay_order.extract(self.source,self.reg['sourceFunctionASTSha256'])

    def test_source_bound_certificates(self):
        result = replay_order.check_replay_order()
        self.assertEqual(result['smt'],{'F-RL-REPLAY-DUE':'unsat'})
        self.assertEqual(result['model']['states'],2876)
        self.assertEqual(result['model']['transitions'],3474)
        self.assertEqual(result['model']['distinctInitialStates'],84)
        self.assertEqual(result['model']['progressBound'],48)
        self.assertFalse(result['model']['runtimeRefinement'])

    def test_source_mutants_fail(self):
        changes = [('self.t + 1 + self.execution.extra_delay','self.t + self.execution.extra_delay'),
                   ('self.pending[0] <= self.t','self.pending[0] >= self.t'),
                   ('self.t += 1','self.t += 2'),
                   ('self.equity += gross + funding','self.equity += gross'),
                   ('self.pending = None','self.pending = (self.t, 0.0)'),
                   ('self._trade(0.0, terminal=True)','self._trade(0.0, terminal=False)'),
                   ('if isfinite(self.equity) and self.equity > 0:','if True:'),
                   ('self.extra_delay not in (0, 1)','self.extra_delay not in (-1, 0, 1)')]
        for old,new in changes:
            self.assertIn(old,self.source)
            with self.subTest(change=new), self.assertRaises(ValueError):
                replay_order.extract(self.source.replace(old,new),self.reg['sourceFunctionASTSha256'])

    def test_due_and_guard_mutants_fail_smt(self):
        due,guard = self.nodes
        with self.assertRaises(RuntimeError):
            replay_order.due_certificate(ast.parse('self.t + self.execution.extra_delay',mode='eval').body,guard)
        for text in ('self.pending[0] <= self.t',
                     'self.failure is None and self.pending is not None',
                     'self.pending is not None and self.pending[0] <= self.t'):
            with self.subTest(guard=text), self.assertRaises(RuntimeError):
                replay_order.due_certificate(due,ast.parse(text,mode='eval').body)

    def test_model_bypasses_and_stall_fail(self):
        def changed(kind):
            def transition(s):
                if kind == 'gate' and s.phase == 'guard':
                    return [('bypass',replay_order.replace(s,phase='mark',observed=True,bar_valid=True))]
                if kind == 'risk' and s.phase == 'risk':
                    return [('bypass',replay_order.replace(s,phase='fill_check'))]
                if kind == 'early_fill' and s.phase == 'fill_check' and s.pending > s.t:
                    return [('target_fill',replay_order.replace(s,phase='postfill_risk',pending=-1,fills=s.fills+1))]
                if kind == 'cancel' and s.phase == 'cancel':
                    return [('bypass',replay_order.replace(s,phase='liquidate_check'))]
                if kind == 'terminal' and s.phase == 'after_row' and s.terminal:
                    return [('next_bar',replay_order.replace(s,phase='bar'))]
                if kind == 'stall' and s.phase == 'risk': return [('stall',s)]
                return replay_order.successors(s)
            return transition
        for kind in ('gate','risk','early_fill','cancel','terminal','stall'):
            with self.subTest(mutation=kind), self.assertRaises(RuntimeError):
                replay_order.check_model(self.reg,changed(kind))

    def test_actual_grid_and_preserved_terminal_round_trip(self):
        result = replay_order.check_conformance(self.reg)
        self.assertEqual(result,read_json(ROOT/'formal/research/replay-order-fixtures.json')['conformance'])
        self.assertEqual(result['gridTraces'],486)
        self.assertEqual(result['terminalRoundTrip']['costs'],.0005)
        self.assertEqual(result['scenarios']['insolvency']['units'],.0025)
        self.assertEqual(result['scenarios']['solvent_risk']['units'],0.)

    def test_trace_order_mutants_are_not_model_traces(self):
        state = replay_order.State('guard',1,1,0,-1)
        for trace in (['target_fill','mark_old_and_advance','liquidate','append_row'],
                      ['mark_old_and_advance','target_fill','append_row','liquidate'],
                      ['mark_old_and_advance','target_fill','liquidate','append_row','append_row']):
            self.assertEqual(replay_order.matching_states(state,trace),set())

    def test_unknown_fails_and_registration_drift_rejects(self):
        class Unknown:
            def set(self,**kw): pass
            def add(self,*args): pass
            def check(self): return z3.unknown
            def reason_unknown(self): return 'synthetic unknown'
        with patch.object(replay_order.z,'Solver',Unknown), self.assertRaises(RuntimeError):
            replay_order.due_certificate(*self.nodes)
        with patch.object(replay_order,'REGISTRATION_SHA256','0'*64), self.assertRaises(ValueError):
            replay_order.registration()

class ReplayCutoffTests(unittest.TestCase):
    def setUp(self):
        self.reg = replay_cutoff.registration()
        self.source = (ROOT/self.reg['orderingModel']).read_text()
        self.guard = replay_cutoff.extract(self.source,self.reg['orderingModelSha256'])

    def test_cutoff_certificate_and_complete_graph(self):
        result = replay_cutoff.check_replay_cutoff()
        self.assertEqual(result['smt'],{'F-RL-REPLAY-CUTOFF':'unsat'})
        self.assertEqual(result['model']['states'],3596)
        self.assertEqual(result['model']['transitions'],4340)
        self.assertEqual(result['model']['distinctInitialStates'],102)
        self.assertEqual(result['model']['progressBound'],48)
        self.assertFalse(result['model']['runtimeRefinement'])
        self.assertEqual(result['lifts']['reachableClassSevenStates'],720)
        self.assertEqual(result['lifts']['successorComparisons'],3600)
        fixtures = read_json(ROOT/'formal/research/replay-cutoff-fixtures.json')
        self.assertEqual(result['badAbstraction'],fixtures['badAbstraction'])
        self.assertEqual(result['conformance'],fixtures['conformance'])

    def test_cutoff_six_fails_universal_query(self):
        with self.assertRaises(RuntimeError):
            replay_cutoff.cutoff_certificate(self.guard,6)

    def test_source_dependency_drift_rejected_even_after_rehash(self):
        import hashlib
        mutants = [('s.failed or s.t == s.limit','s.failed or s.t >= s.limit'),
                   ('s.pending < 0','s.pending < s.limit'),
                   ('pending=-1,fills=s.fills+1','pending=-1,fills=s.fills+1,limit=6'),
                   ('replace(s,phase=phase,**kw)','replace(s,phase=phase,limit=6,**kw)')]
        for old,new in mutants:
            self.assertIn(old,self.source)
            changed = self.source.replace(old,new)
            with self.subTest(change=new), self.assertRaises(ValueError):
                replay_cutoff.extract(changed,hashlib.sha256(changed.encode()).hexdigest())
        with self.assertRaises(ValueError):
            replay_cutoff.extract(self.source+'\n',self.reg['orderingModelSha256'])

    def test_lift_mutant_with_new_limit_dependence_rejected(self):
        def mutant(state):
            if state.limit > 7 and state.phase == 'terminal_check':
                return [('incorrect_terminal',replay_order.replace(state,phase='return'))]
            return replay_order.successors(state)
        with self.assertRaises(RuntimeError):
            replay_cutoff.check_lifts(self.reg,mutant)

    def test_nonterminal_six_bar_call_and_forced_terminal_mismatch(self):
        actual = replay_order.conformance_case(horizon=6,remaining=7)
        replay_cutoff.match_actual(actual,6,0)
        self.assertFalse(actual['done'])
        self.assertNotIn('liquidate',actual['trace'])
        wrong = dict(actual,done=True)
        with self.assertRaises(RuntimeError):
            replay_cutoff.match_actual(wrong,6,0)

    def test_unknown_and_registration_drift_fail(self):
        class Unknown:
            def set(self,**kw): pass
            def add(self,*args): pass
            def check(self): return z3.unknown
            def reason_unknown(self): return 'synthetic unknown'
        with patch.object(replay_cutoff.z,'Solver',Unknown), self.assertRaises(RuntimeError):
            replay_cutoff.cutoff_certificate(self.guard)
        with patch.object(replay_cutoff,'REGISTRATION_SHA256','0'*64), self.assertRaises(ValueError):
            replay_cutoff.registration()


class RewardAccountingTests(unittest.TestCase):
    def setUp(self):
        self.reg = reward_accounting.registration()
        self.sources = [(ROOT/p).read_text() for p in reward_accounting.SOURCES]
        self.nodes = reward_accounting.extract(self.sources,self.reg['sourceFunctionASTSha256'])

    def test_source_certificates_and_registered_grid(self):
        result = reward_accounting.check_reward_accounting()
        self.assertEqual(len(result['smt']),3)
        self.assertEqual(result['satUnsatPairs'],7)
        self.assertEqual(result['nonadditivity']['status'],'refuted')
        self.assertEqual(result['conformance']['gridEpisodes'],1944)
        self.assertTrue(result['conformance']['cashResidualWithinTolerance'])
        self.assertTrue(result['conformance']['rewardResidualWithinTolerance'])
        expected = read_json(ROOT/'formal/research/reward-accounting-fixtures.json')
        self.assertEqual(result['nonadditivity'],expected['nonadditivity'])
        self.assertEqual(result['conformance']['failureScenarios'],expected['failureScenarios'])
        self.assertEqual(result['conformance']['nonadditiveReplay'],expected['nonadditiveReplay'])

    def test_arithmetic_mutants_fail_universal_queries(self):
        mutations = [('reward','100 * ((self.equity - before) / self.equity - self.execution.risk_penalty * penalty)'),
                     ('reward','100 * ((self.equity - before) / before + self.execution.risk_penalty * penalty)'),
                     ('mark','gross + funding - 1'),
                     ('debit',"sum(terms[k] for k in ('fee', 'spread', 'slippage'))"),
                     ('debit',"2 * sum(terms[k] for k in ('fee', 'spread', 'slippage', 'impact'))"),
                     ('merge','costs[k]'),('net','self.equity / old_equity + 1'),
                     ('rowPenalty','self.execution.risk_penalty * exposure**2 * x[5]**2'),
                     ('metric','equity[-1]'),('initialPenalty','1')]
        for key,value in mutations:
            nodes = dict(self.nodes); nodes[key]=ast.parse(value,mode='eval').body
            with self.subTest(key=key,expression=value), self.assertRaises(RuntimeError):
                reward_accounting.certificates(nodes)

    def test_source_drift_rejected(self):
        for index,old,new in [(0,'self.equity += gross + funding','self.equity += gross'),
                              (0,'penalty += exposure**2 * x[5]**2','penalty += exposure * x[5]**2'),
                              (1,'"netReturn": equity[-1] - 1','"netReturn": sum(r)'),
                              (1,'rel_tol=1e-10','rel_tol=1')]:
            sources=list(self.sources); self.assertIn(old,sources[index]); sources[index]=sources[index].replace(old,new)
            with self.subTest(change=new), self.assertRaises(ValueError):
                reward_accounting.extract(sources,self.reg['sourceFunctionASTSha256'])

    def test_report_rejects_cash_or_return_tampering(self):
        from sequential_evaluation import economic
        for field,delta in [('fee',.01),('net',.01),('equity',.01)]:
            env = reward_accounting.replay_case(.25,1,0,3,1.,0.,'alternating',.01)
            reward_accounting.check_episode(env,.25,1e-11)
            env.rows[1][field]+=delta
            with self.subTest(field=field), self.assertRaises(ValueError): economic(env)

    def test_nonadditive_witness_reports_economic_loss(self):
        env = reward_accounting.replay_case(.25,1,0,3,0.,0.,'flat',0.)
        env.prices[25:28]=[100.,110.,99.]
        result=reward_accounting.check_episode(env,.25,1e-11)
        self.assertAlmostEqual(sum(result['rewards']),0.,places=11)
        self.assertAlmostEqual(result['netReturn'],-1/1600,places=11)
        self.assertGreater(abs(sum(result['rewards'])-100*result['netReturn']),.06)

    def test_unsupported_translation_fails_closed(self):
        for value in ('unknown(1)', 'x**3', "sum(terms[k] for k in ('fee', unknown))", 'sum(terms[k] for k in ())'):
            with self.subTest(expression=value), self.assertRaises(ValueError):
                reward_accounting.real(ast.parse(value,mode='eval').body,{'x':z3.RealVal(1)})

    def test_unknown_and_registration_drift_fail(self):
        class Unknown:
            def set(self,**kw): pass
            def add(self,*args): pass
            def check(self): return z3.unknown
            def reason_unknown(self): return 'synthetic unknown'
        with patch.object(reward_accounting.z,'Solver',Unknown), self.assertRaises(RuntimeError):
            reward_accounting.certificates(self.nodes)
        with patch.object(reward_accounting.z,'Solver',Unknown), self.assertRaises(RuntimeError):
            reward_accounting.nonadditive_certificate(self.nodes)
        with patch.object(reward_accounting,'REGISTRATION_SHA256','0'*64), self.assertRaises(ValueError):
            reward_accounting.registration()


class ObligationClosureTests(unittest.TestCase):
    def setUp(self):
        self.ledger = read_json(ROOT / 'formal/research/proof-ledger.json')

    def test_real_affected_scope_obligations_closed_others_remain(self):
        from verify import validate_obligations
        self.assertEqual(validate_obligations(self.ledger['missionObligations'], self.ledger['entries']), 18)
        self.assertEqual([o['number'] for o in self.ledger['missionObligations'] if o['status']=='exhaustively_checked'], [2,3,5,7,8,11,12,15,23,24,25,26,28,29,30,31])
        self.assertEqual([o['number'] for o in self.ledger['missionObligations'] if o['status']=='smt_verified'], [9])
        self.assertEqual([o['number'] for o in self.ledger['missionObligations'] if o['status']=='model_checked'], [14, 16, 35])

    def test_certified_completion_is_reachable_but_not_economic_acceptance(self):
        from verify import acceptance_summary
        # Validator-only fixture: no assertion about other trading requirements.
        obligations = copy.deepcopy(self.ledger['missionObligations'])
        for item in obligations:
            item.update(status='exhaustively_checked', blockers=[],
                        requiredCertificates=['F-RL-DEFAULT-PATH'], evidenceRequirements=['F-RL-DEFAULT-PATH'])
        contracts = [{k:o[k] for k in ('number','claim','scope','closureCriteria','requiredCertificates','implementationFiles')} for o in obligations]
        for contract in contracts:
            contract['closureClass'] = 'exhaustively_checked'
        result = acceptance_summary(obligations, self.ledger['entries'],
                                    {'F-RL-DEFAULT-PATH': 'exhaustively_checked'},
                                    self.ledger['researchAcceptanceGates'], ROOT, contracts)
        self.assertEqual(result['openMissionObligations'], 0)
        self.assertTrue(result['formalObligationsComplete'])
        self.assertFalse(result['missionComplete'])
        # Acceptance becomes reachable only with separate reviewed, hashed evidence.
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'evidence.txt').write_text('synthetic validator fixture, not market evidence')
            gates = copy.deepcopy(self.ledger['researchAcceptanceGates'])
            for gate in gates:
                gate.update(status='satisfied', blockers=[], evidence=[{'path': 'evidence.txt',
                    'sha256': hashlib.sha256((root / 'evidence.txt').read_bytes()).hexdigest()}])
            self.assertTrue(acceptance_summary(obligations, self.ledger['entries'],
                {'F-RL-DEFAULT-PATH': 'exhaustively_checked'}, gates, root, contracts)['missionComplete'])
            (root / 'evidence.txt').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'evidence drift'):
                acceptance_summary(obligations, self.ledger['entries'],
                    {'F-RL-DEFAULT-PATH': 'exhaustively_checked'}, gates, root, contracts)

    def test_missing_wrong_and_unverified_certificates_cannot_close(self):
        from verify import validate_obligations
        for mutate in (
            lambda o: o.update(requiredCertificates=[]),
            lambda o: o.update(requiredCertificates=['F-RL-INTEGRITY'], evidenceRequirements=['F-RL-INTEGRITY']),
            lambda o: o.update(requiredCertificates=['missing']),
            lambda o: o.update(blockers=['unresolved implementation path']),
            lambda o: o.update(scope=''),
            lambda o: o.update(status='proved'),
            lambda o: o.update(status='smt_verified'),
            lambda o: o.update(requiredCertificates=['F-RL-DEFAULT-PATH','F-RL-DEFAULT-PATH']),
        ):
            data = copy.deepcopy(self.ledger['missionObligations'])
            mutate(data[30])
            with self.assertRaises(ValueError):
                validate_obligations(data, self.ledger['entries'])
        relabel = read_json(ROOT / 'formal/research/closure-counterexamples.json')['entries'][1]['witness']
        data = copy.deepcopy(self.ledger['missionObligations'])
        data[relabel['obligation'] - 1]['status'] = relabel['incorrectlyAdmittedAggregateClass']
        with self.assertRaisesRegex(ValueError, 'aggregate verification class'):
            validate_obligations(data, self.ledger['entries'])
        for reproduced in ({}, {'F-RL-DEFAULT-PATH':'unknown'}, {'F-RL-DEFAULT-PATH':'property_tested'}):
            with self.assertRaisesRegex(ValueError, 'not reproduced'):
                validate_obligations(self.ledger['missionObligations'], self.ledger['entries'], reproduced)

    def test_every_remaining_obligation_has_action_and_code(self):
        for item in self.ledger['missionObligations']:
            self.assertTrue(item['closureCriteria'])
            self.assertTrue(item['nextAction'])
            for path in item['implementationFiles']:
                self.assertTrue((ROOT / path).is_file(), path)
        witness = read_json(ROOT / 'formal/research/closure-counterexamples.json')['entries'][0]['witness']
        # Preserves the previous validator's unreachable-completion defect.
        self.assertEqual(set(witness['permittedStatuses']), set(witness['countedAsOpen']))
        self.assertEqual(witness['minimumOpenCount'], 38)

    def test_research_gate_failures_are_independent(self):
        from verify import acceptance_summary
        reproduced = {e['requirementId']:e['status'] for e in self.ledger['entries']}
        for mutate in (lambda g:g.pop(), lambda g:g[0].update(status='satisfied'),
                       lambda g:g[0].update(status='unknown'), lambda g:g[0].update(blockers=[])):
            gates = copy.deepcopy(self.ledger['researchAcceptanceGates'])
            mutate(gates)
            with self.assertRaises(ValueError):
                acceptance_summary(self.ledger['missionObligations'], self.ledger['entries'], reproduced, gates, ROOT)


class DefaultPathTests(unittest.TestCase):
    def setUp(self):
        import default_paths as d
        self.d = d
        self.sources = {p:(ROOT / p).read_text() for p in {d.HASKELL, *(p for p, _ in d.ENTRY_POINTS)}}

    def test_actual_source_paths(self):
        receipt = self.d.check_defaults(self.sources)
        self.assertEqual(receipt['booleanCases'], 14)
        self.assertEqual(receipt['haskellModes'], 2)

    def test_source_mutations_rejected(self):
        for path, name in self.d.ENTRY_POINTS:
            source = self.sources[path]
            for old, new in [('= False', '= True'), ('enabled is not True', 'enabled is True'),
                             ('if enabled is not True or', 'if enabled is not True and')]:
                with self.subTest(entry=name, mutation=new), self.assertRaises(ValueError):
                    self.d.check_python(source.replace(old, new), name)
            tree = ast.parse(source)
            fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
            fn.body.insert(0, ast.parse('dangerous_effect()').body[0])
            with self.assertRaises(ValueError):
                self.d.check_python(ast.unparse(tree), name)
        hs = self.sources[self.d.HASKELL]
        for old,new in [('defaultResearchMode = Disabled','defaultResearchMode = OfflineReplayV1'),
                        ('mode /= OfflineReplayV1 = Nothing','mode == OfflineReplayV1 = Nothing'),
                        ('= Nothing','= Just (ResearchProposal target)')]:
            with self.assertRaises(ValueError):
                self.d.check_haskell(hs.replace(old,new))
        source = self.sources[self.d.ENTRY_POINTS[0][0]]
        for old,new in [('"enabled": False','"enabled": True'),
                        ('raw = (json.dumps(value','value["enabled"] = True\n    raw = (json.dumps(value')]:
            with self.assertRaises(ValueError):
                self.d.check_saved_default(source.replace(old,new))

    def test_defaults_do_not_evaluate_opaque_arguments(self):
        import sys
        sys.path.insert(0, str(ROOT / 'scripts/research'))
        import sequential_learning as learning
        from gae_targets_v2 import step_v2, batch_v2
        from ess_rational_v2 import effective_sample_size_v2
        class Poison:
            def __getattribute__(self, name):
                raise AssertionError('disabled path read an argument: ' + name)
        poison = Poison()
        self.assertIsNone(learning.infer(poison, poison)[0])
        self.assertIsNone(step_v2(poison, poison, poison, poison, poison, poison))
        self.assertIsNone(batch_v2(poison, poison))
        self.assertIsNone(effective_sample_size_v2(poison))



class CapabilityIsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import subprocess
        import capability_isolation as c
        cls.c = c
        cls.directory = tempfile.TemporaryDirectory(prefix='trader-parser-test-')
        cls.exe = str(Path(cls.directory.name) / 'parser')
        subprocess.run(['ghc','-v0','-O0','-package','ghc-9.4.8','-package','Cabal-3.8.1.0',
                        '-outputdir',cls.directory.name,c.HELPER,'-o',cls.exe],cwd=ROOT,check=True)
        cls.parsed = c.parse_receipt(c.parser_output(executable=cls.exe))
        cls.registry = read_json(ROOT / c.REGISTRY)
        cls.source = (ROOT / c.LEARNING).read_text()

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_actual_graph_effects_schema_and_types(self):
        c = self.c
        result = c.graph_certificate(self.parsed,self.registry['executables'])
        self.assertEqual(len(result['roots']),6)
        self.assertTrue(all(v['states'] > 0 for v in result['roots'].values()))
        self.assertEqual(c.effect_certificate(self.source,self.parsed,self.registry)['parameterWrites'],0)
        emitted,required = c.artifact_keys(self.source,self.parsed,self.registry)
        self.assertEqual(c.prove_disjoint(emitted,required)['membershipBits'],12)
        self.assertTrue(c.check_types()['allRejected'])

    def test_graph_mutations_fail_with_path_or_reason(self):
        c = self.c
        mutations = [
            (lambda p:p['modules']['haskell/app/Trader/LSTM.hs']['imports'].append('Trader.Research.PolicyProposalV1'),'research import reachable'),
            (lambda p:p['roots']['trader-hs']['declared'].append('Trader.Research.PolicyProposalV1'),'declared in executable'),
            (lambda p:p['modules']['haskell/app/Main.hs']['imports'].append('Trader.Missing'),'missing local'),
            (lambda p:p['modules']['haskell/app/Main.hs']['imports'].append('Main'),'ambiguous import'),
            (lambda p:p['modules'].update({'duplicate.hs':p['modules'][c.PROPOSAL]}),'duplicate local'),
            (lambda p:p['roots'].pop('trader-hs'),'inventory drift')]
        for mutate,reason in mutations:
            parsed = copy.deepcopy(self.parsed); mutate(parsed)
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError,reason):
                c.graph_certificate(parsed,self.registry['executables'])

    def parse_fixture(self,source,cabal_suffix=''):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'haskell/app').mkdir(parents=True)
            (root / 'haskell/app/Main.hs').write_text(source)
            (root / 'haskell/trader.cabal').write_text(
                'name: fixture\nversion: 0.1.0.0\nbuild-type: Simple\ncabal-version: >=1.10\n'
                'executable fixture\n  hs-source-dirs: app\n  main-is: Main.hs\n'
                '  build-depends: base\n  default-language: Haskell2010\n' + cabal_suffix)
            return self.c.parse_receipt(self.c.parser_output(root,self.exe))

    def test_real_parser_preserves_multiline_qualified_import(self):
        parsed = self.parse_fixture('module Main where\n-- import Trader.Fake\nimport qualified\n Trader.Research.PolicyProposalV1 as P\nmain = pure ()\n')
        self.assertEqual(parsed['modules']['haskell/app/Main.hs']['imports'],['Trader.Research.PolicyProposalV1'])

    def test_unsupported_parser_and_build_constructs_refused(self):
        import subprocess
        cases = [
            '{-# LANGUAGE TemplateHaskell #-}\nmodule Main where\nmain = pure ()',
            '{-# OPTIONS_GHC -fplugin=Evil #-}\nmodule Main where\nmain = pure ()',
            'module Main where\nimport {-# SOURCE #-} Trader.X\nmain = pure ()',
            '{-# LANGUAGE PackageImports #-}\nmodule Main where\nimport "base" Prelude\nmain = pure ()',
            'module Main where\nforeign import ccall "evil" evil :: IO ()\nmain = pure ()',
            '#if EVIL\nmodule Main where\n#endif',
            'module Main where\nimport ???']
        for source in cases:
            with self.subTest(source=source), self.assertRaises(subprocess.CalledProcessError):
                self.parse_fixture(source)
        for suffix in ['  ghc-options: -fplugin=Evil\n','  if flag(extra)\n    ghc-options: -O0\n',
                       '  hs-source-dirs: injected\n']:
            with self.subTest(suffix=suffix), self.assertRaises(subprocess.CalledProcessError):
                self.parse_fixture('module Main where\nmain = pure ()',suffix)

    def test_effect_writes_calls_and_keyword_mutants_rejected(self):
        for old,new in [('hidden = x @','self.steps = 4\n                hidden = x @'),
                        ('hidden = x @','place_order()\n                hidden = x @'),
                        ('np.tanh(hidden)','np.tanh(hidden, out=self.p["b1"])'),
                        ('hidden = x @','hidden = (x * 2) @')]:
            self.assertIn(old,self.source)
            with self.subTest(new=new), self.assertRaises(ValueError):
                self.c.effect_certificate(self.source.replace(old,new),self.parsed,self.registry)

    def test_schema_overlap_unknown_and_source_drift_refused(self):
        c = self.c
        with self.assertRaises(ValueError): c.prove_disjoint(['version'],['version'])
        class Unknown:
            def set(self,**kwargs): pass
            def add(self,*args): pass
            def check(self): return z3.unknown
        with patch.object(c.z,'Solver',Unknown), self.assertRaises(ValueError):
            c.prove_disjoint(['schema'],['version'])
        registry = copy.deepcopy(self.registry); registry['savePolicyAST'] = '0'*64
        with self.assertRaisesRegex(ValueError,'writer drift'):
            c.artifact_keys(self.source,self.parsed,registry)
        parsed = copy.deepcopy(self.parsed)
        parsed['declarations']['haskell/app/Main.hs'].append('decoder changed')
        with self.assertRaisesRegex(ValueError,'decoder semantics drift'):
            c.artifact_keys(self.source,parsed,self.registry)

    def test_actual_inference_preserves_numeric_policy_state(self):
        import sys
        sys.path.insert(0,str(ROOT / "scripts/research"))
        import sequential_learning as learning
        net = learning.Network(11)
        before = copy.deepcopy(net.__dict__)
        for observation in [np.zeros(learning.FEATURE_COUNT),np.full(learning.FEATURE_COUNT,np.nan)]:
            learning.infer(net,observation,enabled=True)
            self.assertEqual(net.steps,before['steps'])
            for field in ('p','m','v'):
                for key in before[field]:
                    np.testing.assert_array_equal(getattr(net,field)[key],before[field][key])


class SnapshotV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import sys
        sys.path.insert(0,str(ROOT / 'scripts/research'))
        import optimizer_snapshot_v2 as v
        import snapshot_v2 as proof
        import sequential_learning as legacy
        cls.v,cls.proof,cls.legacy=v,proof,legacy

    def test_source_bound_model_and_smt(self):
        r=self.proof.check_snapshot()
        self.assertEqual(r['storeAttributes'],['_state'])
        self.assertEqual(len(r['smt']),2)
        self.assertGreater(r['model']['orphanLockStates'],0)
        self.assertGreater(r['model']['staleComparisons'],0)
        with self.assertRaisesRegex(ValueError,'mixed snapshot'):
            self.proof.check_model(mutant=True)

    def test_source_and_solver_mutants_fail(self):
        source=(ROOT/self.proof.SOURCE).read_text()
        for old,new in [('self._state = candidate','self._state = expected'),
                        ('self._state is not expected','self._state is expected'),
                        ('blocking=False','blocking=True'),
                        ('not np.isfinite(value).all()','False'),
                        ('frozen=True','frozen=False'),
                        ('base = net.snapshot()','base = net._state')]:
            self.assertIn(old,source)
            with self.subTest(new=new),self.assertRaises(ValueError):
                self.proof.extract(source.replace(old,new))
        # Mutate a translated formula without going through the source hash gate.
        guard=self.proof.extract()
        bad=ast.parse('candidate.step != expected.step + 2 or not 0 <= expected.step < STEP_CAP or candidate.outputs != expected.outputs',mode='eval').body
        with self.assertRaises(ValueError): self.proof.smt((bad,guard[1]))
        class Unknown:
            def set(self,**kwargs): pass
            def add(self,*args): pass
            def check(self): return z3.unknown
        with patch.object(self.proof.z,'Solver',Unknown),self.assertRaises(ValueError):self.proof.smt(guard)

    def test_parity_all_registered_seeds_shapes_and_batches(self):
        v=self.v
        for outputs in (1,3):
            for seed in (11,23,47):
                for rows in (1,8,256):
                    legacy=self.legacy.Network(seed,outputs);net=v.create_v2(seed,outputs,enabled=True)
                    rng=np.random.default_rng(seed+rows)
                    x=rng.normal(0,.02,(rows,12));dz=rng.normal(0,.01,(rows,outputs))
                    for step in range(8):
                        old=net.snapshot()
                        legacy.update(x,dz,.0003)
                        state=v.update_v2(net,x,dz,.0003,enabled=True)
                        self.assertEqual(state.step,step+1);self.assertIs(net.snapshot(),state)
                        self.assertEqual(old.step,step)
                        for field in ('p','m','v'):
                            for key,a in zip(v.KEYS,v._arrays(getattr(state,field),outputs)):
                                np.testing.assert_array_equal(a,getattr(legacy,field)[key])
                        actual=v.forward_v2(net,x,enabled=True);expected=legacy.forward(x)
                        np.testing.assert_array_equal(actual,expected)

    def test_byte_view_parity_counterexample_regression(self):
        from fractions import Fraction
        r=read_json(ROOT/'formal/research/snapshot-v2-counterexample.json')
        self.assertEqual(Fraction(float.fromhex(r['legacyHex']))-Fraction(float.fromhex(r['viewHex'])),Fraction(1,2**63))
        v=self.v;params=tuple(np.array(a,dtype=np.float64) for a in r['parameters'])
        zeros=tuple(np.zeros_like(a) for a in params)
        net=v._Optimizer(v._pack(r['outputs'],r['step'],params,zeros,zeros))
        legacy=self.legacy.Network(r['seed'],r['outputs']);legacy.p=dict(zip(v.KEYS,params))
        x=np.array(r['x'],dtype=np.float64)
        np.testing.assert_array_equal(v.forward_v2(net,x,enabled=True),legacy.forward(x))
        # The recorded raw-view difference is backend-dependent; CI verifies the
        # corrected path, without pretending every backend repeats the old bit pattern.

    def test_immutable_payloads_and_views(self):
        from dataclasses import FrozenInstanceError
        v=self.v;net=v.create_v2(11,enabled=True);state=net.snapshot()
        with self.assertRaises(FrozenInstanceError):state.step=9
        with self.assertRaises(TypeError):state.p[0][0]=1
        for group in (state.p,state.m,state.v):
            arrays=v._arrays(group,3)
            with self.assertRaises(ValueError):arrays[0].setflags(write=True)
            with self.assertRaises(ValueError):arrays[0][0,0]=3
            arrays[0].shape=(192,)
            self.assertEqual(v._arrays(group,3)[0].shape,(12,16))
        self.assertIs(net.snapshot(),state)

    def test_defaults_and_invalid_controls_preserve_state(self):
        v=self.v
        class Poison:
            def __getattribute__(self,name):raise AssertionError('disabled input inspected')
        p=Poison()
        self.assertIsNone(v.create_v2(p,p));self.assertIsNone(v.update_v2(p,p,p,p));self.assertIsNone(v.forward_v2(p,p))
        for seed in (-1,True,2**32,float('nan')):self.assertIsNone(v.create_v2(seed,enabled=True))
        for outputs in (0,2,True):self.assertIsNone(v.create_v2(11,outputs,enabled=True))
        net=v.create_v2(11,enabled=True);before=net.snapshot()
        x=np.ones((8,12));dz=np.ones((8,3))*.002
        bad=[None,np.zeros((0,12)),np.zeros((257,12)),np.zeros((8,11)),np.ones((8,12),dtype=np.float32),np.full((8,12),np.nan),np.ma.array(x)]
        for value in bad:
            self.assertIsNone(v.update_v2(net,value,dz,.001,enabled=True))
            self.assertIsNone(v.forward_v2(net,value,enabled=True));self.assertIs(net.snapshot(),before)
        for lr in (-1,0,True,float('nan'),float('inf'),2):
            self.assertIsNone(v.update_v2(net,x,dz,lr,enabled=True));self.assertIs(net.snapshot(),before)
        self.assertIsNone(v.update_v2(net,x,dz[:-1],.001,enabled=True))

    def test_unsupported_runtime_rejects_before_inputs(self):
        v=self.v;net=v.create_v2(11,enabled=True);before=net.snapshot()
        with patch.object(v.sys,'_is_gil_enabled',return_value=False):
            self.assertIsNone(v.create_v2(11,enabled=True))
            self.assertIsNone(v.update_v2(net,None,None,None,enabled=True))
            self.assertIsNone(v.forward_v2(net,None,enabled=True))
        self.assertIs(net.snapshot(),before)

    def test_staging_failures_and_step_cap(self):
        from dataclasses import replace
        v=self.v;net=v.create_v2(11,enabled=True);old=net.snapshot()
        x=np.ones((8,12));dz=np.ones((8,3))
        for error in (ValueError('staging'),MemoryError('staging')):
            with patch.object(v,'_stage',side_effect=error):self.assertIsNone(v.update_v2(net,x,dz,.001,enabled=True))
            self.assertIs(net.snapshot(),old)
        with patch.object(v,'_stage',side_effect=KeyboardInterrupt('before publication')):
            with self.assertRaises(KeyboardInterrupt):v.update_v2(net,x,dz,.001,enabled=True)
        self.assertIs(net.snapshot(),old)
        self.assertIsNone(v.update_v2(net,np.full((8,12),1e308),np.full((8,3),1e308),.001,enabled=True))
        self.assertIs(net.snapshot(),old)
        # Controlled internal fixture for cap admission; no public restore operation.
        capped=v._Optimizer(replace(old,step=v.STEP_CAP))
        self.assertIsNone(v.update_v2(capped,x,dz,.001,enabled=True))

    def test_two_real_writers_and_stale_replay(self):
        import threading
        v=self.v;net=v.create_v2(11,enabled=True);old=net.snapshot()
        barrier=threading.Barrier(2,timeout=5);stage=v._stage;answers=[];errors=[]
        def staged(*args):
            result=stage(*args);barrier.wait();return result
        def writer(sign):
            try:answers.append(v.update_v2(net,np.ones((8,12))*.01,np.ones((8,3))*.002*sign,.001,enabled=True))
            except BaseException as e:errors.append(e)
        with patch.object(v,'_stage',side_effect=staged):
            threads=[threading.Thread(target=writer,args=(s,)) for s in (-1,1)]
            for t in threads:t.start()
            for t in threads:t.join(10)
            self.assertFalse(any(t.is_alive() for t in threads))
        self.assertFalse(errors);self.assertEqual(sum(s is not None for s in answers),1)
        self.assertEqual(net.snapshot().step,1)
        winner=next(s for s in answers if s is not None)
        self.assertIsNone(net._publish(old,winner));self.assertIs(net.snapshot(),winner)
        net._lock.acquire()
        try:
            self.assertIsNone(v.update_v2(net,np.ones((1,12)),np.ones((1,3)),.001,enabled=True))
            self.assertIsNotNone(v.forward_v2(net,np.ones((1,12)),enabled=True))
            self.assertIs(net.snapshot(),winner)
        finally:net._lock.release()

    def test_reader_uses_one_captured_snapshot_during_update(self):
        import threading
        v=self.v;net=v.create_v2(23,enabled=True);base=net.snapshot();x=np.ones((8,12))*.02
        expected=v.forward_v2(net,x,enabled=True);entered=threading.Event();resume=threading.Event();answers=[]
        arrays=v._arrays
        def paused(buffers,outputs):
            if threading.current_thread().name=='snapshot-reader' and buffers is base.p:
                entered.set()
                if not resume.wait(5):raise RuntimeError('reader fixture timeout')
            return arrays(buffers,outputs)
        def reader():answers.append(v.forward_v2(net,x,enabled=True))
        with patch.object(v,'_arrays',side_effect=paused):
            thread=threading.Thread(target=reader,name='snapshot-reader');thread.start()
            try:
                self.assertTrue(entered.wait(5))
                self.assertIsNotNone(v.update_v2(net,x,np.ones((8,3))*.01,.001,enabled=True))
            finally:resume.set();thread.join(10)
        self.assertFalse(thread.is_alive());self.assertEqual(len(answers),1)
        np.testing.assert_array_equal(answers[0],expected);self.assertEqual(base.step,0)

    def test_nonthrowing_publication_trace_sees_whole_snapshots(self):
        import sys
        v=self.v;net=v.create_v2(47,enabled=True);before=net.snapshot();seen=[]
        def trace(frame,event,arg):
            if frame.f_code is v._Optimizer._publish.__code__:
                frame.f_trace_opcodes=True
                seen.append(net.snapshot())
            return trace
        previous=sys.gettrace()
        try:
            sys.settrace(trace)
            after=v.update_v2(net,np.ones((8,12))*.01,np.ones((8,3))*.002,.001,enabled=True)
        finally:sys.settrace(previous)
        self.assertIsNotNone(after);self.assertTrue(seen)
        self.assertTrue(all(s is before or s is after for s in seen))
        self.assertIn(before,seen);self.assertIn(after,seen)


class InferenceProcessTests(unittest.TestCase):
    def test_source_admission_rejects_mutated_boundaries(self):
        import inference_process as process
        source = (ROOT / process.SOURCE).read_text()
        process.extract(source)
        for old, new in [
            ('elapsed >= budgetNS', 'elapsed > budgetNS'),
            ('env = Just []', 'env = Nothing'),
            ('signalProcess sigKILL identity', 'pure ()'),
            ('admission (ended - started) clean', 'admission 0 True'),
            ('_ -> putStrLn "(Absent,True)"', '_ -> offline'),
        ]:
            with self.subTest(change=old), self.assertRaises(ValueError):
                process.extract(source.replace(old, new))

    def test_lifecycle_rejects_unbounded_poll_and_restart(self):
        import inference_process as process
        for phase, destination in [('pending', 'pending'), ('kill', 'launch')]:
            def mutated(state):
                if state[0] == phase:
                    return [(destination, *state[1:])]
                return process.successors(state)
            with self.subTest(phase=phase), self.assertRaises(ValueError):
                process.check_model(mutated)

    def test_process_guard_and_model(self):
        import inference_process as process
        self.assertEqual(process.prove_guard(20000000), {'F-RL-PROCESS-ADMISSION': 'unsat'})
        self.assertGreater(process.check_model()['expiredReplyStatesRejected'], 0)
        self.assertIsNone(process.admission(20000000, True, 1))
        self.assertIsNone(process.admission(0, False, 1))
        self.assertIsNone(process.admission(0, True, 9))

class ShutdownDeadlineTests(unittest.TestCase):
    def test_shutdown_counterexamples_and_integer_proof(self):
        import shutdown_deadline as shutdown
        self.assertEqual(len(shutdown.counterexamples()), 3)
        self.assertEqual(shutdown.prove_budget(), {'F-SHUTDOWN-BUDGET': 'unsat'})

    def test_shutdown_model_rejects_late_ack_and_skipped_stage(self):
        import shutdown_deadline as shutdown
        for kind in ('late', 'skip', 'history'):
            def mutation(state):
                steps = shutdown.successors(state)
                if not steps:
                    return steps
                index, now, outcomes = state
                if kind == 'late':
                    return [(index + 1, 20, outcomes + (True,))]
                if kind == 'skip':
                    return [(index + 2, now, outcomes + (False, False))]
                return [(index + 1, now, (True,) * index + (False,))]
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                shutdown.check_model(mutation)


class WorkerRegistryTests(unittest.TestCase):
    def test_atomic_predicates_and_model(self):
        import worker_registry as worker
        self.assertEqual(worker.prove_predicates(), {'F-WORKER-REGISTRY-INVARIANTS': 'unsat'})
        receipt = worker.check_model()
        self.assertGreater(receipt['deliveryWithoutCompletionStates'], 0)
        self.assertEqual(receipt['initialRank'], 24)
        self.assertGreater(receipt['requestAfterCompletionTransitions'], 0)
        self.assertEqual(worker.check_capture_trace(), 'MC-WORKER-001')

    def test_mutated_protocol_is_rejected(self):
        import worker_registry as worker
        for kind in ('reopen', 'premature', 'forget', 'duplicate', 'delivery', 'loop'):
            def mutated(state):
                closed, workers, requests, delivered, callers, ticks, captured = state
                steps = worker.successors(state)
                if kind == 'reopen' and closed:
                    return [('mutant', (False, workers, requests, delivered, callers, ticks, captured))]
                if kind == 'premature' and closed and 1 in workers:
                    return [('mutant', (closed, workers, requests, delivered, (3, callers[1]), ticks, captured))]
                if kind == 'forget' and 1 in workers:
                    return [('mutant', (closed, worker.replace(workers, workers.index(1), 3), requests, delivered, callers, ticks, captured))]
                if kind == 'duplicate' and 1 in requests:
                    return [('mutant', (closed, workers, worker.replace(requests, requests.index(1), 2), delivered, callers, ticks, captured))]
                if kind == 'delivery' and any(d and w == 1 for d, w in zip(delivered, workers)):
                    i = next(i for i in range(2) if delivered[i] and workers[i] == 1)
                    return [('deliver', (closed, worker.replace(workers, i, 2), requests, delivered, callers, ticks, captured))]
                if kind == 'loop':
                    return [('mutant', state)]
                return steps
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                worker.check_model(mutated)

    def test_source_drift_rejected(self):
        import worker_registry as worker
        from unittest.mock import patch
        original = Path.read_bytes
        def altered(path):
            data = original(path)
            if str(path).endswith(worker.SOURCE):
                return data.replace(b'WorkerRegistryState True retained', b'WorkerRegistryState False retained')
            return data
        with patch.object(Path, 'read_bytes', altered), self.assertRaises(ValueError):
            worker.extract()


class AsyncAdmissionTests(unittest.TestCase):
    def test_numeric_and_model(self):
        import async_job_admission as admission
        self.assertEqual(admission.prove_numeric(), {'F-ASYNC-ADMISSION-NUMERIC': 'unsat'})
        self.assertEqual(admission.check_model()['states'], 369)

    def test_mutated_ownership_and_publication(self):
        import async_job_admission as admission
        for kind in ('leak', 'double-release', 'early-execute', 'loop', 'overbook'):
            def mutation(state, capacity):
                count, callers = state
                if kind == 'loop':
                    return [('mutant', state)]
                if kind == 'overbook' and count == capacity and callers[1][0] == 'new':
                    row = ('prepare', 'none', None, False, False)
                    return [('mutant', (count+1, (callers[0], row)))]
                if callers[0][0] == 'prepare' and kind in ('leak', 'double-release'):
                    row = ('failed', 'none', None, False, False)
                    return [('mutant', (count if kind == 'leak' else count-2, (row, callers[1])))]
                if callers[0][0] == 'waiting' and kind == 'early-execute':
                    row = ('waiting', 'running', True, False, True)
                    return [('mutant', (count, (row, callers[1])))]
                return admission.successors(state, capacity)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                admission.check_model(mutation)

    def test_source_boundary_drift(self):
        import async_job_admission as admission
        original = Path.read_bytes
        def changed(path):
            content = original(path)
            if str(path).endswith(admission.SOURCE):
                return content.replace(b'putMVar gate False', b'putMVar gate True')
            return content
        with patch.object(Path, 'read_bytes', changed), self.assertRaises(ValueError):
            admission.extract()


class AsyncSealTests(unittest.TestCase):
    def test_model_and_smt(self):
        import async_shutdown_seal as seal
        self.assertEqual(seal.prove_invariants(), {'F-ASYNC-SEAL-INVARIANTS': 'unsat'})
        self.assertGreater(seal.check_model()['states'], 369)

    def test_invalid_transitions_rejected(self):
        import async_shutdown_seal as seal
        for kind in ('early', 'reopen', 'consume', 'loop', 'post-seal'):
            def mutated(state, capacity):
                base, closed, closers, pending, signal = state
                count, callers = base
                if kind == 'early' and count:
                    return [('bad', (base, closed, closers, pending, True))]
                if kind == 'reopen' and closed:
                    return [('bad', (base, False, closers, pending, signal))]
                if kind == 'consume' and signal:
                    return [('bad', (base, closed, closers, pending, False))]
                if kind == 'loop':
                    return [('bad', state)]
                if kind == 'post-seal' and closed and callers[0][0] == 'new' and count < capacity:
                    row = ('prepare', 'none', None, False, False)
                    return [('bad', ((count+1, (row, callers[1])), closed, closers, pending, signal))]
                return seal.successors(state, capacity)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                seal.check_model(mutated)

    def test_seal_source_drift_rejected(self):
        import async_shutdown_seal as seal
        original = Path.read_bytes
        def changed(path):
            content = original(path)
            if str(path).endswith(seal.SOURCE):
                return content.replace(b'closed && next == 0', b'next == 0')
            return content
        with patch.object(Path, 'read_bytes', changed), self.assertRaises(ValueError):
            seal.extract()


class AdmissionProgressTests(unittest.TestCase):
    def test_source_numeric_model_and_actual_haskell(self):
        import admission_progress as p
        result = p.check_progress()
        self.assertEqual(result['queries'], 8)
        self.assertEqual(result['model']['states'], 838)
        self.assertEqual(result['model']['transitions'], 3356)
        self.assertEqual(result['conformance']['regression'], {'old': 'barged=True', 'new': 'barged=False'})

    def test_barging_mutant_has_fair_starvation(self):
        import admission_progress as p
        graph, _ = p.explore(1, barging=True)
        self.assertTrue(p.fair_bad_components(graph))

    def test_cancellation_leak_is_detected(self):
        import admission_progress as p
        with self.assertRaisesRegex(ValueError, 'orphan'):
            p.explore(1, leak_cancel=True)

    def test_scc_against_all_three_vertex_graphs(self):
        import admission_progress as p
        import itertools
        vertices = set(range(3))
        possible = list(itertools.product(vertices, repeat=2))
        for mask in range(1 << len(possible)):
            graph = {v: [] for v in vertices}
            for i, (v, w) in enumerate(possible):
                if mask & (1 << i):
                    graph[v].append(('edge', w))
            reach = {v: {v} for v in vertices}
            for v in vertices:
                for _, w in graph[v]:
                    reach[v].add(w)
            for k in vertices:
                for v in vertices:
                    if k in reach[v]:
                        reach[v].update(reach[k])
            expected = {frozenset(w for w in vertices if w in reach[v] and v in reach[w]) for v in vertices}
            self.assertEqual({frozenset(g) for g in p.components(vertices, graph)}, expected)

    def test_weak_fairness_distinguishes_intermittent_enable(self):
        import admission_progress as p
        a = (False, (0,), ('wait', 'new', 'new'), 0)
        b = (False, (0,), ('wait', 'active', 'new'), 1)
        done = (False, (), ('reserved', 'new', 'new'), 1)
        graph = {a: [('compete', b), ('acquire:0', done)], b: [('release:1', a)], done: [('stutter', done)]}
        self.assertTrue(p.fair_bad_components(graph))  # intermittently enabled is not weak fairness
        graph[b].append(('acquire:0', done))
        self.assertFalse(p.fair_bad_components(graph))  # continuously enabled must eventually run
        graph[a].append(('acquire:0', b))
        self.assertTrue(p.fair_bad_components(graph))  # a fair edge inside the SCC is exercised

    def test_numeric_unknown_and_vacuity_fail(self):
        import admission_progress as p
        from unittest.mock import patch
        for result in (p.z.unknown, p.z.unsat):
            with patch.object(p.z.Solver, 'check', return_value=result):
                with self.assertRaises(ValueError):
                    p.prove_numeric()

    def test_ledger_model_bounds_are_reproduced(self):
        from verify import validate_progress_bounds
        ledger = read_json(ROOT / 'formal/research/proof-ledger.json')
        result = {'backtestGate': {'model': {'states':1792, 'transitions':3456}},
                  'drainPool': {'model': {'states':5376, 'transitions':17872}}}
        validate_progress_bounds(ledger['entries'], result)
        for field in ('modelBounds', 'bounds'):
            changed = copy.deepcopy(ledger['entries'])
            entry = next(e for e in changed if e['requirementId'] == 'F-BACKTEST-GATE-LIFECYCLE')
            entry[field] = {} if field == 'modelBounds' else '1656 states, 2984 edges'
            with self.assertRaisesRegex(ValueError, 'stale'):
                validate_progress_bounds(changed, result)

    def test_preserved_legacy_lasso(self):
        import admission_progress as p
        fixture = read_json(ROOT / 'formal/research/fixtures/admission-progress.json')
        self.assertEqual(p.legacy_witness(), fixture['legacy'])


class BacktestGateTests(unittest.TestCase):
    def test_numeric_and_lifecycle(self):
        import backtest_gate as gate
        self.assertEqual(gate.prove_numeric(), {'F-BACKTEST-GATE-NUMERIC': 'unsat'})
        self.assertEqual(gate.check_model()['states'], 1792)

    def test_invalid_lifecycle_rejected(self):
        import backtest_gate as gate
        for kind in ('leak','double-release','swallow','early','stuck'):
            def mutation(state, capacity, modes):
                count,callers=state
                phase,cause,outcome,ran=callers[0]
                if kind=='stuck': return [('bad',state)]
                if phase=='cleanup' and kind in ('leak','double-release'):
                    row=('done',cause,outcome,ran)
                    return [('bad',(count if kind=='leak' else count-2,(row,callers[1])))]
                if phase=='active' and kind=='swallow':
                    row=('cleanup','cancel','error',ran)
                    return [('bad',(count,(row,callers[1])))]
                if phase=='new' and kind=='early':
                    row=('new',cause,outcome,True)
                    return [('bad',(count,(row,callers[1])))]
                return gate.successors(state,capacity,modes)
            with self.subTest(kind=kind), self.assertRaises(ValueError):gate.check_model(mutation)

    def test_source_rejects_broad_exception_handler(self):
        import backtest_gate as gate
        original=Path.read_bytes
        def changed(path):
            value=original(path)
            if str(path).endswith(gate.SOURCE):return value.replace(b'Just _ -> Nothing',b'Just _ -> Just ex')
            return value
        with patch.object(Path,'read_bytes',changed), self.assertRaises(ValueError):gate.extract()


class DrainPoolTests(unittest.TestCase):
    def test_numeric_and_model(self):
        import drain_pool as pool
        self.assertEqual(pool.prove_order(), {'F-DRAIN-POOL-ORDER': 'unsat'})
        self.assertEqual(pool.check_model()['states'], 5376)

    def test_invalid_order_rejected(self):
        import drain_pool as pool
        for kind in ('late', 'leak', 'reopen', 'stuck'):
            def mutation(state, capacity, assignment):
                draining, counts, closed, callers, drainers = state
                if kind == 'stuck': return [('bad', state)]
                if kind == 'reopen' and draining:
                    return [('bad', (False, counts, closed, callers, drainers))]
                if kind == 'late' and draining and callers[0][0] == 1:
                    return [('bad', (draining, pool.replace(counts, assignment[0], counts[assignment[0]]+1), closed,
                                         pool.replace(callers, 0, (2, False, True)), drainers))]
                if kind == 'leak' and callers[0][0] == 3:
                    return [('release', (draining, counts, closed, pool.replace(callers, 0, (4, False, False)), drainers))]
                return pool.successors(state, capacity, assignment)
            with self.subTest(kind=kind), self.assertRaises(ValueError): pool.check_model(mutation)

    def test_disconnected_controller_rejected(self):
        import drain_pool as pool
        original = Path.read_bytes
        def changed(path):
            value = original(path)
            if str(path).endswith('haskell/app/Main.hs'):
                return value.replace(b'newBacktestGateWithDrain drain', b'newBacktestGateWithDrain other')
            return value
        with patch.object(Path, 'read_bytes', changed), self.assertRaises(ValueError): pool.extract()



class DataCompositionTests(unittest.TestCase):
    def sources(self):
        import data_composition as dc
        return {f: (ROOT / 'scripts/research' / f).read_text() for f in dc.FILES}

    def test_complete_source_graph(self):
        import data_composition as dc
        result = dc.check_composition()
        self.assertEqual(result['scaleUseSites'], 28)
        self.assertEqual(result['keys']['violationChecks'], 6)
        self.assertFalse(result['publicationWitnessesVerified'])

    def test_leakage_mutants_fail_semantic_admission(self):
        import data_composition as dc
        mutants = [
            ('run_sequential_screen.py', 'Scale.fit(list(train.values()))', 'Scale.fit(list(prices.values()))'),
            ('run_sequential_screen.py', 'bars[bars.symbol == symbol]', 'bars[bars.symbol == "FOREIGN"]'),
            ('run_sequential_screen.py', 'events[events.symbol == symbol]', 'events[events.symbol == "FOREIGN"]'),
            ('sequential_env.py', 'p = prices[symbol]', 'p = prices["FOREIGN"]'),
            ('sequential_evaluation.py', 'Replay(prices[sym], funding[sym]', 'Replay(prices[sym], funding["FOREIGN"]'),
            ('sequential_env.py', 'dataclass(frozen=True)', 'dataclass(frozen=False)'),
            ('sequential_env.py', 'np.frombuffer(np.asarray(value, dtype=float).tobytes(), dtype=float)', 'np.asarray(value, dtype=float)'),
            ('run_sequential_screen.py', 'controls = Baselines(train, funds, scale, h)', 'stolen = scale\n                controls = Baselines(train, funds, scale, h)'),
        ]
        for file, old, new in mutants:
            with self.subTest(file=file, mutation=new):
                sources = self.sources()
                self.assertIn(old, sources[file])
                sources[file] = sources[file].replace(old, new)
                # These fail before source hashes; a hash mismatch is not a semantic proof.
                with self.assertRaises((ValueError, RuntimeError)) as caught:
                    dc.check_composition(sources)
                self.assertNotIn('unreviewed helper/source drift', str(caught.exception))

    def test_missing_reviewed_coverage_cannot_certify(self):
        import data_composition as dc
        original = read_json(ROOT / dc.REGISTRY)
        for mutate in (lambda r: r.update(blocks={}),
                       lambda r: r['blocks']['sequential_env.py'].pop('Scale.__post_init__'),
                       lambda r: r.update(sourceHashes={}),
                       lambda r: r.update(sharedInputs=[]),
                       lambda r: r.update(schemaVersion=2)):
            registry = copy.deepcopy(original)
            mutate(registry)
            with self.assertRaises(ValueError):
                dc.check_composition(registry=registry)
        fixture = read_json(ROOT / 'formal/research/data-composition-counterexamples.json')
        self.assertEqual(fixture['entries'][0]['id'], 'CE-DATA-COMPOSITION-001')
        self.assertEqual(fixture['entries'][0]['mutation'], {'blocks': {}})

    def test_equal_foreign_keys_are_not_declared_symbol(self):
        import data_composition as dc
        sources = self.sources()
        sources['sequential_env.py'] = sources['sequential_env.py'].replace(
            'p = prices[symbol]', 'p = prices["FOREIGN"]').replace(
            'Replay(p, funding[symbol]', 'Replay(p, funding["FOREIGN"]')
        with self.assertRaisesRegex(RuntimeError, 'violating source bound'):
            dc.check_keys({file: ast.parse(source) for file, source in sources.items()})

    def test_wrong_key_yields_smt_counterexample(self):
        import data_composition as dc
        sources = self.sources()
        sources['sequential_env.py'] = sources['sequential_env.py'].replace(
            'p = prices[symbol]', 'p = prices["FOREIGN"]')
        with self.assertRaisesRegex(RuntimeError, 'violating source bound'):
            dc.check_keys({file: ast.parse(source) for file, source in sources.items()})


class PPOSuccessorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import sys
        sys.path.insert(0, str(ROOT / 'scripts/research'))
        import ppo_successor_v2
        cls.impl = ppo_successor_v2

    def fixture(self):
        x = np.arange(180, dtype=float)
        prices = {'ALPHA': 100 * np.exp(.0002*x + .002*np.sin(x/9)),
                  'BETA': 200 * np.exp(-.0001*x + .003*np.sin(x/11))}
        return prices, {s: np.zeros(len(p)) for s, p in prices.items()}

    def test_registered_seed_horizon_budget_matrix_and_determinism(self):
        p, f = self.fixture()
        before = {s: v.tobytes() for s, v in p.items()}
        for seed in (11, 23, 47):
            for horizon in (1, 3, 6):
                for steps in (1, 17, 257):
                    with self.subTest(seed=seed, horizon=horizon, steps=steps):
                        first = self.impl.train_ppo_v2(p, f, horizon, seed, steps, enabled=True)
                        self.assertIsNotNone(first)
                        second = self.impl.train_ppo_v2(p, f, horizon, seed, steps, enabled=True)
                        self.assertEqual(first, second)
                        self.assertEqual(first.actor.step, 4*((steps+255)//256))
                        self.assertEqual(first.actor.step, first.critic.step)
                        self.assertEqual(len(first.losses), first.actor.step)
                        self.assertTrue(all(np.isfinite(first.losses)))
                        for snapshot in (first.actor, first.critic):
                            for group in (snapshot.p, snapshot.m, snapshot.v):
                                for raw in group:
                                    self.assertIs(type(raw), bytes)
                                    self.assertTrue(np.isfinite(np.frombuffer(raw, dtype='<f8')).all())
                        self.assertEqual(first.symbols, ('ALPHA', 'BETA'))
        self.assertEqual(before, {s: v.tobytes() for s, v in p.items()})

    def test_disabled_version_and_config_reject_before_effects(self):
        with patch.object(self.impl, '_prefixes', side_effect=RuntimeError('must not run')):
            self.assertIsNone(self.impl.train_ppo_v2(object(), object(), object(), object()))
            for flag in (False, None, 1, 'true', np.bool_(True)):
                self.assertIsNone(self.impl.train_ppo_v2(None, None, 1, 11, enabled=flag))
            self.assertIsNone(self.impl.train_ppo_v2(None, None, 1, 11, enabled=True, version='v1'))
            for horizon, seed, steps in ((True, 11, 1), (2, 11, 1), (1, -1, 1),
                                         (1, 2**32, 1), (1, True, 1), (1, 11, 0),
                                         (1, 11, 4097), (1, 11, 1.0)):
                self.assertIsNone(self.impl.train_ppo_v2(None, None, horizon, seed, steps, enabled=True))

    def test_invalid_prefixes_funding_and_snapshot_aliases(self):
        p, f = self.fixture()
        cp, cf = self.impl._prefixes(p, f)
        for symbol in p:
            self.assertFalse(np.shares_memory(p[symbol], cp[symbol]))
            with self.assertRaises(ValueError):
                cp[symbol].setflags(write=True)
        for bad in (float('nan'), float('inf'), -float('inf')):
            for dest in (p, f):
                mutated = {s: v.copy() for s, v in dest.items()}
                mutated['ALPHA'][0] = bad
                pp, ff = (mutated, f) if dest is p else (p, mutated)
                with patch.object(self.impl, 'collect', side_effect=RuntimeError('must not collect')):
                    self.assertIsNone(self.impl.train_ppo_v2(pp, ff, 1, 11, 1, enabled=True))
        for invalid in ({}, {'ALPHA': p['ALPHA'][:120]}, {'ALPHA': np.zeros(5000)},
                        {'ALPHA': np.zeros(180)}, {'ALPHA': p['ALPHA'].astype(np.float32)}):
            self.assertIsNone(self.impl.train_ppo_v2(invalid, f, 1, 11, 1, enabled=True))

    def test_partial_actor_update_cannot_publish_training_result(self):
        p, f = self.fixture()
        created = []
        create, update = self.impl.create_v2, self.impl.update_v2
        def recording(*a, **kw):
            net = create(*a, **kw); created.append(net); return net
        calls = []
        def failed_critic(*a, **kw):
            calls.append(a[0])
            return None if len(calls) == 2 else update(*a, **kw)
        with patch.object(self.impl, 'create_v2', recording), patch.object(self.impl, 'update_v2', failed_critic):
            self.assertIsNone(self.impl.train_ppo_v2(p, f, 1, 11, 17, enabled=True))
        self.assertEqual([n.snapshot().step for n in created], [1, 0])

    def test_each_stage_failure_publishes_nothing(self):
        p, f = self.fixture()
        for name in ('_prefixes', '_rollout', '_targets', '_update', 'collect'):
            with self.subTest(name=name), patch.object(self.impl, name, side_effect=ValueError('fixture failure')):
                self.assertIsNone(self.impl.train_ppo_v2(p, f, 1, 11, 17, enabled=True))
        for name in ('create_v2', 'forward_v2', 'batch_v2', 'update_v2'):
            with self.subTest(name=name), patch.object(self.impl, name, return_value=None):
                self.assertIsNone(self.impl.train_ppo_v2(p, f, 1, 11, 17, enabled=True))

    def test_nonfinite_objective_and_gradient_are_rejected(self):
        p, f = self.fixture()
        for loss, gradient in ((float('nan'), np.zeros((17,3))),
                               (0.0, np.full((17,3), np.nan)),
                               (float('inf'), np.zeros((17,3))), (0.0, np.zeros((17,2)))):
            with patch.object(self.impl, 'ppo_gradient', return_value=(loss, gradient)):
                self.assertIsNone(self.impl.train_ppo_v2(p, f, 1, 11, 17, enabled=True))

    def test_source_and_model_mutations_refuse_partial_publication(self):
        import ppo_successor as proof
        source = (ROOT / proof.SOURCE).read_text()
        mutations = [source.replace('enabled: object = False', 'enabled: object = True'),
                     source.replace('steps - batch * 256', 'steps'),
                     source.replace('range(4)', 'range(0)'),
                     source.replace('batch_v2(rows, gamma, enabled=True)', 'None'),
                     source.replace('critic.snapshot()', 'actor.snapshot()'),
                     source.replace('if pairs is None:', 'if False:'),
                     source.replace('losses.append(loss)', 'losses.append(float("nan"))')]
        for mutant in mutations:
            self.assertNotEqual(source, mutant)
            with self.assertRaises(ValueError):
                proof.extract(mutant)
        with self.assertRaisesRegex(ValueError, 'partial result published'):
            proof.check_model(mutant=True)
        registry = read_json(ROOT / proof.REGISTRY)
        for key in ('definitions', 'helperHashes'):
            missing = copy.deepcopy(registry); missing[key] = {}
            with self.assertRaisesRegex(ValueError, 'coverage omitted'):
                proof.extract(registry=missing)

    def test_preserved_numeric_counterexamples_at_composed_boundary(self):
        terminal = read_json(ROOT / 'formal/research/terminal-counterexamples.json')['entries']
        one = terminal[0]
        data = {'s': np.zeros((1,12)), 'next': np.zeros((1,12)),
                'r': np.array([float.fromhex(one['rewardHex'][0])]), 'done': np.array([True])}
        with patch.object(self.impl, '_forward', return_value=np.array([[float.fromhex(one['criticHex'])]])):
            _, targets = self.impl._targets(data, object(), float.fromhex(one['gammaHex']))
            self.assertEqual(targets[0].hex(), one['rewardHex'][0])
        two = terminal[1]
        data = {'s': np.zeros((2,12)), 'next': np.zeros((2,12)),
                'r': np.array([float.fromhex(v) for v in two['rewardHex']]), 'done': np.array([True,True])}
        with patch.object(self.impl, '_forward', return_value=np.full((2,1), float.fromhex(two['criticHex']))):
            with self.assertRaisesRegex(ValueError, 'rejected GAE'):
                self.impl._targets(data, object(), float.fromhex(two['gammaHex']))
        ce = read_json(ROOT / 'formal/research/ppo-counterexamples.json')['entries'][0]
        data = {'s': np.zeros((1,12)), 'a': np.array([ce['action']]),
                'prob': np.array([float.fromhex(ce['oldProbabilityHex'])])}
        for mode in ('ignore', 'raise'):
            with np.errstate(all=mode), patch.object(self.impl, '_forward', return_value=np.zeros((1,3))), patch.object(self.impl, 'update_v2') as update:
                with self.assertRaises((ValueError, ArithmeticError)):
                    self.impl._update(object(), object(), data,
                                      np.array([float.fromhex(ce['advantageHex'])]), np.zeros(1), 1)
                update.assert_not_called()

    def test_generated_budget_partition_properties(self):
        import random
        rng = random.Random(20261004)
        for _ in range(512):
            steps = rng.randint(1, 4096)
            counts = [min(256, steps - b*256) for b in range((steps+255)//256)]
            self.assertEqual(sum(counts), steps)
            self.assertTrue(all(1 <= c <= 256 for c in counts))
            self.assertTrue(self.impl._configuration(rng.choice((1, 3, 6)), rng.randrange(2**32-10000), steps))
        p, f = self.fixture()
        for _ in range(8):
            h, seed, steps = rng.choice((1, 3, 6)), rng.randrange(10000), rng.randrange(1, 34)
            result = self.impl.train_ppo_v2(p, f, h, seed, steps, enabled=True)
            self.assertIsNotNone(result)
            self.assertEqual(result.actor.step, result.critic.step)
            self.assertTrue(np.isfinite(result.losses).all())

    def test_smt_and_model_receipts(self):
        import ppo_successor as proof
        result = proof.check_successor()
        self.assertEqual(result['smt'], {'F-RL-PPO-V2-BOUNDS': 'unsat', 'F-RL-PPO-V2-FINITE': 'unsat'})
        self.assertEqual([c['batches'] for c in result['model']['configurations']], list(range(1, 17)))
        self.assertGreater(result['model']['states'], 0)




class PPOProcessBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import sys
        sys.path.insert(0, str(ROOT / 'scripts/research'))
        import ppo_inference_v3
        cls.impl = ppo_inference_v3

    def fixture(self):
        from dataclasses import replace
        from optimizer_snapshot_v2 import create_v2
        from ppo_successor_v2 import TrainingResult
        snapshot = create_v2(11, 3, enabled=True).snapshot()
        # Deliberately constructed metadata tests compatibility, not provenance.
        actor = replace(snapshot, step=4)
        return TrainingResult('ppo-successor-v2', 11, 1, 17, ('ALPHA',), (), actor, actor, ())

    def test_default_disabled_invalid_types_versions_and_shapes(self):
        from dataclasses import replace
        result = self.fixture(); obs = np.zeros(12)
        with patch.object(self.impl, '_words', side_effect=RuntimeError('must not encode')):
            self.assertIsNone(self.impl.encode_request_v3(object(), object()))
            for flag in (False, None, 1, 'true', np.bool_(True)):
                self.assertIsNone(self.impl.encode_request_v3(result, obs, enabled=flag))
            self.assertIsNone(self.impl.encode_request_v3(result, obs, enabled=True, version='v2'))
        for observation in (None, obs.tolist(), np.zeros(11), np.zeros((1,12)),
                            np.zeros(12, dtype=np.float32), np.full(12, np.nan),
                            np.full(12, np.inf), np.full(12, -np.inf), np.full(12, 1001)):
            self.assertIsNone(self.impl.encode_request_v3(result, observation, enabled=True))
        actors = [replace(result.actor, **fields) for fields in
                  ({'version':'v1'}, {'outputs':True}, {'outputs':1}, {'step':0}, {'step':True},
                   {'step':8}, {'p':list(result.actor.p)}, {'p':result.actor.p[:-1]},
                   {'p':(bytearray(result.actor.p[0]), *result.actor.p[1:])},
                   {'p':(result.actor.p[0][:-8], *result.actor.p[1:])})]
        for value in (np.nan, np.inf, -np.inf, 1001.0, -1001.0):
            actors.append(replace(result.actor, p=(np.full(192,value).astype('<f8').tobytes(), *result.actor.p[1:])))
        for actor in actors:
            self.assertIsNone(self.impl.encode_request_v3(replace(result,actor=actor), obs, enabled=True))
        for fields in ({'version':'v1'}, {'seed':True}, {'steps':4097}, {'horizon':2}):
            self.assertIsNone(self.impl.encode_request_v3(replace(result,**fields), obs, enabled=True))

    def test_generated_bit_preservation_and_immutable_request(self):
        import ast
        from dataclasses import replace
        rng = np.random.default_rng(20261005)
        result = self.fixture()
        for _ in range(128):
            steps = int(rng.integers(1,4097))
            value = replace(result, steps=steps, actor=replace(result.actor, step=4*((steps+255)//256)))
            obs = rng.uniform(-1000,1000,12)
            frame = self.impl.encode_request_v3(value, obs, enabled=True)
            self.assertIs(type(frame), bytes)
            self.assertLess(len(frame),32768)
            tag, step, ow, pw = ast.literal_eval(frame.decode('ascii'))
            self.assertEqual(tag, self.impl.VERSION)
            self.assertEqual(step,value.actor.step)
            self.assertEqual(ow,[int(w) for w in obs.astype('<f8').view('<u8')])
            self.assertEqual(pw,[int(w) for w in np.frombuffer(b''.join(value.actor.p),dtype='<u8')])
            frozen = frame[:]; obs[:] = np.nan
            self.assertEqual(frame,frozen)
        for error in (ValueError, MemoryError, FloatingPointError):
            with patch.object(self.impl,'_words',side_effect=error('fixture')):
                self.assertIsNone(self.impl.encode_request_v3(result,np.zeros(12),enabled=True))

    def test_source_coverage_mutants_and_model_bypass_reject(self):
        import ppo_process_bridge as proof
        source = (ROOT / proof.SOURCE).read_text()
        for old,new in [('enabled: object = False','enabled: object = True'),
                        ('actor.step != 4','actor.step == 4'),
                        ('np.abs(value) > 1000','np.abs(value) > 10000'),
                        ('type(b) is not bytes','type(b) is not bytearray'),
                        ('len(frame) < 32768','len(frame) < 65536')]:
            mutated = source.replace(old,new); self.assertNotEqual(source,mutated)
            with self.assertRaises(ValueError): proof.extract(source=mutated)
        decoder = (ROOT / proof.DECODER).read_text()
        for old,new in [('word < 0','word < -1'),('step > 64','step > 128'),
                        ('length digits > 20','length digits > 200'),('remaining - 1','remaining + 1'),
                        ('abs value > 1000','abs value > 10000'),('length observation /= 12','length observation /= 11')]:
            mutated = decoder.replace(old,new); self.assertNotEqual(decoder,mutated)
            with self.assertRaises(ValueError): proof.extract(decoder=mutated)
        registry = read_json(ROOT / proof.REGISTRY)
        for key in ('definitions','helperHashes'):
            missing = copy.deepcopy(registry); missing[key] = {}
            with self.assertRaisesRegex(ValueError,'coverage omitted'): proof.extract(registry=missing)
        with self.assertRaisesRegex(ValueError,'bypassed decoder'): proof.check_model(mutant=True)

    def test_source_bound_smt_model_and_actual_trained_process(self):
        import ppo_process_bridge as proof
        result = proof.check_bridge()
        self.assertEqual(result['smt'],{'F-RL-BRIDGE-V3-CODEC':'unsat'})
        self.assertEqual(result['model']['states'],83)
        self.assertEqual(result['model']['transitions'],189)
        self.assertEqual(result['conformance']['syntheticFits'],9)
        self.assertEqual(result['conformance']['trainedObservationCases'],27)

class PPOArtifactV4Tests(unittest.TestCase):
    def test_source_and_mandatory_coverage(self):
        import json
        import artifact_v4 as proof
        source = (proof.ROOT / proof.SOURCE).read_text()
        proof.extract()
        for before, after in [('enabled is not True', 'enabled is True'),
                              ('hashlib.sha256(raw).hexdigest() != expected_sha256', 'False'),
                              ('value["enabled"] is not False', 'False'),
                              ('return restored', 'return None')]:
            changed = source.replace(before, after, 1)
            self.assertNotEqual(source, changed)
            with self.assertRaises(ValueError): proof.extract(changed)
        registry = json.loads((proof.ROOT / proof.REGISTRY).read_text())
        registry['helperHashes'].pop(next(iter(registry['helperHashes'])))
        with self.assertRaises(ValueError): proof.extract(registry=registry)

    def test_gate_bypass_counterexample(self):
        import artifact_v4 as proof
        self.assertEqual(proof.check_model()['states'], 21)
        with self.assertRaises(ValueError): proof.check_model(mutant=True)
        self.assertEqual(proof.prove_guards(proof.extract()), {'F-RL-ARTIFACT-V4-GUARD':'unsat'})

    def test_trained_artifact_conformance(self):
        import artifact_v4 as proof
        result = proof.conformance()
        self.assertEqual(result['fits'], 9)
        self.assertEqual(result['compiledBitRoundtrips'], 9)
        self.assertEqual(result['artifactSizeChecks'], 9)
        self.assertEqual(result['maximumArtifactBytes'], 65536)
        self.assertNotIn('artifactBytes', result)
        self.assertEqual(result['generatedBitCases'], 128)
        self.assertGreaterEqual(result['invalidArtifacts'], 30)


class QuantityRoundingTests(unittest.TestCase):
    def test_source_guard_and_adapter_mutations(self):
        import quantity_rounding as q
        source = (q.ROOT/q.SOURCE).read_text(); adapter = (q.ROOT/q.ADAPTER).read_text()
        q.extract()
        for before, after in [('isNaN x || ', ''), ('scale <= 0', 'scale < 0'),
                              ('rounded > x', 'rounded > x + 1e-9'), ('`div`', '`quot`')]:
            changed = source.replace(before,after,1)
            self.assertNotEqual(changed,source)
            with self.assertRaises(ValueError): q.extract(source=changed)
        with self.assertRaises(ValueError): q.extract(adapter=adapter.replace('(stepScale st) (stepInt st)', '(stepInt st) (stepScale st)',1))
        main = (q.ROOT/'haskell/app/Main.hs').read_text()
        with self.assertRaises(ValueError): q.extract(main=main.replace('validateQuantityInput (fmap', 'pure () -- (fmap',1))

    def test_rounding_smt_and_compiled_conformance(self):
        import quantity_rounding as q
        result = q.check_rounding()
        self.assertEqual(len(result['smt']),2)
        self.assertEqual(result['conformance']['cases'],4226)



class UpwardRoundingTests(unittest.TestCase):
    def test_caller_and_guard_mutations(self):
        import upward_rounding as u
        core=(u.ROOT/u.CORE).read_text(); main=(u.ROOT/u.MAIN).read_text()
        u.extract()
        for before,after in [('rounded < x','rounded < x - 1e-9'),('+ divisor - 1','+ divisor'),('validateOrderNumber "Invalid maker price." price','Right ()')]:
            with self.assertRaises(ValueError): u.extract(core=core.replace(before,after))
        for before,after in [(u.DELEGATE,u.DELEGATE.replace('stepScale','stepInt')),('then pure baseOut{aorMessage = "No order: invalid maker price."}','then fallback "price unavailable"')]:
            with self.assertRaises(ValueError): u.extract(main=main.replace(before,after,1))

    def test_smt_model_and_compiled_callers(self):
        import upward_rounding as u
        result=u.check_upward()
        self.assertEqual(len(result['smt']),2)
        self.assertEqual(result['model']['states'],8)
        self.assertEqual(result['conformance']['cases'],4226)
        self.assertEqual(result['conformance']['dispatchCases'],8452)



class OrderNumberTests(unittest.TestCase):
    def test_source_and_coverage_mutations(self):
        import json
        import order_numbers as n
        core=(n.ROOT/n.CORE).read_text();adapter=(n.ROOT/n.ADAPTER).read_text()
        n.extract()
        for before,after in [('isNaN value || ',''),('value <= 0','value < 0'),('case quantity of','case quoteOrderQty of'),('units > 0','units >= 0'),('renderOrderNumber value','show value')]:
            with self.assertRaises(ValueError): n.extract(core=core.replace(before,after,1))
        for line in adapter.splitlines(keepends=True):
            if 'either (throwIO . userError) pure (validate' in line and ('validateOrderNumber ' in line or 'validateMarketNumbers ' in line):
                with self.assertRaises(ValueError): n.extract(adapter=adapter.replace(line,'',1))
        with self.assertRaises(ValueError): n.extract(adapter=adapter.replace('renderDouble = BS.pack . renderOrderNumber','renderDouble = BS.pack . show'))
        registry=json.loads((n.ROOT/n.REGISTRY).read_text())
        del registry['prefixes'][n.ROSTER[-1]]
        with self.assertRaises(ValueError): n.extract(registry=registry)

    def test_smt_model_and_actual_prefixes(self):
        import order_numbers as n
        result=n.check_order_numbers()
        self.assertEqual(len(result['smt']),3)
        self.assertEqual(result['model']['states'],80)
        self.assertEqual(result['conformance']['rows'],5728)
        self.assertEqual(result['conformance']['currentPrefixCases'],28640)



class SizingInputTests(unittest.TestCase):
    def test_source_and_caller_mutations(self):
        import sizing_inputs as s
        core=(s.ROOT/s.CORE).read_text(); main=(s.ROOT/s.MAIN).read_text()
        s.extract()
        for before,after in [('x < 0 = Left','x <= 0 = Left'),('x >= 0','x > 0'),('lo > hi','lo < hi'),('p <= 0','p < 0'),
                             ('Invalid sizing price.','Quantity below minQty invalid price')]:
            changed=core.replace(before,after,1)
            self.assertNotEqual(core,changed)
            with self.assertRaises(ValueError): s.extract(core=changed)
        for line in main.splitlines(keepends=True):
            if any(x in line for x in ['validateSizingInputs (effectiveMinQty sf)', 'validateSizingInputs (mSf >>= effectiveMinQty)',
                                       'validateMinimumNotional (mSf >>= sfMinNotional)', 'validateQuantityInput Nothing qty2']):
                with self.assertRaises(ValueError): s.extract(main=main.replace(line,'',1))
        import json
        registry=json.loads((s.ROOT/s.REGISTRY).read_text());registry['functions'].pop()
        with self.assertRaises(ValueError): s.extract(registry=registry)

    def test_smt_model_and_actual_functions(self):
        import sizing_inputs as s
        result=s.check_sizing()
        self.assertEqual(len(result['smt']),3)
        self.assertEqual(result['model']['states'],56)
        self.assertEqual(result['model']['transitions'],32)
        self.assertEqual(result['conformance']['rows'],6214)
        self.assertEqual(result['conformance']['currentFunctionCases'],24856)


class ClosedTradeRecoveryTests(unittest.TestCase):
    def test_closed_trade_recovery_source_mutations(self):
        import closed_trade_recovery as r
        s=(r.ROOT/r.CORE).read_text();r.extract()
        for before,after in [('entryEquity > 0','entryEquity >= 0'),('if finite value','if True'),
                             ('holdingPeriods >= 0','holdingPeriods < 0'),('exitIdx > toInteger','exitIdx < toInteger'),
                             ('go (exitIdx + 1) rest','go exitIdx rest'),
                             ('unless (validTradeMetadata holdingPeriods entryHighVolProb)','unless True')]:
            changed=s.replace(before,after,1)
            self.assertNotEqual(s,changed)
            with self.assertRaises(ValueError):r.extract(changed)

    def test_closed_trade_recovery_conformance(self):
        import closed_trade_recovery as r
        x=r.check_recovery()
        self.assertEqual(len(x['smt']),2)
        self.assertEqual((x['model']['states'],x['model']['transitions']),(22,22))
        self.assertEqual(x['model']['finiteHistories'],341)
        self.assertEqual(x['conformance']['numericRows'],2675)
        self.assertEqual(x['conformance']['indexHistories'],1031)


class InventoryReadinessTests(unittest.TestCase):
    def test_inventory_readiness_source_mutations(self):
        import inventory_readiness as r
        import hashlib
        import json
        source=(r.ROOT/r.MAIN).read_text();registry=r.extract()
        mutations=[('&& not (raiStarting info)', '&& raiStarting info'),
                   ('amount == 0 = True','amount == 0 = False'),
                   ('&& raiSide info == Just positionSide','&& True'),
                   ('                                positions\n                    pure (Right (dedupeStable orphanSymbols', '                                openPositions\n                    pure (Right (dedupeStable orphanSymbols'),
                   ('writeIORef recoveryReadyRef NotReady','writeIORef recoveryReadyRef (ReadyAt 0 0)'),
                   ('orphanScanReady && inventoryReconciled','orphanScanReady && null orphanSymbols')]
        for before,after in mutations:
            changed=source.replace(before,after,1);self.assertNotEqual(source,changed)
            updated=copy.deepcopy(registry);updated['sha256']=hashlib.sha256(changed.encode()).hexdigest()
            # Even a refreshed whole-file hash cannot excuse changed fragments.
            with patch.object(r.json,'loads',return_value=updated):
                with self.assertRaises(ValueError):r.extract(changed)
        for key in registry['fragments']:
            updated=copy.deepcopy(registry);del updated['fragments'][key]
            with patch.object(r.json,'loads',return_value=updated):
                with self.assertRaises(ValueError):r.extract(source)

    def test_inventory_readiness_model_mutations(self):
        import inventory_readiness as r
        def optimistic(state):
            out=r.successors(state)
            return [(label, (t[0],t[1],True,t[3],t[4]) if label=='start-ack' else t) for label,t in out]
        def retain(state):
            out=r.successors(state)
            return [(label, (t[0],t[1],state[2],state[3],t[4]) if label=='clear' else t) for label,t in out]
        for mutation in (optimistic,retain):
            with self.assertRaises(ValueError):r.model(mutation)

    def test_inventory_readiness_conformance(self):
        import inventory_readiness as r
        result=r.check_readiness()
        self.assertEqual((result['model']['states'],result['model']['transitions']),(25,33))
        self.assertEqual(result['conformance']['rows'],6468)
        self.assertEqual(result['conformance']['accepted'],1188)
        self.assertEqual(result['referenceCount'],12)


class PromotionBoundaryTests(unittest.TestCase):
    def setUp(self):
        import promotion_boundary as p
        self.p = p
        self.sources = {name:(p.ROOT/p.source_path(name)).read_text() for name in p.MODULES}
        self.registry = read_json(p.ROOT/p.REGISTRY)

    def refreshed(self, name, source):
        import json
        sources = dict(self.sources); sources[name] = source
        registry = copy.deepcopy(self.registry)
        registry['modules'][name] = json.loads(json.dumps(self.p.inventory(source)))
        return sources, registry

    def test_promotion_roster_mutations(self):
        for name in self.p.MODULES:
            registry = copy.deepcopy(self.registry);del registry['modules'][name]
            with self.assertRaises(ValueError):self.p.extract(self.sources,registry)
        for field in ('callSites','imports','controlFields'):
            registry = copy.deepcopy(self.registry)
            registry['modules']['run_sequential_screen'][field].pop()
            with self.assertRaises(ValueError):self.p.extract(self.sources,registry)

    def test_promotion_effect_mutations_even_with_refreshed_inventory(self):
        mutations=[('run_sequential_screen','import argparse','import socket'),
                   ('run_sequential_screen','ledger.flush()', 'ledger.close()'),
                   ('run_sequential_screen','scale, choose, cfg','scale, arbitrary_policy, cfg')]
        for name,before,after in mutations:
            source=self.sources[name].replace(before,after,1);self.assertNotEqual(source,self.sources[name])
            args=self.refreshed(name,source)
            with self.assertRaises(ValueError):self.p.extract(*args)
        source=self.sources['run_sequential_screen']+'\nexec("pass")\n'
        with self.assertRaises(ValueError):self.p.extract(*self.refreshed('run_sequential_screen',source))

    def test_promotion_destination_mutations_even_with_refreshed_inventory(self):
        mutations=[('run_sequential_screen','output / "training.json"','output.parent / "training.json"'),
                   ('run_sequential_screen','save_policy(artifact, net, provenance)','save_policy(panel, net, provenance)'),
                   ('run_sequential_screen','output / "policies" / (trial','panel / "policies" / (trial'),
                   ('summarize_sequential_screen',"csvfile('experiment-registry.csv'",'csvfile(manifest["name"]'),
                   ('summarize_sequential_screen','summary.update(trainingSecondsSum=', 'summary.update(promotionAllowed=True, trainingSecondsSum=')]
        for name,before,after in mutations:
            source=self.sources[name].replace(before,after,1);self.assertNotEqual(source,self.sources[name])
            with self.assertRaises(ValueError):self.p.extract(*self.refreshed(name,source))

    def test_promotion_writer_and_decoder_mutations(self):
        trees={name:ast.parse(source) for name,source in self.sources.items()}
        for name,function in [('sequential_learning','save_policy'),('ppo_artifact_v4','encode_artifact_v4')]:
            for key,bad in [('promotion','live'),('enabled',True)]:
                changed=copy.deepcopy(trees)
                function_node=self.p.definition(changed[name],function)
                record=self.p.one_assignment(function_node,'value')
                i=next(i for i,k in enumerate(record.keys) if k.value==key)
                record.values[i]=ast.Constant(bad)
                with self.assertRaises(ValueError):self.p.metadata(changed)
        for name,function in [('sequential_learning','load_policy'),('ppo_artifact_v4','decode_artifact_v4')]:
            for fragment in ("['promotion']", "['enabled']"):
                changed=copy.deepcopy(trees)
                function_node=self.p.definition(changed[name],function)
                class Weaken(ast.NodeTransformer):
                    def visit_Compare(self,node):
                        return ast.Constant(False) if fragment in ast.unparse(node) else self.generic_visit(node)
                Weaken().visit(function_node)
                with self.assertRaises((ValueError,RuntimeError)):self.p.metadata(changed)

    def test_promotion_native_overlap_and_live_edge(self):
        trees,_=self.p.extract();facts,_=self.p.metadata(trees)
        changed=copy.deepcopy(facts);changed['v4']['keys']+=['hiddenSize','params','trainBars','version']
        with self.assertRaises((ValueError,AssertionError)):
            self.p.schema(changed,['hiddenSize','params','trainBars','version'])
        with self.assertRaisesRegex(ValueError,'promotion/authority reached'):self.p.lifecycle(facts,mutant=True)

    def test_promotion_receipt_json_roundtrip(self):
        import json
        # The saved graph supplies schema inputs for a transport test only.
        # verify.run separately reproduces the actual graph before composition.
        isolation = read_json(ROOT/'formal/research/results.json')['capabilityIsolation']
        result = self.p.check_promotion(isolation)
        self.assertEqual(result, json.loads(json.dumps(result, allow_nan=False)))

    def test_promotion_conformance(self):
        trees,surface=self.p.extract();facts,meta=self.p.metadata(trees)
        result=self.p.conformance(facts);model=self.p.lifecycle(facts)
        self.assertEqual((surface['moduleCount'],meta['queries']),(20,7))
        self.assertEqual((result['metadataCases'],result['accepted'],result['exclusiveCreateCases']),(72,2,2))
        self.assertGreater(model['rejectedMetadataEdges'],0)
        self.assertGreater(model['states'],0)




class ArtifactCompositionTests(unittest.TestCase):
    def test_snapshot_race_and_compatibility(self):
        import artifact_composition as c
        result=c.conformance()
        self.assertTrue(result['legacyRaceReproduced'])
        self.assertTrue(result['fixedRaceRejected'])
        self.assertEqual(result['rejections'],29)

    def test_source_derived_identity_and_snapshot_model(self):
        import artifact_composition as c
        trees,predicates=c.extract()
        self.assertEqual(c.identity(trees,predicates)[1],6)
        self.assertEqual(c.model()['states'],848)
        self.assertIsNone(c.model()['counterexample'])
        self.assertIsNotNone(c.model(True)['counterexample'])

    def test_snapshot_and_publication_source_mutations_reject(self):
        import artifact_composition as c
        sources={p:(ROOT/p).read_text() for p in c.SOURCES}
        mutations=[(c.SOURCES[0],'validate_provenance(json.loads(identity))','validate_provenance(provenance)'),
                   (c.SOURCES[0],'!= expected_identity','!= json.dumps(expected_provenance, sort_keys=True)'),
                   (c.SOURCES[0],'json.loads(raw,','json.loads(path.read_bytes(),'),
                   (c.SOURCES[0],'net.p = snapshots','net.p = parameters'),
                   (c.SOURCES[1],'return value.copy()','return value'),
                   (c.SOURCES[1],'return restored','return value'),
                   (c.SOURCES[1],'value["version"] == "optimizer-snapshot-v2"','True')]
        for path,old,new in mutations:
            self.assertIn(old,sources[path])
            changed=dict(sources);changed[path]=sources[path].replace(old,new)
            with self.subTest(mutation=new),self.assertRaises(ValueError):c.extract(changed)

    def test_identity_predicate_mutations_find_counterexamples(self):
        import artifact_composition as c
        trees,predicates=c.extract()
        # Bypass source locks deliberately to test the solver, not merely hashing.
        changed=copy.deepcopy(predicates)
        changed[0]=ast.parse('False',mode='eval').body
        with self.assertRaises(RuntimeError):c.identity(trees,changed)
        altered=copy.deepcopy(trees)
        condition=altered[c.SOURCES[1]]['_snapshot'].body[1].value.args[0]
        condition.values=[n for n in condition.values if ast.unparse(n)!="value['version'] == 'optimizer-snapshot-v2'"]
        with self.assertRaises(RuntimeError):c.identity(altered,predicates)


class BotWorkerPublicationTests(unittest.TestCase):
    def test_worker_publication_predicates_and_finite_protocol(self):
        import bot_worker_publication as b
        self.assertEqual(b.extract()['productionCallSites'],1)
        self.assertEqual(b.gates(),{'F-BOT-PUBLICATION-GATES':'unsat'})
        self.assertEqual((b.model()['states'],b.model()['transitions']),(158,308))
        self.assertIsNotNone(b.model(True)['legacyCounterexample'])

    def test_worker_publication_compiled_regressions(self):
        import bot_worker_publication as b
        result=b.conformance()
        self.assertEqual(result['compiledCases'],6)
        self.assertEqual(result['publicationFailureCuts'],3)

    def test_worker_publication_mutations_reject(self):
        import bot_worker_publication as b
        source=(ROOT/b.SOURCE).read_text();main=(ROOT/b.MAIN).read_text()
        for old,new in [('when accepted (unmask action)','unmask action'),
                        ('tryPutMVar gate False','tryPutMVar gate True'),
                        ('modifyMVarMasked','modifyMVar'),
                        ('nextOutcome <- evaluate outcome','let nextOutcome = outcome')]:
            self.assertIn(old,source)
            with self.subTest(mutation=new),self.assertRaises(ValueError):b.extract(core=source.replace(old,new))
        with self.assertRaises(ValueError):b.extract(main=main.replace('publishWorker (bcRuntime ctrl)','modifyMVar (bcRuntime ctrl)'))
        with self.assertRaises(ValueError):b.gates(mutant=True)
        with self.assertRaises(ValueError):b.model(next_states=lambda s,_:b.successors(s,True))
        with self.assertRaises(ValueError):b.prove(z3.BoolVal(False),z3.BoolVal(True))
        with self.assertRaises(ValueError):b.prove(z3.BoolVal(True),z3.BoolVal(False))


class ChampionArchiveTests(unittest.TestCase):
    def test_actual_archive_writers_and_legacy_collisions(self):
        import champion_archive as p
        result = p.conformance()
        self.assertEqual((result['cases'], result['compatibleReportFiles'], result['legacyCounterexamples']), (16, 7, 3))
        self.assertEqual(result['trainingRuns'], 0)

    def test_source_bound_smt_and_interleavings(self):
        import champion_archive as p
        self.assertEqual(len(p.extract()['writerSites']), 5)
        self.assertEqual(p.preserve(), {'F-RL-ARCHIVE-PRESERVE':'unsat'})
        fixed, legacy = p.model(), p.model(True)
        self.assertEqual((fixed['states'], fixed['transitions']), (110, 359))
        self.assertEqual((legacy['states'], legacy['transitions']), (242, 815))
        self.assertEqual(legacy['legacyCounterexample'], ['insert-collision', '0:select-1', '0:open'])
        self.assertEqual(fixed, p.model())

    def test_source_solver_and_composition_mutations(self):
        import champion_archive as p
        source = (ROOT/p.SOURCE).read_text()
        for before, after in [('open("xb")', 'open("wb")'), ('output / name', 'output.parent / name'),
                              ('exist_ok=False', 'exist_ok=True'), ('stream.write(content)', 'stream.write(b"bad")')]:
            changed = source.replace(before, after)
            self.assertNotEqual(changed, source)
            with self.assertRaises(ValueError): p.extract(changed)
        with self.assertRaisesRegex(ValueError, 'counterexample'): p.preserve(False)
        with self.assertRaisesRegex(ValueError, 'unsatisfied'): p.prove(z3.BoolVal(False), z3.BoolVal(True))
        with self.assertRaisesRegex(ValueError, 'production-root composition'):
            p.check_archive({'surface':{'moduleCount':20}, 'composition':{'productionRoots':[]}})
        original = p.successors
        def overwrite(state, legacy=False):
            names, data, phases, handles, keys, injected = state
            return original(state, legacy) + [('bad-write', (names, (0, *data[1:]), phases, handles, keys, injected))]
        with patch.object(p, 'successors', overwrite), self.assertRaisesRegex(ValueError, 'protected object modified'):
            p.model()
        for missing in read_json(ROOT/p.REGISTRY)['supportHashes']:
            registry = read_json(ROOT/p.REGISTRY); del registry['supportHashes'][missing]
            with self.assertRaisesRegex(ValueError, 'support coverage omitted'): p.extract(registry=registry)
        registry = read_json(ROOT/p.REGISTRY); registry['supportHashes'][p.FIXTURE] = '0'*64
        with self.assertRaisesRegex(ValueError, 'support drift'): p.extract(registry=registry)

    def test_composed_receipt_roundtrip(self):
        import champion_archive as p
        import promotion_boundary, json
        # Transport test only; verify.run reproduces the graph before closure.
        prior_graph = read_json(ROOT/'formal/research/results.json')['capabilityIsolation']
        result = p.check_archive(promotion_boundary.check_promotion(prior_graph))
        self.assertEqual(result, json.loads(json.dumps(result, allow_nan=False)))

    def test_champion_closure_requires_every_reproduced_constituent(self):
        from verify import validate_obligations
        ledger = read_json(ROOT/'formal/research/proof-ledger.json')
        target = next(o for o in ledger['missionObligations'] if o['number'] == 28)
        reproduced = {e['requirementId']:e['status'] for e in ledger['entries']}
        # Validator mutation fixture, not substituted proof evidence.
        for certificate in target['requiredCertificates']:
            changed = dict(reproduced); del changed[certificate]
            with self.subTest(certificate=certificate), self.assertRaisesRegex(ValueError, 'not reproduced'):
                validate_obligations(ledger['missionObligations'], ledger['entries'], changed)

    def test_generated_existing_files_are_never_replaced(self):
        import champion_archive as p
        import random, tempfile
        import summarize_sequential_screen as exporter
        rng = random.Random(20261005)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve(); source, kwargs, _ = p.prepare_fixture(root)
            for i in range(32):
                output = root/('case-'+str(i)); output.mkdir()
                path = output/'sentinel'; value = rng.randbytes(rng.randrange(0, 512)); path.write_bytes(value)
                with self.assertRaises(FileExistsError): exporter.export(source, output, **kwargs)
                self.assertEqual(path.read_bytes(), value)
                self.assertEqual(list(output.iterdir()), [path])


class ShieldConsumerTests(unittest.TestCase):
    def test_source_origin_and_repeated_call_model(self):
        import shield_consumers as p
        surface=p.extract()
        self.assertEqual((len(surface['constructors']),len(surface['consumers']),surface['receiverUses']), (4,6,89))
        self.assertEqual(p.origins(surface),{'F-RL-SHIELD-ORIGIN':'unsat'})
        result=p.model()
        self.assertEqual(result,p.model())
        self.assertEqual(result['twoInstanceProductStates'],result['states']**2)
        self.assertEqual(result['twoInstanceProductTransitions'],2*result['states']*result['transitions'])

    def test_actual_shield_consumer_trace_conformance(self):
        import shield_consumers as p
        result=p.conformance()
        self.assertEqual((result['cases'],result['stepCalls'],result['ordinaryTrades'],result['terminalTrades']), (28,187,149,29))
        self.assertEqual(result['marketDataReads'],0)
        self.assertEqual(result['trainingRuns'],0)

    def test_complete_receiver_roster_required(self):
        import shield_consumers as p
        registry=read_json(ROOT/p.REGISTRY)
        with self.assertRaisesRegex(ValueError,'unregistered research consumer'):
            p.research_importers({'scripts/research/unregistered.py':'from sequential_evaluation import replay_policy'})
        for key in ('receiverUses','constructors','consumers'):
            for variant in ('omit','empty','first','last'):
                changed=copy.deepcopy(registry)
                if variant=='omit': del changed[key]
                elif variant=='empty': changed[key]=[]
                elif variant=='first': changed[key].pop(0)
                else: changed[key].pop()
                with self.subTest(key=key,variant=variant),self.assertRaises(ValueError):p.extract(registry=changed)

    def test_semantic_mutations_fail_even_with_updated_inventory(self):
        import ast, shield_consumers as p
        trees,surface=p.promotion_extract()
        for before,after in [('self._trade(self.pending[1])','self._trade(action)'),
                             ('self.execution.extra_delay, proposal','self.execution.extra_delay, action'),
                             ('if proposal is None:', 'if False:'),
                             ('self.pending = None','self.pending = (0, action)')]:
            changed=dict(trees);source=(ROOT/'scripts/research/sequential_env.py').read_text()
            changed['sequential_env']=ast.parse(source.replace(before,after))
            registry={'schemaVersion':1,**p.roster(changed)}
            with patch.object(p,'promotion_extract',return_value=(changed,surface)),self.assertRaises(ValueError):
                p.extract(changed,registry)
        changed=dict(trees)
        changed['sequential_evaluation']=ast.parse((ROOT/'scripts/research/sequential_evaluation.py').read_text().replace('target, ms = action(obs)','target, ms = action(env)'))
        with patch.object(p,'promotion_extract',return_value=(changed,surface)),self.assertRaisesRegex(ValueError,'callback escape'):
            p.extract(changed,{'schemaVersion':1,**p.roster(changed)})

    def test_unsafe_model_edges_and_unsatisfied_premises_fail(self):
        import shield_consumers as p
        from dataclasses import replace
        for mutant in (lambda s: ('target-fill',replace(s,phase='after')),
                       lambda s: ('liquidate-zero',replace(s,phase='done')),
                       lambda s: ('bad-origin',replace(s,pending=1,tagged=False))):
            with self.assertRaises(ValueError):p.model(lambda s:p.local_steps(s)+[mutant(s)])
        with self.assertRaises(RuntimeError):p.certify('nonvacuity',z3.BoolVal(False),z3.BoolVal(True),[])

    def test_generated_shield_rejection_cannot_consume_pending(self):
        import random,numpy as np
        import sequential_env as e
        rng=random.Random(20261005);prices=np.full(40,100.);funding=np.zeros(40);scale=e.Scale.fit([prices[:25]])
        for _ in range(64):
            env=e.Replay(prices,funding,24,35,1,scale,e.Execution(extra_delay=1),enabled=True)
            env.step(rng.choice([-.25,0.,.25]));before=(env.t,env.units,env.equity,env.pending,len(env.rows),env.fills)
            invalid=rng.choice([float('nan'),float('inf'),float('-inf'),rng.uniform(.26,100),True,None])
            env.step(invalid)
            self.assertTrue(env.done)
            self.assertEqual((env.t,env.units,env.equity,env.pending,len(env.rows),env.fills),before)

    def test_closure_requires_every_reproduced_constituent(self):
        from verify import validate_obligations
        ledger=read_json(ROOT/'formal/research/proof-ledger.json')
        target=next(o for o in ledger['missionObligations'] if o['number']==12)
        self.assertTrue(target['requiredCertificates'])
        reproduced={e['requirementId']:e['status'] for e in ledger['entries']}
        for certificate in target['requiredCertificates']:
            changed=dict(reproduced);del changed[certificate]
            with self.subTest(certificate=certificate),self.assertRaisesRegex(ValueError,'not reproduced'):
                validate_obligations(ledger['missionObligations'],ledger['entries'],changed)



class PointInTimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import sys
        sys.path.insert(0,str(ROOT/'scripts/research'))
        import point_in_time_v3 as impl
        import point_in_time as proof
        cls.p=impl;cls.v=proof

    def test_source_mutations_and_omissions_fail(self):
        source=(ROOT/self.v.SOURCE).read_text()
        for before,after in [('enabled: object = False','enabled: object = True'),
                             ('available > decision','available < decision'),
                             ('record.revision > old.revision','record.revision >= old.revision'),
                             ('if admitted is None:','if False:'),
                             ('ambiguous.add(key)','ambiguous.discard(key)'),
                             ('np.asarray(row, dtype="<f8")','np.asarray(row, dtype="<f4")')]:
            mutated=source.replace(before,after,1);self.assertNotEqual(source,mutated)
            with self.assertRaises(ValueError):self.v.extract(mutated)
        registry=read_json(ROOT/self.v.REGISTRY)
        for name in registry['definitions']:
            altered=copy.deepcopy(registry);del altered['definitions'][name]
            with self.assertRaises(ValueError):self.v.extract(source,altered)

    def test_smt_and_model_mutants_fail(self):
        nodes,_=self.v.extract()
        bad=copy.deepcopy(nodes)
        bad['_available']=ast.parse('def _available(record, processing_us):\n return record.close + processing_us').body[0]
        with self.assertRaises(ValueError):self.v.prove(bad)
        with self.assertRaises(ValueError):self.v.check_model(mutant=True)
        self.assertEqual(self.v.prove(nodes),{'F-RL-PIT-TIME':'unsat'})
        self.assertEqual(self.v.check_model()['states'],126)

    def test_unknown_solver_and_refreshed_witness_mutants_fail(self):
        class Unknown:
            def set(self,**kw):pass
            def add(self,*args):pass
            def check(self):return z3.unknown
        nodes,_=self.v.extract()
        with patch.object(self.v.z,'Solver',Unknown),self.assertRaises(ValueError):self.v.prove(nodes)
        import hashlib
        source=(ROOT/self.v.SOURCE).read_text()
        for before,after in [('record.close <= record.first_seen <= record.collected','record.close <= record.first_seen'),
                             ('record.revision > 0 and record.revised is None','False'),
                             ('MAX_TIME = 2**63 - 1','MAX_TIME = 2**64 - 1')]:
            changed=source.replace(before,after,1);tree=ast.parse(changed)
            registry={'schemaVersion':1,'astSha256':hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest(),
                      'definitions':{n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}}
            with self.assertRaises(ValueError):self.v.extract(changed,registry)

    def test_schema_missingness_and_nonfinite_reject(self):
        from dataclasses import replace
        p=self.p;args=self.v.fixture();record=args[0][0]
        bad=[replace(record,first_seen=None),replace(record,collected=None),
             replace(record,first_seen=-1),replace(record,collected=0),
             replace(record,released=100),replace(record,revised=-1),
             replace(record,revision=1),replace(record,revision=True),
             replace(record,close=True),replace(record,symbol='OTHER'),
             replace(record,kind='order'),replace(record,value=float('nan')),
             replace(record,value=float('inf')),replace(record,value=True),
             replace(record,value=0.0),replace(record,value=-1.0)]
        for invalid in bad:
            with self.subTest(record=invalid):
                self.assertIsNone(p.admit_v3((invalid,)+args[0][1:],*args[1:],enabled=True))
        self.assertIsNone(p.admit_v3(args[0][:-1],*args[1:],enabled=True))
        for symbols,closes,decisions,lag in [(('FIXTURE','FIXTURE'),args[2],args[3],0),
              (args[1],args[2][:-1],args[3][:-1],0),(args[1],args[2],args[2],-1),
              (args[1],args[2],(1001,)+args[3][1:],0),(args[1],args[2],args[3],True)]:
            self.assertIsNone(p.admit_v3(args[0],symbols,closes,decisions,lag,enabled=True))

    def test_no_future_payload_read_or_revision_backdate(self):
        from dataclasses import replace
        p=self.p;args=self.v.fixture();baseline=p.admit_v3(*args,5,enabled=True)
        class Poison:
            def __float__(self):raise AssertionError('future value accessed')
        late=replace(args[0][0],revision=1,revised=30,first_seen=40,collected=101,value=Poison())
        self.assertEqual(p.admit_v3(args[0]+(late,),*args[1:],5,enabled=True),baseline)
        timely=replace(late,collected=50,value=101.)
        updated=p.admit_v3(args[0]+(timely,),*args[1:],5,enabled=True)
        self.assertIsNotNone(updated);self.assertEqual(updated.witnesses[0].revision,1)
        self.assertIsNone(p.admit_v3(args[0]+(timely,timely),*args[1:],5,enabled=True))
        self.assertEqual(p.admit_v3(args[0]+(args[0][0],timely),*args[1:],5,enabled=True),updated)

    def test_symbol_isolation_and_temporal_bounds(self):
        from dataclasses import replace
        p=self.p;args=self.v.fixture()
        other=tuple(replace(r,symbol='SECOND',value=200. if r.kind=='price' else 0.) for r in args[0])
        admitted=p.admit_v3(args[0]+other,('FIXTURE','SECOND'),*args[2:],5,enabled=True)
        self.assertIsNotNone(admitted);self.assertNotEqual(admitted.prices[0],admitted.prices[1])
        modified=list(other);modified[0]=replace(modified[0],value=300.)
        updated=p.admit_v3(args[0]+tuple(modified),('FIXTURE','SECOND'),*args[2:],5,enabled=True)
        self.assertEqual(admitted.prices[0],updated.prices[0]);self.assertEqual(admitted.witnesses,updated.witnesses)
        shift=p.MAX_TIME-args[3][-1]
        shifted=tuple(replace(r,close=r.close+shift,first_seen=r.first_seen+shift,collected=r.collected+shift) for r in args[0])
        c=tuple(t+shift for t in args[2]);d=tuple(t+shift for t in args[3])
        self.assertIsNotNone(p.admit_v3(shifted,args[1],c,d,80,enabled=True))
        self.assertIsNone(p.admit_v3(shifted,args[1],c,d,81,enabled=True))
        self.assertGreater(p._available(shifted[-1],p.MAX_TIME),p.MAX_TIME)
        self.assertIsNone(p.admit_v3(shifted,args[1],c,d,p.MAX_TIME,enabled=True))

    def test_default_and_failure_never_publish_or_train(self):
        p=self.p;args=self.v.fixture()
        for value in (False,None,0,1,'true'):
            with patch.object(p,'train_ppo_v2',side_effect=AssertionError('unadmitted')):
                self.assertIsNone(p.train_point_in_time_v3(*args,1,31,1,enabled=value))
        with patch.object(p,'train_ppo_v2',side_effect=AssertionError('bad-version')):
            self.assertIsNone(p.train_point_in_time_v3(*args,1,31,1,enabled=True,version='v2'))
        for failure in (None,ValueError('injected'),MemoryError('injected')):
            with patch.object(p,'train_ppo_v2',return_value=failure if failure is None else None,
                              side_effect=failure if isinstance(failure,Exception) else None):
                self.assertIsNone(p.train_point_in_time_v3(*args,1,31,1,enabled=True))

    def test_private_array_handoff_and_immutable_envelope(self):
        from dataclasses import FrozenInstanceError
        import numpy as np
        p=self.p;args=self.v.fixture();admitted=p.admit_v3(*args,enabled=True)
        def learner(prices,funding,*pos,**kw):
            self.assertEqual(prices['FIXTURE'].tolist(),[100.]*121)
            prices['FIXTURE'][0]=123.;funding['FIXTURE'][0]=3.
            return None
        with patch.object(p,'train_ppo_v2',side_effect=learner):
            self.assertIsNone(p.train_point_in_time_v3(*args,1,31,1,enabled=True))
        self.assertEqual(p.admit_v3(*args,enabled=True),admitted)
        with self.assertRaises(FrozenInstanceError):admitted.processing_us=999
        with self.assertRaises(ValueError):np.frombuffer(admitted.prices[0],dtype='<f8').setflags(write=True)

    def test_oracle_and_actual_training_conformance(self):
        result=self.v.conformance();self.assertEqual(result['generatedOracleCases'],64)
        self.assertEqual(result['syntheticTraining']['steps'],1)

    def test_full_shield_composition_includes_new_entry(self):
        import shield_consumers as c
        from promotion_boundary import extract
        _,surface=extract()
        production=['analyze-close-timing','lstm-bench','merge-top-combos','optimize-equity','outbox-publisher','trader-hs']
        result=c.check_consumers({'surface':surface,'composition':{'productionRoots':production}})
        self.assertEqual(result['surface']['modules'],20)
        with self.assertRaisesRegex(ValueError,'complete consumer composition'):
            c.check_consumers({'surface':dict(surface,moduleCount=12),'composition':{'productionRoots':production}})

    def test_existing_closures_require_new_boundary(self):
        from verify import validate_obligations
        ledger=read_json(ROOT/'formal/research/proof-ledger.json')
        reproduced={e['requirementId']:e['status'] for e in ledger['entries']}
        del reproduced['F-RL-PIT-BOUNDARY']
        with self.assertRaisesRegex(ValueError,'not reproduced'):
            validate_obligations(ledger['missionObligations'],ledger['entries'],reproduced)


class ExactOpeTests(unittest.TestCase):
    def setUp(self):
        import sys
        sys.path.insert(0,str(ROOT/'scripts/research'))
        import ope_rational_v2 as impl
        import ope_rational as proof
        self.p,self.f=impl,proof
        self.e=impl.Episode((1.0,),(1,),(0.5,),(0.5,),(0.0,),(0.0,0.0))

    def test_actual_proof_and_oracle(self):
        result=self.f.check_ope()
        self.assertEqual(result['queries'],12)
        self.assertEqual(result['conformance']['oracleCases'],96)
        self.assertEqual(result['model']['states'],23)

    def test_defaults_and_versions_have_no_arithmetic(self):
        with patch.object(self.p,'_episode',side_effect=AssertionError('arithmetic')):
            for enabled in (False,None,1,'true'):
                self.assertIsNone(self.p.estimate_v2((self.e,),1.0,enabled=enabled))
            for version in (None,2,'ope-rational-v1'):
                self.assertIsNone(self.p.estimate_v2((self.e,),1.0,enabled=True,version=version))

    def test_entire_batch_validates_before_computation(self):
        from dataclasses import replace
        variants=[replace(self.e,rewards=(float('nan'),)),replace(self.e,behavior=(0.0,)),
                  replace(self.e,behavior=(1.1,)),replace(self.e,target=(-0.1,)),
                  replace(self.e,target=(float('inf'),)),replace(self.e,actions=(True,)),
                  replace(self.e,actions=(3,)),replace(self.e,v=(0.0,1.0)),
                  replace(self.e,q=[0.0]),replace(self.e,q=(False,)),
                  replace(self.e,rewards=(1.0,2.0)),None]
        with patch.object(self.p,'_episode',side_effect=AssertionError('arithmetic')):
            for bad in variants:
                with self.subTest(bad=bad):
                    self.assertIsNone(self.p.estimate_v2((self.e,bad),1.0,enabled=True))

    def test_shapes_budgets_and_scalar_types(self):
        from dataclasses import replace
        for batch in ((),[self.e],(self.e,)*257,(replace(self.e,rewards=()),),
                      (replace(self.e,rewards=(1.0,)*33),)):
            self.assertIsNone(self.p.estimate_v2(batch,1.0,enabled=True))
        for gamma in (1,True,float('nan'),float('inf'),-0.1,1.1):
            self.assertIsNone(self.p.estimate_v2((self.e,),gamma,enabled=True))

    def test_extreme_ratios_do_not_overflow_or_underflow(self):
        from dataclasses import replace
        from fractions import Fraction
        tiny=float.fromhex('0x0.0000000000001p-1022')
        for behavior,target in ((tiny,1.0),(1.0,tiny)):
            episode=replace(self.e,behavior=(behavior,),target=(target,))
            got=self.p.estimate_v2((episode,episode),1.0,enabled=True)
            expected=Fraction.from_float(target)/Fraction.from_float(behavior)
            self.assertIsNotNone(got)
            self.assertEqual(got.max_weight,expected)
            self.assertEqual(got.ordinary_is,expected)
            self.assertEqual(got.weighted_is,1)
            self.assertEqual(got.effective_sample_size,2)
            self.assertGreater(got.max_weight,0)

    def test_nonzero_cumulative_underflow_regression(self):
        from fractions import Fraction
        episode=self.p.Episode((1.0,1.0),(1,1),(1.0,1.0),(1e-200,1e-200),
                               (0.0,0.0),(0.0,0.0,0.0))
        got=self.p.estimate_v2((episode,),1.0,enabled=True)
        self.assertEqual(1e-200*1e-200,0.0)
        self.assertIsNotNone(got)
        self.assertEqual(got.max_weight,Fraction.from_float(1e-200)**2)
        self.assertEqual(got.effective_sample_size,1)
        self.assertEqual(got.weighted_is,2)

    def test_zero_support_preserves_absence(self):
        from dataclasses import replace
        got=self.p.estimate_v2((replace(self.e,target=(0.0,)),),0.0,enabled=True)
        self.assertIsNone(got.weighted_is)
        self.assertEqual((got.effective_sample_size,got.nonzero_trajectories),(0,0))
        self.assertFalse(got.reliable)

    def test_size_and_allocation_failure_never_publish(self):
        h=32;tiny=float.fromhex('0x0.0000000000001p-1022')
        episode=self.p.Episode((1.0,)*h,(1,)*h,(tiny,)*h,(1.0,)*h,(0.0,)*h,(0.0,)*(h+1))
        self.assertIsNone(self.p.estimate_v2((episode,),1.0,enabled=True))
        with patch.object(self.p,'_episode',side_effect=[self.p._episode(self.e,self.f.Fraction(1)),MemoryError()]):
            self.assertIsNone(self.p.estimate_v2((self.e,self.e),1.0,enabled=True))

    def test_checked_size_threshold(self):
        from fractions import Fraction
        bound=2**8192
        self.assertEqual(self.p._bounded(Fraction(bound-1)),bound-1)
        for value in (Fraction(bound),Fraction(-bound),Fraction(1,bound)):
            with self.assertRaises(ArithmeticError):self.p._bounded(value)

    def test_result_is_immutable_and_replay_is_exact(self):
        from dataclasses import FrozenInstanceError
        got=self.p.estimate_v2((self.e,),1.0,enabled=True)
        self.assertEqual(got,self.p.estimate_v2((self.e,),1.0,enabled=True))
        with self.assertRaises(FrozenInstanceError):got.reliable=True
        with self.assertRaises(FrozenInstanceError):got.episodes[0].weight=0

    def test_source_mutations_and_refreshed_arithmetic_mutant(self):
        import hashlib,json
        source=(ROOT/self.f.SOURCE).read_text()
        for old,new in [('enabled is not True','enabled is False'),('MAX_BITS = 8192','MAX_BITS = 8193'),
                        ('return _bounded(a / b)','return a / b'),('episode.v[-1] == 0','True')]:
            with self.assertRaises(ValueError):self.f.extract(source.replace(old,new))
        mutant=source.replace('weight = _mul(weight, _div(target, behavior))','weight = _mul(weight, _div(behavior, target))')
        tree=ast.parse(mutant);registry=json.loads((ROOT/self.f.REGISTRY).read_text())
        registry['astSha256']=hashlib.sha256(self.f.shape(tree).encode()).hexdigest()
        registry['definitions']={n.name:self.f.shape(n) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        extracted,_=self.f.extract(mutant,registry)
        with self.assertRaises((ValueError,RuntimeError)):self.f.prove(extracted)

    def test_partial_publication_and_unknown_solver_rejected(self):
        with self.assertRaisesRegex(ValueError,'partial publication'):self.f.check_model(mutant=True)
        with patch('ppo_successor.z.Solver') as solver:
            solver.return_value.check.return_value=z3.unknown
            with self.assertRaises((ValueError,RuntimeError)):self.f.prove(self.f.extract()[0])


class ReportBundleTests(unittest.TestCase):
    def setUp(self):
        import sys
        sys.path.insert(0,str(ROOT/'scripts/research'))
        import report_bundle_v2 as impl
        import report_bundle as proof
        self.b,self.p=impl,proof
        self.reports={name:b'synthetic\n' for name in impl.NAMES}

    def test_actual_models_and_exporter_parity(self):
        result=self.p.check_bundle()
        self.assertEqual((result['model']['states'],result['model']['transitions']),(446,2127))
        self.assertEqual(result['model']['recoverableSameIntentStates'],254)
        self.assertEqual(result['conformance']['reportByteParity'],7)

    def test_default_disabled_before_filesystem(self):
        with patch.object(self.b.os,'open',side_effect=AssertionError('filesystem')):
            for enabled in (False,None,1,'true'):
                self.assertIsNone(self.b.publish_bundle_v2('/invalid',self.reports,enabled=enabled))
            for version in (None,True,'v1'):
                self.assertIsNone(self.b.publish_bundle_v2('/invalid',self.reports,enabled=True,version=version))
            for reports in ({},[],{**self.reports,'unknown':b'x'},dict(self.reports,**{'ope-report.json':b'\xff'})):
                self.assertIsNone(self.b.publish_bundle_v2('/invalid',reports,enabled=True))

    def test_encoding_schema_and_limits(self):
        import json
        raw=self.b.encode_bundle_v2(self.reports);value=json.loads(raw)
        self.assertEqual(set(value),{'schema','promotion','enabled','liveAuthorization','reports'})
        self.assertEqual((value['schema'],value['promotion'],value['enabled'],value['liveAuthorization']),('report-bundle-v2','research',False,False))
        self.assertIsNone(self.b.encode_bundle_v2(dict(self.reports,**{'ope-report.json':b'x'*(self.b.MAX_REPORT+1)})))
        self.assertIsNone(self.b.encode_bundle_v2({name:b'x'*self.b.MAX_REPORT for name in self.b.NAMES}))
        # Valid decoded-size inputs may expand beyond the encoded-byte cap.
        large={name:(b'\x00'*(self.b.MAX_REPORT//2)) for name in self.b.NAMES}
        self.assertIsNone(self.b.encode_bundle_v2(large))

    def test_existing_conflicts_and_links_preserve_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);champion=root/'champion';champion.write_bytes(b'KEEP')
            for kind in ('regular','symlink','hardlink','directory','fifo'):
                out=root/kind;out.mkdir();target=out/self.b.TARGET
                if kind=='regular':target.write_bytes(b'KEEP')
                elif kind=='symlink':target.symlink_to(champion)
                elif kind=='hardlink':target.hardlink_to(champion)
                elif kind=='directory':target.mkdir()
                else:self.b.os.mkfifo(target)
                before=champion.stat().st_ino
                self.assertIsNone(self.b.publish_bundle_v2(str(out),self.reports,enabled=True))
                self.assertEqual(champion.read_bytes(),b'KEEP');self.assertEqual(champion.stat().st_ino,before)
                if kind in ('regular','hardlink'):self.assertEqual(target.read_bytes(),b'KEEP')

    def test_short_write_and_zero_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            original=self.b.os.write
            with patch.object(self.b.os,'write',lambda fd,data:original(fd,data[:3])):
                self.assertIsNotNone(self.b.publish_bundle_v2(tmp,self.reports,enabled=True))
            target=Path(tmp)/self.b.TARGET;expected=target.read_bytes();target.unlink()
            with patch.object(self.b.os,'write',return_value=0):
                self.assertIsNone(self.b.publish_bundle_v2(tmp,self.reports,enabled=True))
            self.assertFalse(target.exists())
            self.assertIsNotNone(self.b.publish_bundle_v2(tmp,self.reports,enabled=True))
            self.assertEqual(target.read_bytes(),expected)

    def test_failure_and_retry_at_effect_boundaries(self):
        import os,stat
        original={name:getattr(os,name) for name in ('open','write','fsync','link','unlink')}
        def classify(name,args):
            if name=='open':return 'stage-open' if args[1]&os.O_CREAT else 'open'
            if name=='fsync':return 'directory-sync' if stat.S_ISDIR(os.fstat(args[0]).st_mode) else 'file-sync'
            return name
        for point in ('stage-open','write','file-sync','link','directory-sync','unlink'):
            for after in (False,True):
                with self.subTest(point=point,after=after), tempfile.TemporaryDirectory() as tmp:
                    fired=[]
                    def hook(name):
                        def call(*args,**kwargs):
                            hit=not fired and classify(name,args)==point
                            if hit and not after:fired.append(True);raise OSError('injected')
                            result=original[name](*args,**kwargs)
                            if hit:fired.append(True);raise OSError('injected')
                            return result
                        return call
                    from contextlib import ExitStack
                    with ExitStack() as stack:
                        for name in original:stack.enter_context(patch.object(os,name,hook(name)))
                        result=self.b.publish_bundle_v2(tmp,self.reports,enabled=True)
                    self.assertTrue(fired)
                    if point!='unlink':self.assertIsNone(result)
                    self.assertIsNotNone(self.b.publish_bundle_v2(tmp,self.reports,enabled=True))
                    self.assertEqual((Path(tmp)/self.b.TARGET).read_bytes(),self.b.encode_bundle_v2(self.reports))

    def test_staging_collision_is_not_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/('.report-bundle-v2.'+'0'*32);path.write_bytes(b'KEEP')
            with patch.object(self.b.secrets,'token_hex',return_value='0'*32):
                self.assertIsNone(self.b.publish_bundle_v2(tmp,self.reports,enabled=True))
            self.assertEqual(path.read_bytes(),b'KEEP')
            self.assertFalse((Path(tmp)/self.b.TARGET).exists())

    def test_process_crashes_before_and_after_boundaries(self):
        import os,subprocess,sys
        script=r'''
import os,stat,sys
import report_bundle_v2 as b
point,when,directory=sys.argv[1:]
original={name:getattr(os,name) for name in ('open','write','fsync','link','unlink')}
def label(name,args):
    if name=='open':return 'stage-open' if args[1]&os.O_CREAT else 'parent-open' if args[1]&os.O_DIRECTORY else 'read-open'
    if name=='fsync':return 'directory-sync' if stat.S_ISDIR(os.fstat(args[0]).st_mode) else 'file-sync'
    return name
def hook(name):
    def call(*args,**kwargs):
        hit=label(name,args)==point
        if hit and when=='before':os._exit(73)
        result=original[name](*args,**kwargs)
        if hit and when=='after':os._exit(73)
        return result
    return call
for name in original:setattr(os,name,hook(name))
b.publish_bundle_v2(directory,{name:b'synthetic\n' for name in b.NAMES},enabled=True)
raise SystemExit(74)
'''
        for point in ('parent-open','stage-open','write','file-sync','link','directory-sync','unlink'):
            for when in ('before','after'):
                with self.subTest(point=point,when=when),tempfile.TemporaryDirectory() as tmp:
                    result=subprocess.run([sys.executable,'-c',script,point,when,tmp],env={'PATH':os.defpath,'PYTHONPATH':str(ROOT/'scripts/research')},capture_output=True,timeout=15)
                    self.assertEqual(result.returncode,73,result.stderr.decode())
                    self.assertIsNotNone(self.b.publish_bundle_v2(tmp,self.reports,enabled=True))
                    target=Path(tmp)/self.b.TARGET;inode=target.stat().st_ino
                    self.assertEqual(target.read_bytes(),self.b.encode_bundle_v2(self.reports))
                    self.assertIsNotNone(self.b.publish_bundle_v2(tmp,self.reports,enabled=True))
                    self.assertEqual(target.stat().st_ino,inode)

    def test_concurrent_matching_and_conflicting_writers(self):
        import os,subprocess,sys
        script="import sys;from report_bundle_v2 import *;r=publish_bundle_v2(sys.argv[1],{n:sys.argv[2].encode() for n in NAMES},enabled=True);print(r or 'absent')"
        for values in (('same','same'),('first','second')):
            with tempfile.TemporaryDirectory() as tmp:
                workers=[subprocess.Popen([sys.executable,'-c',script,tmp,value],env={'PATH':os.defpath,'PYTHONPATH':str(ROOT/'scripts/research')},stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) for value in values]
                outputs=[]
                for worker in workers:
                    out,err=worker.communicate(timeout=15);self.assertEqual(worker.returncode,0,err);outputs.append(out.strip())
                successes=[o for o in outputs if o!='absent']
                self.assertEqual(len(successes),2 if values[0]==values[1] else 1)
                self.assertEqual(len(set(successes)),1)
                import hashlib
                self.assertEqual(successes[0],hashlib.sha256((Path(tmp)/self.b.TARGET).read_bytes()).hexdigest())

    def test_source_and_model_mutants_fail(self):
        import hashlib,json
        source=(ROOT/self.p.SOURCE).read_text()
        for before,after in [('os.O_EXCL','os.O_TRUNC'),('os.fsync(parent)','None'),('enabled is not True','enabled is False')]:
            with self.assertRaises(ValueError):self.p.extract(source.replace(before,after))
        mutant=source.replace('os.O_EXCL','os.O_TRUNC');tree=ast.parse(mutant);reg=read_json(ROOT/self.p.REGISTRY)
        reg['astSha256']=hashlib.sha256(self.p.shape(tree).encode()).hexdigest()
        reg['definitions']={n.name:self.p.shape(n) for n in tree.body if isinstance(n,ast.FunctionDef)}
        reg['effects']=sorted(ast.unparse(n) for n in ast.walk(tree) if isinstance(n,ast.Call) and ast.unparse(n.func).startswith('os.'))
        with self.assertRaisesRegex(ValueError,'exclusive staging'):self.p.extract(mutant,reg)
        with self.assertRaisesRegex(ValueError,'partial target'):self.p.check_model(mutant=True)
        with patch('ppo_successor.z.Solver') as solver:
            solver.return_value.check.return_value=z3.unknown
            with self.assertRaises((ValueError,RuntimeError)):self.p.arithmetic(self.p.extract()[0])


class LifecycleStageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import lifecycle_stage as stage
        cls.s = stage
        cls.trees, _ = stage.extract()
        cls.facts, _ = stage.metadata(cls.trees)

    def test_actual_stage_composition(self):
        r = self.s.check_stages()
        self.assertEqual(r['surface']['researchModules'], 20)
        self.assertEqual(r['queries'], 8)
        self.assertEqual(r['premiseChecks'], 8)
        self.assertEqual((r['model']['states'], r['model']['transitions']), (28, 168))
        self.assertEqual(r['model']['proposedStates'], 2)
        self.assertEqual(r['model']['retainedProposalCases'], 12)
        self.assertEqual(r['conformance']['metadataCases'], 204)
        self.assertEqual(r['conformance']['acceptedResearchCases'], 2)

    def test_missing_stage_guard_fails_without_relying_on_source_hash(self):
        class Remove(ast.NodeTransformer):
            def visit_Compare(self, node):
                return ast.Constant(value=False) if "['promotion']" in ast.unparse(node.left) else self.generic_visit(node)
        for version in self.s.TAGS:
            guards = self.s.predicates(self.trees)
            node, container = guards[version]
            guards[version] = (Remove().visit(copy.deepcopy(node)), container)
            with self.subTest(version=version), self.assertRaises(ValueError):
                self.s.prove_steps(self.facts, guards)

    def test_missing_enable_guard_fails_without_relying_on_source_hash(self):
        class Remove(ast.NodeTransformer):
            def visit_Compare(self, node):
                return ast.Constant(value=False) if "['enabled']" in ast.unparse(node.left) else self.generic_visit(node)
        guards = self.s.predicates(self.trees)
        node, container = guards['v4']
        guards['v4'] = (Remove().visit(copy.deepcopy(node)), container)
        with self.assertRaises(ValueError):
            self.s.prove_steps(self.facts, guards)

    def test_writer_stage_and_format_coverage_cannot_be_relabeled(self):
        for mutate in (lambda x:x['v1'].update(promotion='shadow-eligible'),
                       lambda x:x['v4'].update(enabled=True), lambda x:x.pop('v4')):
            facts=copy.deepcopy(self.facts); mutate(facts)
            with self.assertRaises(ValueError):self.s.prove_steps(facts,self.s.predicates(self.trees))

        # The finite model must derive the stage from metadata too; hard-coding
        # Research in its transfer would hide a bad writer despite the SMT gate.
        import inspect,json
        fixture=read_json(ROOT/'formal/research/fixtures/lifecycle-stage-skip.json')['projectionRegression']
        facts=copy.deepcopy(self.facts); facts[fixture['writerFormat']]['promotion']=fixture['writerTag']
        with self.assertRaisesRegex(ValueError,'promotion counterexample') as raised:
            self.s.check_model(facts)
        self.assertEqual(json.loads(str(raised.exception).split('promotion counterexample: ',1)[1]),fixture['correctTrace'])
        old_transfer=inspect.getsource(self.s.successors)
        for before in ("next_stage=stage_index(facts[version]['promotion'])",'next_stage=stage_index(value) if admitted else stage'):
            self.assertEqual(old_transfer.count(before),1)
            old_transfer=old_transfer.replace(before,'next_stage=stage')
        namespace=dict(vars(self.s));exec(compile(old_transfer,'<old-stage-transfer>','exec'),namespace)
        with patch.object(self.s,'successors',namespace['successors']):
            self.assertEqual(self.s.check_model(facts)['states'],28)

    def test_unknown_metadata_is_not_neutral_research(self):
        for text,is_text in [('unknown',True),('research-only',False),('shadow-eligible',True)]:
            self.assertNotEqual(z3.simplify(self.s.projection(z3.StringVal(text),z3.BoolVal(is_text))).as_long(),0)
        for value in (*self.s.STAGES,*self.s.TAGS.values(),'unknown',None,False,0,[],{}):
            result=z3.simplify(self.s.projection(z3.StringVal(str(value)),z3.BoolVal(type(value) is str))).as_long()
            self.assertEqual(self.s.stage_index(value),result)

    def test_stage_relation_forbids_skips_even_with_all_evidence(self):
        for before in range(7):
            for after in range(7):
                for evidence in (False, True):
                    allowed = z3.is_true(z3.simplify(self.s.lawful(z3.IntVal(before),z3.IntVal(after),
                                                [z3.BoolVal(evidence)]*6,z3.BoolVal(evidence))))
                    self.assertEqual(allowed, after==before or (after==before+1 and after<6 and evidence))

    def test_preserved_stage_skip_counterexample(self):
        import json
        fixture=read_json(ROOT/'formal/research/fixtures/lifecycle-stage-skip.json')
        with self.assertRaisesRegex(ValueError,'promotion counterexample') as raised:
            self.s.check_model(self.facts,mutant=True)
        trace=json.loads(str(raised.exception).split('promotion counterexample: ',1)[1])
        self.assertEqual(trace,fixture['trace'])
        self.assertEqual(fixture['kind'],'model_mutation_not_observed_runtime_defect')

    def test_unknown_solver_and_false_premise_refused(self):
        with patch('ppo_successor.z.Solver') as solver:
            solver.return_value.check.return_value=z3.unknown
            with self.assertRaises(ValueError):self.s.prove_steps(self.facts,self.s.predicates(self.trees))
        with self.assertRaises(ValueError):self.s.certify(z3.BoolVal(False),z3.BoolVal(True))

    def test_closure_requires_every_composed_certificate(self):
        from verify import validate_obligations
        ledger=read_json(ROOT/'formal/research/proof-ledger.json')
        obligation=ledger['missionObligations'][24]
        self.assertEqual(obligation['closureCriteria'],'Candidate promotion follows all mandatory evidence stages without skipping or self-promotion.')
        self.assertEqual(obligation['status'],'exhaustively_checked')
        reproduced={e['requirementId']:e['status'] for e in ledger['entries']}
        for certificate in obligation['requiredCertificates']:
            missing=dict(reproduced); missing.pop(certificate)
            with self.subTest(certificate=certificate), self.assertRaises(ValueError):
                validate_obligations(ledger['missionObligations'],ledger['entries'],missing)


class ReplayAccountingTests(unittest.TestCase):
    @staticmethod
    def modules():
        import sys
        sys.path.insert(0, str(ROOT / 'scripts/research'))
        import replay_accounting_v2 as impl
        import replay_accounting as proof
        from fractions import Fraction
        return impl, proof, Fraction

    def test_source_smt_model_and_haskell(self):
        a,p,F=self.modules()
        self.assertEqual(p.check_accounting()['smt'], {'F-RL-ACCOUNT-V2-ARITH':'unsat'})

    def test_mutants_and_solver_fail_closed(self):
        a,p,F=self.modules()
        source=(ROOT/p.SOURCE).read_text()
        for old,new in [('funding = -_mul', 'funding = _mul'),
                        ('enabled: object = False', 'enabled: object = True'),
                        ('equity = _add(equity, -_debit(closing))','equity = equity'),
                        ('liquidation = "failed"','liquidation = "flat"')]:
            with self.subTest(old=old), self.assertRaises(ValueError):
                p.extract(source.replace(old,new))
        with self.assertRaisesRegex(ValueError,'terminal accounting status'):
            p.model(mutant=True)
        for answer in (z3.sat,z3.unknown):
            with patch('ppo_successor.z.Solver') as solver:
                solver.return_value.check.return_value=answer
                with self.assertRaises(Exception):
                    p.prove(p.extract()[0])

    def test_default_bad_types_and_rejection_atomicity(self):
        a,p,F=self.modules()
        self.assertIsNone(a.initial_v2(F(100)))
        state=a.initial_v2(F(100),enabled=True)
        self.assertIsNone(a.advance_v2(state,F(100),(),F(0)))
        for kw in ({'enabled':1},{'version':'sequential_replay_v1'}, {'fill':F(0)},
                   {'multiplier':F(-1)}, {'impact':float('nan')}, {'terminal':1}):
            args={'enabled':True,**kw}
            self.assertIsNone(a.advance_v2(state,F(100),(),F(0),**args))
        for price in (100.0,True,F(0),F(-1),F(1<<8192)):
            self.assertIsNone(a.advance_v2(state,price,(),F(0),enabled=True))
        for events in ([],((F(1),),),((F(0),F(1)),),((F(1),F(1)),)*129):
            self.assertIsNone(a.advance_v2(state,F(100),events,F(0),enabled=True))
        before=repr(state)
        for error in (ArithmeticError,MemoryError):
            with patch.object(a,'_cost',side_effect=error):
                self.assertIsNone(a.advance_v2(state,F(100),(),F(1,4),enabled=True))
        self.assertEqual(repr(state),before)

    def test_exact_cost_and_terminal_debit(self):
        a,p,F=self.modules()
        s=a.initial_v2(F(100),enabled=True)
        entry=a.advance_v2(s,F(100),(),F(1,4),enabled=True)
        self.assertEqual(entry.after.units,F(1,400))
        self.assertEqual(entry.after.equity,F(3999,4000))
        closed=a.advance_v2(entry.after,F(100),(),F(-1,4),terminal=True,enabled=True)
        self.assertEqual(closed.after.units,0)
        self.assertEqual(closed.after.equity,F(1999,2000))
        self.assertEqual(closed.liquidation,'flat')
        self.assertIsNone(a.advance_v2(closed.after,F(100),(),F(0),enabled=True))

    def test_insolvent_gap_is_explicit_failed_liquidation(self):
        a,p,F=self.modules()
        s=a.initial_v2(F(100),enabled=True)
        entry=a.advance_v2(s,F(100),(),F(-1,4),enabled=True)
        gap=a.advance_v2(entry.after,F(1000),(),F(0),enabled=True)
        self.assertLess(gap.after.equity,0)
        self.assertEqual(gap.after.units,entry.after.units)
        self.assertTrue(gap.after.terminal)
        self.assertEqual((gap.reason,gap.liquidation),('equity_exhausted','failed'))
        self.assertEqual(gap.costs,a.Costs())

    def test_extreme_funding_and_small_debits_are_exact_or_rejected(self):
        a,p,F=self.modules()
        s=a.initial_v2(F(1),enabled=True)
        entry=a.advance_v2(s,F(1),(),F(1,4),multiplier=F(0),enabled=True)
        tiny=F(1,2**1074)
        got=a.advance_v2(entry.after,F(1),((F(1),tiny),),F(0),terminal=True,multiplier=F(0),enabled=True)
        self.assertEqual(got.funding,-tiny/4)
        self.assertEqual(got.after.equity,F(1)-tiny/4)
        # Raw products that overflow binary64 remain exact within the bit budget.
        huge=F(2**1023)
        got=a.advance_v2(entry.after,F(1),((huge,huge),),F(0),terminal=True,enabled=True)
        self.assertEqual(got.funding,-huge*huge/4)
        self.assertEqual(got.liquidation,'failed')
        self.assertIsNone(a.advance_v2(entry.after,F(1),((F(2**8191),F(2**8191)),),F(0),enabled=True))

    def test_exact_sqrt_enclosure(self):
        a,p,F=self.modules()
        for value in [F(0),F(1),F(4),F(1,2),F(1,2**1074),F(2**1023)]:
            bound=a._sqrt_up(value)
            self.assertGreaterEqual(bound*bound,value)
            if bound:
                self.assertLess((bound-F(1,a.SQRT_GRID))**2,value)

    def test_tick_bound_turnover_risk_and_partial_fill(self):
        a,p,F=self.modules()
        s=a.State(4095,F(100),F(1,400),F(1),F(1),False)
        got=a.advance_v2(s,F(100),(),F(1,4),enabled=True)
        self.assertTrue(got.after.terminal)
        self.assertEqual(got.liquidation,'flat')
        s=a.State(0,F(100),F(3,1000),F(1),F(1),False)
        got=a.advance_v2(s,F(100),(),F(-1,4),enabled=True)
        self.assertEqual(got.reason,'turnover_limit')
        self.assertEqual(got.after.units,0)
        s=a.initial_v2(F(100),enabled=True)
        got=a.advance_v2(s,F(100),(),F(1,4),fill=F(1,2),enabled=True)
        self.assertEqual(got.after.units,F(1,800))
        got=a.advance_v2(s,F(100),(),F(1,4),multiplier=F(5000),enabled=True)
        self.assertEqual((got.reason,got.liquidation),('equity_exhausted','failed'))


class FundingEventsTests(unittest.TestCase):
    @staticmethod
    def modules():
        import sys
        sys.path.insert(0, str(ROOT / 'scripts/research'))
        import funding_events_v2 as impl
        import funding_events as proof
        from fractions import Fraction
        return impl, proof, Fraction

    def test_source_smt_exhaustive_model_and_haskell(self):
        a,p,F=self.modules()
        r=p.check_funding_events()
        self.assertEqual(r['smt'],{'F-RL-FUNDING-V2-ARITH':'unsat'})
        self.assertEqual(r['exhaustive']['cases'],256)
        self.assertEqual((r['model']['states'],r['model']['transitions']),(8,12))
        self.assertEqual(r['conformance']['haskellRows'],146)

    def test_mutants_and_solver_fail_closed(self):
        a,p,F=self.modules()
        source=(ROOT/p.SOURCE).read_text()
        for old,new in [('closes[j] < time','closes[j] <= time'),
                        ('enabled: object = False','enabled: object = True'),
                        ('total = _add(total, _mul(mark, rate))','total = total + mark * rate'),
                        ('{0,19}','{0,400}'),
                        ('if mark <= 0:','if mark < 0:'),
                        ('if parsed and time <= parsed[-1][0]:','if parsed and time < parsed[-1][0]:'),
                        ('import re','import re\nimport os')]:
            with self.subTest(old=old), self.assertRaises(ValueError):
                p.extract(source.replace(old,new))
        with self.assertRaisesRegex(ValueError,'publication without every stage'):
            p.model(mutant=True)
        for answer in (z3.sat,z3.unknown):
            with patch('ppo_successor.z.Solver') as solver:
                solver.return_value.check.return_value=answer
                with self.assertRaises(Exception):
                    p.prove(p.extract()[0])

    def test_default_domain_and_atomic_rejection(self):
        a,p,F=self.modules()
        grid=(9,19,29)
        self.assertIsNone(a.load_v2(grid,(('10','1','1'),)))
        for kw in ({'enabled':1},{'version':'funding-events-v1'}):
            self.assertIsNone(a.load_v2(grid,(),**{'enabled':True,**kw}))
        bad=[('10',1.0,'1'),('10','1',True),(10,'1','1'),('10','+1','1'),('10',' 1','1'),('10','1 ','1'),
             ('10','nan','1'),('10','inf','1'),('10','1e3','1'),('10','٣','1'),('10','01','1'),
             ('10','1.','1'),('10','.5','1'),('10','1','0'),('10','1','-0.5'),('-1','1','1'),('010','1','1'),
             ('9223372036854775808','1','1'),('10','1'*21,'1'),('10','0.'+'1'*21,'1')]
        for row in bad:
            with self.subTest(row=row):
                self.assertIsNone(a.load_v2(grid,(row,),enabled=True))
        for closes in ((),[9],(9,9),(9,8),(9.0,),(True,),(-1,),(2**63,)):
            self.assertIsNone(a.load_v2(closes,(),enabled=True))
        self.assertIsNone(a.load_v2(grid,[('10','1','1')],enabled=True))
        self.assertIsNone(a.load_v2(grid,(('10','1','1'),('10','1','1')),enabled=True))
        self.assertIsNone(a.load_v2(grid,(('30','1','1'),),enabled=True))
        with patch.object(a,'_per_unit',side_effect=MemoryError):
            self.assertIsNone(a.load_v2(grid,(('10','1','1'),),enabled=True))

    def test_endpoint_semantics_and_bucket_limit(self):
        a,p,F=self.modules()
        got=a.load_v2((9,19,29),(('0','1','1'),('9','1','2'),('10','1','3'),('19','1','4'),('29','1','5')),enabled=True)
        self.assertEqual(tuple(len(b) for b in got.events),(2,2,1))
        self.assertEqual(got.per_unit,(F(3),F(7),F(5)))
        self.assertEqual(got.events[0],((F(1),F(1)),(F(2),F(1))))
        ok=a.load_v2((1000,),tuple((str(k),'1','1') for k in range(128)),enabled=True)
        self.assertEqual(ok.per_unit,(F(128),))
        self.assertIsNone(a.load_v2((1000,),tuple((str(k),'1','1') for k in range(129)),enabled=True))
        self.assertEqual(a.load_v2((9,),(),enabled=True).events,((),))

    def test_ce_rl_019_witnesses_reject_or_stay_exact(self):
        a,p,F=self.modules()
        huge='%.0f' % 1.7976931348623157e308
        self.assertIsNone(a.load_v2((9,19,29),(('10','2',huge),),enabled=True))
        self.assertIsNone(a.load_v2((9,19,29),(('10','0.75',huge),('11','0.75',huge)),enabled=True))
        top='99999999999999999999.99999999999999999999'
        got=a.load_v2((9,1000),tuple((str(10+k),'-'+top,top) for k in range(128)),enabled=True)
        exact=F(top)
        self.assertEqual(got.per_unit[1],-128*exact*exact)
        self.assertGreater(float(abs(got.per_unit[1])),1e40)
        tiny=a.load_v2((9,),(('1','0.00000000000000000001','0.00000000000000000001'),),enabled=True)
        self.assertEqual(tiny.per_unit,(F(1,10**40),))

    def test_published_value_is_immutable_and_composes_with_replay(self):
        a,p,F=self.modules()
        import replay_accounting_v2 as replay
        got=a.load_v2((9,19),(('10','0.0001','100'),('11','-0.0003','101')),enabled=True)
        with self.assertRaises(Exception):
            got.per_unit=()
        self.assertIsInstance(got.events,tuple)
        state=replay.initial_v2(F(100),enabled=True)
        entry=replay.advance_v2(state,F(100),(),F(1,4),multiplier=F(0),enabled=True)
        step=replay.advance_v2(entry.after,F(100),got.events[1],F(0),terminal=True,multiplier=F(0),enabled=True)
        self.assertEqual(step.funding,-entry.after.units*got.per_unit[1])
        self.assertEqual(got.per_unit[1],F(100)*F(1,10000)+F(101)*F(-3,10000))


class ReplayRunnerTests(unittest.TestCase):
    @staticmethod
    def modules():
        import sys
        sys.path.insert(0, str(ROOT / 'scripts/research'))
        import replay_runner_v2 as impl
        import funding_events_v2 as fe
        import replay_runner as proof
        from fractions import Fraction
        return impl, fe, proof, Fraction

    def test_source_smt_model_oracle_and_metamorphic(self):
        r,fe,p,F=self.modules()
        got=p.check_runner()
        self.assertEqual(got['smt'],{'F-RL-RUNNER-V2-ARITH':'unsat'})
        self.assertEqual((got['model']['states'],got['model']['transitions']),(180,172))
        self.assertEqual((got['conformance']['episodes'],got['conformance']['haskellRows']),(64,157))
        self.assertGreater(got['conformance']['earlyTerminations'],0)
        self.assertEqual(got['metamorphic']['episodes'],64)

    def test_source_mutants_model_mutant_and_solver_fail_closed(self):
        r,fe,p,F=self.modules()
        source=(ROOT/p.SOURCE).read_text()
        for old,new in [('prices[left + k], buckets','prices[left + k + 1], buckets'),
                        ('buckets.events[left + k]','buckets.events[left + k - 1]'),
                        ('terminal=k == len(targets)','terminal=False'),
                        ('enabled: object = False','enabled: object = True'),
                        ('        if state.terminal:\n            break\n',''),
                        ('left + len(targets) >= len(prices)','left + len(targets) > len(prices)')]:
            with self.subTest(old=old), self.assertRaises(ValueError):
                p.extract(source.replace(old,new))
        with self.assertRaisesRegex(ValueError,'publication without terminal episode'):
            p.model(mutant=True)
        for answer in (z3.sat,z3.unknown):
            with patch('ppo_successor.z.Solver') as solver:
                solver.return_value.check.return_value=answer
                with self.assertRaises(Exception):
                    p.prove(p.extract()[0])

    def test_metamorphic_detects_lookahead_variants(self):
        import types
        r,fe,p,F=self.modules()
        source=(ROOT/p.SOURCE).read_text()
        for old,new in [('prices[left + k], buckets','prices[min(left + k + 1, len(prices) - 1)], buckets'),
                        ('buckets.events[left + k]','buckets.events[min(left + k + 1, len(prices) - 1)]')]:
            import sys
            leaky=types.ModuleType('leaky_runner')
            with patch.dict(sys.modules,{'leaky_runner':leaky}):
                exec(compile(source.replace(old,new),'leaky_runner',"exec"),leaky.__dict__)
                with self.subTest(old=old), self.assertRaisesRegex(ValueError,'future data changed'):
                    p.metamorphic(leaky.run_v2)

    def test_default_admission_and_atomic_rejection(self):
        r,fe,p,F=self.modules()
        b=fe.load_v2((9,19,29,39),(('10','0.0001','100'),),enabled=True)
        prices=(F(100),F(101),F(99),F(100))
        t=(F(1,4),F(0),F(0))
        self.assertIsNone(r.run_v2(prices,b,0,t))
        for kw in ({'enabled':1},{'version':'replay-runner-v1'}):
            self.assertIsNone(r.run_v2(prices,b,0,t,**{'enabled':True,**kw}))
        bad=[(prices[:3],b,0,t),(list(prices),b,0,t),(prices,None,0,t),(prices,b,-1,t),(prices,b,True,t),
             (prices,b,1,t),(prices,b,0,()),(prices,b,0,list(t)),(prices,b,0,(F(1,2),F(0),F(0))),
             ((F(100),F(0),F(99),F(100)),b,0,t),((100.0,F(101),F(99),F(100)),b,0,t)]
        for args in bad:
            with self.subTest(args=args):
                self.assertIsNone(r.run_v2(*args,enabled=True))
        forged=fe.Buckets('funding-events-v1',b.closes,b.events,b.per_unit)
        self.assertIsNone(r.run_v2(prices,forged,0,t,enabled=True))
        import replay_accounting_v2 as acc
        with patch.object(acc,'advance_v2',side_effect=[acc.advance_v2(acc.initial_v2(F(100),enabled=True),F(101),(),F(1,4),enabled=True),None]):
            self.assertIsNone(r.run_v2(prices,b,0,t,enabled=True))
        with patch.object(acc,'advance_v2',side_effect=MemoryError):
            self.assertIsNone(r.run_v2(prices,b,0,t,enabled=True))

    def test_bucket_alignment_terminal_and_early_stop(self):
        r,fe,p,F=self.modules()
        b=fe.load_v2((9,19,29,39),(('5','1','1'),('10','0.0001','100'),('25','-0.0002','101')),enabled=True)
        prices=(F(100),F(101),F(99),F(100))
        ep=r.run_v2(prices,b,0,(F(1,4),F(1,4),F(0)),enabled=True)
        self.assertEqual(len(ep.receipts),3)
        self.assertEqual(ep.receipts[0].funding,0)  # bucket 0 prehistory never charged; flat at step 1
        self.assertEqual(ep.receipts[1].funding,-ep.receipts[0].after.units*F(-101,5000))
        self.assertTrue(ep.final.terminal and ep.final.units==0 and ep.receipts[-1].liquidation=='flat')
        with self.assertRaises(Exception):
            ep.receipts=()
        stop=r.run_v2((F(100),F(100),F(1000),F(1000)),b,0,(F(-1,4),F(0),F(0)),enabled=True)
        self.assertEqual(len(stop.receipts),2)
        self.assertEqual(stop.receipts[-1].reason,'equity_exhausted')
        self.assertEqual(stop.receipts[-1].liquidation,'failed')


class ValueObjectiveV2Tests(unittest.TestCase):
    @staticmethod
    def modules():
        import sys
        sys.path.insert(0, str(ROOT / 'scripts/research'))
        import value_objective_v2 as impl
        import value_objective_successor as proof
        return impl, proof

    def test_source_smt_bounds_model_and_oracles(self):
        v,p=self.modules()
        r=p.check_value_objective()
        self.assertEqual(r['smt'],{'F-RL-VALUE-V2-ARITH':'unsat'})
        self.assertEqual((r['bounds']['lossBoundLog2'],r['bounds']['gradientBoundLog2']),(201,101))
        self.assertEqual((r['model']['states'],r['model']['transitions']),(6,9))
        self.assertEqual((r['conformance']['oracleBatches'],r['conformance']['frozenDifferential']),(96,96))
        self.assertGreater(r['conformance']['finiteDifferenceChecks'],0)

    def test_mutants_and_solver_fail_closed(self):
        v,p=self.modules()
        source=(ROOT/p.SOURCE).read_text()
        for old,new in [('if alpha > 0.0:','if alpha >= 0.0:'),
                        ('gap = (m - row[action]) + math.log(total)','gap = m + math.log(total) - row[action]'),
                        ('    loss += 0.0\n',''),
                        ('tuple(g + 0.0 for g in r)','tuple(g for g in r)'),
                        ('abs(x) <= BOUND','abs(x) <= BOUND * 2'),
                        ('enabled: object = False','enabled: object = True'),
                        ('import math','import math\nimport numpy as np'),
                        ('math.fsum(e * e for e in residuals)','sum(e * e for e in residuals)')]:
            with self.subTest(old=old), self.assertRaises(ValueError):
                p.extract(source.replace(old,new))
        with self.assertRaisesRegex(ValueError,'publication without finite guard'):
            p.model(mutant=True)
        for answer in (z3.sat,z3.unknown):
            with patch('ppo_successor.z.Solver') as solver:
                solver.return_value.check.return_value=answer
                with self.assertRaises(Exception):
                    p.prove(p.extract()[0])

    def test_oracle_rejects_frozen_style_penalty(self):
        import math
        v,p=self.modules()
        def frozen_style(row, action):
            m=max(row); t=[math.exp(x-m) for x in row]; total=math.fsum(t)
            return (m+math.log(total))-row[action], tuple(x/total for x in t), 0
        S=2.0**54
        with patch.object(v,'_penalty',side_effect=frozen_style):
            shifted=v.objective_v2(((S,S,S),),(0,),(S,),0.1,enabled=True)
        self.assertEqual(shifted.loss,0.0)  # the CE-RL-015 defect reappears
        with patch.object(v,'_penalty',side_effect=frozen_style):
            with self.assertRaisesRegex(ValueError,'CE-RL-015'):
                p.witnesses()

    def test_domain_default_and_atomic_rejection(self):
        import math
        v,p=self.modules()
        ok=(((0.0,1.0,2.0),),(1,),(0.5,),0.1)
        self.assertIsNone(v.objective_v2(*ok))
        for kw in ({'enabled':1},{'version':'value-objective-v1'}):
            self.assertIsNone(v.objective_v2(*ok,**{'enabled':True,**kw}))
        B=2.0**100
        bad=[(((0.0,1.0),),(1,),(0.5,),0.1),(((0.0,1.0,2.0),),(3,),(0.5,),0.1),(((0.0,1.0,2.0),),(True,),(0.5,),0.1),
             (((0.0,1.0,2.0),),(1,),(0.5,),1.5),(((0.0,1.0,2.0),),(1,),(0.5,),-0.0-1e-300),(((0.0,1.0,2.0),),(1,),(0.5,),1),
             (((0.0,1.0,math.nan),),(1,),(0.5,),0.1),(((0.0,1.0,math.inf),),(1,),(0.5,),0.1),(((0.0,1.0,B*2),),(1,),(0.5,),0.1),
             (((0,1.0,2.0),),(1,),(0.5,),0.1),([(0.0,1.0,2.0)],(1,),(0.5,),0.1),((),(),(),0.1),
             (((0.0,1.0,2.0),)*257,(1,)*257,(0.5,)*257,0.1),(((0.0,1.0,2.0),),(1,),(0.5,0.5),0.1)]
        for args in bad:
            with self.subTest(args=str(args)[:60]):
                self.assertIsNone(v.objective_v2(*args,enabled=True))
        self.assertIsNotNone(v.objective_v2(((B,-B,0.0),),(1,),(B,),1.0,enabled=True))
        with patch.object(v,'_penalty',side_effect=MemoryError):
            self.assertIsNone(v.objective_v2(*ok,enabled=True))
        with patch.object(v.math,'fsum',return_value=math.inf):
            self.assertIsNone(v.objective_v2(*ok,enabled=True))

    def test_alpha_zero_skips_penalty_and_underflow_is_reported(self):
        v,p=self.modules()
        with patch.object(v,'_penalty',side_effect=AssertionError('evaluated')):
            got=v.objective_v2(((1.0,2.0,3.0),),(0,),(0.0,),0.0,enabled=True)
        self.assertEqual((got.loss,got.underflow),(0.5,0))
        far=v.objective_v2(((0.0,-1000.0,-2000.0),),(0,),(0.0,),0.5,enabled=True)
        self.assertEqual(far.underflow,2)
        self.assertEqual(far.loss,0.0)
        with self.assertRaises(Exception):
            far.loss=1.0


class SplitIsolationTests(unittest.TestCase):
    @staticmethod
    def proof():
        import split_isolation as proof
        return proof

    def test_source_regions_registered_and_conformance(self):
        p=self.proof()
        r=p.check_split()
        self.assertEqual(r['smt'],{'F-RL-SPLIT-REGIONS':'unsat'})
        self.assertEqual((r['source']['foldBindings'],r['source']['fullPanelUses'],r['source']['replayReads']),(9,5,6))
        self.assertEqual(r['registered']['holdoutStartOpenTime']-r['registered']['lastDevelopmentClose'],1)
        self.assertEqual(r['conformance']['checks'],18)

    def test_source_mutants_fail_closed(self):
        p=self.proof()
        cases={p.RUNNER:[('p[:split["trainStop"]] for s, p in prices.items()','p[:split["testStart"]] for s, p in prices.items()'),
                         ('split["testStart"], split["testStop"], h, scale, choose, cfg','split["testStart"], split["testStop"] + 1, h, scale, choose, cfg'),
                         ("controls = Baselines(train, funds, scale, h)","controls = Baselines(prices, funding, scale, h)"),
                         ("net = None\n","net = net\n")],
               p.ENV:[("end = min(self.t + self.horizon, self.stop - 1)","end = min(self.t + self.horizon, self.stop)"),
                      ("f = self.funding[left + 1]","f = self.funding[left + 2]")],
               p.EVAL:[("t = int(rng.integers(start, stop - 6 * horizon))","t = int(rng.integers(start, stop))"),
                       ("y.append(p[t + 1 + horizon] / p[t + 1] - 1)","y.append(p[t + 2 + horizon] / p[t + 1] - 1)")]}
        for path,pairs in cases.items():
            source=(ROOT/path).read_text()
            for old,new in pairs:
                self.assertIn(old,source)
                with self.subTest(path=path,old=old), self.assertRaises(ValueError):
                    p.bind({path:source.replace(old,new)})

    def test_solver_and_registration_fail_closed(self):
        p=self.proof()
        for answer in (z3.sat,z3.unknown):
            with patch('ppo_successor.z.Solver') as solver:
                solver.return_value.check.return_value=answer
                with self.assertRaises(Exception):
                    p.regions()
        real=p.json.loads
        def overlap(text):
            value=real(text)
            if isinstance(value,dict) and 'validation' in value and 'outerFolds' in value['validation']:
                value['validation']['outerFolds'][0]['testStop']=5000
            return value
        with patch.object(p.json,'loads',side_effect=overlap):
            with self.assertRaisesRegex(ValueError,'fold inside panel'):
                p.registered()

    def test_conformance_detects_future_read(self):
        import sys
        sys.path.insert(0,str(ROOT/'scripts/research'))
        import sequential_evaluation as ev
        p=self.proof()
        original=ev.Replay
        class Leaky(original):
            def __init__(self,prices,funding,start,stop,*a,**k):
                super().__init__(prices,funding,start,min(stop+3,len(prices)),*a,**k)
        with patch.object(ev,'Replay',Leaky):
            with self.assertRaisesRegex(ValueError,'after testStop|outside validation region'):
                p.conformance()


class ObservationCausalityTests(unittest.TestCase):
    def test_witness_and_fixed_scale_probe(self):
        import observation_causality as o
        r=o.check_observation()
        self.assertEqual(r['trainScale']['counterexample'],'CE-RL-025')
        self.assertGreater(r['trainScale']['maxAbsObservationDifference'],1.0)
        self.assertEqual(r['replayCausal']['decisionChecks'],250)

    def test_source_drift_fails_closed(self):
        import observation_causality as o
        runner=(ROOT/o.RUNNER).read_text(); env=(ROOT/o.ENV).read_text()
        for path,old,new in [(o.RUNNER,'scale = Scale.fit(list(train.values()))','scale = Scale.fit(list(train.values())[:1])'),
                             (o.ENV,'start = int(rng.integers(24, len(p) - 96))','start = int(rng.integers(30, len(p) - 96))'),
                             (o.ENV,'normalized = self.scale.transform(x)','normalized = x')]:
            source=runner if path==o.RUNNER else env
            self.assertIn(old,source)
            with self.subTest(old=old), self.assertRaises(ValueError):
                o.bind({path:source.replace(old,new)})

    def test_probe_detects_lookahead(self):
        import sys
        sys.path.insert(0,str(ROOT/'scripts/research'))
        import sequential_env as env
        import observation_causality as o
        original=env.market_features
        def peek(prices,t):
            x=original(prices,t)
            return None if x is None else x+0*prices[min(t+1,len(prices)-1)]+1e-3*prices[min(t+1,len(prices)-1)]
        with patch.object(env,'market_features',peek):
            with self.assertRaisesRegex(ValueError,'depends on data after t'):
                o.replay_causal()


class BoundedValuesTests(unittest.TestCase):
    def test_source_smt_and_stress_probe(self):
        import bounded_values as b
        r=b.check_bounds()
        self.assertEqual(r['smt'],{'F-RL-BOUNDS-COMPOSE':'unsat'})
        self.assertEqual((r['source']['riskClauses'],r['source']['stresses']),(5,9))
        c=r['conformance']
        self.assertEqual(c['episodes'],288)
        self.assertGreater(c['breachTerminations'],0)
        self.assertLess(c['flatTerminals'],c['episodes'])

    def test_source_mutants_fail_closed(self):
        import bounded_values as b
        env=(ROOT/b.ENV).read_text(); runner=(ROOT/b.RUNNER).read_text()
        for path,old,new in [(b.ENV,'if self.equity < 0.8:','if self.equity < 0.7:'),
                             (b.ENV,'> 0.35:','> 0.5:'),
                             (b.ENV,'if isfinite(self.equity) and self.equity > 0:\n                    liquidation','if False:\n                    liquidation'),
                             (b.ENV,'if not terminal and turnover > 0.50 + 1e-12:','if not terminal and turnover > 0.90:'),
                             (b.ENV,'action not in (-0.25, 0.0, 0.25)','action not in (-0.5, 0.0, 0.5)'),
                             (b.RUNNER,'Execution(cost_multiplier=2.5)','Execution(cost_multiplier=4)')]:
            source=env if path==b.ENV else runner
            self.assertIn(old,source)
            with self.subTest(old=old), self.assertRaises(ValueError):
                b.bind({path:source.replace(old,new)})
        for answer in (z3.sat,z3.unknown):
            with patch('ppo_successor.z.Solver') as solver:
                solver.return_value.check.return_value=answer
                with self.assertRaises(Exception):
                    b.prove()

    def test_propagation_fails_closed_on_weaker_arithmetic(self):
        import bounded_values as b
        with patch.object(b,'U',b.Q(1,100)):
            with self.assertRaisesRegex(ValueError,'exact bound propagation failed'):
                b.propagate()
        r=b.propagate()
        self.assertGreater(r['equityAfterBarRatio'],0.8)
        self.assertLess(r['detectionExposure'],0.66)

    def test_probe_detects_out_of_bounds_state(self):
        import sys
        sys.path.insert(0,str(ROOT/'scripts/research'))
        import sequential_env as env
        import bounded_values as b
        import math
        with patch.object(env.Replay,'_risk',lambda self: None if math.isfinite(self.equity) and self.equity > 0 else 'equity_exhausted'):
            with self.assertRaisesRegex(ValueError,'outside realized bounds|not liquidated flat'):
                b.conformance()


class AccountingReconciliationTests(unittest.TestCase):
    def test_source_lemmas_constants_and_ledger_probe(self):
        import accounting_reconciliation as a
        r=a.check_reconciliation()
        self.assertEqual(r['smt'],{'F-RL-RECON-ERROR':'unsat'})
        self.assertEqual((r['certified']['rollForwardRoundings'],r['certified']['rollForwardQueries']),(14,15))
        self.assertEqual(r['certified']['terms']['impact']['multiplications'],3)
        self.assertGreater(r['conformance']['rows'],1000)

    def test_source_mutants_and_bad_constants_fail_closed(self):
        import accounting_reconciliation as a
        env=(ROOT/a.ENV).read_text()
        for old,new in [("self.equity += gross + funding","self.equity += gross"),
                        ("costs = {k: costs[k] + liquidation[k] for k in costs}","costs = liquidation"),
                        ('"spread": cash * 0.00005 * cfg.cost_multiplier','"spread": cash * 0.0 * cfg.cost_multiplier'),
                        ('"equity": self.equity, "gross": gross','"equity": self.equity + 0.0, "gross": gross'),
                        ("            if self.done:\n                break\n","            self.equity -= 0.0\n            if self.done:\n                break\n"),
                        ("            terminal = self.failure is not None","            costs['fee'] = 0.0\n            terminal = self.failure is not None"),
                        ("            terminal = self.failure is not None","            costs.update(fee=0.0)\n            terminal = self.failure is not None"),
                        ("            terminal = self.failure is not None","            if self.t % 7 == 0:\n                alias = costs\n                alias['fee'] = 0.0\n            terminal = self.failure is not None"),
                        ("        self.units = new\n","        view = terms\n        view['fee'] = 0.0\n        self.units = new\n"),
                        ("                **costs})","                **costs, **({'fee': 0.0} if self.t % 7 == 0 else {})})"),
                        ("                costs = self._trade(self.pending[1])\n","                costs = self._trade(self.pending[1])\n                if self.t % 7 == 0:\n                    self._trade(0.0)\n"),
                        ("                costs = self._trade(self.pending[1])\n","                costs = self._trade(self.pending[1])\n                if self.t % 7 == 0:\n                    Replay._trade(self, 0.0)\n"),
                        ("        self.units = new\n","        self.units = new\n        if self.t % 5 == 0:\n            return terms\n"),
                        ("            terminal = self.failure is not None","            if self.t % 3 == 0:\n                costs = dict.fromkeys(('turnover', 'fee', 'spread', 'slippage', 'impact'), 0.0)\n            terminal = self.failure is not None")]:
            self.assertIn(old,env)
            with self.subTest(old=old), self.assertRaises(ValueError):
                a.bind({a.ENV:env.replace(old,new)})
        with patch.object(a,'STATED',dict(a.STATED,costTerm=(a.Q(3),a.Q(8)))):
            with self.assertRaisesRegex(ValueError,'stated bound below the AST-derived certified constant'):
                a.lemmas()
        with patch.object(a,'STATED',dict(a.STATED,rollForward=(a.Q(14),a.Q(14)))):
            with self.assertRaises(ValueError):     # 14 is a hair too tight: the generated Z3 query is satisfiable
                a._roll_forward()
        for answer in (z3.sat,z3.unknown):
            with patch('ppo_successor.z.Solver') as solver:
                solver.return_value.check.return_value=answer
                with self.assertRaises(Exception):
                    a.lemmas()

    def test_transitive_helper_edit_and_sum_semantics(self):
        import accounting_reconciliation as a
        env=(ROOT/a.ENV).read_text()
        old='def _finite_real(value) -> bool:\n'
        self.assertIn(old,env)
        mutated=env.replace(old,old+'    return True\n',1)
        with self.assertRaisesRegex(ValueError,'reviewed source lock drift: module'):
            a.bind({a.ENV:mutated})
        self.assertTrue(a.sum_semantics())

    def test_probe_detects_unrecorded_debit(self):
        import sys
        sys.path.insert(0,str(ROOT/'scripts/research'))
        import sequential_env as env
        import accounting_reconciliation as a
        original=env.Replay._trade
        def leaky(self,target,terminal=False):
            terms=original(self,target,terminal)
            self.equity-=1e-6   # an unrecorded debit
            return terms
        with patch.object(env.Replay,'_trade',leaky):
            with self.assertRaisesRegex(ValueError,'roll-forward outside the stated bound'):
                a.conformance()


class AsyncStopTests(unittest.TestCase):
    def test_source_and_compiled_witness(self):
        import async_stop as a
        r = a.check_async_stop()
        self.assertEqual(r['source']['modules'], len(list((ROOT / 'haskell/app').rglob('*.hs'))))
        self.assertIn('follow-up order', r['conformance']['observed'])

    def test_reintroduced_capture_fails_closed(self):
        import async_stop as a
        main = (ROOT / a.MAIN).read_text()
        anchor = 'import Control.Exception ('
        cases = [
            ('imports capture', main.replace(anchor, anchor + 'try, ', 1)),
            ('explicit unqualified list', main.replace(anchor, 'import qualified Control.Exception as E\n' + anchor, 1)),
            ('explicit unqualified list', main.replace(anchor, 'import Control.Exception\n' + anchor, 1)),
            ('imports capture', main.replace(anchor, 'import Control.Monad.Catch (handle)\n' + anchor, 1)),
            ('unreviewed capture', main + '\nleak = tryForwardingAll (pure ())\n'),
        ]
        for reason, text in cases:
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                a.bind({a.MAIN: text})
        cache = 'haskell/app/Trader/Cache.hs'
        with self.assertRaisesRegex(ValueError, 'imports capture'):
            a.bind({cache: (ROOT / cache).read_text().replace(anchor, anchor + 'catch, ', 1)})
        helper = (ROOT / a.ASYNC).read_text()
        with self.assertRaisesRegex(ValueError, 'async exception|async-safe fragment'):
            a.bind({a.ASYNC: helper.replace('Left ex | isAsyncException (toException ex) -> throwIO ex', 'Left ex | False -> throwIO ex')})


class PositionOwnershipTests(unittest.TestCase):
    def test_source_model_and_compiled_regression(self):
        import position_ownership as o
        r = o.check_position_ownership()
        self.assertEqual(r['source']['orderCapableDefinitions'], len(o.CLASSIFICATION))
        self.assertIsNone(r['model']['fixed']['violation'])
        self.assertEqual(set(r['model']['counterexamples']), {'CE-LIVE-002', 'CE-LIVE-003', 'no-bot-claim-check'})

    def test_weakened_source_fails_closed(self):
        import position_ownership as o
        main = (ROOT / o.MAIN).read_text()
        claimed = 'runManualTrade botCtrl argsFinal (computeTradeFromArgsWithLimits limits mOps argsFinal)'
        self.assertEqual(main.count(claimed), 2)
        cases = [
            ('not run under a manual-trade claim', main.replace(claimed, '(computeTradeFromArgsWithLimits limits mOps argsFinal)', 1)),
            ('does not refuse a claimed or owned identity', main.replace('else botOwnerConflict ctrl mrt mOwnerKey', 'else pure Nothing', 1)),
            ('no longer reduce-only', main.replace('OrderLive sym side qty (Just True)', 'OrderLive sym side qty Nothing', 1)),
            ('order-capable definitions drifted', main + '\nsneakyOrder :: IO ()\nsneakyOrder = void (placeMarketOrder env OrderLive sym Buy q Nothing Nothing Nothing)\n'),
            ('account identity differs', main.replace('<$> resolveEnv "BINANCE_API_KEY" (argBinanceApiKey args)', '<$> pure (argBinanceApiKey args)', 1)),
            ('bot owner identity changed', main.replace('bsTradeEnabled settings && (argPlatform args /= PlatformBinance || argBinanceLive args) =', 'bsTradeEnabled settings && manualTradeMayBeLive args =', 1)),
            ('queued before the ownership refusal', main.replace('ownedNow <- manualTradeOwnedNow botCtrl argsFinal', 'ownedNow <- pure Nothing', 1)),
        ]
        for reason, text in cases:
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                o.bind({o.MAIN: text})
        helper = (ROOT / o.OWNERSHIP).read_text()
        with self.assertRaisesRegex(ValueError, 'manual claim protocol changed'):
            o.bind({o.OWNERSHIP: helper.replace('uninterruptibleMask_ $', 'id $', 1)})

    def test_role_gate_and_account_identity_fail_closed(self):
        import position_ownership as o
        r = o.check_position_ownership()['roleGate']
        self.assertEqual(r['source']['gatedPrimitives'], 10)
        binance = (ROOT / 'haskell/app/Trader/Binance.hs').read_text()
        gate = '    Control.Monad.when (mode == OrderLive) requireLiveOrderRole\n'
        self.assertEqual(binance.count(gate), 5)
        cases = [
            ('role gate is not the first action', {'haskell/app/Trader/Binance.hs': binance.replace(gate, '', 1)}),
            ('ungated venue action', {'haskell/app/Trader/Coinbase.hs': (ROOT / 'haskell/app/Trader/Coinbase.hs').read_text() + '\nplaceCoinbaseLimitOrder :: IO ()\nplaceCoinbaseLimitOrder = do\n    pure ()\n'}),
            ('lost its role label', {'deploy/hetzner/trader.research.env.managed': (ROOT / 'deploy/hetzner/trader.research.env.managed').read_text().replace('TRADER_SERVER_ROLE=research', 'TRADER_SERVER_ROLE=standalone')}),
            ('role gate decision changed', {o.LIVE_ROLE: (ROOT / o.LIVE_ROLE).read_text().replace('"research", "read-only", "readonly", "fly"]', '"read-only", "readonly", "fly"]')}),
        ]
        for reason, sources in cases:
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                o.bind_role_gate(sources)
        main = (ROOT / o.MAIN).read_text()
        with self.assertRaisesRegex(ValueError, 'fail-closed account UID'):
            o.bind({o.MAIN: main.replace('| argPlatform args == PlatformBinance && not (argBinanceTestnet args) -> do', '| False -> do', 1)})
        with self.assertRaisesRegex(ValueError, 'echo request headers'):
            o.bind({o.MAIN: main.replace('Just _ -> "Binance account request failed."', 'Just _ -> displayException ex', 1)})


class CausalReplayV3Tests(unittest.TestCase):
    def setUp(self):
        import causal_replay
        self.proof = causal_replay
        self.c = causal_replay.modules()
        from fractions import Fraction as F
        self.F = F
        self.bars = tuple(self.c.Bar("BTC", i * 1000, i * 1000 + 10, F(100 + i % 3)) for i in range(60))
        self.session = self.c.start_v3(self.bars[:50], enabled=True)
        self.assertIsNotNone(self.session)

    def test_source_smt_and_model(self):
        self.proof.extract()
        self.assertEqual(self.proof.prove(), {'F-RL-CAUSAL-V3-ARITH': 'unsat'})
        self.assertGreater(self.proof.model()['states'], 0)
        with self.assertRaises(ValueError):
            self.proof.model(mutant=True)

    def test_future_fit_and_shield_mutants_reject(self):
        source = (ROOT / self.proof.SOURCE).read_text()
        for old, new in (('_fit(bars[:-1])', '_fit(bars)'),
                         ('else F(0)', 'else target'),
                         ('session.scale, receipt.after', '_fit(session.history), receipt.after'),
                         ('bars[-25:]', 'bars[:25]')):
            self.assertIn(old, source)
            with self.assertRaises(ValueError):
                self.proof.extract(source.replace(old, new))

    def test_disabled_version_invalid_and_late_inputs(self):
        c, F = self.c, self.F
        self.assertIsNone(c.start_v3(self.bars[:50]))
        self.assertIsNone(c.observe_v3(self.session))
        self.assertIsNone(c.step_v3(self.session, self.bars[50], (), F(0)))
        for version in ('sequential_replay_v1', None, 3, True):
            self.assertIsNone(c.start_v3(self.bars[:50], enabled=True, version=version))
        for bar in (None, c.Bar("BTC", 50000, 50010, F(-1)), c.Bar("BTC", 50000, 49999, F(100)),
                    c.Bar("BTC", 50000, 50010, float('nan')), c.Bar("BTC", 49010, 50010, F(100)),
                    c.Bar("BTC", True, 50010, F(100)), c.Bar("BTC", 50000, 2**63, F(100))):
            self.assertIsNone(c.step_v3(self.session, bar, (), F(0), enabled=True))
        for action in (True, float('nan'), float('inf'), F(1), 0.25):
            self.assertIsNone(c.step_v3(self.session, self.bars[50], (), action, enabled=True))
        self.assertEqual(self.session.history, self.bars[:50])

    def test_symbol_isolation(self):
        c, F = self.c, self.F
        other = c.Bar("ETH", 50000, 50010, F(100))
        self.assertIsNone(c.start_v3(self.bars[:49] + (other,), enabled=True))
        self.assertIsNone(c.step_v3(self.session, other, (), F(0), enabled=True))
        self.assertEqual(self.session.history[-1].symbol, "BTC")

    def test_scale_cannot_be_injected_or_refitted_from_later_rows(self):
        from dataclasses import replace
        c, F = self.c, self.F
        modified = self.bars[:50] + tuple(c.Bar("BTC", b.close, b.available, F(500)) for b in self.bars[50:])
        self.assertEqual(c.start_v3(modified[:50], enabled=True), self.session)
        contaminated = c._fit(modified)
        self.assertNotEqual(contaminated, self.session.scale)
        self.assertIsNone(c.observe_v3(replace(self.session, scale=contaminated), enabled=True))
        self.assertIsNone(c.observe_v3(replace(self.session, burn=26), enabled=True))
        # The initial decision bar also does not enter the fitted scale.
        changed = self.bars[:49] + (c.Bar("BTC", 49000, 49010, F(103)),)
        other = c.start_v3(changed, enabled=True)
        self.assertEqual(other.scale, self.session.scale)
        self.assertNotEqual(c.observe_v3(other, enabled=True).values, c.observe_v3(self.session, enabled=True).values)

    def test_neutralization_and_terminal_accounting(self):
        c, F = self.c, self.F
        flat = tuple(c.Bar("BTC", i*1000, i*1000+10, F(100)) for i in range(50))
        s = c.start_v3(flat, enabled=True)
        s, first = c.step_v3(s, c.Bar("BTC", 50000,50010,F(101)), (), F(1,4), enabled=True)
        self.assertNotEqual(first.after.units,F(0))
        self.assertFalse(c.observe_v3(s,enabled=True).supported)
        s, second = c.step_v3(s,c.Bar("BTC", 51000,51010,F(101)),(),F(1,4),enabled=True)
        self.assertEqual(second.after.units,F(0))
        s, terminal = c.step_v3(s,c.Bar("BTC", 52000,52010,F(101)),(),F(0),terminal=True,enabled=True)
        self.assertTrue(s.state.terminal)
        self.assertIsNone(c.step_v3(s,c.Bar("BTC", 53000,53010,F(101)),(),F(0),enabled=True))
        self.assertIsNone(c.observe_v3(s,enabled=True))

    def test_bounded_arithmetic_and_missing_history(self):
        c, F = self.c, self.F
        for bars in ((), self.bars[:25], list(self.bars), self.bars[:49] + (c.Bar("BTC", 49000,49010,F(2**8192)),)):
            self.assertIsNone(c.start_v3(bars,enabled=True))
        self.assertIsNone(c.step_v3(self.session,self.bars[50],((F(2**8191),F(2**8191)),),F(0),enabled=True))
        self.assertEqual(self.session.state.tick,0)


class ReadinessEpochTests(unittest.TestCase):
    def test_source_model_and_compiled_regression(self):
        import readiness_epoch as r
        out = r.check_readiness_epoch()
        self.assertEqual(out['source']['mutationSites'], 6)
        self.assertIn('CE-READY-001', out['model']['counterexamples'])

    def test_weakened_readiness_fails_closed(self):
        import readiness_epoch as r
        main = (ROOT / r.MAIN).read_text()
        cases = [
            ('does not bump', main.replace('unless (null targets) (bumpRuntimeEpoch ctrl)', 'pure ()', 1)),
            ('without epoch validation', main.replace('botRecoveryReady <- botRecoveryReadyNow botCtrl botRecoveryReadyRef', 'botRecoveryReady <- readinessReadyBool <$> readIORef botRecoveryReadyRef', 1)),
            ('snapshot with its epoch|epoch written outside|in-flight count', main.replace('(,,) m <$> readIORef (bcRuntimeEpoch botCtrl) <*> readIORef (bcManualInFlight botCtrl)', '(,,) m <$> pure 0 <*> pure 0', 1)),
            ('mutation sites drifted', main + '\nsneaky :: BotController -> IO ()\nsneaky ctrl = modifyMVar_ (bcRuntime ctrl) pure\n'),
        ]
        for reason, text in cases:
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                r.bind({r.MAIN: text})
        helper = (ROOT / r.READINESS).read_text()
        with self.assertRaisesRegex(ValueError, 'readiness decision changed'):
            r.bind({r.READINESS: helper.replace('epoch == epochNow && ', '', 1)})


class MinSizeCapTests(unittest.TestCase):
    def test_source_smt_wire_and_compiled_render(self):
        import min_size_cap as m
        out = m.check_min_size_cap()
        self.assertEqual(out['smt']['F-ROUND-MIN-CAP'], 'unsat')
        self.assertGreater(out['conformance']['cases'], 3000)

    def test_removed_guard_or_new_bump_fails_closed(self):
        import min_size_cap as m
        main = (ROOT / m.MAIN).read_text()
        guard = '                                            , fractionSized && wireValue q * toRational currentPrice > toRational cap ->'
        self.assertIn(guard, main)
        with self.assertRaisesRegex(ValueError, 'futures cap guard changed'):
            m.bind({m.MAIN: main.replace(guard, '                                            , False ->', 1)})
        with self.assertRaisesRegex(ValueError, 'unreviewed minimum-bump caller'):
            m.bind({m.MAIN: main + '\nextra = normalizeEntryQty sf currentPrice 1\n'})
        numeric = (ROOT / m.NUMERIC).read_text()
        with self.assertRaisesRegex(ValueError, 'wire renderer changed'):
            m.bind({m.NUMERIC: numeric.replace('else truncate (toRational (abs x) * 10 ^ scaleExp)', 'else ceiling (toRational (abs x) * 10 ^ scaleExp)', 1)})
        dex = (ROOT / 'haskell/app/Trader/Dex.hs').read_text()
        with self.assertRaisesRegex(ValueError, 'DEX token amounts bypass'):
            m.bind({'haskell/app/Trader/Dex.hs': dex.replace('let scaled = gridUnits decimals amt', 'let scaled = floor (scaledRaw + 1e-9)', 1)})


class DrainGatesTests(unittest.TestCase):
    def test_source_and_model(self):
        import drain_gates as d
        out = d.check_drain_gates()
        self.assertEqual(len(out['source']['drainedRoutes']), 9)
        self.assertTrue(all(out['model']['mutants'].values()))

    def test_removed_gate_fails_closed(self):
        import drain_gates as d
        main = (ROOT / d.MAIN).read_text()
        cases = [
            ('bot publication not drain-latched', main.replace('drainingNow <- isDraining (bcDrain ctrl)', 'drainingNow <- pure False', 1)),
            ('decides orders while draining', main.replace('placeIfEnabled drain args settings sig env sym =\n    refuseWhileDraining drain sym $\n', 'placeIfEnabled drain args settings sig env sym =\n    id $\n', 1)),
            ('candles are decided while draining', main.replace('            then pure (Right st)\n', '            then trySync (botApplyKline mOps metrics mJournal mWebhook topCombosCtx ctrl st k)\n', 1)),
            ('optimizer launch not drain-gated', main.replace('    draining <- isDraining drain\n    exeResult <-', '    let draining = False\n    exeResult <-', 1)),
        ]
        for reason, text in cases:
            self.assertNotEqual(text, main, reason)
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                d.bind({d.MAIN: text})
        shutdown = (ROOT / d.SHUTDOWN).read_text()
        with self.assertRaisesRegex(ValueError, 'work-starting route not drained'):
            d.bind({d.SHUTDOWN: shutdown.replace('        , ["optimizer", "run"]\n', '', 1)})


class OrderIdempotenceTests(unittest.TestCase):
    def test_source_and_model(self):
        import order_idempotence as o
        out = o.check_order_idempotence()
        self.assertEqual(set(out['model']['counterexamples']), {'delta-sizing', 'retry-writes'})

    def test_weakened_retry_fails_closed(self):
        import order_idempotence as o
        http = (ROOT / o.HTTP).read_text()
        with self.assertRaisesRegex(ValueError, 'may re-send writes'):
            o.bind({o.HTTP: http.replace('defaultRetryConfig{rcRetryWrites = False}', 'defaultRetryConfig', 1)})
        with self.assertRaisesRegex(ValueError, 'implicitly re-send'):
            o.bind({o.HTTP: http.replace(', managerRetryableException = const False', '', 1)})
        binance = (ROOT / o.BINANCE).read_text()
        with self.assertRaisesRegex(ValueError, 'outside the venue retry policy'):
            o.bind({o.BINANCE: binance.replace('httpLbsWithRetry venueRetryConfig Nothing', 'httpLbsWithRetry defaultRetryConfig Nothing', 1)})
        main = (ROOT / o.MAIN).read_text()
        with self.assertRaisesRegex(ValueError, 're-read the venue position'):
            o.bind({o.MAIN: main.replace('    placeFutures mSf quoteAsset dir = do\n        summary <- fetchFuturesPositionSummary env sym', '    placeFutures mSf quoteAsset dir = do\n        let summary = emptySummary', 1)})


if __name__ == '__main__':
    unittest.main()
