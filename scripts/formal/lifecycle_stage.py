"""Source-bound eligibility preservation; no promotion service or runtime changes."""
import ast
from collections import deque
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import z3 as z
from artifact_admission import expression, extract as loader_extract
from data_composition import definition
from ppo_successor import certify
from promotion_boundary import extract, metadata

ROOT = Path(__file__).resolve().parents[2]
STAGES = ('research', 'backtest-eligible', 'replay-eligible', 'shadow-eligible',
          'paper-eligible', 'micro-live-review-eligible', 'live-authorized')
TAGS = {'v1': 'rejected_research_only', 'v4': 'research-only'}


def require(ok, reason):
    if not ok:
        raise ValueError('lifecycle stage: ' + reason)


def predicates(trees):
    return {
        'v1': (loader_extract(ast.unparse(trees['sequential_learning']))[0][1], 'a'),
        'v4': (definition(trees['ppo_artifact_v4'], 'decode_artifact_v4').body[2].body[5].test, 'value'),
    }


def rejection(node, container, label):
    """Abstract all unrelated rejection atoms nondeterministically, not as false.

    Ordinary JSON non-string values cannot equal either policy string. The exact
    False identity guard likewise abstracts all other JSON values as not false.
    """
    text = z.String(label + '_text')
    is_text, exact_false = z.Bools(label + '_is_text ' + label + '_exact_false')
    atoms = {}
    for i, comparison in enumerate(n for n in ast.walk(node) if isinstance(n, ast.Compare)):
        key = ast.unparse(comparison)
        left = ast.unparse(comparison.left)
        if left == container + "['promotion']":
            require(len(comparison.ops) == 1 and isinstance(comparison.ops[0], ast.NotEq)
                    and isinstance(comparison.comparators[0], ast.Constant)
                    and type(comparison.comparators[0].value) is str, 'unreviewed promotion predicate')
            atoms[key] = z.Not(z.And(is_text, text == comparison.comparators[0].value))
        elif left == container + "['enabled']":
            require(len(comparison.ops) == 1 and isinstance(comparison.ops[0], ast.IsNot)
                    and isinstance(comparison.comparators[0], ast.Constant)
                    and comparison.comparators[0].value is False, 'unreviewed enable predicate')
            atoms[key] = z.Not(exact_false)
        else:
            atoms[key] = z.Bool(label + '_unrelated_reject_' + str(i))
    return expression(node, atoms), text, is_text, exact_false


def projection(text, is_text):
    # Unrecognized labels are not Research; they must be rejected, never defaulted.
    result = z.IntVal(-1)
    for i, stage in reversed(list(enumerate(STAGES))):
        result = z.If(z.And(is_text, text == stage), i, result)
    return z.If(z.And(is_text, z.Or(text == TAGS['v1'], text == TAGS['v4'])), 0, result)


def lawful(before, after, evidence, review):
    return z.Or(after == before,
                z.And(before >= 0, after == before + 1, after < len(STAGES) - 1,
                      review, *[z.Implies(after >= i + 1, g) for i, g in enumerate(evidence)]))


def prove_steps(facts, guards):
    require(set(facts) == set(guards) == set(TAGS), 'policy format coverage')
    queries = 0
    for version, (node, container) in guards.items():
        require(facts[version]['promotion'] == TAGS[version] and facts[version]['enabled'] is False,
                'writer leaves research stage')
        reject, text, is_text, exact_false = rejection(node, container, version)
        admitted = z.Not(reject)
        selected = projection(text, is_text)
        # These predicates come from actual decoder ASTs, not a model-only guard.
        certify(admitted, z.And(is_text, text == facts[version]['promotion'], exact_false)); queries += 1
        certify(admitted, selected == 0); queries += 1
        before = z.Int(version + '_before')
        evidence = z.Bools(' '.join(version + '_evidence_' + str(i) for i in range(6)))
        review = z.Bool(version + '_external_review')
        certify(z.And(admitted, before == 0),
                z.And(lawful(before, selected, evidence, review), selected == before, selected < 6)); queries += 1
        certify(z.Or(z.Not(is_text), text != facts[version]['promotion'], z.Not(exact_false)), reject); queries += 1
    return {'queries': queries, 'premiseChecks': queries,
            'smt': {'F-RL-STAGE-STEP': 'unsat'}}


def stage_index(value):
    if type(value) is not str:
        return -1
    if value in TAGS.values():
        return 0
    return next((i for i, stage in enumerate(STAGES) if value == stage), -1)


def successors(state, facts, mutant=False):
    version, phase, stored, retained, stage = state
    def go(label, next_phase, new_stored=stored, new_retained=retained, next_stage=stage):
        return label, (version, next_phase, new_stored, new_retained, next_stage)
    yield go('failure', 'failed')
    yield go('disable', 'disabled')
    if phase == 'fresh':
        yield go('save-research', 'saved', True, next_stage=stage_index(facts[version]['promotion']))
    elif phase == 'saved':
        # Nonstrings and unknown strings are explicit equivalence classes. Other
        # codec gates are relaxed: accepting extra research inputs is conservative
        # for stage safety, not an implementation acceptance equivalence claim.
        for value in (*STAGES, *TAGS.values(), 'unknown', None):
            for exact_false in (True, False):
                admitted = type(value) is str and value == facts[version]['promotion'] and exact_false
                yield go('load:' + repr(value) + ':' + str(exact_false), 'loaded' if admitted else 'failed',
                         next_stage=stage_index(value) if admitted else stage)
    elif phase == 'loaded':
        yield go('infer-research', 'proposed', new_retained=True)
    elif phase == 'proposed':
        yield go('successor-research', 'fresh')
        if mutant:
            yield go('unreviewed-skip-to-shadow', 'proposed', next_stage=3)
    elif phase in ('failed', 'disabled'):
        yield go('retry-research', 'fresh')


def check_model(facts, mutant=False):
    initial = [(v, 'fresh', False, False, 0) for v in sorted(TAGS)]
    queue = deque(initial); paths = {s: [] for s in initial}
    transitions = rejected = proposals = 0
    while queue:
        state = queue.popleft()
        if state[-1] != 0:
            raise ValueError('lifecycle stage: promotion counterexample: ' + json.dumps(paths[state]))
        proposals += state[1] == 'proposed'
        for label, nxt in successors(state, facts, mutant):
            transitions += 1
            rejected += label.startswith('load:') and nxt[1] == 'failed'
            if nxt not in paths:
                paths[nxt] = paths[state] + [label]; queue.append(nxt)
    require(proposals > 0 and rejected > 0, 'missing ordinary inference or rejection paths')
    return {'status': 'model_checked', 'states': len(paths), 'transitions': transitions,
            'maxShortestDepth': max(map(len, paths.values())), 'formats': 2,
            'eligibilityStages': len(STAGES), 'proposedStates': proposals,
            'rejectedMetadataEdges': rejected, 'retainedProposalCases': sum(s[3] for s in paths),
            'search': 'reachable fixed point, including arbitrary retry/successor cycles',
            'scope': 'source-derived stage safety; no promotion liveness, service, concurrency or storage theorem'}


def conformance():
    sys.path.insert(0, str(ROOT / 'scripts/research'))
    import numpy as np
    import sequential_learning as learning
    import ppo_artifact_v4 as artifact
    from optimizer_snapshot_v2 import Snapshot
    from ppo_successor_v2 import TrainingResult
    prov = {'codeCommit': 'a'*40, 'registrationSha256': 'b'*64, 'dataSha256': 'c'*64,
            'seed': 11, 'horizon': 1, 'algorithm': 'ppo', 'fold': 0,
            'promotion': 'live-authorized', 'enabled': True}
    provenance = {k: ('a'*40 if k == 'codeCommit' else 'b'*64) for k in
                  ('codeCommit', 'registrationSha256', 'dataSha256', 'fundingSha256', 'splitSha256', 'proofSha256')}
    provenance['datasetRole'] = 'development'
    def snapshot(outputs):
        buffers = tuple(np.zeros(n, dtype='<f8').tobytes() for n in (192, 16, 16*outputs, outputs))
        return Snapshot('optimizer-snapshot-v2', outputs, 4, buffers, buffers, buffers)
    fixture = TrainingResult('ppo-successor-v2', 11, 1, 1, ('FIXTURE',),
                            tuple(np.zeros(6, dtype='<f8').tobytes() for _ in range(4)),
                            snapshot(3), snapshot(1), (0.0,)*4)
    raw4 = artifact.encode_artifact_v4(fixture, provenance, enabled=True)
    require(raw4 is not None, 'ordinary v4 research rejected')
    values = (*STAGES, *TAGS.values(), 'unknown', None, False, 0, [], {}, 'Research', 'OfflineReplayV1')
    enabled_values = (False, True, 0, None, 'False', {})
    cases = accepted = 0
    with tempfile.TemporaryDirectory(prefix='trader-stage-') as directory:
        root = Path(directory); original_path = root / 'original.json'
        learning.save_policy(original_path, learning.Network(11), prov)
        original_bytes = original_path.read_bytes()
        for version, raw in (('v1', original_bytes), ('v4', raw4)):
            original = json.loads(raw)
            require(original['promotion'] == TAGS[version] and original['enabled'] is False, 'writer stage')
            for value in values:
                for enabled in enabled_values:
                    item = dict(original, promotion=value, enabled=enabled)
                    encoded = ((json.dumps(item, sort_keys=True, allow_nan=False) + '\n').encode()
                               if version == 'v1' else artifact._json(item))
                    digest = hashlib.sha256(encoded).hexdigest()  # Correct hash must not bypass stage refusal.
                    if version == 'v1':
                        path = root / (str(cases) + '.json'); path.write_bytes(encoded)
                        try:
                            learning.load_policy(path, digest, prov); ok = True
                        except ValueError:
                            ok = False
                    else:
                        ok = artifact.decode_artifact_v4(encoded, digest, provenance, enabled=True) is not None
                    expected = type(value) is str and value == TAGS[version] and enabled is False
                    require(ok == expected, 'actual promoted metadata accepted or research rejected')
                    cases += 1; accepted += ok
        require(original_path.read_bytes() == original_bytes, 'research source artifact modified')
        require(json.loads(original_bytes)['provenance']['promotion'] == 'live-authorized', 'nested test missing')
    return {'status': 'property_tested', 'metadataCases': cases, 'acceptedResearchCases': accepted,
            'rejectedCases': cases-accepted, 'correctHashOnEveryCase': True,
            'nestedPromotionDoesNotOverride': True, 'trainingRuns': 0, 'marketDataReads': 0}


def check_stages():
    registration = json.loads((ROOT / 'research-notes/registrations/lifecycle-stage-engineering.json').read_text())
    require(tuple(registration['eligibilityStages']) == STAGES and registration['policyFormats'] == ['v1', 'v4'], 'registration drift')
    require(registration['financialTrials'] == registration['historicalDataReads'] == 0
            and registration['holdoutOpened'] is False and registration['runtimeChanges'] is False, 'research boundary')
    trees, surface = extract()
    facts, _ = metadata(trees)
    proof = prove_steps(facts, predicates(trees))
    return {'surface': {'status': 'exhaustively_checked', 'researchModules': surface['moduleCount'],
                        'reviewedCalls': surface['callSites'], 'writerTags': TAGS,
                        'newPromotionServices': 0, 'newProductionConsumers': 0,
                        'scope': 'reviewed stage projection and complete existing effect inventory; constituent certificates mandatory'},
            **proof, 'model': check_model(facts), 'conformance': conformance()}
