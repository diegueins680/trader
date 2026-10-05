#!/usr/bin/env python3
"""Reproduce scoped certificates; fail closed on drift or unknown proof results."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
STATUSES = {'proved', 'model_checked', 'probabilistically_model_checked', 'smt_verified',
            'refinement_verified', 'exhaustively_checked', 'property_tested',
            'empirically_supported', 'assumption', 'partially_verified', 'open', 'refuted'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key: ' + key)
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('non-finite JSON: ' + value)
    def finite_float(value):
        parsed = float(value)
        require(math.isfinite(parsed), "overflowing JSON number")
        return parsed
    return json.loads(path.read_text(), object_pairs_hook=pairs, parse_constant=invalid, parse_float=finite_float)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_ledger(ledger, root):
    require(ledger['schemaVersion'] == 1, 'unsupported proof ledger')
    entries = ledger['entries']
    require(entries and len({e['requirementId'] for e in entries}) == len(entries), 'missing/duplicate requirements')
    canonical = read_json(root / 'formal/specifications.json')
    ids = {c['id'] for s in canonical['specifications'] for k in ('requires', 'ensures', 'invariants', 'failures') for c in s[k]}
    required = {'requirementId', 'claim', 'criticality', 'formalStatement', 'assumptions', 'tool', 'toolVersion',
                'sourceModel', 'artifact', 'implementationFiles', 'testFiles', 'ciCommand', 'result', 'bounds',
                'limitations', 'counterexample', 'status'}
    assumptions = {a['id'] for a in ledger['assumptions']}
    scope = next(s for s in canonical['specifications'] if s['id'] == 'A-FORMAL-RESEARCH')
    scoped_clauses = {c['id']: c['statement'] for k in ('requires', 'ensures', 'invariants', 'failures') for c in scope[k]}
    mapped = {e['requirementId'] for e in entries}
    for entry in entries:
        related = entry.get('relatedRequirements', [])
        require(isinstance(related, list) and set(related) <= scoped_clauses.keys(), 'unknown related requirement')
        mapped.update(related)
        require(scoped_clauses.get(entry['requirementId']) == entry['formalStatement'], 'canonical formal statement drift')
    require(mapped == scoped_clauses.keys(), 'unmapped scoped formal requirement')
    supported = {"F-RL-DATA-COMPOSITION": "exhaustively_checked", "F-DRAIN-POOL-LIFECYCLE": "model_checked", "F-DRAIN-POOL-CONFORMANCE": "property_tested", "F-BACKTEST-GATE-LIFECYCLE": "model_checked", "F-BACKTEST-GATE-CONFORMANCE": "property_tested", "F-ASYNC-SEAL-LIFECYCLE": "model_checked", "F-ASYNC-SEAL-CONFORMANCE": "property_tested", "F-ASYNC-ADMISSION-LIFECYCLE": "model_checked", "F-ASYNC-ADMISSION-CONFORMANCE": "property_tested", "F-WORKER-REGISTRY-LIFECYCLE": "model_checked", "F-WORKER-REGISTRY-CONFORMANCE": "property_tested", "F-SHUTDOWN-STAGES": "model_checked", "F-SHUTDOWN-CONFORMANCE": "property_tested", "F-RL-PROCESS-LIFECYCLE": "model_checked", "F-RL-PROCESS-CONFORMANCE": "property_tested", "F-RL-PROCESS-ISOLATION": "exhaustively_checked", "F-RL-LIFECYCLE": "model_checked", "F-RL-SNAPSHOT-PUBLISH": "model_checked", "F-RL-SNAPSHOT-CONFORMANCE": "property_tested", "F-RL-SNAPSHOT-ISOLATION": "exhaustively_checked", "F-RL-COMPONENT-ISOLATION": "exhaustively_checked", "F-RL-POLICY-EFFECTS": "exhaustively_checked", "F-RL-CONFORMANCE": "property_tested",
                 "F-RL-INTEGRITY": "property_tested", "F-RL-CLOSURE": "property_tested", "F-RL-DEFAULT-PATH": "exhaustively_checked", "F-RL-REFINEMENT": "open",
                 "F-RL-ARTIFACT-PATH": "model_checked", "F-RL-TARGET-V2-PUBLISH": "model_checked",
                 "F-RL-ESS-V2-PUBLISH": "model_checked", "F-RL-FUNDING-FINITE": "refuted", "F-RL-REPLAY-ORDER": "model_checked", "F-RL-REPLAY-QUOTIENT": "model_checked", "F-RL-REWARD-ADDITIVE": "refuted", "F-RL-OPE-FP-SUPPORT": "refuted", "F-RL-INFER-PATH": "model_checked", "F-RL-INFER-DEADLINE": "refuted",
                 "F-RL-OPTIMIZER-PUBLISH": "model_checked", "F-RL-OPTIMIZER-ATOMIC": "refuted",
                 "F-RL-Q-FINITE": "refuted", "F-RL-CQL-SHIFT": "refuted",
                 "F-RL-PPO-FINITE": "refuted", "F-RL-PPO-UNIFORM-CLIP": "refuted",
                 "F-RL-TERMINAL-EXACT": "refuted", "F-RL-TERMINAL-FINITE": "refuted",
                 "F-RL-UNCONDITIONAL-FLOOR": "refuted", "F-RL-GAP-CONFORMANCE": "exhaustively_checked"}
    require(set(supported) <= {e["requirementId"] for e in entries}, "missing required certificate")
    for entry in entries:
        require(required <= entry.keys(), 'incomplete ledger entry')
        require(entry['requirementId'] in ids, 'unknown canonical requirement')
        require(entry['status'] in STATUSES, 'unknown proof status')
        require(entry['status'] == supported.get(entry['requirementId'], 'smt_verified'), 'unsupported proof promotion')
        require(entry['criticality'] in ('critical', 'high', 'medium'), 'invalid criticality')
        require(entry['assumptions'] and set(entry['assumptions']) <= assumptions, 'unknown/missing assumptions')
        require(entry['implementationFiles'] and entry['testFiles'], 'missing implementation/test traceability')
        for key in ('claim', 'formalStatement', 'tool', 'toolVersion', 'ciCommand', 'result', 'bounds', 'limitations'):
            require(isinstance(entry[key], str) and bool(entry[key].strip()), 'empty field: ' + key)
        for relative in [entry['sourceModel'], entry['artifact'], *entry['implementationFiles'], *entry['testFiles']]:
            path = root / relative
            require(not Path(relative).is_absolute() and '..' not in Path(relative).parts and path.is_file(), 'missing/unsafe traceability path: ' + relative)
    open_count = validate_obligations(ledger['missionObligations'], entries)
    for obligation in ledger['missionObligations']:
        for relative in obligation['implementationFiles']:
            require(not Path(relative).is_absolute() and '..' not in Path(relative).parts and (root / relative).is_file(), 'missing/unsafe obligation implementation')
    covered = {p for e in entries for p in e['implementationFiles'] + e['testFiles'] + [e['sourceModel']]}
    require(set(ledger['criticalFiles']) <= covered, 'critical file lacks reverse traceability')
    source_files = {str(p.relative_to(root)) for pattern in ('scripts/formal/*.py', 'formal/research/*.hs') for p in root.glob(pattern)}
    require(source_files <= set(ledger['criticalFiles']), 'new proof source missing from critical-file roster')
    return open_count


VERIFIED = {'proved', 'model_checked', 'probabilistically_model_checked',
            'smt_verified', 'refinement_verified', 'exhaustively_checked'}


def validate_obligations(obligations, entries, reproduced=None, contracts=None):
    """Validate reviewed sufficiency mappings; never infer closure from lemma count."""
    require([o['number'] for o in obligations] == list(range(1, 39)), 'mandatory obligation coverage incomplete')
    if contracts is None:
        contracts = read_json(ROOT / 'formal/research/obligation-contracts.json')['obligations']
    require([c['number'] for c in contracts] == list(range(1, 39)), 'incomplete canonical closure contracts')
    by_id = {e['requirementId']: e for e in entries}
    remaining = 0
    for item, contract in zip(obligations, contracts):
        require('closureClass' in contract and all(item.get(key) == value for key, value in contract.items() if key != 'closureClass'), 'canonical closure contract drift')
        for key in ('scope', 'closureCriteria', 'nextAction', 'limitations'):
            require(isinstance(item.get(key), str) and bool(item[key].strip()), 'missing obligation closure criterion')
        require(item['implementationFiles'] and all(isinstance(p, str) and p for p in item['implementationFiles']), 'missing obligation implementation')
        require(isinstance(item['blockers'], list) and all(isinstance(b, str) and b.strip() for b in item['blockers']), 'invalid blockers')
        require(set(item['evidenceRequirements']) <= by_id.keys(), 'unknown obligation evidence')
        required = item['requiredCertificates']
        require(isinstance(required, list) and len(required) == len(set(required)) and set(required) <= set(item['evidenceRequirements']), 'invalid closure certificate mapping')
        sufficient = bool(required) and not item['blockers'] and all(by_id[c]['status'] in VERIFIED for c in required)
        closed = item['status'] in VERIFIED
        require(closed == sufficient, 'unsupported or unrecorded obligation closure')
        require(closed or item['status'] in ('open', 'partially_verified', 'refuted'), 'unsupported mission status')
        if closed:
            require(item['status'] == contract['closureClass'] and item['status'] in {by_id[c]['status'] for c in required}, 'unsupported aggregate verification class')
            if reproduced is not None:
                require(all(reproduced.get(c) == by_id[c]['status'] for c in required), 'closure certificate not reproduced')
        else:
            require(bool(item['blockers']), 'unresolved obligation lacks concrete blocker')
            remaining += 1
    return remaining


def acceptance_summary(obligations, entries, reproduced, gates, root, contracts=None):
    remaining = validate_obligations(obligations, entries, reproduced, contracts)
    require([g['id'] for g in gates] == ['economic_evidence', 'reproducible_delivery'], 'missing research acceptance gate')
    blocked = []
    for gate in gates:
        require(gate['criteria'] and gate['status'] in ('open', 'satisfied'), 'invalid research gate')
        if gate['status'] == 'open':
            require(gate['blockers'], 'open research gate lacks reason')
            blocked.append(gate['id'])
        else:
            require(not gate['blockers'] and gate['evidence'], 'unsupported research acceptance')
            for item in gate['evidence']:
                relative = Path(item['path'])
                require(not relative.is_absolute() and '..' not in relative.parts and (root / relative).is_file(), 'missing acceptance evidence')
                require(digest(root / relative) == item['sha256'], 'acceptance evidence drift')
    return {'openMissionObligations': remaining, 'closedMissionObligations': 38 - remaining,
            'formalObligationsComplete': remaining == 0,
            'blockedResearchGates': blocked,
            'missionComplete': remaining == 0 and not blocked}


def run(record=False, require_complete=False):
    import numpy
    import z3
    from conformance import check_haskell
    from lifecycle import check_model
    from proofs import check_all
    from gap_risk import check_counterexamples
    from gap_conformance import check_replay
    from causal_footprint import check_source
    from training_prefix import check_training, REGISTRATION
    from artifact_admission import check_artifact
    from transition_admission import check_transition
    from terminal_numerics import check_terminal
    from target_v2 import check_targets
    from ppo_objective import check_ppo
    from value_objective import check_values
    from optimizer_publication import check_optimizer
    from inference_boundary import check_inference
    from ope_algebra import check_ope, REGISTRATION as OPE_REGISTRATION
    from ess_v2 import check_ess, REGISTRATION as ESS_REGISTRATION
    from funding_boundary import check_funding
    from replay_order import check_replay_order
    from replay_cutoff import check_replay_cutoff
    from reward_accounting import check_reward_accounting
    from data_composition import check_composition
    from default_paths import check_defaults
    from capability_isolation import check_isolation
    from snapshot_v2 import check_snapshot
    from inference_process import check_process
    from shutdown_deadline import check_shutdown
    from worker_registry import check_worker_registry
    from async_job_admission import check_async_admission
    from async_shutdown_seal import check_async_seal
    from drain_pool import check_drain_pool
    from backtest_gate import check_backtest_gate
    started = time.monotonic()
    lock = read_json(ROOT / 'formal/research/toolchain.json')
    require(z3.get_version_string() == lock['z3'], 'Z3 version mismatch')
    require(numpy.__version__ == lock['numpy'], 'NumPy version mismatch')
    require('.'.join(map(str, sys.version_info[:3])) == lock['python'], 'Python version mismatch')
    ghc = subprocess.check_output(['ghc', '--numeric-version'], text=True).strip()
    require(ghc == lock['ghc'], 'GHC version mismatch')
    ledger = read_json(ROOT / 'formal/research/proof-ledger.json')
    open_count = validate_ledger(ledger, ROOT)
    for path, expected in lock['sourceHashes'].items():
        require(digest(ROOT / path) == expected, 'unreviewed model/implementation drift: ' + path)
    # Scan executable proof sources, not prose explaining forbidden placeholders.
    for path in lock['proofSources']:
        text = (ROOT / path).read_text()
        require(not re.search(r'\b(sorry|admit|Admitted|axiom)\b', text), 'proof placeholder: ' + path)
        require(not re.search(r'@(?:unittest\.)?skip|assert\s+(?:True|False)', text), 'disabled proof/test: ' + path)
    result = {'schemaVersion': 1, 'solver': z3.get_version_string(), 'numpy': numpy.__version__, 'smt': check_all(),
              'model': check_model(), 'conformance': check_haskell(ROOT), 'sourceHashes': lock['sourceHashes']}
    gaps = read_json(ROOT / 'formal/research/gap-counterexamples.json')
    result['gapRefutations'] = check_counterexamples(gaps)
    result['replayConformance'] = check_replay(gaps)
    result['sourceFootprint'] = check_source()
    result['smt'][result['sourceFootprint']['requirement']] = 'unsat'
    result['trainingPrefix'] = check_training(read_json(ROOT / REGISTRATION))
    result['smt'].update(result['trainingPrefix']['smt'])
    result['artifactAdmission'] = check_artifact()
    result['smt'][result['artifactAdmission']['metadata']['requirement']] = 'unsat'
    result['transitionAdmission'] = check_transition()
    result['smt'][result['transitionAdmission']['requirement']] = 'unsat'
    result['terminalNumerics'] = check_terminal(read_json(ROOT / 'formal/research/terminal-counterexamples.json'))
    result['smt'].update(result['terminalNumerics']['smt'])
    result['targetV2'] = check_targets()
    result['smt'].update(result['targetV2']['smt'])
    result['ppoObjective'] = check_ppo(read_json(ROOT / 'formal/research/ppo-counterexamples.json'))
    result['smt'].update(result['ppoObjective']['smt'])
    result['valueObjective'] = check_values(read_json(ROOT / 'formal/research/value-counterexamples.json'))
    result['smt'].update(result['valueObjective']['smt'])
    result['optimizerPublication'] = check_optimizer(read_json(ROOT / 'formal/research/optimizer-counterexamples.json'))
    result['smt'].update(result['optimizerPublication']['smt'])
    result['inferenceBoundary'] = check_inference(read_json(ROOT / 'formal/research/inference-counterexamples.json'))
    result['smt'].update(result['inferenceBoundary']['smt'])
    result['opeAlgebra'] = check_ope(read_json(ROOT / 'formal/research/ope-counterexamples.json'), read_json(ROOT / OPE_REGISTRATION))
    result['smt'].update(result['opeAlgebra']['smt'])
    result['exactESSV2'] = check_ess(read_json(ROOT / ESS_REGISTRATION))
    result['smt'].update(result['exactESSV2']['smt'])
    result['fundingBoundary'] = check_funding()
    result['smt'].update(result['fundingBoundary']['smt'])
    result['replayOrder'] = check_replay_order()
    result['smt'].update(result['replayOrder']['smt'])
    result['replayCutoff'] = check_replay_cutoff()
    result['smt'].update(result['replayCutoff']['smt'])
    result['dataComposition'] = check_composition()
    result['smt'].update(result['dataComposition']['smt'])
    result['defaultPaths'] = check_defaults()
    result['drainPool'] = check_drain_pool()
    result['smt'].update(result['drainPool']['smt'])
    result['backtestGate'] = check_backtest_gate()
    result['smt'].update(result['backtestGate']['smt'])
    result['asyncShutdownSeal'] = check_async_seal()
    result['smt'].update(result['asyncShutdownSeal']['smt'])
    result['asyncJobAdmission'] = check_async_admission()
    result['smt'].update(result['asyncJobAdmission']['smt'])
    result['workerRegistry'] = check_worker_registry()
    result['smt'].update(result['workerRegistry']['smt'])
    result['shutdownDeadline'] = check_shutdown()
    result['smt'].update(result['shutdownDeadline']['smt'])
    result['inferenceProcess'] = check_process()
    result['smt'].update(result['inferenceProcess']['smt'])
    result['snapshotV2'] = check_snapshot()
    result['smt'].update(result['snapshotV2']['smt'])
    result['capabilityIsolation'] = check_isolation()
    result['smt'].update(result['capabilityIsolation']['smt'])
    result['rewardAccounting'] = check_reward_accounting()
    result['smt'].update(result['rewardAccounting']['smt'])
    require(set(result['smt']) == {e['requirementId'] for e in ledger['entries'] if e['status'] == 'smt_verified'}, 'SMT obligation roster mismatch')
    counterexamples = read_json(ROOT / 'formal/research/counterexamples.json')
    require(result['model']['counterexampleToRevocation'] == counterexamples['entries'][0]['trace'], 'counterexample regression drift')
    reproduced = {key: 'smt_verified' for key, value in result['smt'].items() if value == 'unsat'}
    for requirement, path in {
        'F-RL-LIFECYCLE': ('model',),
        'F-DRAIN-POOL-LIFECYCLE': ('drainPool', 'model'),
        'F-BACKTEST-GATE-LIFECYCLE': ('backtestGate', 'model'),
        'F-ASYNC-SEAL-LIFECYCLE': ('asyncShutdownSeal', 'model'),
        'F-ASYNC-ADMISSION-LIFECYCLE': ('asyncJobAdmission', 'model'),
        'F-WORKER-REGISTRY-LIFECYCLE': ('workerRegistry', 'model'),
        'F-SHUTDOWN-STAGES': ('shutdownDeadline', 'model'),
        'F-RL-PROCESS-LIFECYCLE': ('inferenceProcess', 'model'),
        'F-RL-SNAPSHOT-PUBLISH': ('snapshotV2', 'model'),
        'F-RL-ARTIFACT-PATH': ('artifactAdmission', 'model'),
        'F-RL-TARGET-V2-PUBLISH': ('targetV2', 'model'),
        'F-RL-ESS-V2-PUBLISH': ('exactESSV2', 'model'),
        'F-RL-REPLAY-ORDER': ('replayOrder', 'model'),
        'F-RL-REPLAY-QUOTIENT': ('replayCutoff', 'model'),
        'F-RL-OPTIMIZER-PUBLISH': ('optimizerPublication', 'singleWriter'),
        'F-RL-INFER-PATH': ('inferenceBoundary', 'model'),
    }.items():
        receipt = result
        for key in path:
            receipt = receipt[key]
        require(receipt['states'] > 0 and receipt['transitions'] > 0, 'missing model reproduction')
        reproduced[requirement] = 'model_checked'
    require(result['replayConformance']['boundedAccountingTraces'] == 180, 'incomplete gap conformance')
    reproduced['F-RL-GAP-CONFORMANCE'] = 'exhaustively_checked'
    reproduced['F-RL-DATA-COMPOSITION'] = result['dataComposition']['status']
    reproduced['F-RL-DEFAULT-PATH'] = result['defaultPaths']['status']
    for part in ('graph', 'effects'):
        reproduced[result['capabilityIsolation'][part]['requirement']] = 'exhaustively_checked'
    reproduced['F-RL-PROCESS-ISOLATION'] = result['inferenceProcess']['isolation']['status']
    reproduced['F-RL-SNAPSHOT-ISOLATION'] = result['snapshotV2']['isolation']['status']
    acceptance = acceptance_summary(ledger['missionObligations'], ledger['entries'], reproduced, ledger['researchAcceptanceGates'], ROOT)
    result['obligationClosure'] = acceptance
    path = ROOT / 'formal/research/results.json'
    if record:
        path.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    else:
        require(result == read_json(path), 'certificate differs from reviewed receipt; investigate before recording')
    print(json.dumps({'scopedCertificates': 'pass', 'smtObligations': len(result['smt']),
                      'model': result['model'], 'conformance': result['conformance'],
                      'gapRefutations': result['gapRefutations'],
                      'replayConformance': result['replayConformance'],
                      'dataComposition': result['dataComposition'],
                      'sourceFootprint': result['sourceFootprint'],
                      'trainingPrefix': result['trainingPrefix'],
                      'artifactAdmission': result['artifactAdmission'],
                      'transitionAdmission': result['transitionAdmission'],
                      'terminalNumerics': result['terminalNumerics'],
                      'targetV2': result['targetV2'],
                      'ppoObjective': result['ppoObjective'],
                      'valueObjective': result['valueObjective'],
                      'optimizerPublication': result['optimizerPublication'],
                      'inferenceBoundary': result['inferenceBoundary'],
                      'inferenceProcess': result['inferenceProcess'],
                      'shutdownDeadline': result['shutdownDeadline'],
                      'workerRegistry': result['workerRegistry'],
                      'drainPool': result['drainPool'],
                      'backtestGate': result['backtestGate'],
                      'asyncShutdownSeal': result['asyncShutdownSeal'],
                      'asyncJobAdmission': result['asyncJobAdmission'],
                      'opeAlgebra': result['opeAlgebra'],
                      'exactESSV2': result['exactESSV2'],
                      'fundingBoundary': result['fundingBoundary'],
                      'replayOrder': result['replayOrder'],
                      'replayCutoff': result['replayCutoff'],
                      'rewardAccounting': result['rewardAccounting'],
                      **acceptance,
                      'seconds': round(time.monotonic() - started, 3)}, indent=2))
    require(not require_complete or acceptance['missionComplete'], 'research acceptance blocked by open obligations or research evidence gates')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', action='store_true', help='explicitly record successfully reproduced scoped results for review')
    parser.add_argument('--require-complete', action='store_true', help='also reject open mission obligations')
    args = parser.parse_args()
    run(args.record, args.require_complete)
