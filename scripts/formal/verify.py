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
    supported = {"F-RL-LIFECYCLE": "model_checked", "F-RL-CONFORMANCE": "property_tested",
                 "F-RL-INTEGRITY": "property_tested", "F-RL-REFINEMENT": "open"}
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
    obligations = ledger['missionObligations']
    require([o['number'] for o in obligations] == list(range(1, 39)), 'mandatory obligation coverage incomplete')
    for item in obligations:
        require(item['status'] in ('open', 'partially_verified') and item['limitations'], 'unsupported mission completion')
        require(set(item['evidenceRequirements']) <= {e['requirementId'] for e in entries}, 'unknown obligation evidence')
    covered = {p for e in entries for p in e['implementationFiles'] + e['testFiles'] + [e['sourceModel']]}
    require(set(ledger['criticalFiles']) <= covered, 'critical file lacks reverse traceability')
    source_files = {str(p.relative_to(root)) for pattern in ('scripts/formal/*.py', 'formal/research/*.hs') for p in root.glob(pattern)}
    require(source_files <= set(ledger['criticalFiles']), 'new proof source missing from critical-file roster')
    return sum(o['status'] in ('open', 'partially_verified') for o in obligations)


def run(record=False, require_complete=False):
    import z3
    from conformance import check_haskell
    from lifecycle import check_model
    from proofs import check_all
    started = time.monotonic()
    lock = read_json(ROOT / 'formal/research/toolchain.json')
    require(z3.get_version_string() == lock['z3'], 'Z3 version mismatch')
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
    result = {'schemaVersion': 1, 'solver': z3.get_version_string(), 'smt': check_all(),
              'model': check_model(), 'conformance': check_haskell(ROOT), 'sourceHashes': lock['sourceHashes']}
    require(set(result['smt']) == {e['requirementId'] for e in ledger['entries'] if e['status'] == 'smt_verified'}, 'SMT obligation roster mismatch')
    counterexamples = read_json(ROOT / 'formal/research/counterexamples.json')
    require(result['model']['counterexampleToRevocation'] == counterexamples['entries'][0]['trace'], 'counterexample regression drift')
    path = ROOT / 'formal/research/results.json'
    if record:
        path.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    else:
        require(result == read_json(path), 'certificate differs from reviewed receipt; investigate before recording')
    print(json.dumps({'scopedCertificates': 'pass', 'smtObligations': len(result['smt']),
                      'model': result['model'], 'conformance': result['conformance'],
                      'openMissionObligations': open_count, 'missionComplete': False,
                      'seconds': round(time.monotonic() - started, 3)}, indent=2))
    require(not require_complete or open_count == 0, 'research acceptance blocked by open obligations')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', action='store_true', help='explicitly record successfully reproduced scoped results for review')
    parser.add_argument('--require-complete', action='store_true', help='also reject open mission obligations')
    args = parser.parse_args()
    run(args.record, args.require_complete)
