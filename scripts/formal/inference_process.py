"""Conditional process lifecycle model, source-bound guard SMT and compiled tests."""
from collections import deque
import hashlib
import itertools
import os
import json
from pathlib import Path
import random
import re
import subprocess
import tempfile

import z3 as z

ROOT = Path(__file__).resolve().parents[2]
SOURCE = 'haskell/research/InferenceProcessV2.hs'
REGISTRY = 'formal/research/inference-process-source.json'


def require(ok, reason):
    if not ok:
        raise ValueError('inference process: ' + reason)


def extract(source=None):
    source = (ROOT / SOURCE).read_text() if source is None else source
    registry = json.loads((ROOT / REGISTRY).read_text())
    require(hashlib.sha256(source.encode()).hexdigest() == registry['sha256'], 'reviewed source drift')
    require(re.findall(r'^import (.+)$', source, re.M) == registry['imports'], 'unreviewed import')
    require(source.count('createProcess (proc executable ["--offline-inference-v2", "--worker"])') == 1,
            'launch identity drift')
    require('env = Just [], close_fds = True, create_group = True' in source, 'child isolation drift')
    require('executable <- getExecutablePath' in source and 'module Main (main) where' in source, 'self boundary drift')
    require(source.rstrip().endswith('_ -> putStrLn "(Absent,True)"'), 'default dispatch drift')
    require(source.count('readMaybe') == 4, 'unreviewed decoder')
    expected = '''admission elapsed clean frame
    | not clean || elapsed < 0 || elapsed >= budgetNS = Absent
    | frame == "IPV2 -1" = QuarterTarget (-1)
    | frame == "IPV2 0" = QuarterTarget 0
    | frame == "IPV2 1" = QuarterTarget 1
    | otherwise = Absent'''
    require(expected in source, 'guard semantics drift')
    cap = re.findall(r'^budgetNS = ([0-9]+)$', source, re.M)
    require(cap == ['20000000'], 'admission deadline changed')
    require(set(registry['helperHashes']) == {'haskell/research/SnapshotRequestV3.hs'}, 'missing decoder coverage')
    for path, expected in registry['helperHashes'].items():
        require(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, 'decoder helper drift')
    # Exact reviewed IO bodies are separately locked, not interpreted as an IO theorem.
    for name, body in registry['controlBodies'].items():
        require(body in source and source.count(name + ' ::') == 1, 'control skeleton drift: ' + name)
    require('haskell/research' not in (ROOT / 'haskell/trader.cabal').read_text(), 'research source added to production components')
    require(not any('InferenceProcessV2' in p.read_text() for p in (ROOT / 'haskell/app').rglob('*.hs')),
            'production reference introduced')
    return int(cap[0]), registry


def admission(elapsed, clean, code, cap=20000000):
    return code if clean and 0 <= elapsed < cap and code in (-1, 0, 1) else None


def prove_guard(cap):
    elapsed, code = z.Ints('process_elapsed process_code')
    clean = z.Bool('process_clean')
    accepted = z.And(clean, elapsed >= 0, elapsed < cap, z.Or(code == -1, code == 0, code == 1))
    solver = z.Solver(); solver.set(timeout=10000, random_seed=0)
    solver.add(accepted)
    require(solver.check() == z.sat, 'vacuous admission premise')
    solver.add(z.Or(z.Not(clean), elapsed < 0, elapsed >= cap, code < -1, code > 1))
    require(solver.check() == z.unsat, 'unsafe admission or unknown solver result')
    return {'F-RL-PROCESS-ADMISSION': 'unsat'}


# State = phase, request-time bucket (0,1,expired=2), candidate code (9=absent),
# remaining poll steps. Each poll step represents a bounded OS wait, not a CPU step.
# Time is saturated at expiry. A deadline event may race with a valid response.
# Initialization has its own bounded wait; no request exists before Ready.
def successors(state):
    phase, elapsed, code, polls = state
    if phase == 'disabled' or phase in ('done', 'failed'):
        return []
    if phase == 'launch':
        return [('ready', 0, 9, 0), ('term', 0, 9, 2)]
    if phase == 'ready':
        return [('pending', 0, 9, 0), ('term', 0, 9, 2)]
    if phase == 'pending':
        replies = [('term', elapsed, c, 2) for c in (-1, 0, 1, 9)]
        if elapsed < 2:
            replies.append(('pending', elapsed + 1, 9, 0))
        else:
            replies.append(('term', 2, 9, 2))
        return replies
    if phase in ('term', 'kill'):
        result = [('done', later, code, 0) for later in range(elapsed, 3)] + [('failed', elapsed, 9, 0)]
        if polls:
            result.append((phase, min(2, elapsed + 1), code, polls - 1))
        elif phase == 'term':
            result.append(('kill', elapsed, code, 2))
        else:
            result.append(('failed', elapsed, 9, 0))
        return result
    raise ValueError('unknown lifecycle phase')


def rank(state):
    phase, elapsed, _, polls = state
    if phase in ('disabled', 'done', 'failed'):
        return 0
    if phase == 'kill':
        return 1 + polls
    if phase == 'term':
        return 4 + polls
    return {'pending': 9 - elapsed, 'ready': 10, 'launch': 11}[phase]


def check_model(next_states=successors):
    initial = [('disabled', 0, 9, 0), ('launch', 0, 9, 0)]
    seen = set(initial); queue = deque((s, 0) for s in initial)
    edges = depth = accepted = failed = late = 0
    while queue:
        state, distance = queue.popleft(); depth = max(depth, distance)
        phase, elapsed, code, _ = state
        result = admission(elapsed, phase == 'done', code, cap=2)
        if result is not None:
            require(phase == 'done' and elapsed < 2 and code in (-1, 0, 1), 'unsafe accepted state')
            accepted += 1
        failed += phase == 'failed'
        late += phase == 'done' and elapsed == 2 and code != 9
        following = next_states(state)
        require(bool(following) == (phase not in ('disabled', 'done', 'failed')), 'unexpected deadlock')
        for target in following:
            require(rank(target) < rank(state), 'nonterminating bounded control path')
            require(target[0] != 'launch', 'successor worker/retry')
            edges += 1
            if target not in seen:
                seen.add(target); queue.append((target, distance + 1))
    require(accepted and failed and late, 'missing race/failure coverage')
    return {'states': len(seen), 'transitions': edges, 'maxShortestDepth': depth,
            'initialRank': 11, 'acceptedStates': accepted, 'cleanupFailureStates': failed,
            'expiredReplyStatesRejected': late, 'children': 1, 'requests': 1, 'retries': 0,
            'timeBuckets': [0, 1, 2], 'pollStepsPerWindow': 2,
            'scope': 'conditional finite control abstraction; OS primitives and compiler trusted'}


def compile_source(source, directory, optimization="-O0"):
    require(optimization in ("-O0", "-O2"), "unsupported compiler optimization")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'Main.hs'; path.write_text(source)
    exe = directory / 'worker'
    result = subprocess.run(['ghc', '-v0', optimization, '-threaded', '-with-rtsopts=-V0.001',
                    '-package', 'process-1.6.18.0', '-package', 'unix-2.7.3', '-package', 'bytestring-0.11.5.3',
                    '-ihaskell/research', '-outputdir', str(directory), str(path), '-o', str(exe)],
                   cwd=ROOT, capture_output=True, text=True, timeout=90)
    require(result.returncode == 0, 'GHC fixture failed: ' + result.stderr)
    return exe


def invoke(exe, args, payload='', seconds=3):
    result = subprocess.run([str(exe), *args], input=payload, capture_output=True,
                            text=True, timeout=seconds, check=True)
    require(not result.stderr, 'unexpected worker diagnostics')
    return result.stdout.strip()


def conformance(source):
    with tempfile.TemporaryDirectory(prefix='trader-inference-process-') as directory:
        root = Path(directory); exe = compile_source(source, root / 'base')
        # No input is needed for default/unknown modes. A pipe with no EOF stays open.
        for args in ([], ['--worker'], ['--unknown']):
            with subprocess.Popen([str(exe), *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True) as child:
                child.wait(timeout=3)
                require(child.stdout.read().strip() == '(Absent,True)', 'default read/spawned work')
        frames = ['IPV2 -1', 'IPV2 0', 'IPV2 1', 'IPV2 absent', 'IPV1 1', 'IPV2 2', 'IPV2 NaN']
        edges = [-2**80, -1, 0, 1, 19999999, 20000000, 2**80]
        cases = list(itertools.product(edges, (False, True), frames))
        rng = random.Random(20261003)
        cases += [(rng.randint(-30000000, 30000000), bool(rng.getrandbits(1)), rng.choice(frames)) for _ in range(128)]
        for elapsed, clean, frame in cases:
            payload = f'({elapsed},{clean},{json.dumps(frame)})\n'
            code = {'IPV2 -1': -1, 'IPV2 0': 0, 'IPV2 1': 1}.get(frame, 9)
            expected = admission(elapsed, clean, code)
            wanted = 'Absent' if expected is None else ('QuarterTarget (-1)' if expected == -1 else f'QuarterTarget {expected}')
            require(invoke(exe, ['--offline-inference-v2', '--contract'], payload) == wanted, 'compiled admission mismatch')
        request = repr(([0.0] * 12, [0.0] * 256 + [0.0, 0.0, 1.0])) + '\n'
        successes = 0
        for _ in range(10):
            answer = invoke(exe, ['--offline-inference-v2'], request)
            require(answer in ('(QuarterTarget 1,True)', '(Absent,True)'), 'wrong/unclean normal output')
            successes += answer == '(QuarterTarget 1,True)'
        require(successes > 0, 'no successful inference in ten attempts; performance boundary infeasible')
        # Synthetic nonconstant networks exercise the 12-16-3 evaluator. Numerical
        # parity here is a differential test, not a binary64 or tanh theorem.
        import numpy as np
        network_cases = 0
        for seed in (11, 23, 47):
            gen = np.random.default_rng(seed)
            for _ in range(4):
                x = gen.normal(size=12) * .1
                w1 = gen.normal(size=(12, 16)) * .1; b1 = gen.normal(size=16) * .1
                w2 = gen.normal(size=(16, 3)) * .1; b2 = gen.normal(size=3) * .1
                scores = np.tanh(x @ w1 + b1) @ w2 + b2
                winner = int(np.argmax(scores)) - 1
                data = repr((x.tolist(), np.concatenate((w1.ravel(), b1, w2.ravel(), b2)).tolist())) + '\n'
                require(invoke(exe, ['--offline-inference-v2', '--worker'], data) == f'IPV2 ready\nIPV2 {winner}', 'synthetic network mismatch')
                network_cases += 1
        bad = ['([] ,[])\n', 'not a request\n', 'x' * 32768 + '\n',
               repr(([0.0] * 12, [0.0] * 259)) + '\n',
               repr(([1001.0] * 12, [0.0] * 259)) + '\n',
               request.replace('0.0', 'NaN', 1), request.replace('0.0', '1e999', 1)]
        for data in bad:
            require(invoke(exe, ['--offline-inference-v2'], data) == '(Absent,True)', 'invalid request accepted')
        # With input held open, the request deadline still cancels the child.
        with subprocess.Popen([str(exe), '--offline-inference-v2'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True) as child:
            child.wait(timeout=3)
            require(child.stdout.read().strip() == '(Absent,True)', 'blocked request was not cancelled')
        line = '    let answer = raw >>= readMaybe >>= workerCompute'
        mutants = {
            'hang': source.replace(line, '    threadDelay 5000000\n' + line),
            'ignore_term': source.replace('import System.Posix.Signals (', 'import System.Posix.Signals (Handler (Ignore), installHandler, sigTERM, ').replace('    putStrLn "IPV2 ready"', '    void (installHandler sigTERM Ignore Nothing)\n    putStrLn "IPV2 ready"').replace(line, '    threadDelay 5000000\n' + line),
            'late': source.replace(line, '    threadDelay 80000\n' + line),
            'bad_reply': source.replace('maybe "IPV2 absent" (("IPV2 " ++) . show) answer', 'maybe "IPV2 absent" (const "IPV2 2") answer'),
            'eof': source.replace('    putStrLn (maybe "IPV2 absent" (("IPV2 " ++) . show) answer)', '    void (pure answer)'),
            'startup_hang': source.replace('    putStrLn "IPV2 ready"', '    threadDelay 5000000\n    putStrLn "IPV2 ready"'),
            'cleanup_failed': source.replace('pure (stopped && a && b && c)', 'pure (stopped && a && b && c && False)'),
        }
        mutants['cpu_hang'] = mutants['ignore_term'].replace('    threadDelay 5000000\n' + line,
            '    let spin n = n `seq` spin (n + 1)\n        answer = spin (0 :: Integer) :: Maybe Int')
        for name, mutation in mutants.items():
            pid_file = root / (name + '.pid')
            mutation = mutation.replace('import System.Posix.Signals (', 'import System.Posix.Process (getProcessID)\nimport System.Posix.Signals (')
            mutation = mutation.replace('worker = do', 'worker = do\n    getProcessID >>= writeFile ' + json.dumps(str(pid_file)) + ' . show')
            mutant = compile_source(mutation, root / name)
            expected = '(Absent,False)' if name == 'cleanup_failed' else '(Absent,True)'
            require(invoke(mutant, ['--offline-inference-v2'], request) == expected, 'fault escaped boundary: ' + name)
            require(pid_file.is_file(), 'worker never entered fault fixture: ' + name)
            try:
                os.kill(int(pid_file.read_text()), 0)
            except ProcessLookupError:
                pass
            else:
                raise ValueError('worker still exists after cleanup: ' + name)
    return {'guardCases': len(cases), 'generatedSeed': 20261003, 'networkSeeds': [11, 23, 47],
            'syntheticNetworkCases': network_cases, 'normalAttempts': 10, 'invalidRequestCases': len(bad),
            'defaultModes': 3, 'heldOpenInput': True, 'faultFixtures': sorted(mutants),
            'scope': 'compiled conformance and process tests, not OS or real-time proof'}


def check_process():
    cap, registry = extract()
    lock = json.loads((ROOT / 'formal/research/toolchain.json').read_text())
    require(lock['process'] == '1.6.18.0' and lock['unix'] == '2.7.3' and lock['bytestring'] == '0.11.5.3' and lock['researchWorkerRts'] == '-threaded -with-rtsopts=-V0.001', 'worker toolchain drift')
    return {'smt': prove_guard(cap), 'model': check_model(),
            'conformance': conformance((ROOT / SOURCE).read_text()),
            'isolation': {'requirement': 'F-RL-PROCESS-ISOLATION', 'status': 'exhaustively_checked',
                          'sourceSha256': registry['sha256'], 'dispatchCases': 6,
                          'launchCommands': 1, 'childEnvironment': [], 'productionComponents': 0,
                          'scope': 'reviewed closed source boundary under A-INFERENCE-PROCESS, not OS sandbox'}}
