"""Conditional trained-request composition, codec SMT and compiled conformance."""
import ast
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import z3 as z
import inference_process as process
from ppo_successor import certify

ROOT = Path(__file__).resolve().parents[2]
SOURCE = 'scripts/research/ppo_inference_v3.py'
DECODER = 'haskell/research/SnapshotRequestV3.hs'
REGISTRY = 'formal/research/ppo-process-bridge-source.json'
HELPERS = {'scripts/research/ppo_successor_v2.py', 'scripts/research/optimizer_snapshot_v2.py',
           process.SOURCE}
DEFINITIONS = {'_words', 'encode_request_v3'}


def require(ok, reason):
    if not ok:
        raise ValueError('PPO process bridge: ' + reason)


def extract(source=None, decoder=None, registry=None):
    source = (ROOT / SOURCE).read_text() if source is None else source
    decoder = (ROOT / DECODER).read_text() if decoder is None else decoder
    registry = json.loads((ROOT / REGISTRY).read_text()) if registry is None else registry
    tree = ast.parse(source)
    nodes = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    require(registry['schemaVersion'] == 1, 'schema')
    require(set(nodes) == DEFINITIONS and set(registry['definitions']) == DEFINITIONS and
            set(registry['helperHashes']) == HELPERS, 'mandatory coverage omitted')
    require(ast.dump(tree, include_attributes=False) == registry['moduleAST'], 'unreviewed Python source')
    for name, node in nodes.items():
        require(ast.dump(node, include_attributes=False) == registry['definitions'][name], 'body drift')
    require(decoder == registry['decoderSource'], 'unreviewed Haskell decoder')
    require(re.findall(r'^import (.+)$', decoder, re.M) ==
            ["Data.Char (isDigit)", "Data.List (foldl', stripPrefix)", 'Data.Word (Word64)', 'GHC.Float (castDoubleToWord64, castWord64ToDouble)', 'Text.Read (readMaybe)'], 'effectful decoder import')
    for path, expected in registry['helperHashes'].items():
        require(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, 'helper drift')
    entry = nodes['encode_request_v3']
    require(ast.unparse(entry.body[0]) == 'if enabled is not True or type(version) is not str or version != VERSION:\n    return None', 'public guard')
    require([ast.unparse(v) for v in entry.args.kw_defaults] == ['False', 'VERSION'], 'public defaults')
    # Fixed restricted Haskell guard grammar. Unknown syntax has no translation.
    word = re.search(r'word < (\d+) \|\| word >= (\d+) = Nothing', decoder)
    tag = re.search(r'b <- literal "\\"([^\\]+)\\"" a', decoder)
    metadata = re.search(r'if not \(null \(spaces end\)\) \|\| step < (\d+) \|\| step > (\d+) \|\| step `mod` (\d+) /= 0 \|\| length observation /= (\d+) \|\| length parameters /= (\d+)', decoder)
    value = re.search(r'if isNaN value \|\| isInfinite value \|\| abs value > (\d+) then Nothing else Just value', decoder)
    require(word and tag and metadata and value, 'unsupported guard grammar')
    require('castWord64ToDouble (fromInteger word :: Word64)' in decoder, 'bit conversion drift')
    require("span isDigit" in decoder and "length digits > 20" in decoder and
            "n * 10 + toInteger (fromEnum c - fromEnum '0')" in decoder and
            "go (remaining - 1) next" in decoder and "remaining <= 0" in decoder, 'bounded parser skeleton drift')
    require('actor.step != 4 * ((result.steps + 255) // 256)' in source, 'completed step relation drift')
    process.extract()
    return tuple(map(int, word.groups())), (tag[1], *metadata.groups()), int(value[1])


def prove_codec(extracted):
    (low, high), (tag, lower, upper, divisor, obs, params), bound = extracted
    lower, upper, divisor, obs, params = map(int, (lower, upper, divisor, obs, params))
    word, step, steps, width, count = z.Ints('bridge_word bridge_step bridge_steps bridge_width bridge_count')
    accepted_word = z.And(word >= low, word < high)
    # Checked Integer -> Word64 has the mathematical modulo conversion semantics.
    certify(accepted_word, z.And(word >= 0, word < 2**64, word % 2**64 == word))
    version = z.String('bridge_version')
    metadata = z.And(version == tag, step >= lower, step <= upper, step % divisor == 0,
                     width == obs, count == params)
    certify(metadata, z.And(version == 'PPO-SNAPSHOT-V3', step >= 4, step <= 64,
                           step % 4 == 0, width == 12, count == 192 + 16 + 48 + 3))
    completed = 4 * ((steps + 255) / 256)
    certify(z.And(steps >= 1, steps <= 4096),
            z.And(completed >= lower, completed <= upper, completed % divisor == 0))
    value = z.FP('bridge_value', z.Float64())
    safe = z.Not(z.Or(z.fpIsNaN(value), z.fpIsInf(value),
                      z.fpGT(z.fpAbs(value), z.FPVal(bound, z.Float64()))))
    certify(safe, z.And(z.Not(z.fpIsNaN(value)), z.Not(z.fpIsInf(value)),
                       z.fpLEQ(z.fpAbs(value), z.FPVal(1000, z.Float64()))))
    prefix, digit, remaining = z.Ints('bridge_prefix bridge_digit bridge_remaining')
    for width in range(20):
        certify(z.And(prefix >= 0, prefix < 10**width, digit >= 0, digit <= 9),
                z.And(prefix*10+digit >= 0, prefix*10+digit < 10**(width+1)))
    certify(remaining > 0, z.And(remaining - 1 >= 0, remaining - 1 < remaining))
    return {'F-RL-BRIDGE-V3-CODEC': 'unsat'}


# Extend the actual supervised-process abstraction with decoder and revalidation
# outcomes. No neural accuracy, parser/runtime or OS termination theorem follows.
def next_states(state, mutant=False):
    phase, decoded, valid, elapsed, code, polls = state
    if phase == 'decode':
        return [('validate', True, False, elapsed, 9, 0),
                ('pending' if mutant else 'term', False, False, elapsed, 9, 2)]
    if phase == 'validate':
        return [('pending', decoded, True, elapsed, 9, 0), ('term', decoded, False, elapsed, 9, 2)]
    result = []
    for p, t, c, n in process.successors((phase, elapsed, code, polls)):
        result.append(('decode' if phase == 'ready' and p == 'pending' else p, decoded, valid, t, c, n))
    return result


def rank(state):
    phase, _, _, elapsed, code, polls = state
    if phase == 'launch': return 13
    if phase == 'ready': return 12
    if phase == 'decode': return 11
    if phase == 'validate': return 10
    return process.rank((phase, elapsed, code, polls))


def check_model(mutant=False):
    initial = [('disabled', False, False, 0, 9, 0), ('launch', False, False, 0, 9, 0)]
    seen = {s: 0 for s in initial}; queue = deque(initial); edges = accepted = 0
    while queue:
        state = queue.popleft()
        phase, decoded, valid, elapsed, code, _ = state
        if code != 9:
            require(decoded and valid, 'request bypassed decoder or validation')
        action = process.admission(elapsed, phase == 'done', code, cap=2)
        if action is not None:
            require(decoded and valid and phase == 'done' and elapsed < 2, 'unsafe publication')
            accepted += 1
        targets = next_states(state, mutant)
        require(bool(targets) == (phase not in ('disabled', 'done', 'failed')), 'unexpected deadlock')
        for target in targets:
            require(rank(target) < rank(state), 'nonterminating control path')
            edges += 1
            if target not in seen:
                seen[target] = seen[state] + 1; queue.append(target)
    require(accepted > 0, 'no successful path')
    return {'states': len(seen), 'transitions': edges, 'maxShortestDepth': max(seen.values()),
            'acceptedStates': accepted, 'initialRank': 13, 'children': 1, 'requests': 1,
            'timeBuckets': [0, 1, 2], 'pollStepsPerWindow': 2,
            'scope': 'all abstract gate outcomes; reviewed source control plus process primitive assumptions'}


def conformance():
    sys.path.insert(0, str(ROOT / 'scripts/research'))
    import numpy as np
    from ppo_successor_v2 import train_ppo_v2
    from ppo_inference_v3 import encode_request_v3
    source = (ROOT / process.SOURCE).read_text()
    x = np.arange(180, dtype=float)
    prices = {'ALPHA': 100 * np.exp(.0002*x + .002*np.sin(x/9)),
              'BETA': 200 * np.exp(-.0001*x + .003*np.sin(x/11))}
    funding = {s: np.zeros(len(p)) for s, p in prices.items()}
    rng = np.random.default_rng(20261005)
    cases = fits = successes = 0
    with tempfile.TemporaryDirectory(prefix='trader-ppo-bridge-') as directory:
        root = Path(directory); exe = process.compile_source(source, root / 'base', optimization='-O2')
        for seed in (11, 23, 47):
            for horizon in (1, 3, 6):
                result = train_ppo_v2(prices, funding, horizon, seed, 17, enabled=True)
                require(result is not None, 'training failed'); fits += 1
                w1, b1, w2, b2 = [np.frombuffer(b, dtype='<f8').reshape(shape) for b, shape in
                                  zip(result.actor.p, ((12,16),(16,),(16,3),(3,)))]
                policy_successes = successes
                for _ in range(3):
                    observation = rng.normal(0, .5, 12)
                    frame = encode_request_v3(result, observation, enabled=True)
                    require(frame is not None, 'request rejected')
                    text = frame.decode('ascii')
                    _, _, ow, pw = ast.literal_eval(text)
                    bits = process.invoke(exe, ['--snapshot-contract-v3'], text)
                    require(bits.startswith('Just '), 'compiled decoder rejected')
                    require(ast.literal_eval(bits[5:]) == (ow, pw), 'bit roundtrip differs')
                    scores = np.tanh(observation @ w1 + b1) @ w2 + b2
                    require(np.diff(np.sort(scores))[-1] > 1e-9, 'fixture near tie requires separate analysis')
                    action = int(scores.argmax()) - 1
                    expected = f'(QuarterTarget {action if action >= 0 else "(-1)"},True)'
                    actual = process.invoke(exe, ['--offline-snapshot-v3'], text)
                    require(actual in (expected, '(Absent,True)'), 'trained policy action differs')
                    successes += actual == expected; cases += 1
                require(successes > policy_successes, 'no actual inference for trained policy')
        require(successes > 0, 'no actual trained-policy success')
        # Bit patterns include signed zero and subnormals; no neural parity claim.
        special = [0, 2**63, 1, 2**63+1, 0x0010000000000000, 0x3ff0000000000000]
        for step in range(4, 65, 4):
            ow = (special * 2); pw = (special * 44)[:259]
            text = f'("PPO-SNAPSHOT-V3",{step},{ow},{pw})\n'
            bits = process.invoke(exe, ['--snapshot-contract-v3'], text)
            require(bits.startswith('Just ') and ast.literal_eval(bits[5:]) == (ow, pw), 'edge bit transport')
        ow, pw = [0]*12, [0]*259
        bad = [f'("{tag}",{step},{o},{p})\n' for tag, step, o, p in
               [('v2',4,ow,pw),('PPO-SNAPSHOT-V3',0,ow,pw),('PPO-SNAPSHOT-V3',5,ow,pw),
                ('PPO-SNAPSHOT-V3',68,ow,pw),('PPO-SNAPSHOT-V3',4,ow[:-1],pw),
                ('PPO-SNAPSHOT-V3',4,ow,pw[:-1])]]
        for word in (-1, 2**64, 0x7ff0000000000000, 0xfff0000000000000,
                     0x7ff8000000000000, 0x408f480000000000):
            bad.append(f'("PPO-SNAPSHOT-V3",4,{[word]+ow[1:]},{pw})\n')
        bad += ['malformed\n', 'x'*32768+'\n',
                f'("PPO-SNAPSHOT-V3",٤,{ow},{pw})\n',
                f'("PPO-SNAPSHOT-V3",４,{ow},{pw})\n',
                f'("PPO-SNAPSHOT-V3",4,{ow},{pw})junk\n',
                f'("PPO-SNAPSHOT-V3",+4,{ow},{pw})\n',
                f'("PPO-SNAPSHOT-V3",4,{ow},[0,])\n',
                f'("PPO-SNAPSHOT-V3",4,{[10**20]+ow[1:]},{pw})\n']
        for text in bad:
            require(process.invoke(exe, ['--snapshot-contract-v3'], text) == 'Nothing', 'invalid codec accepted')
            require(process.invoke(exe, ['--offline-snapshot-v3'], text) == '(Absent,True)', 'invalid request acted')
        tie = f'("PPO-SNAPSHOT-V3",4,{ow},{pw})\n'
        require(process.invoke(exe, ['--offline-snapshot-v3'], tie) == '(Absent,True)', 'tie did not abstain')
        with subprocess.Popen([str(exe), '--offline-snapshot-v3'], stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, text=True) as child:
            child.wait(timeout=3)
            require(child.stdout.read().strip() == '(Absent,True)', 'input deadline failed')
        # Real child hangs/invalid reply and cleanup-failure branches in the new mode.
        line = '    let answer = raw >>= readMaybe >>= workerCompute'
        faults = {'hang': source.replace(line, '    threadDelay 1000000\n'+line),
                  'cleanup_failed': source.replace('pure (stopped && a && b && c)', 'pure (stopped && a && b && c && False)'),
                  'bad_reply': source.replace('putStrLn (maybe "IPV2 absent" (("IPV2 " ++) . show) answer)', 'putStrLn "IPV2 7"')}
        for name, mutated in faults.items():
            require(mutated != source, 'fixture failed to mutate')
            pid_file = root / (name + '.pid')
            mutated = mutated.replace('import System.Posix.Signals (', 'import System.Posix.Process (getProcessID)\nimport System.Posix.Signals (')
            mutated = mutated.replace('worker = do', 'worker = do\n    getProcessID >>= writeFile ' + json.dumps(str(pid_file)) + ' . show')
            faulty = process.compile_source(mutated, root / name, optimization="-O2")
            expected = '(Absent,False)' if name == 'cleanup_failed' else '(Absent,True)'
            require(process.invoke(faulty, ['--offline-snapshot-v3'], frame.decode('ascii')) == expected, 'fault escaped: '+name)
            require(pid_file.is_file(), 'worker never reached fixture')
            try:
                os.kill(int(pid_file.read_text()), 0)
            except ProcessLookupError:
                pass
            else:
                raise ValueError('bridge worker remains after cleanup')
    return {'syntheticFits': fits, 'seeds': [11,23,47], 'horizons': [1,3,6], 'steps': 17,
            'trainedObservationCases': cases, 'successfulInferenceObserved': successes > 0,
            'bitEdgeCases': 16, 'invalidCases': len(bad), 'heldOpenInput': True,
            'faultFixtures': sorted(faults), 'financialTrials': 0,
            'scope': 'actual process/codec tests; neither neural numerical equivalence nor OS proof'}


def check_bridge():
    lock = json.loads((ROOT / 'formal/research/toolchain.json').read_text())
    require(lock['base'] == '4.17.2.1' and lock['researchBridgeOptimization'] == '-O2', 'bridge toolchain drift')
    require(subprocess.check_output(['ghc-pkg', 'field', 'base', 'version', '--simple-output'], text=True).strip() == lock['base'], 'base version mismatch')
    extracted = extract()
    return {'smt': prove_codec(extracted), 'model': check_model(), 'conformance': conformance(),
            'boundary': {'requirement': 'F-RL-BRIDGE-V3-BOUNDARY', 'status': 'exhaustively_checked',
                         'definitions': sorted(DEFINITIONS), 'helperSources': sorted(HELPERS),
                         'defaultEnabled': False, 'authorizationCapabilities': 0,
                         'scope': 'complete reviewed source boundary under named runtime/helper assumptions'}}


def benchmark():
    """Reproduce engineering timing only; repeated existing synthetic configuration."""
    import time
    import statistics
    sys.path.insert(0, str(ROOT / 'scripts/research'))
    import numpy as np
    from ppo_successor_v2 import train_ppo_v2
    from ppo_inference_v3 import encode_request_v3
    x = np.arange(180, dtype=float)
    p = {'ALPHA':100*np.exp(.0002*x+.002*np.sin(x/9)),
         'BETA':200*np.exp(-.0001*x+.003*np.sin(x/11))}
    f = {s:np.zeros(len(v)) for s,v in p.items()}
    start = time.perf_counter(); result = train_ppo_v2(p,f,1,11,17,enabled=True)
    training_ms = (time.perf_counter()-start)*1000
    require(result is not None, 'benchmark training failed')
    rng = np.random.default_rng(20261005)
    start = time.perf_counter()
    frames = [encode_request_v3(result,rng.normal(0,.5,12),enabled=True) for _ in range(3)]
    encoding_ms = (time.perf_counter()-start)*1000/3
    source = (ROOT / process.SOURCE).read_text(); runs = []
    with tempfile.TemporaryDirectory(prefix='trader-bridge-benchmark-') as directory:
        for flag in ('-O0','-O2'):
            exe = process.compile_source(source,Path(directory)/flag,optimization=flag)
            latencies=[]; absences=0
            for frame in frames*10:
                start=time.perf_counter()
                answer=process.invoke(exe,['--offline-snapshot-v3'],frame.decode('ascii'))
                latencies.append((time.perf_counter()-start)*1000)
                require(answer.endswith(',True)'), 'benchmark cleanup failed')
                absences += answer == '(Absent,True)'
            runs.append({'ghcOptimization':flag,'requests':len(latencies),'absences':absences,
                         'endToEndMilliseconds':{'minimum':min(latencies),'median':statistics.median(latencies),'maximum':max(latencies)},
                         'temporaryExecutableBytes':exe.stat().st_size})
    return {'schemaVersion':1,'kind':'synthetic engineering benchmark; not economic evidence',
            'seed':11,'horizon':1,'steps':17,'observations':3,'repetitions':10,
            'trainingMilliseconds':training_ms,'meanEncodingMilliseconds':encoding_ms,
            'requestBytes':[len(f) for f in frames],'runs':runs,
            'scope':'wall time includes process startup and cleanup; scheduling dependent, not a hard timing bound'}


if __name__ == '__main__':
    require(sys.argv[1:] == ['--benchmark'], 'only explicit benchmark command supported')
    print(json.dumps(benchmark(),indent=2))
