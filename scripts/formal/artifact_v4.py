"""Source-bound artifact gate lemmas, finite lifecycle and compiled conformance."""
import ast
from collections import deque
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import z3 as z
import inference_process as process
from ppo_successor import certify

ROOT = Path(__file__).resolve().parents[2]
SOURCE = 'scripts/research/ppo_artifact_v4.py'
REGISTRY = 'formal/research/ppo-artifact-v4-source.json'
DEFINITIONS = {'_check', '_provenance', '_buffer', '_snapshot', '_restore', '_pack_snapshot',
               '_pack', '_json', '_unique', 'encode_artifact_v4', 'decode_artifact_v4', 'request_from_artifact_v4'}
PUBLIC = {'encode_artifact_v4', 'decode_artifact_v4', 'request_from_artifact_v4'}
HELPERS = {'scripts/research/ppo_successor_v2.py', 'scripts/research/optimizer_snapshot_v2.py',
           'scripts/research/ppo_inference_v3.py', 'haskell/research/SnapshotRequestV3.hs'}


def require(ok, reason):
    if not ok:
        raise ValueError('PPO artifact v4: ' + reason)


def extract(source=None, registry=None):
    source = (ROOT / SOURCE).read_text() if source is None else source
    registry = json.loads((ROOT / REGISTRY).read_text()) if registry is None else registry
    tree = ast.parse(source)
    nodes = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    require(set(nodes) == DEFINITIONS and set(registry['helperHashes']) == HELPERS, 'mandatory coverage')
    require(ast.dump(tree, include_attributes=False) == registry['moduleAST'], 'unreviewed source')
    for name in PUBLIC:
        require(ast.unparse(nodes[name].body[0]) == 'if enabled is not True:\n    return None', 'enable guard')
        require([ast.unparse(v) for v in nodes[name].args.kw_defaults] == ['False'], 'default enabled')
    for path, digest in registry['helperHashes'].items():
        require(hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest, 'helper drift')
    require(not any('ppo_artifact_v4' in p.read_text() for p in (ROOT/'haskell/app').rglob('*.hs')), 'production reference')
    # Exact AST of the reviewed source is the control correspondence boundary.
    # The compared digest values are modeled as strings, not a SHA implementation.
    decoder = nodes['decode_artifact_v4']
    digest_if = decoder.body[2].body[2]
    require(ast.unparse(digest_if.test) == 'hashlib.sha256(raw).hexdigest() != expected_sha256', 'digest predicate')
    size = decoder.body[1].test.values[1].operand
    require(ast.unparse(size) == '0 < len(raw) <= 65536', 'byte bound')
    meta = decoder.body[2].body[5]
    expected_meta = ast.parse("value['schema'] != VERSION or value['contracts'] != CONTRACTS or value['enabled'] is not False or value['promotion'] != 'research-only' or _provenance(value['provenance']) != expected", mode='eval').body
    require(ast.dump(meta.test) == ast.dump(expected_meta), 'metadata predicate')
    return int(size.comparators[1].value)


def prove_guards(cap):
    actual, expected = z.Strings('artifact_actual_digest artifact_expected_digest')
    certify(z.Not(actual != expected), actual == expected)
    schema, contracts, enabled, promotion, provenance = z.Bools('artifact_schema artifact_contracts artifact_disabled artifact_promotion artifact_provenance')
    certify(z.Not(z.Or(z.Not(schema), z.Not(contracts), z.Not(enabled), z.Not(promotion), z.Not(provenance))),
            z.And(schema, contracts, enabled, promotion, provenance))
    size, steps = z.Ints('artifact_size artifact_steps')
    certify(z.And(size > 0, size <= cap), z.And(size >= 1, size <= 65536))
    completed = 4 * ((steps + 255) / 256)
    certify(z.And(steps >= 1, steps <= 4096), z.And(completed >= 4, completed <= 64, completed % 4 == 0))
    return {'F-RL-ARTIFACT-V4-GUARD': 'unsat'}


GATES = ('enabled', 'bytes', 'expected_provenance', 'digest', 'json', 'fields',
         'metadata', 'restore', 'canonical', 'bridge')


def check_model(mutant=False):
    initial = (0, 0, False)
    queue = deque([initial]); seen = {initial: 0}; edges = 0
    while queue:
        index, passed, failed = state = queue.popleft()
        require(not failed or index < len(GATES), 'failure published')
        require(passed == (1 << index)-1, 'gate skipped')
        if failed or index == len(GATES):
            targets = []
        else:
            targets = [(index+1, passed | (1 << index), False), (index, passed, True)]
            if mutant and GATES[index] == 'digest':
                targets.append((len(GATES), passed, False))
        for target in targets:
            edges += 1
            require(target[2] or target[0] > index, 'nontermination')
            if target not in seen:
                seen[target] = seen[state]+1; queue.append(target)
    require((len(GATES), (1 << len(GATES))-1, False) in seen, 'vacuous publication')
    return {'states': len(seen), 'transitions': edges, 'maxShortestDepth': max(seen.values()),
            'gates': list(GATES), 'requests': 1, 'retries': 0, 'authorizationCapabilities': 0,
            'scope': 'atomic returning gates, sequential one-shot byte-to-request path'}


def provenance():
    return {**{k: 'a'*n for k, n in [('codeCommit',40), ('registrationSha256',64),
            ('dataSha256',64), ('fundingSha256',64), ('splitSha256',64), ('proofSha256',64)]},
            'datasetRole': 'development'}


def conformance():
    sys.path.insert(0, str(ROOT/'scripts/research'))
    import numpy as np
    # Distinct module name avoids collision with this proof module.
    import importlib.util
    spec = importlib.util.spec_from_file_location('artifact_implementation_v4', ROOT/SOURCE)
    codec = importlib.util.module_from_spec(spec); spec.loader.exec_module(codec)
    from ppo_successor_v2 import train_ppo_v2
    from ppo_inference_v3 import encode_request_v3
    x = np.arange(180, dtype=float)
    prices = {'ALPHA':100*np.exp(.0002*x+.002*np.sin(x/9)),
              'BETA':200*np.exp(-.0001*x+.003*np.sin(x/11))}
    funding = {s: np.zeros(len(p)) for s,p in prices.items()}
    prov = provenance(); cases = 0; sizes = []
    with tempfile.TemporaryDirectory(prefix='trader-artifact-v4-') as directory:
        exe = process.compile_source((ROOT/process.SOURCE).read_text(), Path(directory), optimization='-O2')
        for seed in (11,23,47):
            for horizon in (1,3,6):
                result = train_ppo_v2(prices,funding,horizon,seed,17,enabled=True)
                require(result is not None, 'training failed')
                raw = codec.encode_artifact_v4(result, prov, enabled=True)
                require(raw is not None, 'encoding failed'); sizes.append(len(raw))
                digest = hashlib.sha256(raw).hexdigest()
                restored = codec.decode_artifact_v4(raw,digest,prov,enabled=True)
                require(restored == result and codec.encode_artifact_v4(restored,prov,enabled=True) == raw, 'state or bytes drift')
                observation = np.linspace(-.5,.5,12)
                frame = codec.request_from_artifact_v4(raw,digest,prov,observation,enabled=True)
                require(frame == encode_request_v3(result,observation,enabled=True), 'bridge drift')
                _, _, ow, pw = ast.literal_eval(frame.decode('ascii'))
                bits = process.invoke(exe, ['--snapshot-contract-v3'], frame.decode('ascii'))
                require(bits.startswith('Just ') and ast.literal_eval(bits[5:]) == (ow,pw), 'compiled roundtrip')
                require(codec.encode_artifact_v4(result,prov) is None and
                        codec.decode_artifact_v4(raw,digest,prov) is None and
                        codec.request_from_artifact_v4(raw,digest,prov,observation) is None, 'default enabled')
                cases += 1
    invalid = [b'', b'x'*65537, b'null', b'[]', b'NaN', b'1e999', b'\xff', b'['*3000,
               raw+b' ', raw.replace(b'"schema":', b'"schema":"ppo-artifact-v4","schema":',1)]
    document = json.loads(raw)
    mutations = [(['schema'],'unknown'), (['contracts',0],'old'), (['enabled'],True),
                 (['enabled'],0), (['promotion'],'paper'), (['provenance','datasetRole'],'holdout'),
                 (['provenance','codeCommit'],'b'*40), (['result','seed'],True), (['result','steps'],4097),
                 (['result','symbols'],['BETA','ALPHA']), (['result','actor','step'],0),
                 (['result','actor','outputs'],True), (['result','actor','version'],'v1'),
                 (['result','scale',0],'00'), (['result','losses',0],'nan'),
                 (['result','actor','p',0],(np.full(192,np.nan).astype('<f8').tobytes()).hex()),
                 (['result','actor','v',0],(np.full(192,-1.).astype('<f8').tobytes()).hex())]
    for path, value in mutations:
        changed = json.loads(raw); target = changed
        for key in path[:-1]: target = target[key]
        target[path[-1]] = value
        invalid.append(codec._json(changed))
    for field in document:
        changed = json.loads(raw); del changed[field]; invalid.append(codec._json(changed))
    changed = json.loads(raw); changed['extra']=0; invalid.append(codec._json(changed))
    for bad in invalid:
        digest = hashlib.sha256(bad).hexdigest()
        require(codec.decode_artifact_v4(bad,digest,prov,enabled=True) is None, 'malformed restored')
        require(codec.request_from_artifact_v4(bad,digest,prov,np.zeros(12),enabled=True) is None, 'malformed request')
    require(codec.decode_artifact_v4(raw,'0'*64,prov,enabled=True) is None, 'hash bypass')
    require(codec.decode_artifact_v4(bytearray(raw),hashlib.sha256(raw).hexdigest(),prov,enabled=True) is None, 'mutable raw')
    reference_failures = 0
    for field in prov:
        changed = dict(prov); changed[field] = 'holdout' if field == 'datasetRole' else 'b'*len(prov[field])
        require(codec.decode_artifact_v4(raw,hashlib.sha256(raw).hexdigest(),changed,enabled=True) is None, 'reference mismatch')
        reference_failures += 1
    for enabled in (False, None, 0, 1, 'true'):
        require(codec.decode_artifact_v4(raw,hashlib.sha256(raw).hexdigest(),prov,enabled=enabled) is None, 'non-Boolean enablement')
    for observation in (np.zeros(11), np.full(12,np.nan), np.full(12,1001.), None):
        require(codec.request_from_artifact_v4(raw,hashlib.sha256(raw).hexdigest(),prov,observation,enabled=True) is None, 'invalid observation request')
    generated = 0
    rng = np.random.default_rng(20261005)
    for _ in range(128):
        words = rng.integers(0,2**64,size=192,dtype=np.uint64)
        # Keep every generated bit pattern finite, including signed zero/subnormals.
        words &= np.uint64(0xffefffffffffffff)
        words[:4] = [0,2**63,1,2**63+1]
        actor = replace(result.actor,p=(words.astype('<u8').tobytes(),*result.actor.p[1:]))
        changed = replace(result,actor=actor)
        encoded = codec.encode_artifact_v4(changed,prov,enabled=True)
        require(encoded is not None and codec.decode_artifact_v4(encoded,hashlib.sha256(encoded).hexdigest(),prov,enabled=True) == changed, 'generated bit loss')
        generated += 1
    return {'fits':cases,'seeds':[11,23,47],'horizons':[1,3,6],'steps':17,'compiledBitRoundtrips':cases,
            'invalidArtifacts':len(invalid),'referenceMismatches':reference_failures,'generatedBitCases':generated,'artifactBytes':sizes,
            'scope':'synthetic engineering state/frame equality; no market or provenance-authenticity evidence'}


def check_artifact_v4():
    return {'smt':prove_guards(extract()),'model':check_model(),'conformance':conformance(),
            'boundary':{'status':'exhaustively_checked','definitions':sorted(DEFINITIONS),
                        'helpers':sorted(HELPERS),'public':sorted(PUBLIC),'defaultEnabled':False,
                        'scope':'complete reviewed source/effect inventory under runtime assumptions'}}


def benchmark():
    import statistics
    import time
    sys.path.insert(0, str(ROOT/'scripts/research'))
    import numpy as np
    import ppo_artifact_v4 as codec
    from ppo_successor_v2 import train_ppo_v2
    x = np.arange(180,dtype=float)
    prices = {'ALPHA':100*np.exp(.0002*x+.002*np.sin(x/9)),
              'BETA':200*np.exp(-.0001*x+.003*np.sin(x/11))}
    result = train_ppo_v2(prices,{s:np.zeros(180) for s in prices},1,11,17,enabled=True)
    require(result is not None,'benchmark training')
    prov = provenance(); raw = codec.encode_artifact_v4(result,prov,enabled=True)
    digest = hashlib.sha256(raw).hexdigest(); timings = {'encode':[], 'decode':[], 'request':[]}
    calls = {'encode':lambda:codec.encode_artifact_v4(result,prov,enabled=True),
             'decode':lambda:codec.decode_artifact_v4(raw,digest,prov,enabled=True),
             'request':lambda:codec.request_from_artifact_v4(raw,digest,prov,np.zeros(12),enabled=True)}
    for name, call in calls.items():
        for _ in range(100):
            start = time.perf_counter_ns(); value = call(); elapsed = time.perf_counter_ns()-start
            require(value is not None,'benchmark absence'); timings[name].append(elapsed/1e6)
    return {'schemaVersion':1,'kind':'synthetic engineering benchmark','seed':11,'horizon':1,'steps':17,
            'artifactBytes':len(raw),'iterations':100,
            'milliseconds':{k:{'min':min(v),'median':statistics.median(v),'max':max(v)} for k,v in timings.items()},
            'scope':'byte codec only; excludes disk/network, Haskell inference, startup and economic evidence; not a deadline guarantee'}
