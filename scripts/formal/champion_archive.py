"""Source-bound archive preservation; explicit filesystem/primitive assumptions.

Existing file identities are protected without assuming disjoint output paths.
This is not a filesystem sandbox, crash-durability or whole-language proof.
"""
import ast
from collections import deque
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import z3 as z
from data_composition import definition
from promotion_boundary import extract as promotion_extract, shape, source_path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = 'scripts/research/summarize_sequential_screen.py'
LEGACY = 'formal/research/champion-archive-legacy.json'
FIXTURE = 'formal/research/fixtures/champion-archive.json'
REGISTRY = 'formal/research/champion-archive-source.json'


def require(ok, reason):
    if not ok:
        raise ValueError('champion archive: ' + reason)


def extract(source=None, registry=None):
    source = (ROOT/SOURCE).read_text() if source is None else source
    registry = json.loads((ROOT/REGISTRY).read_text()) if registry is None else registry
    require(set(registry) == {'schemaVersion', 'exporter', 'supportHashes'} and registry['schemaVersion'] == 1, 'registry schema')
    expected_support = {LEGACY, FIXTURE, source_path('run_sequential_screen'), source_path('sequential_learning'),
                        'scripts/formal/promotion_boundary.py', 'formal/research/promotion-boundary-source.json'}
    require(set(registry['supportHashes']) == expected_support, 'support coverage omitted')
    require(source == registry['exporter'], 'complete exporter drift')
    for path, expected in registry['supportHashes'].items():
        require(hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == expected, 'support drift: ' + path)
    # Require the independently reviewed complete reviewed effect/callback roster.
    trees, surface = promotion_extract()
    require(shape(ast.parse(source)) == shape(trees['summarize_sequential_screen']), 'exporter/effect composition drift')
    fn = definition(ast.parse(source), 'export')
    tail = ast.parse('''output.mkdir(parents=True, exist_ok=False)
for name, content in reports.items():
    with (output / name).open("xb") as stream:
        stream.write(content)
''').body
    require([shape(n) for n in fn.body[-2:]] == [shape(n) for n in tail], 'exclusive export tail')
    writes = [entry for entry in surface['fileOpens'] if entry[2] != 'rb']
    require(len(writes) == 5 and all(mode in ('x', 'xb') for _, _, mode in writes), 'exclusive writer coverage')
    require(not any('write_bytes' in call or 'write_text' in call for _, call in surface['writeSites']), 'unguarded path write')
    runner = definition(trees['run_sequential_screen'], 'write_json')
    expected = ast.parse('''with path.open("x") as stream:
    json.dump(value, stream, sort_keys=True, allow_nan=False, indent=2)
    stream.write("\\n")
''').body
    require([shape(n) for n in runner.body] == [shape(n) for n in expected], 'runner writer body')
    # save_policy's complete serialization and exclusive-create tail is checked
    # by promotion_extract -> check_saved_default; no other writer is admitted.
    return {'status': 'exhaustively_checked', 'writerSites': writes,
            'researchModules': surface['moduleCount'], 'reviewedCalls': surface['callSites'],
            'callbacks': surface['callbacks'], 'destinations': surface['destinations'],
            'processes': surface['processes'],
            'scope': 'actual writer tails plus complete reviewed effect/consumer composition; trusted primitive semantics'}


def prove(premise, claim):
    solver = z.Solver(); solver.set(timeout=10000, random_seed=0)
    solver.add(premise)
    require(solver.check() == z.sat, 'unsatisfied or unknown premise')
    solver.add(z.Not(claim))
    require(solver.check() == z.unsat, 'counterexample or unknown')


def preserve(exclusive=True):
    # Map keys/identities and opaque byte contents use unbounded SMT integers.
    names = z.Array('archive_names', z.IntSort(), z.IntSort())
    data = z.Array('archive_bytes', z.IntSort(), z.IntSort())
    key, protected, epoch, payload = z.Ints('archive_key archive_protected archive_epoch archive_payload')
    old = z.Select(names, key); existing = old >= 0
    accepted = z.Or(z.Not(existing), z.Not(z.BoolVal(exclusive)))
    fresh = epoch + 1
    handle = z.If(existing, old, fresh)
    after_names = z.If(accepted, z.Store(names, key, handle), names)
    # Successful creation starts empty; a truncating mode also empties old data.
    opened = z.If(accepted, z.Store(data, handle, 0), data)
    written = z.If(accepted, z.Store(opened, handle, payload), data)
    p = z.Select(names, protected)
    premise = z.And(epoch >= 0, old >= -1, old <= epoch, p >= 0, p <= epoch)
    prove(premise, z.Select(after_names, protected) == p)
    prove(premise, z.Select(opened, p) == z.Select(data, p))
    prove(premise, z.Select(written, p) == z.Select(data, p))
    prove(z.And(premise, existing), z.And(z.Not(accepted), after_names == names, written == data))
    prove(z.And(premise, z.Not(existing)), z.And(accepted, handle > epoch, handle != p))
    # Failed acquisition cannot provide a writable handle. A later retry at a
    # successfully created pathname rejects; failure never triggers unlink/retry.
    retry = z.Select(after_names, key) < 0
    prove(z.And(premise, accepted), z.Not(retry))
    other = z.Int('archive_other')
    prove(z.And(premise, other != key), z.Select(after_names, other) == z.Select(names, other))
    epoch2 = z.Int('archive_epoch2')
    prove(z.And(epoch >= 0, epoch2 > epoch), epoch + 1 != epoch2 + 1)
    return {'F-RL-ARCHIVE-PRESERVE': 'unsat'}


# Two namespace keys; object0 is initially protected; object3 is an inserted
# collision. Objects1/2 are fresh writer identities. No deletion/parent retarget.
INITIAL = ((0, -1), (7, 0, 0, 7), ('new', 'new'), (-1, -1), (-1, -1), False)


def successors(state, legacy=False):
    names, data, phases, handles, keys, injected = state
    out = []
    def replace(values, index, value):
        return values[:index] + (value,) + values[index+1:]
    if not injected and names[1] == -1:
        out.append(('insert-collision', ((names[0], 3), data, phases, handles, keys, True)))
    for i, phase in enumerate(phases):
        def add(label, phase=phase, handle=handles[i], key=keys[i], ns=names, bs=data):
            out.append((str(i)+':'+label, (ns, bs, replace(phases, i, phase),
                       replace(handles, i, handle), replace(keys, i, key), injected)))
        if phase == 'new':
            for key in ((1,) if i == 0 else (0, 1)):
                add('select-'+str(key), phase='ready', key=key)
        elif phase == 'ready':
            key = keys[i]; old = names[key]
            add('open-error', phase='failed')
            if old < 0 or (legacy and i == 0):
                handle = old if old >= 0 else i+1
                add('open', phase='open', handle=handle, ns=replace(names, key, handle), bs=replace(data, handle, 0))
            else:
                add('collision-rejected', phase='failed')
        elif phase == 'open':
            add('partial-write', phase='partial', bs=replace(data, handles[i], i+1))
            add('full-write', phase='written', bs=replace(data, handles[i], i+4))
            add('write-error', phase='failed', handle=-1)
        elif phase == 'partial':
            add('full-write', phase='written', bs=replace(data, handles[i], i+4))
            add('write-error', phase='failed', handle=-1)
        elif phase == 'written':
            add('close', phase='done', handle=-1)
            add('close-error', phase='failed', handle=-1)
        elif phase in ('failed', 'done'):
            add('retry', phase='ready')
    return out


def model(legacy=False):
    queue = deque([INITIAL]); seen = {INITIAL: []}; edges = 0; witness = None
    failures = partial = successes = 0
    while queue:
        state = queue.popleft(); names, data, phases, handles, keys, injected = state
        preserved = names[0] == 0 and data[0] == 7 and (not injected or (names[1] == 3 and data[3] == 7))
        if not preserved:
            if not legacy:
                raise ValueError('champion archive: protected object modified')
            if witness is None:
                witness = seen[state]
        if not legacy:
            for i, handle in enumerate(handles):
                require(handle == -1 or (handle == i+1 and names[keys[i]] == handle), 'unowned write handle')
            require(handles[0] == -1 or handles[1] == -1 or handles[0] != handles[1], 'aliased write ownership')
        failures += 'failed' in phases; partial += 'partial' in phases; successes += 'done' in phases
        for label, target in successors(state, legacy):
            edges += 1
            if target not in seen:
                seen[target] = seen[state] + [label]; queue.append(target)
    require(failures and partial and successes, 'vacuous publication/failure coverage')
    require((witness is not None) == legacy, 'counterexample expectation')
    return {'states': len(seen), 'transitions': edges, 'maxShortestDepth': max(map(len, seen.values())),
            'writers': 2, 'keys': 2, 'objectIdentities': 4, 'opaqueContentClasses': 6,
            'failureStates': failures, 'partialWriteStates': partial, 'completedStates': successes,
            'legacyCounterexample': witness, 'search': 'reachable fixed point; retries unbounded, no retry-depth cutoff',
            'scope': 'stable parent/object identities with leaf insertion; no crash durability, liveness or archive atomicity claim'}


def prepare_fixture(root):
    fixture = json.loads((ROOT/FIXTURE).read_text())
    source = root/'source'; source.mkdir()
    for name, content in fixture['archive'].items():
        (source/name).write_bytes(content.encode())
    require(hashlib.sha256((source/'evidence-index.json').read_bytes()).hexdigest() == fixture['indexSha256'], 'fixture index hash')
    kwargs = dict(rss_unit='bytes', platform_label='synthetic-champion-fixture', expected_index_sha256=fixture['indexSha256'])
    return source, kwargs, {k:v.encode() for k,v in fixture['reports'].items()}


def conformance():
    sys.path.insert(0, str(ROOT/'scripts/research'))
    import summarize_sequential_screen as exporter
    import sequential_learning as learning
    namespace = dict(vars(exporter))
    legacy = json.loads((ROOT/LEGACY).read_text())['exportFunction']
    exec(compile(legacy, '<preserved-legacy-export>', 'exec'), namespace)
    old_export = namespace['export']
    writer = definition(ast.parse((ROOT/source_path('run_sequential_screen')).read_text()), 'write_json')
    writer_ns = {'Path': Path, 'json': json}
    exec(compile(ast.Module(body=[writer], type_ignores=[]), '<actual-write_json>', 'exec'), writer_ns)
    count = 0
    with tempfile.TemporaryDirectory(prefix='trader-champion-') as tmp:
        root = Path(tmp).resolve(); source, kwargs, expected = prepare_fixture(root)
        for name, function in (('legacy-control', old_export), ('control', exporter.export)):
            output = root/name; function(source, output, **kwargs)
            require({p.name:p.read_bytes() for p in output.iterdir()} == expected, 'stable output byte compatibility')
            count += 1
        for legacy_mode, function in ((True, old_export), (False, exporter.export)):
            for kind in ('file', 'symlink', 'hardlink'):
                output = root/(str(legacy_mode)+'-'+kind); champion = root/('champion-'+str(legacy_mode)+'-'+kind)
                sentinel = b'CHAMPION: existing configuration\n'; champion.write_bytes(sentinel)
                original_open = Path.open; injected = []
                def collide(path, mode='r', *args, **kw):
                    if path.parent == output and mode in ('wb', 'xb') and not injected:
                        injected.append(path)
                        if kind == 'symlink': path.symlink_to(champion)
                        elif kind == 'hardlink': path.hardlink_to(champion)
                        else:
                            with original_open(path, 'xb') as stream: stream.write(sentinel)
                    return original_open(path, mode, *args, **kw)
                with patch.object(Path, 'open', collide):
                    if legacy_mode: function(source, output, **kwargs)
                    else:
                        try: function(source, output, **kwargs)
                        except FileExistsError: pass
                        else: raise ValueError('champion archive: collision accepted')
                require(len(injected) == 1, 'collision not scheduled')
                protected = injected[0] if kind == 'file' else champion
                require((protected.read_bytes() != sentinel) == legacy_mode, 'collision preservation/counterexample')
                count += 1
        # Partial newly created output is permitted, but cannot replace old data.
        output = root/'partial'; champion = root/'failure-champion'; champion.write_bytes(b'keep')
        original_open = Path.open; touched = []
        class FailingWrite:
            def __init__(self, stream): self.stream = stream
            def __enter__(self): return self
            def __exit__(self, *args): return self.stream.__exit__(*args)
            def write(self, content):
                self.stream.write(content[:3]); raise OSError('injected write failure')
        def fail_write(path, mode='r', *args, **kw):
            stream = original_open(path, mode, *args, **kw)
            if path.parent == output and mode == 'xb':
                touched.append(path); return FailingWrite(stream)
            return stream
        with patch.object(Path, 'open', fail_write):
            try: exporter.export(source, output, **kwargs)
            except OSError as exc: require(str(exc) == 'injected write failure', 'wrong failure')
            else: raise ValueError('champion archive: failed write hidden')
        require(len(touched) == 1 and touched[0].read_bytes() == expected[touched[0].name][:3], 'partial write fixture')
        before = {p.name:p.read_bytes() for p in output.iterdir()}
        try: exporter.export(source, output, **kwargs)
        except FileExistsError: pass
        else: raise ValueError('champion archive: retry replaced output')
        require(before == {p.name:p.read_bytes() for p in output.iterdir()} and champion.read_bytes() == b'keep', 'failure/retry preservation')
        count += 2
        net = learning.Network(11)
        provenance = {'codeCommit':'a'*40, 'registrationSha256':'b'*64, 'dataSha256':'c'*64,
                      'seed':11, 'horizon':1, 'algorithm':'ppo', 'fold':0}
        for name, function in [('json', lambda p: writer_ns['write_json'](p, {'candidate': 1})),
                               ('policy', lambda p: learning.save_policy(p, net, provenance))]:
            for kind in ('file', 'symlink', 'hardlink'):
                champion = root/(name+'-'+kind+'-target'); champion.write_bytes(b'keep')
                path = champion
                if kind != 'file':
                    path = root/(name+'-'+kind+'-alias')
                    if kind == 'symlink': path.symlink_to(champion)
                    else: path.hardlink_to(champion)
                try: function(path)
                except FileExistsError: pass
                else: raise ValueError('champion archive: writer replaced champion')
                require(champion.read_bytes() == b'keep', 'writer changed champion')
                count += 1
    return {'cases': count, 'compatibleReportFiles': len(expected), 'collisionKinds': ['file','symlink','hardlink'],
            'legacyCounterexamples': 3, 'newPartialOutputPermitted': True, 'trainingRuns': 0, 'marketDataReads': 0,
            'scope': 'actual full exporter and actual writer bodies on deterministic synthetic archives'}


def check_archive(promotion=None):
    surface = extract(); fixed = model(); legacy = model(True)
    counters = json.loads((ROOT/'formal/research/counterexamples.json').read_text())
    counter = next(e for e in counters['entries'] if e['id'] == 'CE-RL-ARCHIVE-COLLISION')
    require(counter['trace'] == legacy['legacyCounterexample'], 'preserved counterexample drift')
    if promotion is None:
        from promotion_boundary import check_promotion
        promotion = check_promotion()
    require(promotion['surface']['moduleCount'] == surface['researchModules'], 'composition coverage')
    expected_roots = ['analyze-close-timing','lstm-bench','merge-top-combos','optimize-equity','outbox-publisher','trader-hs']
    require(promotion['composition']['productionRoots'] == expected_roots, 'production-root composition coverage')
    result = {'surface': surface, 'smt': preserve(all(mode in ('x', 'xb') for _, _, mode in surface['writerSites'])), 'model': fixed, 'legacyModel': legacy,
            'conformance': conformance(), 'composition': {'productionRoots': promotion['composition']['productionRoots'],
            'wholeProductionCorrectness': False, 'filesystemKernelProved': False, 'crashDurability': False}}
    return json.loads(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    print(json.dumps(check_archive(), indent=2))
