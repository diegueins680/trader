"""Source-derived research isolation; trusted compiler/runtime, not an OS sandbox."""
import ast
from collections import deque
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

import z3 as z

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = 'formal/research/capability-source-contract.json'
PROPOSAL = 'haskell/app/Trader/Research/PolicyProposalV1.hs'
LEARNING = 'scripts/research/sequential_learning.py'
HELPER = 'formal/research/ComponentGraph.hs'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def parse_receipt(output):
    modules, roots, declarations, exports = {}, {}, {}, {}
    for line in output.splitlines():
        fields = line.split('\t')
        if fields[0] == 'M' and len(fields) == 4:
            _, path, name, imports = fields
            require(path not in modules, 'duplicate source module')
            modules[path] = {'name': name, 'imports': imports.split(',') if imports else []}
        elif fields[0] == 'R' and len(fields) == 4:
            _, name, path, declared = fields
            require(name not in roots, 'duplicate executable')
            roots[name] = {'path': path, 'declared': declared.split(',') if declared else []}
        elif fields[0] == 'D' and len(fields) == 3:
            declarations.setdefault(fields[1], []).append(ast.literal_eval(fields[2]))
        elif fields[0] == 'E' and len(fields) == 3:
            require(fields[1] not in exports, 'duplicate export surface')
            exports[fields[1]] = ast.literal_eval(fields[2])
        else:
            raise ValueError('unknown GHC adapter output')
    return {'modules': modules, 'roots': roots, 'declarations': declarations, 'exports': exports}


def parser_output(root=ROOT, executable=None, files=None):
    paths = files if files is not None else sorted(str(p.relative_to(root)) for p in (root / 'haskell/app').rglob('*.hs'))
    require(paths and all(not Path(p).is_absolute() and '..' not in Path(p).parts for p in paths), 'unsafe source inventory')
    require(all(not (root / p).is_symlink() for p in paths), 'symlinked source is outside contract')
    libdir = subprocess.check_output(['ghc', '--print-libdir'], text=True).strip()
    if executable is not None:
        return subprocess.check_output([executable, libdir, 'haskell/trader.cabal', *paths], cwd=root, text=True, timeout=90)
    with tempfile.TemporaryDirectory(prefix='trader-component-') as directory:
        exe = str(Path(directory) / 'components')
        subprocess.run(['ghc', '-v0', '-O0', '-package', 'ghc-9.4.8', '-package', 'Cabal-3.8.1.0',
                        '-outputdir', directory, HELPER, '-o', exe], cwd=root, check=True, timeout=90,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return subprocess.check_output([exe, libdir, 'haskell/trader.cabal', *paths], cwd=root, text=True, timeout=90)


def graph_certificate(parsed, expected_roots):
    modules, roots = parsed['modules'], parsed['roots']
    require(sorted(roots) == sorted(expected_roots), 'executable inventory drift')
    local = {}
    for path, item in modules.items():
        if item['name'] != 'Main':
            require(item['name'] not in local, 'duplicate local module name')
            local[item['name']] = path
    require(local.get('Trader.Research.PolicyProposalV1') == PROPOSAL, 'research module identity drift')
    graph = {}
    for path, item in modules.items():
        edges = []
        for name in item['imports']:
            require(name != 'Main', 'ambiguous import of executable Main')
            require(not name.startswith('Trader.') or name in local, 'missing local imported module: ' + name)
            if name in local:
                edges.append(local[name])
        graph[path] = sorted(set(edges))
    require(not modules[PROPOSAL]['imports'], 'research proposal imports an effect dependency')
    summaries = {}
    for name, root in roots.items():
        require(root['path'] in graph, 'missing executable source')
        require('Trader.Research.PolicyProposalV1' not in root['declared'], 'research module declared in executable')
        queue = deque([root['path']]); parents = {root['path']: None}; edges = 0; depth = {root['path']: 0}
        while queue:
            path = queue.popleft()
            if path == PROPOSAL:
                trace = []
                while path is not None:
                    trace.append(path); path = parents[path]
                raise ValueError('research import reachable: ' + ' -> '.join(reversed(trace)))
            for target in graph[path]:
                edges += 1
                if target not in parents:
                    parents[target] = path; depth[target] = depth[path] + 1; queue.append(target)
        summaries[name] = {'states': len(parents), 'edges': edges, 'maxShortestDepth': max(depth.values()),
                           'reachableFiles': sorted(parents)}
    return {'requirement': 'F-RL-COMPONENT-ISOLATION', 'search': 'reachable_fixed_point',
            'moduleFiles': len(modules), 'localEdges': sum(map(len, graph.values())), 'roots': summaries}


def functions(source):
    tree = ast.parse(source)
    result = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    network = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Network')
    result.update({'Network.' + n.name: n for n in network.body if isinstance(n, ast.FunctionDef)})
    return result


def effect_certificate(source, parsed, registry):
    fns = functions(source)
    allowed = {'infer': {'time.perf_counter_ns', '_finite_real_vector', 'net.forward', 'float', 'int', 'np.argmax'},
               'Network.forward': {'np.errstate', 'np.isfinite', 'array.all', 'np.tanh', 'ValueError'},
               '_finite_real_vector': {'isinstance', 'np.ma.isMaskedArray', 'bool', 'np.isfinite', 'array.all'}}
    calls = {}
    for name, allowed_calls in allowed.items():
        node = fns[name]
        require(not node.decorator_list, 'effectful inference decorator')
        found = []
        for item in ast.walk(node):
            require(not isinstance(item, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.AugAssign,
                                          ast.Delete, ast.Yield, ast.YieldFrom, ast.Await, ast.Lambda)), 'unknown inference effect')
            if isinstance(item, ast.Assign):
                require(all(isinstance(t, ast.Name) for t in item.targets), 'policy state write')
            if isinstance(item, ast.AnnAssign):
                require(isinstance(item.target, ast.Name), 'annotated policy state write')
            if isinstance(item, ast.Call):
                label = ast.unparse(item.func)
                if isinstance(item.func, ast.Attribute) and item.func.attr == 'all' and isinstance(item.func.value, ast.Call) and ast.unparse(item.func.value.func) == 'np.isfinite':
                    label = 'array.all'
                require(label in allowed_calls, 'unreviewed inference call: ' + label)
                permitted_keywords = {'over','invalid','divide'} if label == 'np.errstate' else set()
                require(all(k.arg in permitted_keywords for k in item.keywords), 'mutable or unreviewed call option')
                found.append(label)
        # Complete reviewed AST semantics, not just a call-name denylist.
        require(sha(ast.dump(node, include_attributes=False)) == registry['inferenceAST'][name], 'inference semantic drift')
        calls[name] = sorted(set(found))
    declarations = parsed['declarations'][PROPOSAL]
    require([sha(d) for d in declarations] == registry['proposalDeclarations'], 'proposal declaration drift')
    require(parsed['exports'][PROPOSAL] == registry['proposalExports'], 'proposal export drift')
    require('orderAuthorized _ = False' in declarations, 'proposal gained authority')
    require('newtype ResearchProposal\n  = ResearchProposal Double\n  deriving (Eq, Show)' in declarations, 'proposal capability representation changed')
    return {'requirement': 'F-RL-POLICY-EFFECTS', 'functions': calls, 'parameterWrites': 0,
            'orderOperations': 0, 'scope': 'ordinary admitted Network/base arrays; trusted primitive effects; no OS sandbox'}


def artifact_keys(source, parsed, registry):
    save = functions(source)['save_policy']
    assignments = [n for n in save.body if isinstance(n, ast.Assign) and any(isinstance(t,ast.Name) and t.id == 'value' for t in n.targets)]
    require(len(assignments) == 1 and isinstance(assignments[0].value, ast.Dict), 'policy object construction drift')
    keys = assignments[0].value.keys
    require(all(isinstance(k, ast.Constant) and isinstance(k.value, str) for k in keys), 'dynamic artifact keys')
    emitted = [k.value for k in keys]
    require(len(set(emitted)) == len(emitted), 'duplicate emitted artifact key')
    require(sha(ast.dump(save, include_attributes=False)) == registry['savePolicyAST'], 'artifact writer drift')
    ds = parsed['declarations']['haskell/app/Main.hs']
    require([sha(d) for d in ds] == registry['nativeDeclarations'], 'native decoder semantics drift')
    definition = next(d for d in ds if d.startswith('data PersistedLstmModel'))
    fields = re.findall(r'\b(plm[A-Za-z]+)\s*::\s*!', definition)
    require(len(fields) == 4 and definition.count('::') == 4 and 'Maybe' not in definition, 'unsupported native fields')
    require('instance FromJSON PersistedLstmModel where\n  parseJSON = Aeson.genericParseJSON (jsonOptions 3)' in ds, 'native decoder not generic required-field decoder')
    required = [f[3].lower() + f[4:] for f in fields]
    return sorted(emitted), sorted(required)


def prove_disjoint(emitted, required):
    require(emitted and required, 'empty schema domain')
    universe = sorted(set(emitted) | set(required))
    present = {name: z.Bool('key_' + name) for name in universe}
    premise = z.And(*[present[name] == (name in emitted) for name in universe])
    native_accepts = z.And(*[present[name] for name in required])
    for formula, expected in ((premise, z.sat), (z.And(premise, native_accepts), z.unsat)):
        solver = z.Solver(); solver.set(timeout=10000, random_seed=0); solver.add(formula)
        require(solver.check() == expected, 'schema proof unexpected SAT/UNKNOWN/timeout')
    return {'requirement': 'F-RL-ARTIFACT-DISJOINT', 'result': 'unsat', 'emittedKeys': emitted,
            'requiredNativeKeys': required, 'membershipBits': len(universe), 'premiseChecks': 1, 'violationChecks': 1,
            'scope': 'emitted offline JSON versus unchanged native LSTM decoder; trusted Aeson semantics'}


def check_types(root=ROOT):
    probes = {
        'constructor': 'forged :: ResearchProposal\nforged = ResearchProposal 0.25',
        'coercion': 'forged :: ResearchProposal\nforged = coerce (0.25 :: Double)',
        'effect': 'forged :: ResearchProposal -> IO ()\nforged = proposalTarget',
    }
    with tempfile.TemporaryDirectory(prefix='trader-capability-types-') as directory:
        for name, body in probes.items():
            path = Path(directory) / 'Forgery.hs'
            path.write_text('module Forgery where\nimport Trader.Research.PolicyProposalV1\nimport Data.Coerce (coerce)\n' + body + '\n')
            result = subprocess.run(['ghc','-v0','-fno-code','-ihaskell/app','-outputdir',directory,str(path)],cwd=root,text=True,capture_output=True,timeout=60)
            require(result.returncode != 0 and ('not in scope' in result.stderr or "Couldn't match" in result.stderr or 'Illegal term-level' in result.stderr), 'unexpected type-forgery admission/error: ' + name)
    return {'negativeCompilationCases': sorted(probes), 'allRejected': True}


def check_isolation(root=ROOT):
    registry = json.loads((root / REGISTRY).read_text())
    inventory = sorted(str(p.relative_to(root)) for p in (root / 'haskell/app').rglob('*.hs'))
    require(inventory == registry['sourceInventory'], 'application source inventory drift')
    for path, expected in registry['supportSourceHashes'].items():
        require(sha((root / path).read_text()) == expected, 'reviewed external boundary drift: ' + path)
    parsed = parse_receipt(parser_output(root))
    require(sorted(parsed['modules']) == inventory, 'parser omitted source')
    source = (root / LEARNING).read_text()
    graph = graph_certificate(parsed, registry['executables'])
    effects = effect_certificate(source, parsed, registry)
    emitted, required = artifact_keys(source, parsed, registry)
    schema = prove_disjoint(emitted, required)
    return {'graph': graph, 'effects': effects, 'schema': schema, 'typeConformance': check_types(root),
            'smt': {'F-RL-ARTIFACT-DISJOINT': 'unsat'},
            'sourceHashes': {p: sha((root / p).read_text()) for p in inventory},
            'packaging': registry['packagingReview'],
            'legacyCompilerProfile': 'Dockerfile.optimized names GHC 8.10.4; not certified buildable',
            'productionAuthorizationVerified': False}
