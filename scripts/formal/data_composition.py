"""Finite source dataflow composition with source-derived symbolic key checks.

See data-composition-contract.md. Library/runtime semantics are explicit trusted
contracts; this is not a general Python verifier or an empirical trading result.
"""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path

import z3 as z

from causal_footprint import structure
from training_prefix import check_training, function, prove

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = 'formal/research/data-composition-source.json'
FILES = ('sequential_env.py', 'sequential_learning.py',
         'sequential_evaluation.py', 'run_sequential_screen.py')
BLOCK_ROSTER = {
    'run_sequential_screen.py': {'load_development'},
    'sequential_env.py': {'Scale.__post_init__', 'Scale.transform', 'Scale.supported', 'Replay.__init__', 'collect'},
    'sequential_evaluation.py': {'short_ope', 'replay_policy', 'Baselines.__init__'},
    'sequential_learning.py': {'train_ppo', 'train_q'},
}
SHARED_INPUTS = [
    'Scale fitted over registered training prefixes of all declared symbols',
    'Shared policy and baseline parameters fitted over those training prefixes',
]



def require(condition, message):
    if not condition:
        raise ValueError('data composition: ' + message)


def definition(tree, name):
    for part in name.split('.'):
        matches = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == part]
        require(len(matches) == 1, 'missing/duplicate definition ' + name)
        tree = matches[0]
    return tree


def source_uses(file, tree):
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    uses = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Name) and node.id in ('scale', 'Scale') or
                isinstance(node, ast.Attribute) and node.attr == 'scale'):
            continue
        parent, current, scope = parents[node], node, []
        while current in parents:
            current = parents[current]
            if isinstance(current, (ast.FunctionDef, ast.ClassDef)):
                scope.insert(0, current.name)
        if isinstance(parent, ast.FunctionDef) and parent.returns is node:
            context, effect = 'return_annotation', 'annotation'
        elif isinstance(parent, ast.arg):
            context, effect = 'argument_annotation', 'annotation'
        else:
            context = ast.unparse(parent)
            value = ast.unparse(node)
            if isinstance(parent, ast.Attribute):
                allowed = ('fit',) if value == 'Scale' else ('transform', 'supported', 'mean', 'std')
                require(parent.attr in allowed, 'unreviewed scale member access')
                effect = 'fit' if value == 'Scale' else 'read'
            elif isinstance(parent, ast.Call):
                callee = ast.unparse(parent.func)
                positions = {'Replay': 5, 'collect': 2, 'Baselines': 2, 'train_ppo': 2,
                             'train_q': 2, 'short_ope': 2, 'replay_policy': 5, 'getattr': 0}
                require(callee in positions and len(parent.args) > positions[callee] and
                        parent.args[positions[callee]] is node, 'unreviewed scale escape/call binding')
                effect = 'serialize' if callee == 'getattr' else 'forward:' + callee
            elif isinstance(parent, ast.Tuple):
                assignment = parents[parent]
                require(isinstance(assignment, ast.Assign) and scope[-1] == '__init__' and
                        scope[0] in ('Replay', 'Baselines'), 'unreviewed scale alias')
                expected = ('self.horizon, self.scale, self.execution = horizon, scale, execution'
                            if scope[0] == 'Replay' else 'self.scale, self.horizon = scale, horizon')
                require(structure(assignment) == structure(ast.parse(expected).body[0]), 'scale owner binding drift')
                effect = 'owner_binding'
            elif isinstance(parent, ast.Assign):
                require(scope == ['run'] and structure(parent) == structure(
                    ast.parse('scale = Scale.fit(list(train.values()))').body[0]), 'scale rebind/write')
                effect = 'fit_binding'
            else:
                raise ValueError('data composition: unsupported scale use ' + context)
        uses.append(({'file': file, 'scope': '.'.join(scope), 'value': ast.unparse(node),
                      'context': context}, effect))
    return uses


def expression(text):
    return ast.parse(text, mode='eval').body


def assignments(node, target):
    return [n for n in ast.walk(node) if isinstance(n, ast.Assign) and
            any(ast.unparse(t) == target for t in n.targets)]


def one_assignment(node, target):
    values = assignments(node, target)
    require(len(values) == 1, 'ambiguous binding ' + target)
    return values[0].value


def key(node, symbols):
    if isinstance(node, ast.Name) and node.id in symbols:
        return symbols[node.id]
    if isinstance(node, ast.Constant) and type(node.value) is str:
        return z.StringVal(node.value)
    raise ValueError('data composition: unsupported symbol expression ' + ast.unparse(node))


def dictionary_key(node, dictionary, symbols):
    require(isinstance(node, ast.Subscript) and ast.unparse(node.value) == dictionary,
            'unreviewed market dictionary')
    return key(node.slice, symbols)


def selection_key(node, table, symbols):
    require(isinstance(node, ast.Subscript) and ast.unparse(node.value) == table,
            'unreviewed panel selection')
    part = node.slice
    require(isinstance(part, ast.Compare) and len(part.ops) == 1 and isinstance(part.ops[0], ast.Eq)
            and ast.unparse(part.left) == table + '.symbol', 'panel must filter exact symbol equality')
    return key(part.comparators[0], symbols)


def check_keys(trees):
    """Prove actual selected keys agree, for arbitrary strings (no finite universe assumption)."""
    name = z.String('declared_symbol')
    symbols = {'symbol': name, 'sym': name, 's': name}
    loader = definition(trees['run_sequential_screen.py'], 'load_development')
    checks = []
    for local, table in (('rows', 'bars'), ('es', 'events')):
        selected = selection_key(one_assignment(loader, local), table, symbols)
        checks.append((local + '_selection', selected == name))
    output = one_assignment(loader, '(prices[symbol], funding[symbol])')
    require(ast.unparse(output) == '(p, f)', 'loader output values do not match selected arrays')
    # Every construction edge binds the same local price and funding symbol.
    collect = definition(trees['sequential_env.py'], 'collect')
    selected_price = dictionary_key(one_assignment(collect, 'p'), 'prices', symbols)
    calls = [n for n in ast.walk(collect) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'Replay']
    require(len(calls) == 1 and ast.unparse(calls[0].args[0]) == 'p', 'collector price binding')
    checks.append(('collection_pair', z.And(selected_price == name,
                  dictionary_key(calls[0].args[1], 'funding', symbols) == name)))
    for owner, callee, count in (('short_ope', 'Replay', 2), ('run', 'replay_policy', 1)):
        file = 'run_sequential_screen.py' if owner == 'run' else 'sequential_evaluation.py'
        calls = [n for n in ast.walk(definition(trees[file], owner))
                 if isinstance(n, ast.Call) and ast.unparse(n.func) == callee]
        require(len(calls) == count, 'market consumer call roster drift')
        for index, call in enumerate(calls):
            checks.append((owner + str(index), z.And(dictionary_key(call.args[0], 'prices', symbols) == name,
                           dictionary_key(call.args[1], 'funding', symbols) == name)))
    # Forwarding single arrays does not introduce a second symbol-local market source.
    forward = definition(trees['sequential_evaluation.py'], 'replay_policy')
    constructor = one_assignment(forward, 'env')
    require(ast.unparse(constructor.func) == 'Replay' and
            [ast.unparse(x) for x in constructor.args[:2]] == ['p', 'funding'], 'replay pair forwarding')
    initial = definition(trees['sequential_env.py'], 'Replay.__init__')
    require(ast.unparse(one_assignment(initial, '(self.prices, self.funding)')) == '(prices, funding)',
            'replay pair storage')
    for label, claim in checks:
        prove('F-RL-DATA-KEYS/' + label, z.BoolVal(True), claim)
    return {'requirement': 'F-RL-DATA-KEYS', 'result': 'unsat',
            'keyChecks': [label for label, _ in checks], 'premiseChecks': len(checks),
            'violationChecks': len(checks), 'domain': 'arbitrary mathematical symbol strings'}


def check_snapshot(tree):
    scale = definition(tree, 'Scale')
    require([ast.unparse(d) for d in scale.decorator_list] == ['dataclass(frozen=True)'], 'scale is mutable')
    constructor = definition(tree, 'Scale.__post_init__')
    snapshot = one_assignment(constructor, 'snapshot')
    require(structure(snapshot) == structure(expression(
        'np.frombuffer(np.asarray(value, dtype=float).tobytes(), dtype=float)')), 'scale bytes ownership drift')
    loops = [n for n in constructor.body if isinstance(n, ast.For)]
    require(len(loops) == 2 and ast.unparse(loops[0].iter) == "('mean', 'std', 'low', 'high')",
            'scale field coverage')
    require(ast.unparse(loops[1].iter) == 'snapshots.items()', 'scale snapshot publication drift')
    require(ast.unparse(loops[1].body[0]) == 'object.__setattr__(self, name, value)' and len(loops[1].body) == 1,
            'scale snapshot publication not direct')
    return {'fields': ['mean', 'std', 'low', 'high'], 'backing': 'immutable bytes',
            'librarySemantics': 'A-DATA-COMPOSITION', 'frozenOwner': True}


def check_composition(sources=None, registry=None):
    registry = json.loads((ROOT / REGISTRY).read_text()) if registry is None else registry
    require(type(registry.get('schemaVersion')) is int and registry['schemaVersion'] == 1, 'unsupported source contract version')
    require({file: set(blocks) for file, blocks in registry.get('blocks', {}).items()} == BLOCK_ROSTER,
            'mandatory source skeleton coverage missing or changed')
    require(set(registry.get('sourceHashes', {})) == {'scripts/research/' + f for f in FILES},
            'mandatory source hash inventory missing or changed')
    require(registry.get('sharedInputs') == SHARED_INPUTS, 'explicit shared input contract drift')
    sources = ({f: (ROOT / 'scripts/research' / f).read_text() for f in FILES}
               if sources is None else sources)
    require(set(sources) == set(FILES), 'incomplete source inventory')
    trees = {file: ast.parse(source) for file, source in sources.items()}
    uses = [use for file in FILES for use in source_uses(file, trees[file])]
    require([use for use, _ in uses] == registry['uses'], 'scale use/dispatch inventory drift')
    keys = check_keys(trees)
    snapshot = check_snapshot(trees['sequential_env.py'])
    for file, blocks in registry['blocks'].items():
        for name, reviewed in blocks.items():
            require(structure(definition(trees[file], name)) == structure(ast.parse(reviewed).body[0]),
                    'reviewed primitive/dataflow skeleton drift: ' + name)
    # Full source locks cover unmodified helper bodies and foreign effects. They
    # complement, rather than substitute for, the semantic checks above.
    for file, source in sources.items():
        require(hashlib.sha256(source.encode()).hexdigest() == registry['sourceHashes']['scripts/research/' + file],
                'unreviewed helper/source drift: ' + file)
    registration = json.loads((ROOT / 'research-notes/registrations/sequential-control-screen-v1.json').read_text())
    prefixes = check_training(registration, sources['sequential_env.py'], sources['run_sequential_screen.py'])
    return {'requirement': 'F-RL-DATA-COMPOSITION', 'status': 'exhaustively_checked',
            'sourceFiles': list(FILES), 'scaleUseSites': len(uses),
            'useEffects': dict(sorted(Counter(effect for _, effect in uses).items())),
            'snapshot': snapshot, 'keys': keys, 'prefixRequirements': list(prefixes['smt']),
            'sharedInputs': registry['sharedInputs'],
            'smt': {'F-RL-DATA-KEYS': 'unsat'},
            'universalInterpreterRefinement': False, 'publicationWitnessesVerified': False}


if __name__ == '__main__':
    print(json.dumps(check_composition(), indent=2))
