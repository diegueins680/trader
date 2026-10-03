"""Source-derived finite default-control proof; not a compiler correctness proof."""
import ast
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
HASKELL = 'haskell/app/Trader/Research/PolicyProposalV1.hs'
ENTRY_POINTS = (
    ('scripts/research/sequential_learning.py', 'infer'),
    ('scripts/research/gae_targets_v2.py', 'step_v2'),
    ('scripts/research/gae_targets_v2.py', 'batch_v2'),
    ('scripts/research/ess_rational_v2.py', 'effective_sample_size_v2'),
)


def require(value, message):
    if not value:
        raise ValueError(message)


def shape(node):
    return ast.dump(node, include_attributes=False)


def expression(text):
    return shape(ast.parse(text, mode='eval').body)


def check_python(source, name):
    functions = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name]
    require(len(functions) == 1, 'missing/duplicate default entry')
    fn = functions[0]
    require(not fn.decorator_list, 'decorated default entry')
    args = fn.args
    require(not args.vararg and not args.kwarg, 'unsupported argument forwarding')
    defaults = dict(zip((a.arg for a in args.kwonlyargs), args.kw_defaults))
    require('enabled' in defaults and isinstance(defaults['enabled'], ast.Constant) and defaults['enabled'].value is False, 'default must be singleton False')
    # Python defaults run at definition time. Admit only constants or version names.
    require(all(isinstance(n, ast.Constant) or isinstance(n, ast.Name) and n.id == 'VERSION'
                for n in [*args.defaults, *args.kw_defaults] if n is not None), 'effectful default')
    body = list(fn.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        body.pop(0)
    if name == 'infer':
        require(shape(body.pop(0)) == shape(ast.parse('start = time.perf_counter_ns()').body[0]), 'inference pre-guard effects')
    require(body and isinstance(body[0], ast.If), 'guard is not first operation')
    gate = body[0]
    require(isinstance(gate.test, ast.BoolOp) and isinstance(gate.test.op, ast.Or)
            and shape(gate.test.values[0]) == expression('enabled is not True'), 'unsafe short-circuit prefix')
    wanted = 'return None, (time.perf_counter_ns() - start) / 1e6' if name == 'infer' else 'return None'
    require(not gate.orelse and len(gate.body) == 1 and shape(gate.body[0]) == shape(ast.parse(wanted).body[0]), 'default branch is not absent')
    # Exact abstraction of the admitted identity test. Disabled RHS is unreachable.
    cases = []
    for enabled in (False, True):
        rejected_by_prefix = enabled is not True
        require(rejected_by_prefix == (not enabled), 'default abstraction mismatch')
        cases.append({'enabled': enabled, 'rejectsBeforeOtherInputs': rejected_by_prefix})
    return {'entry': name, 'cases': cases}


def check_haskell(source):
    # A deliberately restricted declaration grammar; changes require review.
    require('{-#' not in source, 'unreviewed Haskell pragmas')
    clean = re.sub(r'\{-.*?-\}', '', source, flags=re.S)
    clean = re.sub(r'--[^\n]*', '', clean)
    require(not re.search(r'^\s*(import|instance)\b|\(/=\)|\bmode\s*=', clean, re.M), 'unreviewed Haskell name semantics')
    require(len(re.findall(r'^defaultResearchMode\s*=', clean, re.M)) == 1, 'ambiguous default')
    require(re.search(r'^defaultResearchMode = Disabled\s*$', clean, re.M), 'Haskell default changed')
    require(re.search(r'^data ResearchMode = Disabled \| OfflineReplayV1\s*\n', clean, re.M), 'mode domain changed')
    require(len(re.findall(r'^screenProposal\s+mode\s+evidence\s+target\s*$', clean, re.M)) == 1, 'ambiguous Haskell entry')
    require(re.search(r'^screenProposal mode evidence target\s*\n\s*\| mode /= OfflineReplayV1 = Nothing\s*\n', clean, re.M), 'Haskell initial guard changed')
    require(len(re.findall(r'^screenProposal\b(?!\s*::)', clean, re.M)) == 1, 'extra Haskell equation')
    modes = ('Disabled', 'OfflineReplayV1')
    cases = [{'mode': mode, 'initialGuardRejects': mode != 'OfflineReplayV1'} for mode in modes]
    require(cases[0]['initialGuardRejects'] and not cases[1]['initialGuardRejects'], 'Haskell default path mismatch')
    return {'modes': len(cases), 'cases': cases, 'default': 'Disabled', 'defaultResult': 'Nothing'}


def check_saved_default(source):
    fn = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'save_policy')
    assignments = [n for n in fn.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'value' for t in n.targets)]
    require(len(assignments) == 1 and isinstance(assignments[0].value, ast.Dict), 'artifact mapping drift')
    fields = assignments[0].value
    keys = [k.value if isinstance(k, ast.Constant) else None for k in fields.keys]
    require(keys.count('enabled') == 1, 'ambiguous artifact enabled flag')
    require(shape(fields.values[keys.index('enabled')]) == expression('False'), 'artifact default enabled')
    tail = fn.body[fn.body.index(assignments[0]) + 1:]
    expected = ast.parse("""raw = (json.dumps(value, sort_keys=True, allow_nan=False) + "\\n").encode()
if len(raw) > 65536:
    raise ValueError("artifact too large")
with path.open("xb") as stream:
    stream.write(raw)
return hashlib.sha256(raw).hexdigest()
""").body
    require([shape(n) for n in tail] == [shape(n) for n in expected], 'artifact publication tail drift')
    # Existing artifact-admission source checks and metadata SMT cover decoding.
    return {'savedEnabled': False}


def check_defaults(sources=None):
    if sources is None:
        sources = {p: (ROOT / p).read_text() for p in {HASKELL, *(p for p, _ in ENTRY_POINTS)}}
    return {'requirement': 'F-RL-DEFAULT-PATH', 'status': 'exhaustively_checked',
            'python': [check_python(sources[p], name) for p, name in ENTRY_POINTS],
            'haskell': check_haskell(sources[HASKELL]),
            'artifact': check_saved_default(sources[ENTRY_POINTS[0][0]]),
            'booleanCases': 8, 'haskellModes': 2}
