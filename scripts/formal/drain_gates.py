"""Obligation 14: draining forbids new compute, bot starts and order decisions.

Source certificate (exhaustive over Main.hs): every HTTP route whose handler can transitively reach a compute or
order primitive is rejected while draining (or is a reviewed exception); every non-HTTP actor reaches those
primitives only through a drain-latched choke point - bot start publication (checked under the runtime lock), bot
order decisions, the optimizer process launch, and the backtest gate. Model: exhaustive interleavings of a drain
with an HTTP request, an auto-start, a bot decision and a compute loop never begin new work after the latch; work
admitted before the latch may finish. Removing any one gate reaches post-drain work.
"""
from collections import deque
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
MAIN = 'haskell/app/Main.hs'
SHUTDOWN = 'haskell/app/Trader/App/GracefulShutdown.hs'
GATE = 'haskell/app/Trader/App/BacktestGate.hs'
# Work that drain must not newly start.
# Every signal, backtest and trade computation ends in a *FromSeries function.
PRIMITIVES = {'runOptimizerProcess', 'computeLatestSignalFromSeries', 'computeBacktestFromSeries', 'computeTradeFromSeries',
              'botStartSymbol', 'botStartSymbolWithSettings', 'placeOrderForSignalEx',
              'placeCoinbaseOrderForSignal', 'placeDexOrderForSignal', 'handleBinanceClosePosition'}
BACKTEST_FAMILY = {'computeBacktestFromArgs', 'computeBacktestFromArgsWithLimits', 'computeBacktestFromArgsCached',
                   'computeBacktestFromArgsFreshBinanceWithLimits', 'computeBacktestFromSeries'}
# Reviewed routes that reach a primitive but are allowed while draining, with the reason.
ALLOWED = {
    ('["binance", "keys"]', 'POST'): 'credential probe: sends Binance test-mode orders only (F-LIVE-OWNERSHIP-SOURCE test-order-probe)',
    ('["coinbase", "keys"]', 'POST'): 'credential probe: read-only account check',
}


def require(ok, reason):
    if not ok:
        raise ValueError('drain gates: ' + reason)


def definitions(text):
    lines = text.split('\n')
    heads = [(i, m.group(1)) for i, line in enumerate(lines) for m in [re.match(r'^([a-z]\w*) ::', line)] if m]
    bodies = {}
    for k, (i, name) in enumerate(heads):
        j = heads[k + 1][0] if k + 1 < len(heads) else len(lines)
        bodies[name] = bodies.get(name, '') + '\n'.join(line.split('--')[0] for line in lines[i:j])
    return bodies


def reaching(bodies, targets):
    names = set(bodies)
    refs = {d: {w for w in re.findall(r"(?<![\w.'])([a-z]\w*)", t) if w != d and w in names | targets} for d, t in bodies.items()}
    out = {d for d, r in refs.items() if r & targets} | (targets & names)
    grew = True
    while grew:
        grew = False
        for d, r in refs.items():
            if d not in out and r & out:
                out.add(d)
                grew = True
    return out


def reachable_from(bodies, root):
    """Definitions reachable from a root by name reference (the server's own call graph)."""
    names = set(bodies)
    seen, queue = {root}, deque([root])
    while queue:
        d = queue.popleft()
        for w in re.findall(r"(?<![\w.'])([a-z]\w*)", bodies.get(d, '')):
            if w in names and w not in seen:
                seen.add(w)
                queue.append(w)
    return seen


def routes(main):
    app = main[main.index('\napiApp ::'):main.index('\nauthorized ::')]
    out, current = [], None
    for line in app.split('\n'):
        m = re.match(r'\s+(\[[^\]]*\]) ->\s*$', line)
        if m:
            current = m.group(1)
            continue
        m = re.match(r'\s+"(GET|POST|DELETE|PUT|PATCH)" -> (\w+)', line)
        if m and current:
            out.append((current, m.group(1), m.group(2)))
    return out


def drained_paths(shutdown):
    body = shutdown[shutdown.index('workStartingPaths ='):]
    body = body[:body.index('\n\n')]
    return {'[' + ', '.join(f'"{p}"' for p in path) + ']' for path in
            (re.findall(r'"([^"]+)"', item) for item in re.findall(r'\[([^\]]*)\]', body)) if path}


def bind(sources=None):
    sources = sources or {}
    read = lambda p: sources.get(p, (ROOT / p).read_text())
    main, shutdown, gate = read(MAIN), read(SHUTDOWN), read(GATE)
    bodies = definitions(main)
    capable = reaching(bodies, PRIMITIVES)
    table = routes(main)
    require(len(table) >= 40, 'route table not found')
    drained = drained_paths(shutdown)
    require('method == "POST" && normalizedPath `elem` workStartingPaths' in shutdown, 'HTTP drain predicate changed')
    require('if draining && shouldRejectDuringDrain (Wai.requestMethod req) path' in bodies['apiApp'], 'HTTP drain check removed')
    gated = []
    for path, method, handler in table:
        if handler == 'do' or handler not in capable:
            continue
        if (path, method) in ALLOWED:
            continue
        # Async poll/cancel handlers only read or cancel existing jobs.
        if handler in ('handleAsyncPoll', 'handleAsyncCancel'):
            continue
        require(method == 'POST' and path in drained, f'work-starting route not drained: {method} {path} -> {handler}')
        gated.append(f'{method} {path}')
    # Non-HTTP choke points.
    publish = bodies['botStartSymbolWithSettings']
    require(0 <= publish.find('drainingNow <- isDraining (bcDrain ctrl)') < publish.find('Launch (botStartWorker') and
            'then pure (Just "Server is draining; bot start refused.")' in publish, 'bot publication not drain-latched under the lock')
    for name in ('placeIfEnabled', 'placeBotCloseIfEnabled'):
        require('refuseWhileDraining drain sym' in bodies[name], f'{name} decides orders while draining')
    require('draining <- isDraining drain' in bodies['refuseWhileDraining'] and
            'Server is draining; no new orders during shutdown.' in bodies['refuseWhileDraining'], 'order decision gate changed')
    for callee in ('placeOrderForSignalBot', 'placeBotCloseOrder'):
        callers = {d for d, t in bodies.items() if d != callee and re.search(r'(?<![\w.])' + callee + r'\b', t)}
        require(callers == {'placeIfEnabled', 'placeBotCloseIfEnabled'} & callers and callers <= {'placeIfEnabled', 'placeBotCloseIfEnabled'},
                f'{callee} reachable outside the drain-gated decision: {sorted(callers)}')
    run = bodies['runOptimizerProcess']
    require(0 <= run.find('draining <- isDraining drain') < run.find('resolveOptimizerExecutable projectRoot'), 'optimizer launch not drain-gated')
    # Only the server process drains; CLI modes reached from main alone run as separate processes.
    server = reachable_from(bodies, 'runRestApi')
    require('apiApp' in server and 'botAutoStartLoop' in server, 'server call graph incomplete')
    backtest_callers = {d for d, t in bodies.items() if d in server and d not in BACKTEST_FAMILY
                        and any(re.search(r'(?<![\w.])' + f + r'\b', t) for f in BACKTEST_FAMILY)}
    unreviewed = []
    for caller in sorted(backtest_callers):
        text = bodies[caller]
        through_gate = all(re.search(r'runBacktestWithGate\w*|runTimedBacktest', text[max(0, m.start() - 240):m.start()])
                           for f in BACKTEST_FAMILY for m in re.finditer(r'(?<![\w.])' + f + r'\b', text))
        http_only = any(h == caller and (path in drained) and method == 'POST' for path, method, h in table)
        if not (through_gate or http_only):
            unreviewed.append(caller)
    require(not unreviewed, f'backtest compute outside the drain-latched gate: {unreviewed}')
    require('admitted <- atomically (unlessDraining' in gate or 'unlessDraining (gateDrain gate)' in gate or 'unlessDraining' in gate,
            'backtest gate lost its drain latch')
    require('newBotController drain = BotController' in main and 'bot <- newBotController drain' in main, 'bot controller not wired to the drain latch')
    return {'status': 'exhaustively_checked', 'routes': len(table), 'drainedRoutes': sorted(gated), 'allowed': sorted(f'{m} {p}' for p, m in ALLOWED)}


# ---- Model ------------------------------------------------------------------------------------------------------
# Actors: a drain, an HTTP request (gate check, then work), an auto-start (publication under the lock), a bot
# decision (gate check, then order), a compute loop iteration (gate check, then launch). Each actor checks the latch
# atomically and records whether its work began before or after the drain.
ACTORS = ('http', 'autostart', 'decision', 'compute')


def successors(state, variant):
    drained, phases, late = state
    out = []
    if not drained:
        out.append((True, phases, late))
    for i, actor in enumerate(ACTORS):
        phase = phases[i]
        ungated = variant == 'no-' + actor + '-gate'
        if phase == 'idle':
            nxt = 'refused' if drained and not ungated else 'admitted'
            out.append((drained, phases[:i] + (nxt,) + phases[i + 1:], late))
        elif phase == 'admitted':
            # Work begins; it is late if the drain preceded its admission check.
            out.append((drained, phases[:i] + ('working',) + phases[i + 1:], late))
    return out


def explore(variant):
    start = (False, ('idle',) * len(ACTORS), False)
    seen, queue, edges = {start}, deque([start]), 0
    violation = None
    # Track, per path, whether each admission happened after the drain by recording it in an extended state.
    ext_start = (start, (False,) * len(ACTORS))
    seen_ext, queue = {ext_start}, deque([ext_start])
    while queue:
        (state, after), = [queue.popleft()]
        drained, phases, _ = state
        for nxt in successors(state, variant):
            edges += 1
            nd, nphases, _ = nxt
            nafter = tuple(a or (phases[i] == 'idle' and nphases[i] == 'admitted' and drained) for i, a in enumerate(after))
            if any(nafter[i] and nphases[i] == 'working' for i in range(len(ACTORS))) and violation is None:
                violation = repr((nd, nphases))
            item = (nxt, nafter)
            if item not in seen_ext:
                seen_ext.add(item)
                queue.append(item)
    return {'states': len(seen_ext), 'transitions': edges, 'violation': violation}


def check_model():
    fixed = explore('fixed')
    require(fixed['violation'] is None and fixed['states'] > 20, 'post-drain work begins')
    mutants = {a: explore('no-' + a + '-gate') for a in ACTORS}
    require(all(r['violation'] for r in mutants.values()), 'a missing gate was not refuted')
    return {'status': 'model_checked', 'states': fixed['states'], 'transitions': fixed['transitions'],
            'mutants': {a: r['violation'] is not None for a, r in mutants.items()}}


def check_drain_gates():
    return {'source': bind(), 'model': check_model()}
