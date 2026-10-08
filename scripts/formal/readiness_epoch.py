"""Obligation 16: live readiness cannot outlive the reconciliation it reports.

Source certificate: every mutation of the bot runtime map bumps the runtime epoch inside the same lock, the scan reads
its runtime snapshot together with the epoch, publication tags readiness with that epoch, and every reader validates
the epoch and freshness. Model certificate: exhaustive interleavings of a scanner, bot starts/stops and manual trades
never let a reader report ready while the current runtime and inventory are unreconciled; the pre-fix protocol
reproduces CE-READY-001. Conformance: a compiled regression against the actual Trader.App.Readiness module.
"""
from collections import deque
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
MAIN = 'haskell/app/Main.hs'
READINESS = 'haskell/app/Trader/App/Readiness.hs'
# Every definition that mutates the runtime map, and the reviewed fragment that bumps the epoch inside that mutation.
MUTATIONS = {
    'botStop': 'unless (null targets) (bumpRuntimeEpoch ctrl)',
    'botStartWorker': ['| bsrThreadId rt == tid -> do\n                                    bumpRuntimeEpoch ctrl\n                                    let tenantMap\' = HM.delete sym tenantMap',
                       '| bsrThreadId rt == tid -> do\n                                        bumpRuntimeEpoch ctrl\n                                        let tenantMap\' = HM.insert sym (BotRunning'],
    'botLoop': 'Just (BotRunning rt) | brThreadId rt == tid -> do\n                                bumpRuntimeEpoch ctrl',
    'botStartSymbolWithSettings': 'Nothing -> do\n                                                                                bumpRuntimeEpoch ctrl\n                                                                                stopSig <- newEmptyMVar',
    'runManualTrade': 'uninterruptibleMask_ (modifyMVar_ (bcRuntime ctrl) (\\mrt -> bumpRuntimeEpoch ctrl >> pure mrt))',
}
# Lock-taking uses of the runtime map that leave it unchanged.
READ_ONLY = {'withManualTradeClaim', 'botOwnerConflict'}


def require(ok, reason):
    if not ok:
        raise ValueError('readiness epoch: ' + reason)


def definitions(text):
    lines = text.split('\n')
    heads = [(i, m.group(1)) for i, line in enumerate(lines) for m in [re.match(r'^([a-z]\w*) ::', line)] if m]
    bodies = {}
    for k, (i, name) in enumerate(heads):
        j = heads[k + 1][0] if k + 1 < len(heads) else len(lines)
        bodies[name] = bodies.get(name, '') + '\n'.join(line.split('--')[0] for line in lines[i:j])
    return bodies


def bind(sources=None):
    sources = sources or {}
    main = sources.get(MAIN, (ROOT / MAIN).read_text())
    helper = sources.get(READINESS, (ROOT / READINESS).read_text())
    bodies = definitions(main)
    mutating = {name for name, body in bodies.items()
                if re.search(r'(modifyMVar_?|modifyMVarMasked|publishWorker) \(bcRuntime', body)}
    require(mutating == set(MUTATIONS), f'runtime mutation sites drifted: {sorted(mutating ^ set(MUTATIONS))}')
    for name, fragments in MUTATIONS.items():
        for fragment in ([fragments] if isinstance(fragments, str) else fragments):
            require(fragment in bodies[name], f'{name}: mutation does not bump the runtime epoch under the lock')
    # Six bump call sites plus the definition; the epoch field is otherwise only declared, snapshotted and read.
    require(len(re.findall(r'bumpRuntimeEpoch ctrl', main)) == 7 and
            main.count('modifyIORef\' (bcRuntimeEpoch ctrl) (+ 1)') == 1 and
            len(re.findall(r'bcRuntimeEpoch', main)) == 4, 'epoch written outside the reviewed sites')
    require('BotController <$> newMVar HM.empty <*> newManualTradeClaims <*> newMVar HM.empty <*> newIORef 0' in main, 'epoch initialisation')
    loop = bodies['botAutoStartLoop']
    revoke = loop.find('when (bsTradeEnabled settings) (writeIORef recoveryReadyRef NotReady)')
    snap = loop.find('(mrt, scanEpoch) <- withMVar (bcRuntime botCtrl) (\\m -> (,) m <$> readIORef (bcRuntimeEpoch botCtrl))')
    scan = loop.find('resolveOrphanOpenPositionActions mOps argsWithKeys tenantMap0')
    publish = loop.find('then ReadyAt scanEpoch scannedAtMs')
    require(0 <= revoke < snap < scan < publish and 'else NotReady' in loop[publish:publish + 200],
            'scan does not revoke, snapshot with its epoch, then publish epoch-tagged readiness')
    require(len(re.findall(r'writeIORef recoveryReadyRef', main)) == 2, 'readiness written outside the scan loop')
    readers = [line.strip() for line in main.split('\n') if re.search(r'readIORef (bot)?[rR]ecoveryReadyRef', line)]
    require(readers == [], 'readiness read without epoch validation')
    now = bodies['botRecoveryReadyNow']
    require('epochNow <- readIORef (bcRuntimeEpoch ctrl)' in now and
            'readinessHolds epochNow nowMs (readinessMaxAgeMs pollSec) readiness' in now, 'reader validation')
    require(main.count('botRecoveryReady <- botRecoveryReadyNow botCtrl botRecoveryReadyRef') == 2, 'HTTP readers')
    for fragment in ('ReadyAt epoch scannedAtMs ->\n            epoch == epochNow && scannedAtMs <= nowMs && nowMs - scannedAtMs <= maxAgeMs',
                     'ReadinessNotRequired -> True', 'NotReady -> False',
                     'readinessMaxAgeMs pollSec = 1000 * fromIntegral (max 300 (10 * max 1 pollSec))'):
        require(fragment in helper, 'readiness decision changed: ' + fragment.split('\n')[0])
    return {'status': 'exhaustively_checked', 'mutationSites': sum(1 if isinstance(v, str) else len(v) for v in MUTATIONS.values()),
            'readers': 2}


# ---- Model ------------------------------------------------------------------------------------------------------
# Symbols A and B. Runtime: set of symbols with a live bot. Venue: set of symbols with an open position.
# Reconciled(runtime, venue): every open position has a live bot.
# Actors: a scanner (snapshot -> venue read -> publish), a bot that may stop on A, a bot that may start on B,
# and a manual trade that opens B. Each step under the runtime lock is atomic.

def reconciled(runtime, venue):
    return venue <= runtime


def initial():
    # runtime, venue, epoch, scanner pc, snapshot, snapshot epoch, venue seen, published, events used
    return (frozenset({'A'}), frozenset({'A'}), 0, 'idle', None, None, None, ('none',), frozenset())


def successors(state, variant):
    runtime, venue, epoch, pc, snap, snap_epoch, seen, published, used = state
    bump = (lambda e: e) if variant == 'no-bump-on-stop' else (lambda e: e + 1)
    out = []
    if pc == 'idle':
        out.append(('revoke', (runtime, venue, epoch, 'snapped', runtime, epoch, None, ('none',), used)))
    elif pc == 'snapped':
        out.append(('venue', (runtime, venue, epoch, 'seen', snap, snap_epoch, venue, published, used)))
    elif pc == 'seen':
        ok = reconciled(snap, seen)
        out.append(('publish', (runtime, venue, epoch, 'done', snap, snap_epoch, seen, ('ready', snap_epoch) if ok else ('none',), used)))
    if 'stop' not in used and 'A' in runtime:  # bot A stops; its position stays open on the venue
        out.append(('stop', (runtime - {'A'}, venue, bump(epoch), pc, snap, snap_epoch, seen, published, used | {'stop'})))
    if 'start' not in used:  # a bot for B starts (adopting nothing yet)
        out.append(('start', (runtime | {'B'}, venue, epoch + 1, pc, snap, snap_epoch, seen, published, used | {'start'})))
    if 'trade' not in used:  # a manual trade opens B, then invalidates under the lock on completion
        out.append(('trade', (runtime, venue | {'B'}, epoch + 1, pc, snap, snap_epoch, seen, published, used | {'trade'})))
    return out


def reader_ready(state, variant):
    runtime, venue, epoch, pc, snap, snap_epoch, seen, published, used = state
    if published[0] != 'ready':
        return False
    return True if variant == 'no-epoch-check' else published[1] == epoch


def explore(variant):
    start = initial()
    seen_states, queue, edges = {start}, deque([start]), 0
    while queue:
        state = queue.popleft()
        if reader_ready(state, variant) and not reconciled(state[0], state[1]):
            return {'states': len(seen_states), 'transitions': edges, 'violation': repr(state)}
        for _, nxt in successors(state, variant):
            edges += 1
            if nxt not in seen_states:
                seen_states.add(nxt)
                queue.append(nxt)
    return {'states': len(seen_states), 'transitions': edges, 'violation': None}


def check_model():
    fixed = explore('fixed')
    require(fixed['violation'] is None and fixed['states'] > 20, 'epoch-validated readiness reports an unreconciled state')
    reachable_ready = any(reader_ready(s, 'fixed') for s in _all_states('fixed'))
    require(reachable_ready, 'readiness unreachable (vacuous)')
    mutants = {name: explore(name) for name in ('no-epoch-check', 'no-bump-on-stop')}
    require(all(r['violation'] for r in mutants.values()), 'a weakened protocol was not refuted')
    return {'status': 'model_checked', 'states': fixed['states'], 'transitions': fixed['transitions'],
            'counterexamples': {'CE-READY-001': mutants['no-epoch-check']['violation'],
                                'no-bump-on-stop': mutants['no-bump-on-stop']['violation']}}


def _all_states(variant):
    start = initial()
    seen_states, queue = {start}, deque([start])
    while queue:
        state = queue.popleft()
        for _, nxt in successors(state, variant):
            if nxt not in seen_states:
                seen_states.add(nxt)
                queue.append(nxt)
    return seen_states


PROGRAM = r'''
import Trader.App.Readiness
main :: IO ()
main = do
  let maxAge = readinessMaxAgeMs 30
  print maxAge
  print (map (readinessHolds 7 1000000 maxAge)
    [ReadinessNotRequired, NotReady, ReadyAt 7 900000, ReadyAt 6 900000, ReadyAt 7 600000, ReadyAt 7 1000001])
'''
EXPECTED = '300000\n[True,False,True,False,False,False]'


def conformance():
    with tempfile.TemporaryDirectory(prefix='trader-readiness-') as directory:
        path = Path(directory) / 'Readiness.hs'
        path.write_text(PROGRAM)
        out = subprocess.run(['runghc', '-i' + str(ROOT / 'haskell/app'), str(path)],
                             capture_output=True, text=True, timeout=300, check=True).stdout.strip()
    require(out == EXPECTED, 'compiled readiness regression: ' + out)
    return {'status': 'property_tested', 'observed': out}


def check_readiness_epoch():
    return {'source': bind(), 'model': check_model(), 'conformance': conformance()}
