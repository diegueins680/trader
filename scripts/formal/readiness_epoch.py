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
    'runManualTrade': 'modifyMVar_ (bcRuntime ctrl) $ \\mrt -> do\n                modifyIORef\' (bcManualInFlight ctrl) (+ delta)\n                bumpRuntimeEpoch ctrl',
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
    require(0 <= now.find('manualInFlightNow <- readIORef (bcManualInFlight ctrl)') < now.find('epochNow <- readIORef (bcRuntimeEpoch ctrl)') and
            'readinessHolds manualInFlightNow epochNow nowMs (readinessMaxAgeMs pollSec) readiness' in now, 'reader validation')
    # A possibly-live manual trade is counted in flight from admission to completion, bumping the epoch at both ends.
    run = bodies['runManualTrade']
    require('runManualTrade ctrl args action = bracketManualTrade $ do' in run and
            '| manualTradeMayBeLive args = bracket_ (manualInFlight 1) (manualInFlight (-1))' in run and
            'uninterruptibleMask_ $\n            modifyMVar_ (bcRuntime ctrl)' in run, 'manual trade not counted in flight for its whole duration')
    require(len(re.findall(r'bcManualInFlight', main)) == 3, 'in-flight count written outside the reviewed site')
    require(main.count('botRecoveryReady <- botRecoveryReadyNow botCtrl botRecoveryReadyRef') == 2, 'HTTP readers')
    for fragment in ('ReadyAt epoch scannedAtMs ->\n            manualInFlight == 0 && epoch == epochNow && scannedAtMs <= nowMs && nowMs - scannedAtMs <= maxAgeMs',
                     'ReadinessNotRequired -> True', 'NotReady -> False',
                     'readinessMaxAgeMs pollSec = 1000 * fromIntegral (max 300 (10 * max 1 pollSec))'):
        require(fragment in helper, 'readiness decision changed: ' + fragment.split('\n')[0])
    return {'status': 'exhaustively_checked', 'mutationSites': sum(1 if isinstance(v, str) else len(v) for v in MUTATIONS.values()),
            'readers': 2}


# ---- Model ------------------------------------------------------------------------------------------------------
# Symbols A and B. Runtime: symbols with a live bot. Venue: symbols with an open position.
# Reconciled(runtime, venue): every open position has a live bot.
# Actors: the scanner (revoke -> snapshot+epoch -> venue read -> publish); bot A stops (its position stays open);
# a bot for B starts; a manual trade is admitted, fills (opens B) and completes. Lock-protected steps are atomic.
# A reader validates (in-flight count, epoch) as the server does.

def reconciled(runtime, venue):
    return venue <= runtime


def initial():
    # runtime, venue, epoch, in-flight trades, scanner pc, snapshot, snapshot epoch, venue seen, published, trade phase, events
    return (frozenset({'A'}), frozenset({'A'}), 0, 0, 'idle', None, None, None, ('none',), 'none', frozenset())


def successors(state, variant):
    runtime, venue, epoch, flight, pc, snap, snap_epoch, seen, published, trade, used = state
    out = []
    def with_(**kw):
        d = dict(runtime=runtime, venue=venue, epoch=epoch, flight=flight, pc=pc, snap=snap, snap_epoch=snap_epoch,
                 seen=seen, published=published, trade=trade, used=used)
        d.update(kw)
        return (d['runtime'], d['venue'], d['epoch'], d['flight'], d['pc'], d['snap'], d['snap_epoch'], d['seen'],
                d['published'], d['trade'], d['used'])
    if pc == 'idle':
        out.append(with_(pc='snapped', snap=runtime, snap_epoch=epoch, published=('none',)))
    elif pc == 'snapped':
        out.append(with_(pc='seen', seen=venue))
    elif pc == 'seen':
        out.append(with_(pc='done', published=('ready', snap_epoch) if reconciled(snap, seen) else ('none',)))
    if 'stop' not in used and 'A' in runtime:
        out.append(with_(runtime=runtime - {'A'}, epoch=epoch if variant == 'no-bump-on-stop' else epoch + 1, used=used | {'stop'}))
    if 'start' not in used:
        out.append(with_(runtime=runtime | {'B'}, epoch=epoch + 1, used=used | {'start'}))
    completion_only = variant == 'completion-only-trade'
    if trade == 'none':
        out.append(with_(trade='admitted', flight=flight if completion_only else flight + 1, epoch=epoch if completion_only else epoch + 1))
    elif trade == 'admitted':
        out.append(with_(trade='filled', venue=venue | {'B'}))
    elif trade == 'filled':
        out.append(with_(trade='done', flight=flight if completion_only else flight - 1, epoch=epoch + 1))
    return out


def reader_ready(state, variant):
    runtime, venue, epoch, flight, pc, snap, snap_epoch, seen, published, trade, used = state
    if published[0] != 'ready':
        return False
    if variant == 'no-epoch-check':
        return True
    return flight == 0 and published[1] == epoch


def _all_states(variant):
    start = initial()
    seen_states, queue, edges = {start}, deque([start]), 0
    while queue:
        state = queue.popleft()
        for nxt in successors(state, variant):
            edges += 1
            if nxt not in seen_states:
                seen_states.add(nxt)
                queue.append(nxt)
    return seen_states, edges


def explore(variant):
    states, edges = _all_states(variant)
    bad = sorted((repr(s) for s in states if reader_ready(s, variant) and not reconciled(s[0], s[1])), key=len)
    return {'states': len(states), 'transitions': edges, 'violation': bad[0] if bad else None,
            'readyReachable': any(reader_ready(s, variant) for s in states)}


def check_model():
    fixed = explore('fixed')
    require(fixed['violation'] is None and fixed['states'] > 20, 'epoch-validated readiness reports an unreconciled state')
    require(fixed['readyReachable'], 'readiness unreachable (vacuous)')
    mutants = {name: explore(name) for name in ('no-epoch-check', 'no-bump-on-stop', 'completion-only-trade')}
    require(all(r['violation'] for r in mutants.values()), 'a weakened protocol was not refuted')
    return {'status': 'model_checked', 'states': fixed['states'], 'transitions': fixed['transitions'],
            'counterexamples': {'CE-READY-001': mutants['no-epoch-check']['violation'],
                                'no-bump-on-stop': mutants['no-bump-on-stop']['violation'],
                                'completion-only-trade': mutants['completion-only-trade']['violation']}}


PROGRAM = r'''
import Trader.App.Readiness
main :: IO ()
main = do
  let maxAge = readinessMaxAgeMs 30
  print maxAge
  print (map (readinessHolds 0 7 1000000 maxAge)
    [ReadinessNotRequired, NotReady, ReadyAt 7 900000, ReadyAt 6 900000, ReadyAt 7 600000, ReadyAt 7 1000001])
  print (readinessHolds 1 7 1000000 maxAge (ReadyAt 7 900000))
'''
EXPECTED = '300000\n[True,False,True,False,False,False]\nFalse'


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
