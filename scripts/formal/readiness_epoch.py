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
    'runManualTrade': 'bumpRuntimeEpoch ctrl\n                atomicModifyIORef\' (bcManualInFlight ctrl) (\\n -> (n + delta, ()))',
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
            main.count("atomicModifyIORef' (bcRuntimeEpoch ctrl) (\\e -> (e + 1, ()))") == 1 and
            len(re.findall(r'bcRuntimeEpoch', main)) == 4, 'epoch written outside the reviewed sites')
    require('BotController <$> newMVar HM.empty <*> newManualTradeClaims <*> newMVar HM.empty <*> newIORef 0' in main, 'epoch initialisation')
    loop = bodies['botAutoStartLoop']
    revoke = loop.find('when (bsTradeEnabled settings) (writeIORef recoveryReadyRef NotReady)')
    snap = loop.find('(,,) m <$> readIORef (bcRuntimeEpoch botCtrl) <*> readIORef (bcManualInFlight botCtrl)')
    scan = loop.find('resolveOrphanOpenPositionActions mOps argsWithKeys tenantMap0')
    publish = loop.find('then ReadyAt scanEpoch scannedAtMs')
    require(0 <= revoke < snap < scan < publish and 'else NotReady' in loop[publish:publish + 200] and
            'null adoptionStartingSymbols && manualInFlightAtScan == 0' in loop,
            'scan does not revoke, snapshot with its epoch, then publish epoch-tagged readiness')
    require(len(re.findall(r'writeIORef recoveryReadyRef', main)) == 2, 'readiness written outside the scan loop')
    readers = [line.strip() for line in main.split('\n') if re.search(r'readIORef (bot)?[rR]ecoveryReadyRef', line)]
    require(readers == [], 'readiness read without epoch validation')
    now = bodies['botRecoveryReadyNow']
    require(0 <= now.find("manualInFlightNow <- atomicModifyIORef' (bcManualInFlight ctrl) (\\n -> (n, n))") < now.find("epochNow <- atomicModifyIORef' (bcRuntimeEpoch ctrl) (\\e -> (e, e))") and
            'readinessHolds manualInFlightNow epochNow nowMs (readinessMaxAgeMs pollSec) readiness' in now, 'reader validation')
    # A possibly-live manual trade is counted in flight from admission to completion, bumping the epoch at both ends.
    run = bodies['runManualTrade']
    require('affectsReadiness <- manualTradeAffectsReadiness ctrl args' in run and
            '(if affectsReadiness then bracket_ (manualInFlight 1) (manualInFlight (-1)) else id) $ do' in run and
            'uninterruptibleMask_ $\n            modifyMVar_ (bcRuntime ctrl)' in run, 'manual trade not counted in flight for its whole duration')
    require(len(re.findall(r'bcManualInFlight', main)) == 4, 'in-flight count written outside the reviewed site')
    # Only trades that can change the readiness account's inventory are counted; unknown identity counts.
    affects = bodies['manualTradeAffectsReadiness']
    require("| not (manualTradeMayBeLive args) || argPlatform args /= PlatformBinance = pure False" in affects and
            'serverAccount <- orderAccountKey ctrl (sanitizeArgsKeys args)' in affects and
            '(Right (Just trade), Right (Just server)) -> trade == server' in affects and '_ -> True' in affects,
            'readiness-account trade selection changed')
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
    # runtime, venue, epoch, in-flight, scanner pc, snapshot, snapshot epoch, venue seen, published, trade phase,
    # reader (phase, count read), events used
    return (frozenset({'A'}), frozenset({'A'}), 0, 0, 'idle', None, None, None, ('none',), 'none', ('idle', None), frozenset())


FIELDS = ('runtime', 'venue', 'epoch', 'flight', 'pc', 'snap', 'snap_epoch', 'seen', 'published', 'trade', 'reader', 'used')


def successors(state, variant):
    d = dict(zip(FIELDS, state))
    out = []
    def step(**kw):
        n = dict(d)
        n.update(kw)
        out.append(tuple(n[f] for f in FIELDS))
    runtime, venue, epoch, flight, used = d['runtime'], d['venue'], d['epoch'], d['flight'], d['used']
    # Admission and completion hold the runtime lock across their two writes: only the lock-free reader interleaves.
    locked = d['trade'] in ('admit1', 'done1')
    ignore_flight = variant == 'scan-ignores-flight'
    if locked:
        pass
    elif d['pc'] == 'idle':
        step(pc='snapped', snap=(runtime, flight), snap_epoch=epoch, published=('none',))
    elif d['pc'] == 'snapped':
        step(pc='seen', seen=venue)
    elif d['pc'] == 'seen':
        snap_runtime, snap_flight = d['snap']
        ok = reconciled(snap_runtime, d['seen']) and (ignore_flight or snap_flight == 0)
        step(pc='done', published=('ready', d['snap_epoch']) if ok else ('none',))
    if not locked and 'stop' not in used and 'A' in runtime:
        step(runtime=runtime - {'A'}, epoch=epoch if variant == 'no-bump-on-stop' else epoch + 1, used=used | {'stop'})
    if not locked and 'start' not in used:
        step(runtime=runtime | {'B'}, epoch=epoch + 1, used=used | {'start'})
    # Manual trade: each end is two visible writes (epoch, then count), as lock-free readers observe them.
    completion_only = variant == 'completion-only-trade'
    count_first = variant == 'count-before-epoch'
    t = d['trade']
    if t == 'none':
        step(trade='admit1', **({} if completion_only else ({'flight': flight + 1} if count_first else {'epoch': epoch + 1})))
    elif t == 'admit1':
        step(trade='admitted', **({} if completion_only else ({'epoch': epoch + 1} if count_first else {'flight': flight + 1})))
    elif t == 'admitted':
        step(trade='filled', venue=venue | {'B'})
    elif t == 'filled':
        step(trade='done1', **({'flight': flight - 1} if count_first and not completion_only else {'epoch': epoch + 1}))
    elif t == 'done1':
        step(trade='done', **({'epoch': epoch + 1} if count_first and not completion_only else ({} if completion_only else {'flight': flight - 1})))
    # Reader: reads the in-flight count, then the epoch, and decides at the second read.
    phase, count = d['reader']
    if phase == 'idle':
        step(reader=('counted', flight))
    elif phase == 'counted':
        published = d['published']
        ok = published[0] == 'ready' and (variant == 'no-epoch-check' or (count == 0 and published[1] == epoch))
        # The verdict is judged at the instant it is given (the epoch read): ready must mean reconciled now.
        step(reader=(('ready' if reconciled(runtime, venue) else 'violation') if ok else 'not-ready', None))
    return out


def violates(state):
    return dict(zip(FIELDS, state))['reader'][0] == 'violation'


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


def render(state):
    """Deterministic text for a state: frozenset reprs depend on per-process hash randomization."""
    return repr(tuple(sorted(x) if isinstance(x, frozenset) else x for x in state))


def explore(variant):
    states, edges = _all_states(variant)
    bad = sorted((render(s) for s in states if violates(s)), key=lambda x: (len(x), x))
    return {'states': len(states), 'transitions': edges, 'violation': bad[0] if bad else None,
            'readyReachable': any(dict(zip(FIELDS, s))['reader'][0] == 'ready' for s in states)}


def check_model():
    fixed = explore('fixed')
    require(fixed['violation'] is None and fixed['states'] > 20, 'epoch-validated readiness reports an unreconciled state')
    require(fixed['readyReachable'], 'readiness unreachable (vacuous)')
    mutants = {name: explore(name) for name in ('no-epoch-check', 'no-bump-on-stop', 'completion-only-trade', 'scan-ignores-flight')}
    # Defense in depth, not load-bearing: with the scan refusing to publish while a trade is in flight, the order of
    # the epoch and count writes no longer matters (recorded as checked-safe rather than refuted).
    order = explore('count-before-epoch')
    require(order['violation'] is None, 'write order became load-bearing')
    require(all(r['violation'] for r in mutants.values()), 'a weakened protocol was not refuted')
    return {'status': 'model_checked', 'states': fixed['states'], 'transitions': fixed['transitions'],
            'counterexamples': {'CE-READY-001': mutants['no-epoch-check']['violation'],
                                'no-bump-on-stop': mutants['no-bump-on-stop']['violation'],
                                'completion-only-trade': mutants['completion-only-trade']['violation'],
                                'scan-ignores-flight': mutants['scan-ignores-flight']['violation']},
            'countBeforeEpochSafe': True}


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
