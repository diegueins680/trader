"""Obligation 15, in-process part: at most one live actor per position identity (credential account, symbol).

Source certificate: every Main.hs definition that can transitively reach an order-placing or order-cancelling venue
primitive is classified, and the manual-trade claim and bot-start publication checks are bound to the source.
Model certificate: exhaustive interleavings of bot starts/stops and manual trades (admission, action, release, timeout) under
the single runtime lock never put two live actors on one identity; the pre-fix protocols reproduce CE-LIVE-002/003.
"""
import bisect
from collections import deque
import hashlib
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
MAIN = 'haskell/app/Main.hs'
OWNERSHIP = 'haskell/app/Trader/App/ManualTradeOwnership.hs'
# Venue primitives that place, cancel or swap: anything reaching one of these can act on a position.
PRIMITIVES = {'placeMarketOrder', 'placeFuturesMarketOrderWithPositionSide', 'placeFuturesPostOnlyLimitOrder',
              'placeFuturesTriggerMarketOrder', 'placeFuturesAlgoTriggerMarketOrder', 'placeCoinbaseMarketOrder',
              'cancelFuturesOrderByClientId', 'cancelFuturesAlgoOrderByClientId', 'cancelFuturesOpenOrdersByClientPrefix',
              'swapDexExactIn'}
ADAPTERS = ('haskell/app/Trader/Binance.hs', 'haskell/app/Trader/Coinbase.hs', 'haskell/app/Trader/Dex.hs')
# Reviewed role of every order-capable definition. A new one fails closed until it is classified here.
CLASSIFICATION = {
    'botStartWorker': 'bot-worker', 'initBotState': 'bot-worker', 'botLoop': 'bot-worker', 'botApplyKline': 'bot-worker',
    'botApplyKlineSafe': 'bot-worker', 'placeIfEnabled': 'bot-worker', 'placeBotCloseIfEnabled': 'bot-worker',
    'placeBotCloseOrder': 'bot-worker', 'placeOrderForSignalBot': 'bot-worker',
    'botStartSymbolWithSettings': 'bot-publication', 'botStartSymbol': 'bot-publication',
    'botAutoStartLoop': 'bot-publication', 'handleBotStart': 'bot-publication',
    'placeOrderForSignalEx': 'shared-order-path', 'placeOrderForSignal': 'shared-order-path',
    'placeOrderForSignalPlatform': 'shared-order-path', 'placeCoinbaseOrderForSignal': 'shared-order-path',
    'placeDexOrderForSignal': 'shared-order-path',
    'computeTradeFromSeries': 'manual-trade', 'computeTradeFromArgs': 'manual-trade',
    'computeTradeFromArgsWithLimits': 'manual-trade', 'handleTrade': 'manual-trade-claimed',
    'handleTradeAsync': 'manual-trade-claimed',
    'handleBinanceClosePosition': 'reduce-only-operator-override',
    'computeBinanceKeysStatusFromArgs': 'test-order-probe', 'handleBinanceKeys': 'test-order-probe',
    'maybeSendOrder': 'separate-cli-process', 'runTradeOnly': 'separate-cli-process',
    'runBacktestPipeline': 'separate-cli-process',
    'apiApp': 'server-root', 'runRestApi': 'server-root', 'main': 'process-root',
}


def require(ok, reason):
    if not ok:
        raise ValueError('position ownership: ' + reason)


def definitions(text):
    lines = text.split('\n')
    heads = [(i, m.group(1)) for i, line in enumerate(lines) for m in [re.match(r'^([a-z]\w*) ::', line)] if m]
    bodies = {}
    for k, (i, name) in enumerate(heads):
        j = heads[k + 1][0] if k + 1 < len(heads) else len(lines)
        bodies[name] = bodies.get(name, '') + '\n'.join(line.split('--')[0] for line in lines[i:j])
    return bodies


def order_capable(bodies):
    names = set(bodies)
    refs = {d: {w for w in re.findall(r"(?<![\w.'])([a-z]\w*)", t) if w != d and (w in names or w in PRIMITIVES)}
            for d, t in bodies.items()}
    capable = {d for d, r in refs.items() if r & PRIMITIVES}
    grew = True
    while grew:
        grew = False
        for d, r in refs.items():
            if d not in capable and r & capable:
                capable.add(d)
                grew = True
    return capable


def bind(sources=None):
    sources = sources or {}
    read = lambda p: sources.get(p, (ROOT / p).read_text())
    main, ownership = read(MAIN), read(OWNERSHIP)
    for adapter in ADAPTERS:
        exported = set(re.findall(r'^((?:place|cancel|swap)[A-Z]\w*) ::', read(adapter), re.M))
        require(exported <= PRIMITIVES, f'unclassified venue primitive in {adapter}: {sorted(exported - PRIMITIVES)}')
    bodies = definitions(main)
    capable = order_capable(bodies)
    require(capable == set(CLASSIFICATION),
            f'order-capable definitions drifted: new {sorted(capable - set(CLASSIFICATION))}, gone {sorted(set(CLASSIFICATION) - capable)}')
    # Manual HTTP trades hold a claim for the whole trade.
    for handler in ('handleTrade', 'handleTradeAsync'):
        uses = re.findall(r'computeTradeFromArgsWithLimits', bodies[handler])
        claimed = re.findall(r'runManualTrade botCtrl argsFinal \(computeTradeFromArgsWithLimits limits mOps argsFinal\)', bodies[handler])
        require(len(uses) == len(claimed) == 1, f'{handler}: trade not run under a manual-trade claim')
    for name, body in bodies.items():
        if name not in ('handleTrade', 'handleTradeAsync', 'computeTradeFromArgsWithLimits'):
            require('computeTradeFromArgsWithLimits' not in body, f'unclaimed manual trade entry in {name}')
        if name not in ('computeTradeFromArgs',):
            require(not re.search(r'\bcomputeTradeFromArgs\b', body), f'unclaimed manual trade entry in {name}')
    require(re.search(r'"POST" -> handleTrade botCtrl ', bodies['apiApp']) and
            re.search(r'"POST" -> handleTradeAsync botCtrl ', bodies['apiApp']), 'trade routes lost the bot controller')
    run = bodies['runManualTrade']
    require('| manualTradeMayBeLive args -> do' in run and
            'withManualTradeClaim (bcRuntime ctrl) ownedBotKeys (bcManualTrades ctrl) (account, normalizeSymbol symRaw) action' in run and
            'Left err -> throwIO (AccountIdentityUnavailable err)' in run and
            0 <= run.find('requireLiveOrderRole') < run.find('orderAccountKey ctrl args'),
            'manual claim keyed by account and normalized symbol under the runtime lock, failing closed')
    require('manualTradeMayBeLive args = not (argDryRun args) && (argPlatform args /= PlatformBinance || argBinanceLive args)'
            in bodies['manualTradeMayBeLive'], 'live-capability predicate changed')
    # Ownership identity uses exactly the credentials makeBinanceEnv/makeCoinbaseEnv sign with.
    account, env = bodies['orderCredentialKey'], bodies['makeBinanceEnv'] + bodies['makeCoinbaseEnv']
    # Live mainnet Binance identity is the exchange account UID (shared by every key pair of one account), fail-closed.
    uid = bodies['orderAccountKey']
    require("| argPlatform args == PlatformBinance && not (argBinanceTestnet args) -> do" in uid and
            'fetchSpotAccountUid' in uid and 'pure (Left ("Cannot resolve the Binance account identity' in uid and
            '("binance-uid:" ++)' in uid and 'HM.insert cred uid' in uid,
            'live Binance identity is not the fail-closed account UID')
    require('Just _ -> "Binance account request failed."' in bodies['accountLookupError'],
            'account lookup errors may echo request headers (API key)')
    for var, field in (('BINANCE_API_KEY', 'argBinanceApiKey'), ('BINANCE_API_SECRET', 'argBinanceApiSecret'),
                       ('COINBASE_API_KEY', 'argCoinbaseApiKey'), ('COINBASE_API_SECRET', 'argCoinbaseApiSecret'),
                       ('COINBASE_API_PASSPHRASE', 'argCoinbaseApiPassphrase')):
        fragment = f'resolveEnv "{var}" ({field} args)'
        require(fragment in account and fragment in env, 'account identity differs from signing credentials: ' + var)
    # A bot owns its identity whenever its order path can go live; that path ignores argDryRun.
    require('bsTradeEnabled settings && (argPlatform args /= PlatformBinance || argBinanceLive args) = fmap (fmap (,sym)) <$> orderAccountKey ctrl args'
            in bodies['botOwnerKey'] and 'argDryRun' not in bodies['botOwnerKey'], 'bot owner identity changed')
    for name in ('placeOrderForSignalEx', 'placeIfEnabled', 'placeBotCloseIfEnabled', 'placeBotCloseOrder', 'placeOrderForSignalBot'):
        require('argDryRun' not in bodies[name],
                'bot order path gained a dry-run gate; revisit the bot owner predicate: ' + name)
    async_body = bodies['handleTradeAsync']
    require(0 <= async_body.find('ownedNow <- manualTradeOwnedNow botCtrl argsFinal') < async_body.find('startJob mOps store') and
            'respond (jsonError refusalStatus msg)' in async_body, 'async trade is queued before the ownership refusal')
    pre = bodies['manualTradeOwnedNow']
    require(0 <= pre.find('roleRefusal <- liveOrderRoleRefusal <$> resolveServerRole') < pre.find('orderAccountKey ctrl args') and
            'Just msg -> pure (Just (status403, msg))' in pre and '(status409,) <$> manualTradeConflict' in pre,
            'async pre-check does not refuse a non-trading role before ownership')
    # Bot start publication: the only insertion of a new owner, checked under the lock.
    require(main.count('publishWorker (bcRuntime ctrl)') == 1, 'more than one bot publication site')
    publish = bodies['botStartSymbolWithSettings']
    i = publish.find('publishWorker (bcRuntime ctrl)')
    check, launch = (publish.find('ownerConflict <- botOwnerConflict ctrl mrt mOwnerKey', i), publish.find('Launch (botStartWorker', i))
    lookup = publish.find('ownerOrErr <- botOwnerKey ctrl argsSym settings sym')
    refuse = publish.find('Left err -> pure (Left err)', max(lookup, 0))
    require(0 <= lookup < refuse < i,
            'bot start does not refuse an unresolved owner identity before publication')
    require(i >= 0 and 0 <= check < launch and
            'Just err -> pure (Retain (Left err))' in publish and 'bsrOwnerKey = mOwnerKey' in publish,
            'bot publication does not refuse a claimed or owned identity before launch')
    conflict = bodies['botOwnerConflict']
    require('claimed <- manualTradeClaimed (bcManualTrades ctrl) key' in conflict and
            'key `elem` ownedBotKeys mrt' in conflict, 'bot owner conflict check')
    require(len(re.findall(r'HM\.insert sym \((?:BotStarting|BotRunning)', main)) == 1 and
            'HM.insert sym (BotRunning (BotRuntime tid stVar stopSig mOptimizerRt (bsrOwnerKey rt))) tenantMap' in main,
            'runtime insertion other than the guarded publication and its owner-preserving promotion')
    require(len(re.findall(r'\bbcManualTrades\b', main)) == 3, 'manual claims accessed outside the reviewed sites')
    require('st = BotStarting rt' in publish, 'publication inserts the starting runtime')
    # Reviewed non-claimed order paths.
    probe = bodies['computeBinanceKeysStatusFromArgs']
    require(re.findall(r'placeMarketOrder env (\w+)', probe) and set(re.findall(r'placeMarketOrder env (\w+)', probe)) == {'OrderTest'},
            'key probe sends a non-test order')
    require(re.search(r'placeFuturesMarketOrderWithPositionSide env OrderLive sym side qty \(Just True\)', bodies['handleBinanceClosePosition']),
            'close-position is no longer reduce-only')
    # Helper module: admission and release under the lock, release uninterruptible.
    for fragment in ('modifyMVarMasked lock $ \\runtime ->\n            case manualTradeConflict (owned runtime) key of',
                     "modifyIORef' ref (Map.insertWith (+) key 1)", 'uninterruptibleMask_ $\n                    modifyMVar_ lock',
                     'bracket acquire release (either (throwIO . ManualTradeOwnershipConflict) (const action))',
                     '| key `elem` owners ='):
        require(fragment in ownership, 'manual claim protocol changed: ' + fragment.split('\n')[0])
    return {'status': 'exhaustively_checked', 'orderCapableDefinitions': len(capable),
            'roles': sorted(set(CLASSIFICATION.values())),
            'sha256': {p: hashlib.sha256(read(p).encode()).hexdigest() for p in (MAIN, OWNERSHIP)}}


# ---- Server-role gate ------------------------------------------------------------------------------------------
LIVE_ROLE = 'haskell/app/Trader/App/LiveRole.hs'
OBSERVABILITY = 'haskell/app/Trader/App/Observability.hs'
GATED = {'haskell/app/Trader/Binance.hs': {'placeMarketOrder', 'placeFuturesMarketOrderWithPositionSide', 'placeFuturesPostOnlyLimitOrder',
                                          'placeFuturesTriggerMarketOrder', 'placeFuturesAlgoTriggerMarketOrder',
                                          'cancelFuturesOrderByClientId', 'cancelFuturesAlgoOrderByClientId',
                                          'cancelFuturesOpenOrdersByClientPrefix'},
         'haskell/app/Trader/Coinbase.hs': {'placeCoinbaseMarketOrder'},
         'haskell/app/Trader/Dex.hs': {'sendDexTx'}}
MODE_GATED = {'placeMarketOrder', 'placeFuturesMarketOrderWithPositionSide', 'placeFuturesPostOnlyLimitOrder',
              'placeFuturesTriggerMarketOrder', 'placeFuturesAlgoTriggerMarketOrder'}


def bind_role_gate(sources=None):
    sources = sources or {}
    read = lambda p: sources.get(p, (ROOT / p).read_text())
    sites = 0
    for path, names in GATED.items():
        text = read(path)
        exported = set(re.findall(r'^((?:place|cancel|swap|send)[A-Z]\w*) ::', text, re.M))
        require(exported - {'swapDexExactIn'} <= names, f'ungated venue action in {path}: {sorted(exported - names - {"swapDexExactIn"})}')
        for name in names:
            m = re.search(r'^' + name + r' [^\n]*= do\n( +)(\S[^\n]*)\n', text, re.M)
            want = 'Control.Monad.when (mode == OrderLive) requireLiveOrderRole' if name in MODE_GATED else 'requireLiveOrderRole'
            require(m and m.group(2).strip() == want, f'{name}: role gate is not the first action')
            sites += 1
    # On-chain effects only go through sendDexTx; the 1inch helpers only quote and build transactions.
    dex = definitions(read('haskell/app/Trader/Dex.hs'))
    for name in ('swapDexExactIn', 'ensureAllowance'):
        require('sendDexTx' in dex[name] and not re.search(r'readProcess|createProcess|callProcess|httpLbs', dex[name]),
                f'DEX {name} acts outside sendDexTx')
    require(not re.search(r'readProcess|createProcess|callProcess|proc ', ''.join(v for k, v in dex.items() if k not in ('sendDexTx', 'waitForReceipt'))),
            'DEX module launches a process outside sendDexTx')
    require(re.findall(r'proc "(\w+)" \[?"?(\w+)', dex['waitForReceipt']) == [('cast', 'receipt')], 'receipt polling is not read-only')
    role = read(LIVE_ROLE)
    require('nonTradingRoles = ["research", "read-only", "readonly", "fly"]' in role and
            'explicit <- nonEmpty <$> lookupEnv "TRADER_SERVER_ROLE"' in role and
            '| isJust flyMachine || isJust flyApp -> "fly"' in role and '| otherwise -> "local"' in role and
            'maybe (pure ()) (throwIO . LiveOrderRoleRefused) (liveOrderRoleRefusal role)' in role, 'role gate decision changed')
    obs = read(OBSERVABILITY)
    require('explicitRole <- lookupTextEnv "TRADER_SERVER_ROLE"' in obs and 'fallbackRole = if onFly then Just "fly" else Just "local"' in obs,
            'server identity role resolution diverged from the gate')
    # Checked-in non-trading deployments carry a denied role; the trading deployment does not.
    deployments = {'fly.toml': r'^\s*TRADER_SERVER_ROLE = "read-only"$', 'fly.research.toml': r'^\s*TRADER_SERVER_ROLE = "research"$',
                   'deploy/hetzner/trader.research.env.managed': r'^TRADER_SERVER_ROLE=research$'}
    for path, pattern in deployments.items():
        require(len(re.findall(r'TRADER_SERVER_ROLE', read(path))) == 1 and re.search(pattern, read(path), re.M),
                'non-trading deployment lost its role label: ' + path)
    require('TRADER_SERVER_ROLE' not in read('deploy/hetzner/trader.trading.env.managed'), 'trading deployment role changed')
    require('merge_env_overlay "$MANAGED_ENV_FILE" "$ENV_FILE"' in read('deploy/hetzner/deploy-remote.sh'),
            'managed env overlay no longer applied on deploy')
    return {'status': 'exhaustively_checked', 'gatedPrimitives': sites, 'deniedRoles': ['research', 'read-only', 'readonly', 'fly'],
            'labelledDeployments': sorted(deployments)}


ROLE_PROGRAM = r'''
import System.Environment
import Trader.App.LiveRole
import Control.Exception
main :: IO ()
main = do
  let decide = map (maybe "allowed" (const "refused") . liveOrderRoleRefusal)
  print (decide ["research", "read-only", "fly", "trading", "standalone", "local"])
  setEnv "TRADER_SERVER_ROLE" "Research"
  r <- try requireLiveOrderRole :: IO (Either LiveOrderRoleRefused ())
  setEnv "TRADER_SERVER_ROLE" "trading"
  ok <- try requireLiveOrderRole :: IO (Either LiveOrderRoleRefused ())
  unsetEnv "TRADER_SERVER_ROLE"
  setEnv "FLY_APP_NAME" "trader-hs"
  fly <- resolveServerRole
  print (either (const "refused") (const "allowed") r, either (const "refused") (const "allowed") ok, fly)
'''
ROLE_EXPECTED = '["refused","refused","refused","allowed","allowed","allowed"]\n("refused","allowed","fly")'


def role_conformance():
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory(prefix='trader-role-') as directory:
        path = Path(directory) / 'Role.hs'
        path.write_text(ROLE_PROGRAM)
        env = {k: v for k, v in __import__('os').environ.items() if k not in ('TRADER_SERVER_ROLE', 'FLY_APP_NAME', 'FLY_MACHINE_ID')}
        out = subprocess.run(['runghc', '-i' + str(ROOT / 'haskell/app'), str(path)], env=env,
                             capture_output=True, text=True, timeout=300, check=True).stdout.strip()
    require(out == ROLE_EXPECTED, 'compiled role-gate regression: ' + out)
    return {'status': 'property_tested', 'observed': out}


# ---- Protocol model -------------------------------------------------------------------------------------------
# Identities: account A and B on one symbol. Bots: (tenant, account). Manual trades: account.
BOTS = (('t1', 'A'), ('t2', 'A'), ('t3', 'B'))
TRADES = ('A', 'A', 'B')


def successors(state, variant):
    bots, trades = state  # bots: tuple of running flags; trades: tuple of phase 'idle'|'acting'|'done'
    owners = [BOTS[i][1] for i, up in enumerate(bots) if up]
    claims = [TRADES[i] for i, phase in enumerate(trades) if phase == 'acting']
    for i, up in enumerate(bots):
        tenant, account = BOTS[i]
        if not up:
            if variant == 'tenant-keyed':  # CE-LIVE-003: uniqueness only per (tenant, symbol)
                ok = True
            else:
                ok = account not in owners and (variant == 'no-bot-claim-check' or account not in claims)
            if ok:
                yield (bots[:i] + (True,) + bots[i + 1:], trades)
        else:
            yield (bots[:i] + (False,) + bots[i + 1:], trades)  # stop: removed under the lock
    for i, phase in enumerate(trades):
        if phase == 'idle':
            if variant == 'unguarded-manual' or TRADES[i] not in owners:  # CE-LIVE-002 when unguarded
                yield (bots, trades[:i] + ('acting',) + trades[i + 1:])
            else:
                yield (bots, trades[:i] + ('done',) + trades[i + 1:])  # refused with 409
        elif phase == 'acting':
            yield (bots, trades[:i] + ('done',) + trades[i + 1:])  # completes, fails or times out; release is uninterruptible


def violation(state):
    """Owners of an identity are one bot worker or the account's manual operator (concurrent manual trades on one
    account are the same owner; idempotency keys order them). Two bots, or a bot and a manual trade, violate it."""
    bots, trades = state
    for account in {'A', 'B'}:
        live_bots = sum(1 for i, up in enumerate(bots) if up and BOTS[i][1] == account)
        manual = any(p == 'acting' and TRADES[i] == account for i, p in enumerate(trades))
        if live_bots > 1 or (live_bots and manual):
            return account
    return None


def explore(variant):
    start = ((False,) * len(BOTS), ('idle',) * len(TRADES))
    seen, queue, edges = {start}, deque([start]), 0
    while queue:
        state = queue.popleft()
        bad = violation(state)
        if bad:
            return {'states': len(seen), 'edges': edges, 'violation': {'account': bad, 'state': repr(state)}}
        for nxt in successors(state, variant):
            edges += 1
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return {'states': len(seen), 'edges': edges, 'violation': None}


def check_model():
    fixed = explore('fixed')
    require(fixed['violation'] is None and fixed['states'] > 50, 'fixed protocol admits two live actors')
    mutants = {name: explore(name) for name in ('unguarded-manual', 'tenant-keyed', 'no-bot-claim-check')}
    require(all(r['violation'] for r in mutants.values()), 'a weakened protocol was not refuted')
    return {'status': 'model_checked', 'states': fixed['states'], 'transitions': fixed['edges'], 'fixed': fixed,
            'counterexamples': {'CE-LIVE-002': mutants['unguarded-manual']['violation'],
                                'CE-LIVE-003': mutants['tenant-keyed']['violation'],
                                'no-bot-claim-check': mutants['no-bot-claim-check']['violation']}}




PROGRAM = r'''
{-# LANGUAGE OverloadedStrings #-}
import Control.Concurrent
import Control.Exception
import Control.Monad
import qualified Data.Map.Strict as Map
import Data.IORef
import System.Timeout (timeout)
import Trader.App.ManualTradeOwnership

type Runtime = Map.Map String [OwnershipKey]

owned :: Runtime -> [OwnershipKey]
owned = concat . Map.elems

startBot :: MVar Runtime -> ManualTradeClaims -> String -> OwnershipKey -> IO Bool
startBot lock claims tenant key = modifyMVar lock $ \rt -> do
  claimed <- manualTradeClaimed claims key
  if claimed || key `elem` owned rt then pure (rt, False) else pure (Map.insertWith (++) tenant [key] rt, True)

main :: IO ()
main = do
  let a = ("binance:a", "BTCUSDT"); b = ("binance:b", "BTCUSDT")
  lock <- newMVar Map.empty
  claims <- newManualTradeClaims
  s1 <- startBot lock claims "server" a
  s2 <- startBot lock claims "header" a
  s3 <- startBot lock claims "header" b
  refused <- try (withManualTradeClaim lock owned claims a (pure ())) :: IO (Either ManualTradeOwnershipConflict ())
  lock2 <- newMVar Map.empty
  during <- withManualTradeClaim lock2 owned claims a (startBot lock2 claims "server" a)
  _ <- timeout 20000 (withManualTradeClaim lock2 owned claims b (threadDelay 2000000))
  leaked <- manualTradeClaimed claims b
  overlaps <- newIORef (0 :: Int)
  replicateM_ 300 $ do
    l <- newMVar Map.empty
    c <- newManualTradeClaims
    d1 <- newEmptyMVar
    d2 <- newEmptyMVar
    _ <- forkIO (startBot l c "t" a >> putMVar d1 ())
    _ <- forkIO $ do
      _ <- try (withManualTradeClaim l owned c a (threadDelay 50 >> readMVar l >>= \rt -> when (a `elem` owned rt) (modifyIORef' overlaps (+ 1)))) :: IO (Either ManualTradeOwnershipConflict ())
      putMVar d2 ()
    takeMVar d1 >> takeMVar d2
  n <- readIORef overlaps
  print (s1, s2, s3, either (const "refused") (const "admitted") refused, during, leaked, n)
'''
EXPECTED = '(True,False,True,"refused",False,False,0)'


def conformance():
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory(prefix='trader-ownership-') as directory:
        path = Path(directory) / 'Ownership.hs'
        path.write_text(PROGRAM)
        out = subprocess.run(['runghc', '-i' + str(ROOT / 'haskell/app'), str(path)],
                             capture_output=True, text=True, timeout=300, check=True).stdout.strip()
    require(out == EXPECTED, 'compiled ownership regression: ' + out)
    return {'status': 'property_tested', 'observed': out}


def check_position_ownership():
    return {'source': bind(), 'model': check_model(), 'conformance': conformance(),
            'roleGate': {'source': bind_role_gate(), 'conformance': role_conformance()}}
