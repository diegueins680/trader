"""Obligation 35: retries do not duplicate externally visible order intents (or ownership).

Source certificate: exchange writes are never re-sent by the transport (venueRetryConfig), the only order re-send is
the timestamp retry after an explicit -1021 rejection, every live Binance/Coinbase order decision re-reads venue
position or balance and is a no-op for an already-held target, keyed orders reconcile an unknown outcome by client
order id, and retried bot starts are refused by the registry. Model: an order whose response is lost, followed by
re-decisions, never leaves more than the target exposure; a delta-sized decision or a write-retrying transport does.
"""
from collections import deque
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
MAIN = 'haskell/app/Main.hs'
HTTP = 'haskell/app/Trader/Http.hs'
BINANCE = 'haskell/app/Trader/Binance.hs'
COINBASE = 'haskell/app/Trader/Coinbase.hs'


def require(ok, reason):
    if not ok:
        raise ValueError('order idempotence: ' + reason)


def bind(sources=None):
    sources = sources or {}
    read = lambda p: sources.get(p, (ROOT / p).read_text())
    main, http, binance, coinbase = read(MAIN), read(HTTP), read(BINANCE), read(COINBASE)
    # 1. Transport: venue writes are never re-sent.
    require('venueRetryConfig = defaultRetryConfig{rcRetryWrites = False}' in http, 'venue retry policy may re-send writes')
    for method in ('"put" -> rcRetryWrites cfg', '"delete" -> rcRetryWrites cfg', '"post" -> rcRetryWrites cfg'):
        require(method in http, 'write methods no longer governed by rcRetryWrites')
    require('respOrErr <- trySync (httpLbsWithRetry venueRetryConfig Nothing (beManager env) req)' in binance and
            len(re.findall(r'httpLbsWithRetry', binance)) == 2, 'Binance request outside the venue retry policy')
    require('coinbaseHttp env label = httpLbsWithRetry venueRetryConfig (Just label) (ceManager env)' in coinbase and
            'resp <- coinbaseHttp env "coinbase.order" req' in coinbase, 'Coinbase order outside the venue retry policy')
    # 2. The only order re-send: after an explicit timestamp rejection.
    retry = binance[binance.index('withBinanceTimestampRetry env send = do'):]
    retry = retry[:retry.index('\n\n')]
    require('if code >= 200 && code < 300\n        then pure resp' in retry and 'if isBinanceTimestampError resp' in retry and
            len(re.findall(r'(?<![\w])send ts', retry)) == 2, 'timestamp retry re-sends outside an explicit rejection')
    require('Just code | code == binanceTimestampErrorCode -> True' in binance, 'timestamp rejection predicate changed')
    # 3. Target-position decisions: venue state is read before every live order decision.
    futures = main[main.index('    placeFutures mSf quoteAsset dir = do'):]
    require(0 <= futures.find('summary <- fetchFuturesPositionSummary env sym') < futures.find('if posAmt > 0') and
            futures.find('summary <- fetchFuturesPositionSummary env sym') < futures.find('\n    place') and
            '"No market order: already long."' in futures and '"No market order: already short."' in futures,
            'futures decisions do not re-read the venue position')
    spot = main[main.index('    placeSpotOrMargin mSf baseAsset quoteAsset dir = do'):]
    require(0 <= spot.find('baseBal <- fetchFreeBalance env baseAsset') < spot.find('if alreadyLong') and
            'then pure baseResult{aorMessage = "No order: already long."}' in spot, 'spot decisions do not re-read the venue balance')
    cb = main[main.index('placeCoinbaseOrderForSignal args symRaw sig env = do'):]
    require('if isLongCoinbaseSpot mBaseMinQty baseBal\n            then noOrder "No order: already long."' in main and
            'baseBal <-' in cb[:4000], 'Coinbase decisions do not re-read the venue balance')
    # 4. Keyed orders reconcile an unknown outcome by client order id.
    require('r2 <- trySync (fetchOrderByClientId env sym cid)' in main and
            'aorMessage = "Order reconciled by clientOrderId after error: " ++ shortErr ex' in main, 'client-order-id reconciliation removed')
    # 5. Ownership: a retried start is refused while the bot exists.
    require('BotRunning _ -> pure (Retain (Left "Bot is already running"))' in main and
            'BotStarting _ -> pure (Retain (Left "Bot is starting"))' in main, 'retried bot start not refused')
    require('TradeIdemCached cachedValue -> respond (jsonValue status200 cachedValue)' in main, 'manual trade idempotency key replay removed')
    return {'status': 'exhaustively_checked', 'venues': ['binance', 'coinbase']}


# ---- Model ------------------------------------------------------------------------------------------------------
# Target exposure 1. Each step: a decision computes an order (target semantics: target - observed; delta semantics:
# always 1), the venue may apply it, and the response may be lost (outcome unknown to the caller). The transport may
# re-send a write (mutant). Up to three decisions, each observing the venue position at decision time.

def explore(variant):
    start = (0, 0, 0)  # venue position, decisions made, in-flight resend allowance
    seen, queue, edges, violation = {start}, deque([start]), 0, None
    while queue:
        pos, decisions, resend = queue.popleft()
        if pos > 1 and violation is None:
            violation = (pos, decisions)
        nxt = []
        if decisions < 3:
            size = 1 if variant == 'delta-sizing' else max(0, 1 - pos)
            for applied in (True, False):
                for lost in (True, False):
                    new_pos = pos + (size if applied else 0)
                    allow = 1 if variant == 'retry-writes' and lost and size else 0
                    nxt.append((new_pos, decisions + 1, allow))
        if resend:
            nxt.append((pos + 1, decisions, 0))  # the transport re-sends the same write and it applies again
        for n in nxt:
            edges += 1
            if n not in seen:
                seen.add(n)
                queue.append(n)
    return {'states': len(seen), 'transitions': edges, 'violation': violation}


def check_model():
    fixed = explore('fixed')
    require(fixed['violation'] is None and fixed['states'] > 3, 'target-position decisions duplicated exposure')
    mutants = {v: explore(v) for v in ('delta-sizing', 'retry-writes')}
    require(all(m['violation'] for m in mutants.values()), 'a duplicating protocol was not refuted')
    return {'status': 'model_checked', 'states': fixed['states'], 'transitions': fixed['transitions'],
            'counterexamples': {k: repr(v['violation']) for k, v in mutants.items()}}


def check_order_idempotence():
    return {'source': bind(), 'model': check_model()}
