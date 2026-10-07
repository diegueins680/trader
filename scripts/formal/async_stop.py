"""CE-LIVE-001: a killed bot worker must not keep acting. Source binding of every exception capture on the
worker and order paths, plus a compiled regression against the actual Trader.App.AsyncSafe module."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
MAIN = 'haskell/app/Main.hs'
BINANCE = 'haskell/app/Trader/Binance.hs'
ASYNC = 'haskell/app/Trader/App/AsyncSafe.hs'
COUNTEREXAMPLE = 'formal/research/async-stop-counterexample.json'
# Top-level definitions whose bodies run on a live bot worker or place/cancel exchange orders.
WORKER_SPAN = ('botStartWorker ::', 'reconcileBotPositionWithExchange ::')
ORDER_SPAN = ('placeOrderForSignal ::', 'computeBacktestFromArgs ::')
BARE_TRY = re.compile(r'\btry\b(?!\w)')
OTHER_TRY = re.compile(r'\btry(ReadMVar|PutMVar|TakeMVar|Sync)\b|\bretry')


def require(ok, reason):
    if not ok:
        raise ValueError('async stop: ' + reason)


def _span(text, start, end):
    i = text.index('\n' + start) + 1
    return text[i:text.index('\n' + end, i)]


def bind(sources=None):
    sources = sources or {}
    main = sources.get(MAIN, (ROOT / MAIN).read_text())
    binance = sources.get(BINANCE, (ROOT / BINANCE).read_text())
    helper = sources.get(ASYNC, (ROOT / ASYNC).read_text())
    require('import Trader.App.AsyncSafe (trySync)' in main and 'import Trader.App.AsyncSafe (trySync)' in binance, 'trySync imports')
    sites = 0
    for name, (a, b) in (('worker', WORKER_SPAN), ('order', ORDER_SPAN)):
        body = _span(main, a, b)
        bare = [line.strip() for line in body.split('\n') if BARE_TRY.search(OTHER_TRY.sub('', line))]
        require(not bare, f'bare try on the {name} path: ' + (bare[0] if bare else ''))
        require(not re.search(r'(?<![\w-])(catch|handle|catchAny|tryAny)(?![\w-])', body), f'other capture forms on the {name} path')
        sites += len(re.findall(r'\btrySync\b', body))
    require(sites == 32, f'reviewed trySync site count drifted: {sites}')
    require('( trySync $\n            fetchWithCache binanceTimeOffsetCache' in binance, 'signing timestamp offset capture')
    require('(\\cid -> trySync (cancelFuturesOrderByClientId env symbol cid)' in binance, 'cancel-by-prefix capture')
    body = helper[helper.index('trySync :: IO a -> IO (Either SomeException a)'):]
    require('Left ex | isAsyncException ex -> throwIO ex' in body and
            'isAsyncException ex = isJust (fromException ex :: Maybe SomeAsyncException)' in helper, 'trySync rethrows every async exception')
    return {'status': 'exhaustively_checked', 'workerAndOrderSites': sites, 'binanceSites': 2,
            'scope': 'botStartWorker..reconcileBotPositionWithExchange (worker, autostart, optimizer and bot loops) and placeOrderForSignal..computeBacktestFromArgs (Binance and Coinbase order paths) bodies, Binance signing/cancel helpers'}


PROGRAM = r'''
import Control.Concurrent
import Control.Exception
import Data.IORef
import Trader.App.AsyncSafe (trySync)

routine :: (IO String -> IO (Either SomeException String)) -> IO [String]
routine capture = do
  steps <- newIORef []
  done <- newEmptyMVar
  tid <- forkIO $ (do
      r <- capture (threadDelay 500000 >> pure "entry")
      modifyIORef' steps (either (const "entry failed") id r :)
      modifyIORef' steps ("follow-up order" :)) `finally` putMVar done ()
  threadDelay 50000
  killThread tid
  takeMVar done
  reverse <$> readIORef steps

main :: IO ()
main = do
  legacy <- routine try
  fixed <- routine trySync
  print (legacy, fixed)
'''


def conformance():
    with tempfile.TemporaryDirectory(prefix='trader-async-stop-') as directory:
        path = Path(directory) / 'AsyncStop.hs'
        path.write_text(PROGRAM)
        out = subprocess.run(['runghc', '-i' + str(ROOT / 'haskell/app'), str(path)],
                             capture_output=True, text=True, timeout=120, check=True).stdout.strip()
    recorded = json.loads((ROOT / COUNTEREXAMPLE).read_text())
    require(out == recorded['observed'], 'compiled regression differs from the preserved counterexample: ' + out)
    return {'status': 'property_tested', 'observed': out}


def check_async_stop():
    return {'source': bind(), 'conformance': conformance(),
            'helperSha256': hashlib.sha256((ROOT / ASYNC).read_bytes()).hexdigest()}
