module Trader.App.BacktestGate where
import Control.Concurrent (threadDelay)
import Control.Concurrent.MVar
import Control.Exception
import System.Timeout (timeout)
data BacktestGate = BacktestGate
    { btRunning :: !(MVar Int)
    , btMaxRunning :: !Int
    , btTimeoutSec :: !Int
    }

data BacktestFailure
    = BacktestBusy
    | BacktestTimedOut
    | BacktestException SomeException
    deriving (Show)

newBacktestGate :: Int -> Int -> IO BacktestGate
newBacktestGate maxRunning timeoutSec = do
    running <- newMVar 0
    pure BacktestGate{btRunning = running, btMaxRunning = max 1 maxRunning, btTimeoutSec = max 1 timeoutSec}

runBacktestWithGate :: BacktestGate -> IO a -> IO (Either BacktestFailure a)
runBacktestWithGate gate action = do
    let maxRunning = max 1 (btMaxRunning gate)
    acquired <-
        modifyMVar (btRunning gate) $ \n ->
            if n >= maxRunning
                then pure (n, False)
                else pure (n + 1, True)
    if not acquired
        then pure (Left BacktestBusy)
        else do
            let release = modifyMVar_ (btRunning gate) (pure . max 0 . subtract 1)
            result <- timeout (btTimeoutSec gate * 1000000) (try action) `finally` release
            case result of
                Nothing -> pure (Left BacktestTimedOut)
                Just (Left ex) -> pure (Left (BacktestException ex))
                Just (Right v) -> pure (Right v)

runBacktestWithGateWait :: BacktestGate -> IO a -> IO (Either BacktestFailure a)
runBacktestWithGateWait gate action = do
    let maxRunning = max 1 (btMaxRunning gate)
        tryAcquire =
            modifyMVar (btRunning gate) $ \n ->
                if n >= maxRunning
                    then pure (n, False)
                    else pure (n + 1, True)
        waitLoop = do
            acquired <- tryAcquire
            if acquired
                then pure ()
                else do
                    threadDelay 1000000
                    waitLoop
    waitLoop
    let release = modifyMVar_ (btRunning gate) (pure . max 0 . subtract 1)
    result <- timeout (btTimeoutSec gate * 1000000) (try action) `finally` release
    case result of
        Nothing -> pure (Left BacktestTimedOut)
        Just (Left ex) -> pure (Left (BacktestException ex))
        Just (Right v) -> pure (Right v)


backtestRunningCount :: BacktestGate -> IO Int
backtestRunningCount = readMVar . btRunning

timeoutMicroseconds :: Int -> Int
timeoutMicroseconds seconds = seconds * 1000000
