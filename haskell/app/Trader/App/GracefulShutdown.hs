{-# LANGUAGE OverloadedStrings #-}

module Trader.App.GracefulShutdown (
    DrainController,
    ShutdownBudget,
    ShutdownPhase (..),
    shutdownBudget,
    shutdownRemainingUs,
    runShutdownStep,
    WorkerRegistry,
    beginDrain,
    forkSupervisedWorker,
    isDraining,
    newDrainController,
    unlessDraining,
    newWorkerRegistry,
    runCleanupStepBounded,
    shouldRejectDuringDrain,
    stopSupervisedWorkersBounded,
    stopThreadIdsBounded,
    supervisedWorkerCount,
) where

import Control.Concurrent (ThreadId, forkIO, forkIOWithUnmask, killThread, threadDelay)
import Control.Concurrent.MVar (MVar, isEmptyMVar, modifyMVar, modifyMVar_, newEmptyMVar, newMVar, putMVar, readMVar, takeMVar)
import Control.Concurrent.STM (STM, TVar, atomically, newTVarIO, readTVar, readTVarIO, writeTVar)
import Control.Exception (AsyncException, SomeException, displayException, finally, fromException, mask_, throwIO, try)
import Control.Monad (filterM, forM, unless, void)
import Data.ByteString (ByteString)
import Data.Text (Text)
import GHC.Clock (getMonotonicTimeNSec)
import System.IO (hPutStrLn, stderr)
import System.Timeout (timeout)

newtype DrainController = DrainController (TVar Bool)

newtype WorkerRegistry = WorkerRegistry (MVar WorkerRegistryState)

data WorkerRegistryState = WorkerRegistryState
    { registryClosed :: !Bool
    , registryWorkers :: [SupervisedWorker]
    }

data SupervisedWorker = SupervisedWorker
    { workerThread :: !ThreadId
    , workerFinished :: !(MVar ())
    , workerCancelRequested :: !(MVar Bool)
    }

newDrainController :: IO DrainController
newDrainController = DrainController <$> newTVarIO False

beginDrain :: DrainController -> IO Bool
beginDrain (DrainController ref) = atomically $ do
    draining <- readTVar ref
    writeTVar ref True
    pure (not draining)

isDraining :: DrainController -> IO Bool
isDraining (DrainController ref) = readTVarIO ref

-- Compose the guard and resource reservation in the same transaction. A readiness
-- snapshot alone is not an admission permit. The guarded action cannot perform IO.
unlessDraining :: DrainController -> STM a -> STM (Maybe a)
unlessDraining (DrainController ref) action = do
    draining <- readTVar ref
    if draining then pure Nothing else Just <$> action

-- Keep polling and cancellation available while draining, but reject endpoints
-- that can launch expensive compute, orders, bots, or optimizer processes.
shouldRejectDuringDrain :: ByteString -> [Text] -> Bool
shouldRejectDuringDrain method path =
    method == "POST" && normalizedPath `elem` workStartingPaths
  where
    normalizedPath =
        case path of
            "api" : rest -> rest
            _ -> path
    workStartingPaths =
        [ ["signal"]
        , ["signal", "async"]
        , ["trade"]
        , ["trade", "async"]
        , ["backtest"]
        , ["backtest", "async"]
        , ["binance", "positions", "close"]
        , ["bot", "start"]
        , ["optimizer", "run"]
        ]

newWorkerRegistry :: IO WorkerRegistry
newWorkerRegistry = WorkerRegistry <$> newMVar (WorkerRegistryState False [])

unfinishedWorkers :: [SupervisedWorker] -> IO [SupervisedWorker]
unfinishedWorkers = filterM (isEmptyMVar . workerFinished)

supervisedWorkerCount :: WorkerRegistry -> IO Int
supervisedWorkerCount (WorkerRegistry workers) = do
    snapshot <- readMVar workers
    length <$> unfinishedWorkers (registryWorkers snapshot)

-- Nothing rejects a closed registry without creating a thread. Publication is
-- masked and holds the same lock that the child reads before its first action.
forkSupervisedWorker :: WorkerRegistry -> String -> IO () -> IO (Maybe ThreadId)
forkSupervisedWorker (WorkerRegistry workers) name action = mask_ $
    modifyMVar workers $ \state ->
        if registryClosed state
            then pure (state, Nothing)
            else do
                retained <- unfinishedWorkers (registryWorkers state)
                finished <- newEmptyMVar
                requested <- newMVar False
                tid <- forkIOWithUnmask $ \unmask -> unmask (loop 0) `finally` putMVar finished ()
                let entry = SupervisedWorker tid finished requested
                pure (state{registryWorkers = entry : retained}, Just tid)
  where
    restartDelayUs = 1000000

    loop :: Int -> IO ()
    loop restartCount = do
        closed <- registryClosed <$> readMVar workers
        unless closed $ do
            result <- try action
            case result of
                Right () -> restart "exited" restartCount Nothing
                Left ex ->
                    case fromException ex :: Maybe AsyncException of
                        Just asyncEx -> throwIO asyncEx
                        Nothing -> restart "crashed" restartCount (Just ex)

    restart :: String -> Int -> Maybe SomeException -> IO ()
    restart outcome restartCount problem = do
        hPutStrLn stderr (workerMessage outcome restartCount problem)
        threadDelay restartDelayUs
        loop (restartCount + 1)

    workerMessage :: String -> Int -> Maybe SomeException -> String
    workerMessage outcome restartCount mException =
        "Background worker '"
            ++ name
            ++ "' "
            ++ outcome
            ++ maybe "" ((": " ++) . displayException) mException
            ++ "; restarting (count="
            ++ show (restartCount + 1)
            ++ ")."

-- Run cleanup in a separate thread so an uninterruptible foreign call cannot
-- make the caller exceed its deadline. A timed-out cleanup thread receives a
-- best-effort asynchronous cancellation without waiting for delivery.
runCleanupStepBounded :: Int -> IO () -> IO Bool
runCleanupStepBounded timeoutUs action = do
    done <- newEmptyMVar
    tid <-
        forkIO $ do
            result <- try action :: IO (Either SomeException ())
            putMVar done result
    result <- timeout (max 1 timeoutUs) (takeMVar done)
    case result of
        Just (Right ()) -> pure True
        Just (Left _) -> pure False
        Nothing -> do
            void (forkIO (killThread tid))
            pure False

stopThreadIdsBounded :: Int -> [ThreadId] -> IO Bool
stopThreadIdsBounded timeoutUs tids = do
    completions <-
        forM tids $ \tid -> do
            done <- newEmptyMVar
            void $ forkIO (killThread tid `finally` putMVar done ())
            pure done
    isJustResult <- timeout (max 1 timeoutUs) (mapM_ takeMVar completions)
    pure $ case isJustResult of
        Just () -> True
        Nothing -> False

-- Retrying shares both the cancellation request and the completion cell. A
-- delivered ThreadKilled is not evidence that callback finalizers have finished.
requestWorkerCancellation :: SupervisedWorker -> IO ()
requestWorkerCancellation worker = mask_ $
    modifyMVar_ (workerCancelRequested worker) $ \requested -> do
        unless requested (void (forkIO (killThread (workerThread worker))))
        pure True

stopSupervisedWorkersBounded :: Int -> WorkerRegistry -> IO Bool
stopSupervisedWorkersBounded timeoutUs (WorkerRegistry workers) = do
    completed <- timeout (max 1 timeoutUs) $ do
        captured <- modifyMVar workers $ \state -> do
            retained <- unfinishedWorkers (registryWorkers state)
            pure (WorkerRegistryState True retained, retained)
        mapM_ requestWorkerCancellation captured
        mapM_ (readMVar . workerFinished) captured
    pure $ case completed of
        Just () -> True
        Nothing -> False

-- Exact nanoseconds keep deadline construction independent of Int overflow.
-- Constructors stay private so callers cannot fabricate inconsistent deadlines.
data ShutdownBudget = ShutdownBudget !Integer !Integer !Integer

data ShutdownPhase = WorkCleanup | FinalCleanup
    deriving (Eq, Show)

shutdownBudget :: Integer -> Int -> ShutdownBudget
shutdownBudget started seconds =
    let total = max 0 (toInteger seconds) * 1000000000
        reserve = min 2000000000 (total `div` 4)
     in ShutdownBudget started (started + total - reserve) (started + total)

shutdownRemainingUs :: ShutdownBudget -> ShutdownPhase -> Integer -> Int
shutdownRemainingUs (ShutdownBudget started workDeadline finalDeadline) phase now
    | started < 0 || now < started = 0
    | otherwise = fromInteger (min (toInteger (maxBound :: Int)) (max 0 ((deadline - now) `div` 1000)))
  where
    deadline = case phase of
        WorkCleanup -> workDeadline
        FinalCleanup -> finalDeadline

-- Success means the action acknowledged completion before its deadline. It is
-- not proof that uninterruptible child threads have terminated after cancellation.
runShutdownStep :: ShutdownBudget -> ShutdownPhase -> (Int -> IO ()) -> IO Bool
runShutdownStep budget phase action = do
    now <- toInteger <$> getMonotonicTimeNSec
    let remaining = shutdownRemainingUs budget phase now
    if remaining <= 0
        then pure False
        else do
            completed <- runCleanupStepBounded remaining (action remaining)
            ended <- toInteger <$> getMonotonicTimeNSec
            pure (completed && shutdownRemainingUs budget phase ended > 0)
