-- Exact registry/cancellation bodies extracted from e39fbfa8 for refutation only.
module Trader.App.GracefulShutdown where
import Control.Concurrent (ThreadId, forkIO, killThread, myThreadId, threadDelay)
import Control.Concurrent.MVar (MVar, modifyMVar_, newEmptyMVar, newMVar, putMVar, readMVar, swapMVar, takeMVar)
import Control.Exception (AsyncException, SomeException, displayException, finally, fromException, throwIO, try)
import Control.Monad (forM, void)
import System.IO (hPutStrLn, stderr)
import System.Timeout (timeout)
newtype WorkerRegistry = WorkerRegistry (MVar [(String, ThreadId)])

newWorkerRegistry :: IO WorkerRegistry
newWorkerRegistry = WorkerRegistry <$> newMVar []

supervisedWorkerCount :: WorkerRegistry -> IO Int
supervisedWorkerCount (WorkerRegistry workers) = length <$> readMVar workers

forkSupervisedWorker :: WorkerRegistry -> String -> IO () -> IO ThreadId
forkSupervisedWorker (WorkerRegistry workers) name action = do
    ready <- newEmptyMVar
    tid <-
        forkIO $ do
            takeMVar ready
            current <- myThreadId
            loop 0 `finally` modifyMVar_ workers (pure . filter ((/= current) . snd))
    modifyMVar_ workers (pure . ((name, tid) :))
    putMVar ready ()
    pure tid
  where
    restartDelayUs = 1000000

    loop :: Int -> IO ()
    loop restartCount = do
        result <- try action
        case result of
            Right () -> do
                hPutStrLn stderr (workerMessage "exited" restartCount Nothing)
                threadDelay restartDelayUs
                loop (restartCount + 1)
            Left ex ->
                case fromException ex :: Maybe AsyncException of
                    Just asyncEx -> throwIO asyncEx
                    Nothing -> do
                        hPutStrLn stderr (workerMessage "crashed" restartCount (Just ex))
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

stopSupervisedWorkersBounded :: Int -> WorkerRegistry -> IO Bool
stopSupervisedWorkersBounded timeoutUs (WorkerRegistry workers) = do
    registered <- swapMVar workers []
    stopThreadIdsBounded timeoutUs (map snd registered)

