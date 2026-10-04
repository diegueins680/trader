-- Control-slice adapter of Main.startJob at 02555718. Preparation replaces job
-- ID/result allocation and persistence; publication replaces HM.insert. This
-- executes the inherited reserve/fork/finally/publish ordering, not HTTP routing.
module Trader.App.AsyncJobAdmission (JobSlots, newJobSlots, runningJobSlots, startBoundedJob) where
import Control.Concurrent (ThreadId, forkIO)
import Control.Concurrent.MVar
import Control.Exception (finally)
data JobSlots = JobSlots Int (MVar Int)
newJobSlots :: Int -> IO JobSlots
newJobSlots limit = JobSlots (max 1 limit) <$> newMVar 0
runningJobSlots :: JobSlots -> IO Int
runningJobSlots (JobSlots _ running) = readMVar running
startBoundedJob :: JobSlots -> IO a -> (a -> IO ()) -> (ThreadId -> a -> IO ()) -> IO (Either Int a)
startBoundedJob (JobSlots limit slots) prepare execute publish = do
    (running, ok) <- modifyMVar slots $ \n ->
        if n >= limit then pure (n, (n, False)) else pure (n + 1, (n, True))
    if not ok then pure (Left running) else do
        payload <- prepare
        tid <- forkIO (execute payload `finally` modifyMVar_ slots (pure . max 0 . subtract 1))
        publish tid payload
        pure (Right payload)
