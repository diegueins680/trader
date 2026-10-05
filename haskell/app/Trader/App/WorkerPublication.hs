module Trader.App.WorkerPublication (
    WorkerPlan (..),
    publishWorker,
) where

import Control.Concurrent (ThreadId, forkIOWithUnmask)
import Control.Concurrent.MVar (MVar, modifyMVarMasked, newEmptyMVar, readMVar, tryPutMVar)
import Control.Exception (evaluate, mask, onException)
import Control.Monad (void, when)

{- | Preparation either preserves the state or supplies a worker and a pure
publication using its real thread identity. Publication is forced to WHNF.
-}
data WorkerPlan state result
    = Retain result
    | Launch (IO ()) (ThreadId -> (state, result))

{- | A child cannot execute before its state has been published. Failed
preparation/publication restores the old MVar and releases a rejecting gate;
cleanup never waits for cancellation delivery to an arbitrary worker action.
This does not supervise the action after successful publication.
-}
publishWorker :: MVar state -> (state -> IO (WorkerPlan state result)) -> IO result
publishWorker cell prepare = mask $ \restore -> do
    gate <- newEmptyMVar
    result <-
        modifyMVarMasked
            cell
            ( \state -> do
                plan <- restore (prepare state)
                case plan of
                    Retain unchanged -> pure (state, unchanged)
                    Launch action publish -> do
                        tid <- forkIOWithUnmask $ \unmask -> do
                            accepted <- readMVar gate
                            when accepted (unmask action)
                        (next, outcome) <- evaluate (publish tid)
                        nextState <- evaluate next
                        nextOutcome <- evaluate outcome
                        pure (nextState, nextOutcome)
            )
            `onException` void (tryPutMVar gate False)
    void (tryPutMVar gate True)
    pure result
