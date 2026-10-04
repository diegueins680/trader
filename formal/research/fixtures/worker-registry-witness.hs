module Main (main) where
import Control.Concurrent (threadDelay)
import Control.Concurrent.MVar
import Control.Exception (finally, uninterruptibleMask_)
import Trader.App.GracefulShutdown
main :: IO ()
main = do
  registry <- newWorkerRegistry
  entered <- newEmptyMVar
  release <- newEmptyMVar
  _ <- forkSupervisedWorker registry "blocked" (uninterruptibleMask_ (putMVar entered () >> takeMVar release))
  takeMVar entered
  first <- stopSupervisedWorkersBounded 20000 registry
  count <- supervisedWorkerCount registry
  second <- stopSupervisedWorkersBounded 20000 registry
  print ("unfinished retry",first,count,second)
  putMVar release ()
  threadDelay 50000
  fresh <- newWorkerRegistry
  stopped <- stopSupervisedWorkersBounded 20000 fresh
  _ <- forkSupervisedWorker fresh "after-close" (threadDelay 10000000)
  count2 <- supervisedWorkerCount fresh
  print ("post-close admission",stopped,count2)
  _ <- stopSupervisedWorkersBounded 500000 fresh
  finalizing <- newWorkerRegistry
  active <- newEmptyMVar
  cleanup <- newEmptyMVar
  finish <- newEmptyMVar
  _ <- forkSupervisedWorker finalizing "finalizer" ((putMVar active () >> threadDelay 10000000) `finally` uninterruptibleMask_ (putMVar cleanup () >> takeMVar finish))
  takeMVar active
  delivered <- stopSupervisedWorkersBounded 500000 finalizing
  takeMVar cleanup
  print ("unfinished finalizer acknowledged",delivered)
  putMVar finish ()
  threadDelay 50000
