module Main (main) where
import Control.Concurrent (killThread, threadDelay)
import Control.Concurrent.MVar
import Control.Exception (SomeException, try)
import Data.Maybe (isJust)
import System.Timeout (timeout)
import Trader.App.AsyncJobAdmission
main :: IO ()
main = do
    slots <- newJobSlots 1
    _ <- try (startBoundedJob slots (ioError (userError "prepare")) (const (pure ())) (\_ _ -> pure ())) :: IO (Either SomeException (Either Int ()))
    count <- runningJobSlots slots
    print ("prepare failure", count)
    other <- newJobSlots 1
    entered <- newEmptyMVar
    captured <- newEmptyMVar
    observed <- newEmptyMVar
    let action () = putMVar entered () >> threadDelay 10000000
        publish tid () = do
            putMVar captured tid
            ran <- isJust <$> timeout 1000000 (takeMVar entered)
            putMVar observed ran
            ioError (userError "publication")
    _ <- try (startBoundedJob other (pure ()) action publish) :: IO (Either SomeException (Either Int ()))
    ran <- takeMVar observed
    -- Allow the aborted child's nonblocking release to be scheduled.
    threadDelay 50000
    remaining <- runningJobSlots other
    print ("failed publication callback ran", ran, remaining)
    killThread =<< takeMVar captured
