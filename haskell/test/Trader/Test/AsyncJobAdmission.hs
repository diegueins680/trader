{-# LANGUAGE LambdaCase #-}

module Trader.Test.AsyncJobAdmission (asyncJobAdmissionSuite) where

import Control.Concurrent (forkIO, killThread, threadDelay)
import Control.Concurrent.MVar (newEmptyMVar, putMVar, takeMVar)
import Control.Exception (AsyncException (ThreadKilled), MaskingState (Unmasked), SomeException, finally, getMaskingState, mask_, throwIO, try)
import Control.Monad (forM, forM_, void)
import Data.IORef (newIORef, readIORef, writeIORef)
import System.Timeout (timeout)
import Trader.App.AsyncJobAdmission (JobSlots, newJobSlots, runningJobSlots, startBoundedJob)

asyncJobAdmissionSuite :: [(String, IO ())]
asyncJobAdmissionSuite =
    [ ("preparation failure releases reservation", testPreparationFailure)
    , ("interrupted preparation releases reservation", testPreparationCancellation)
    , ("failed publication never executes callback", testPublicationFailure)
    , ("interrupted publication releases only its own reservation", testPublicationCancellation)
    , ("publication precedes unmasked callback and queue limits hold", testPublicationAndCapacity)
    , ("callback failure and cancellation release reservation", testCallbackFailure)
    , ("concurrent admission respects capacity", testConcurrentAdmission)
    ]

expect :: (Eq a, Show a) => String -> a -> a -> IO ()
expect label wanted actual =
    if wanted == actual then pure () else ioError (userError (label ++ ": expected " ++ show wanted ++ ", got " ++ show actual))

await :: IO a -> IO a
await action = do
    result <- timeout 3000000 action
    maybe (ioError (userError "async admission fixture timed out")) pure result

awaitCount :: JobSlots -> Int -> IO ()
awaitCount slots wanted = await loop
  where
    loop = do
        count <- runningJobSlots slots
        if count == wanted then pure () else threadDelay 1000 >> loop

expectFailure :: IO a -> IO ()
expectFailure action = do
    result <- try (void action) :: IO (Either SomeException ())
    case result of
        Left _ -> pure ()
        Right () -> ioError (userError "expected admission failure")

ignorePublication :: a -> b -> IO ()
ignorePublication _ _ = pure ()

testPreparationFailure :: IO ()
testPreparationFailure = do
    slots <- newJobSlots 1
    expectFailure (startBoundedJob slots (ioError (userError "prepare")) (const (pure ())) ignorePublication)
    expect "failed prepare releases synchronously" 0 =<< runningJobSlots slots

testPreparationCancellation :: IO ()
testPreparationCancellation = do
    slots <- newJobSlots 1
    entered <- newEmptyMVar
    block <- newEmptyMVar
    done <- newEmptyMVar
    parent <- forkIO (void (startBoundedJob slots (putMVar entered () >> takeMVar block) (const (pure ())) ignorePublication) `finally` putMVar done ())
    await (takeMVar entered)
    expect "preparation owns one slot" 1 =<< runningJobSlots slots
    killThread parent
    await (takeMVar done)
    expect "interrupted prepare releases" 0 =<< runningJobSlots slots

testPublicationFailure :: IO ()
testPublicationFailure = do
    slots <- newJobSlots 1
    ran <- newIORef False
    expectFailure (startBoundedJob slots (pure ()) (const (writeIORef ran True)) (\_ _ -> ioError (userError "publish")))
    awaitCount slots 0
    expect "unpublished callback absent" False =<< readIORef ran

testPublicationCancellation :: IO ()
testPublicationCancellation = do
    slots <- newJobSlots 2
    healthy <- newEmptyMVar
    release <- newEmptyMVar
    _ <- startBoundedJob slots (pure ()) (\() -> putMVar healthy () >> takeMVar release) ignorePublication
    await (takeMVar healthy)
    publishing <- newEmptyMVar
    block <- newEmptyMVar
    done <- newEmptyMVar
    ran <- newIORef False
    parent <- forkIO (void (startBoundedJob slots (pure ()) (const (writeIORef ran True)) (\_ _ -> putMVar publishing () >> takeMVar block)) `finally` putMVar done ())
    await (takeMVar publishing)
    expect "two reservations" 2 =<< runningJobSlots slots
    killThread parent
    await (takeMVar done)
    awaitCount slots 1
    expect "interrupted publication cannot execute" False =<< readIORef ran
    putMVar release ()
    awaitCount slots 0

testPublicationAndCapacity :: IO ()
testPublicationAndCapacity = do
    slots <- newJobSlots 0
    published <- newIORef False
    entered <- newEmptyMVar
    release <- newEmptyMVar
    let execute () = do
            visible <- readIORef published
            masking <- getMaskingState
            putMVar entered (visible, masking)
            takeMVar release
    accepted <- mask_ (startBoundedJob slots (pure ()) execute (\_ _ -> writeIORef published True))
    expect "initial admission" (Right ()) accepted
    expect "published and unmasked" (True, Unmasked) =<< await (takeMVar entered)
    rejected <- startBoundedJob slots (ioError (userError "full queue must not prepare")) (const (pure ())) ignorePublication
    expect "zero limit sanitizes to one" (Left 1 :: Either Int ()) rejected
    putMVar release ()
    awaitCount slots 0

testCallbackFailure :: IO ()
testCallbackFailure = do
    slots <- newJobSlots 1
    _ <- startBoundedJob slots (pure ()) (\() -> throwIO ThreadKilled) ignorePublication
    awaitCount slots 0
    entered <- newEmptyMVar
    identity <- newEmptyMVar
    _ <- startBoundedJob slots (pure ()) (\() -> putMVar entered () >> threadDelay 10000000) (\tid _ -> putMVar identity tid)
    await (takeMVar entered)
    killThread =<< takeMVar identity
    awaitCount slots 0

testConcurrentAdmission :: IO ()
testConcurrentAdmission = forM_ seeds $ \seed -> do
    slots <- newJobSlots 2
    outcomes <- forM [1, 3, 9, 27 :: Integer] $ \divisor -> do
        done <- newEmptyMVar
        _ <- forkIO $ do
            threadDelay (fromInteger ((seed `div` divisor) `mod` 3) * 1000)
            result <- try (startBoundedJob slots (threadDelay 1000) (const (threadDelay 5000)) ignorePublication) :: IO (Either SomeException (Either Int ()))
            putMVar done result
        pure done
    results <- await (mapM takeMVar outcomes)
    forM_ results $ \case
        Left ex -> ioError (userError (show ex))
        Right (Left count) -> expect "full queue count" 2 count
        Right (Right ()) -> pure ()
    observed <- runningJobSlots slots
    expect "bounded concurrent count" True (observed >= 0 && observed <= 2)
    awaitCount slots 0
  where
    seeds = take 32 (tail (iterate (\n -> (1664525 * n + 1013904223) `mod` 4294967296) 20261004))
