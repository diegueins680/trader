module Main (main) where

import Control.Monad (forM_)
import System.Environment (getArgs)
import Text.Read (readMaybe)

import Trader.App.GracefulShutdown (ShutdownPhase (..), shutdownBudget, shutdownRemainingUs)
import Trader.Test.GracefulShutdown (gracefulShutdownSuite)

main :: IO ()
main = do
    args <- getArgs
    case args of
        ["--cases"] -> do
            inputs <- lines <$> getContents
            forM_ inputs $ \input ->
                case readMaybe input of
                    Nothing -> ioError (userError "invalid conformance input")
                    Just (started, seconds, final, now) ->
                        print (shutdownRemainingUs (shutdownBudget started seconds) (if final then FinalCleanup else WorkCleanup) now)
        ["--int-bound"] -> print (maxBound :: Int)
        [] -> mapM_ snd gracefulShutdownSuite
        _ -> ioError (userError "invalid fixture mode")
