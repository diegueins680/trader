module Main (main) where

import Trader.Test.GracefulShutdown (workerRegistrySuite)

main :: IO ()
main = mapM_ snd workerRegistrySuite
