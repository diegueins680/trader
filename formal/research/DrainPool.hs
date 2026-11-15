module Main (main) where

import Trader.Test.DrainPool (drainPoolSuite)

main :: IO ()
main = mapM_ snd drainPoolSuite
