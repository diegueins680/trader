module Main (main) where

import Trader.Test.AdmissionProgress (admissionProgressSuite)

main :: IO ()
main = mapM_ snd admissionProgressSuite
