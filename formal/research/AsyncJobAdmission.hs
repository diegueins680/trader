module Main (main) where

import System.Environment (getArgs)
import Trader.App.AsyncJobAdmission (admitSlot, releaseSlot)
import Trader.Test.AsyncJobAdmission (asyncJobAdmissionSuite)

main :: IO ()
main = do
    args <- getArgs
    case args of
        ["--cases"] -> interact (unlines . map (show . evaluate . read) . lines)
        ["--int-bound"] -> print (maxBound :: Int)
        _ -> mapM_ snd asyncJobAdmissionSuite
  where
    evaluate :: (Int, Int) -> ((Int, Bool), Int)
    evaluate (limit, count) = (admitSlot limit count, releaseSlot count)
