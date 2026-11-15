module Main (main) where
import System.Environment (getArgs)
import Trader.App.BacktestGate (timeoutMicroseconds)
import Trader.Test.BacktestGate (backtestGateSuite)
main :: IO ()
main = do
    args <- getArgs
    case args of
        ["--cases"] -> interact (unlines . map (show . timeoutMicroseconds . read) . lines)
        ["--int-bound"] -> print (maxBound :: Int)
        _ -> mapM_ snd backtestGateSuite
