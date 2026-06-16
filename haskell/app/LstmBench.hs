module Main where

import System.Clock
import Trader.LSTM

main :: IO ()
main = do
    let series = [sin (fromIntegral i / 10.0) | i <- [0 .. 999]]
        cfg =
            LSTMConfig
                { lcLookback = 20
                , lcHiddenSize = 16
                , lcEpochs = 10
                , lcLearningRate = 1e-3
                , lcAdamBeta1 = defaultLstmAdamBeta1
                , lcAdamBeta2 = defaultLstmAdamBeta2
                , lcAdamEps = defaultLstmAdamEps
                , lcValRatio = 0.3
                , lcPatience = 5
                , lcGradClip = Nothing
                , lcSeed = 42
                }
    putStrLn "Starting..."
    t0 <- getTime Monotonic
    let (model, history) = trainLSTM cfg series
    t1 <- getTime Monotonic
    let elapsed = fromIntegral (toNanoSecs (diffTimeSpec t1 t0)) / 1e9
    putStrLn $ "Done: params=" ++ show (length (lmParams model)) ++ " epochs=" ++ show (length history) ++ " time=" ++ show elapsed ++ "s"
