module Main where

import System.CPUTime (getCPUTime)
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
    t0 <- getCPUTime
    let (model, history) = trainLSTM cfg series
    -- Force the training result before taking the end timestamp. CPU time is
    -- provided by base and keeps this small diagnostic target dependency-free.
    sum (lmParams model) `seq` length history `seq` pure ()
    t1 <- getCPUTime
    let elapsed = fromIntegral (t1 - t0) / 1e12 :: Double
    putStrLn $ "Done: params=" ++ show (length (lmParams model)) ++ " epochs=" ++ show (length history) ++ " time=" ++ show elapsed ++ "s"
