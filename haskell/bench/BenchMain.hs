module Main where

import Control.Exception (evaluate)
import Control.Monad (when)
import qualified Data.Vector as V
import System.CPUTime (getCPUTime)
import System.Environment (getArgs)
import System.Exit (exitSuccess)
import Text.Printf (printf)
import Text.Read (readMaybe)

import Trader.Predictors.GBDT (GBDTModel (..), Stump (..), predictGBDT, trainGBDT)
import Trader.Predictors.HMM (HMM3 (..), HMMFilter (..), filterPosterior, fitHMM3)
import Trader.Predictors.Quantile (LinModel (..), QuantileModel (..), predictQuantiles, trainQuantileModel)
import Trader.Predictors.TCN (TCNModel (..), predictTCN, trainTCN)

data BenchConfig = BenchConfig
    { bcSamples :: !Int
    , bcFeatures :: !Int
    , bcTrees :: !Int
    , bcQuantileEpochs :: !Int
    , bcHmmIters :: !Int
    , bcLookback :: !Int
    , bcPredictIters :: !Int
    }
    deriving (Eq, Show)

defaultConfig :: BenchConfig
defaultConfig =
    BenchConfig
        { bcSamples = 5000
        , bcFeatures = 16
        , bcTrees = 50
        , bcQuantileEpochs = 50
        , bcHmmIters = 10
        , bcLookback = 32
        , bcPredictIters = 5000
        }

main :: IO ()
main = do
    args <- getArgs
    when ("--help" `elem` args) $ do
        putStrLn usage
        exitSuccess
    let cfg = normalizeConfig (parseArgs args)
    putStrLn ("Bench config: " ++ show cfg)
    benchGBDT cfg
    benchQuantile cfg
    benchTCN cfg
    benchHMM cfg

usage :: String
usage =
    unlines
        [ "Usage: cabal run trader-bench -- [options]"
        , ""
        , "Options:"
        , "  --samples N         Number of samples (default 5000)"
        , "  --features N        Feature dimension (default 16)"
        , "  --trees N           GBDT tree count (default 50)"
        , "  --quantile-epochs N Quantile epochs (default 50)"
        , "  --hmm-iters N       HMM EM iterations (default 10)"
        , "  --lookback N        TCN lookback bars (default 32)"
        , "  --predict-iters N   Prediction iterations (default 5000)"
        ]

parseArgs :: [String] -> BenchConfig
parseArgs = go defaultConfig
  where
    go cfg [] = cfg
    go cfg ("--samples" : v : rest) = go cfg{bcSamples = readInt "--samples" v} rest
    go cfg ("--features" : v : rest) = go cfg{bcFeatures = readInt "--features" v} rest
    go cfg ("--trees" : v : rest) = go cfg{bcTrees = readInt "--trees" v} rest
    go cfg ("--quantile-epochs" : v : rest) = go cfg{bcQuantileEpochs = readInt "--quantile-epochs" v} rest
    go cfg ("--hmm-iters" : v : rest) = go cfg{bcHmmIters = readInt "--hmm-iters" v} rest
    go cfg ("--lookback" : v : rest) = go cfg{bcLookback = readInt "--lookback" v} rest
    go cfg ("--predict-iters" : v : rest) = go cfg{bcPredictIters = readInt "--predict-iters" v} rest
    go _ (flag : _) = error ("Unknown flag: " ++ flag)

readInt :: String -> String -> Int
readInt label raw =
    case readMaybe raw of
        Just n -> n
        Nothing -> error ("Invalid " ++ label ++ " value: " ++ raw)

normalizeConfig :: BenchConfig -> BenchConfig
normalizeConfig cfg =
    cfg
        { bcSamples = max 1 (bcSamples cfg)
        , bcFeatures = max 1 (bcFeatures cfg)
        , bcTrees = max 1 (bcTrees cfg)
        , bcQuantileEpochs = max 1 (bcQuantileEpochs cfg)
        , bcHmmIters = max 1 (bcHmmIters cfg)
        , bcLookback = max 1 (bcLookback cfg)
        , bcPredictIters = max 1 (bcPredictIters cfg)
        }

timeIt :: String -> IO a -> IO a
timeIt label action = do
    start <- getCPUTime
    result <- action
    end <- getCPUTime
    let elapsed = fromIntegral (end - start) / 1e12 :: Double
    printf "%-18s %.3fs\n" label elapsed
    pure result

benchGBDT :: BenchConfig -> IO ()
benchGBDT cfg = do
    let dataset = mkDataset (bcSamples cfg) (bcFeatures cfg)
        inputs = take (bcPredictIters cfg) (cycle (map fst dataset))
    model <- timeIt "GBDT train" $ do
        let m = trainGBDT (bcTrees cfg) 0.1 dataset
        evaluate (sum (map stThreshold (gmStumps m)))
        pure m
    _ <- timeIt "GBDT predict" $ do
        let preds = map (fst . predictGBDT model) inputs
        evaluate (sum preds)
    pure ()

benchQuantile :: BenchConfig -> IO ()
benchQuantile cfg = do
    let dataset = mkDataset (bcSamples cfg) (bcFeatures cfg)
        inputs = take (bcPredictIters cfg) (cycle (map fst dataset))
    model <- timeIt "Quantile train" $ do
        let m = trainQuantileModel (bcQuantileEpochs cfg) 1e-3 1e-4 dataset
            sumW =
                sum (lmW (qm10 m))
                    + sum (lmW (qm50 m))
                    + sum (lmW (qm90 m))
        evaluate sumW
        pure m
    _ <- timeIt "Quantile predict" $ do
        let preds =
                map
                    (\x -> maybe 0 (\(lo, mid, hi, _, _) -> lo + mid + hi) (predictQuantiles model x))
                    inputs
        evaluate (sum preds)
    pure ()

benchTCN :: BenchConfig -> IO ()
benchTCN cfg = do
    let priceCount = max (bcSamples cfg) (bcLookback cfg + 2)
        prices = mkPriceSeries priceCount
        targets = mkTargets prices
        lookback = min (bcLookback cfg) (priceCount - 2)
        predictIdxs =
            take
                (bcPredictIters cfg)
                (cycle [lookback .. V.length prices - 1])
    model <- timeIt "TCN train" $ do
        let m = trainTCN lookback prices targets
        evaluate (sum (tmWeights m))
        pure m
    _ <- timeIt "TCN predict" $ do
        let preds =
                map
                    (\t -> maybe 0 fst (predictTCN model prices t))
                    predictIdxs
        evaluate (sum preds)
    pure ()

benchHMM :: BenchConfig -> IO ()
benchHMM cfg = do
    let obs = mkObs (bcSamples cfg)
    model <- timeIt "HMM fit" $ do
        let m = fitHMM3 (bcHmmIters cfg) obs
        evaluate (sum (hmmMu m) + sum (hmmVar m))
        pure m
    _ <- timeIt "HMM filter" $ do
        let filt = filterPosterior model obs
        evaluate (sum (hfPosterior filt))
    pure ()

mkDataset :: Int -> Int -> [([Double], Double)]
mkDataset samples dims =
    let weights = [fromIntegral (j + 1) / fromIntegral dims | j <- [0 .. dims - 1]]
        featRow i =
            [ sin (fromIntegral (i + j) * 0.01)
                + cos (fromIntegral (i * (j + 1)) * 0.005)
            | j <- [0 .. dims - 1]
            ]
        target feats = sum (zipWith (*) weights feats) / fromIntegral dims
     in [let feats = featRow i in (feats, target feats) | i <- [0 .. samples - 1]]

mkPriceSeries :: Int -> V.Vector Double
mkPriceSeries n =
    V.generate n $ \i ->
        let x = fromIntegral i
         in 100 + 0.1 * sin (x * 0.01) + 0.05 * cos (x * 0.02)

mkTargets :: V.Vector Double -> [(Int, Double)]
mkTargets prices =
    [ (t, retAt t)
    | t <- [1 .. V.length prices - 1]
    ]
  where
    retAt t =
        let p0 = prices V.! (t - 1)
            p1 = prices V.! t
         in if p0 == 0 then 0 else p1 / p0 - 1

mkObs :: Int -> [Double]
mkObs n =
    [ sin (fromIntegral i * 0.01) + cos (fromIntegral i * 0.02)
    | i <- [0 .. n - 1]
    ]
