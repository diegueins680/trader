module Trader.Optimization
  ( TuneObjective(..)
  , tuneObjectiveCode
  , parseTuneObjective
  , TuneConfig(..)
  , defaultTuneConfig
  , TuneStats(..)
  , bestFinalEquity
  , optimizeOperations
  , optimizeOperationsWithHL
  , optimizeOperationsWith
  , optimizeOperationsWithHLWith
  , sweepThreshold
  , sweepThresholdWithHL
  , sweepThresholdWith
  , sweepThresholdWithHLWith
  ) where

import Data.List (foldl', group, intercalate, sort)
import qualified Data.Vector as V

import Trader.Method (Method(..))
import Trader.Trading (BacktestResult(..), EnsembleConfig(..), StepMeta(..), simulateEnsembleVWithHL)
import Trader.Metrics (BacktestMetrics(..), computeMetrics)

data TuneObjective
  = TuneFinalEquity
  | TuneSharpe
  | TuneCalmar
  | TuneEquityDd
  | TuneEquityDdTurnover
  deriving (Eq, Show)

tuneObjectiveCode :: TuneObjective -> String
tuneObjectiveCode o =
  case o of
    TuneFinalEquity -> "final-equity"
    TuneSharpe -> "sharpe"
    TuneCalmar -> "calmar"
    TuneEquityDd -> "equity-dd"
    TuneEquityDdTurnover -> "equity-dd-turnover"

parseTuneObjective :: String -> Either String TuneObjective
parseTuneObjective raw =
  case normalize raw of
    "finalequity" -> Right TuneFinalEquity
    "final-equity" -> Right TuneFinalEquity
    "final_equity" -> Right TuneFinalEquity
    "sharpe" -> Right TuneSharpe
    "calmar" -> Right TuneCalmar
    "equitydd" -> Right TuneEquityDd
    "equity-dd" -> Right TuneEquityDd
    "equity_dd" -> Right TuneEquityDd
    "equityddturnover" -> Right TuneEquityDdTurnover
    "equity-dd-turnover" -> Right TuneEquityDdTurnover
    "equity_dd_turnover" -> Right TuneEquityDdTurnover
    _ ->
      Left
        ( "Invalid tune objective (expected one of: "
            ++ intercalate ", " (map tuneObjectiveCode [TuneFinalEquity, TuneSharpe, TuneCalmar, TuneEquityDd, TuneEquityDdTurnover])
            ++ ")"
        )
  where
    normalize = map (\c -> if c == '_' then '-' else c) . filter (/= ' ') . map toLower
    toLower c =
      if 'A' <= c && c <= 'Z' then toEnum (fromEnum c + 32) else c

data TuneConfig = TuneConfig
  { tcObjective :: !TuneObjective
  , tcPenaltyMaxDrawdown :: !Double
  , tcPenaltyTurnover :: !Double
  , tcPeriodsPerYear :: !Double
  , tcWalkForwardFolds :: !Int
  , tcMinRoundTrips :: !Int
  } deriving (Eq, Show)

data TuneStats = TuneStats
  { tsFoldCount :: !Int
  , tsFoldScores :: ![Double]
  , tsMeanScore :: !Double
  , tsStdScore :: !Double
  } deriving (Eq, Show)

defaultTuneConfig :: Double -> TuneConfig
defaultTuneConfig periodsPerYear =
  TuneConfig
    { tcObjective = TuneFinalEquity
    , tcPenaltyMaxDrawdown = 1.0
    , tcPenaltyTurnover = 0.0
    , tcPeriodsPerYear = max 1e-12 periodsPerYear
    , tcWalkForwardFolds = 1
    , tcMinRoundTrips = 0
    }

scoreBacktest :: TuneConfig -> BacktestResult -> Double
scoreBacktest cfg br =
  let m = computeMetrics (max 1e-12 (tcPeriodsPerYear cfg)) br
      finalEq = bmFinalEquity m
      maxDd = max 0 (bmMaxDrawdown m)
      turnover = max 0 (bmTurnover m)
      pDd = max 0 (tcPenaltyMaxDrawdown cfg)
      pTurn = max 0 (tcPenaltyTurnover cfg)
   in case tcObjective cfg of
        TuneFinalEquity -> finalEq
        TuneSharpe -> bmSharpe m
        TuneCalmar ->
          let denom = max 1e-12 maxDd
           in bmAnnualizedReturn m / denom
        TuneEquityDd -> finalEq - pDd * maxDd
        TuneEquityDdTurnover -> finalEq - pDd * maxDd - pTurn * turnover

mean :: [Double] -> Double
mean xs =
  if null xs
    then 0
    else sum xs / fromIntegral (length xs)

stddev :: [Double] -> Double
stddev xs =
  case xs of
    [] -> 0
    [_] -> 0
    _ ->
      let m = mean xs
          var = sum (map (\x -> (x - m) ** 2) xs) / fromIntegral (length xs - 1)
       in sqrt var

foldRanges :: Int -> Int -> [(Int, Int)]
foldRanges stepCount foldsReq =
  let steps = max 0 stepCount
      k0 = max 1 foldsReq
      k = max 1 (min steps k0)
      base = if k <= 0 then 0 else steps `div` k
      extra = if k <= 0 then 0 else steps `mod` k
      go i start =
        if i >= k
          then []
          else
            let len = base + if i < extra then 1 else 0
                end = start + len - 1
             in if len <= 0 then [] else (start, end) : go (i + 1) (end + 1)
   in go 0 0

bestFinalEquity :: BacktestResult -> Double
bestFinalEquity br =
  case brEquityCurve br of
    [] -> 1.0
    xs -> last xs

optimizeOperations :: EnsembleConfig -> [Double] -> [Double] -> [Double] -> Maybe [StepMeta] -> Either String (Method, Double, Double, BacktestResult)
optimizeOperations baseCfg prices kalPred lstmPred mMeta =
  case optimizeOperationsWith (defaultTuneConfig 1) baseCfg prices kalPred lstmPred mMeta of
    Left e -> Left e
    Right (m, openThr, closeThr, bt, _stats) -> Right (m, openThr, closeThr, bt)

optimizeOperationsWithHL :: EnsembleConfig -> [Double] -> [Double] -> [Double] -> [Double] -> [Double] -> Maybe [StepMeta] -> Either String (Method, Double, Double, BacktestResult)
optimizeOperationsWithHL baseCfg closes highs lows kalPred lstmPred mMeta =
  case optimizeOperationsWithHLWith (defaultTuneConfig 1) baseCfg closes highs lows kalPred lstmPred mMeta of
    Left e -> Left e
    Right (m, openThr, closeThr, bt, _stats) -> Right (m, openThr, closeThr, bt)

optimizeOperationsWith :: TuneConfig -> EnsembleConfig -> [Double] -> [Double] -> [Double] -> Maybe [StepMeta] -> Either String (Method, Double, Double, BacktestResult, TuneStats)
optimizeOperationsWith cfg baseCfg prices kalPred lstmPred mMeta =
  optimizeOperationsWithHLWith cfg baseCfg prices prices prices kalPred lstmPred mMeta

optimizeOperationsWithHLWith :: TuneConfig -> EnsembleConfig -> [Double] -> [Double] -> [Double] -> [Double] -> [Double] -> Maybe [StepMeta] -> Either String (Method, Double, Double, BacktestResult, TuneStats)
optimizeOperationsWithHLWith cfg baseCfg closes highs lows kalPred lstmPred mMeta =
  let eps = 1e-12
      methodRank m =
        case m of
          MethodBoth -> 2 :: Int
          MethodKalmanOnly -> 1
          MethodLstmOnly -> 0
      eval m =
        case sweepThresholdWithHLWith cfg m baseCfg closes highs lows kalPred lstmPred mMeta of
          Left e -> Left e
          Right (openThr, closeThr, bt, stats) ->
            Right (tsMeanScore stats, tsStdScore stats, m, openThr, closeThr, bt, stats)
      candidates = [MethodBoth, MethodKalmanOnly, MethodLstmOnly]
      results = map eval candidates
      evaluated = [v | Right v <- results]
      errors = [e | Left e <- results]
      pick (bestSc, bestStd, bestM, bestOpenThr, bestCloseThr, bestBt, bestStats) (sc, std, m, openThr, closeThr, bt, stats) =
        if sc > bestSc + eps
          then (sc, std, m, openThr, closeThr, bt, stats)
          else if abs (sc - bestSc) <= eps
            then
              if std < bestStd - eps
                then (sc, std, m, openThr, closeThr, bt, stats)
                else if abs (std - bestStd) <= eps
                  then
                    let r = methodRank m
                        bestR = methodRank bestM
                     in if r > bestR || (r == bestR && (openThr, closeThr) > (bestOpenThr, bestCloseThr))
                          then (sc, std, m, openThr, closeThr, bt, stats)
                          else (bestSc, bestStd, bestM, bestOpenThr, bestCloseThr, bestBt, bestStats)
                  else (bestSc, bestStd, bestM, bestOpenThr, bestCloseThr, bestBt, bestStats)
            else (bestSc, bestStd, bestM, bestOpenThr, bestCloseThr, bestBt, bestStats)
   in case evaluated of
        [] ->
          Left
            ( "optimizeOperations: no eligible candidates"
                ++ if null errors
                  then ""
                  else " (" ++ intercalate "; " errors ++ ")"
            )
        c : cs ->
          let (_, _, bestM, bestOpenThr, bestCloseThr, bestBt, bestStats) = foldl' pick c cs
           in Right (bestM, bestOpenThr, bestCloseThr, bestBt, bestStats)

sweepThreshold :: Method -> EnsembleConfig -> [Double] -> [Double] -> [Double] -> Maybe [StepMeta] -> Either String (Double, Double, BacktestResult)
sweepThreshold method baseCfg prices kalPred lstmPred mMeta =
  case sweepThresholdWith (defaultTuneConfig 1) method baseCfg prices kalPred lstmPred mMeta of
    Left e -> Left e
    Right (openThr, closeThr, bt, _stats) -> Right (openThr, closeThr, bt)

sweepThresholdWithHL :: Method -> EnsembleConfig -> [Double] -> [Double] -> [Double] -> [Double] -> [Double] -> Maybe [StepMeta] -> Either String (Double, Double, BacktestResult)
sweepThresholdWithHL method baseCfg closes highs lows kalPred lstmPred mMeta =
  case sweepThresholdWithHLWith (defaultTuneConfig 1) method baseCfg closes highs lows kalPred lstmPred mMeta of
    Left e -> Left e
    Right (openThr, closeThr, bt, _stats) -> Right (openThr, closeThr, bt)

sweepThresholdWith :: TuneConfig -> Method -> EnsembleConfig -> [Double] -> [Double] -> [Double] -> Maybe [StepMeta] -> Either String (Double, Double, BacktestResult, TuneStats)
sweepThresholdWith cfg method baseCfg prices kalPred lstmPred mMeta =
  sweepThresholdWithHLWith cfg method baseCfg prices prices prices kalPred lstmPred mMeta

sweepThresholdWithHLWith :: TuneConfig -> Method -> EnsembleConfig -> [Double] -> [Double] -> [Double] -> [Double] -> [Double] -> Maybe [StepMeta] -> Either String (Double, Double, BacktestResult, TuneStats)
sweepThresholdWithHLWith cfg method baseCfg closes highs lows kalPred lstmPred mMeta =
  let pricesV = V.fromList closes
      highsV = V.fromList highs
      lowsV = V.fromList lows
      n = V.length pricesV
      stepCount = n - 1
      eps = 1e-12
      baseOpenThreshold = max 0 (ecOpenThreshold baseCfg)
      baseCloseThreshold = max 0 (ecCloseThreshold baseCfg)
      maxCandidates = 60 :: Int
      minRoundTripsReq = max 0 (tcMinRoundTrips cfg)
      ineligibleScore = -1e18 :: Double

      downsample :: Int -> [Double] -> [Double]
      downsample k xs
        | k <= 0 = []
        | otherwise =
            let v = V.fromList xs
                n = V.length v
             in if n <= k
                  then xs
                  else
                    let denom = max 1 (k - 1)
                        pick i = (i * (n - 1)) `div` denom
                     in [v V.! pick i | i <- [0 .. k - 1]]

      kalV = V.fromList kalPred
      lstmV = V.fromList lstmPred

      metaV = V.fromList <$> mMeta
      metaUsed =
        case method of
          MethodLstmOnly -> Nothing
          _ -> metaV

      (kalUsedV, lstmUsedV) =
        case method of
          MethodBoth -> (kalV, lstmV)
          MethodKalmanOnly -> (kalV, kalV)
          MethodLstmOnly -> (lstmV, lstmV)

      predSources =
        case method of
          MethodBoth -> [kalV, lstmV]
          MethodKalmanOnly -> [kalV]
          MethodLstmOnly -> [lstmV]
      mags =
        [ v
        | t <- [0 .. stepCount - 1]
        , let prev = pricesV V.! t
        , prev /= 0
        , predsV <- predSources
        , let pred = predsV V.! t
        , let v = abs (pred / prev - 1)
        , not (isNaN v)
        , not (isInfinite v)
        ]

      uniqueSorted = map head . group . sort
      candidates0 = uniqueSorted (0 : map (\v -> max 0 (v - eps)) mags)
      candidates =
        uniqueSorted
          ( baseOpenThreshold
              : baseCloseThreshold
              : downsample maxCandidates candidates0
          )

      eligibleTune bt =
        if minRoundTripsReq <= 0
          then True
          else
            let m = computeMetrics (max 1e-12 (tcPeriodsPerYear cfg)) bt
             in bmRoundTrips m >= minRoundTripsReq

      eval openThr closeThr =
        let btCfg = baseCfg { ecOpenThreshold = openThr, ecCloseThreshold = closeThr }
            btFull = simulateEnsembleVWithHL btCfg 1 pricesV highsV lowsV kalUsedV lstmUsedV metaUsed
            eligible = eligibleTune btFull
            foldsReq = max 1 (tcWalkForwardFolds cfg)
            foldRs = foldRanges stepCount foldsReq
            foldScores =
              if not eligible
                then [ineligibleScore]
                else
                  if length foldRs <= 1
                    then [scoreBacktest cfg btFull]
                    else
                      [ let steps = t1 - t0 + 1
                            pricesF = V.slice t0 (steps + 1) pricesV
                            highsF = V.slice t0 (steps + 1) highsV
                            lowsF = V.slice t0 (steps + 1) lowsV
                            kalF = V.slice t0 steps kalUsedV
                            lstmF = V.slice t0 steps lstmUsedV
                            metaF = fmap (\mv -> V.slice t0 steps mv) metaUsed
                            btFold = simulateEnsembleVWithHL btCfg 1 pricesF highsF lowsF kalF lstmF metaF
                         in scoreBacktest cfg btFold
                      | (t0, t1) <- foldRs
                      , t1 >= t0
                      ]
            m = mean foldScores
            s = stddev foldScores
            stats =
              TuneStats
                { tsFoldCount = length foldScores
                , tsFoldScores = foldScores
                , tsMeanScore = m
                , tsStdScore = s
                }
         in (eligible, m, s, openThr, closeThr, btFull, stats)

      (baseEligible, baseMean, baseStd, baseOpenThr, baseCloseThr, baseBt, baseStats) = eval baseOpenThreshold baseCloseThreshold
      eqEps = 1e-12
      pick (bestEligible, bestMean, bestStd, bestOpenThr, bestCloseThr, bestBt, bestStats) (openThr, closeThr) =
        let (eligible, m, s, openThr', closeThr', bt, stats) = eval openThr closeThr
         in case (bestEligible, eligible) of
              (False, True) -> (True, m, s, openThr', closeThr', bt, stats)
              (True, False) -> (bestEligible, bestMean, bestStd, bestOpenThr, bestCloseThr, bestBt, bestStats)
              _ ->
                if m > bestMean + eqEps
                  then (eligible, m, s, openThr', closeThr', bt, stats)
                  else if abs (m - bestMean) <= eqEps
                    then
                      if s < bestStd - eqEps
                        then (eligible, m, s, openThr', closeThr', bt, stats)
                        else if abs (s - bestStd) <= eqEps && (openThr', closeThr') > (bestOpenThr, bestCloseThr)
                          then (eligible, m, s, openThr', closeThr', bt, stats)
                          else (bestEligible, bestMean, bestStd, bestOpenThr, bestCloseThr, bestBt, bestStats)
                    else (bestEligible, bestMean, bestStd, bestOpenThr, bestCloseThr, bestBt, bestStats)

      foldClose acc openThr = foldl' (\acc0 closeThr -> pick acc0 (openThr, closeThr)) acc candidates
      (bestEligible, _, _, bestOpenThr, bestCloseThr, bestBt, bestStats) =
        foldl' foldClose (baseEligible, baseMean, baseStd, baseOpenThr, baseCloseThr, baseBt, baseStats) candidates

      result = (bestOpenThr, bestCloseThr, bestBt, bestStats)
   in
    if n < 2
      then Left "sweepThreshold: need at least 2 prices"
      else if V.length highsV /= n || V.length lowsV /= n
        then Left "sweepThreshold: high/low series must match closes length"
        else if maybe False (\mv -> V.length mv < stepCount) metaUsed
          then Left "sweepThreshold: meta vector too short"
          else if minRoundTripsReq > 0 && not bestEligible
            then Left ("sweepThreshold: no eligible candidates (minRoundTrips=" ++ show minRoundTripsReq ++ ")")
            else
              case method of
                MethodBoth
                  | V.length kalV < stepCount -> Left "sweepThreshold: kalPred too short"
                  | V.length lstmV < stepCount -> Left "sweepThreshold: lstmPred too short"
                  | otherwise -> Right result
                MethodKalmanOnly
                  | V.length kalV < stepCount -> Left "sweepThreshold: kalPred too short"
                  | otherwise -> Right result
                MethodLstmOnly
                  | V.length lstmV < stepCount -> Left "sweepThreshold: lstmPred too short"
                  | otherwise -> Right result
