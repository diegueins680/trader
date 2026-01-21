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
import Trader.Trading (BacktestResult(..), EnsembleConfig(..), StepMeta(..), simulateEnsembleVWithHLChecked)
import Trader.Metrics (BacktestMetrics(..), computeMetrics)

data TuneObjective
  = TuneFinalEquity
  | TuneAnnualizedEquity
  | TuneSharpe
  | TuneCalmar
  | TuneEquityDd
  | TuneEquityDdTurnover
  deriving (Eq, Show)

tuneObjectiveCode :: TuneObjective -> String
tuneObjectiveCode o =
  case o of
    TuneFinalEquity -> "final-equity"
    TuneAnnualizedEquity -> "annualized-equity"
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
    "annualizedequity" -> Right TuneAnnualizedEquity
    "annualized-equity" -> Right TuneAnnualizedEquity
    "annualized_equity" -> Right TuneAnnualizedEquity
    "annualizedreturn" -> Right TuneAnnualizedEquity
    "annualized-return" -> Right TuneAnnualizedEquity
    "annualized_return" -> Right TuneAnnualizedEquity
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
            ++ intercalate
              ", "
              (map tuneObjectiveCode [TuneAnnualizedEquity, TuneFinalEquity, TuneSharpe, TuneCalmar, TuneEquityDd, TuneEquityDdTurnover])
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
  , tcStressVolMultiplier :: !Double
  , tcStressShock :: !Double
  , tcStressWeight :: !Double
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
    , tcStressVolMultiplier = 1.0
    , tcStressShock = 0.0
    , tcStressWeight = 0.0
    }

scoreBacktest :: TuneConfig -> BacktestResult -> Double
scoreBacktest cfg br =
  let ppy = max 1e-12 (tcPeriodsPerYear cfg)
      m = computeMetrics ppy br
      baseScore = scoreObjective cfg m
      stressWeight = max 0 (tcStressWeight cfg)
      stressScore =
        if stressWeight <= 0
          then baseScore
          else
            let mult = max 0 (tcStressVolMultiplier cfg)
                shock = tcStressShock cfg
                eq = brEquityCurve br
                stressEq = stressEquityCurve mult shock eq
                brStress = br { brEquityCurve = stressEq }
                mStress = computeMetrics ppy brStress
             in scoreObjective cfg mStress
      penalty = max 0 (baseScore - stressScore)
   in baseScore - stressWeight * penalty

scoreObjective :: TuneConfig -> BacktestMetrics -> Double
scoreObjective cfg m =
  let finalEq = bmFinalEquity m
      maxDd = max 0 (bmMaxDrawdown m)
      turnover = max 0 (bmTurnover m)
      pDd = max 0 (tcPenaltyMaxDrawdown cfg)
      pTurn = max 0 (tcPenaltyTurnover cfg)
   in case tcObjective cfg of
        TuneFinalEquity -> finalEq
        TuneAnnualizedEquity -> bmAnnualizedReturn m
        TuneSharpe -> bmSharpe m
        TuneCalmar ->
          if maxDd <= 0
            then bmAnnualizedReturn m
            else
              let denom = max 1e-12 maxDd
               in bmAnnualizedReturn m / denom
        TuneEquityDd -> finalEq - pDd * maxDd
        TuneEquityDdTurnover -> finalEq - pDd * maxDd - pTurn * turnover

stressEquityCurve :: Double -> Double -> [Double] -> [Double]
stressEquityCurve volMult shock eq =
  let rets = returnsFromEquity eq
      step acc r =
        let r' = r * volMult + shock
            next = acc * (1 + r')
            next' = if isNaN next || isInfinite next || next < 0 then 0 else next
         in next'
   in case rets of
        [] -> eq
        _ -> scanl step 1.0 rets

returnsFromEquity :: [Double] -> [Double]
returnsFromEquity eq =
  case eq of
    [] -> []
    [_] -> []
    _ -> zipWith ret eq (tail eq)
  where
    bad x = isNaN x || isInfinite x
    ret a b =
      if bad a || bad b || a <= 0
        then 0
        else b / a - 1

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

clamp01 :: Double -> Double
clamp01 x = max 0 (min 1 x)

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
  case optimizeOperationsWith (defaultTuneConfig (ecPeriodsPerYear baseCfg)) baseCfg prices kalPred lstmPred mMeta of
    Left e -> Left e
    Right (m, openThr, closeThr, bt, _stats) -> Right (m, openThr, closeThr, bt)

optimizeOperationsWithHL :: EnsembleConfig -> [Double] -> [Double] -> [Double] -> [Double] -> [Double] -> Maybe [StepMeta] -> Either String (Method, Double, Double, BacktestResult)
optimizeOperationsWithHL baseCfg closes highs lows kalPred lstmPred mMeta =
  case optimizeOperationsWithHLWith (defaultTuneConfig (ecPeriodsPerYear baseCfg)) baseCfg closes highs lows kalPred lstmPred mMeta of
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
          MethodBoth -> 3 :: Int
          MethodRouter -> 3
          MethodBlend -> 2
          MethodKalmanOnly -> 1
          MethodLstmOnly -> 0
      eval m =
        case sweepThresholdWithHLWith cfg m baseCfg closes highs lows kalPred lstmPred mMeta of
          Left e -> Left e
          Right (openThr, closeThr, bt, stats) ->
            Right (tsMeanScore stats, tsStdScore stats, m, openThr, closeThr, bt, stats)
      candidates = [MethodBoth, MethodRouter, MethodBlend, MethodKalmanOnly, MethodLstmOnly]
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
  case sweepThresholdWith (defaultTuneConfig (ecPeriodsPerYear baseCfg)) method baseCfg prices kalPred lstmPred mMeta of
    Left e -> Left e
    Right (openThr, closeThr, bt, _stats) -> Right (openThr, closeThr, bt)

sweepThresholdWithHL :: Method -> EnsembleConfig -> [Double] -> [Double] -> [Double] -> [Double] -> [Double] -> Maybe [StepMeta] -> Either String (Double, Double, BacktestResult)
sweepThresholdWithHL method baseCfg closes highs lows kalPred lstmPred mMeta =
  case sweepThresholdWithHLWith (defaultTuneConfig (ecPeriodsPerYear baseCfg)) method baseCfg closes highs lows kalPred lstmPred mMeta of
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
      minEdge = max 0 (ecMinEdge baseCfg)
      maxCandidates = 60 :: Int
      minRoundTripsReq = max 0 (tcMinRoundTrips cfg)
      ineligibleScore = -1e18 :: Double
      routerLookback = max 2 (ecRouterLookback baseCfg)
      routerMinScore = clamp01 (ecRouterMinScore baseCfg)
      routerScorePnlWeight = clamp01 (ecRouterScorePnlWeight baseCfg)
      perSideCost =
        let fee = max 0 (ecFee baseCfg)
            slip = max 0 (ecSlippage baseCfg)
            spr = max 0 (ecSpread baseCfg)
            c = fee + slip + spr / 2
         in min 0.999999 (max 0 c)
      roundTripCost = min 0.999999 (2 * perSideCost)

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
          MethodRouter -> metaV
          _ -> metaV

      blendWeight = clamp01 (ecBlendWeight baseCfg)
      blendV = V.zipWith (\k l -> blendWeight * k + (1 - blendWeight) * l) kalV lstmV

      (kalUsedV0, lstmUsedV0) =
        case method of
          MethodBoth -> (kalV, lstmV)
          MethodRouter -> (kalV, lstmV)
          MethodBlend -> (blendV, blendV)
          MethodKalmanOnly -> (kalV, kalV)
          MethodLstmOnly -> (lstmV, lstmV)

      validationError =
        if n < 2
          then Just "sweepThreshold: need at least 2 prices"
          else if V.length highsV /= n || V.length lowsV /= n
            then Just "sweepThreshold: high/low series must match closes length"
            else if maybe False (\mv -> V.length mv < stepCount) metaUsed
              then Just "sweepThreshold: meta vector too short"
              else
                case method of
                  MethodBoth
                    | V.length kalV < stepCount ->
                        Just
                          ( "sweepThreshold: kalPred has length "
                              ++ show (V.length kalV)
                              ++ " but needs at least "
                              ++ show stepCount
                          )
                    | V.length lstmV < stepCount ->
                        Just
                          ( "sweepThreshold: lstmPred has length "
                              ++ show (V.length lstmV)
                              ++ " but needs at least "
                              ++ show stepCount
                          )
                    | otherwise -> Nothing
                  MethodRouter
                    | V.length kalV < stepCount ->
                        Just
                          ( "sweepThreshold: kalPred has length "
                              ++ show (V.length kalV)
                              ++ " but needs at least "
                              ++ show stepCount
                          )
                    | V.length lstmV < stepCount ->
                        Just
                          ( "sweepThreshold: lstmPred has length "
                              ++ show (V.length lstmV)
                              ++ " but needs at least "
                              ++ show stepCount
                          )
                    | otherwise -> Nothing
                  MethodBlend
                    | V.length kalV < stepCount ->
                        Just
                          ( "sweepThreshold: kalPred has length "
                              ++ show (V.length kalV)
                              ++ " but needs at least "
                              ++ show stepCount
                          )
                    | V.length lstmV < stepCount ->
                        Just
                          ( "sweepThreshold: lstmPred has length "
                              ++ show (V.length lstmV)
                              ++ " but needs at least "
                              ++ show stepCount
                          )
                    | otherwise -> Nothing
                  MethodKalmanOnly
                    | V.length kalV < stepCount ->
                        Just
                          ( "sweepThreshold: kalPred has length "
                              ++ show (V.length kalV)
                              ++ " but needs at least "
                              ++ show stepCount
                          )
                    | otherwise -> Nothing
                  MethodLstmOnly
                    | V.length lstmV < stepCount ->
                        Just
                          ( "sweepThreshold: lstmPred has length "
                              ++ show (V.length lstmV)
                              ++ " but needs at least "
                              ++ show stepCount
                          )
                    | otherwise -> Nothing

      predSources =
        case method of
          MethodBoth -> [kalV, lstmV]
          MethodRouter -> [kalV, lstmV, blendV]
          MethodBlend -> [blendV]
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
      epsilonFor v =
        let rel = abs v * 1e-9
         in max eps rel
      candidates0 = uniqueSorted (0 : map (\v -> max 0 (v - epsilonFor v)) mags)
      candidates =
        uniqueSorted
          ( baseOpenThreshold
              : baseCloseThreshold
              : downsample maxCandidates candidates0
          )
      ppy = max 1e-12 (tcPeriodsPerYear cfg)
      emptyBacktest =
        BacktestResult
          { brEquityCurve = [1]
          , brPositions = []
          , brAgreementOk = []
          , brAgreementValid = []
          , brPositionChanges = 0
          , brTrades = []
          }
      emptyMetrics = computeMetrics ppy emptyBacktest

      lstmFlipEnabled =
        case method of
          MethodBoth -> True
          MethodLstmOnly -> True
          _ -> False

      applyLstmFlip cfg =
        if lstmFlipEnabled
          then cfg
          else cfg { ecLstmExitFlipBars = 0, ecLstmExitFlipGraceBars = 0 }

      evalForOpen openThr =
        let (kalUsedV, lstmUsedV, metaMask) =
              case method of
                MethodRouter ->
                  let routerOpenThr = max openThr minEdge
                      (routerPredV, routerModelsV) =
                        routerPredictionsWithModelsV
                          routerOpenThr
                          roundTripCost
                          routerScorePnlWeight
                          routerLookback
                          routerMinScore
                          pricesV
                          kalV
                          lstmV
                          blendV
                      routerMaskV = V.map (== Just RouterKalman) routerModelsV
                   in (routerPredV, routerPredV, Just routerMaskV)
                _ -> (kalUsedV0, lstmUsedV0, Nothing)
            evalClose closeThr =
              let btCfg0 =
                    baseCfg
                      { ecOpenThreshold = openThr
                      , ecCloseThreshold = closeThr
                      , ecMetaMask = metaMask
                      }
                  btCfg = applyLstmFlip btCfg0
                  btFullE = simulateEnsembleVWithHLChecked btCfg 1 pricesV highsV lowsV kalUsedV lstmUsedV metaUsed
                  (btFull, metrics, eligible, foldScores) =
                    case btFullE of
                      Left _ ->
                        (emptyBacktest, emptyMetrics, False, [ineligibleScore])
                      Right btFull' ->
                        let metrics' = computeMetrics ppy btFull'
                            eligible' =
                              if minRoundTripsReq <= 0
                                then True
                                else bmRoundTrips metrics' >= minRoundTripsReq
                            foldsReq = max 1 (tcWalkForwardFolds cfg)
                            foldRs = foldRanges stepCount foldsReq
                            foldScores' =
                              if not eligible'
                                then [ineligibleScore]
                                else
                                  if length foldRs <= 1
                                    then [scoreBacktest cfg btFull']
                                    else
                                      [ let steps = t1 - t0 + 1
                                            pricesF = V.slice t0 (steps + 1) pricesV
                                            highsF = V.slice t0 (steps + 1) highsV
                                            lowsF = V.slice t0 (steps + 1) lowsV
                                            kalF = V.slice t0 steps kalUsedV
                                            lstmF = V.slice t0 steps lstmUsedV
                                            metaF = fmap (\mv -> V.slice t0 steps mv) metaUsed
                                            openTimesF = fmap (\ot -> V.slice t0 (steps + 1) ot) (ecOpenTimes btCfg)
                                            metaMaskF = fmap (\mask -> V.slice t0 steps mask) metaMask
                                            btCfgFold = btCfg { ecOpenTimes = openTimesF, ecMetaMask = metaMaskF }
                                            btFoldE = simulateEnsembleVWithHLChecked btCfgFold 1 pricesF highsF lowsF kalF lstmF metaF
                                         in case btFoldE of
                                              Left _ -> ineligibleScore
                                              Right btFold -> scoreBacktest cfg btFold
                                      | (t0, t1) <- foldRs
                                      , t1 >= t0
                                      ]
                         in (btFull', metrics', eligible', foldScores')
                  m = mean foldScores
                  s = stddev foldScores
                  stats =
                    TuneStats
                      { tsFoldCount = length foldScores
                      , tsFoldScores = foldScores
                      , tsMeanScore = m
                      , tsStdScore = s
                      }
               in (eligible, m, s, openThr, closeThr, btFull, stats, metrics)
         in evalClose

      (baseEligible, baseMean, baseStd, baseOpenThr, baseCloseThr, baseBt, baseStats, baseMetrics) =
        evalForOpen baseOpenThreshold baseCloseThreshold
      eqEps = 1e-12
      preferTie metrics openThr closeThr bestMetrics bestOpen bestClose =
        let eq = bmFinalEquity metrics
            bestEq = bmFinalEquity bestMetrics
            turnover = bmTurnover metrics
            bestTurnover = bmTurnover bestMetrics
            roundTrips = bmRoundTrips metrics
            bestRoundTrips = bmRoundTrips bestMetrics
            inverted = closeThr > openThr + eqEps
            bestInverted = bestClose > bestOpen + eqEps
         in if eq > bestEq + eqEps
              then True
              else if abs (eq - bestEq) <= eqEps
                then
                  if turnover < bestTurnover - eqEps
                    then True
                    else if abs (turnover - bestTurnover) <= eqEps
                      then
                        if roundTrips > bestRoundTrips
                          then True
                          else if roundTrips == bestRoundTrips
                            then
                              if not inverted && bestInverted
                                then True
                                else inverted == bestInverted && (openThr, closeThr) > (bestOpen, bestClose)
                            else False
                      else False
                else False
      pickResult (bestEligible, bestMean, bestStd, bestOpenThr, bestCloseThr, bestBt, bestStats, bestMetrics) (eligible, m, s, openThr', closeThr', bt, stats, metrics) =
        case (bestEligible, eligible) of
              (False, True) -> (True, m, s, openThr', closeThr', bt, stats, metrics)
              (True, False) -> (bestEligible, bestMean, bestStd, bestOpenThr, bestCloseThr, bestBt, bestStats, bestMetrics)
              _ ->
                if m > bestMean + eqEps
                  then (eligible, m, s, openThr', closeThr', bt, stats, metrics)
                  else if abs (m - bestMean) <= eqEps
                    then
                      if s < bestStd - eqEps
                        then (eligible, m, s, openThr', closeThr', bt, stats, metrics)
                        else if abs (s - bestStd) <= eqEps && preferTie metrics openThr' closeThr' bestMetrics bestOpenThr bestCloseThr
                          then (eligible, m, s, openThr', closeThr', bt, stats, metrics)
                          else (bestEligible, bestMean, bestStd, bestOpenThr, bestCloseThr, bestBt, bestStats, bestMetrics)
                    else (bestEligible, bestMean, bestStd, bestOpenThr, bestCloseThr, bestBt, bestStats, bestMetrics)

      foldClose acc openThr =
        let evalClose = evalForOpen openThr
         in foldl' (\acc0 closeThr -> pickResult acc0 (evalClose closeThr)) acc candidates
      (bestEligible, _, _, bestOpenThr, bestCloseThr, bestBt, bestStats, _bestMetrics) =
        foldl' foldClose (baseEligible, baseMean, baseStd, baseOpenThr, baseCloseThr, baseBt, baseStats, baseMetrics) candidates

      result = (bestOpenThr, bestCloseThr, bestBt, bestStats)
   in
    case validationError of
      Just err -> Left err
      Nothing ->
        if minRoundTripsReq > 0 && not bestEligible
          then Left ("sweepThreshold: no eligible candidates (minRoundTrips=" ++ show minRoundTripsReq ++ ")")
          else Right result

data RouterModel
  = RouterKalman
  | RouterLstm
  | RouterBlend
  deriving (Eq, Show)

data RouterStats = RouterStats
  { rsScore :: !Double
  , rsAccuracy :: !Double
  , rsCoverage :: !Double
  , rsSignals :: !Int
  } deriving (Eq, Show)

routerStatsWindow :: Double -> Double -> Double -> V.Vector Double -> V.Vector Double -> Int -> Int -> RouterStats
routerStatsWindow openThr roundTripCost pnlWeight pricesV predsV start0 end0 =
  let stepCount = min (V.length predsV) (V.length pricesV - 1)
      start = max 0 start0
      end = min end0 (stepCount - 1)
      bad x = isNaN x || isInfinite x
      direction prev next =
        if prev <= 0 || bad prev || bad next
          then Nothing
          else
            let up = prev * (1 + openThr)
                down = prev * (1 - openThr)
             in if next > up
                  then Just (1 :: Int)
                  else if next < down then Just (-1) else Nothing
      step (correct, wrong, signals, netAcc) i =
        let prev = pricesV V.! i
            next = pricesV V.! (i + 1)
            pred = predsV V.! i
            predDir = direction prev pred
            actualDir = direction prev next
            ret = if prev <= 0 || bad prev || bad next then 0 else next / prev - 1
         in case predDir of
              Nothing -> (correct, wrong, signals, netAcc)
              Just dir ->
                let signals' = signals + 1
                    net = fromIntegral dir * ret - roundTripCost
                    netAcc' = netAcc + if bad net then 0 else net
                 in if actualDir == Just dir
                      then (correct + 1, wrong, signals', netAcc')
                      else (correct, wrong + 1, signals', netAcc')
   in
    if stepCount <= 0 || end < start
      then RouterStats { rsScore = 0, rsAccuracy = 0, rsCoverage = 0, rsSignals = 0 }
      else
        let windowLen = end - start + 1
            (correct, _wrong, signals, netAcc) = foldl' step (0, 0, 0, 0) [start .. end]
            accuracy =
              if signals <= 0
                then 0
                else fromIntegral correct / fromIntegral signals
            coverage =
              if windowLen <= 0
                then 0
                else fromIntegral signals / fromIntegral windowLen
            avgNet =
              if signals <= 0
                then 0
                else netAcc / fromIntegral signals
            denom = max 1e-12 (openThr + roundTripCost)
            pnlScore = clamp01 (0.5 + avgNet / denom)
            pnlWeight' = clamp01 pnlWeight
            scoreAcc = accuracy * coverage
            score = (1 - pnlWeight') * scoreAcc + pnlWeight' * pnlScore
         in RouterStats { rsScore = score, rsAccuracy = accuracy, rsCoverage = coverage, rsSignals = signals }

routerSelectModelAt
  :: Double
  -> Double
  -> Double
  -> Int
  -> Double
  -> V.Vector Double
  -> V.Vector Double
  -> V.Vector Double
  -> V.Vector Double
  -> Int
  -> (Maybe RouterModel, Double, Maybe String)
routerSelectModelAt openThr roundTripCost pnlWeight lookback0 minScore0 pricesV kalPredV lstmPredV blendPredV t =
  let stepCount =
        minimum
          [ V.length pricesV - 1
          , V.length kalPredV
          , V.length lstmPredV
          , V.length blendPredV
          ]
      lookback = max 1 lookback0
      minScore = max 0 (min 1 minScore0)
      windowEnd = min (t - 1) (stepCount - 1)
      modelRank m =
        case m of
          RouterBlend -> 2 :: Int
          RouterKalman -> 1
          RouterLstm -> 0
      scoreKey (m, stats) = (rsScore stats, rsCoverage stats, rsAccuracy stats, modelRank m)
      pick best cand =
        if scoreKey cand > scoreKey best
          then cand
          else best
   in
    if stepCount <= 0 || windowEnd < 0
      then (Nothing, 0, Just "ROUTER_WARMUP")
      else
        let windowStart = max 0 (windowEnd - lookback + 1)
            statsKal = routerStatsWindow openThr roundTripCost pnlWeight pricesV kalPredV windowStart windowEnd
            statsLstm = routerStatsWindow openThr roundTripCost pnlWeight pricesV lstmPredV windowStart windowEnd
            statsBlend = routerStatsWindow openThr roundTripCost pnlWeight pricesV blendPredV windowStart windowEnd
            (bestModel, bestStats) =
              foldl' pick (RouterKalman, statsKal) [(RouterLstm, statsLstm), (RouterBlend, statsBlend)]
            bestScore = rsScore bestStats
         in if bestScore < minScore
              then (Nothing, bestScore, Just "ROUTER_MIN_SCORE")
              else (Just bestModel, bestScore, Nothing)

routerPredictionsWithModelsV
  :: Double
  -> Double
  -> Double
  -> Int
  -> Double
  -> V.Vector Double
  -> V.Vector Double
  -> V.Vector Double
  -> V.Vector Double
  -> (V.Vector Double, V.Vector (Maybe RouterModel))
routerPredictionsWithModelsV openThr roundTripCost pnlWeight lookback minScore pricesV kalPredV lstmPredV blendPredV =
  let stepCount =
        minimum
          [ V.length pricesV - 1
          , V.length kalPredV
          , V.length lstmPredV
          , V.length blendPredV
          ]
      pickPred t =
        case routerSelectModelAt openThr roundTripCost pnlWeight lookback minScore pricesV kalPredV lstmPredV blendPredV t of
          (Just RouterKalman, _, _) -> (kalPredV V.! t, Just RouterKalman)
          (Just RouterLstm, _, _) -> (lstmPredV V.! t, Just RouterLstm)
          (Just RouterBlend, _, _) -> (blendPredV V.! t, Just RouterBlend)
          _ -> (pricesV V.! t, Nothing)
      picks = V.generate (max 0 stepCount) pickPred
   in (V.map fst picks, V.map snd picks)
