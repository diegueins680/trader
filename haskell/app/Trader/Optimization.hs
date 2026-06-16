module Trader.Optimization (
    TuneObjective (..),
    tuneObjectiveCode,
    parseTuneObjective,
    TuneConfig (..),
    defaultMaxThresholdCandidates,
    defaultTuneConfig,
    TuneStats (..),
    bestFinalEquity,
    optimizeOperations,
    optimizeOperationsWithHL,
    optimizeOperationsWith,
    optimizeOperationsWithHLWith,
    sweepThreshold,
    sweepThresholdWithHL,
    sweepThresholdWith,
    sweepThresholdWithHLWith,
) where

import qualified Data.Char
import qualified Data.Either
import Data.List (foldl', intercalate, sort)
import Data.Maybe (fromMaybe, mapMaybe)
import qualified Data.Set as Set
import qualified Data.Vector as V

import Trader.Formal.Optimization (
    preferTieBreakImplementation,
    roiImplementationScore,
    tieBreakCandidateFromMetrics,
 )
import Trader.Method (Method (..))
import Trader.Metrics (BacktestMetrics (..), computeMetrics)
import Trader.SignalGates (signalEntryHeadroomThresholdCap)
import Trader.Trading (BacktestResult (..), EnsembleConfig (..), StepMeta (..), emptyBacktestCostAttribution, simulateEnsembleVWithHLChecked)

data TuneObjective
    = TuneFinalEquity
    | TuneAnnualizedEquity
    | TuneRoi
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
        TuneRoi -> "roi"
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
        "roi" -> Right TuneRoi
        "riskadjustedroi" -> Right TuneRoi
        "risk-adjusted-roi" -> Right TuneRoi
        "risk_adjusted_roi" -> Right TuneRoi
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
                        (map tuneObjectiveCode [TuneAnnualizedEquity, TuneRoi, TuneFinalEquity, TuneSharpe, TuneCalmar, TuneEquityDd, TuneEquityDdTurnover])
                    ++ ")"
                )
  where
    normalize = map (\c -> if c == '_' then '-' else c) . filter (not . Data.Char.isSpace) . map toLower
    toLower c =
        if Data.Char.isAsciiUpper c then toEnum (fromEnum c + 32) else c

data TuneConfig = TuneConfig
    { tcObjective :: !TuneObjective
    , tcPenaltyMaxDrawdown :: !Double
    , tcPenaltyTurnover :: !Double
    , tcPeriodsPerYear :: !Double
    , tcWalkForwardFolds :: !Int
    , tcWalkForwardEmbargoBars :: !Int
    , tcMinRoundTrips :: !Int
    , tcMaxThresholdCandidates :: !Int
    , tcStressVolMultiplier :: !Double
    , tcStressShock :: !Double
    , tcStressWeight :: !Double
    }
    deriving (Eq, Show)

defaultMaxThresholdCandidates :: Int
defaultMaxThresholdCandidates = 60

data TuneStats = TuneStats
    { tsFoldCount :: !Int
    , tsFoldScores :: ![Double]
    , tsMeanScore :: !Double
    , tsStdScore :: !Double
    }
    deriving (Eq, Show)

defaultTuneConfig :: Double -> TuneConfig
defaultTuneConfig periodsPerYear =
    TuneConfig
        { tcObjective = TuneRoi
        , tcPenaltyMaxDrawdown = 1.0
        , tcPenaltyTurnover = 0.0
        , tcPeriodsPerYear = max 1e-12 periodsPerYear
        , tcWalkForwardFolds = 1
        , tcWalkForwardEmbargoBars = 0
        , tcMinRoundTrips = 0
        , tcMaxThresholdCandidates = defaultMaxThresholdCandidates
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
                        brStress = br{brEquityCurve = stressEq}
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
            TuneRoi -> roiImplementationScore pDd pTurn m
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
        bad x = isNaN x || isInfinite x
        clamp x =
            if bad x || x < 0
                then 0
                else x
        startEq =
            case eq of
                (x : _) -> clamp x
                [] -> 1.0
        step acc r =
            let r' = r * volMult + shock
                next = acc * (1 + r')
                next' = if isNaN next || isInfinite next || next < 0 then 0 else next
             in next'
     in case eq of
            [] -> []
            [_] -> [startEq]
            _ -> scanl step startEq rets

returnsFromEquity :: [Double] -> [Double]
returnsFromEquity eq =
    zipWith ret eq' (drop1 eq')
  where
    bad x = isNaN x || isInfinite x
    clamp x =
        if bad x || x < 0
            then 0
            else x
    eq' = map clamp eq
    drop1 xs =
        case xs of
            [] -> []
            _ : ys -> ys
    ret a b =
        if a <= 0
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

isFiniteDouble :: Double -> Bool
isFiniteDouble x = not (isNaN x || isInfinite x)

neutralPredFromPrev :: Double -> Double
neutralPredFromPrev prev =
    if isFiniteDouble prev
        then prev
        else 0

finiteBlendOrNeutral :: Double -> Double -> Double -> Double -> Double
finiteBlendOrNeutral weight prev kalPred lstmPred =
    let w = clamp01 weight
        blended = w * kalPred + (1 - w) * lstmPred
     in if isFiniteDouble blended
            then blended
            else neutralPredFromPrev prev

blendPredFromPreds :: Double -> Double -> Double -> Double -> Double
blendPredFromPreds fallbackWeight prev kalPred lstmPred =
    let bad x = isNaN x || isInfinite x
        w = clamp01 fallbackWeight
     in case (bad kalPred, bad lstmPred) of
            (False, False) -> finiteBlendOrNeutral w prev kalPred lstmPred
            (False, True) -> kalPred
            (True, False) -> lstmPred
            (True, True) -> neutralPredFromPrev prev

blendPredictionsV :: Double -> V.Vector Double -> V.Vector Double -> V.Vector Double -> V.Vector Double
blendPredictionsV fallbackWeight pricesV kalPredV lstmPredV =
    let stepCount = minimum [V.length pricesV - 1, V.length kalPredV, V.length lstmPredV]
        pick t =
            let prev = pricesV V.! t
                kalPred = kalPredV V.! t
                lstmPred = lstmPredV V.! t
             in blendPredFromPreds fallbackWeight prev kalPred lstmPred
     in V.generate (max 0 stepCount) pick

scale01 :: Double -> Double -> Double -> Double
scale01 lo hi x =
    let lo' = min lo hi
        hi' = max lo hi
     in if hi' <= lo' + 1e-12
            then if x >= hi' then 1 else 0
            else clamp01 ((x - lo') / (hi' - lo'))

lstmConfidenceScoreFromPred :: Double -> Double -> Double -> Maybe Double
lstmConfidenceScoreFromPred openThr prev next =
    let bad x = isNaN x || isInfinite x
        thr = max 1e-12 openThr
     in if prev <= 0 || bad prev || bad next
            then Nothing
            else
                let edge = abs (next / prev - 1)
                    raw = edge / (2 * thr)
                 in if bad edge || bad raw
                        then Nothing
                        else Just (clamp01 raw)

kalmanZFromMeta :: StepMeta -> Maybe Double
kalmanZFromMeta m =
    let v = max 0 (smKalmanVar m)
        s = sqrt v
        z = if s <= 0 then 0 else abs (smKalmanMean m) / s
     in if isNaN z || isInfinite z then Nothing else Just z

confidenceBlendWeightFromPreds ::
    Double ->
    Double ->
    Double ->
    Double ->
    Double ->
    Double ->
    Double ->
    Maybe Double ->
    Double
confidenceBlendWeightFromPreds fallbackWeight zMin zMax openThr prev kalPred lstmPred mKalZ =
    let wFallback = clamp01 fallbackWeight
        kalScore =
            case mKalZ of
                Just z | not (isNaN z || isInfinite z) -> scale01 zMin zMax z
                _ -> fromMaybe 0 (lstmConfidenceScoreFromPred openThr prev kalPred)
        lstmScore = fromMaybe 0 (lstmConfidenceScoreFromPred openThr prev lstmPred)
        denom = kalScore + lstmScore
     in if denom <= 1e-12
            then wFallback
            else clamp01 (kalScore / denom)

confidenceBlendPredFromPreds ::
    Double ->
    Double ->
    Double ->
    Double ->
    Double ->
    Double ->
    Double ->
    Maybe Double ->
    Double
confidenceBlendPredFromPreds fallbackWeight zMin zMax openThr prev kalPred lstmPred mKalZ =
    let bad x = isNaN x || isInfinite x
        wFallback = clamp01 fallbackWeight
     in case (bad kalPred, bad lstmPred) of
            (False, False) ->
                let w = confidenceBlendWeightFromPreds fallbackWeight zMin zMax openThr prev kalPred lstmPred mKalZ
                 in w * kalPred + (1 - w) * lstmPred
            (False, True) -> kalPred
            (True, False) -> lstmPred
            (True, True) -> finiteBlendOrNeutral wFallback prev kalPred lstmPred

confidenceBlendPredictionsV ::
    Double ->
    Double ->
    Double ->
    Double ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double ->
    Maybe (V.Vector StepMeta) ->
    V.Vector Double
confidenceBlendPredictionsV fallbackWeight zMin zMax openThr pricesV kalPredV lstmPredV mMetaV =
    let stepCount = minimum [V.length pricesV - 1, V.length kalPredV, V.length lstmPredV]
        kalZAt t =
            case mMetaV of
                Just metaV
                    | t >= 0 && t < V.length metaV ->
                        kalmanZFromMeta (metaV V.! t)
                _ -> Nothing
        pick t =
            let prev = pricesV V.! t
                kalPred = kalPredV V.! t
                lstmPred = lstmPredV V.! t
             in confidenceBlendPredFromPreds fallbackWeight zMin zMax openThr prev kalPred lstmPred (kalZAt t)
     in V.generate (max 0 stepCount) pick

confidencePickPredFromPreds ::
    Double ->
    Double ->
    Double ->
    Double ->
    Double ->
    Double ->
    Double ->
    Maybe Double ->
    Double
confidencePickPredFromPreds fallbackWeight zMin zMax openThr prev kalPred lstmPred mKalZ =
    let bad x = isNaN x || isInfinite x
     in case (bad kalPred, bad lstmPred) of
            (False, False) ->
                let w = confidenceBlendWeightFromPreds fallbackWeight zMin zMax openThr prev kalPred lstmPred mKalZ
                 in if w >= 0.5 then kalPred else lstmPred
            (False, True) -> kalPred
            (True, False) -> lstmPred
            (True, True) ->
                let wFallback = clamp01 fallbackWeight
                 in finiteBlendOrNeutral wFallback prev kalPred lstmPred

confidencePickPredictionsV ::
    Double ->
    Double ->
    Double ->
    Double ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double ->
    Maybe (V.Vector StepMeta) ->
    V.Vector Double
confidencePickPredictionsV fallbackWeight zMin zMax openThr pricesV kalPredV lstmPredV mMetaV =
    let stepCount = minimum [V.length pricesV - 1, V.length kalPredV, V.length lstmPredV]
        kalZAt t =
            case mMetaV of
                Just metaV
                    | t >= 0 && t < V.length metaV ->
                        kalmanZFromMeta (metaV V.! t)
                _ -> Nothing
        pick t =
            let prev = pricesV V.! t
                kalPred = kalPredV V.! t
                lstmPred = lstmPredV V.! t
             in confidencePickPredFromPreds fallbackWeight zMin zMax openThr prev kalPred lstmPred (kalZAt t)
     in V.generate (max 0 stepCount) pick

costPickWeightFromPreds ::
    Double ->
    Double ->
    Double ->
    Double ->
    Double ->
    Double
costPickWeightFromPreds fallbackWeight roundTripCost prev kalPred lstmPred =
    let netEdge x =
            if prev <= 0 || isNaN prev || isInfinite prev || isNaN x || isInfinite x
                then Nothing
                else
                    let raw = abs (x / prev - 1) - max 0 roundTripCost
                        v = max 0 raw
                     in if isNaN v || isInfinite v then Nothing else Just v
        wFallback = clamp01 fallbackWeight
     in case (netEdge kalPred, netEdge lstmPred) of
            (Just eKal, Just eLstm) ->
                let denom = eKal + eLstm
                 in if denom <= 1e-12
                        then wFallback
                        else clamp01 (eKal / denom)
            (Just _, Nothing) -> 1
            (Nothing, Just _) -> 0
            (Nothing, Nothing) -> wFallback

costPickPredFromPreds ::
    Double ->
    Double ->
    Double ->
    Double ->
    Double ->
    Double
costPickPredFromPreds fallbackWeight roundTripCost prev kalPred lstmPred =
    let bad x = isNaN x || isInfinite x
     in case (bad kalPred, bad lstmPred) of
            (False, False) ->
                let w = costPickWeightFromPreds fallbackWeight roundTripCost prev kalPred lstmPred
                 in if w >= 0.5 then kalPred else lstmPred
            (False, True) -> kalPred
            (True, False) -> lstmPred
            (True, True) ->
                let wFallback = clamp01 fallbackWeight
                 in finiteBlendOrNeutral wFallback prev kalPred lstmPred

costPickPredictionsV ::
    Double ->
    Double ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double
costPickPredictionsV fallbackWeight roundTripCost pricesV kalPredV lstmPredV =
    let stepCount = minimum [V.length pricesV - 1, V.length kalPredV, V.length lstmPredV]
        pick t =
            let prev = pricesV V.! t
                kalPred = kalPredV V.! t
                lstmPred = lstmPredV V.! t
             in costPickPredFromPreds fallbackWeight roundTripCost prev kalPred lstmPred
     in V.generate (max 0 stepCount) pick

harmonicBlendPredFromPreds ::
    Double ->
    Double ->
    Double ->
    Double ->
    Double
harmonicBlendPredFromPreds fallbackWeight prev kalPred lstmPred =
    let bad x = isNaN x || isInfinite x
        w = clamp01 fallbackWeight
        arithmetic = finiteBlendOrNeutral w prev kalPred lstmPred
        eps = 1e-12
     in case (bad prev || prev <= 0, bad kalPred, bad lstmPred) of
            (False, False, False) ->
                if kalPred > 0 && lstmPred > 0
                    then
                        let rKal = kalPred / prev
                            rLstm = lstmPred / prev
                            denom = w / max eps rKal + (1 - w) / max eps rLstm
                            pred = if denom <= eps then arithmetic else prev / denom
                         in if isNaN pred || isInfinite pred then arithmetic else pred
                    else arithmetic
            (_, False, True) -> kalPred
            (_, True, False) -> lstmPred
            (_, False, False) -> arithmetic
            _ -> arithmetic

harmonicBlendPredictionsV ::
    Double ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double
harmonicBlendPredictionsV fallbackWeight pricesV kalPredV lstmPredV =
    let stepCount = minimum [V.length pricesV - 1, V.length kalPredV, V.length lstmPredV]
        pick t =
            let prev = pricesV V.! t
                kalPred = kalPredV V.! t
                lstmPred = lstmPredV V.! t
             in harmonicBlendPredFromPreds fallbackWeight prev kalPred lstmPred
     in V.generate (max 0 stepCount) pick

disagreementGuardPredFromPreds ::
    Double ->
    Double ->
    Double ->
    Double ->
    Double
disagreementGuardPredFromPreds fallbackWeight prev kalPred lstmPred =
    let bad x = isNaN x || isInfinite x
        wFallback = clamp01 fallbackWeight
        edge x =
            if prev <= 0 || bad prev || bad x
                then Nothing
                else
                    let v = abs (x / prev - 1)
                     in if bad v then Nothing else Just v
        dir x
            | bad x || bad prev = Nothing
            | x > prev = Just (1 :: Int)
            | x < prev = Just (-1 :: Int)
            | otherwise = Just 0
     in case (bad kalPred, bad lstmPred) of
            (False, False) ->
                case (edge kalPred, edge lstmPred, dir kalPred, dir lstmPred) of
                    (Just eKal, Just eLstm, Just dKal, Just dLstm) ->
                        if dKal == dLstm
                            then
                                if eKal > eLstm
                                    then kalPred
                                    else
                                        if eLstm > eKal
                                            then lstmPred
                                            else if wFallback >= 0.5 then kalPred else lstmPred
                            else
                                if eKal < eLstm
                                    then kalPred
                                    else
                                        if eLstm < eKal
                                            then lstmPred
                                            else if wFallback >= 0.5 then kalPred else lstmPred
                    _ -> if wFallback >= 0.5 then kalPred else lstmPred
            (False, True) -> kalPred
            (True, False) -> lstmPred
            (True, True) -> finiteBlendOrNeutral wFallback prev kalPred lstmPred

disagreementGuardPredictionsV ::
    Double ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double
disagreementGuardPredictionsV fallbackWeight pricesV kalPredV lstmPredV =
    let stepCount = minimum [V.length pricesV - 1, V.length kalPredV, V.length lstmPredV]
        pick t =
            let prev = pricesV V.! t
                kalPred = kalPredV V.! t
                lstmPred = lstmPredV V.! t
             in disagreementGuardPredFromPreds fallbackWeight prev kalPred lstmPred
     in V.generate (max 0 stepCount) pick

medianBlendPredFromPreds ::
    Double ->
    Double ->
    Double ->
    Double ->
    Double
medianBlendPredFromPreds fallbackWeight prev kalPred lstmPred =
    let bad x = isNaN x || isInfinite x
        w = clamp01 fallbackWeight
        arithmetic = finiteBlendOrNeutral w prev kalPred lstmPred
     in case (bad prev || prev <= 0, bad kalPred, bad lstmPred) of
            (False, False, False) ->
                let rKal = kalPred / prev
                    rLstm = lstmPred / prev
                    rBlend = arithmetic / prev
                    rMedian = median3 rKal rLstm rBlend
                    pred = prev * rMedian
                 in if isNaN pred || isInfinite pred then arithmetic else pred
            (_, False, True) -> kalPred
            (_, True, False) -> lstmPred
            (_, False, False) -> arithmetic
            _ -> arithmetic
  where
    median3 a b c
        | a <= b = if b <= c then b else max a c
        | a <= c = a
        | b <= c = c
        | otherwise = b

medianBlendPredictionsV ::
    Double ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double
medianBlendPredictionsV fallbackWeight pricesV kalPredV lstmPredV =
    let stepCount = minimum [V.length pricesV - 1, V.length kalPredV, V.length lstmPredV]
        pick t =
            let prev = pricesV V.! t
                kalPred = kalPredV V.! t
                lstmPred = lstmPredV V.! t
             in medianBlendPredFromPreds fallbackWeight prev kalPred lstmPred
     in V.generate (max 0 stepCount) pick

neutralGuardPredFromPreds ::
    Double ->
    Double ->
    Double ->
    Double ->
    Double
neutralGuardPredFromPreds fallbackWeight prev kalPred lstmPred =
    let bad x = isNaN x || isInfinite x
        wFallback = clamp01 fallbackWeight
        blend = finiteBlendOrNeutral wFallback prev kalPred lstmPred
        neutralPred =
            if bad prev || isInfinite prev
                then blend
                else prev
        edge x =
            if prev <= 0 || bad prev || bad x
                then Nothing
                else
                    let v = abs (x / prev - 1)
                     in if bad v then Nothing else Just v
        dir x
            | bad x || bad prev = Nothing
            | x > prev = Just (1 :: Int)
            | x < prev = Just (-1 :: Int)
            | otherwise = Just 0
     in case (bad kalPred, bad lstmPred) of
            (False, False) ->
                case (dir kalPred, dir lstmPred, edge kalPred, edge lstmPred) of
                    (Just dKal, Just dLstm, _, _)
                        | dKal /= dLstm ->
                            neutralPred
                    (_, _, Just eKal, Just eLstm) ->
                        if eKal < eLstm
                            then kalPred
                            else
                                if eLstm < eKal
                                    then lstmPred
                                    else if wFallback >= 0.5 then kalPred else lstmPred
                    _ -> if wFallback >= 0.5 then kalPred else lstmPred
            (False, True) -> kalPred
            (True, False) -> lstmPred
            (True, True) -> blend

neutralGuardPredictionsV ::
    Double ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double
neutralGuardPredictionsV fallbackWeight pricesV kalPredV lstmPredV =
    let stepCount = minimum [V.length pricesV - 1, V.length kalPredV, V.length lstmPredV]
        pick t =
            let prev = pricesV V.! t
                kalPred = kalPredV V.! t
                lstmPred = lstmPredV V.! t
             in neutralGuardPredFromPreds fallbackWeight prev kalPred lstmPred
     in V.generate (max 0 stepCount) pick

riskParityBlendWeightFromPreds ::
    Double ->
    Double ->
    Double ->
    Double ->
    Double
riskParityBlendWeightFromPreds fallbackWeight prev kalPred lstmPred =
    let edge x =
            if prev <= 0 || isNaN prev || isInfinite prev || isNaN x || isInfinite x
                then Nothing
                else
                    let v = abs (x / prev - 1)
                     in if isNaN v || isInfinite v then Nothing else Just v
        wFallback = clamp01 fallbackWeight
        eps = 1e-12
     in case (edge kalPred, edge lstmPred) of
            (Just eKal, Just eLstm) ->
                if eKal <= eps && eLstm <= eps
                    then wFallback
                    else
                        let invKal = 1 / max eps eKal
                            invLstm = 1 / max eps eLstm
                            denom = invKal + invLstm
                         in if denom <= eps
                                then wFallback
                                else clamp01 (invKal / denom)
            (Just _, Nothing) -> 1
            (Nothing, Just _) -> 0
            (Nothing, Nothing) -> wFallback

riskParityBlendPredFromPreds ::
    Double ->
    Double ->
    Double ->
    Double ->
    Double
riskParityBlendPredFromPreds fallbackWeight prev kalPred lstmPred =
    let bad x = isNaN x || isInfinite x
        wFallback = clamp01 fallbackWeight
     in case (bad kalPred, bad lstmPred) of
            (False, False) ->
                let w = riskParityBlendWeightFromPreds fallbackWeight prev kalPred lstmPred
                 in w * kalPred + (1 - w) * lstmPred
            (False, True) -> kalPred
            (True, False) -> lstmPred
            (True, True) -> finiteBlendOrNeutral wFallback prev kalPred lstmPred

riskParityBlendPredictionsV ::
    Double ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double
riskParityBlendPredictionsV fallbackWeight pricesV kalPredV lstmPredV =
    let stepCount = minimum [V.length pricesV - 1, V.length kalPredV, V.length lstmPredV]
        pick t =
            let prev = pricesV V.! t
                kalPred = kalPredV V.! t
                lstmPred = lstmPredV V.! t
             in riskParityBlendPredFromPreds fallbackWeight prev kalPred lstmPred
     in V.generate (max 0 stepCount) pick

consensusBoostPredFromPreds ::
    Double ->
    Double ->
    Double ->
    Double ->
    Double
consensusBoostPredFromPreds fallbackWeight prev kalPred lstmPred =
    let bad x = isNaN x || isInfinite x
        wFallback = clamp01 fallbackWeight
        blend = finiteBlendOrNeutral wFallback prev kalPred lstmPred
        neutralPred =
            if bad prev || isInfinite prev
                then blend
                else prev
        edge x =
            if prev <= 0 || bad prev || bad x
                then Nothing
                else
                    let v = abs (x / prev - 1)
                     in if bad v then Nothing else Just v
        dir x
            | bad x || bad prev = Nothing
            | x > prev = Just (1 :: Int)
            | x < prev = Just (-1 :: Int)
            | otherwise = Just 0
     in case (bad kalPred, bad lstmPred) of
            (False, False) ->
                case (dir kalPred, dir lstmPred, edge kalPred, edge lstmPred) of
                    (Just dKal, Just dLstm, Just eKal, Just eLstm)
                        | dKal == dLstm && dKal /= 0 ->
                            if eKal > eLstm
                                then kalPred
                                else
                                    if eLstm > eKal
                                        then lstmPred
                                        else if wFallback >= 0.5 then kalPred else lstmPred
                        | dKal /= dLstm ->
                            neutralPred
                        | otherwise ->
                            neutralPred
                    _ -> if wFallback >= 0.5 then kalPred else lstmPred
            (False, True) -> kalPred
            (True, False) -> lstmPred
            (True, True) -> blend

consensusBoostPredictionsV ::
    Double ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double
consensusBoostPredictionsV fallbackWeight pricesV kalPredV lstmPredV =
    let stepCount = minimum [V.length pricesV - 1, V.length kalPredV, V.length lstmPredV]
        pick t =
            let prev = pricesV V.! t
                kalPred = kalPredV V.! t
                lstmPred = lstmPredV V.! t
             in consensusBoostPredFromPreds fallbackWeight prev kalPred lstmPred
     in V.generate (max 0 stepCount) pick

anchorBlendPredFromPreds ::
    Double ->
    Double ->
    Double ->
    Double ->
    Double ->
    Double ->
    Double ->
    Double
anchorBlendPredFromPreds conflictBaseRaw conflictScaleRaw alignedScaleRaw fallbackWeight prev kalPred lstmPred =
    let bad x = isNaN x || isInfinite x
        wFallback = clamp01 fallbackWeight
        conflictBase = clamp01 conflictBaseRaw
        conflictScale = clamp01 conflictScaleRaw
        alignedScale = clamp01 alignedScaleRaw
        blend = finiteBlendOrNeutral wFallback prev kalPred lstmPred
        neutralPred =
            if bad prev || isInfinite prev
                then blend
                else prev
        edge x =
            if prev <= 0 || bad prev || bad x
                then Nothing
                else
                    let v = abs (x / prev - 1)
                     in if bad v then Nothing else Just v
        dir x
            | bad x || bad prev = Nothing
            | x > prev = Just (1 :: Int)
            | x < prev = Just (-1 :: Int)
            | otherwise = Just 0
     in case (bad kalPred, bad lstmPred) of
            (False, False) ->
                case (edge kalPred, edge lstmPred, dir kalPred, dir lstmPred) of
                    (Just eKal, Just eLstm, Just dKal, Just dLstm) ->
                        let total = eKal + eLstm
                            gap = abs (eKal - eLstm)
                            conflictScore =
                                if total <= 1e-12
                                    then 0
                                    else clamp01 (gap / total)
                            anchorWeight =
                                if dKal /= dLstm
                                    then min 1 (conflictBase + conflictScale * conflictScore)
                                    else alignedScale * conflictScore
                            pred = (1 - anchorWeight) * blend + anchorWeight * neutralPred
                         in if bad pred then blend else pred
                    _ -> blend
            (False, True) -> kalPred
            (True, False) -> lstmPred
            (True, True) -> blend

anchorBlendPredictionsV ::
    Double ->
    Double ->
    Double ->
    Double ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double
anchorBlendPredictionsV conflictBase conflictScale alignedScale fallbackWeight pricesV kalPredV lstmPredV =
    let stepCount = minimum [V.length pricesV - 1, V.length kalPredV, V.length lstmPredV]
        pick t =
            let prev = pricesV V.! t
                kalPred = kalPredV V.! t
                lstmPred = lstmPredV V.! t
             in anchorBlendPredFromPreds conflictBase conflictScale alignedScale fallbackWeight prev kalPred lstmPred
     in V.generate (max 0 stepCount) pick

tensionGatePredFromPreds ::
    Double ->
    Double ->
    Double ->
    Double ->
    Double ->
    Double ->
    Double
tensionGatePredFromPreds conflictShrinkRaw neutralShrinkRaw fallbackWeight prev kalPred lstmPred =
    let bad x = isNaN x || isInfinite x
        wFallback = clamp01 fallbackWeight
        conflictShrink = clamp01 conflictShrinkRaw
        neutralShrink = clamp01 neutralShrinkRaw
        blend = finiteBlendOrNeutral wFallback prev kalPred lstmPred
        neutralPred =
            if bad prev || isInfinite prev
                then blend
                else prev
        edge x =
            if prev <= 0 || bad prev || bad x
                then Nothing
                else
                    let v = abs (x / prev - 1)
                     in if bad v then Nothing else Just v
        dir x
            | bad x || bad prev = Nothing
            | x > prev = Just (1 :: Int)
            | x < prev = Just (-1 :: Int)
            | otherwise = Just 0
        chooseStrong eKal eLstm =
            case compare eKal eLstm of
                GT -> kalPred
                LT -> lstmPred
                EQ -> if wFallback >= 0.5 then kalPred else lstmPred
        chooseWeak eKal eLstm =
            case compare eKal eLstm of
                LT -> kalPred
                GT -> lstmPred
                EQ -> if wFallback >= 0.5 then kalPred else lstmPred
        shrink alpha pred = (1 - alpha) * neutralPred + alpha * pred
     in case (bad kalPred, bad lstmPred) of
            (False, False) ->
                case (dir kalPred, dir lstmPred, edge kalPred, edge lstmPred) of
                    (Just dKal, Just dLstm, Just eKal, Just eLstm)
                        | dKal == dLstm && dKal /= 0 ->
                            chooseStrong eKal eLstm
                        | dKal /= dLstm ->
                            let pred = shrink conflictShrink (chooseWeak eKal eLstm)
                             in if bad pred then neutralPred else pred
                        | otherwise ->
                            shrink neutralShrink (if wFallback >= 0.5 then kalPred else lstmPred)
                    _ -> if wFallback >= 0.5 then kalPred else lstmPred
            (False, True) -> kalPred
            (True, False) -> lstmPred
            (True, True) -> blend

tensionGatePredictionsV ::
    Double ->
    Double ->
    Double ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double
tensionGatePredictionsV conflictShrink neutralShrink fallbackWeight pricesV kalPredV lstmPredV =
    let stepCount = minimum [V.length pricesV - 1, V.length kalPredV, V.length lstmPredV]
        pick t =
            let prev = pricesV V.! t
                kalPred = kalPredV V.! t
                lstmPred = lstmPredV V.! t
             in tensionGatePredFromPreds conflictShrink neutralShrink fallbackWeight prev kalPred lstmPred
     in V.generate (max 0 stepCount) pick

entropyBlendPredFromPreds ::
    Double ->
    Double ->
    Double ->
    Double ->
    Double ->
    Double ->
    Double ->
    Double ->
    Double
entropyBlendPredFromPreds conflictFloorRaw conflictScaleRaw alignedBaseRaw alignedEntropyScaleRaw fallbackWeight prev kalPred lstmPred =
    let bad x = isNaN x || isInfinite x
        wFallback = clamp01 fallbackWeight
        conflictFloor = clamp01 conflictFloorRaw
        conflictScale = clamp01 conflictScaleRaw
        alignedBase = clamp01 alignedBaseRaw
        alignedEntropyScale = clamp01 alignedEntropyScaleRaw
        blend = finiteBlendOrNeutral wFallback prev kalPred lstmPred
        neutralPred =
            if bad prev || isInfinite prev
                then blend
                else prev
        edge x =
            if prev <= 0 || bad prev || bad x
                then Nothing
                else
                    let v = abs (x / prev - 1)
                     in if bad v then Nothing else Just v
        dir x
            | bad x || bad prev = Nothing
            | x > prev = Just (1 :: Int)
            | x < prev = Just (-1 :: Int)
            | otherwise = Just 0
        entropy01 p0 =
            let eps = 1e-12
                p = max eps (min (1 - eps) (clamp01 p0))
                q = 1 - p
                h = -(((p * log p) + (q * log q)) / log 2)
             in if bad h then 1 else clamp01 h
     in case (bad kalPred, bad lstmPred) of
            (False, False) ->
                case (edge kalPred, edge lstmPred, dir kalPred, dir lstmPred) of
                    (Just eKal, Just eLstm, Just dKal, Just dLstm) ->
                        let denom = eKal + eLstm
                            pKal =
                                if denom <= 1e-12
                                    then wFallback
                                    else clamp01 (eKal / denom)
                            h = entropy01 pKal
                            conflict = dKal /= dLstm
                            alpha =
                                if conflict
                                    then clamp01 (conflictFloor + conflictScale * h)
                                    else clamp01 (alignedBase - alignedEntropyScale * h)
                            pred = neutralPred + alpha * (blend - neutralPred)
                         in if bad pred then blend else pred
                    _ -> blend
            (False, True) -> kalPred
            (True, False) -> lstmPred
            (True, True) -> blend

entropyBlendPredictionsV ::
    Double ->
    Double ->
    Double ->
    Double ->
    Double ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double
entropyBlendPredictionsV conflictFloor conflictScale alignedBase alignedEntropyScale fallbackWeight pricesV kalPredV lstmPredV =
    let stepCount = minimum [V.length pricesV - 1, V.length kalPredV, V.length lstmPredV]
        pick t =
            let prev = pricesV V.! t
                kalPred = kalPredV V.! t
                lstmPred = lstmPredV V.! t
             in entropyBlendPredFromPreds conflictFloor conflictScale alignedBase alignedEntropyScale fallbackWeight prev kalPred lstmPred
     in V.generate (max 0 stepCount) pick

coherenceGatePredFromPreds ::
    Double ->
    Double ->
    Double ->
    Double ->
    Double ->
    Double ->
    Double ->
    Double ->
    Double ->
    Double
coherenceGatePredFromPreds conflictFloorRaw conflictScaleRaw boostThresholdRaw boostGainRaw boostSpanRaw fallbackWeight prev kalPred lstmPred =
    let bad x = isNaN x || isInfinite x
        wFallback = clamp01 fallbackWeight
        conflictFloor = max 0 conflictFloorRaw
        conflictScale = max 0 conflictScaleRaw
        boostThreshold = clamp01 boostThresholdRaw
        boostGain = max 0 boostGainRaw
        boostSpan = max 1e-12 boostSpanRaw
        blend = finiteBlendOrNeutral wFallback prev kalPred lstmPred
        neutralPred =
            if bad prev || isInfinite prev
                then blend
                else prev
        ret x =
            if prev <= 0 || bad prev || bad x
                then Nothing
                else
                    let v = x / prev - 1
                     in if bad v then Nothing else Just v
        edge x =
            if prev <= 0 || bad prev || bad x
                then Nothing
                else
                    let v = abs (x / prev - 1)
                     in if bad v then Nothing else Just v
        dir x
            | bad x || bad prev = Nothing
            | x > prev = Just (1 :: Int)
            | x < prev = Just (-1 :: Int)
            | otherwise = Just 0
        chooseStrong eKal eLstm =
            case compare eKal eLstm of
                GT -> kalPred
                LT -> lstmPred
                EQ -> if wFallback >= 0.5 then kalPred else lstmPred
        chooseWeak eKal eLstm =
            case compare eKal eLstm of
                LT -> kalPred
                GT -> lstmPred
                EQ -> if wFallback >= 0.5 then kalPred else lstmPred
        shrink alpha pred = neutralPred + alpha * (pred - neutralPred)
     in case (bad kalPred, bad lstmPred) of
            (False, False) ->
                case (ret kalPred, ret lstmPred, edge kalPred, edge lstmPred, dir kalPred, dir lstmPred) of
                    (Just rKal, Just rLstm, Just eKal, Just eLstm, Just dKal, Just dLstm) ->
                        let denom = abs rKal + abs rLstm + 1e-12
                            coherence =
                                if denom <= 1e-12
                                    then 1
                                    else clamp01 (1 - abs (rKal - rLstm) / denom)
                            weakPred = chooseWeak eKal eLstm
                            pred
                                | dKal /= dLstm =
                                    shrink (clamp01 (conflictFloor + conflictScale * coherence)) weakPred
                                | dKal == 0 =
                                    neutralPred
                                | coherence >= boostThreshold =
                                    let gain = 1 + boostGain * ((coherence - boostThreshold) / boostSpan)
                                     in neutralPred + gain * (blend - neutralPred)
                                | otherwise =
                                    weakPred
                         in if bad pred then blend else pred
                    _ -> if wFallback >= 0.5 then kalPred else lstmPred
            (False, True) -> kalPred
            (True, False) -> lstmPred
            (True, True) -> blend

coherenceGatePredictionsV ::
    Double ->
    Double ->
    Double ->
    Double ->
    Double ->
    Double ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double
coherenceGatePredictionsV conflictFloor conflictScale boostThreshold boostGain boostSpan fallbackWeight pricesV kalPredV lstmPredV =
    let stepCount = minimum [V.length pricesV - 1, V.length kalPredV, V.length lstmPredV]
        pick t =
            let prev = pricesV V.! t
                kalPred = kalPredV V.! t
                lstmPred = lstmPredV V.! t
             in coherenceGatePredFromPreds conflictFloor conflictScale boostThreshold boostGain boostSpan fallbackWeight prev kalPred lstmPred
     in V.generate (max 0 stepCount) pick

fractalBlendPredFromPreds ::
    Double ->
    Double ->
    Double ->
    Double ->
    Double ->
    Double ->
    Double ->
    Double
fractalBlendPredFromPreds returnClampRaw alignedGainRaw conflictGainRaw fallbackWeight prev kalPred lstmPred =
    let bad x = isNaN x || isInfinite x
        wFallback = clamp01 fallbackWeight
        returnClamp = max 1e-12 returnClampRaw
        alignedGain = max 0 alignedGainRaw
        conflictGain = max 0 conflictGainRaw
        blend = finiteBlendOrNeutral wFallback prev kalPred lstmPred
        neutralPred =
            if bad prev || isInfinite prev
                then blend
                else prev
        ret x =
            if prev <= 0 || bad prev || bad x
                then Nothing
                else
                    let v = x / prev - 1
                     in if bad v then Nothing else Just v
        signedRoot r = signum r * sqrt (abs r)
        signedSquare r = signum r * r * r
        clampRet r = max (negate returnClamp) (min returnClamp r)
     in case (bad kalPred, bad lstmPred) of
            (False, False) ->
                case (ret kalPred, ret lstmPred) of
                    (Just rKal, Just rLstm) ->
                        let sKal = signedRoot rKal
                            sLstm = signedRoot rLstm
                            sBlend = wFallback * sKal + (1 - wFallback) * sLstm
                            aligned = signum rKal == signum rLstm && signum rKal /= 0
                            gain = if aligned then alignedGain else conflictGain
                            predRet = clampRet (signedSquare (gain * sBlend))
                            pred = neutralPred * (1 + predRet)
                         in if bad pred then blend else pred
                    _ -> blend
            (False, True) -> kalPred
            (True, False) -> lstmPred
            (True, True) -> blend

fractalBlendPredictionsV ::
    Double ->
    Double ->
    Double ->
    Double ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double
fractalBlendPredictionsV returnClamp alignedGain conflictGain fallbackWeight pricesV kalPredV lstmPredV =
    let stepCount = minimum [V.length pricesV - 1, V.length kalPredV, V.length lstmPredV]
        pick t =
            let prev = pricesV V.! t
                kalPred = kalPredV V.! t
                lstmPred = lstmPredV V.! t
             in fractalBlendPredFromPreds returnClamp alignedGain conflictGain fallbackWeight prev kalPred lstmPred
     in V.generate (max 0 stepCount) pick

phaseCancelPredFromPreds ::
    Double ->
    Double ->
    Double ->
    Double ->
    Double ->
    Double ->
    Double ->
    Double ->
    Double
phaseCancelPredFromPreds returnClampRaw conflictFloorRaw conflictScaleRaw alignmentScaleRaw fallbackWeight prev kalPred lstmPred =
    let bad x = isNaN x || isInfinite x
        wFallback = clamp01 fallbackWeight
        returnClamp = max 1e-12 returnClampRaw
        conflictFloor = max 0 conflictFloorRaw
        conflictScale = max 0 conflictScaleRaw
        alignmentScale = max 0 alignmentScaleRaw
        blend = finiteBlendOrNeutral wFallback prev kalPred lstmPred
        neutralPred =
            if bad prev || isInfinite prev
                then blend
                else prev
        ret x =
            if prev <= 0 || bad prev || bad x
                then Nothing
                else
                    let v = x / prev - 1
                     in if bad v then Nothing else Just v
        clampRet r = max (negate returnClamp) (min returnClamp r)
     in case (bad kalPred, bad lstmPred) of
            (False, False) ->
                case (ret kalPred, ret lstmPred) of
                    (Just rKal, Just rLstm) ->
                        let absSum = abs rKal + abs rLstm + 1e-12
                            sumMag = abs (rKal + rLstm)
                            diffMag = abs (rKal - rLstm)
                            alignment = clamp01 (sumMag / absSum)
                            cancellation =
                                clamp01
                                    (1 - (sumMag / (sumMag + diffMag + 1e-12)))
                            conflict =
                                signum rKal /= signum rLstm
                                    && signum rKal /= 0
                                    && signum rLstm /= 0
                            blendRet = wFallback * rKal + (1 - wFallback) * rLstm
                            predRet =
                                if conflict
                                    then (conflictFloor + conflictScale * (1 - cancellation)) * blendRet
                                    else (1 + alignmentScale * alignment) * blendRet
                            pred = neutralPred * (1 + clampRet predRet)
                         in if bad pred then blend else pred
                    _ -> blend
            (False, True) -> kalPred
            (True, False) -> lstmPred
            (True, True) -> blend

phaseCancelPredictionsV ::
    Double ->
    Double ->
    Double ->
    Double ->
    Double ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double
phaseCancelPredictionsV returnClamp conflictFloor conflictScale alignmentScale fallbackWeight pricesV kalPredV lstmPredV =
    let stepCount = minimum [V.length pricesV - 1, V.length kalPredV, V.length lstmPredV]
        pick t =
            let prev = pricesV V.! t
                kalPred = kalPredV V.! t
                lstmPred = lstmPredV V.! t
             in phaseCancelPredFromPreds returnClamp conflictFloor conflictScale alignmentScale fallbackWeight prev kalPred lstmPred
     in V.generate (max 0 stepCount) pick

softmaxBlendWeightFromPreds ::
    Double ->
    Double ->
    Double ->
    Double ->
    Double ->
    Double
softmaxBlendWeightFromPreds edgeScale fallbackWeight prev kalPred lstmPred =
    let bad x = isNaN x || isInfinite x
        scale = max 1e-12 edgeScale
        wFallback = clamp01 fallbackWeight
        edge x =
            if prev <= 0 || bad prev || bad x
                then Nothing
                else
                    let v = abs (x / prev - 1)
                     in if bad v then Nothing else Just v
     in case (edge kalPred, edge lstmPred) of
            (Just eKal, Just eLstm) ->
                let d = eKal - eLstm
                    w0 = 1 / (1 + exp (negate (scale * d)))
                    w = clamp01 (wFallback + (w0 - 0.5))
                 in if bad w then wFallback else w
            (Just _, Nothing) -> 1
            (Nothing, Just _) -> 0
            (Nothing, Nothing) -> wFallback

softmaxBlendPredFromPreds ::
    Double ->
    Double ->
    Double ->
    Double ->
    Double ->
    Double
softmaxBlendPredFromPreds edgeScale fallbackWeight prev kalPred lstmPred =
    let bad x = isNaN x || isInfinite x
        wFallback = clamp01 fallbackWeight
     in case (bad kalPred, bad lstmPred) of
            (False, False) ->
                let w = softmaxBlendWeightFromPreds edgeScale wFallback prev kalPred lstmPred
                    pred = w * kalPred + (1 - w) * lstmPred
                 in if bad pred then finiteBlendOrNeutral wFallback prev kalPred lstmPred else pred
            (False, True) -> kalPred
            (True, False) -> lstmPred
            (True, True) -> finiteBlendOrNeutral wFallback prev kalPred lstmPred

softmaxBlendPredictionsV ::
    Double ->
    Double ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double
softmaxBlendPredictionsV edgeScale fallbackWeight pricesV kalPredV lstmPredV =
    let stepCount = minimum [V.length pricesV - 1, V.length kalPredV, V.length lstmPredV]
        pick t =
            let prev = pricesV V.! t
                kalPred = kalPredV V.! t
                lstmPred = lstmPredV V.! t
             in softmaxBlendPredFromPreds edgeScale fallbackWeight prev kalPred lstmPred
     in V.generate (max 0 stepCount) pick

netSoftmaxBlendWeightFromPreds ::
    Double ->
    Double ->
    Double ->
    Double ->
    Double ->
    Double ->
    Double
netSoftmaxBlendWeightFromPreds edgeScale fallbackWeight roundTripCost prev kalPred lstmPred =
    let bad x = isNaN x || isInfinite x
        scale = max 1e-12 edgeScale
        wFallback = clamp01 fallbackWeight
        cost = max 0 roundTripCost
        edge x =
            if prev <= 0 || bad prev || bad x
                then Nothing
                else
                    let v = abs (x / prev - 1)
                     in if bad v then Nothing else Just v
        netEdge x = max 0 (x - cost)
     in case (edge kalPred, edge lstmPred) of
            (Just eKal, Just eLstm) ->
                let d = netEdge eKal - netEdge eLstm
                    w0 = 1 / (1 + exp (negate (scale * d)))
                    w = clamp01 (wFallback + (w0 - 0.5))
                 in if bad w then wFallback else w
            (Just _, Nothing) -> 1
            (Nothing, Just _) -> 0
            (Nothing, Nothing) -> wFallback

netSoftmaxBlendPredFromPreds ::
    Double ->
    Double ->
    Double ->
    Double ->
    Double ->
    Double ->
    Double
netSoftmaxBlendPredFromPreds edgeScale fallbackWeight roundTripCost prev kalPred lstmPred =
    let bad x = isNaN x || isInfinite x
        wFallback = clamp01 fallbackWeight
     in case (bad kalPred, bad lstmPred) of
            (False, False) ->
                let w = netSoftmaxBlendWeightFromPreds edgeScale wFallback roundTripCost prev kalPred lstmPred
                    pred = w * kalPred + (1 - w) * lstmPred
                 in if bad pred then finiteBlendOrNeutral wFallback prev kalPred lstmPred else pred
            (False, True) -> kalPred
            (True, False) -> lstmPred
            (True, True) -> finiteBlendOrNeutral wFallback prev kalPred lstmPred

netSoftmaxBlendPredictionsV ::
    Double ->
    Double ->
    Double ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double
netSoftmaxBlendPredictionsV edgeScale fallbackWeight roundTripCost pricesV kalPredV lstmPredV =
    let stepCount = minimum [V.length pricesV - 1, V.length kalPredV, V.length lstmPredV]
        pick t =
            let prev = pricesV V.! t
                kalPred = kalPredV V.! t
                lstmPred = lstmPredV V.! t
             in netSoftmaxBlendPredFromPreds edgeScale fallbackWeight roundTripCost prev kalPred lstmPred
     in V.generate (max 0 stepCount) pick

smoothSoftmaxBlendPredictionsV ::
    Double ->
    Double ->
    Double ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double
smoothSoftmaxBlendPredictionsV edgeScale alphaRaw fallbackWeight pricesV kalPredV lstmPredV =
    let stepCount = minimum [V.length pricesV - 1, V.length kalPredV, V.length lstmPredV]
        alpha = clamp01 alphaRaw
        bad x = isNaN x || isInfinite x
        wFallback = clamp01 fallbackWeight
        step (t, wPrev0) =
            if t >= stepCount
                then Nothing
                else
                    let prev = pricesV V.! t
                        kalPred = kalPredV V.! t
                        lstmPred = lstmPredV V.! t
                        wPrev = clamp01 wPrev0
                        wSoft = softmaxBlendWeightFromPreds edgeScale wFallback prev kalPred lstmPred
                        wRaw = (1 - alpha) * wPrev + alpha * wSoft
                        w =
                            if bad wRaw
                                then wPrev
                                else clamp01 wRaw
                        pred =
                            case (bad kalPred, bad lstmPred) of
                                (False, False) ->
                                    let v = w * kalPred + (1 - w) * lstmPred
                                     in if bad v then finiteBlendOrNeutral wFallback prev kalPred lstmPred else v
                                (False, True) -> kalPred
                                (True, False) -> lstmPred
                                (True, True) -> finiteBlendOrNeutral wFallback prev kalPred lstmPred
                     in Just (pred, (t + 1, w))
     in V.unfoldrN (max 0 stepCount) step (0, wFallback)

divergenceGatePredFromPreds ::
    Double ->
    Double ->
    Double ->
    Double ->
    Double ->
    Double ->
    Double
divergenceGatePredFromPreds divergenceK fallbackWeight openThr prev kalPred lstmPred =
    let bad x = isNaN x || isInfinite x
        wFallback = clamp01 fallbackWeight
        thr = max 1e-12 (abs openThr)
        k = max 1e-12 divergenceK
        blend = finiteBlendOrNeutral wFallback prev kalPred lstmPred
        neutralPred =
            if bad prev || isInfinite prev
                then blend
                else prev
        ret x =
            if prev <= 0 || bad prev || bad x
                then Nothing
                else
                    let v = x / prev - 1
                     in if bad v then Nothing else Just v
     in case (bad kalPred, bad lstmPred) of
            (False, False) ->
                case (ret kalPred, ret lstmPred) of
                    (Just rKal, Just rLstm) ->
                        let rBlend = wFallback * rKal + (1 - wFallback) * rLstm
                            disp = abs (rKal - rLstm)
                            alpha = 1 / (1 + disp / (k * thr))
                            a = if bad alpha then 1 else clamp01 alpha
                            pred = neutralPred * (1 + a * rBlend)
                         in if bad pred then blend else pred
                    _ -> blend
            (False, True) -> kalPred
            (True, False) -> lstmPred
            (True, True) -> blend

divergenceGatePredictionsV ::
    Double ->
    Double ->
    Double ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double
divergenceGatePredictionsV divergenceK fallbackWeight openThr pricesV kalPredV lstmPredV =
    let stepCount = minimum [V.length pricesV - 1, V.length kalPredV, V.length lstmPredV]
        pick t =
            let prev = pricesV V.! t
                kalPred = kalPredV V.! t
                lstmPred = lstmPredV V.! t
             in divergenceGatePredFromPreds divergenceK fallbackWeight openThr prev kalPred lstmPred
     in V.generate (max 0 stepCount) pick

edgeBlendWeightFromPreds ::
    Double ->
    Double ->
    Double ->
    Double ->
    Double ->
    Double
edgeBlendWeightFromPreds edgePowerRaw fallbackWeight prev kalPred lstmPred =
    let edge x =
            if prev <= 0 || isNaN prev || isInfinite prev || isNaN x || isInfinite x
                then Nothing
                else
                    let v = abs (x / prev - 1)
                     in if isNaN v || isInfinite v then Nothing else Just v
        edgePower = max 1e-12 edgePowerRaw
        wFallback = clamp01 fallbackWeight
     in case (edge kalPred, edge lstmPred) of
            (Just eKal, Just eLstm) ->
                let eKalPowered = eKal ** edgePower
                    eLstmPowered = eLstm ** edgePower
                    denom = eKalPowered + eLstmPowered
                 in if denom <= 1e-12
                        then wFallback
                        else clamp01 (eKalPowered / denom)
            (Just _, Nothing) -> 1
            (Nothing, Just _) -> 0
            (Nothing, Nothing) -> wFallback

edgeBlendPredFromPreds ::
    Double ->
    Double ->
    Double ->
    Double ->
    Double ->
    Double
edgeBlendPredFromPreds edgePower fallbackWeight prev kalPred lstmPred =
    let bad x = isNaN x || isInfinite x
        wFallback = clamp01 fallbackWeight
     in case (bad kalPred, bad lstmPred) of
            (False, False) ->
                let w = edgeBlendWeightFromPreds edgePower fallbackWeight prev kalPred lstmPred
                 in w * kalPred + (1 - w) * lstmPred
            (False, True) -> kalPred
            (True, False) -> lstmPred
            (True, True) -> finiteBlendOrNeutral wFallback prev kalPred lstmPred

edgeBlendPredictionsV ::
    Double ->
    Double ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double
edgeBlendPredictionsV edgePower fallbackWeight pricesV kalPredV lstmPredV =
    let stepCount = minimum [V.length pricesV - 1, V.length kalPredV, V.length lstmPredV]
        pick t =
            let prev = pricesV V.! t
                kalPred = kalPredV V.! t
                lstmPred = lstmPredV V.! t
             in edgeBlendPredFromPreds edgePower fallbackWeight prev kalPred lstmPred
     in V.generate (max 0 stepCount) pick

edgePickPredFromPreds ::
    Double ->
    Double ->
    Double ->
    Double ->
    Double ->
    Double
edgePickPredFromPreds edgePower fallbackWeight prev kalPred lstmPred =
    let bad x = isNaN x || isInfinite x
     in case (bad kalPred, bad lstmPred) of
            (False, False) ->
                let w = edgeBlendWeightFromPreds edgePower fallbackWeight prev kalPred lstmPred
                 in if w >= 0.5 then kalPred else lstmPred
            (False, True) -> kalPred
            (True, False) -> lstmPred
            (True, True) ->
                let wFallback = clamp01 fallbackWeight
                 in finiteBlendOrNeutral wFallback prev kalPred lstmPred

edgePickPredictionsV ::
    Double ->
    Double ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double
edgePickPredictionsV edgePower fallbackWeight pricesV kalPredV lstmPredV =
    let stepCount = minimum [V.length pricesV - 1, V.length kalPredV, V.length lstmPredV]
        pick t =
            let prev = pricesV V.! t
                kalPred = kalPredV V.! t
                lstmPred = lstmPredV V.! t
             in edgePickPredFromPreds edgePower fallbackWeight prev kalPred lstmPred
     in V.generate (max 0 stepCount) pick

geometricBlendPredFromPreds ::
    Double ->
    Double ->
    Double ->
    Double ->
    Double
geometricBlendPredFromPreds fallbackWeight prev kalPred lstmPred =
    let bad x = isNaN x || isInfinite x
        w = clamp01 fallbackWeight
        arithmetic = finiteBlendOrNeutral w prev kalPred lstmPred
     in case (bad prev || prev <= 0, bad kalPred, bad lstmPred) of
            (False, False, False) ->
                if kalPred > 0 && lstmPred > 0
                    then
                        let rKal = kalPred / prev
                            rLstm = lstmPred / prev
                            pred = prev * exp (w * log rKal + (1 - w) * log rLstm)
                         in if isNaN pred || isInfinite pred then arithmetic else pred
                    else arithmetic
            (_, False, True) -> kalPred
            (_, True, False) -> lstmPred
            (_, False, False) -> arithmetic
            _ -> arithmetic

geometricBlendPredictionsV ::
    Double ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double
geometricBlendPredictionsV fallbackWeight pricesV kalPredV lstmPredV =
    let stepCount = minimum [V.length pricesV - 1, V.length kalPredV, V.length lstmPredV]
        pick t =
            let prev = pricesV V.! t
                kalPred = kalPredV V.! t
                lstmPred = lstmPredV V.! t
             in geometricBlendPredFromPreds fallbackWeight prev kalPred lstmPred
     in V.generate (max 0 stepCount) pick

regimeSwitchPredFromPreds ::
    Double ->
    Double ->
    Double ->
    Double ->
    Double ->
    Double ->
    Maybe StepMeta ->
    Double
regimeSwitchPredFromPreds fallbackWeight highVolCutoff kalZCutoff prev kalPred lstmPred mMeta =
    let bad x = isNaN x || isInfinite x
        blend = blendPredFromPreds fallbackWeight prev kalPred lstmPred
        kalZMeta = mMeta >>= kalmanZFromMeta
        hvMeta = mMeta >>= smHighVolProb
     in case (bad kalPred, bad lstmPred) of
            (False, False) ->
                case (kalZMeta, hvMeta) of
                    (Just _, Just hv) | hv >= highVolCutoff -> blend
                    (Just z, _) | z >= kalZCutoff -> kalPred
                    _ -> lstmPred
            (False, True) -> kalPred
            (True, False) -> lstmPred
            (True, True) -> blend

regimeSwitchPredictionsV ::
    Double ->
    Double ->
    Double ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double ->
    Maybe (V.Vector StepMeta) ->
    V.Vector Double
regimeSwitchPredictionsV fallbackWeight highVolCutoff kalZCutoff pricesV kalPredV lstmPredV mMetaV =
    let stepCount = minimum [V.length pricesV - 1, V.length kalPredV, V.length lstmPredV]
        metaAt t =
            case mMetaV of
                Just metaV
                    | t >= 0 && t < V.length metaV -> Just (metaV V.! t)
                _ -> Nothing
        pick t =
            let prev = pricesV V.! t
                kalPred = kalPredV V.! t
                lstmPred = lstmPredV V.! t
             in regimeSwitchPredFromPreds fallbackWeight highVolCutoff kalZCutoff prev kalPred lstmPred (metaAt t)
     in V.generate (max 0 stepCount) pick

conformalClipBoundsFromMeta :: StepMeta -> Maybe (Double, Double)
conformalClipBoundsFromMeta m =
    let bad x = isNaN x || isInfinite x
     in case (smConformalLo m, smConformalHi m) of
            (Just lo, Just hi) | not (bad lo || bad hi) -> Just (lo, hi)
            _ ->
                case (smQuantile10 m, smQuantile90 m) of
                    (Just lo, Just hi) | not (bad lo || bad hi) -> Just (lo, hi)
                    _ -> Nothing

conformalClipPredFromPreds ::
    Double ->
    Double ->
    Double ->
    Double ->
    Maybe (Double, Double) ->
    Double
conformalClipPredFromPreds fallbackWeight prev kalPred lstmPred mBounds =
    let bad x = isNaN x || isInfinite x
        w = clamp01 fallbackWeight
        cand0 =
            case (bad kalPred, bad lstmPred) of
                (False, False) ->
                    let v = w * kalPred + (1 - w) * lstmPred
                     in if bad v then kalPred else v
                (False, True) -> kalPred
                (True, False) -> lstmPred
                (True, True) -> prev
        cand = if bad cand0 then prev else cand0
     in if prev <= 0 || bad prev || bad cand
            then cand
            else
                let rCand = cand / prev - 1
                 in case mBounds of
                        Just (lo0, hi0) ->
                            let lo = min lo0 hi0
                                hi = max lo0 hi0
                                rClipped = max lo (min hi rCand)
                                rSafe = max (-0.999999) rClipped
                                pred = prev * (1 + rSafe)
                             in if bad pred then cand else pred
                        Nothing -> cand

conformalClipPredictionsV ::
    Double ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double ->
    Maybe (V.Vector StepMeta) ->
    V.Vector Double
conformalClipPredictionsV fallbackWeight pricesV kalPredV lstmPredV mMetaV =
    let stepCount = minimum [V.length pricesV - 1, V.length kalPredV, V.length lstmPredV]
        boundsAt t =
            case mMetaV of
                Just metaV
                    | t >= 0 && t < V.length metaV ->
                        conformalClipBoundsFromMeta (metaV V.! t)
                _ -> Nothing
        pick t =
            let prev = pricesV V.! t
                kalPred = kalPredV V.! t
                lstmPred = lstmPredV V.! t
             in conformalClipPredFromPreds fallbackWeight prev kalPred lstmPred (boundsAt t)
     in V.generate (max 0 stepCount) pick

hedgeBlendPredictionsV ::
    Double ->
    Double ->
    Double ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double
hedgeBlendPredictionsV etaRaw maxErrRaw initWeight pricesV kalPredV lstmPredV =
    let stepCount = minimum [V.length pricesV - 1, V.length kalPredV, V.length lstmPredV]
        bad x = isNaN x || isInfinite x
        w0Raw = clamp01 initWeight
        epsW = 1e-6
        w0 = max epsW (min (1 - epsW) (if bad w0Raw then 0.5 else w0Raw))
        logit p = log (p / (1 - p))
        sigmoid z
            | z >= 30 = 1
            | z <= (-30) = 0
            | otherwise = 1 / (1 + exp (negate z))
        eta = max 0 etaRaw
        maxErr = max 1e-12 maxErrRaw
        safePred w prev kalPred lstmPred =
            case (bad kalPred, bad lstmPred) of
                (False, False) ->
                    let v = w * kalPred + (1 - w) * lstmPred
                        fallback = finiteBlendOrNeutral w0 prev kalPred lstmPred
                     in if bad v then fallback else v
                (False, True) -> kalPred
                (True, False) -> lstmPred
                (True, True) ->
                    neutralPredFromPrev prev
        ret prev x =
            if prev <= 0 || bad prev || bad x
                then Nothing
                else
                    let r = x / prev - 1
                     in if bad r then Nothing else Just r
        lossFromR rPred rReal =
            let e = abs (rPred - rReal)
                e' = min maxErr (max 0 e)
             in if bad e' then maxErr else e'
        updateZ z prev actual kalPred lstmPred =
            case ret prev actual of
                Nothing -> z
                Just rReal ->
                    let mRKal = ret prev kalPred
                        mRLstm = ret prev lstmPred
                        lKal = maybe maxErr (`lossFromR` rReal) mRKal
                        lLstm = maybe maxErr (`lossFromR` rReal) mRLstm
                        z' = z - eta * (lKal - lLstm)
                     in if bad z' then z else z'
        step (t, z) =
            if t >= stepCount
                then Nothing
                else
                    let prev = pricesV V.! t
                        actual = pricesV V.! (t + 1)
                        kalPred = kalPredV V.! t
                        lstmPred = lstmPredV V.! t
                        w = sigmoid z
                        pred = safePred w prev kalPred lstmPred
                        z' = updateZ z prev actual kalPred lstmPred
                     in Just (pred, (t + 1, z'))
     in V.unfoldrN (max 0 stepCount) step (0, logit w0)

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
bestFinalEquity br = sanitizeEquity (foldl' (\_ x -> x) 1.0 (brEquityCurve br))
  where
    sanitizeEquity x =
        if isNaN x || isInfinite x || x < 0
            then 0
            else x

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
optimizeOperationsWith cfg baseCfg prices = optimizeOperationsWithHLWith cfg baseCfg prices prices prices

optimizeOperationsWithHLWith :: TuneConfig -> EnsembleConfig -> [Double] -> [Double] -> [Double] -> [Double] -> [Double] -> Maybe [StepMeta] -> Either String (Method, Double, Double, BacktestResult, TuneStats)
optimizeOperationsWithHLWith cfg baseCfg closes highs lows kalPred lstmPred mMeta =
    let eps = 1e-12
        methodRank m =
            case m of
                MethodBoth -> 3 :: Int
                MethodRouter -> 3
                MethodBanditRouter -> 3
                MethodConfBlend -> 2
                MethodConfPick -> 2
                MethodConformalClip -> 2
                MethodCostPick -> 2
                MethodHarmonicBlend -> 2
                MethodDisagreementGuard -> 2
                MethodMedianBlend -> 2
                MethodNeutralGuard -> 2
                MethodRiskParityBlend -> 2
                MethodConsensusBoost -> 2
                MethodAnchorBlend -> 2
                MethodTensionGate -> 2
                MethodEntropyBlend -> 2
                MethodCoherenceGate -> 2
                MethodDivergenceGate -> 2
                MethodFractalBlend -> 2
                MethodPhaseCancel -> 2
                MethodSoftmaxBlend -> 2
                MethodSmoothSoftmaxBlend -> 2
                MethodHedgeBlend -> 2
                MethodNetSoftmaxBlend -> 2
                MethodEdgeBlend -> 2
                MethodEdgePick -> 2
                MethodGeoBlend -> 2
                MethodRegimeSwitch -> 2
                MethodBlend -> 2
                MethodTaTrend -> 1
                MethodTaReversion -> 1
                MethodTaBreakout -> 1
                MethodTaBest -> 1
                MethodTaRegimeSwitch -> 1
                MethodKalmanOnly -> 1
                MethodKalmanPhysicsError -> 1
                MethodLstmOnly -> 0
        eval m =
            case sweepThresholdWithHLWith cfg m baseCfg closes highs lows kalPred lstmPred mMeta of
                Left e -> Left e
                Right (openThr, closeThr, bt, stats) ->
                    Right (tsMeanScore stats, tsStdScore stats, m, openThr, closeThr, bt, stats)
        candidates =
            [ MethodBoth
            , MethodRouter
            , MethodBanditRouter
            , MethodConfBlend
            , MethodConfPick
            , MethodConformalClip
            , MethodCostPick
            , MethodHarmonicBlend
            , MethodDisagreementGuard
            , MethodMedianBlend
            , MethodNeutralGuard
            , MethodRiskParityBlend
            , MethodConsensusBoost
            , MethodAnchorBlend
            , MethodTensionGate
            , MethodEntropyBlend
            , MethodCoherenceGate
            , MethodDivergenceGate
            , MethodFractalBlend
            , MethodPhaseCancel
            , MethodSoftmaxBlend
            , MethodSmoothSoftmaxBlend
            , MethodHedgeBlend
            , MethodNetSoftmaxBlend
            , MethodEdgeBlend
            , MethodEdgePick
            , MethodGeoBlend
            , MethodRegimeSwitch
            , MethodBlend
            , MethodKalmanOnly
            , MethodLstmOnly
            ]
        results = map eval candidates
        evaluated = Data.Either.rights results
        errors = Data.Either.lefts results
        pick (bestSc, bestStd, bestM, bestOpenThr, bestCloseThr, bestBt, bestStats) (sc, std, m, openThr, closeThr, bt, stats)
            | sc > bestSc + eps = (sc, std, m, openThr, closeThr, bt, stats)
            | abs (sc - bestSc) <= eps =
                if std < bestStd - eps
                    then (sc, std, m, openThr, closeThr, bt, stats)
                    else
                        if abs (std - bestStd) <= eps
                            then
                                let r = methodRank m
                                    bestR = methodRank bestM
                                 in if r > bestR || (r == bestR && (openThr, closeThr) > (bestOpenThr, bestCloseThr))
                                        then (sc, std, m, openThr, closeThr, bt, stats)
                                        else (bestSc, bestStd, bestM, bestOpenThr, bestCloseThr, bestBt, bestStats)
                            else (bestSc, bestStd, bestM, bestOpenThr, bestCloseThr, bestBt, bestStats)
            | otherwise = (bestSc, bestStd, bestM, bestOpenThr, bestCloseThr, bestBt, bestStats)
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
sweepThresholdWith cfg method baseCfg prices = sweepThresholdWithHLWith cfg method baseCfg prices prices prices

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
        maxCandidates = max 0 (tcMaxThresholdCandidates cfg)
        minRoundTripsReq = max 0 (tcMinRoundTrips cfg)
        ineligibleScore = -1e18 :: Double
        routerLookback = max 2 (ecRouterLookback baseCfg)
        routerRegimeMinBars = max 0 (ecRouterRegimeMinBars baseCfg)
        routerRegimeMinFraction =
            if isNaN (ecRouterRegimeMinFraction baseCfg) || isInfinite (ecRouterRegimeMinFraction baseCfg)
                then 0
                else clamp01 (ecRouterRegimeMinFraction baseCfg)
        routerMinScore = clamp01 (ecRouterMinScore baseCfg)
        routerScorePnlWeight = clamp01 (ecRouterScorePnlWeight baseCfg)
        costPerSideTotal size volPerBar =
            let s = max 0 (abs size)
             in if s <= 0
                    then 0
                    else
                        let vol = max 0 volPerBar
                            feeRate = max 0 (ecFee baseCfg)
                            feeFixed = max 0 (ecFeeFixed baseCfg)
                            feeMin = max 0 (ecFeeMin baseCfg)
                            slipBase = max 0 (ecSlippage baseCfg)
                            slipVol = max 0 (ecSlippageVolMult baseCfg) * vol
                            impactPower = max 0 (ecSlippageImpactPower baseCfg)
                            impactRate = max 0 (ecSlippageImpact baseCfg) * (s ** impactPower)
                            spreadBase = max 0 (ecSpread baseCfg)
                            spreadVol = max 0 (ecSpreadVolMult baseCfg) * vol
                            spreadTotal = spreadBase + spreadVol
                            perNotional = feeRate + slipBase + slipVol + impactRate + spreadTotal / 2
                            total0 = perNotional * s + feeFixed
                            total1 = if feeMin > 0 then max feeMin total0 else total0
                         in min 0.999999 (max 0 total1)

        volPerBarAvg =
            if stepCount <= 1
                then 0
                else
                    let rets = V.generate stepCount $ \i ->
                            let p0 = pricesV V.! i
                                p1 = pricesV V.! (i + 1)
                                r = if p0 == 0 then 0 else p1 / p0 - 1
                             in if isNaN r || isInfinite r then 0 else r
                        m = if stepCount <= 0 then 0 else V.sum rets / fromIntegral stepCount
                        var =
                            if stepCount < 2
                                then 0
                                else V.sum (V.map (\x -> (x - m) ** 2) rets) / fromIntegral (stepCount - 1)
                     in max 0 (sqrt (max 0 var))

        sizeRef = max 1e-6 (ecMaxPositionSize baseCfg)
        perSideCost =
            let total = costPerSideTotal sizeRef volPerBarAvg
             in if sizeRef <= 0 then 0 else total / sizeRef
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
        blendSoftmaxScale = max 1e-12 (ecBlendSoftmaxScale baseCfg)
        blendNetSoftmaxScale = max 1e-12 (ecBlendNetSoftmaxScale baseCfg)
        blendEdgePower = max 1e-12 (ecBlendEdgePower baseCfg)
        blendSmoothAlpha = clamp01 (ecBlendSmoothAlpha baseCfg)
        blendHedgeEta = max 0 (ecBlendHedgeEta baseCfg)
        blendHedgeMaxError = max 1e-12 (ecBlendHedgeMaxError baseCfg)
        blendDivergenceK = max 1e-12 (ecBlendDivergenceK baseCfg)
        blendRegimeHighVolCutoff = clamp01 (ecBlendRegimeHighVolCutoff baseCfg)
        blendRegimeKalmanZCutoff = max 0 (ecBlendRegimeKalmanZCutoff baseCfg)
        blendBanditExploreScale = max 0 (ecBlendBanditExploreScale baseCfg)
        blendFractalReturnClamp = max 1e-12 (ecBlendFractalReturnClamp baseCfg)
        blendFractalAlignedGain = max 0 (ecBlendFractalAlignedGain baseCfg)
        blendFractalConflictGain = max 0 (ecBlendFractalConflictGain baseCfg)
        blendCoherenceConflictFloor = max 0 (ecBlendCoherenceConflictFloor baseCfg)
        blendCoherenceConflictScale = max 0 (ecBlendCoherenceConflictScale baseCfg)
        blendCoherenceBoostThreshold = clamp01 (ecBlendCoherenceBoostThreshold baseCfg)
        blendCoherenceBoostGain = max 0 (ecBlendCoherenceBoostGain baseCfg)
        blendCoherenceBoostSpan = max 1e-12 (ecBlendCoherenceBoostSpan baseCfg)
        blendAnchorConflictBase = clamp01 (ecBlendAnchorConflictBase baseCfg)
        blendAnchorConflictScale = clamp01 (ecBlendAnchorConflictScale baseCfg)
        blendAnchorAlignedScale = clamp01 (ecBlendAnchorAlignedScale baseCfg)
        blendTensionConflictShrink = clamp01 (ecBlendTensionConflictShrink baseCfg)
        blendTensionNeutralShrink = clamp01 (ecBlendTensionNeutralShrink baseCfg)
        blendEntropyConflictFloor = clamp01 (ecBlendEntropyConflictFloor baseCfg)
        blendEntropyConflictScale = clamp01 (ecBlendEntropyConflictScale baseCfg)
        blendEntropyAlignedBase = clamp01 (ecBlendEntropyAlignedBase baseCfg)
        blendEntropyAlignedEntropyScale = clamp01 (ecBlendEntropyAlignedEntropyScale baseCfg)
        blendPhaseCancelReturnClamp = max 1e-12 (ecBlendPhaseCancelReturnClamp baseCfg)
        blendPhaseCancelConflictFloor = max 0 (ecBlendPhaseCancelConflictFloor baseCfg)
        blendPhaseCancelConflictScale = max 0 (ecBlendPhaseCancelConflictScale baseCfg)
        blendPhaseCancelAlignmentScale = max 0 (ecBlendPhaseCancelAlignmentScale baseCfg)
        blendV = blendPredictionsV blendWeight pricesV kalV lstmV
        edgeBlendV0 = edgeBlendPredictionsV blendEdgePower blendWeight pricesV kalV lstmV
        edgePickV0 = edgePickPredictionsV blendEdgePower blendWeight pricesV kalV lstmV
        costPickV0 = costPickPredictionsV blendWeight roundTripCost pricesV kalV lstmV
        harmonicBlendV0 = harmonicBlendPredictionsV blendWeight pricesV kalV lstmV
        disagreementGuardV0 = disagreementGuardPredictionsV blendWeight pricesV kalV lstmV
        medianBlendV0 = medianBlendPredictionsV blendWeight pricesV kalV lstmV
        neutralGuardV0 = neutralGuardPredictionsV blendWeight pricesV kalV lstmV
        riskParityBlendV0 = riskParityBlendPredictionsV blendWeight pricesV kalV lstmV
        consensusBoostV0 = consensusBoostPredictionsV blendWeight pricesV kalV lstmV
        anchorBlendV0 =
            anchorBlendPredictionsV
                blendAnchorConflictBase
                blendAnchorConflictScale
                blendAnchorAlignedScale
                blendWeight
                pricesV
                kalV
                lstmV
        tensionGateV0 =
            tensionGatePredictionsV
                blendTensionConflictShrink
                blendTensionNeutralShrink
                blendWeight
                pricesV
                kalV
                lstmV
        entropyBlendV0 =
            entropyBlendPredictionsV
                blendEntropyConflictFloor
                blendEntropyConflictScale
                blendEntropyAlignedBase
                blendEntropyAlignedEntropyScale
                blendWeight
                pricesV
                kalV
                lstmV
        coherenceGateV0 =
            coherenceGatePredictionsV
                blendCoherenceConflictFloor
                blendCoherenceConflictScale
                blendCoherenceBoostThreshold
                blendCoherenceBoostGain
                blendCoherenceBoostSpan
                blendWeight
                pricesV
                kalV
                lstmV
        fractalBlendV0 =
            fractalBlendPredictionsV
                blendFractalReturnClamp
                blendFractalAlignedGain
                blendFractalConflictGain
                blendWeight
                pricesV
                kalV
                lstmV
        phaseCancelV0 =
            phaseCancelPredictionsV
                blendPhaseCancelReturnClamp
                blendPhaseCancelConflictFloor
                blendPhaseCancelConflictScale
                blendPhaseCancelAlignmentScale
                blendWeight
                pricesV
                kalV
                lstmV
        softmaxBlendV0 = softmaxBlendPredictionsV blendSoftmaxScale blendWeight pricesV kalV lstmV
        smoothSoftmaxBlendV0 = smoothSoftmaxBlendPredictionsV blendSoftmaxScale blendSmoothAlpha blendWeight pricesV kalV lstmV
        netSoftmaxBlendV0 = netSoftmaxBlendPredictionsV blendNetSoftmaxScale blendWeight roundTripCost pricesV kalV lstmV
        geoBlendV0 = geometricBlendPredictionsV blendWeight pricesV kalV lstmV
        kalZMinForBlend = max 0 (ecKalmanZMin baseCfg)
        kalZMaxForBlend = max kalZMinForBlend (ecKalmanZMax baseCfg)
        confBlendOpenThr0 = max baseOpenThreshold minEdge
        confBlendV0 = confidenceBlendPredictionsV blendWeight kalZMinForBlend kalZMaxForBlend confBlendOpenThr0 pricesV kalV lstmV metaV
        confPickV0 = confidencePickPredictionsV blendWeight kalZMinForBlend kalZMaxForBlend confBlendOpenThr0 pricesV kalV lstmV metaV
        conformalClipV0 = conformalClipPredictionsV blendWeight pricesV kalV lstmV metaV
        divergenceGateV0 = divergenceGatePredictionsV blendDivergenceK blendWeight confBlendOpenThr0 pricesV kalV lstmV
        hedgeBlendV0 = hedgeBlendPredictionsV blendHedgeEta blendHedgeMaxError blendWeight pricesV kalV lstmV
        regimeSwitchV0 = regimeSwitchPredictionsV blendWeight blendRegimeHighVolCutoff blendRegimeKalmanZCutoff pricesV kalV lstmV metaV

        (kalUsedV0, lstmUsedV0) =
            case method of
                MethodBoth -> (kalV, lstmV)
                MethodRouter -> (kalV, lstmV)
                MethodBanditRouter -> (kalV, lstmV)
                MethodBlend -> (blendV, blendV)
                MethodConfBlend -> (confBlendV0, confBlendV0)
                MethodConfPick -> (confPickV0, confPickV0)
                MethodConformalClip -> (conformalClipV0, conformalClipV0)
                MethodCostPick -> (costPickV0, costPickV0)
                MethodHarmonicBlend -> (harmonicBlendV0, harmonicBlendV0)
                MethodDisagreementGuard -> (disagreementGuardV0, disagreementGuardV0)
                MethodMedianBlend -> (medianBlendV0, medianBlendV0)
                MethodNeutralGuard -> (neutralGuardV0, neutralGuardV0)
                MethodRiskParityBlend -> (riskParityBlendV0, riskParityBlendV0)
                MethodConsensusBoost -> (consensusBoostV0, consensusBoostV0)
                MethodAnchorBlend -> (anchorBlendV0, anchorBlendV0)
                MethodTensionGate -> (tensionGateV0, tensionGateV0)
                MethodEntropyBlend -> (entropyBlendV0, entropyBlendV0)
                MethodCoherenceGate -> (coherenceGateV0, coherenceGateV0)
                MethodDivergenceGate -> (divergenceGateV0, divergenceGateV0)
                MethodFractalBlend -> (fractalBlendV0, fractalBlendV0)
                MethodPhaseCancel -> (phaseCancelV0, phaseCancelV0)
                MethodSoftmaxBlend -> (softmaxBlendV0, softmaxBlendV0)
                MethodSmoothSoftmaxBlend -> (smoothSoftmaxBlendV0, smoothSoftmaxBlendV0)
                MethodHedgeBlend -> (hedgeBlendV0, hedgeBlendV0)
                MethodNetSoftmaxBlend -> (netSoftmaxBlendV0, netSoftmaxBlendV0)
                MethodEdgeBlend -> (edgeBlendV0, edgeBlendV0)
                MethodEdgePick -> (edgePickV0, edgePickV0)
                MethodGeoBlend -> (geoBlendV0, geoBlendV0)
                MethodRegimeSwitch -> (regimeSwitchV0, regimeSwitchV0)
                MethodTaTrend -> (kalV, kalV)
                MethodTaReversion -> (kalV, kalV)
                MethodTaBreakout -> (kalV, kalV)
                MethodTaBest -> (kalV, kalV)
                MethodTaRegimeSwitch -> (kalV, kalV)
                MethodKalmanOnly -> (kalV, kalV)
                MethodKalmanPhysicsError -> (kalV, kalV)
                MethodLstmOnly -> (lstmV, lstmV)

        validationError
            | n < 2 = Just "sweepThreshold: need at least 2 prices"
            | V.length highsV /= n || V.length lowsV /= n = Just "sweepThreshold: high/low series must match closes length"
            | maybe False (\mv -> V.length mv < stepCount) metaUsed = Just "sweepThreshold: meta vector too short"
            | otherwise = case method of
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
                MethodBanditRouter
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
                MethodConfBlend
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
                MethodConfPick
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
                MethodConformalClip
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
                MethodCostPick
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
                MethodHarmonicBlend
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
                MethodDisagreementGuard
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
                MethodMedianBlend
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
                MethodNeutralGuard
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
                MethodRiskParityBlend
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
                MethodConsensusBoost
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
                MethodAnchorBlend
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
                MethodTensionGate
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
                MethodEntropyBlend
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
                MethodCoherenceGate
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
                MethodDivergenceGate
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
                MethodFractalBlend
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
                MethodPhaseCancel
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
                MethodSoftmaxBlend
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
                MethodSmoothSoftmaxBlend
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
                MethodHedgeBlend
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
                MethodNetSoftmaxBlend
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
                MethodEdgeBlend
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
                MethodEdgePick
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
                MethodGeoBlend
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
                MethodRegimeSwitch
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
                MethodKalmanPhysicsError
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
                MethodTaTrend
                    | V.length kalV < stepCount ->
                        Just
                            ( "sweepThreshold: taPred has length "
                                ++ show (V.length kalV)
                                ++ " but needs at least "
                                ++ show stepCount
                            )
                    | otherwise -> Nothing
                MethodTaReversion
                    | V.length kalV < stepCount ->
                        Just
                            ( "sweepThreshold: taPred has length "
                                ++ show (V.length kalV)
                                ++ " but needs at least "
                                ++ show stepCount
                            )
                    | otherwise -> Nothing
                MethodTaBreakout
                    | V.length kalV < stepCount ->
                        Just
                            ( "sweepThreshold: taPred has length "
                                ++ show (V.length kalV)
                                ++ " but needs at least "
                                ++ show stepCount
                            )
                    | otherwise -> Nothing
                MethodTaBest
                    | V.length kalV < stepCount ->
                        Just
                            ( "sweepThreshold: taPred has length "
                                ++ show (V.length kalV)
                                ++ " but needs at least "
                                ++ show stepCount
                            )
                    | otherwise -> Nothing
                MethodTaRegimeSwitch
                    | V.length kalV < stepCount ->
                        Just
                            ( "sweepThreshold: taPred has length "
                                ++ show (V.length kalV)
                                ++ " but needs at least "
                                ++ show stepCount
                            )
                    | otherwise -> Nothing

        predSources =
            case method of
                MethodBoth -> [kalV, lstmV]
                MethodRouter -> [kalV, lstmV, blendV]
                MethodBanditRouter -> [kalV, lstmV, blendV]
                MethodBlend -> [blendV]
                MethodConfBlend -> [confBlendV0]
                MethodConfPick -> [confPickV0]
                MethodConformalClip -> [conformalClipV0]
                MethodCostPick -> [costPickV0]
                MethodHarmonicBlend -> [harmonicBlendV0]
                MethodDisagreementGuard -> [disagreementGuardV0]
                MethodMedianBlend -> [medianBlendV0]
                MethodNeutralGuard -> [neutralGuardV0]
                MethodRiskParityBlend -> [riskParityBlendV0]
                MethodConsensusBoost -> [consensusBoostV0]
                MethodAnchorBlend -> [anchorBlendV0]
                MethodTensionGate -> [tensionGateV0]
                MethodEntropyBlend -> [entropyBlendV0]
                MethodCoherenceGate -> [coherenceGateV0]
                MethodDivergenceGate -> [divergenceGateV0]
                MethodFractalBlend -> [fractalBlendV0]
                MethodPhaseCancel -> [phaseCancelV0]
                MethodSoftmaxBlend -> [softmaxBlendV0]
                MethodSmoothSoftmaxBlend -> [smoothSoftmaxBlendV0]
                MethodHedgeBlend -> [hedgeBlendV0]
                MethodNetSoftmaxBlend -> [netSoftmaxBlendV0]
                MethodEdgeBlend -> [edgeBlendV0]
                MethodEdgePick -> [edgePickV0]
                MethodGeoBlend -> [geoBlendV0]
                MethodRegimeSwitch -> [regimeSwitchV0]
                MethodTaTrend -> [kalV]
                MethodTaReversion -> [kalV]
                MethodTaBreakout -> [kalV]
                MethodTaBest -> [kalV]
                MethodTaRegimeSwitch -> [kalV]
                MethodKalmanOnly -> [kalV]
                MethodKalmanPhysicsError -> [kalV]
                MethodLstmOnly -> [lstmV]
        epsilonFor v =
            let rel = abs v * 1e-9
             in max eps rel
        magsSet =
            foldl'
                ( \acc t ->
                    let prev = pricesV V.! t
                     in if prev == 0
                            then acc
                            else
                                foldl'
                                    ( \acc' predsV ->
                                        let pred = predsV V.! t
                                            v = abs (pred / prev - 1)
                                            headroomCap = signalEntryHeadroomThresholdCap v
                                         in if isNaN v || isInfinite v
                                                then acc'
                                                else
                                                    Set.insert (max 0 (headroomCap - epsilonFor headroomCap)) $
                                                        Set.insert (max 0 (v - epsilonFor v)) acc'
                                    )
                                    acc
                                    predSources
                )
                Set.empty
                [0 .. stepCount - 1]
        candidates0 = Set.toAscList (Set.insert 0 magsSet)
        isFinite v = not (isNaN v || isInfinite v)
        baseCandidates = filter isFinite [baseOpenThreshold, baseCloseThreshold]
        candidates =
            Set.toAscList
                (Set.fromList (baseCandidates ++ downsample maxCandidates candidates0))
        ppy = max 1e-12 (tcPeriodsPerYear cfg)
        emptyBacktest =
            BacktestResult
                { brEquityCurve = [1]
                , brPositions = []
                , brAgreementOk = []
                , brAgreementValid = []
                , brPositionChanges = 0
                , brCostAttribution = emptyBacktestCostAttribution [1]
                , brTrades = []
                }
        emptyMetrics = computeMetrics ppy emptyBacktest
        foldsReq = max 1 (tcWalkForwardFolds cfg)
        foldRs = foldRanges stepCount foldsReq
        embargoBars = max 0 (tcWalkForwardEmbargoBars cfg)
        applyEmbargo e (t0, t1) =
            let t0' = t0 + e
                t1' = t1 - e
             in if t1' < t0' then Nothing else Just (t0', t1')
        foldSingle =
            case foldRs of
                [] -> True
                [_] -> True
                _ -> False
        foldRsEval =
            if foldSingle || embargoBars <= 0
                then foldRs
                else mapMaybe (applyEmbargo embargoBars) foldRs
        foldEvalOk = foldSingle || not (null foldRsEval)

        lstmFlipEnabled =
            case method of
                MethodBoth -> True
                MethodLstmOnly -> True
                _ -> False

        applyLstmFlip cfg =
            if lstmFlipEnabled
                then cfg
                else cfg{ecLstmExitFlipBars = 0, ecLstmExitFlipGraceBars = 0}

        evalForOpen openThr =
            let (kalUsedV, lstmUsedV, metaMask) =
                    case method of
                        MethodRouter ->
                            let routerOpenThr = max openThr minEdge
                                (routerPredV, routerModelsV) =
                                    routerPredictionsWithModelsV
                                        blendRegimeHighVolCutoff
                                        routerOpenThr
                                        roundTripCost
                                        routerScorePnlWeight
                                        routerLookback
                                        routerRegimeMinBars
                                        routerRegimeMinFraction
                                        routerMinScore
                                        pricesV
                                        kalV
                                        lstmV
                                        blendV
                                        metaV
                                routerMaskV = V.map (== Just RouterKalman) routerModelsV
                             in (routerPredV, routerPredV, Just routerMaskV)
                        MethodBanditRouter ->
                            let routerOpenThr = max openThr minEdge
                                (routerPredV, routerModelsV) =
                                    banditPredictionsWithModelsV
                                        blendRegimeHighVolCutoff
                                        blendBanditExploreScale
                                        routerOpenThr
                                        roundTripCost
                                        routerScorePnlWeight
                                        routerLookback
                                        routerRegimeMinBars
                                        routerRegimeMinFraction
                                        routerMinScore
                                        pricesV
                                        kalV
                                        lstmV
                                        blendV
                                        metaV
                                routerMaskV = V.map (== Just RouterKalman) routerModelsV
                             in (routerPredV, routerPredV, Just routerMaskV)
                        MethodConfBlend ->
                            let confBlendOpenThr = max openThr minEdge
                                confBlendV =
                                    confidenceBlendPredictionsV
                                        blendWeight
                                        kalZMinForBlend
                                        kalZMaxForBlend
                                        confBlendOpenThr
                                        pricesV
                                        kalV
                                        lstmV
                                        metaV
                             in (confBlendV, confBlendV, Nothing)
                        MethodConfPick ->
                            let confPickOpenThr = max openThr minEdge
                                confPickV =
                                    confidencePickPredictionsV
                                        blendWeight
                                        kalZMinForBlend
                                        kalZMaxForBlend
                                        confPickOpenThr
                                        pricesV
                                        kalV
                                        lstmV
                                        metaV
                             in (confPickV, confPickV, Nothing)
                        MethodDivergenceGate ->
                            let divergenceGateOpenThr = max openThr minEdge
                                divergenceGateV =
                                    divergenceGatePredictionsV
                                        blendDivergenceK
                                        blendWeight
                                        divergenceGateOpenThr
                                        pricesV
                                        kalV
                                        lstmV
                             in (divergenceGateV, divergenceGateV, Nothing)
                        MethodRegimeSwitch ->
                            let regimeSwitchV =
                                    regimeSwitchPredictionsV
                                        blendWeight
                                        blendRegimeHighVolCutoff
                                        blendRegimeKalmanZCutoff
                                        pricesV
                                        kalV
                                        lstmV
                                        metaV
                             in (regimeSwitchV, regimeSwitchV, Nothing)
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
                                            ((minRoundTripsReq <= 0) || (bmRoundTrips metrics' >= minRoundTripsReq))
                                        eligible'' = eligible' && foldEvalOk
                                        foldScores'
                                            | not eligible'' = [ineligibleScore]
                                            | foldSingle = [scoreBacktest cfg btFull']
                                            | otherwise =
                                                [ let steps = t1 - t0 + 1
                                                      pricesF = V.slice t0 (steps + 1) pricesV
                                                      highsF = V.slice t0 (steps + 1) highsV
                                                      lowsF = V.slice t0 (steps + 1) lowsV
                                                      kalF = V.slice t0 steps kalUsedV
                                                      lstmF = V.slice t0 steps lstmUsedV
                                                      metaF = fmap (V.slice t0 steps) metaUsed
                                                      openTimesF = fmap (V.slice t0 (steps + 1)) (ecOpenTimes btCfg)
                                                      metaMaskF = fmap (V.slice t0 steps) metaMask
                                                      openPricesF = fmap (V.slice t0 (steps + 1)) (ecOpenPrices btCfg)
                                                      btCfgFold = btCfg{ecOpenTimes = openTimesF, ecOpenPrices = openPricesF, ecMetaMask = metaMaskF}
                                                      btFoldE = simulateEnsembleVWithHLChecked btCfgFold 1 pricesF highsF lowsF kalF lstmF metaF
                                                   in case btFoldE of
                                                        Left _ -> ineligibleScore
                                                        Right btFold -> scoreBacktest cfg btFold
                                                | (t0, t1) <- foldRsEval
                                                , t1 >= t0
                                                ]
                                     in (btFull', metrics', eligible'', foldScores')
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
            preferTieBreakImplementation
                (tieBreakCandidateFromMetrics metrics openThr closeThr)
                (tieBreakCandidateFromMetrics bestMetrics bestOpen bestClose)
        pickResult (bestEligible, bestMean, bestStd, bestOpenThr, bestCloseThr, bestBt, bestStats, bestMetrics) (eligible, m, s, openThr', closeThr', bt, stats, metrics) =
            case (bestEligible, eligible) of
                (False, True) -> (True, m, s, openThr', closeThr', bt, stats, metrics)
                (True, False) -> (bestEligible, bestMean, bestStd, bestOpenThr, bestCloseThr, bestBt, bestStats, bestMetrics)
                _
                    | m > bestMean + eqEps -> (eligible, m, s, openThr', closeThr', bt, stats, metrics)
                    | abs (m - bestMean) <= eqEps
                    , s < bestStd - eqEps
                        || (abs (s - bestStd) <= eqEps && preferTie metrics openThr' closeThr' bestMetrics bestOpenThr bestCloseThr) ->
                        (eligible, m, s, openThr', closeThr', bt, stats, metrics)
                    | otherwise -> (bestEligible, bestMean, bestStd, bestOpenThr, bestCloseThr, bestBt, bestStats, bestMetrics)

        foldClose acc openThr =
            let evalClose = evalForOpen openThr
             in foldl' (\acc0 closeThr -> pickResult acc0 (evalClose closeThr)) acc candidates
        (bestEligible, _, _, bestOpenThr, bestCloseThr, bestBt, bestStats, _bestMetrics) =
            foldl' foldClose (baseEligible, baseMean, baseStd, baseOpenThr, baseCloseThr, baseBt, baseStats, baseMetrics) candidates

        result = (bestOpenThr, bestCloseThr, bestBt, bestStats)
     in case validationError of
            Just err -> Left err
            Nothing -> Right result

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
    }
    deriving (Eq, Show)

routerStatsWindowWith :: Double -> Double -> Double -> V.Vector Double -> V.Vector Double -> (Int -> Bool) -> Int -> Int -> RouterStats
routerStatsWindowWith openThr roundTripCost pnlWeight pricesV predsV useIdx start0 end0 =
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
        step (correct, wrong, signals, netAcc, netAccSq, bars) i =
            if not (useIdx i)
                then (correct, wrong, signals, netAcc, netAccSq, bars)
                else
                    let bars' = bars + 1
                        prev = pricesV V.! i
                        next = pricesV V.! (i + 1)
                        pred = predsV V.! i
                        predDir = direction prev pred
                        actualDir = direction prev next
                        ret = if prev <= 0 || bad prev || bad next then 0 else next / prev - 1
                     in case predDir of
                            Nothing -> (correct, wrong, signals, netAcc, netAccSq, bars')
                            Just dir ->
                                let signals' = signals + 1
                                    net = fromIntegral dir * ret - roundTripCost
                                    netAcc' = netAcc + if bad net then 0 else net
                                    netAccSq' = netAccSq + if bad net then 0 else net * net
                                 in if actualDir == Just dir
                                        then (correct + 1, wrong, signals', netAcc', netAccSq', bars')
                                        else (correct, wrong + 1, signals', netAcc', netAccSq', bars')
     in if stepCount <= 0 || end < start
            then RouterStats{rsScore = 0, rsAccuracy = 0, rsCoverage = 0, rsSignals = 0}
            else
                let (correct, _wrong, signals, netAcc, netAccSq, bars) = foldl' step (0, 0, 0, 0, 0, 0) [start .. end]
                    accuracy =
                        if signals <= 0
                            then 0
                            else fromIntegral correct / fromIntegral signals
                    coverage =
                        if bars <= 0
                            then 0
                            else fromIntegral signals / fromIntegral bars
                    avgNet =
                        if signals <= 0
                            then 0
                            else netAcc / fromIntegral signals
                    meanSq =
                        if signals <= 0
                            then 0
                            else netAccSq / fromIntegral signals
                    varNet = max 0 (meanSq - avgNet * avgNet)
                    stdNet = sqrt varNet
                    denom = max 1e-12 (openThr + roundTripCost)
                    riskAdj =
                        if signals <= 0
                            then 0
                            else avgNet / (stdNet + denom)
                    pnlScore =
                        if signals <= 0
                            then 0
                            else clamp01 (0.5 + 0.5 * riskAdj)
                    pnlWeight' = clamp01 pnlWeight
                    scoreAcc = accuracy * coverage
                    score = (1 - pnlWeight') * scoreAcc + pnlWeight' * pnlScore
                 in RouterStats{rsScore = score, rsAccuracy = accuracy, rsCoverage = coverage, rsSignals = signals}

routerStatsWindow :: Double -> Double -> Double -> V.Vector Double -> V.Vector Double -> Int -> Int -> RouterStats
routerStatsWindow openThr roundTripCost pnlWeight pricesV predsV =
    routerStatsWindowWith openThr roundTripCost pnlWeight pricesV predsV (const True)

routerRegimeMinBarsFromKnobs :: Int -> Double -> Int -> Int
routerRegimeMinBarsFromKnobs minBarsRaw minFractionRaw lookback =
    let minBars = max 0 minBarsRaw
        minFraction =
            if isNaN minFractionRaw || isInfinite minFractionRaw
                then 0
                else clamp01 minFractionRaw
        fractionBars = floor (fromIntegral (max 0 lookback) * minFraction)
     in max minBars fractionBars

routerSelectModelAt ::
    Double ->
    Double ->
    Double ->
    Double ->
    Int ->
    Int ->
    Double ->
    Double ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double ->
    Maybe (V.Vector StepMeta) ->
    Int ->
    (Maybe RouterModel, Double, Maybe String)
routerSelectModelAt highVolCutoffRaw openThr roundTripCost pnlWeight lookback0 regimeMinBars0 regimeMinFraction0 minScore0 pricesV kalPredV lstmPredV blendPredV mMetaV t =
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
        volCutoff = clamp01 highVolCutoffRaw
        minRegimeBars = routerRegimeMinBarsFromKnobs regimeMinBars0 regimeMinFraction0 lookback
        regimeAt i =
            case mMetaV of
                Just metaV
                    | i >= 0 && i < V.length metaV ->
                        case smHighVolProb (metaV V.! i) of
                            Just hv -> Just (hv >= volCutoff)
                            Nothing -> Nothing
                _ -> Nothing
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
     in if stepCount <= 0 || windowEnd < 0
            then (Nothing, 0, Just "ROUTER_WARMUP")
            else
                let windowStart = max 0 (windowEnd - lookback + 1)
                    mRegNow = regimeAt windowEnd
                    sameReg i = regimeAt i == mRegNow
                    regimeBars =
                        case mRegNow of
                            Nothing -> 0
                            Just _ -> length [i | i <- [windowStart .. windowEnd], sameReg i]
                    useRegime = regimeBars >= minRegimeBars
                    useIdx = if useRegime then sameReg else const True
                    statsKal = routerStatsWindowWith openThr roundTripCost pnlWeight pricesV kalPredV useIdx windowStart windowEnd
                    statsLstm = routerStatsWindowWith openThr roundTripCost pnlWeight pricesV lstmPredV useIdx windowStart windowEnd
                    statsBlend = routerStatsWindowWith openThr roundTripCost pnlWeight pricesV blendPredV useIdx windowStart windowEnd
                    (bestModel, bestStats) =
                        foldl' pick (RouterKalman, statsKal) [(RouterLstm, statsLstm), (RouterBlend, statsBlend)]
                    bestScore = rsScore bestStats
                 in if bestScore < minScore
                        then (Nothing, bestScore, Just "ROUTER_MIN_SCORE")
                        else (Just bestModel, bestScore, Nothing)

routerPredictionsWithModelsV ::
    Double ->
    Double ->
    Double ->
    Double ->
    Int ->
    Int ->
    Double ->
    Double ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double ->
    Maybe (V.Vector StepMeta) ->
    (V.Vector Double, V.Vector (Maybe RouterModel))
routerPredictionsWithModelsV highVolCutoff openThr roundTripCost pnlWeight lookback regimeMinBars regimeMinFraction minScore pricesV kalPredV lstmPredV blendPredV mMetaV =
    let stepCount =
            minimum
                [ V.length pricesV - 1
                , V.length kalPredV
                , V.length lstmPredV
                , V.length blendPredV
                ]
        pickPred t =
            case routerSelectModelAt highVolCutoff openThr roundTripCost pnlWeight lookback regimeMinBars regimeMinFraction minScore pricesV kalPredV lstmPredV blendPredV mMetaV t of
                (Just RouterKalman, _, _) -> (kalPredV V.! t, Just RouterKalman)
                (Just RouterLstm, _, _) -> (lstmPredV V.! t, Just RouterLstm)
                (Just RouterBlend, _, _) -> (blendPredV V.! t, Just RouterBlend)
                _ -> (pricesV V.! t, Nothing)
        picks = V.generate (max 0 stepCount) pickPred
     in (V.map fst picks, V.map snd picks)

banditSelectModelAt ::
    Double ->
    Double ->
    Double ->
    Double ->
    Double ->
    Int ->
    Int ->
    Double ->
    Double ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double ->
    Maybe (V.Vector StepMeta) ->
    Int ->
    (Maybe RouterModel, Double, Maybe String)
banditSelectModelAt highVolCutoffRaw exploreScaleRaw openThr roundTripCost pnlWeight lookback0 regimeMinBars0 regimeMinFraction0 minScore0 pricesV kalPredV lstmPredV blendPredV mMetaV t =
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
        volCutoff = clamp01 highVolCutoffRaw
        minRegimeBars = routerRegimeMinBarsFromKnobs regimeMinBars0 regimeMinFraction0 lookback
        regimeAt i =
            case mMetaV of
                Just metaV
                    | i >= 0 && i < V.length metaV ->
                        case smHighVolProb (metaV V.! i) of
                            Just hv -> Just (hv >= volCutoff)
                            Nothing -> Nothing
                _ -> Nothing
        modelRank m =
            case m of
                RouterBlend -> 2 :: Int
                RouterKalman -> 1
                RouterLstm -> 0
        bonusScale = max 0 exploreScaleRaw
        scoreKey totalSignals (m, stats) =
            let n = max 0 (rsSignals stats)
                explore = sqrt (2 * log (max 2 totalSignals) / fromIntegral (n + 1))
                score = rsScore stats + bonusScale * explore
             in (score, rsScore stats, rsCoverage stats, rsAccuracy stats, modelRank m)
        pick totalSignals best cand =
            if scoreKey totalSignals cand > scoreKey totalSignals best
                then cand
                else best
     in if stepCount <= 0 || windowEnd < 0
            then (Nothing, 0, Just "BANDIT_WARMUP")
            else
                let windowStart = max 0 (windowEnd - lookback + 1)
                    mRegNow = regimeAt windowEnd
                    sameReg i = regimeAt i == mRegNow
                    regimeBars =
                        case mRegNow of
                            Nothing -> 0
                            Just _ -> length [i | i <- [windowStart .. windowEnd], sameReg i]
                    useRegime = regimeBars >= minRegimeBars
                    useIdx = if useRegime then sameReg else const True
                    statsKal = routerStatsWindowWith openThr roundTripCost pnlWeight pricesV kalPredV useIdx windowStart windowEnd
                    statsLstm = routerStatsWindowWith openThr roundTripCost pnlWeight pricesV lstmPredV useIdx windowStart windowEnd
                    statsBlend = routerStatsWindowWith openThr roundTripCost pnlWeight pricesV blendPredV useIdx windowStart windowEnd
                    totalSignals = fromIntegral (1 + rsSignals statsKal + rsSignals statsLstm + rsSignals statsBlend)
                    (bestModel, bestStats) =
                        foldl' (pick totalSignals) (RouterKalman, statsKal) [(RouterLstm, statsLstm), (RouterBlend, statsBlend)]
                    bestScore = rsScore bestStats
                 in if bestScore < minScore
                        then (Nothing, bestScore, Just "BANDIT_MIN_SCORE")
                        else (Just bestModel, bestScore, Nothing)

banditPredictionsWithModelsV ::
    Double ->
    Double ->
    Double ->
    Double ->
    Double ->
    Int ->
    Int ->
    Double ->
    Double ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double ->
    Maybe (V.Vector StepMeta) ->
    (V.Vector Double, V.Vector (Maybe RouterModel))
banditPredictionsWithModelsV highVolCutoff exploreScale openThr roundTripCost pnlWeight lookback regimeMinBars regimeMinFraction minScore pricesV kalPredV lstmPredV blendPredV mMetaV =
    let stepCount =
            minimum
                [ V.length pricesV - 1
                , V.length kalPredV
                , V.length lstmPredV
                , V.length blendPredV
                ]
        pickPred t =
            case banditSelectModelAt highVolCutoff exploreScale openThr roundTripCost pnlWeight lookback regimeMinBars regimeMinFraction minScore pricesV kalPredV lstmPredV blendPredV mMetaV t of
                (Just RouterKalman, _, _) -> (kalPredV V.! t, Just RouterKalman)
                (Just RouterLstm, _, _) -> (lstmPredV V.! t, Just RouterLstm)
                (Just RouterBlend, _, _) -> (blendPredV V.! t, Just RouterBlend)
                _ -> (pricesV V.! t, Nothing)
        picks = V.generate (max 0 stepCount) pickPred
     in (V.map fst picks, V.map snd picks)
