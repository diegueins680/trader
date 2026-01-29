module Trader.Trading (
    Positioning (..),
    EnsembleConfig (..),
    IntrabarFill (..),
    PositionSide (..),
    ExitReason (..),
    exitReasonCode,
    exitReasonFromCode,
    StepMeta (..),
    Trade (..),
    BacktestResult (..),
    simulateEnsemble,
    simulateEnsembleV,
    simulateEnsembleWithHL,
    simulateEnsembleWithHLChecked,
    simulateEnsembleVWithHL,
    simulateEnsembleVWithHLChecked,
    simulateEnsembleLongFlat,
    simulateEnsembleLongFlatV,
    simulateEnsembleLongFlatWithHL,
    simulateEnsembleLongFlatVWithHL,
) where

import Data.Aeson (FromJSON (..), ToJSON (..), withText)
import qualified Data.Aeson as Aeson
import Data.Int (Int64)
import Data.List (foldl')
import qualified Data.Text as T
import qualified Data.Vector as V
import Trader.Duration (TimeWindow, minuteOfDayFromMs, timeWindowContains)
import Trader.Kalman3 (KalmanRunV (..), runConstantAcceleration1DVec)

data Positioning
    = LongFlat
    | LongShort
    deriving (Eq, Show)

data EnsembleConfig = EnsembleConfig
    { ecOpenThreshold :: !Double
    , ecCloseThreshold :: !Double
    , ecFee :: !Double
    , ecSlippage :: !Double -- fractional per side, e.g. 0.0002
    , ecSpread :: !Double -- fractional total spread, e.g. 0.0005 (half per side)
    , ecStopLoss :: !(Maybe Double) -- fractional, e.g. 0.02
    , ecTakeProfit :: !(Maybe Double) -- fractional, e.g. 0.03
    , ecTrailingStop :: !(Maybe Double) -- fractional, e.g. 0.01
    , ecStopLossVolMult :: !Double -- per-bar sigma multiple (0 disables)
    , ecTakeProfitVolMult :: !Double -- per-bar sigma multiple (0 disables)
    , ecTrailingStopVolMult :: !Double -- per-bar sigma multiple (0 disables)
    , ecMinHoldBars :: !Int -- bars; 0 disables (signal exits allowed immediately)
    , ecCooldownBars :: !Int -- bars; 0 disables (wait after exiting before re-entering)
    , ecMaxHoldBars :: !(Maybe Int) -- bars; 0 disables (force exit after N bars)
    , ecMaxDrawdown :: !(Maybe Double) -- fraction, e.g. 0.2 (20%); halts and exits to flat
    , ecMaxDailyLoss :: !(Maybe Double) -- fraction, e.g. 0.05 (5%); halts and exits to flat
    , ecMaxWeeklyLoss :: !(Maybe Double) -- fraction, e.g. 0.1 (10%); halts and exits to flat
    , ecRiskPerTrade :: !(Maybe Double) -- fraction, e.g. 0.01 (1%); sizes via stop-loss fraction
    , ecMaxTradesPerDay :: !(Maybe Int) -- number of entries per day; 0 disables
    , ecExpectancyLookback :: !Int -- trades; 0 disables expectancy gating
    , ecMinExpectancy :: !(Maybe Double) -- avg trade return threshold
    , ecNoTradeWindows :: ![TimeWindow] -- UTC windows to block entries
    , ecIntervalSeconds :: !(Maybe Int) -- inferred from CLI interval; UTC day/week windows require bar timestamps
    , ecOpenTimes :: !(Maybe (V.Vector Int64)) -- optional bar open times (ms since epoch) for daily-loss day keys
    , ecMetaMask :: !(Maybe (V.Vector Bool)) -- optional per-bar mask to apply Kalman meta gating
    , ecPositioning :: !Positioning
    , ecIntrabarFill :: !IntrabarFill
    , ecMaxPositionSize :: !Double -- 0..N, caps position sizing (1=full)
    , ecMinEdge :: !Double -- min predicted return magnitude required for entry
    , ecMinSignalToNoise :: !Double -- min edge / per-bar sigma required for entry
    , ecSnrSizeWeight :: !Double -- weight for soft SNR sizing (0=hard gate only, 1=fully scaled)
    -- Dynamic threshold factor (multiplicative)
    , ecThresholdFactorEnabled :: !Bool
    , ecThresholdFactorAlpha :: !Double
    , ecThresholdFactorMin :: !Double
    , ecThresholdFactorMax :: !Double
    , ecThresholdFactorFloor :: !Double
    , ecThresholdFactorEdgeKalWeight :: !Double
    , ecThresholdFactorEdgeLstmWeight :: !Double
    , ecThresholdFactorKalmanZWeight :: !Double
    , ecThresholdFactorHighVolWeight :: !Double
    , ecThresholdFactorConformalWeight :: !Double
    , ecThresholdFactorQuantileWeight :: !Double
    , ecThresholdFactorLstmConfWeight :: !Double
    , ecThresholdFactorLstmHealthWeight :: !Double
    , ecLstmTrainingHealth :: !(Maybe Double)
    , ecTrendLookback :: !Int -- bars; 0 disables
    , ecPeriodsPerYear :: !Double -- for annualized volatility sizing
    , ecVolTarget :: !(Maybe Double) -- target annualized volatility (0 disables)
    , ecVolLookback :: !Int -- bars; used when EWMA alpha is not set
    , ecVolEwmaAlpha :: !(Maybe Double) -- 0..1, uses EWMA variance when set
    , ecVolFloor :: !Double -- annualized volatility floor
    , ecVolScaleMax :: !Double -- caps volatility scaling
    , ecMaxVolatility :: !(Maybe Double) -- if set, block trades above this annualized vol
    , ecRebalanceBars :: !Int -- bars; 0 disables size rebalancing
    , ecRebalanceThreshold :: !Double -- min abs size delta required to rebalance
    , ecRebalanceGlobal :: !Bool -- when True, rebalance cadence anchors to global bars
    , ecRebalanceResetOnSignal :: !Bool -- when True, reset rebalance anchor on same-side open signals
    , ecFundingRate :: !Double -- annualized funding/borrow rate (fraction; 0 disables)
    , ecFundingBySide :: !Bool -- when True, apply funding sign by side
    , ecFundingOnOpen :: !Bool -- when True, charge funding for bars opened with a position
    , ecBlendWeight :: !Double -- Kalman weight for blend method
    , ecRouterLookback :: !Int -- bars; router scoring window
    , ecRouterMinScore :: !Double -- 0..1; router min score for accepting a model
    , ecRouterScorePnlWeight :: !Double -- 0..1; router score blend weight for PnL-aware scoring
    -- Tri-layer gating (Kalman cloud + price action trigger)
    , ecKalmanDt :: !Double
    , ecKalmanProcessVar :: !Double
    , ecKalmanMeasurementVar :: !Double
    , ecTriLayer :: !Bool
    , ecTriLayerFastMult :: !Double
    , ecTriLayerSlowMult :: !Double
    , ecTriLayerCloudPadding :: !Double
    , ecTriLayerCloudSlope :: !Double
    , ecTriLayerCloudWidth :: !Double
    , ecTriLayerTouchLookback :: !Int
    , ecTriLayerRequirePriceAction :: !Bool
    , ecTriLayerPriceActionBody :: !Double
    , ecTriLayerExitOnSlow :: !Bool
    , ecKalmanBandLookback :: !Int
    , ecKalmanBandStdMult :: !Double
    , -- LSTM flip exit
      ecLstmExitFlipBars :: !Int
    , ecLstmExitFlipGraceBars :: !Int
    , ecLstmExitFlipStrong :: !Bool
    , ecLstmConfidenceSoft :: !Double
    , ecLstmConfidenceHard :: !Double
    , -- Confidence gating/sizing (Kalman sensors + HMM/intervals)
      ecKalmanZMin :: !Double
    , ecKalmanZMax :: !Double
    , ecMaxHighVolProb :: !(Maybe Double)
    , ecMaxConformalWidth :: !(Maybe Double)
    , ecMaxQuantileWidth :: !(Maybe Double)
    , ecConfirmConformal :: !Bool
    , ecConfirmQuantiles :: !Bool
    , ecConfidenceSizing :: !Bool
    , ecMinPositionSize :: !Double
    }
    deriving (Eq, Show)

data IntrabarFill
    = StopFirst
    | TakeProfitFirst
    deriving (Eq, Show)

data PositionSide
    = SideLong
    | SideShort
    deriving (Eq, Show)

sideSign :: PositionSide -> Double
sideSign side =
    case side of
        SideLong -> 1
        SideShort -> -1

data ExitReason
    = ExitSignal
    | ExitStopLoss
    | ExitTrailingStop
    | ExitTakeProfit
    | ExitMaxDrawdown
    | ExitMaxDailyLoss
    | ExitMaxWeeklyLoss
    | ExitLiquidation
    | ExitEod
    | ExitOther !String
    deriving (Eq, Show)

exitReasonCode :: ExitReason -> String
exitReasonCode why =
    case why of
        ExitSignal -> "SIGNAL"
        ExitStopLoss -> "STOP_LOSS"
        ExitTrailingStop -> "TRAILING_STOP"
        ExitTakeProfit -> "TAKE_PROFIT"
        ExitMaxDrawdown -> "MAX_DRAWDOWN"
        ExitMaxDailyLoss -> "MAX_DAILY_LOSS"
        ExitMaxWeeklyLoss -> "MAX_WEEKLY_LOSS"
        ExitLiquidation -> "LIQUIDATION"
        ExitEod -> "EOD"
        ExitOther s -> s

exitReasonFromCode :: String -> ExitReason
exitReasonFromCode code =
    case code of
        "SIGNAL" -> ExitSignal
        "STOP_LOSS" -> ExitStopLoss
        "TRAILING_STOP" -> ExitTrailingStop
        "TAKE_PROFIT" -> ExitTakeProfit
        "MAX_DRAWDOWN" -> ExitMaxDrawdown
        "MAX_DAILY_LOSS" -> ExitMaxDailyLoss
        "MAX_WEEKLY_LOSS" -> ExitMaxWeeklyLoss
        "LIQUIDATION" -> ExitLiquidation
        "EOD" -> ExitEod
        other -> ExitOther other

instance ToJSON ExitReason where
    toJSON = Aeson.String . T.pack . exitReasonCode

instance FromJSON ExitReason where
    parseJSON = withText "ExitReason" (pure . exitReasonFromCode . T.unpack)

data StepMeta = StepMeta
    { smKalmanMean :: !Double
    , smKalmanVar :: !Double
    , smHighVolProb :: !(Maybe Double)
    , smQuantile10 :: !(Maybe Double)
    , smQuantile90 :: !(Maybe Double)
    , smConformalLo :: !(Maybe Double)
    , smConformalHi :: !(Maybe Double)
    }
    deriving (Eq, Show)

data Trade = Trade
    { trEntryIndex :: !Int
    , trExitIndex :: !Int
    , trEntryEquity :: !Double
    , trExitEquity :: !Double
    , trReturn :: !Double
    , trHoldingPeriods :: !Int
    , trExitReason :: !(Maybe ExitReason)
    }
    deriving (Eq, Show)

data OpenTrade = OpenTrade
    { otEntryIndex :: !Int
    , otRebalanceAnchor :: !Int
    , otEntryEquity :: !Double
    , otHoldingPeriods :: !Int
    , otEntryPrice :: !Double
    , otTrail :: !Double
    , otSide :: !PositionSide
    , otBaseSize :: !Double
    , otLstmFlipCount :: !Int
    }
    deriving (Eq, Show)

data RiskState = RiskState
    { rsPeakEquity :: !Double
    , rsDayKey :: !Int
    , rsDayStartEquity :: !Double
    , rsWeekKey :: !Int
    , rsWeekStartEquity :: !Double
    , rsDayTrades :: !Int
    , rsHaltReason :: !(Maybe ExitReason)
    }
    deriving (Eq, Show)

data BacktestResult = BacktestResult
    { brEquityCurve :: [Double] -- length n
    , brPositions :: [Double] -- length n-1 (signed position size at bar open for t->t+1, -1..1)
    , brAgreementOk :: [Bool] -- length n-1 (True when both models emit a direction and agree)
    , brAgreementValid :: [Bool] -- length n-1 (True when both models emit a direction)
    , brPositionChanges :: !Int
    , brTrades :: [Trade]
    }
    deriving (Eq, Show)

simulateEnsemble ::
    EnsembleConfig ->
    Int -> -- lookback (for LSTM alignment)
    [Double] -> -- prices length n
    [Double] -> -- kalman predicted next prices length n-1 (for t=0..n-2)
    [Double] -> -- lstm predicted next prices length n-1 (for t=0..n-2) or n-lookback (for t=lookback-1..n-2)
    Maybe [StepMeta] -> -- optional per-step confidence meta (length n-1)
    BacktestResult
simulateEnsemble = simulateEnsembleLongFlat

simulateEnsembleWithHL ::
    EnsembleConfig ->
    Int -> -- lookback (for LSTM alignment)
    [Double] -> -- closes length n
    [Double] -> -- highs length n (aligned to closes; bar i high is for close[i-1]..close[i])
    [Double] -> -- lows length n
    [Double] -> -- kalman predicted next prices length n-1 (for t=0..n-2)
    [Double] -> -- lstm predicted next prices length n-1 (for t=0..n-2) or n-lookback (for t=lookback-1..n-2)
    Maybe [StepMeta] -> -- optional per-step confidence meta (length n-1)
    BacktestResult
simulateEnsembleWithHL cfg lookback closes highs lows kalPredNext lstmPredNext mMeta =
    case simulateEnsembleWithHLChecked cfg lookback closes highs lows kalPredNext lstmPredNext mMeta of
        Left err -> error err
        Right bt -> bt

simulateEnsembleWithHLChecked ::
    EnsembleConfig ->
    Int -> -- lookback (for LSTM alignment)
    [Double] -> -- closes length n
    [Double] -> -- highs length n (aligned to closes; bar i high is for close[i-1]..close[i])
    [Double] -> -- lows length n
    [Double] -> -- kalman predicted next prices length n-1 (for t=0..n-2)
    [Double] -> -- lstm predicted next prices length n-1 (for t=0..n-2) or n-lookback (for t=lookback-1..n-2)
    Maybe [StepMeta] -> -- optional per-step confidence meta (length n-1)
    Either String BacktestResult
simulateEnsembleWithHLChecked cfg lookback closes highs lows kalPredNext lstmPredNext mMeta =
    simulateEnsembleLongFlatWithHLChecked cfg lookback closes highs lows kalPredNext lstmPredNext mMeta

simulateEnsembleV ::
    EnsembleConfig ->
    Int -> -- lookback (for LSTM alignment)
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double ->
    Maybe (V.Vector StepMeta) -> -- optional per-step confidence meta
    BacktestResult
simulateEnsembleV = simulateEnsembleLongFlatV

simulateEnsembleVWithHL ::
    EnsembleConfig ->
    Int -> -- lookback (for LSTM alignment)
    V.Vector Double -> -- closes
    V.Vector Double -> -- highs
    V.Vector Double -> -- lows
    V.Vector Double ->
    V.Vector Double ->
    Maybe (V.Vector StepMeta) -> -- optional per-step confidence meta
    BacktestResult
simulateEnsembleVWithHL cfg lookback pricesV highsV lowsV kalPredNextV lstmPredNextV mMetaV =
    case simulateEnsembleVWithHLChecked cfg lookback pricesV highsV lowsV kalPredNextV lstmPredNextV mMetaV of
        Left err -> error err
        Right bt -> bt

simulateEnsembleVWithHLChecked ::
    EnsembleConfig ->
    Int -> -- lookback (for LSTM alignment)
    V.Vector Double -> -- closes
    V.Vector Double -> -- highs
    V.Vector Double -> -- lows
    V.Vector Double ->
    V.Vector Double ->
    Maybe (V.Vector StepMeta) -> -- optional per-step confidence meta
    Either String BacktestResult
simulateEnsembleVWithHLChecked cfg lookback pricesV highsV lowsV kalPredNextV lstmPredNextV mMetaV =
    simulateEnsembleLongFlatVWithHLChecked cfg lookback pricesV highsV lowsV kalPredNextV lstmPredNextV mMetaV

{-# DEPRECATED simulateEnsembleLongFlat "Use simulateEnsemble" #-}
{-# DEPRECATED simulateEnsembleLongFlatWithHL "Use simulateEnsembleWithHL" #-}
{-# DEPRECATED simulateEnsembleLongFlatV "Use simulateEnsembleV" #-}
{-# DEPRECATED simulateEnsembleLongFlatVWithHL "Use simulateEnsembleVWithHL" #-}

simulateEnsembleLongFlat ::
    EnsembleConfig ->
    Int -> -- lookback (for LSTM alignment)
    [Double] -> -- prices length n
    [Double] -> -- kalman predicted next prices length n-1 (for t=0..n-2)
    [Double] -> -- lstm predicted next prices length n-1 (for t=0..n-2) or n-lookback (for t=lookback-1..n-2)
    Maybe [StepMeta] -> -- optional per-step confidence meta (length n-1)
    BacktestResult
simulateEnsembleLongFlat cfg lookback prices kalPredNext lstmPredNext mMeta =
    simulateEnsembleLongFlatV
        cfg
        lookback
        (V.fromList prices)
        (V.fromList kalPredNext)
        (V.fromList lstmPredNext)
        (V.fromList <$> mMeta)

simulateEnsembleLongFlatWithHL ::
    EnsembleConfig ->
    Int -> -- lookback (for LSTM alignment)
    [Double] -> -- closes length n
    [Double] -> -- highs length n (aligned to closes; bar i high is for close[i-1]..close[i])
    [Double] -> -- lows length n
    [Double] -> -- kalman predicted next prices length n-1 (for t=0..n-2)
    [Double] -> -- lstm predicted next prices length n-1 (for t=0..n-2) or n-lookback (for t=lookback-1..n-2)
    Maybe [StepMeta] -> -- optional per-step confidence meta (length n-1)
    BacktestResult
simulateEnsembleLongFlatWithHL cfg lookback closes highs lows kalPredNext lstmPredNext mMeta =
    case simulateEnsembleLongFlatWithHLChecked cfg lookback closes highs lows kalPredNext lstmPredNext mMeta of
        Left err -> error err
        Right bt -> bt

simulateEnsembleLongFlatWithHLChecked ::
    EnsembleConfig ->
    Int -> -- lookback (for LSTM alignment)
    [Double] -> -- closes length n
    [Double] -> -- highs length n (aligned to closes; bar i high is for close[i-1]..close[i])
    [Double] -> -- lows length n
    [Double] -> -- kalman predicted next prices length n-1 (for t=0..n-2)
    [Double] -> -- lstm predicted next prices length n-1 (for t=0..n-2) or n-lookback (for t=lookback-1..n-2)
    Maybe [StepMeta] -> -- optional per-step confidence meta (length n-1)
    Either String BacktestResult
simulateEnsembleLongFlatWithHLChecked cfg lookback closes highs lows kalPredNext lstmPredNext mMeta =
    simulateEnsembleLongFlatVWithHLChecked
        cfg
        lookback
        (V.fromList closes)
        (V.fromList highs)
        (V.fromList lows)
        (V.fromList kalPredNext)
        (V.fromList lstmPredNext)
        (V.fromList <$> mMeta)

simulateEnsembleLongFlatV ::
    EnsembleConfig ->
    Int -> -- lookback (for LSTM alignment)
    V.Vector Double ->
    V.Vector Double ->
    V.Vector Double ->
    Maybe (V.Vector StepMeta) -> -- optional per-step confidence meta
    BacktestResult
simulateEnsembleLongFlatV cfg lookback pricesV kalPredNextV lstmPredNextV mMetaV =
    simulateEnsembleLongFlatVWithHL cfg lookback pricesV pricesV pricesV kalPredNextV lstmPredNextV mMetaV

simulateEnsembleLongFlatVWithHL ::
    EnsembleConfig ->
    Int -> -- lookback (for LSTM alignment)
    V.Vector Double -> -- closes
    V.Vector Double -> -- highs
    V.Vector Double -> -- lows
    V.Vector Double ->
    V.Vector Double ->
    Maybe (V.Vector StepMeta) -> -- optional per-step confidence meta
    BacktestResult
simulateEnsembleLongFlatVWithHL cfg lookback pricesV highsV lowsV kalPredNextV lstmPredNextV mMetaV =
    case simulateEnsembleLongFlatVWithHLChecked cfg lookback pricesV highsV lowsV kalPredNextV lstmPredNextV mMetaV of
        Left err -> error err
        Right bt -> bt

simulateEnsembleLongFlatVWithHLChecked ::
    EnsembleConfig ->
    Int -> -- lookback (for LSTM alignment)
    V.Vector Double -> -- closes
    V.Vector Double -> -- highs
    V.Vector Double -> -- lows
    V.Vector Double ->
    V.Vector Double ->
    Maybe (V.Vector StepMeta) -> -- optional per-step confidence meta
    Either String BacktestResult
simulateEnsembleLongFlatVWithHLChecked cfg lookback pricesV highsV lowsV kalPredNextV lstmPredNextV mMetaV =
    let n = V.length pricesV
        startT = max 0 (lookback - 1)
        stepCount = n - 1
        kalLen = V.length kalPredNextV
        kalNeed = max 0 (stepCount - startT)
        openTimesV =
            case ecOpenTimes cfg of
                Just ts | V.length ts == n -> Just ts
                _ -> Nothing
        openTimesMismatch =
            case ecOpenTimes cfg of
                Just ts
                    | V.length ts /= n ->
                        Just
                            ( "open time vector length ("
                                ++ show (V.length ts)
                                ++ ") must match closes length ("
                                ++ show n
                                ++ ")"
                            )
                _ -> Nothing
        hasCalendar =
            case openTimesV of
                Just _ -> True
                Nothing -> False
        hasDailyKey = hasCalendar
        hasWeeklyKey = hasCalendar
        dailyLossReq =
            case ecMaxDailyLoss cfg of
                Just v | v > 0 && v < 1 && not (isNaN v || isInfinite v) -> Just v
                _ -> Nothing
        weeklyLossReq =
            case ecMaxWeeklyLoss cfg of
                Just v | v > 0 && v < 1 && not (isNaN v || isInfinite v) -> Just v
                _ -> Nothing
        maxTradesPerDayReq =
            case ecMaxTradesPerDay cfg of
                Just n | n > 0 -> Just n
                _ -> Nothing
        expectancyLookback = max 0 (ecExpectancyLookback cfg)
        minExpectancy =
            case ecMinExpectancy cfg of
                Just v | not (isNaN v || isInfinite v) -> Just v
                _ -> Nothing
        noTradeWindows = ecNoTradeWindows cfg
        noTradeReq = not (null noTradeWindows)
        metaMaskV =
            case ecMetaMask cfg of
                Just mask
                    | V.length mask == stepCount -> Just mask
                    | V.length mask > stepCount -> Just (V.drop (V.length mask - stepCount) mask)
                    | otherwise -> Nothing
                _ -> Nothing
        metaMaskMismatch =
            case ecMetaMask cfg of
                Just mask | V.length mask < stepCount -> Just "meta mask vector too short for simulateEnsembleLongFlatVWithHL"
                _ -> Nothing
        metaV =
            case mMetaV of
                Just mv
                    | V.length mv == stepCount -> Just mv
                    | V.length mv > stepCount -> Just (V.drop (V.length mv - stepCount) mv)
                    | otherwise -> Nothing
                _ -> Nothing
        metaMismatch =
            case mMetaV of
                Just mv | V.length mv < stepCount -> Just "meta vector too short for simulateEnsembleLongFlatVWithHL"
                _ -> Nothing
        kalPredAtE
            | kalLen >= stepCount =
                let dropCount = kalLen - stepCount
                    v = if dropCount == 0 then kalPredNextV else V.drop dropCount kalPredNextV
                 in Right (\t -> v V.! t)
            | kalLen >= kalNeed =
                let dropCount = kalLen - kalNeed
                    v = if dropCount == 0 then kalPredNextV else V.drop dropCount kalPredNextV
                 in Right (\t -> v V.! (t - startT))
            | otherwise =
                Left
                    ( "kalPredNext too short: need at least "
                        ++ show kalNeed
                        ++ " (or "
                        ++ show stepCount
                        ++ " for full alignment), got "
                        ++ show kalLen
                    )
        lstmLen = V.length lstmPredNextV
        lstmNeed = max 0 (stepCount - startT)
        lstmPredAtE
            | lstmLen >= stepCount =
                let dropCount = lstmLen - stepCount
                    v = if dropCount == 0 then lstmPredNextV else V.drop dropCount lstmPredNextV
                 in Right (\t -> v V.! t)
            | lstmLen >= lstmNeed =
                let dropCount = lstmLen - lstmNeed
                    v = if dropCount == 0 then lstmPredNextV else V.drop dropCount lstmPredNextV
                 in Right (\t -> v V.! (t - startT))
            | otherwise =
                Left
                    ( "lstmPredNext too short: need at least "
                        ++ show lstmNeed
                        ++ " (or "
                        ++ show stepCount
                        ++ " for full alignment), got "
                        ++ show lstmLen
                    )
        validationError =
            if n < 2
                then Just "Need at least 2 prices to simulate"
                else
                    if V.length highsV /= n || V.length lowsV /= n
                        then Just "high/low vectors must match closes length"
                        else
                            if lookback >= n
                                then Just "lookback must be less than number of prices"
                                else
                                    if (dailyLossReq /= Nothing || weeklyLossReq /= Nothing || maxTradesPerDayReq /= Nothing || noTradeReq) && not hasDailyKey
                                        then Just "--max-daily-loss/--max-weekly-loss/--max-trades-per-day/--no-trade-window require bar timestamps"
                                        else
                                            if minExpectancy /= Nothing && expectancyLookback <= 0
                                                then Just "--min-expectancy requires --expectancy-lookback >= 1"
                                                else case openTimesMismatch of
                                                    Just err
                                                        | dailyLossReq /= Nothing || weeklyLossReq /= Nothing || maxTradesPerDayReq /= Nothing || noTradeReq -> Just err
                                                    _ ->
                                                        case metaMaskMismatch of
                                                            Just err -> Just err
                                                            Nothing ->
                                                                case metaMismatch of
                                                                    Just err -> Just err
                                                                    Nothing ->
                                                                        case kalPredAtE of
                                                                            Left err -> Just err
                                                                            Right _ ->
                                                                                case lstmPredAtE of
                                                                                    Left err -> Just err
                                                                                    Right _ -> Nothing
     in case validationError of
            Just err -> Left err
            Nothing ->
                Right $
                    let startT = max 0 (lookback - 1)
                        openThrRaw = max 0 (ecOpenThreshold cfg)
                        minEdge = max 0 (ecMinEdge cfg)
                        minSignalToNoise = max 0 (ecMinSignalToNoise cfg)
                        snrSizeWeight = clamp01 (ecSnrSizeWeight cfg)
                        factorEnabled = ecThresholdFactorEnabled cfg
                        factorAlpha = clamp01 (ecThresholdFactorAlpha cfg)
                        factorMinRaw = ecThresholdFactorMin cfg
                        factorMaxRaw = ecThresholdFactorMax cfg
                        factorMin = max 1e-6 (min factorMinRaw factorMaxRaw)
                        factorMax = max factorMin (max factorMinRaw factorMaxRaw)
                        factorFloor = max 0 (ecThresholdFactorFloor cfg)
                        factorWEdgeKal = ecThresholdFactorEdgeKalWeight cfg
                        factorWEdgeLstm = ecThresholdFactorEdgeLstmWeight cfg
                        factorWKalmanZ = ecThresholdFactorKalmanZWeight cfg
                        factorWHighVol = ecThresholdFactorHighVolWeight cfg
                        factorWConformal = ecThresholdFactorConformalWeight cfg
                        factorWQuantile = ecThresholdFactorQuantileWeight cfg
                        factorWLstmConf = ecThresholdFactorLstmConfWeight cfg
                        factorWLstmHealth = ecThresholdFactorLstmHealthWeight cfg
                        lstmHealthScore =
                            case ecLstmTrainingHealth cfg of
                                Just v | not (isBad v) -> clamp01 v
                                _ -> 0.5
                        openThr = max openThrRaw minEdge
                        priceActionBodyMin = max 0 (ecTriLayerPriceActionBody cfg)
                        bodyMinFracBase = max 1e-6 (0.25 * openThr)
                        bodyMinFrac =
                            if priceActionBodyMin > 0
                                then priceActionBodyMin
                                else bodyMinFracBase
                        closeThr = max 0 (ecCloseThreshold cfg)
                        minHoldBars = max 0 (ecMinHoldBars cfg)
                        cooldownBars = max 0 (ecCooldownBars cfg)
                        maxHoldBars =
                            case ecMaxHoldBars cfg of
                                Just v | v > 0 -> Just v
                                _ -> Nothing
                        maxDrawdownLim =
                            case ecMaxDrawdown cfg of
                                Just v | v > 0 && v < 1 && not (isNaN v || isInfinite v) -> Just v
                                _ -> Nothing
                        maxDailyLossLim =
                            case dailyLossReq of
                                Just v | hasDailyKey -> Just v
                                _ -> Nothing
                        maxWeeklyLossLim =
                            case weeklyLossReq of
                                Just v | hasWeeklyKey -> Just v
                                _ -> Nothing
                        maxTradesPerDayLim =
                            case maxTradesPerDayReq of
                                Just n | hasDailyKey && n > 0 -> Just n
                                _ -> Nothing
                        riskPerTrade =
                            case ecRiskPerTrade cfg of
                                Just v | v > 0 && v < 1 && not (isNaN v || isInfinite v) -> Just v
                                _ -> Nothing
                        noTradeWindows' = noTradeWindows
                        maxPositionSize = max 0 (ecMaxPositionSize cfg)
                        minPositionSize = max 0 (ecMinPositionSize cfg)
                        rebalanceBars = max 0 (ecRebalanceBars cfg)
                        rebalanceThreshold = max 0 (ecRebalanceThreshold cfg)
                        rebalanceGlobal = ecRebalanceGlobal cfg
                        rebalanceResetOnSignal = ecRebalanceResetOnSignal cfg
                        rebalanceEnabled = rebalanceBars > 0 && rebalanceThreshold > 0
                        trendLookback = max 0 (ecTrendLookback cfg)
                        ppy = max 1e-12 (ecPeriodsPerYear cfg)
                        fundingRate =
                            let r = ecFundingRate cfg
                             in if isNaN r || isInfinite r then 0 else r
                        fundingBySide = ecFundingBySide cfg
                        fundingOnOpen = ecFundingOnOpen cfg
                        triLayerEnabled = ecTriLayer cfg
                        kalDt = max 1e-12 (ecKalmanDt cfg)
                        kalProcessVar = max 0 (ecKalmanProcessVar cfg)
                        kalMeasVar = max 1e-12 (ecKalmanMeasurementVar cfg)
                        fastMult = max 1e-6 (ecTriLayerFastMult cfg)
                        slowMult = max 1e-6 (ecTriLayerSlowMult cfg)
                        cloudPadFrac = max 0 (ecTriLayerCloudPadding cfg)
                        cloudSlopeMin = max 0 (ecTriLayerCloudSlope cfg)
                        cloudWidthMax = max 0 (ecTriLayerCloudWidth cfg)
                        touchLookback = max 1 (ecTriLayerTouchLookback cfg)
                        requirePriceAction = ecTriLayerRequirePriceAction cfg
                        triLayerExitOnSlow = ecTriLayerExitOnSlow cfg
                        kalmanBandLookback = max 0 (ecKalmanBandLookback cfg)
                        kalmanBandStdMult = max 0 (ecKalmanBandStdMult cfg)
                        kalmanBandEnabled =
                            n >= 2 && kalmanBandStdMult > 0 && kalmanBandLookback >= 2
                        kalmanCloudEnabled = (triLayerEnabled || kalmanBandEnabled) && n >= 2
                        cloudReady = triLayerEnabled && n >= 2
                        lstmFlipBars = max 0 (ecLstmExitFlipBars cfg)
                        lstmFlipGraceBars = max 0 (ecLstmExitFlipGraceBars cfg)
                        lstmFlipStrongOnly = ecLstmExitFlipStrong cfg
                        (cloudFastV, cloudSlowV) =
                            if kalmanCloudEnabled
                                then
                                    let run mult =
                                            let mv = max 1e-12 (kalMeasVar * mult)
                                                KalmanRunV{krFilteredV = filts} =
                                                    runConstantAcceleration1DVec kalDt kalProcessVar mv pricesV
                                             in filts
                                     in (run fastMult, run slowMult)
                                else (pricesV, pricesV)
                        kalmanResidualV =
                            if kalmanBandEnabled
                                then V.zipWith (-) pricesV cloudSlowV
                                else V.empty
                        kalmanResidualPrefix =
                            if kalmanBandEnabled
                                then V.scanl' (+) 0 kalmanResidualV
                                else V.empty
                        kalmanResidualSqPrefix =
                            if kalmanBandEnabled
                                then V.scanl' (+) 0 (V.map (\x -> x * x) kalmanResidualV)
                                else V.empty
                        kalmanResidualStdAt :: Int -> Maybe Double
                        kalmanResidualStdAt t =
                            if not kalmanBandEnabled
                                then Nothing
                                else
                                    let end = min (n - 1) t
                                        start = max 0 (end - kalmanBandLookback + 1)
                                        len = end - start + 1
                                     in if len < 2
                                            then Nothing
                                            else
                                                let sumR = (kalmanResidualPrefix V.! (end + 1)) - (kalmanResidualPrefix V.! start)
                                                    sumSq = (kalmanResidualSqPrefix V.! (end + 1)) - (kalmanResidualSqPrefix V.! start)
                                                    lenF = fromIntegral len
                                                    mean = sumR / lenF
                                                    var = max 0 ((sumSq - lenF * mean * mean) / fromIntegral (len - 1))
                                                    std = sqrt var
                                                 in if isBad std then Nothing else Just std
                        kalmanBandAt :: Int -> Maybe (Double, Double)
                        kalmanBandAt t =
                            case kalmanResidualStdAt t of
                                Nothing -> Nothing
                                Just std ->
                                    let slow = cloudSlowV V.! t
                                        upper = slow + kalmanBandStdMult * std
                                        lower = slow - kalmanBandStdMult * std
                                     in if isBad slow || isBad upper || isBad lower
                                            then Nothing
                                            else Just (upper, lower)
                        fundingPerBar =
                            if fundingRate == 0
                                then 0
                                else fundingRate / ppy
                        volTarget =
                            case ecVolTarget cfg of
                                Just v | v > 0 && not (isNaN v || isInfinite v) -> Just v
                                _ -> Nothing
                        volLookback = max 0 (ecVolLookback cfg)
                        volFloor = max 0 (ecVolFloor cfg)
                        volScaleMax = max 0 (ecVolScaleMax cfg)
                        volAlpha =
                            case ecVolEwmaAlpha cfg of
                                Just a | a > 0 && not (isNaN a || isInfinite a) -> Just (max 0 (min 1 a))
                                _ -> Nothing
                        maxVolatility =
                            case ecMaxVolatility cfg of
                                Just v | v > 0 && not (isNaN v || isInfinite v) -> Just v
                                _ -> Nothing
                        dayKeyAt :: Int -> Int
                        dayKeyAt i =
                            case openTimesV of
                                Just tsV
                                    | i >= 0 && i < V.length tsV ->
                                        let dayMs = 86400000 :: Int64
                                         in fromIntegral ((tsV V.! i) `div` dayMs)
                                _ -> 0
                        weekKeyAt :: Int -> Int
                        weekKeyAt i =
                            case openTimesV of
                                Just tsV
                                    | i >= 0 && i < V.length tsV ->
                                        let weekMs = 7 * 86400000 :: Int64
                                         in fromIntegral ((tsV V.! i) `div` weekMs)
                                _ -> 0
                        minuteOfDayAt :: Int -> Int
                        minuteOfDayAt i =
                            case openTimesV of
                                Just tsV
                                    | i >= 0 && i < V.length tsV -> minuteOfDayFromMs (tsV V.! i)
                                _ -> 0
                        direction thr prev pred =
                            let upEdge = prev * (1 + thr)
                                downEdge = prev * (1 - thr)
                             in if pred > upEdge
                                    then Just SideLong
                                    else if pred < downEdge then Just SideShort else Nothing
                        desiredSideFromDir d =
                            case d of
                                SideLong -> Just SideLong
                                SideShort ->
                                    case ecPositioning cfg of
                                        LongFlat -> Nothing
                                        LongShort -> Just SideShort
                        stepCount = n - 1

                        perSideCost =
                            let fee = max 0 (ecFee cfg)
                                slip = max 0 (ecSlippage cfg)
                                spr = max 0 (ecSpread cfg)
                                c = fee + slip + spr / 2
                             in min 0.999999 (max 0 c)

                        applyCost :: Double -> Double -> Double
                        applyCost eq size =
                            let s = max 0 (abs size)
                                cost = min 0.999999 (perSideCost * s)
                             in eq * (1 - cost)

                        clampSignedFrac :: Double -> Double
                        clampSignedFrac x =
                            let cap = 0.999999
                             in max (-cap) (min cap x)

                        applyFunding :: Double -> Double -> PositionSide -> Double
                        applyFunding eq size side =
                            if fundingPerBar == 0
                                then eq
                                else
                                    let s = max 0 (abs size)
                                        signedRate =
                                            if fundingBySide
                                                then fundingPerBar * s * sideSign side
                                                else fundingPerBar * s
                                        rate = clampSignedFrac signedRate
                                     in eq * (1 - rate)

                        barHigh t1 =
                            let h = highsV V.! t1
                                c = pricesV V.! t1
                             in if isNaN h || isInfinite h then c else h

                        barLow t1 =
                            let l = lowsV V.! t1
                                c = pricesV V.! t1
                             in if isNaN l || isInfinite l then c else l

                        barOpen t1 =
                            if t1 <= 0
                                then pricesV V.! t1
                                else pricesV V.! (t1 - 1)

                        candleAt t1 =
                            let o = barOpen t1
                                c = pricesV V.! t1
                                h = barHigh t1
                                l = barLow t1
                             in (o, h, l, c)

                        candleOpen (o, _, _, _) = o
                        candleClose (_, _, _, c) = c
                        candleBody (o, _, _, c) = abs (c - o)
                        candleBull (o, _, _, c) = c > o
                        candleBear (o, _, _, c) = c < o
                        candleUpperWick (o, h, _, c) = max 0 (h - max o c)
                        candleLowerWick (o, _, l, c) = max 0 (min o c - l)

                        bodyOk closePx body =
                            let denom = max 1e-12 (abs closePx)
                             in body / denom >= bodyMinFrac

                        hammer candle =
                            let body = candleBody candle
                                upper = candleUpperWick candle
                                lower = candleLowerWick candle
                                closePx = candleClose candle
                             in candleBull candle
                                    && body > 0
                                    && bodyOk closePx body
                                    && lower >= 2 * body
                                    && upper <= 0.5 * body

                        shootingStar candle =
                            let body = candleBody candle
                                upper = candleUpperWick candle
                                lower = candleLowerWick candle
                                closePx = candleClose candle
                             in candleBear candle
                                    && body > 0
                                    && bodyOk closePx body
                                    && upper >= 2 * body
                                    && lower <= 0.5 * body

                        bullishEngulf cur prev =
                            let bodyCur = candleBody cur
                                bodyPrev = candleBody prev
                                closePx = candleClose cur
                             in candleBull cur
                                    && candleBear prev
                                    && bodyCur >= bodyPrev
                                    && bodyOk closePx bodyCur
                                    && candleClose cur >= candleOpen prev
                                    && candleOpen cur <= candleClose prev

                        bearishEngulf cur prev =
                            let bodyCur = candleBody cur
                                bodyPrev = candleBody prev
                                closePx = candleClose cur
                             in candleBear cur
                                    && candleBull prev
                                    && bodyCur >= bodyPrev
                                    && bodyOk closePx bodyCur
                                    && candleClose cur <= candleOpen prev
                                    && candleOpen cur >= candleClose prev

                        railroadTracksLong cur prev =
                            let bodyCur = candleBody cur
                                bodyPrev = candleBody prev
                                closePx = candleClose cur
                                tol = 0.2
                             in candleBull cur
                                    && candleBear prev
                                    && bodyPrev > 0
                                    && bodyOk closePx bodyCur
                                    && abs (bodyCur - bodyPrev) / bodyPrev <= tol

                        darkCloudCover cur prev =
                            let midPrev = (candleOpen prev + candleClose prev) / 2
                             in candleBear cur && candleBull prev && candleClose cur < midPrev

                        markToMarket :: PositionSide -> Double -> Double -> Double
                        markToMarket side basePx px =
                            if basePx == 0
                                then 1
                                else case side of
                                    SideLong -> px / basePx
                                    SideShort -> 2 - px / basePx

                        isBad :: Double -> Bool
                        isBad x = isNaN x || isInfinite x

                        meanList :: [Double] -> Double
                        meanList xs =
                            let ys = filter (not . isBad) xs
                             in if null ys then 0 else sum ys / fromIntegral (length ys)

                        returnsV :: V.Vector Double
                        returnsV =
                            V.generate stepCount $ \i ->
                                let p0 = pricesV V.! i
                                    p1 = pricesV V.! (i + 1)
                                    r =
                                        if p0 == 0 || isBad p0 || isBad p1
                                            then 0
                                            else p1 / p0 - 1
                                 in if isBad r then 0 else r

                        meanV :: V.Vector Double -> Double
                        meanV v =
                            let n0 = V.length v
                             in if n0 <= 0 then 0 else V.sum v / fromIntegral n0

                        stddevV :: V.Vector Double -> Double
                        stddevV v =
                            let n0 = V.length v
                             in if n0 < 2
                                    then 0
                                    else
                                        let m = meanV v
                                            var = V.sum (V.map (\x -> (x - m) ** 2) v) / fromIntegral (n0 - 1)
                                         in sqrt var

                        ewmaVarV :: V.Vector Double
                        ewmaVarV =
                            case volAlpha of
                                Nothing -> V.empty
                                Just a ->
                                    let update var r = a * var + (1 - a) * (r * r)
                                     in V.tail (V.scanl' update 0 returnsV)

                        ewmaVarAt :: Int -> Maybe Double
                        ewmaVarAt t =
                            let idx = t - 1
                             in if idx < 0 || idx >= V.length ewmaVarV
                                    then Nothing
                                    else Just (ewmaVarV V.! idx)

                        rollingVolAt :: Int -> Maybe Double
                        rollingVolAt t =
                            let end = t - 1
                             in if volLookback <= 1 || end < 1
                                    then Nothing
                                    else
                                        let start = max 0 (end - volLookback + 1)
                                            len = end - start + 1
                                         in if len < 2
                                                then Nothing
                                                else
                                                    let v = V.slice start len returnsV
                                                        std = stddevV v
                                                     in if isBad std then Nothing else Just (std * sqrt ppy)

                        ewmaVolAt :: Int -> Maybe Double
                        ewmaVolAt t =
                            case ewmaVarAt t of
                                Nothing -> Nothing
                                Just var ->
                                    let std = sqrt (max 0 var)
                                     in if isBad std then Nothing else Just (std * sqrt ppy)

                        volEstimateAt :: Int -> Maybe Double
                        volEstimateAt t =
                            case volAlpha of
                                Just _ ->
                                    if t < 2
                                        then Nothing
                                        else ewmaVolAt t
                                Nothing -> rollingVolAt t

                        volPerBarAt :: Int -> Maybe Double
                        volPerBarAt t =
                            case volEstimateAt t of
                                Nothing -> Nothing
                                Just vol ->
                                    let perBar = vol / sqrt ppy
                                     in if isBad perBar then Nothing else Just (max 0 perBar)

                        volScaleAt :: Int -> Double
                        volScaleAt t =
                            case volTarget of
                                Nothing -> 1
                                Just target ->
                                    case volEstimateAt t of
                                        Nothing -> 1
                                        Just vol ->
                                            let volAdj = max volFloor vol
                                                scale0 = if volAdj <= 0 then 1 else target / volAdj
                                                scale1 = min volScaleMax (max 0 scale0)
                                             in if isBad scale1 then 1 else scale1

                        volOkAt :: Int -> Bool
                        volOkAt t =
                            case maxVolatility of
                                Nothing -> True
                                Just maxVol ->
                                    case volEstimateAt t of
                                        Just vol -> vol <= maxVol
                                        Nothing -> False

                        volTargetReadyAt :: Int -> Bool
                        volTargetReadyAt t =
                            case volTarget of
                                Nothing -> True
                                Just _ ->
                                    case volEstimateAt t of
                                        Just _ -> True
                                        Nothing -> False

                        signalToNoiseOkAt :: Int -> Double -> Double -> Bool
                        signalToNoiseOkAt t minSn edge =
                            if minSn <= 0
                                then True
                                else case volPerBarAt t of
                                    Just vol | vol > 0 -> edge / vol >= minSn
                                    _ -> False

                        trendOkAt :: Int -> PositionSide -> Bool
                        trendOkAt t side =
                            if trendLookback <= 1 || t < trendLookback - 1
                                then True
                                else
                                    let start = t - trendLookback + 1
                                        v = V.slice start trendLookback pricesV
                                        sma = meanV v
                                        px = pricesV V.! t
                                     in if isBad sma || isBad px
                                            then True
                                            else case side of
                                                SideLong -> px >= sma
                                                SideShort -> px <= sma

                        touchCloudAt :: Int -> Bool
                        touchCloudAt idx =
                            let fastIdx = cloudFastV V.! idx
                                slowIdx = cloudSlowV V.! idx
                                cloudTopIdx = max fastIdx slowIdx
                                cloudBotIdx = min fastIdx slowIdx
                                pxIdx = pricesV V.! idx
                                padIdx = cloudPadFrac * (if isBad pxIdx then 0 else abs pxIdx)
                                cloudTopPadIdx = cloudTopIdx + padIdx
                                cloudBotPadIdx = cloudBotIdx - padIdx
                                (_, h, l, _) = candleAt idx
                             in not (isBad fastIdx || isBad slowIdx)
                                    && l <= cloudTopPadIdx
                                    && h >= cloudBotPadIdx

                        touchCloudPrefix :: V.Vector Int
                        touchCloudPrefix =
                            if cloudReady
                                then
                                    V.scanl'
                                        (\acc touched -> acc + if touched then 1 else 0)
                                        0
                                        (V.generate n touchCloudAt)
                                else V.replicate (n + 1) 0

                        touchCloudInWindow :: Int -> Bool
                        touchCloudInWindow t =
                            if not cloudReady
                                then True
                                else
                                    let start = max 0 (t - touchLookback + 1)
                                        end = min n (t + 1)
                                     in (touchCloudPrefix V.! end) > (touchCloudPrefix V.! start)

                        cloudOkAt :: Int -> PositionSide -> Bool
                        cloudOkAt t side =
                            if not cloudReady || t <= 0
                                then True
                                else
                                    let fast = cloudFastV V.! t
                                        slow = cloudSlowV V.! t
                                        slope = slow - cloudSlowV V.! (t - 1)
                                        cloudTop = max fast slow
                                        cloudBot = min fast slow
                                        px = pricesV V.! t
                                        slopeFrac =
                                            if isBad px || px == 0
                                                then 0
                                                else slope / abs px
                                        widthFrac =
                                            if isBad px || px == 0
                                                then 0
                                                else (cloudTop - cloudBot) / abs px
                                        widthOk = cloudWidthMax <= 0 || isBad widthFrac || widthFrac <= cloudWidthMax
                                        touchCloud = touchCloudInWindow t
                                        trendOk =
                                            case side of
                                                SideLong -> fast > slow && slopeFrac >= cloudSlopeMin
                                                SideShort -> fast < slow && slopeFrac <= -cloudSlopeMin
                                     in if isBad fast || isBad slow || isBad slope
                                            then True
                                            else touchCloud && trendOk && widthOk

                        priceActionOkAt :: Int -> PositionSide -> Bool
                        priceActionOkAt t side =
                            if not triLayerEnabled || not requirePriceAction || t < 2
                                then True
                                else
                                    let cur = candleAt t
                                        prev = candleAt (t - 1)
                                        bullish = hammer cur || bullishEngulf cur prev || railroadTracksLong cur prev
                                        bearish = shootingStar cur || bearishEngulf cur prev || darkCloudCover cur prev
                                     in case side of
                                            SideLong -> bullish
                                            SideShort -> bearish

                        clamp01 :: Double -> Double
                        clamp01 x = max 0 (min 1 x)

                        clampRange :: Double -> Double -> Double -> Double
                        clampRange lo hi x =
                            let lo' = min lo hi
                                hi' = max lo hi
                             in max lo' (min hi' x)

                        signedScore :: Double -> Double
                        signedScore s = 2 * clamp01 s - 1

                        scoreOrNeutral :: Maybe Double -> Double
                        scoreOrNeutral mv =
                            case mv of
                                Just v | not (isBad v) -> clamp01 v
                                _ -> 0.5

                        edgeScore :: Double -> Double -> Double
                        edgeScore thr edge =
                            let denom = max 1e-12 (abs thr)
                                raw = edge / (2 * denom)
                             in if isBad raw then 0.5 else clamp01 raw

                        lstmConfidenceScore :: Double -> Double -> Maybe Double
                        lstmConfidenceScore prev next =
                            if prev <= 0 || isBad prev || isBad next
                                then Nothing
                                else
                                    let edge = abs (next / prev - 1)
                                        thr = max 1e-12 openThr
                                        raw = edge / (2 * thr)
                                     in if isBad edge || isBad raw
                                            then Nothing
                                            else Just (clamp01 raw)

                        lstmConfidenceSizing :: Double -> Double -> Double
                        lstmConfidenceSizing prev next =
                            if not (ecConfidenceSizing cfg)
                                then 1
                                else
                                    let hard0 = clamp01 (ecLstmConfidenceHard cfg)
                                        soft0 = clamp01 (ecLstmConfidenceSoft cfg)
                                        hard = hard0
                                        soft = min soft0 hard
                                        denom = max 1e-12 (hard - soft)
                                     in if hard <= 0
                                            then 1
                                            else case lstmConfidenceScore prev next of
                                                Nothing -> 1
                                                Just score ->
                                                    let scaleRaw =
                                                            if score <= soft
                                                                then 0
                                                                else
                                                                    if score >= hard
                                                                        then 1
                                                                        else (score - soft) / denom
                                                     in clamp01 scaleRaw

                        clampFrac :: Double -> Double
                        clampFrac x = min 0.999999 (max 0 x)

                        stopFracFromVol :: Int -> Double -> Maybe Double
                        stopFracFromVol t mult =
                            if mult <= 0
                                then Nothing
                                else case volPerBarAt t of
                                    Just vol
                                        | vol > 0 ->
                                            let f = mult * vol
                                             in if isBad f || f <= 0 then Nothing else Just (clampFrac f)
                                    _ -> Nothing

                        stopLossFracAt :: Int -> Maybe Double
                        stopLossFracAt t =
                            case stopFracFromVol t (ecStopLossVolMult cfg) of
                                Just f -> Just f
                                Nothing ->
                                    case ecStopLoss cfg of
                                        Just sl | sl > 0 -> Just (clampFrac sl)
                                        _ -> Nothing

                        takeProfitFracAt :: Int -> Maybe Double
                        takeProfitFracAt t =
                            case stopFracFromVol t (ecTakeProfitVolMult cfg) of
                                Just f -> Just f
                                Nothing ->
                                    case ecTakeProfit cfg of
                                        Just tp | tp > 0 -> Just (clampFrac tp)
                                        _ -> Nothing

                        trailingStopFracAt :: Int -> Maybe Double
                        trailingStopFracAt t =
                            case stopFracFromVol t (ecTrailingStopVolMult cfg) of
                                Just f -> Just f
                                Nothing ->
                                    case ecTrailingStop cfg of
                                        Just ts | ts > 0 -> Just (clampFrac ts)
                                        _ -> Nothing

                        riskScaleAt :: Int -> Double
                        riskScaleAt t =
                            case riskPerTrade of
                                Nothing -> 1
                                Just risk ->
                                    case stopLossFracAt t of
                                        Just sl
                                            | sl > 0 ->
                                                let s = risk / sl
                                                 in if isBad s || s <= 0 then 1 else s
                                        _ -> 1

                        scale01 :: Double -> Double -> Double -> Double
                        scale01 lo hi x =
                            let lo' = min lo hi
                                hi' = max lo hi
                             in if hi' <= lo' + 1e-12
                                    then if x >= hi' then 1 else 0
                                    else clamp01 ((x - lo') / (hi' - lo'))

                        metaAt :: Int -> Maybe StepMeta
                        metaAt t =
                            case metaV of
                                Nothing -> Nothing
                                Just mv ->
                                    case metaMaskV of
                                        Just mask
                                            | t >= 0 && t < V.length mask && not (mask V.! t) -> Nothing
                                        _ ->
                                            if t >= 0 && t < V.length mv
                                                then Just (mv V.! t)
                                                else Nothing

                        kalPredAt :: Int -> Double
                        kalPredAt =
                            case kalPredAtE of
                                Right f -> f
                                Left _ -> \_ -> 0

                        lstmPredAt :: Int -> Double
                        lstmPredAt =
                            case lstmPredAtE of
                                Right f -> f
                                Left _ -> \_ -> 0

                        intervalWidth :: StepMeta -> Maybe Double
                        intervalWidth m =
                            case (smConformalLo m, smConformalHi m) of
                                (Just lo', Just hi') ->
                                    let w = hi' - lo'
                                     in if isNaN w || isInfinite w then Nothing else Just w
                                _ -> Nothing

                        quantileWidth :: StepMeta -> Maybe Double
                        quantileWidth m =
                            case (smQuantile10 m, smQuantile90 m) of
                                (Just q10', Just q90') ->
                                    let w = q90' - q10'
                                     in if isNaN w || isInfinite w then Nothing else Just w
                                _ -> Nothing

                        kalmanZ :: StepMeta -> Double
                        kalmanZ m =
                            let var = max 0 (smKalmanVar m)
                                std = sqrt var
                                mu = smKalmanMean m
                             in if std <= 0 || isNaN std || isInfinite std then 0 else abs mu / std

                        confirmConformal :: Double -> StepMeta -> PositionSide -> Bool
                        confirmConformal thr m side =
                            if not (ecConfirmConformal cfg)
                                then True
                                else case side of
                                    SideLong -> maybe True (> thr) (smConformalLo m)
                                    SideShort -> maybe True (< negate thr) (smConformalHi m)

                        confirmQuantiles :: Double -> StepMeta -> PositionSide -> Bool
                        confirmQuantiles thr m side =
                            if not (ecConfirmQuantiles cfg)
                                then True
                                else case side of
                                    SideLong -> maybe True (> thr) (smQuantile10 m)
                                    SideShort -> maybe True (< negate thr) (smQuantile90 m)

                        confidenceScoreKalman :: StepMeta -> Double
                        confidenceScoreKalman m =
                            let zMin = max 0 (ecKalmanZMin cfg)
                                zMax = max zMin (ecKalmanZMax cfg)
                                zScore = scale01 zMin zMax (kalmanZ m)
                                hvScore =
                                    case (ecMaxHighVolProb cfg, smHighVolProb m) of
                                        (Just maxHv, Just hv) -> clamp01 ((maxHv - hv) / max 1e-12 maxHv)
                                        _ -> 1
                                confScore =
                                    case (ecMaxConformalWidth cfg, intervalWidth m) of
                                        (Just maxW, Just w) -> clamp01 ((maxW - w) / max 1e-12 maxW)
                                        _ -> 1
                                qScore =
                                    case (ecMaxQuantileWidth cfg, quantileWidth m) of
                                        (Just maxW, Just w) -> clamp01 ((maxW - w) / max 1e-12 maxW)
                                        _ -> 1
                             in zScore * hvScore * confScore * qScore

                        gateKalmanDir :: Bool -> StepMeta -> Double -> Double -> Maybe PositionSide -> (Maybe PositionSide, Double)
                        gateKalmanDir useSizing m thr confScore dirRaw =
                            case dirRaw of
                                Nothing -> (Nothing, 0)
                                Just side ->
                                    let kalZ = kalmanZ m
                                        zMin = max 0 (ecKalmanZMin cfg)
                                        hvOk =
                                            case (ecMaxHighVolProb cfg, smHighVolProb m) of
                                                (Just maxHv, Just hv) -> hv <= maxHv
                                                (Just _, Nothing) -> True
                                                _ -> True
                                        confWidthOk =
                                            case (ecMaxConformalWidth cfg, intervalWidth m) of
                                                (Just maxW, Just w) -> w <= maxW
                                                (Just _, Nothing) -> True
                                                _ -> True
                                        qWidthOk =
                                            case (ecMaxQuantileWidth cfg, quantileWidth m) of
                                                (Just maxW, Just w) -> w <= maxW
                                                (Just _, Nothing) -> True
                                                _ -> True
                                        size0 = if useSizing then confScore else 1
                                     in if kalZ < zMin
                                            then (Nothing, 0)
                                            else
                                                if not hvOk
                                                    then (Nothing, 0)
                                                    else
                                                        if not confWidthOk
                                                            then (Nothing, 0)
                                                            else
                                                                if not qWidthOk
                                                                    then (Nothing, 0)
                                                                    else
                                                                        if not (confirmConformal thr m side)
                                                                            then (Nothing, 0)
                                                                            else
                                                                                if not (confirmQuantiles thr m side)
                                                                                    then (Nothing, 0)
                                                                                    else
                                                                                        if useSizing && size0 <= 0
                                                                                            then (Nothing, 0)
                                                                                            else (Just side, size0)

                        stepFn (posSide, posSize, equity, eqAcc, posAcc, agreeAcc, agreeValidAcc, changes, openTrade, tradesAcc, dead, cooldownLeft, riskState0, factorOpenPrev, factorClosePrev) t =
                            if dead
                                then
                                    ( Nothing
                                    , 0
                                    , equity
                                    , equity : eqAcc
                                    , 0 : posAcc
                                    , False : agreeAcc
                                    , False : agreeValidAcc
                                    , changes
                                    , Nothing
                                    , tradesAcc
                                    , True
                                    , 0
                                    , riskState0
                                    , factorOpenPrev
                                    , factorClosePrev
                                    )
                                else
                                    let RiskState
                                            { rsPeakEquity = peakEq0
                                            , rsDayKey = dayKey0
                                            , rsDayStartEquity = dayStartEq0
                                            , rsWeekKey = weekKey0
                                            , rsWeekStartEquity = weekStartEq0
                                            , rsDayTrades = dayTrades0
                                            , rsHaltReason = haltReason0
                                            } = riskState0
                                        (dayKey1, dayStartEq1, dayChanged) =
                                            if hasDailyKey
                                                then
                                                    let dk = dayKeyAt t
                                                        changed = dk /= dayKey0
                                                        dayStart = if changed then equity else dayStartEq0
                                                     in (dk, dayStart, changed)
                                                else (dayKey0, dayStartEq0, False)
                                        (weekKey1, weekStartEq1, weekChanged) =
                                            if hasWeeklyKey
                                                then
                                                    let wk = weekKeyAt t
                                                        changed = wk /= weekKey0
                                                        weekStart = if changed then equity else weekStartEq0
                                                     in (wk, weekStart, changed)
                                                else (weekKey0, weekStartEq0, False)
                                        dayTrades1 = if dayChanged then 0 else dayTrades0
                                        peakEq1 = max peakEq0 equity
                                        drawdown =
                                            if peakEq1 > 0
                                                then max 0 (1 - equity / peakEq1)
                                                else 0
                                        dailyLoss =
                                            if dayStartEq1 > 0
                                                then max 0 (1 - equity / dayStartEq1)
                                                else 0
                                        weeklyLoss =
                                            if weekStartEq1 > 0
                                                then max 0 (1 - equity / weekStartEq1)
                                                else 0
                                        expectancy =
                                            if expectancyLookback <= 0
                                                then Nothing
                                                else
                                                    let recent = take expectancyLookback tradesAcc
                                                     in if length recent < expectancyLookback
                                                            then Nothing
                                                            else Just (meanList (map trReturn recent))
                                        haltReasonBase =
                                            case (haltReason0, dayChanged, weekChanged) of
                                                (Just ExitMaxDailyLoss, True, _) -> Nothing
                                                (Just ExitMaxWeeklyLoss, _, True) -> Nothing
                                                _ -> haltReason0
                                        riskHaltReason =
                                            case haltReasonBase of
                                                Just _ -> Nothing
                                                Nothing ->
                                                    case () of
                                                        _
                                                            | maybe False (\lim -> dailyLoss >= lim) maxDailyLossLim -> Just ExitMaxDailyLoss
                                                            | maybe False (\lim -> weeklyLoss >= lim) maxWeeklyLossLim -> Just ExitMaxWeeklyLoss
                                                            | maybe False (\lim -> drawdown >= lim) maxDrawdownLim -> Just ExitMaxDrawdown
                                                            | maybe False (\lim -> maybe False (< lim) expectancy) minExpectancy -> Just (ExitOther "NEGATIVE_EXPECTANCY")
                                                            | otherwise -> Nothing
                                        haltReason1 =
                                            case haltReasonBase of
                                                Just r -> Just r
                                                Nothing -> riskHaltReason
                                        halted = haltReason1 /= Nothing

                                        prev = pricesV V.! t
                                        nextClose = pricesV V.! (t + 1)
                                        hi = barHigh (t + 1)
                                        lo = barLow (t + 1)
                                        factorOpenBase0 =
                                            if factorEnabled && not (isBad factorOpenPrev)
                                                then factorOpenPrev
                                                else 1
                                        factorCloseBase0 =
                                            if factorEnabled && not (isBad factorClosePrev)
                                                then factorClosePrev
                                                else 1
                                        factorOpenBase =
                                            if factorEnabled
                                                then clampRange factorMin factorMax factorOpenBase0
                                                else 1
                                        factorCloseBase =
                                            if factorEnabled
                                                then clampRange factorMin factorMax factorCloseBase0
                                                else 1
                                        minEdgeAdj = max factorFloor (minEdge * factorOpenBase)
                                        openThrAdj = max minEdgeAdj (max factorFloor (openThr * factorOpenBase))
                                        closeThrAdj = max factorFloor (closeThr * factorCloseBase)
                                        minSignalToNoiseAdj = max factorFloor (minSignalToNoise * factorOpenBase)
                                        openTrade0 =
                                            case posSide of
                                                Nothing -> Nothing
                                                Just _ -> openTrade
                                        posSideStart = posSide
                                        posSizeStart = posSize
                                        holdBars =
                                            case openTrade0 of
                                                Nothing -> 0
                                                Just ot -> otHoldingPeriods ot
                                        cooldownActive = posSide == Nothing && cooldownLeft > 0
                                        cooldownNext0 = if posSide == Nothing then max 0 (cooldownLeft - 1) else 0
                                        (agreeOk, agreeValid, desiredSideRaw, desiredSizeRaw, edgeRaw, edgeKal, edgeLstm, mOpenSignal, lstmCloseDir, lstmEntryScaleRaw, lstmConfScoreMaybe, metaNow) =
                                            if t < startT
                                                then (False, False, posSide, posSize, 0, 0, 0, Nothing, Nothing, 1, Nothing, Nothing)
                                                else
                                                    let kp = kalPredAt t
                                                        lp = lstmPredAt t
                                                        edgePred p =
                                                            if prev == 0 || isBad p
                                                                then 0
                                                                else
                                                                    let edge = abs (p / prev - 1)
                                                                     in if isBad edge then 0 else edge
                                                        edgeKal = edgePred kp
                                                        edgeLstm = edgePred lp
                                                        edgeRaw = min edgeKal edgeLstm
                                                        kalOpenDirRaw = direction openThrAdj prev kp
                                                        kalCloseDirRaw = direction closeThrAdj prev kp
                                                        metaNow = metaAt t
                                                        (kalOpenDir, kalCloseDir, kalSize) =
                                                            case metaNow of
                                                                Nothing ->
                                                                    ( kalOpenDirRaw
                                                                    , kalCloseDirRaw
                                                                    , if kalOpenDirRaw == Nothing then 0 else 1
                                                                    )
                                                                Just m ->
                                                                    let confScore = confidenceScoreKalman m
                                                                        (openDir, openSize) = gateKalmanDir (ecConfidenceSizing cfg) m openThrAdj confScore kalOpenDirRaw
                                                                        (closeDir, _) = gateKalmanDir False m closeThrAdj confScore kalCloseDirRaw
                                                                     in (openDir, closeDir, openSize)
                                                        lstmOpenDir = direction openThrAdj prev lp
                                                        lstmCloseDir = direction closeThrAdj prev lp
                                                        agreeValid =
                                                            case (kalOpenDir, lstmOpenDir) of
                                                                (Just _, Just _) -> True
                                                                _ -> False
                                                        agreeOk =
                                                            case (kalOpenDir, lstmOpenDir) of
                                                                (Just a, Just b) -> a == b
                                                                _ -> False
                                                        closeAgreeDir =
                                                            case (kalCloseDir, lstmCloseDir) of
                                                                (Just a, Just b) | a == b -> Just a
                                                                _ -> Nothing
                                                        openAgreeDir =
                                                            if agreeOk then kalOpenDir else Nothing

                                                        openSignal =
                                                            case openAgreeDir of
                                                                Just dir ->
                                                                    case desiredSideFromDir dir of
                                                                        Nothing -> Nothing
                                                                        Just s -> Just (s, kalSize)
                                                                Nothing -> Nothing

                                                        openBlockedByPositioning =
                                                            case openAgreeDir of
                                                                Just _ -> openSignal == Nothing
                                                                Nothing -> False

                                                        (desiredSide', desiredSize') =
                                                            case openSignal of
                                                                Just (s, sz) -> (Just s, sz)
                                                                Nothing ->
                                                                    if openBlockedByPositioning
                                                                        then (Nothing, 0)
                                                                        else case (posSide, closeAgreeDir) of
                                                                            (Just side, Just dir) | side == dir -> (Just side, posSize)
                                                                            _ -> (Nothing, 0)
                                                        lstmEntryScale = lstmConfidenceSizing prev lp
                                                        lstmScore = lstmConfidenceScore prev lp
                                                     in (agreeOk, agreeValid, desiredSide', desiredSize', edgeRaw, edgeKal, edgeLstm, openSignal, lstmCloseDir, lstmEntryScale, lstmScore, metaNow)

                                        pendingFactorUpdate =
                                            factorEnabled && (posSide /= Nothing || mOpenSignal /= Nothing)

                                        factorTarget thr =
                                            let edgeKalScore = edgeScore thr edgeKal
                                                edgeLstmScore = edgeScore thr edgeLstm
                                                zScore =
                                                    scoreOrNeutral $
                                                        case metaNow of
                                                            Nothing -> Nothing
                                                            Just m ->
                                                                let zMin = max 0 (ecKalmanZMin cfg)
                                                                    zMax = max zMin (ecKalmanZMax cfg)
                                                                 in Just (scale01 zMin zMax (kalmanZ m))
                                                hvScore =
                                                    scoreOrNeutral $
                                                        case metaNow >>= smHighVolProb of
                                                            Nothing -> Nothing
                                                            Just hv ->
                                                                let score =
                                                                        case ecMaxHighVolProb cfg of
                                                                            Just maxHv | maxHv > 0 -> clamp01 ((maxHv - hv) / max 1e-12 maxHv)
                                                                            _ -> clamp01 (1 - hv)
                                                                 in Just score
                                                confScore =
                                                    scoreOrNeutral $
                                                        case metaNow >>= intervalWidth of
                                                            Nothing -> Nothing
                                                            Just w ->
                                                                case ecMaxConformalWidth cfg of
                                                                    Just maxW | maxW > 0 -> Just (clamp01 ((maxW - w) / max 1e-12 maxW))
                                                                    _ -> Nothing
                                                qScore =
                                                    scoreOrNeutral $
                                                        case metaNow >>= quantileWidth of
                                                            Nothing -> Nothing
                                                            Just w ->
                                                                case ecMaxQuantileWidth cfg of
                                                                    Just maxW | maxW > 0 -> Just (clamp01 ((maxW - w) / max 1e-12 maxW))
                                                                    _ -> Nothing
                                                lstmScore = scoreOrNeutral lstmConfScoreMaybe
                                                healthScore = lstmHealthScore
                                                raw =
                                                    1
                                                        + factorWEdgeKal * signedScore edgeKalScore
                                                        + factorWEdgeLstm * signedScore edgeLstmScore
                                                        + factorWKalmanZ * signedScore zScore
                                                        + factorWHighVol * signedScore hvScore
                                                        + factorWConformal * signedScore confScore
                                                        + factorWQuantile * signedScore qScore
                                                        + factorWLstmConf * signedScore lstmScore
                                                        + factorWLstmHealth * signedScore healthScore
                                             in if isBad raw then 1 else raw

                                        updateFactor prev target =
                                            if factorAlpha <= 0
                                                then prev
                                                else
                                                    let next = (1 - factorAlpha) * prev + factorAlpha * target
                                                     in if isBad next then prev else clampRange factorMin factorMax next

                                        factorOpenNext =
                                            if pendingFactorUpdate
                                                then updateFactor factorOpenBase (factorTarget openThr)
                                                else factorOpenBase

                                        factorCloseNext =
                                            if pendingFactorUpdate
                                                then updateFactor factorCloseBase (factorTarget closeThr)
                                                else factorCloseBase

                                        lstmFlipStrongOk =
                                            if not lstmFlipStrongOnly
                                                then True
                                                else case lstmConfScoreMaybe of
                                                    Just score -> score >= max 0 (min 1 (ecLstmConfidenceHard cfg))
                                                    Nothing -> False

                                        (openTradeFlip, lstmFlipExit) =
                                            if lstmFlipBars <= 0
                                                then (openTrade0, False)
                                                else case (openTrade0, posSide, lstmCloseDir) of
                                                    (Just ot, Just side, Just lstmDir) ->
                                                        let inGrace = otHoldingPeriods ot < lstmFlipGraceBars
                                                         in if inGrace
                                                                then (Just ot{otLstmFlipCount = 0}, False)
                                                                else
                                                                    let opposite =
                                                                            case (side, lstmDir) of
                                                                                (SideLong, SideShort) -> True
                                                                                (SideShort, SideLong) -> True
                                                                                _ -> False
                                                                        nextCount =
                                                                            if opposite && lstmFlipStrongOk
                                                                                then otLstmFlipCount ot + 1
                                                                                else 0
                                                                        ot' = ot{otLstmFlipCount = nextCount}
                                                                     in (Just ot', nextCount >= lstmFlipBars)
                                                    (Just ot, Just _, _) -> (Just ot{otLstmFlipCount = 0}, False)
                                                    _ -> (openTrade0, False)

                                        desiredSize0 =
                                            if desiredSideRaw == Nothing
                                                then 0
                                                else max 0 desiredSizeRaw

                                        desiredSide0 =
                                            if desiredSize0 <= 0 then Nothing else desiredSideRaw

                                        needsEntry = desiredSide0 /= Nothing && desiredSide0 /= posSide
                                        lstmEntryScale = if needsEntry then lstmEntryScaleRaw else 1

                                        trendOk =
                                            case desiredSide0 of
                                                Just side | needsEntry -> trendLookback <= 1 || trendOkAt t side
                                                _ -> True

                                        volOk = if needsEntry then volOkAt t else True

                                        snrScale =
                                            if minSignalToNoiseAdj <= 0
                                                then 1
                                                else case volPerBarAt t of
                                                    Just vol | vol > 0 -> clamp01 ((max 0 edgeRaw) / vol / minSignalToNoiseAdj)
                                                    _ -> 0

                                        snrScaleWeighted =
                                            if snrSizeWeight <= 0
                                                then 1
                                                else (1 - snrSizeWeight) + snrSizeWeight * snrScale

                                        snrOk =
                                            if needsEntry
                                                then
                                                    if snrSizeWeight <= 0
                                                        then signalToNoiseOkAt t minSignalToNoiseAdj (max 0 edgeRaw)
                                                        else snrScale > 0
                                                else True

                                        volTargetReady =
                                            if needsEntry
                                                then volTargetReadyAt t
                                                else True

                                        triLayerOk =
                                            case (needsEntry, desiredSide0) of
                                                (True, Just side) -> cloudOkAt t side && priceActionOkAt t side
                                                _ -> True

                                        slowCrossExit =
                                            case posSide of
                                                Just side ->
                                                    triLayerExitOnSlow
                                                        && cloudReady
                                                        && t > 0
                                                        && let slow = cloudSlowV V.! t
                                                               slowPrev = cloudSlowV V.! (t - 1)
                                                               priorClose = pricesV V.! (t - 1)
                                                            in if any isBad [slow, slowPrev, prev, priorClose]
                                                                then False
                                                                else case side of
                                                                    SideLong -> priorClose >= slowPrev && prev < slow
                                                                    SideShort -> priorClose <= slowPrev && prev > slow
                                                Nothing -> False

                                        kalmanBandExit =
                                            case posSide of
                                                Just side ->
                                                    case kalmanBandAt t of
                                                        Nothing -> False
                                                        Just (upper, lower) ->
                                                            let reversal =
                                                                    case (side, lstmCloseDir) of
                                                                        (SideLong, Just SideShort) -> True
                                                                        (SideShort, Just SideLong) -> True
                                                                        _ -> False
                                                                softThr = max 0 (min 1 (ecLstmConfidenceSoft cfg))
                                                                lowConfidence =
                                                                    case lstmConfScoreMaybe of
                                                                        Just score -> score < softThr
                                                                        Nothing -> False
                                                                hiNow = barHigh t
                                                                loNow = barLow t
                                                             in case side of
                                                                    SideLong ->
                                                                        not (isBad hiNow) && hiNow >= upper && (reversal || lowConfidence)
                                                                    SideShort ->
                                                                        not (isBad loNow) && loNow <= lower && (reversal || lowConfidence)
                                                Nothing -> False

                                        kalmanExit = slowCrossExit || kalmanBandExit

                                        desiredSide1 =
                                            if not trendOk || not volOk || not snrOk || not volTargetReady || not triLayerOk
                                                then Nothing
                                                else desiredSide0

                                        desiredSide2 =
                                            if lstmFlipExit || kalmanExit
                                                then Nothing
                                                else desiredSide1

                                        baseSizeTarget =
                                            case desiredSide2 of
                                                Nothing -> 0
                                                Just side ->
                                                    case mOpenSignal of
                                                        Just (sigSide, sigSize) | sigSide == side -> max 0 sigSize
                                                        _ ->
                                                            case openTradeFlip of
                                                                Just ot | otSide ot == side -> max 0 (otBaseSize ot)
                                                                _ -> max 0 desiredSizeRaw

                                        entryScale = if desiredSide2 /= Nothing && desiredSide2 /= posSide then lstmEntryScale else 1
                                        sizeScale =
                                            if desiredSide2 == Nothing
                                                then 1
                                                else volScaleAt t * riskScaleAt t * snrScaleWeighted
                                        sizeScaled = baseSizeTarget * entryScale * sizeScale
                                        sizeCapped = min maxPositionSize (max 0 sizeScaled)
                                        sizeFinal0 =
                                            if sizeCapped < minPositionSize && desiredSide2 /= posSide
                                                then 0
                                                else sizeCapped

                                        desiredSide =
                                            if sizeFinal0 <= 0 then Nothing else desiredSide2

                                        desiredSize = sizeFinal0

                                        holdForced = posSide /= Nothing && desiredSide /= posSide && holdBars < minHoldBars

                                        desiredSideHoldAdjusted =
                                            if holdForced
                                                then posSide
                                                else desiredSide

                                        desiredSizeHoldAdjusted =
                                            if holdForced
                                                then posSize
                                                else desiredSize

                                        noTradeActive =
                                            case noTradeWindows' of
                                                [] -> False
                                                windows -> any (\w -> timeWindowContains w (minuteOfDayAt t)) windows

                                        tradeLimitReached =
                                            case maxTradesPerDayLim of
                                                Just lim -> dayTrades1 >= lim
                                                Nothing -> False

                                        entryAttempt =
                                            case desiredSideHoldAdjusted of
                                                Just side -> posSide /= Just side
                                                Nothing -> False

                                        entryBlockReason =
                                            if entryAttempt && noTradeActive
                                                then Just "NO_TRADE_WINDOW"
                                                else
                                                    if entryAttempt && tradeLimitReached
                                                        then Just "MAX_TRADES_PER_DAY"
                                                        else Nothing

                                        (desiredSideFinal0, desiredSizeFinal0) =
                                            if cooldownActive
                                                then (Nothing, 0)
                                                else case entryBlockReason of
                                                    Nothing -> (desiredSideHoldAdjusted, desiredSizeHoldAdjusted)
                                                    Just _ ->
                                                        case posSide of
                                                            Just side -> (Just side, posSize)
                                                            Nothing -> (Nothing, 0)

                                        holdTooLong =
                                            case maxHoldBars of
                                                Nothing -> False
                                                Just lim -> posSide /= Nothing && holdBars >= lim

                                        (desiredSideFinal1, desiredSizeFinal1) =
                                            if holdTooLong
                                                then (Nothing, 0)
                                                else (desiredSideFinal0, desiredSizeFinal0)

                                        (desiredSideFinal, desiredSizeFinal) =
                                            if halted
                                                then (Nothing, 0)
                                                else (desiredSideFinal1, desiredSizeFinal1)

                                        exitReasonOverride =
                                            case () of
                                                _
                                                    | halted -> haltReason1
                                                    | holdTooLong -> Just (ExitOther "MAX_HOLD")
                                                    | lstmFlipExit -> Just (ExitOther "LSTM_FLIP")
                                                    | slowCrossExit -> Just (ExitOther "KALMAN_SLOW")
                                                    | kalmanBandExit -> Just (ExitOther "KALMAN_BAND")
                                                    | otherwise -> Nothing

                                        switchExitReason =
                                            case (exitReasonOverride, posSide) of
                                                (Just r, Just _) -> r
                                                _ -> ExitSignal

                                        closeTradeAt exitIndex why eqExit ot =
                                            Trade
                                                { trEntryIndex = otEntryIndex ot
                                                , trExitIndex = exitIndex
                                                , trEntryEquity = otEntryEquity ot
                                                , trExitEquity = eqExit
                                                , trReturn = eqExit / otEntryEquity ot - 1
                                                , trHoldingPeriods = otHoldingPeriods ot
                                                , trExitReason = Just why
                                                }

                                        openTradeFor side eqEntry baseSize =
                                            OpenTrade
                                                { otEntryIndex = t
                                                , otRebalanceAnchor = t
                                                , otEntryEquity = eqEntry
                                                , otHoldingPeriods = 0
                                                , otEntryPrice = prev
                                                , otTrail = prev
                                                , otSide = side
                                                , otBaseSize = max 0 baseSize
                                                , otLstmFlipCount = 0
                                                }

                                        openTradeUpdated =
                                            case (openTradeFlip, mOpenSignal) of
                                                (Just ot, Just (sigSide, sigSize))
                                                    | otSide ot == sigSide && sigSize > 0 ->
                                                        let ot' = ot{otBaseSize = max 0 sigSize}
                                                         in Just
                                                                ( if rebalanceResetOnSignal
                                                                    then ot'{otRebalanceAnchor = t}
                                                                    else ot'
                                                                )
                                                _ -> openTradeFlip

                                        rebalanceDelta = abs (desiredSizeFinal - posSize)
                                        rebalanceDue =
                                            if not rebalanceEnabled
                                                then False
                                                else
                                                    if rebalanceGlobal
                                                        then t `mod` rebalanceBars == 0
                                                        else case openTradeUpdated of
                                                            Just ot ->
                                                                let age = max 0 (t - otRebalanceAnchor ot)
                                                                 in age `mod` rebalanceBars == 0
                                                            Nothing -> t `mod` rebalanceBars == 0
                                        rebalanceOk =
                                            rebalanceEnabled
                                                && posSide /= Nothing
                                                && desiredSideFinal == posSide
                                                && rebalanceDue
                                                && rebalanceDelta > 0
                                                && rebalanceDelta >= rebalanceThreshold

                                        (posAfterSwitch, posSizeAfterSwitch, equityAfterSwitch, changes', openTrade', tradesAcc') =
                                            if desiredSideFinal == posSide
                                                then
                                                    if rebalanceOk
                                                        then
                                                            let eqRebalance = applyCost equity rebalanceDelta
                                                             in (posSide, desiredSizeFinal, eqRebalance, changes + 1, openTradeUpdated, tradesAcc)
                                                        else (posSide, posSize, equity, changes, openTradeUpdated, tradesAcc)
                                                else case desiredSideFinal of
                                                    Nothing ->
                                                        let eqExit = applyCost equity posSize
                                                            tradesAcc1 =
                                                                case openTradeFlip of
                                                                    Nothing -> tradesAcc
                                                                    Just ot -> closeTradeAt t switchExitReason eqExit ot : tradesAcc
                                                         in (Nothing, 0, eqExit, changes + 1, Nothing, tradesAcc1)
                                                    Just desiredSideFinal' ->
                                                        case posSide of
                                                            Nothing ->
                                                                let eqEntry = applyCost equity desiredSizeFinal
                                                                 in ( Just desiredSideFinal'
                                                                    , desiredSizeFinal
                                                                    , eqEntry
                                                                    , changes + 1
                                                                    , Just (openTradeFor desiredSideFinal' eqEntry baseSizeTarget)
                                                                    , tradesAcc
                                                                    )
                                                            Just _ ->
                                                                let eqExit = applyCost equity posSize
                                                                    tradesAcc1 =
                                                                        case openTradeFlip of
                                                                            Nothing -> tradesAcc
                                                                            Just ot -> closeTradeAt t ExitSignal eqExit ot : tradesAcc
                                                                    eqEntry = applyCost eqExit desiredSizeFinal
                                                                 in ( Just desiredSideFinal'
                                                                    , desiredSizeFinal
                                                                    , eqEntry
                                                                    , changes + 1
                                                                    , Just (openTradeFor desiredSideFinal' eqEntry baseSizeTarget)
                                                                    , tradesAcc1
                                                                    )

                                        equityAtClose =
                                            case posAfterSwitch of
                                                Just side
                                                    | posSizeAfterSwitch > 0 ->
                                                        let factor = markToMarket side prev nextClose
                                                            eq1 = equityAfterSwitch * (1 + posSizeAfterSwitch * (factor - 1))
                                                         in if isBad eq1 then equityAfterSwitch else eq1
                                                _ -> equityAfterSwitch

                                        (posFinal, posSizeFinal, equityFinal0, changesFinal, openTradeFinal, tradesFinal) =
                                            case (posAfterSwitch, openTrade') of
                                                (Just SideLong, Just ot0) ->
                                                    let otHeld = ot0{otHoldingPeriods = otHoldingPeriods ot0 + 1}
                                                        entryPx = otEntryPrice ot0
                                                        trail0 = otTrail ot0
                                                        mTp =
                                                            case takeProfitFracAt t of
                                                                Just tp -> Just (entryPx * (1 + tp))
                                                                _ -> Nothing
                                                        mSl =
                                                            case stopLossFracAt t of
                                                                Just sl -> Just (entryPx * (1 - sl))
                                                                _ -> Nothing
                                                        stopPx trail =
                                                            let mTs =
                                                                    case trailingStopFracAt t of
                                                                        Just ts -> Just (trail * (1 - ts))
                                                                        _ -> Nothing
                                                             in case (mSl, mTs) of
                                                                    (Nothing, Nothing) -> (Nothing, Nothing)
                                                                    (Just slPx, Nothing) -> (Just slPx, Just ExitStopLoss)
                                                                    (Nothing, Just tsPx) -> (Just tsPx, Just ExitTrailingStop)
                                                                    (Just slPx, Just tsPx) ->
                                                                        if tsPx > slPx
                                                                            then (Just tsPx, Just ExitTrailingStop)
                                                                            else (Just slPx, Just ExitStopLoss)
                                                        tpHit = maybe False (\tpPx -> hi >= tpPx) mTp
                                                        exitOrTrail =
                                                            case ecIntrabarFill cfg of
                                                                StopFirst ->
                                                                    let (mStop, stopWhy) = stopPx trail0
                                                                        stopHit = maybe False (\stPx -> lo <= stPx) mStop
                                                                     in if stopHit
                                                                            then (Just (maybe nextClose id mStop, stopWhy), trail0)
                                                                            else
                                                                                if tpHit
                                                                                    then (Just (maybe nextClose id mTp, Just ExitTakeProfit), trail0)
                                                                                    else (Nothing, max trail0 hi)
                                                                TakeProfitFirst ->
                                                                    if tpHit
                                                                        then (Just (maybe nextClose id mTp, Just ExitTakeProfit), trail0)
                                                                        else
                                                                            let trail1 = max trail0 hi
                                                                                (mStop, stopWhy) = stopPx trail1
                                                                                stopHit = maybe False (\stPx -> lo <= stPx) mStop
                                                                             in if stopHit
                                                                                    then (Just (maybe nextClose id mStop, stopWhy), trail1)
                                                                                    else (Nothing, trail1)
                                                     in case exitOrTrail of
                                                            (Just (exitPx, reason), _trailUsed) ->
                                                                let factor = markToMarket SideLong prev exitPx
                                                                    exitEq0 = equityAfterSwitch * (1 + posSizeAfterSwitch * (factor - 1))
                                                                    exitEq = applyCost exitEq0 posSizeAfterSwitch
                                                                    tr =
                                                                        Trade
                                                                            { trEntryIndex = otEntryIndex otHeld
                                                                            , trExitIndex = t + 1
                                                                            , trEntryEquity = otEntryEquity otHeld
                                                                            , trExitEquity = exitEq
                                                                            , trReturn = exitEq / otEntryEquity otHeld - 1
                                                                            , trHoldingPeriods = otHoldingPeriods otHeld
                                                                            , trExitReason = reason
                                                                            }
                                                                 in (Nothing, 0, exitEq, changes' + 1, Nothing, tr : tradesAcc')
                                                            (Nothing, trail1) ->
                                                                let otCont = otHeld{otTrail = trail1}
                                                                 in (Just SideLong, posSizeAfterSwitch, equityAtClose, changes', Just otCont, tradesAcc')
                                                (Just SideShort, Just ot0) ->
                                                    let otHeld = ot0{otHoldingPeriods = otHoldingPeriods ot0 + 1}
                                                        entryPx = otEntryPrice ot0
                                                        trail0 = otTrail ot0
                                                        mTp =
                                                            case takeProfitFracAt t of
                                                                Just tp -> Just (entryPx * (1 - tp))
                                                                _ -> Nothing
                                                        mSl =
                                                            case stopLossFracAt t of
                                                                Just sl -> Just (entryPx * (1 + sl))
                                                                _ -> Nothing
                                                        stopPx trail =
                                                            let mTs =
                                                                    case trailingStopFracAt t of
                                                                        Just ts -> Just (trail * (1 + ts))
                                                                        _ -> Nothing
                                                             in case (mSl, mTs) of
                                                                    (Nothing, Nothing) -> (Nothing, Nothing)
                                                                    (Just slPx, Nothing) -> (Just slPx, Just ExitStopLoss)
                                                                    (Nothing, Just tsPx) -> (Just tsPx, Just ExitTrailingStop)
                                                                    (Just slPx, Just tsPx) ->
                                                                        if tsPx < slPx
                                                                            then (Just tsPx, Just ExitTrailingStop)
                                                                            else (Just slPx, Just ExitStopLoss)
                                                        tpHit = maybe False (\tpPx -> lo <= tpPx) mTp
                                                        exitOrTrail =
                                                            case ecIntrabarFill cfg of
                                                                StopFirst ->
                                                                    let (mStop, stopWhy) = stopPx trail0
                                                                        stopHit = maybe False (\stPx -> hi >= stPx) mStop
                                                                     in if stopHit
                                                                            then (Just (maybe nextClose id mStop, stopWhy), trail0)
                                                                            else
                                                                                if tpHit
                                                                                    then (Just (maybe nextClose id mTp, Just ExitTakeProfit), trail0)
                                                                                    else (Nothing, min trail0 lo)
                                                                TakeProfitFirst ->
                                                                    if tpHit
                                                                        then (Just (maybe nextClose id mTp, Just ExitTakeProfit), trail0)
                                                                        else
                                                                            let trail1 = min trail0 lo
                                                                                (mStop, stopWhy) = stopPx trail1
                                                                                stopHit = maybe False (\stPx -> hi >= stPx) mStop
                                                                             in if stopHit
                                                                                    then (Just (maybe nextClose id mStop, stopWhy), trail1)
                                                                                    else (Nothing, trail1)
                                                     in case exitOrTrail of
                                                            (Just (exitPx, reason), _trailUsed) ->
                                                                let factor = markToMarket SideShort prev exitPx
                                                                    exitEq0 = equityAfterSwitch * (1 + posSizeAfterSwitch * (factor - 1))
                                                                    exitEq = applyCost exitEq0 posSizeAfterSwitch
                                                                    tr =
                                                                        Trade
                                                                            { trEntryIndex = otEntryIndex otHeld
                                                                            , trExitIndex = t + 1
                                                                            , trEntryEquity = otEntryEquity otHeld
                                                                            , trExitEquity = exitEq
                                                                            , trReturn = exitEq / otEntryEquity otHeld - 1
                                                                            , trHoldingPeriods = otHoldingPeriods otHeld
                                                                            , trExitReason = reason
                                                                            }
                                                                 in (Nothing, 0, exitEq, changes' + 1, Nothing, tr : tradesAcc')
                                                            (Nothing, trail1) ->
                                                                let otCont = otHeld{otTrail = trail1}
                                                                 in (Just SideShort, posSizeAfterSwitch, equityAtClose, changes', Just otCont, tradesAcc')
                                                _ -> (posAfterSwitch, posSizeAfterSwitch, equityAtClose, changes', openTrade', tradesAcc')

                                        (fundingSide, fundingSize) =
                                            if fundingOnOpen
                                                then (posSideStart, posSizeStart)
                                                else (posFinal, posSizeFinal)

                                        equityFinal =
                                            case fundingSide of
                                                Just side -> applyFunding equityFinal0 fundingSize side
                                                _ -> equityFinal0

                                        (posFinal2, posSizeFinal2, equityFinal2, changesFinal2, openTradeFinal2, tradesFinal2, dead2) =
                                            if isBad equityFinal || equityFinal <= 0
                                                then
                                                    let exitEq = if isBad equityFinal then 0 else max 0 equityFinal
                                                        (tradesOut, changesOut) =
                                                            case openTradeFinal of
                                                                Nothing -> (tradesFinal, changesFinal)
                                                                Just otHeld ->
                                                                    let tr =
                                                                            Trade
                                                                                { trEntryIndex = otEntryIndex otHeld
                                                                                , trExitIndex = t + 1
                                                                                , trEntryEquity = otEntryEquity otHeld
                                                                                , trExitEquity = exitEq
                                                                                , trReturn = exitEq / otEntryEquity otHeld - 1
                                                                                , trHoldingPeriods = otHoldingPeriods otHeld
                                                                                , trExitReason = Just ExitLiquidation
                                                                                }
                                                                     in (tr : tradesFinal, changesFinal + 1)
                                                     in (Nothing, 0, exitEq, changesOut, Nothing, tradesOut, True)
                                                else (posFinal, posSizeFinal, equityFinal, changesFinal, openTradeFinal, tradesFinal, False)

                                        -- Re-check risk after the bar and flatten at close if limits are breached.
                                        (posFinal3, posSizeFinal3, equityFinal3, changesFinal3, openTradeFinal3, tradesFinal3, haltReason2) =
                                            if dead2 || haltReason1 /= Nothing
                                                then (posFinal2, posSizeFinal2, equityFinal2, changesFinal2, openTradeFinal2, tradesFinal2, haltReason1)
                                                else
                                                    let peakEqAfter = max peakEq1 equityFinal2
                                                        drawdownAfter =
                                                            if peakEqAfter > 0
                                                                then max 0 (1 - equityFinal2 / peakEqAfter)
                                                                else 0
                                                        dailyLossAfter =
                                                            if dayStartEq1 > 0
                                                                then max 0 (1 - equityFinal2 / dayStartEq1)
                                                                else 0
                                                        weeklyLossAfter =
                                                            if weekStartEq1 > 0
                                                                then max 0 (1 - equityFinal2 / weekStartEq1)
                                                                else 0
                                                        expectancyAfter =
                                                            if expectancyLookback <= 0
                                                                then Nothing
                                                                else
                                                                    let recent = take expectancyLookback tradesFinal2
                                                                     in if length recent < expectancyLookback
                                                                            then Nothing
                                                                            else Just (meanList (map trReturn recent))
                                                        riskHaltReason2 =
                                                            case () of
                                                                _
                                                                    | maybe False (\lim -> dailyLossAfter >= lim) maxDailyLossLim -> Just ExitMaxDailyLoss
                                                                    | maybe False (\lim -> weeklyLossAfter >= lim) maxWeeklyLossLim -> Just ExitMaxWeeklyLoss
                                                                    | maybe False (\lim -> drawdownAfter >= lim) maxDrawdownLim -> Just ExitMaxDrawdown
                                                                    | maybe False (\lim -> maybe False (< lim) expectancyAfter) minExpectancy -> Just (ExitOther "NEGATIVE_EXPECTANCY")
                                                                    | otherwise -> Nothing
                                                     in case riskHaltReason2 of
                                                            Nothing -> (posFinal2, posSizeFinal2, equityFinal2, changesFinal2, openTradeFinal2, tradesFinal2, Nothing)
                                                            Just r ->
                                                                if posFinal2 == Nothing
                                                                    then (posFinal2, posSizeFinal2, equityFinal2, changesFinal2, openTradeFinal2, tradesFinal2, Just r)
                                                                    else
                                                                        let eqExit = applyCost equityFinal2 posSizeFinal2
                                                                            tradesAcc1 =
                                                                                case openTradeFinal2 of
                                                                                    Nothing -> tradesFinal2
                                                                                    Just ot -> closeTradeAt (t + 1) r eqExit ot : tradesFinal2
                                                                            changesOut = changesFinal2 + 1
                                                                         in (Nothing, 0, eqExit, changesOut, Nothing, tradesAcc1, Just r)

                                        exitedToFlat =
                                            posFinal3 == Nothing && (posAfterSwitch /= Nothing || posSide /= Nothing)

                                        maxHoldCooldown =
                                            if holdTooLong
                                                then 1
                                                else 0

                                        cooldownAfterExit = max cooldownBars maxHoldCooldown

                                        cooldownNext =
                                            if posFinal3 == Nothing
                                                then if exitedToFlat then cooldownAfterExit else cooldownNext0
                                                else 0
                                        entryOccurred =
                                            case (posSide, posFinal3) of
                                                (Nothing, Just _) -> True
                                                (Just side0, Just side1) -> side0 /= side1
                                                _ -> False
                                        dayTradesNext =
                                            if entryOccurred
                                                then dayTrades1 + 1
                                                else dayTrades1
                                        riskStateNext =
                                            RiskState
                                                { rsPeakEquity = max peakEq1 equityFinal3
                                                , rsDayKey = dayKey1
                                                , rsDayStartEquity = dayStartEq1
                                                , rsWeekKey = weekKey1
                                                , rsWeekStartEquity = weekStartEq1
                                                , rsDayTrades = dayTradesNext
                                                , rsHaltReason = haltReason2
                                                }
                                     in ( posFinal3
                                        , posSizeFinal3
                                        , equityFinal3
                                        , equityFinal3 : eqAcc
                                        , (maybe 0 sideSign posAfterSwitch * posSizeAfterSwitch) : posAcc
                                        , agreeOk : agreeAcc
                                        , agreeValid : agreeValidAcc
                                        , changesFinal3
                                        , openTradeFinal3
                                        , tradesFinal3
                                        , dead2
                                        , cooldownNext
                                        , riskStateNext
                                        , factorOpenNext
                                        , factorCloseNext
                                        )

                        (_finalPos, finalPosSize, finalEq, eqRev, posRev, agreeRev, agreeValidRev, changes, openTrade, tradesRev, _deadFinal, _cooldownFinal, _riskFinal, _factorOpenFinal, _factorCloseFinal) =
                            foldl'
                                stepFn
                                ( Nothing :: Maybe PositionSide
                                , 0 :: Double
                                , 1.0
                                , [1.0]
                                , []
                                , []
                                , []
                                , 0 :: Int
                                , Nothing :: Maybe OpenTrade
                                , []
                                , False
                                , 0 :: Int
                                , RiskState
                                    { rsPeakEquity = 1.0
                                    , rsDayKey = dayKeyAt 0
                                    , rsDayStartEquity = 1.0
                                    , rsWeekKey = weekKeyAt 0
                                    , rsWeekStartEquity = 1.0
                                    , rsDayTrades = 0
                                    , rsHaltReason = Nothing
                                    }
                                , 1.0
                                , 1.0
                                )
                                [0 .. stepCount - 1]

                        (eqRev', tradesRev') =
                            case openTrade of
                                Nothing -> (eqRev, tradesRev)
                                Just ot ->
                                    let exitEq = applyCost finalEq finalPosSize
                                        tr =
                                            Trade
                                                { trEntryIndex = otEntryIndex ot
                                                , trExitIndex = stepCount
                                                , trEntryEquity = otEntryEquity ot
                                                , trExitEquity = exitEq
                                                , trReturn = exitEq / otEntryEquity ot - 1
                                                , trHoldingPeriods = otHoldingPeriods ot
                                                , trExitReason = Just ExitEod
                                                }
                                        eqRev1 =
                                            case eqRev of
                                                [] -> [exitEq]
                                                (_ : rest) -> exitEq : rest
                                     in (eqRev1, tr : tradesRev)
                        eqCurve = reverse eqRev'
                     in BacktestResult
                            { brEquityCurve = eqCurve
                            , brPositions = reverse posRev
                            , brAgreementOk = reverse agreeRev
                            , brAgreementValid = reverse agreeValidRev
                            , brPositionChanges = changes
                            , brTrades = reverse tradesRev'
                            }
