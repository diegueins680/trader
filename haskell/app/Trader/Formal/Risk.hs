module Trader.Formal.Risk (
    HaltInputs (..),
    RiskVerificationReport (..),
    specRiskHalt,
    verifyFormalRisk,
) where

import Control.Applicative ((<|>))
import Data.Maybe (isJust, isNothing)
import Trader.Trading (ExitReason (..))

-- | Inputs to the risk halt decision, extracted from the simulation loop.
data HaltInputs = HaltInputs
    { hiPrevHaltReason :: !(Maybe ExitReason)
    , hiDayChanged :: !Bool
    , hiWeekChanged :: !Bool
    , hiDailyLoss :: !Double
    , hiWeeklyLoss :: !Double
    , hiDrawdown :: !Double
    , hiExpectancy :: !(Maybe Double)
    , hiMaxDailyLossLim :: !(Maybe Double)
    , hiMaxWeeklyLossLim :: !(Maybe Double)
    , hiMaxDrawdownLim :: !(Maybe Double)
    , hiMinExpectancy :: !(Maybe Double)
    }
    deriving (Eq, Show)

{- | Naive spec of the halt logic.

1. If previously halted for daily loss and the day changed, reset.
2. If previously halted for weekly loss and the week changed, reset.
3. Otherwise preserve the previous halt.
4. If still not halted, check new risk conditions.

This mirrors the logic in 'Trader.Trading' exactly but expressed as a
pure function with no simulation baggage.
-}
specRiskHalt :: HaltInputs -> Maybe ExitReason
specRiskHalt hi =
    let haltReasonBase =
            case (hiPrevHaltReason hi, hiDayChanged hi, hiWeekChanged hi) of
                (Just ExitMaxDailyLoss, True, _) -> Nothing
                (Just ExitMaxWeeklyLoss, _, True) -> Nothing
                _ -> hiPrevHaltReason hi
        riskHaltReason =
            case haltReasonBase of
                Just _ -> Nothing
                Nothing ->
                    case () of
                        _
                            | maybe False (hiDailyLoss hi >=) (hiMaxDailyLossLim hi) ->
                                Just ExitMaxDailyLoss
                            | maybe False (hiWeeklyLoss hi >=) (hiMaxWeeklyLossLim hi) ->
                                Just ExitMaxWeeklyLoss
                            | maybe False (hiDrawdown hi >=) (hiMaxDrawdownLim hi) ->
                                Just ExitMaxDrawdown
                            | maybe False (\lim -> maybe False (< lim) (hiExpectancy hi)) (hiMinExpectancy hi) ->
                                Just (ExitOther "NEGATIVE_EXPECTANCY")
                            | otherwise -> Nothing
     in haltReasonBase <|> riskHaltReason

-- ---------------------------------------------------------------------------
-- Verification report
-- ---------------------------------------------------------------------------

data RiskVerificationReport = RiskVerificationReport
    { fvrRiskHaltMonotone :: !Bool
    , fvrRiskHaltResetDaily :: !Bool
    , fvrRiskHaltResetWeekly :: !Bool
    , fvrRiskHaltPreservesOther :: !Bool
    , fvrRiskHaltNoFalsePositive :: !Bool
    }
    deriving (Eq, Show)

-- | Exhaustive enumeration over bounded risk states.
verifyFormalRisk :: RiskVerificationReport
verifyFormalRisk =
    let
        -- Small grid of doubles
        doubles = [0, 0.01, 0.05, 0.1, 0.5, 1.0]
        mDoubles = [Nothing, Just 0, Just 0.05, Just 0.1, Just 1.0]
        prevReasons =
            [ Nothing
            , Just ExitMaxDailyLoss
            , Just ExitMaxWeeklyLoss
            , Just ExitMaxDrawdown
            , Just (ExitOther "NEGATIVE_EXPECTANENCY")
            , Just (ExitOther "MANUAL")
            ]
        booleans = [False, True]
        expectancys = [Nothing, Just (-0.1), Just 0, Just 0.01]

        allInputs =
            [ HaltInputs
                { hiPrevHaltReason = pr
                , hiDayChanged = dc
                , hiWeekChanged = wc
                , hiDailyLoss = dl
                , hiWeeklyLoss = wl
                , hiDrawdown = dd
                , hiExpectancy = ex
                , hiMaxDailyLossLim = mdl
                , hiMaxWeeklyLossLim = mwl
                , hiMaxDrawdownLim = mdd
                , hiMinExpectancy = me
                }
            | pr <- prevReasons
            , dc <- booleans
            , wc <- booleans
            , dl <- doubles
            , wl <- doubles
            , dd <- doubles
            , ex <- expectancys
            , mdl <- mDoubles
            , mwl <- mDoubles
            , mdd <- mDoubles
            , me <- mDoubles
            ]

        -- If halted, drawdown must be >= limit OR daily loss >= limit OR
        -- weekly loss >= limit OR expectancy < minExpectancy, OR it is a
        -- preserved previous halt.
        haltMonotone =
            all
                ( \hi ->
                    let result = specRiskHalt hi
                     in case result of
                            Just ExitMaxDrawdown ->
                                maybe False (hiDrawdown hi >=) (hiMaxDrawdownLim hi)
                                    || hiPrevHaltReason hi == Just ExitMaxDrawdown
                            Just ExitMaxDailyLoss ->
                                maybe False (hiDailyLoss hi >=) (hiMaxDailyLossLim hi)
                                    || (hiPrevHaltReason hi == Just ExitMaxDailyLoss && not (hiDayChanged hi))
                            Just ExitMaxWeeklyLoss ->
                                maybe False (hiWeeklyLoss hi >=) (hiMaxWeeklyLossLim hi)
                                    || (hiPrevHaltReason hi == Just ExitMaxWeeklyLoss && not (hiWeekChanged hi))
                            Just (ExitOther "NEGATIVE_EXPECTANCY") ->
                                maybe False (\lim -> maybe False (< lim) (hiExpectancy hi)) (hiMinExpectancy hi)
                                    || hiPrevHaltReason hi == Just (ExitOther "NEGATIVE_EXPECTANCY")
                            Just (ExitOther _) ->
                                hiPrevHaltReason hi == result
                            Nothing -> True
                )
                allInputs

        -- Daily loss halt resets on day change (unless the new day is already
        -- breaching the limit).
        resetDaily =
            all
                ( \hi ->
                    let result = specRiskHalt hi
                        stillBreached = maybe False (hiDailyLoss hi >=) (hiMaxDailyLossLim hi)
                     in not
                            ( hiPrevHaltReason hi == Just ExitMaxDailyLoss
                                && hiDayChanged hi
                                && not stillBreached
                            )
                            || (result /= Just ExitMaxDailyLoss)
                )
                allInputs

        -- Weekly loss halt resets on week change (unless the new week is already
        -- breaching the limit).
        resetWeekly =
            all
                ( \hi ->
                    let result = specRiskHalt hi
                        stillBreached = maybe False (hiWeeklyLoss hi >=) (hiMaxWeeklyLossLim hi)
                     in not
                            ( hiPrevHaltReason hi == Just ExitMaxWeeklyLoss
                                && hiWeekChanged hi
                                && not stillBreached
                            )
                            || (result /= Just ExitMaxWeeklyLoss)
                )
                allInputs

        -- If previously halted for a non-time-bound reason, it is preserved.
        preservesOther =
            all
                ( \hi ->
                    let result = specRiskHalt hi
                     in case hiPrevHaltReason hi of
                            Just r
                                | r /= ExitMaxDailyLoss && r /= ExitMaxWeeklyLoss ->
                                    result == Just r || result == hiPrevHaltReason hi
                            _ -> True
                )
                allInputs

        -- If no limits are set and no previous halt, result is Nothing.
        noFalsePositive =
            all
                ( \hi ->
                    let result = specRiskHalt hi
                     in not
                            ( isNothing (hiPrevHaltReason hi)
                                && isNothing (hiMaxDailyLossLim hi)
                                && isNothing (hiMaxWeeklyLossLim hi)
                                && isNothing (hiMaxDrawdownLim hi)
                                && isNothing (hiMinExpectancy hi)
                            )
                            || isNothing result
                )
                allInputs
     in
        RiskVerificationReport
            { fvrRiskHaltMonotone = haltMonotone
            , fvrRiskHaltResetDaily = resetDaily
            , fvrRiskHaltResetWeekly = resetWeekly
            , fvrRiskHaltPreservesOther = preservesOther
            , fvrRiskHaltNoFalsePositive = noFalsePositive
            }
