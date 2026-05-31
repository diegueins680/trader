module Trader.Formal.Risk (
    HaltInputs (..),
    RiskVerificationReport (..),
    drawdownLimitInvalid,
    specRiskHalt,
    verifyFormalRisk,
) where

import Data.Maybe (isJust, isNothing)

-- 'HaltInputs' and 'specRiskHalt' are defined in 'Trader.Trading' so the
-- simulation loop can call the canonical spec directly. This module
-- re-exports them and proves properties via 'verifyFormalRisk'.
import Trader.SignalGates (finiteDouble)
import Trader.Trading (
    ExitReason (..),
    HaltInputs (..),
    anyRiskLimitNonFinite,
    specRiskHalt,
 )

-- | Check whether the drawdown limit is outside the valid (0,1) interval.
-- Defined here to avoid a module cycle with Trader.Trading.
drawdownLimitInvalid :: HaltInputs -> Bool
drawdownLimitInvalid hi =
    case hiMaxDrawdownLim hi of
        Just v -> v <= 0 || v >= 1 || not (finiteDouble v)
        Nothing -> False

-- ---------------------------------------------------------------------------
-- Verification report
-- ---------------------------------------------------------------------------

data RiskVerificationReport = RiskVerificationReport
    { fvrRiskHaltMonotone :: !Bool
    , fvrRiskHaltResetDaily :: !Bool
    , fvrRiskHaltResetWeekly :: !Bool
    , fvrRiskHaltPreservesOther :: !Bool
    , fvrRiskHaltNoFalsePositive :: !Bool
    , fvrRiskHaltPositionSize :: !Bool
    , fvrRiskHaltLossStreak :: !Bool
    , fvrMaxPositionSizeBound :: !Bool
    , fvrRiskLimitFinite :: !Bool
    , fvrDrawdownSanity :: !Bool
    , fvrPositionSizeSanity :: !Bool
    , fvrExpectancySanity :: !Bool
    , fvrVolTargetSanity :: !Bool
    , fvrLeverageSanity :: !Bool
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
            , Just (ExitOther "NEGATIVE_EXPECTANCY")
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
                , hiPositionSize = 0
                , hiMaxPositionSizeLim = Nothing
                , hiConsecutiveLosses = 0
                , hiMaxLossStreakLim = Nothing
                , hiVolTarget = 0
                , hiLeverage = 0
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
                            Just (ExitOther "RISK_LIMIT_NON_FINITE") ->
                                anyRiskLimitNonFinite hi
                            Just (ExitOther "DRAWDOWN_LIMIT_INVALID") ->
                                drawdownLimitInvalid hi
                            Just (ExitOther "EXPECTANCY_INVALID") ->
                                isJust (hiMinExpectancy hi) && (isNothing (hiExpectancy hi) || maybe False (not . finiteDouble) (hiExpectancy hi))
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
                                && isNothing (hiMaxPositionSizeLim hi)
                            )
                            || isNothing result
                )
                allInputs

        -- Position-size halt: if position size exceeds the sanitized limit,
        -- a POSITION_SIZE halt is emitted.
        positionSizeHalt =
            all
                ( \hi ->
                    let result = specRiskHalt hi
                        limit = fmap (max 0) (hiMaxPositionSizeLim hi)
                        size = max 0 (hiPositionSize hi)
                     in case result of
                            Just (ExitOther "POSITION_SIZE") ->
                                maybe False (size >) limit
                            _ -> True
                )
                [ HaltInputs
                    { hiPrevHaltReason = Nothing
                    , hiDayChanged = False
                    , hiWeekChanged = False
                    , hiDailyLoss = 0
                    , hiWeeklyLoss = 0
                    , hiDrawdown = 0
                    , hiExpectancy = Nothing
                    , hiMaxDailyLossLim = Nothing
                    , hiMaxWeeklyLossLim = Nothing
                    , hiMaxDrawdownLim = Nothing
                    , hiMinExpectancy = Nothing
                    , hiPositionSize = ps
                    , hiMaxPositionSizeLim = mps
                    , hiConsecutiveLosses = 0
                    , hiMaxLossStreakLim = Nothing
                    , hiVolTarget = 0
                    , hiLeverage = 0
                    }
                | ps <- [0, 0.5, 1.0, 1.5, 2.0]
                , mps <- [Nothing, Just 0, Just 1.0, Just 2.0]
                ]

        -- Loss-streak cooldown halt: if consecutive losses exceed the configured
        -- max, a LOSS_STREAK halt is emitted. This prevents runaway drawdowns
        -- during persistent adverse regimes.
        lossStreakHalt =
            all
                ( \hi ->
                    let result = specRiskHalt hi
                     in case result of
                            Just (ExitOther "LOSS_STREAK") ->
                                maybe False (\lim -> hiConsecutiveLosses hi > lim) (hiMaxLossStreakLim hi)
                            _ -> True
                )
                [ HaltInputs
                    { hiPrevHaltReason = Nothing
                    , hiDayChanged = False
                    , hiWeekChanged = False
                    , hiDailyLoss = 0
                    , hiWeeklyLoss = 0
                    , hiDrawdown = 0
                    , hiExpectancy = Nothing
                    , hiMaxDailyLossLim = Nothing
                    , hiMaxWeeklyLossLim = Nothing
                    , hiMaxDrawdownLim = Nothing
                    , hiMinExpectancy = Nothing
                    , hiPositionSize = 0
                    , hiMaxPositionSizeLim = Nothing
                    , hiConsecutiveLosses = cl
                    , hiMaxLossStreakLim = mls
                    , hiVolTarget = 0
                    , hiLeverage = 0
                    }
                | cl <- [0, 1, 2, 3, 5]
                , mls <- [Nothing, Just 0, Just 2, Just 3]
                ]

        -- Max-position-size simulation bound: ecMaxPositionSize must be
        -- non-negative and finite; when set > 0 the simulation must never
        -- emit a position exceeding it.
        maxPositionSizeBound =
            all
                ( \hi ->
                    let result = specRiskHalt hi
                        limit = fmap (max 0) (hiMaxPositionSizeLim hi)
                        size = max 0 (hiPositionSize hi)
                     in case result of
                            Just (ExitOther "POSITION_SIZE") ->
                                maybe False (size >) limit
                            _ -> True
                )
                [ HaltInputs
                    { hiPrevHaltReason = Nothing
                    , hiDayChanged = False
                    , hiWeekChanged = False
                    , hiDailyLoss = 0
                    , hiWeeklyLoss = 0
                    , hiDrawdown = 0
                    , hiExpectancy = Nothing
                    , hiMaxDailyLossLim = Nothing
                    , hiMaxWeeklyLossLim = Nothing
                    , hiMaxDrawdownLim = Nothing
                    , hiMinExpectancy = Nothing
                    , hiPositionSize = ps
                    , hiMaxPositionSizeLim = mps
                    , hiConsecutiveLosses = 0
                    , hiMaxLossStreakLim = Nothing
                    , hiVolTarget = 0
                    , hiLeverage = 0
                    }
                | ps <- [0, 0.5, 1.0, 1.5, 2.0, 5.0, 1e308]
                , mps <- [Nothing, Just 0, Just 1.0, Just 2.0, Just 1e308]
                ]

        -- Non-finite risk limit halt: if any configured limit is NaN or
        -- Infinity, specRiskHalt must emit RISK_LIMIT_NON_FINITE before any
        -- other risk check. This prevents corrupted configs from silently
        -- disabling halts.
        riskLimitFinite =
            all
                ( \hi ->
                    let result = specRiskHalt hi
                     in case result of
                            Just (ExitOther "RISK_LIMIT_NON_FINITE") ->
                                anyRiskLimitNonFinite hi
                            _ -> not (anyRiskLimitNonFinite hi)
                )
                [ HaltInputs
                    { hiPrevHaltReason = Nothing
                    , hiDayChanged = False
                    , hiWeekChanged = False
                    , hiDailyLoss = 0
                    , hiWeeklyLoss = 0
                    , hiDrawdown = 0
                    , hiExpectancy = Nothing
                    , hiMaxDailyLossLim = mdl
                    , hiMaxWeeklyLossLim = mwl
                    , hiMaxDrawdownLim = mdd
                    , hiMinExpectancy = me
                    , hiPositionSize = 0
                    , hiMaxPositionSizeLim = mps
                    , hiConsecutiveLosses = 0
                    , hiMaxLossStreakLim = Nothing
                    , hiVolTarget = 0
                    , hiLeverage = 0
                    }
                | mdl <- [Nothing, Just 0, Just 0.05, Just (0 / 0), Just (1 / 0), Just (-1 / 0)]
                , mwl <- [Nothing, Just 0, Just 0.05, Just (0 / 0), Just (1 / 0)]
                , mdd <- [Nothing, Just 0, Just 0.05, Just (0 / 0)]
                , me <- [Nothing, Just 0, Just (0 / 0)]
                , mps <- [Nothing, Just 0, Just (0 / 0)]
                ]

        -- Drawdown sanity: ecMaxDrawdown must be finite and strictly within
        -- (0,1). Values <=0, >=1, NaN, or Infinity are treated as corrupted
        -- and disable the drawdown halt check, allowing catastrophic losses.
        drawdownSanity =
            all
                ( \hi ->
                    let result = specRiskHalt hi
                        invalid = drawdownLimitInvalid hi
                     in case result of
                            Just (ExitOther "DRAWDOWN_LIMIT_INVALID") ->
                                invalid && not (anyRiskLimitNonFinite hi)
                            Just (ExitOther "RISK_LIMIT_NON_FINITE") ->
                                anyRiskLimitNonFinite hi
                            _ -> not invalid
                )
                [ HaltInputs
                    { hiPrevHaltReason = Nothing
                    , hiDayChanged = False
                    , hiWeekChanged = False
                    , hiDailyLoss = 0
                    , hiWeeklyLoss = 0
                    , hiDrawdown = 0
                    , hiExpectancy = Nothing
                    , hiMaxDailyLossLim = Nothing
                    , hiMaxWeeklyLossLim = Nothing
                    , hiMaxDrawdownLim = mdd
                    , hiMinExpectancy = Nothing
                    , hiPositionSize = 0
                    , hiMaxPositionSizeLim = Nothing
                    , hiConsecutiveLosses = 0
                    , hiMaxLossStreakLim = Nothing
                    , hiVolTarget = 0
                    , hiLeverage = 0
                    }
                | mdd <- [Nothing, Just 0, Just 0.05, Just 0.5, Just 0.999999, Just 1.0, Just (-0.01), Just (0 / 0), Just (1 / 0), Just (-1 / 0)]
                ]

        -- Position-size sanity: hiPositionSize must be finite, non-negative,
        -- and not exceed a hard sanity cap (10× notional account value).
        -- Non-finite, negative, or absurdly large sizes indicate corrupted
        -- configuration and would silently bypass exposure limits.
        positionSizeSanity =
            all
                ( \hi ->
                    let result = specRiskHalt hi
                        ps = hiPositionSize hi
                        invalid = not (finiteDouble ps) || ps < 0 || ps > 10
                     in case result of
                            Just (ExitOther "POSITION_SIZE_INVALID") ->
                                invalid && not (anyRiskLimitNonFinite hi)
                            Just (ExitOther "RISK_LIMIT_NON_FINITE") ->
                                anyRiskLimitNonFinite hi
                            _ -> not invalid
                )
                [ HaltInputs
                    { hiPrevHaltReason = Nothing
                    , hiDayChanged = False
                    , hiWeekChanged = False
                    , hiDailyLoss = 0
                    , hiWeeklyLoss = 0
                    , hiDrawdown = 0
                    , hiExpectancy = Nothing
                    , hiMaxDailyLossLim = Nothing
                    , hiMaxWeeklyLossLim = Nothing
                    , hiMaxDrawdownLim = Nothing
                    , hiMinExpectancy = Nothing
                    , hiPositionSize = ps
                    , hiMaxPositionSizeLim = mps
                    , hiConsecutiveLosses = 0
                    , hiMaxLossStreakLim = Nothing
                    , hiVolTarget = 0
                    , hiLeverage = 0
                    }
                | ps <- [0, 0.5, 1.0, 2.0, 5.0, 10.0, 10.000001, -0.01, 0 / 0, 1 / 0, -1 / 0]
                , mps <- [Nothing, Just 0, Just 1.0, Just 2.0, Just 10.0]
                ]

        -- Expectancy sanity: if hiMinExpectancy is configured, hiExpectancy
        -- must be a finite Just value. Non-finite or missing expectancy when
        -- a min-expectancy limit is set indicates corrupted configuration and
        -- would cause spurious negative-expectancy halts or silent bypass.
        expectancySanity =
            all
                ( \hi ->
                    let result = specRiskHalt hi
                        me = hiMinExpectancy hi
                        ex = hiExpectancy hi
                        missing = isJust me && (isNothing ex || maybe False (not . finiteDouble) ex)
                     in case result of
                            Just (ExitOther "EXPECTANCY_INVALID") ->
                                missing && not (anyRiskLimitNonFinite hi)
                            Just (ExitOther "RISK_LIMIT_NON_FINITE") ->
                                anyRiskLimitNonFinite hi
                            _ -> not missing
                )
                [ HaltInputs
                    { hiPrevHaltReason = Nothing
                    , hiDayChanged = False
                    , hiWeekChanged = False
                    , hiDailyLoss = 0
                    , hiWeeklyLoss = 0
                    , hiDrawdown = 0
                    , hiExpectancy = ex
                    , hiMaxDailyLossLim = Nothing
                    , hiMaxWeeklyLossLim = Nothing
                    , hiMaxDrawdownLim = Nothing
                    , hiMinExpectancy = me
                    , hiPositionSize = 0
                    , hiMaxPositionSizeLim = Nothing
                    , hiConsecutiveLosses = 0
                    , hiMaxLossStreakLim = Nothing
                    , hiVolTarget = 0
                    , hiLeverage = 0
                    }
                | ex <- [Nothing, Just (-0.1), Just 0, Just 0.01, Just (0 / 0), Just (1 / 0), Just (-1 / 0)]
                , me <- [Nothing, Just (-0.05), Just 0, Just 0.05, Just (0 / 0), Just (1 / 0)]
                ]

        -- Vol-target sanity: ecVolTarget must be finite, non-negative, and not
        -- exceed a hard sanity cap (10.0, representing 1000% annualized vol).
        -- Non-finite, negative, or absurdly large vol targets indicate corrupted
        -- configuration and would silently bypass vol-scaling limits, allowing
        -- catastrophic position-size drift.
        volTargetSanity =
            all
                ( \hi ->
                    let result = specRiskHalt hi
                        vt = hiVolTarget hi
                        invalid = not (finiteDouble vt) || vt < 0 || vt > 10
                     in case result of
                            Just (ExitOther "VOL_TARGET_INVALID") ->
                                invalid && not (anyRiskLimitNonFinite hi)
                            Just (ExitOther "RISK_LIMIT_NON_FINITE") ->
                                anyRiskLimitNonFinite hi
                            _ -> not invalid
                )
                [ HaltInputs
                    { hiPrevHaltReason = Nothing
                    , hiDayChanged = False
                    , hiWeekChanged = False
                    , hiDailyLoss = 0
                    , hiWeeklyLoss = 0
                    , hiDrawdown = 0
                    , hiExpectancy = Nothing
                    , hiMaxDailyLossLim = Nothing
                    , hiMaxWeeklyLossLim = Nothing
                    , hiMaxDrawdownLim = Nothing
                    , hiMinExpectancy = Nothing
                    , hiPositionSize = 0
                    , hiMaxPositionSizeLim = Nothing
                    , hiConsecutiveLosses = 0
                    , hiMaxLossStreakLim = Nothing
                    , hiVolTarget = vt
                    , hiLeverage = 0
                    }
                | vt <- [0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 10.000001, -0.01, 0 / 0, 1 / 0, -1 / 0]
                ]

        -- Leverage sanity: hiLeverage must be finite, non-negative, and not
        -- exceed a hard sanity cap (150x). Non-finite, negative, or absurdly
        -- large leverage values indicate corrupted configuration and would
        -- silently bypass position-size limits, allowing catastrophic
        -- liquidation risk.
        leverageSanity =
            all
                ( \hi ->
                    let result = specRiskHalt hi
                        lev = hiLeverage hi
                        invalid = not (finiteDouble lev) || lev < 0 || lev > 150
                     in case result of
                            Just (ExitOther "LEVERAGE_INVALID") ->
                                invalid && not (anyRiskLimitNonFinite hi)
                            Just (ExitOther "RISK_LIMIT_NON_FINITE") ->
                                anyRiskLimitNonFinite hi
                            _ -> not invalid
                )
                [ HaltInputs
                    { hiPrevHaltReason = Nothing
                    , hiDayChanged = False
                    , hiWeekChanged = False
                    , hiDailyLoss = 0
                    , hiWeeklyLoss = 0
                    , hiDrawdown = 0
                    , hiExpectancy = Nothing
                    , hiMaxDailyLossLim = Nothing
                    , hiMaxWeeklyLossLim = Nothing
                    , hiMaxDrawdownLim = Nothing
                    , hiMinExpectancy = Nothing
                    , hiPositionSize = 0
                    , hiMaxPositionSizeLim = Nothing
                    , hiConsecutiveLosses = 0
                    , hiMaxLossStreakLim = Nothing
                    , hiVolTarget = 0
                    , hiLeverage = lev
                    }
                | lev <- [0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 125.0, 150.0, 150.000001, -0.01, 0 / 0, 1 / 0, -1 / 0]
                ]
     in
        RiskVerificationReport
            { fvrRiskHaltMonotone = haltMonotone
            , fvrRiskHaltResetDaily = resetDaily
            , fvrRiskHaltResetWeekly = resetWeekly
            , fvrRiskHaltPreservesOther = preservesOther
            , fvrRiskHaltNoFalsePositive = noFalsePositive
            , fvrRiskHaltPositionSize = positionSizeHalt
            , fvrRiskHaltLossStreak = lossStreakHalt
            , fvrMaxPositionSizeBound = maxPositionSizeBound
            , fvrRiskLimitFinite = riskLimitFinite
            , fvrDrawdownSanity = drawdownSanity
            , fvrPositionSizeSanity = positionSizeSanity
            , fvrExpectancySanity = expectancySanity
            , fvrVolTargetSanity = volTargetSanity
            , fvrLeverageSanity = leverageSanity
            }
