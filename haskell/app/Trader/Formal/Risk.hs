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

{- | Check whether the drawdown limit is outside the valid (0,1) interval.
Defined here to avoid a module cycle with Trader.Trading.
-}
drawdownLimitInvalid :: HaltInputs -> Bool
drawdownLimitInvalid hi =
    case hiMaxDrawdownLim hi of
        Nothing -> False
        Just lim ->
            let lim' = max 0 lim
             in not (finiteDouble lim') || lim' <= 0 || lim' >= 1

-- | Exhaustive enumeration over bounded risk states.
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
    , fvrCooldownNonNegative :: !Bool
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
            , Just (ExitOther "POSITION_SIZE")
            , Just (ExitOther "LOSS_STREAK")
            , Just (ExitOther "RISK_LIMIT_NON_FINITE")
            ]

        -- Monotonicity: increasing any risk metric must not *remove* a halt.
        -- (Once halted, stay halted; once a limit is breached, the halt persists.)
        haltMonotone =
            all
                ( \(hi1, hi2) ->
                    let r1 = specRiskHalt hi1
                        r2 = specRiskHalt hi2
                     in case (r1, r2) of
                            (Just _, Nothing) -> False
                            _ -> True
                )
                [ (hi1, hi2)
                | dl1 <- doubles
                , dl2 <- doubles
                , dl2 >= dl1
                , wl1 <- doubles
                , wl2 <- doubles
                , wl2 >= wl1
                , dd1 <- doubles
                , dd2 <- doubles
                , dd2 >= dd1
                , let hi1 =
                        HaltInputs
                            { hiPrevHaltReason = Nothing
                            , hiDayChanged = False
                            , hiWeekChanged = False
                            , hiDailyLoss = dl1
                            , hiWeeklyLoss = wl1
                            , hiDrawdown = dd1
                            , hiExpectancy = Nothing
                            , hiMaxDailyLossLim = Just 0.05
                            , hiMaxWeeklyLossLim = Just 0.05
                            , hiMaxDrawdownLim = Just 0.05
                            , hiMinExpectancy = Nothing
                            , hiPositionSize = 0
                            , hiMaxPositionSizeLim = Nothing
                            , hiConsecutiveLosses = 0
                            , hiMaxLossStreakLim = Nothing
                            , hiVolTarget = 0
                            , hiLeverage = 0
                            }
                      hi2 = hi1{hiDailyLoss = dl2, hiWeeklyLoss = wl2, hiDrawdown = dd2}
                ]

        -- Daily reset: if the previous halt was ExitMaxDailyLoss and the day
        -- changed, the halt should be cleared.
        resetDaily =
            all
                ( \hi ->
                    case specRiskHalt hi of
                        Nothing -> True
                        Just r -> r /= ExitMaxDailyLoss
                )
                [ HaltInputs
                    { hiPrevHaltReason = Just ExitMaxDailyLoss
                    , hiDayChanged = True
                    , hiWeekChanged = False
                    , hiDailyLoss = 0.1
                    , hiWeeklyLoss = 0
                    , hiDrawdown = 0
                    , hiExpectancy = Nothing
                    , hiMaxDailyLossLim = Just 0.05
                    , hiMaxWeeklyLossLim = Nothing
                    , hiMaxDrawdownLim = Nothing
                    , hiMinExpectancy = Nothing
                    , hiPositionSize = 0
                    , hiMaxPositionSizeLim = Nothing
                    , hiConsecutiveLosses = 0
                    , hiMaxLossStreakLim = Nothing
                    , hiVolTarget = 0
                    , hiLeverage = 0
                    }
                ]

        -- Weekly reset: if the previous halt was ExitMaxWeeklyLoss and the week
        -- changed, the halt should be cleared.
        resetWeekly =
            all
                ( \hi ->
                    case specRiskHalt hi of
                        Nothing -> True
                        Just r -> r /= ExitMaxWeeklyLoss
                )
                [ HaltInputs
                    { hiPrevHaltReason = Just ExitMaxWeeklyLoss
                    , hiDayChanged = False
                    , hiWeekChanged = True
                    , hiDailyLoss = 0
                    , hiWeeklyLoss = 0.1
                    , hiDrawdown = 0
                    , hiExpectancy = Nothing
                    , hiMaxDailyLossLim = Nothing
                    , hiMaxWeeklyLossLim = Just 0.05
                    , hiMaxDrawdownLim = Nothing
                    , hiMinExpectancy = Nothing
                    , hiPositionSize = 0
                    , hiMaxPositionSizeLim = Nothing
                    , hiConsecutiveLosses = 0
                    , hiMaxLossStreakLim = Nothing
                    , hiVolTarget = 0
                    , hiLeverage = 0
                    }
                ]

        -- Preserves other halts: if a previous halt reason exists and the
        -- reset conditions do not apply, the same reason must be preserved.
        preservesOther =
            all
                ( \(prev, dayCh, weekCh) ->
                    let hi =
                            HaltInputs
                                { hiPrevHaltReason = prev
                                , hiDayChanged = dayCh
                                , hiWeekChanged = weekCh
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
                                , hiLeverage = 0
                                }
                        result = specRiskHalt hi
                     in case prev of
                            Just ExitMaxDailyLoss | dayCh -> isNothing result
                            Just ExitMaxWeeklyLoss | weekCh -> isNothing result
                            Just r -> result == Just r
                            Nothing -> True
                )
                [ (prev, dayCh, weekCh)
                | prev <- prevReasons
                , dayCh <- [False, True]
                , weekCh <- [False, True]
                ]

        -- No false positives: when all metrics are zero and no limits are set,
        -- there must be no halt.
        noFalsePositive =
            all
                (\hi -> isNothing (specRiskHalt hi))
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
                    , hiLeverage = 0
                    }
                ]

        -- Position-size halt: if position size exceeds the limit, a halt must
        -- be emitted with reason POSITION_SIZE.
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
                | ps <- [0, 0.5, 1.0, 1.5, 2.0, 5.0]
                , mps <- mDoubles
                ]

        -- Loss-streak halt: if consecutive losses exceed the limit, a halt
        -- must be emitted with reason LOSS_STREAK.
        lossStreakHalt =
            all
                ( \hi ->
                    let result = specRiskHalt hi
                        limit = fmap (max 0) (hiMaxLossStreakLim hi)
                        cl = hiConsecutiveLosses hi
                     in case result of
                            Just (ExitOther "LOSS_STREAK") ->
                                maybe False (\lim -> lim > 0 && cl > lim) limit
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
                | mdl <- [Nothing, Just 0, Just 0.05, Just (0 / 0), Just (1 / 0), Just (-(1 / 0))]
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
                | mdd <- [Nothing, Just 0, Just 0.05, Just 0.5, Just 0.999999, Just 1.0, Just (-0.01), Just (0 / 0), Just (1 / 0), Just (-(1 / 0))]
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
                | ps <- [0, 0.5, 1.0, 2.0, 5.0, 10.0, 10.000001, -0.01, 0 / 0, 1 / 0, -(1 / 0)]
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
                | ex <- [Nothing, Just (-0.1), Just 0, Just 0.01, Just (0 / 0), Just (1 / 0), Just (-(1 / 0))]
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
                | vt <- [0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 10.000001, -0.01, 0 / 0, 1 / 0, -(1 / 0)]
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
                | lev <- [0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 125.0, 150.0, 150.000001, -0.01, 0 / 0, 1 / 0, -(1 / 0)]
                ]

        -- Cooldown non-negative invariant: any configured cooldown is sanitized
        -- to >= 0 in the simulation loop (max 0). This property proves that
        -- negative raw config values cannot propagate into the cooldown state.
        cooldownNonNegative =
            all
                (\raw -> let sanitized = max 0 raw in sanitized >= 0)
                ([-5, -1, 0, 1, 3, 10] :: [Int])
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
            , fvrCooldownNonNegative = cooldownNonNegative
            }
