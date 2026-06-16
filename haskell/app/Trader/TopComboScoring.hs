module Trader.TopComboScoring (
    TopComboScoringConfig (..),
    defaultTopComboScoringConfig,
    validateTopComboScoringConfig,
    clampLiveAnnualizedReturnWithConfig,
    topComboDrawdownMultiplier,
    topComboEquityTerm,
    topComboLiveBlendWeight,
    topComboLiveQuarantinedByConfig,
    topComboValidatedAnnualizedReturn,
    topComboWalkForwardMultiplier,
) where

data TopComboScoringConfig = TopComboScoringConfig
    { tcscLiveBlendShrinkageOps :: !Double
    , tcscLiveQuarantineMinOperations :: !Int
    , tcscLiveQuarantineMaxFinalEquity :: !Double
    , tcscLiveAnnualizedReturnFloor :: !Double
    , tcscLiveAnnualizedReturnCeiling :: !Double
    , tcscValidatedAnnualizedReturnCap :: !Double
    , tcscDeployableWalkForwardMultiplier :: !Double
    , tcscBelowFloorWalkForwardMultiplier :: !Double
    , tcscMissingWalkForwardMultiplier :: !Double
    , tcscDrawdownPenaltyScale :: !Double
    , tcscEquityFloor :: !Double
    , tcscEquityLogFloor :: !Double
    }
    deriving (Eq, Show)

defaultTopComboScoringConfig :: TopComboScoringConfig
defaultTopComboScoringConfig =
    TopComboScoringConfig
        { tcscLiveBlendShrinkageOps = 25
        , tcscLiveQuarantineMinOperations = 30
        , tcscLiveQuarantineMaxFinalEquity = 0.99
        , tcscLiveAnnualizedReturnFloor = -0.9999
        , tcscLiveAnnualizedReturnCeiling = 10
        , tcscValidatedAnnualizedReturnCap = 20
        , tcscDeployableWalkForwardMultiplier = 1.0
        , tcscBelowFloorWalkForwardMultiplier = 0.35
        , tcscMissingWalkForwardMultiplier = 0.60
        , tcscDrawdownPenaltyScale = 10
        , tcscEquityFloor = 1.0e-9
        , tcscEquityLogFloor = -1
        }

validateTopComboScoringConfig :: TopComboScoringConfig -> Either String TopComboScoringConfig
validateTopComboScoringConfig config = do
    finiteNonNegative "--score-live-blend-shrinkage-ops" (tcscLiveBlendShrinkageOps config)
    ensure "--score-live-quarantine-min-operations must be >= 0" (tcscLiveQuarantineMinOperations config >= 0)
    finiteNonNegative "--score-live-quarantine-max-final-equity" (tcscLiveQuarantineMaxFinalEquity config)
    finite "--score-live-annualized-return-floor" (tcscLiveAnnualizedReturnFloor config)
    finite "--score-live-annualized-return-ceiling" (tcscLiveAnnualizedReturnCeiling config)
    ensure "--score-live-annualized-return-ceiling must be >= --score-live-annualized-return-floor" $
        tcscLiveAnnualizedReturnCeiling config >= tcscLiveAnnualizedReturnFloor config
    finiteNonNegative "--score-validated-annualized-return-cap" (tcscValidatedAnnualizedReturnCap config)
    finiteNonNegative "--score-wf-deployable-multiplier" (tcscDeployableWalkForwardMultiplier config)
    finiteNonNegative "--score-wf-below-floor-multiplier" (tcscBelowFloorWalkForwardMultiplier config)
    finiteNonNegative "--score-wf-missing-multiplier" (tcscMissingWalkForwardMultiplier config)
    finiteNonNegative "--score-drawdown-penalty-scale" (tcscDrawdownPenaltyScale config)
    finitePositive "--score-equity-floor" (tcscEquityFloor config)
    finite "--score-equity-log-floor" (tcscEquityLogFloor config)
    pure config
  where
    ensure msg ok =
        if ok then Right () else Left msg
    finite name value =
        ensure (name ++ " must be finite") (isFinite value)
    finiteNonNegative name value = do
        finite name value
        ensure (name ++ " must be >= 0") (value >= 0)
    finitePositive name value = do
        finite name value
        ensure (name ++ " must be > 0") (value > 0)

isFinite :: Double -> Bool
isFinite value = not (isNaN value || isInfinite value)

clampLiveAnnualizedReturnWithConfig :: TopComboScoringConfig -> Double -> Double
clampLiveAnnualizedReturnWithConfig config ann =
    max
        (tcscLiveAnnualizedReturnFloor config)
        (min (tcscLiveAnnualizedReturnCeiling config) ann)

topComboLiveBlendWeight :: TopComboScoringConfig -> Int -> Double
topComboLiveBlendWeight config operationCount
    | operationCount <= 0 = 0
    | denominator <= 0 = 1
    | otherwise = n / denominator
  where
    n = fromIntegral operationCount
    denominator = n + max 0 (tcscLiveBlendShrinkageOps config)

topComboLiveQuarantinedByConfig :: TopComboScoringConfig -> Int -> Double -> Bool
topComboLiveQuarantinedByConfig config operationCount finalEquity =
    operationCount >= tcscLiveQuarantineMinOperations config
        && finalEquity <= tcscLiveQuarantineMaxFinalEquity config

topComboValidatedAnnualizedReturn :: TopComboScoringConfig -> Double -> Double
topComboValidatedAnnualizedReturn config ann =
    min (tcscValidatedAnnualizedReturnCap config) (max 0 ann)

topComboWalkForwardMultiplier :: TopComboScoringConfig -> Maybe Bool -> Double
topComboWalkForwardMultiplier config verdict =
    case verdict of
        Just True -> tcscDeployableWalkForwardMultiplier config
        Just False -> tcscBelowFloorWalkForwardMultiplier config
        Nothing -> tcscMissingWalkForwardMultiplier config

topComboDrawdownMultiplier :: TopComboScoringConfig -> Double -> Double
topComboDrawdownMultiplier config drawdown =
    1 / (1 + tcscDrawdownPenaltyScale config * max 0 drawdown)

topComboEquityTerm :: TopComboScoringConfig -> Double -> Double
topComboEquityTerm config finalEquity =
    max (tcscEquityLogFloor config) (log (max (tcscEquityFloor config) finalEquity))
