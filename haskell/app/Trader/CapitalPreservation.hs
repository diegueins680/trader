module Trader.CapitalPreservation (
    CapitalPreservationConfig (..),
    CapitalPreservationReport (..),
    capitalPreservationIsEntryOnlyReason,
    capitalPreservationReport,
    defaultCapitalPreservationConfig,
) where

data CapitalPreservationConfig = CapitalPreservationConfig
    { cpcEnabled :: !Bool
    , cpcLookback :: !Int
    , cpcMinTrades :: !Int
    , cpcMaxDrawdown :: !(Maybe Double)
    , cpcMaxRollingLoss :: !(Maybe Double)
    , cpcMinSharpe :: !(Maybe Double)
    , cpcLossStreakMax :: !Int
    }
    deriving (Eq, Show)

data CapitalPreservationReport = CapitalPreservationReport
    { cprEnabled :: !Bool
    , cprLookback :: !Int
    , cprTrades :: !Int
    , cprRollingReturn :: !(Maybe Double)
    , cprRollingLoss :: !(Maybe Double)
    , cprRollingSharpe :: !(Maybe Double)
    , cprReason :: !(Maybe String)
    }
    deriving (Eq, Show)

defaultCapitalPreservationConfig :: CapitalPreservationConfig
defaultCapitalPreservationConfig =
    CapitalPreservationConfig
        { cpcEnabled = True
        , cpcLookback = 20
        , cpcMinTrades = 6
        , cpcMaxDrawdown = Just 0.10
        , cpcMaxRollingLoss = Just 0.05
        , cpcMinSharpe = Just 0
        , cpcLossStreakMax = 3
        }

capitalPreservationIsEntryOnlyReason :: String -> Bool
capitalPreservationIsEntryOnlyReason reason =
    take (length prefix) reason == prefix
  where
    prefix = "CAPITAL_PRESERVATION_"

capitalPreservationReport :: CapitalPreservationConfig -> Double -> Int -> [Double] -> CapitalPreservationReport
capitalPreservationReport cfg drawdown lossStreak returns0 =
    let lookback = max 0 (cpcLookback cfg)
        recent = take lookback (reverse (filter finiteDouble returns0))
        trades = length recent
        minTrades =
            if lookback <= 0
                then max 1 (cpcMinTrades cfg)
                else max 1 (min lookback (cpcMinTrades cfg))
        ready = cpcEnabled cfg && lookback > 0 && trades >= minTrades
        rollingReturn =
            if trades > 0
                then Just (sum recent)
                else Nothing
        rollingLoss = max 0 . negate <$> rollingReturn
        rollingSharpe =
            if ready
                then tradeSharpe recent
                else Nothing
        reason
            | not (cpcEnabled cfg) = Nothing
            | not (finiteDouble drawdown) = Just "CAPITAL_PRESERVATION_METRICS"
            | maybe False (drawdown >=) (cpcMaxDrawdown cfg) = Just "CAPITAL_PRESERVATION_DRAWDOWN"
            | cpcLossStreakMax cfg > 0 && lossStreak >= cpcLossStreakMax cfg = Just "CAPITAL_PRESERVATION_LOSS_STREAK"
            | not ready = Nothing
            | maybe False (\lim -> maybe False (>= lim) rollingLoss) (cpcMaxRollingLoss cfg) = Just "CAPITAL_PRESERVATION_ROLLING_LOSS"
            | maybe False (\lim -> maybe False (< lim) rollingSharpe) (cpcMinSharpe cfg) = Just "CAPITAL_PRESERVATION_SHARPE"
            | otherwise = Nothing
     in CapitalPreservationReport
            { cprEnabled = cpcEnabled cfg
            , cprLookback = lookback
            , cprTrades = trades
            , cprRollingReturn = rollingReturn
            , cprRollingLoss = rollingLoss
            , cprRollingSharpe = rollingSharpe
            , cprReason = reason
            }

tradeSharpe :: [Double] -> Maybe Double
tradeSharpe xs
    | n < 2 = Nothing
    | not (finiteDouble avg) = Nothing
    | std <= 1e-12 = Just (if avg >= 0 then 999 else -999)
    | otherwise = Just (avg / std)
  where
    n = length xs
    avg = sum xs / fromIntegral n
    variance =
        sum [(x - avg) * (x - avg) | x <- xs]
            / fromIntegral (max 1 (n - 1))
    std = sqrt variance

finiteDouble :: Double -> Bool
finiteDouble x = not (isNaN x || isInfinite x)
