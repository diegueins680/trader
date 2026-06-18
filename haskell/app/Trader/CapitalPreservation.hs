module Trader.CapitalPreservation (
    CapitalPreservationConfig (..),
    CapitalPreservationReport (..),
    PortfolioCapitalPreservationConfig (..),
    PortfolioCapitalPreservationReport (..),
    PortfolioCapitalTrade (..),
    capitalPreservationIsEntryOnlyReason,
    capitalPreservationReport,
    defaultCapitalPreservationConfig,
    defaultPortfolioCapitalPreservationCooldownMs,
    defaultPortfolioCapitalPreservationConfig,
    portfolioCapitalPreservationReport,
) where

import Data.Int (Int64)
import Data.List (sortOn)

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

data PortfolioCapitalPreservationConfig = PortfolioCapitalPreservationConfig
    { pcpcEnabled :: !Bool
    , pcpcLookback :: !Int
    , pcpcMinTrades :: !Int
    , pcpcMaxRollingLoss :: !(Maybe Double)
    , pcpcLossStreakMax :: !Int
    , pcpcCooldownMs :: !Int64
    }
    deriving (Eq, Show)

data PortfolioCapitalTrade = PortfolioCapitalTrade
    { pctReturn :: !Double
    , pctClosedAtMs :: !(Maybe Int64)
    }
    deriving (Eq, Show)

data PortfolioCapitalPreservationReport = PortfolioCapitalPreservationReport
    { pcprEnabled :: !Bool
    , pcprLookback :: !Int
    , pcprTrades :: !Int
    , pcprLossStreak :: !Int
    , pcprRollingReturn :: !(Maybe Double)
    , pcprRollingLoss :: !(Maybe Double)
    , pcprNewestClosedAtMs :: !(Maybe Int64)
    , pcprOpenUntilMs :: !(Maybe Int64)
    , pcprReason :: !(Maybe String)
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

defaultPortfolioCapitalPreservationCooldownMs :: Int64
defaultPortfolioCapitalPreservationCooldownMs = 30 * 60 * 1000

defaultPortfolioCapitalPreservationConfig :: PortfolioCapitalPreservationConfig
defaultPortfolioCapitalPreservationConfig =
    PortfolioCapitalPreservationConfig
        { pcpcEnabled = True
        , pcpcLookback = cpcLookback defaultCapitalPreservationConfig
        , pcpcMinTrades = cpcMinTrades defaultCapitalPreservationConfig
        , pcpcMaxRollingLoss = cpcMaxRollingLoss defaultCapitalPreservationConfig
        , pcpcLossStreakMax = cpcLossStreakMax defaultCapitalPreservationConfig
        , pcpcCooldownMs = defaultPortfolioCapitalPreservationCooldownMs
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

portfolioCapitalPreservationReport ::
    PortfolioCapitalPreservationConfig ->
    Int64 ->
    [PortfolioCapitalTrade] ->
    PortfolioCapitalPreservationReport
portfolioCapitalPreservationReport cfg nowMs trades0 =
    let lookback = max 0 (pcpcLookback cfg)
        sortedTrades =
            map snd3 $
                sortOn
                    (\(closedAt, idx, _) -> (closedAt, idx))
                    [ (closedAt, idx, tr{pctClosedAtMs = Just closedAt})
                    | (idx, tr) <- zip [(0 :: Int) ..] trades0
                    , finiteDouble (pctReturn tr)
                    , Just closedAt <- [pctClosedAtMs tr]
                    ]
        snd3 (_, _, x) = x
        finiteTrades = sortedTrades
        recent = take lookback (reverse finiteTrades)
        returns = map pctReturn recent
        trades = length recent
        minTrades =
            if lookback <= 0
                then max 1 (pcpcMinTrades cfg)
                else max 1 (min lookback (pcpcMinTrades cfg))
        ready = pcpcEnabled cfg && lookback > 0 && trades >= minTrades
        lossStreak =
            length
                ( takeWhile
                    (\tr -> pctReturn tr <= 0)
                    (reverse finiteTrades)
                )
        rollingReturn =
            if trades > 0
                then Just (sum returns)
                else Nothing
        rollingLoss = max 0 . negate <$> rollingReturn
        triggerReason
            | not (pcpcEnabled cfg) = Nothing
            | pcpcLossStreakMax cfg > 0 && lossStreak >= pcpcLossStreakMax cfg = Just "CAPITAL_PRESERVATION_PORTFOLIO_LOSS_STREAK"
            | not ready = Nothing
            | maybe False (\lim -> maybe False (>= lim) rollingLoss) (pcpcMaxRollingLoss cfg) = Just "CAPITAL_PRESERVATION_PORTFOLIO_ROLLING_LOSS"
            | otherwise = Nothing
        newestClosedAtMs =
            case finiteTrades of
                [] -> Nothing
                _ -> pctClosedAtMs (last finiteTrades)
        cooldownMs = max 0 (pcpcCooldownMs cfg)
        openUntilMs = (+ cooldownMs) <$> newestClosedAtMs
        cooldownOpen =
            case openUntilMs of
                Just untilMs -> nowMs < untilMs
                Nothing -> False
        reason =
            case triggerReason of
                Just r | cooldownOpen -> Just r
                _ -> Nothing
     in PortfolioCapitalPreservationReport
            { pcprEnabled = pcpcEnabled cfg
            , pcprLookback = lookback
            , pcprTrades = trades
            , pcprLossStreak = lossStreak
            , pcprRollingReturn = rollingReturn
            , pcprRollingLoss = rollingLoss
            , pcprNewestClosedAtMs = newestClosedAtMs
            , pcprOpenUntilMs = openUntilMs
            , pcprReason = reason
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
