{- | Live-vs-backtest gap analysis: the research optimizer's report card.

The optimizer discovers combos by backtesting historical data; the fleet
then trades them and accumulates a realized @live@ record per combo (see
'Trader.TopCombosStore.ComboLiveStats'). The gap between realized and
backtested annualized return, aggregated per method family, measures how
much of each family's backtest edge survives contact with the market. A
family whose combos chronically deliver less than they backtested is
overfitting; one that holds up live deserves more of the discovery budget.

The output is a per-method sampling multiplier the auto-optimizer applies
to its discovery method weights. It is deliberately gentle: neutral (1.0)
until a family has real live evidence, clamped to a bounded range so the
feedback re-balances the search rather than slamming families on or off,
and ops-weighted so one heavily-traded combo counts for more than ten
barely-traded ones.
-}
module Trader.LiveGap (
    LiveGapConfig (..),
    LiveGapStats (..),
    comboLiveGapEntry,
    comboLiveGapEntryWithConfig,
    defaultLiveGapConfig,
    liveGapStatsByMethod,
    liveGapStatsByMethodWithConfig,
    liveGapMethodMultiplier,
    liveGapMethodMultiplierWithConfig,
    liveGapMinComboOperations,
    liveGapMinTotalOperations,
    liveGapMultiplierFloor,
    liveGapMultiplierCeiling,
    validateLiveGapConfig,
) where

import Control.Applicative ((<|>))
import qualified Data.Aeson as Aeson
import qualified Data.Aeson.Key as AK
import qualified Data.Aeson.KeyMap as KM
import Data.List (foldl')
import qualified Data.Map.Strict as M
import Data.Maybe (mapMaybe)
import qualified Data.Text as T

import Trader.TopCombosStore (
    ComboLiveStats (..),
    comboLiveStats,
    comboMetricDouble,
    comboMetricsDouble,
 )

data LiveGapConfig = LiveGapConfig
    { lgcMinComboOperations :: !Int
    , lgcMinTotalOperations :: !Int
    , lgcMultiplierFloor :: !Double
    , lgcMultiplierCeiling :: !Double
    }
    deriving (Eq, Show)

defaultLiveGapConfig :: LiveGapConfig
defaultLiveGapConfig =
    LiveGapConfig
        { lgcMinComboOperations = 10
        , lgcMinTotalOperations = 30
        , lgcMultiplierFloor = 0.25
        , lgcMultiplierCeiling = 1.5
        }

validateLiveGapConfig :: LiveGapConfig -> Either String LiveGapConfig
validateLiveGapConfig config = do
    ensure "live-gap min combo operations must be >= 0" (lgcMinComboOperations config >= 0)
    ensure "live-gap min total operations must be >= 0" (lgcMinTotalOperations config >= 0)
    finiteNonNegative "live-gap multiplier floor" (lgcMultiplierFloor config)
    finiteNonNegative "live-gap multiplier ceiling" (lgcMultiplierCeiling config)
    ensure "live-gap multiplier ceiling must be >= floor" (lgcMultiplierCeiling config >= lgcMultiplierFloor config)
    pure config
  where
    ensure msg ok =
        if ok then Right () else Left msg
    finiteNonNegative label value = do
        ensure (label ++ " must be finite") (not (isNaN value || isInfinite value))
        ensure (label ++ " must be >= 0") (value >= 0)

{- | Live orders a single combo needs before its gap is a measurement
rather than noise.
-}
liveGapMinComboOperations :: Int
liveGapMinComboOperations = lgcMinComboOperations defaultLiveGapConfig

{- | Total live orders a method family needs before its multiplier moves
off neutral.
-}
liveGapMinTotalOperations :: Int
liveGapMinTotalOperations = lgcMinTotalOperations defaultLiveGapConfig

{- | A chronically overfit family keeps at least a quarter of its discovery
weight: the feedback re-balances the search, it doesn't kill exploration.
-}
liveGapMultiplierFloor :: Double
liveGapMultiplierFloor = lgcMultiplierFloor defaultLiveGapConfig

{- | A family that outperforms its backtests live earns at most 1.5x its
base discovery weight.
-}
liveGapMultiplierCeiling :: Double
liveGapMultiplierCeiling = lgcMultiplierCeiling defaultLiveGapConfig

{- | Per-method-family aggregation of live-vs-backtest gaps.
'lgsOpsWeightedGap' is the operations-weighted mean of
@liveAnnualizedReturn - backtestAnnualizedReturn@ across the family's
combos (negative = the family delivers less live than it backtests).
-}
data LiveGapStats = LiveGapStats
    { lgsCombos :: !Int
    , lgsOperations :: !Int
    , lgsOpsWeightedGap :: !Double
    }
    deriving (Eq, Show)

{- | One combo's contribution: (method, live − backtest annualized return,
live operation count). Nothing when the combo has no method, no live
record with an annualized reading, or fewer than
'liveGapMinComboOperations' live orders.
-}
comboLiveGapEntry :: Aeson.Value -> Maybe (String, Double, Int)
comboLiveGapEntry = comboLiveGapEntryWithConfig defaultLiveGapConfig

comboLiveGapEntryWithConfig :: LiveGapConfig -> Aeson.Value -> Maybe (String, Double, Int)
comboLiveGapEntryWithConfig config val = do
    method <- comboMethodString val
    stats <- comboLiveStats val
    liveAnn <- clsAnnualizedReturn stats
    let ops = clsOperationCount stats
    backtestAnn <-
        comboMetricsDouble "annualizedReturn" val
            <|> comboMetricDouble "annualizedReturn" val
    let gap = liveAnn - backtestAnn
    if ops < lgcMinComboOperations config || isNaN gap || isInfinite gap
        then Nothing
        else Just (method, gap, ops)

comboMethodString :: Aeson.Value -> Maybe String
comboMethodString val = do
    paramsVal <- case val of
        Aeson.Object o -> KM.lookup (AK.fromString "params") o
        _ -> Nothing
    paramsObj <- case paramsVal of
        Aeson.Object p -> Just p
        _ -> Nothing
    methodVal <- KM.lookup (AK.fromString "method") paramsObj
    case methodVal of
        Aeson.String s ->
            let trimmed = T.unpack (T.strip s)
             in if null trimmed then Nothing else Just trimmed
        _ -> Nothing

-- | Aggregate gap evidence per method family from top-combos JSON values.
liveGapStatsByMethod :: [Aeson.Value] -> M.Map String LiveGapStats
liveGapStatsByMethod = liveGapStatsByMethodWithConfig defaultLiveGapConfig

liveGapStatsByMethodWithConfig :: LiveGapConfig -> [Aeson.Value] -> M.Map String LiveGapStats
liveGapStatsByMethodWithConfig config combos =
    M.map finalize (foldl' step M.empty (mapMaybe (comboLiveGapEntryWithConfig config) combos))
  where
    -- During accumulation lgsOpsWeightedGap holds the weighted SUM;
    -- 'finalize' divides it into the weighted mean.
    step acc (method, gap, ops) =
        M.insertWith
            merge
            method
            LiveGapStats{lgsCombos = 1, lgsOperations = ops, lgsOpsWeightedGap = gap * fromIntegral ops}
            acc
    merge new old =
        LiveGapStats
            { lgsCombos = lgsCombos old + lgsCombos new
            , lgsOperations = lgsOperations old + lgsOperations new
            , lgsOpsWeightedGap = lgsOpsWeightedGap old + lgsOpsWeightedGap new
            }
    finalize s =
        s
            { lgsOpsWeightedGap =
                if lgsOperations s > 0
                    then lgsOpsWeightedGap s / fromIntegral (lgsOperations s)
                    else 0
            }

{- | Discovery-weight multiplier for a method family. Neutral without
evidence ('liveGapMinTotalOperations' live orders across the family), then
@1 + gap@ clamped to ['liveGapMultiplierFloor', 'liveGapMultiplierCeiling']
— a family losing 50 points of annualized return versus its backtests
samples at half weight; beyond −75 points it bottoms out at the floor.
-}
liveGapMethodMultiplier :: Maybe LiveGapStats -> Double
liveGapMethodMultiplier = liveGapMethodMultiplierWithConfig defaultLiveGapConfig

liveGapMethodMultiplierWithConfig :: LiveGapConfig -> Maybe LiveGapStats -> Double
liveGapMethodMultiplierWithConfig config mStats =
    case mStats of
        Nothing -> 1
        Just stats
            | lgsOperations stats < lgcMinTotalOperations config -> 1
            | isNaN (lgsOpsWeightedGap stats) || isInfinite (lgsOpsWeightedGap stats) -> 1
            | otherwise ->
                max
                    (lgcMultiplierFloor config)
                    (min (lgcMultiplierCeiling config) (1 + lgsOpsWeightedGap stats))
