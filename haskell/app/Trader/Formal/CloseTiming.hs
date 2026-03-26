{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}

module Trader.Formal.CloseTiming (
    ComboTrade (..),
    ComboTimingStats (..),
    TradeTimingSample (..),
    timingSamples,
    summarizeComboTiming,
    summarizeAllCombos
) where

import Data.Aeson ((.:), (.:?), (.!=), (.=))
import qualified Data.Aeson as Aeson
import Data.Function (on)
import Data.List (groupBy, sort, sortOn)
import qualified Data.Text as T
import qualified Data.Vector as V
import GHC.Generics (Generic)

-- | Trade tagged with combo identity and bar indices.
data ComboTrade = ComboTrade
    { ctComboId :: !String
    , ctEntryIndex :: !Int
    , ctExitIndex :: !Int
    , ctEntryPrice :: !Double
    , ctSide :: !Double -- +1 long, -1 short
    }
    deriving (Eq, Show, Generic)

-- | Per-trade sample with the optimal close bar tm in [ta, ta + 2*(tc-ta)].
data TradeTimingSample = TradeTimingSample
    { ttsComboId :: !String
    , ttsEntryIndex :: !Int
    , ttsExitIndex :: !Int
    , ttsOptimalIndex :: !Int
    , ttsObservedDuration :: !Int
    , ttsOptimalDuration :: !Int
    , ttsObservedReturn :: !Double
    , ttsOptimalReturn :: !Double
    , ttsReturnLift :: !Double
    }
    deriving (Eq, Show, Generic)

-- | Robust summary for tm distribution per combo.
data ComboTimingStats = ComboTimingStats
    { ctsComboId :: !String
    , ctsSampleCount :: !Int
    , ctsMedianRatio :: !Double
    , ctsQ25Ratio :: !Double
    , ctsQ75Ratio :: !Double
    , ctsMadRatio :: !Double
    , ctsMeanLift :: !Double
    , ctsMedianLift :: !Double
    }
    deriving (Eq, Show, Generic)

-- | JSON instances for ComboTrade (input) and result types.
instance Aeson.FromJSON ComboTrade where
    parseJSON = Aeson.withObject "ComboTrade" $ \o -> do
        comboId <- fmap T.unpack (o .:? "comboId" .!= "unknown")
        ta <- o .: "entryIndex"
        tc <- o .: "exitIndex"
        entryPx <- o .:? "entryPrice" .!= 0
        side <- o .:? "side" .!= (1 :: Double)
        pure ComboTrade{ctComboId = comboId, ctEntryIndex = ta, ctExitIndex = tc, ctEntryPrice = entryPx, ctSide = side}

instance Aeson.ToJSON TradeTimingSample where
    toJSON s =
        Aeson.object
            [ "comboId" .= ttsComboId s
            , "entryIndex" .= ttsEntryIndex s
            , "exitIndex" .= ttsExitIndex s
            , "optimalIndex" .= ttsOptimalIndex s
            , "observedDuration" .= ttsObservedDuration s
            , "optimalDuration" .= ttsOptimalDuration s
            , "observedReturn" .= ttsObservedReturn s
            , "optimalReturn" .= ttsOptimalReturn s
            , "returnLift" .= ttsReturnLift s
            ]

instance Aeson.ToJSON ComboTimingStats where
    toJSON s =
        Aeson.object
            [ "comboId" .= ctsComboId s
            , "sampleCount" .= ctsSampleCount s
            , "medianRatio" .= ctsMedianRatio s
            , "q25Ratio" .= ctsQ25Ratio s
            , "q75Ratio" .= ctsQ75Ratio s
            , "madRatio" .= ctsMadRatio s
            , "meanLift" .= ctsMeanLift s
            , "medianLift" .= ctsMedianLift s
            ]

returnAt :: V.Vector Double -> ComboTrade -> Int -> Maybe Double
returnAt prices tr i = do
    px <- prices V.!? i
    pure (ctSide tr * (px / ctEntryPrice tr - 1))

optimalIndexInWindow :: V.Vector Double -> ComboTrade -> Int
optimalIndexInWindow prices tr =
    let ta = ctEntryIndex tr
        tc = ctExitIndex tr
     in if tc <= ta
            then ta
            else
                let obsDuration = tc - ta
                    maxI = min (V.length prices - 1) (ta + 2 * obsDuration)
                    candidates = [ta .. maxI]
                    scored = [(i, returnAt prices tr i) | i <- candidates]
                    valid = [(i, r) | (i, Just r) <- scored]
                 in case valid of
                        [] -> tc
                        _ -> fst (maximumBySnd valid)

maximumBySnd :: [(Int, Double)] -> (Int, Double)
maximumBySnd = foldl1 pick
  where
    pick a@(_, ra) b@(_, rb)
        | rb > ra = b
        | otherwise = a

sampleForTrade :: V.Vector Double -> ComboTrade -> Maybe TradeTimingSample
sampleForTrade prices tr = do
    let ta = ctEntryIndex tr
        tc = ctExitIndex tr
        tm = optimalIndexInWindow prices tr
    observed <- returnAt prices tr tc
    optimal <- returnAt prices tr tm
    let durObs = max 0 (tc - ta)
        durOpt = max 0 (tm - ta)
    pure
        TradeTimingSample
            { ttsComboId = ctComboId tr
            , ttsEntryIndex = ta
            , ttsExitIndex = tc
            , ttsOptimalIndex = tm
            , ttsObservedDuration = durObs
            , ttsOptimalDuration = durOpt
            , ttsObservedReturn = observed
            , ttsOptimalReturn = optimal
            , ttsReturnLift = optimal - observed
            }

timingSamples :: [Double] -> [ComboTrade] -> [TradeTimingSample]
timingSamples priceList trades =
    let prices = V.fromList priceList
     in foldr step [] trades
  where
    step tr acc = case sampleForTrade prices tr of
        Just s -> s : acc
        Nothing -> acc

quantile :: Double -> [Double] -> Double
quantile q xs
    | null xs = 0
    | otherwise =
        let ys = sort xs
            n = length ys
            idx = floor (q * fromIntegral (n - 1))
         in ys !! max 0 (min (n - 1) idx)

median :: [Double] -> Double
median = quantile 0.5

mean :: [Double] -> Double
mean xs =
    let n = length xs
     in if n == 0 then 0 else sum xs / fromIntegral n

mad :: [Double] -> Double
mad xs =
    let m = median xs
        deviations = map (abs . subtract m) xs
     in median deviations

summarizeComboTiming :: [TradeTimingSample] -> Maybe ComboTimingStats
summarizeComboTiming [] = Nothing
summarizeComboTiming ss@(s0 : _) =
    let ratios =
            [ if ttsObservedDuration s <= 0
                then 0
                else fromIntegral (ttsOptimalDuration s) / fromIntegral (ttsObservedDuration s)
            | s <- ss
            ]
        lifts = map ttsReturnLift ss
     in Just
            ComboTimingStats
                { ctsComboId = ttsComboId s0
                , ctsSampleCount = length ss
                , ctsMedianRatio = median ratios
                , ctsQ25Ratio = quantile 0.25 ratios
                , ctsQ75Ratio = quantile 0.75 ratios
                , ctsMadRatio = mad ratios
                , ctsMeanLift = mean lifts
                , ctsMedianLift = median lifts
                }

summarizeAllCombos :: [TradeTimingSample] -> [ComboTimingStats]
summarizeAllCombos ss =
    let grouped = groupBy ((==) `on` ttsComboId) (sortOn ttsComboId ss)
     in foldr step [] grouped
  where
    step grp acc = case summarizeComboTiming grp of
        Just st -> st : acc
        Nothing -> acc
