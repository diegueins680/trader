{- | Bridge that assembles the Tier-1 Binance derivatives features for one
symbol: fetch each series, align it point-in-time onto the bar grid (via
'Trader.Predictors.Exogenous.alignToBars'), and attach the result to a
'FeatureInputs'. This is the single call a data-loading path needs to opt a
symbol into exogenous features.

Network/IO is isolated here so 'Trader.Predictors.Exogenous' stays pure and
unit-testable. Failures degrade gracefully: any series that cannot be fetched is
left unset (its features stay neutral 0), so this never blocks model
construction or trading.
-}
module Trader.Predictors.ExogenousFetch (
    fetchExogenousInputs,
) where

import Control.Exception (SomeException, try)
import Data.Int (Int64)
import qualified Data.Vector as V

import Trader.Binance (
    BinanceEnv,
    fetchBasisHistory,
    fetchFundingRateHistory,
    fetchOpenInterestHist,
    fetchTakerLongShortRatio,
 )
import Trader.Predictors.Exogenous (alignedFeatureSeries)
import Trader.Predictors.Features (FeatureInputs, withExogenousInputs)

{- | Fetch funding / open-interest / taker-flow / basis for @symbol@, align each
onto @barOpenTimes@ (ascending closed-bar open times, @intervalMs@ apart), and
attach them to @inputs@. @period@ is the stats granularity for the
@\/futures\/data@ endpoints (e.g. @"1d"@, @"4h"@); funding uses its own
schedule and is aligned by timestamp regardless.
-}
fetchExogenousInputs ::
    BinanceEnv ->
    String -> -- symbol, e.g. "BTCUSDT"
    String -> -- stats period for OI/taker/basis
    V.Vector Int64 -> -- bar open times (ascending, closed bars)
    Int64 -> -- bar interval (ms)
    FeatureInputs ->
    IO FeatureInputs
fetchExogenousInputs env symbol period barOpenTimes intervalMs inputs = do
    let n = max 1 (V.length barOpenTimes)
        tryFetch io =
            either (const []) id <$> (try io :: IO (Either SomeException [(Int64, Double)]))
        align rows = alignedFeatureSeries barOpenTimes intervalMs rows
    funding <- align <$> tryFetch (fetchFundingRateHistory env symbol n)
    oi <- align <$> tryFetch (fetchOpenInterestHist env symbol period n)
    taker <- align <$> tryFetch (fetchTakerLongShortRatio env symbol period n)
    basis <- align <$> tryFetch (fetchBasisHistory env symbol period n)
    pure (withExogenousInputs funding oi basis taker inputs)
