module Trader.MarketContext
  ( MarketModel(..)
  , buildMarketModel
  , marketMeasurementAt
  , fitLinearRange
  ) where

import Control.Concurrent (threadDelay)
import Control.Exception (SomeException, try)
import Data.List (foldl')
import Data.Maybe (catMaybes)
import qualified Data.Vector as V

import Trader.App.Args (Args(..))
import Trader.Binance (BinanceEnv, fetchCloses, fetchTopSymbolsByQuoteVolume)
import Trader.Symbol (splitSymbol)

data MarketModel = MarketModel
  { mmSymbols :: ![String]
  , mmIntercept :: !Double
  , mmBeta :: !Double
  , mmVar :: !Double
  , mmLag :: !(V.Vector Double)
  } deriving (Eq, Show)

isBad :: Double -> Bool
isBad x = isNaN x || isInfinite x

toUpperAscii :: Char -> Char
toUpperAscii c =
  if 'a' <= c && c <= 'z'
    then toEnum (fromEnum c - 32)
    else c

safeReturn :: Double -> Double -> Double
safeReturn p0 p1
  | p0 == 0 = 0
  | otherwise =
      let r = p1 / p0 - 1
       in if isBad r then 0 else r

returnsFromCloses :: V.Vector Double -> V.Vector Double
returnsFromCloses closes =
  let n = V.length closes
   in V.generate n $ \i ->
        if i <= 0
          then 0
          else safeReturn (closes V.! (i - 1)) (closes V.! i)

forwardReturnsV :: V.Vector Double -> V.Vector Double
forwardReturnsV prices =
  let n = V.length prices
   in V.generate n $ \i ->
        if i >= n - 1
          then 0
          else safeReturn (prices V.! i) (prices V.! (i + 1))

takeLast :: Int -> [a] -> [a]
takeLast n xs
  | n <= 0 = xs
  | otherwise =
      let k = length xs - n
       in if k <= 0 then xs else drop k xs

weightedMarketLag :: Int -> [(Double, V.Vector Double)] -> V.Vector Double
weightedMarketLag n xs =
  let xs' = [(w, r) | (w, r) <- xs, w > 0, V.length r == n]
      totalW = sum (map fst xs')
      norm = if totalW <= 0 then 1 else totalW
   in V.generate n $ \i ->
        if i <= 0
          then 0
          else
            foldl'
              (\acc (w, r) ->
                 let v = r V.! i
                  in if isBad v then acc else acc + (w / norm) * v)
              0
              xs'

-- | Fits y = a + b x over indices [start..endExclusive-1], skipping NaN/Inf values.
-- Returns (a, b, residualVariance).
fitLinearRange :: V.Vector Double -> V.Vector Double -> Int -> Int -> (Double, Double, Double)
fitLinearRange xs ys start endExclusive =
  let n = min (V.length xs) (V.length ys)
      start' = max 0 start
      end' = min n endExclusive
      pairs =
        [ (x, y)
        | i <- [start' .. end' - 1]
        , let x = xs V.! i
        , let y = ys V.! i
        , not (isBad x || isBad y)
        ]
      m = length pairs
      meanD zs =
        case zs of
          [] -> 0
          _ -> sum zs / fromIntegral (length zs)
      meanX = meanD (map fst pairs)
      meanY = meanD (map snd pairs)
      eps = 1e-12
      (sxx, sxy) =
        foldl'
          (\(ax, axy) (x, y) ->
             let dx = x - meanX
                 dy = y - meanY
              in (ax + dx * dx, axy + dx * dy))
          (0, 0)
          pairs
      beta =
        if sxx <= eps
          then 0
          else sxy / sxx
      intercept = meanY - beta * meanX
      sse =
        foldl'
          (\acc (x, y) ->
             let e = y - (intercept + beta * x)
              in if isBad e then acc else acc + e * e)
          0
          pairs
      denom = fromIntegral (max 1 (m - 2))
      var = if m <= 0 then 0 else sse / denom
   in (intercept, beta, max eps var)

-- | Builds a market-context measurement for the Kalman filter using the top-N symbols by quote volume.
-- The returned model predicts the target forward return using the previous-bar, volume-weighted market return.
-- 'fitEnd' is the exclusive end of the training window in price indices (avoids lookahead in backtests).
buildMarketModel :: Args -> BinanceEnv -> String -> Int -> V.Vector Double -> IO (Maybe MarketModel)
buildMarketModel args env targetSymbol fitEnd pricesV = do
  let topN = max 0 (argKalmanMarketTopN args)
      n = V.length pricesV
      measVarFloor = max 1e-12 (argKalmanMeasurementVar args)
  if topN <= 0 || n < 3 || fitEnd < 3
    then pure Nothing
    else do
      let (_base, quote) = splitSymbol targetSymbol
      ranked <- fetchTopSymbolsByQuoteVolume env quote (topN + 5)
      let targetU = map toUpperAscii targetSymbol
          ranked' = filter (\(s, _w) -> map toUpperAscii s /= targetU) ranked
          sleepUs = 60 * 1000

      fetched <-
        fmap catMaybes $
          mapM
            (\(sym, w) -> do
               r <- try (fetchCloses env sym (argInterval args) n) :: IO (Either SomeException [Double])
               threadDelay sleepUs
               case r of
                 Left _ -> pure Nothing
                 Right closes
                   | length closes < n -> pure Nothing
                   | otherwise -> do
                       let v = V.fromList (takeLast n closes)
                           rets = returnsFromCloses v
                       pure (Just (sym, max 0 w, rets)))
            (take topN ranked')

      let usedSyms = [s | (s, _w, _r) <- fetched]
          wrets = [(w, r) | (_s, w, r) <- fetched]

      if length wrets < 5
        then pure Nothing
        else do
          let lag = weightedMarketLag n wrets
              ys = forwardReturnsV pricesV
              (a, b, var0) = fitLinearRange lag ys 1 (min (fitEnd - 1) (n - 1))
              var = max measVarFloor var0
          pure (Just MarketModel { mmSymbols = usedSyms, mmIntercept = a, mmBeta = b, mmVar = var, mmLag = lag })

marketMeasurementAt :: MarketModel -> Int -> Maybe (Double, Double)
marketMeasurementAt m t =
  if t <= 0 || t >= V.length (mmLag m)
    then Nothing
    else
      let x = mmLag m V.! t
          mu0 = mmIntercept m + mmBeta m * x
          mu = if isBad mu0 then 0 else mu0
          var = max 1e-12 (mmVar m)
       in Just (mu, var)

