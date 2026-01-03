module Trader.Duration
  ( parseDurationSeconds
  , parseIntervalSeconds
  , inferPeriodsPerYear
  , lookbackBarsFrom
  ) where

import Data.Char (isDigit, toLower, isSpace)

trim :: String -> String
trim = dropWhileEnd isSpace . dropWhile isSpace

dropWhileEnd :: (a -> Bool) -> [a] -> [a]
dropWhileEnd p = reverse . dropWhile p . reverse

-- | Parse a human duration like \"24h\", \"15m\", \"1d\" into seconds.
-- Supports units: s, m, h, d, w, M (month ~= 30d).
parseDurationSeconds :: String -> Maybe Int
parseDurationSeconds raw =
  let s = trim raw
   in case span isDigit s of
        ("", _) -> Nothing
        (nStr, unitStr) ->
          case unitStr of
            [u] -> do
              n <- readInt nStr
              mult <- unitSeconds u
              pure (n * mult)
            _ -> Nothing

-- | Parse a Binance-style interval like \"5m\", \"1h\", \"1d\" into seconds.
-- This is identical to 'parseDurationSeconds' but kept separate for clarity.
parseIntervalSeconds :: String -> Maybe Int
parseIntervalSeconds = parseDurationSeconds

lookbackBarsFrom :: String -> String -> Either String Int
lookbackBarsFrom intervalStr lookbackWindowStr = do
  intervalSec <-
    maybe
      (Left ("Invalid interval: " ++ show intervalStr ++ " (expected like 5m, 1h, 1d)"))
      Right
      (parseIntervalSeconds intervalStr)
  lookbackSec <-
    maybe
      (Left ("Invalid lookback window: " ++ show lookbackWindowStr ++ " (expected like 24h, 90m, 1d)"))
      Right
      (parseDurationSeconds lookbackWindowStr)
  if intervalSec <= 0
    then Left "Interval must be > 0"
    else if lookbackSec <= 0
      then Left "Lookback window must be > 0"
      else
        -- Use ceiling so the effective history covers at least the requested duration.
        let bars = (lookbackSec + intervalSec - 1) `div` intervalSec
         in Right (max 1 bars)

inferPeriodsPerYear :: String -> Double
inferPeriodsPerYear interval =
  case interval of
    "1m" -> 60 * 24 * 365
    "3m" -> 20 * 24 * 365
    "5m" -> 12 * 24 * 365
    "15m" -> 4 * 24 * 365
    "30m" -> 2 * 24 * 365
    "1h" -> 24 * 365
    "2h" -> 12 * 365
    "4h" -> 6 * 365
    "6h" -> 4 * 365
    "8h" -> 3 * 365
    "12h" -> 2 * 365
    "1d" -> 365
    "3d" -> 365 / 3
    "1w" -> 52
    "1M" -> 12
    _ -> 365

unitSeconds :: Char -> Maybe Int
unitSeconds u =
  case u of
    's' -> Just 1
    'm' -> Just 60
    'h' -> Just (60 * 60)
    'd' -> Just (24 * 60 * 60)
    'w' -> Just (7 * 24 * 60 * 60)
    'M' -> Just (30 * 24 * 60 * 60)
    _ ->
      case toLower u of
        -- Allow uppercase variants for convenience (except 'M' which already maps to month).
        's' -> Just 1
        'm' -> Just 60
        'h' -> Just (60 * 60)
        'd' -> Just (24 * 60 * 60)
        'w' -> Just (7 * 24 * 60 * 60)
        _ -> Nothing

readInt :: String -> Maybe Int
readInt s =
  case reads s of
    [(n, "")] -> Just n
    _ -> Nothing
