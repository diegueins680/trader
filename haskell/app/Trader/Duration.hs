module Trader.Duration (
    parseDurationSeconds,
    parseIntervalSeconds,
    TimeWindow (..),
    parseTimeWindow,
    timeWindowCode,
    timeWindowContains,
    minuteOfDayFromMs,
    inferPeriodsPerYear,
    lookbackBarsFrom,
) where

import Data.Char (isDigit, isSpace, toLower)
import Data.Int (Int64)

trim :: String -> String
trim = dropWhileEnd isSpace . dropWhile isSpace

dropWhileEnd :: (a -> Bool) -> [a] -> [a]
dropWhileEnd p = reverse . dropWhile p . reverse

data TimeWindow = TimeWindow
    { twStartMin :: !Int
    , twEndMin :: !Int
    }
    deriving (Eq, Show)

parseTimeWindow :: String -> Either String TimeWindow
parseTimeWindow raw =
    let s = trim raw
     in case break (== '-') s of
            (startRaw, '-' : endRaw) -> do
                start <- parseTimeOfDay startRaw
                end <- parseTimeOfDay endRaw
                if start == end
                    then Left "Time window start and end must differ."
                    else Right (TimeWindow start end)
            _ -> Left "Expected HH:MM-HH:MM (e.g., 12:30-13:15)."

timeWindowCode :: TimeWindow -> String
timeWindowCode (TimeWindow start end) =
    let (sh, sm) = start `divMod` 60
        (eh, em) = end `divMod` 60
     in pad2 sh ++ ":" ++ pad2 sm ++ "-" ++ pad2 eh ++ ":" ++ pad2 em

timeWindowContains :: TimeWindow -> Int -> Bool
timeWindowContains (TimeWindow start end) minuteRaw =
    let minute = minuteRaw `mod` 1440
     in if start < end
            then minute >= start && minute < end
            else minute >= start || minute < end

minuteOfDayFromMs :: Int64 -> Int
minuteOfDayFromMs ts =
    let minutes = toInteger ts `div` 60000
     in fromInteger (minutes `mod` 1440)

{- | Parse a human duration like \"24h\", \"15m\", \"1d\" into seconds.
Supports units: s, m, h, d, w, M (month ~= 30d).
-}
parseDurationSeconds :: String -> Maybe Int
parseDurationSeconds raw =
    let s = trim raw
     in case span isDigit s of
            ("", _) -> Nothing
            (nStr, [u]) -> do
                n <- readIntegerStrict nStr
                mult <- unitSeconds u
                checkedIntProduct n mult
            _ -> Nothing

{- | Parse a Binance-style interval like \"5m\", \"1h\", \"1d\" into seconds.
This is identical to 'parseDurationSeconds' but kept separate for clarity.
-}
parseIntervalSeconds :: String -> Maybe Int
parseIntervalSeconds = parseDurationSeconds

parseTimeOfDay :: String -> Either String Int
parseTimeOfDay raw =
    let s = trim raw
     in case break (== ':') s of
            (hStr, ':' : mStr)
                | isTwoDigitClockPart hStr && isTwoDigitClockPart mStr -> do
                    h <- readIntEither hStr
                    m <- readIntEither mStr
                    if h < 0 || h > 23
                        then Left "Hour must be between 0 and 23."
                        else
                            if m < 0 || m > 59
                                then Left "Minute must be between 0 and 59."
                                else Right (h * 60 + m)
                | otherwise -> Left "Expected HH:MM."
            _ -> Left "Expected HH:MM."

isTwoDigitClockPart :: String -> Bool
isTwoDigitClockPart part =
    case part of
        [a, b] -> isDigit a && isDigit b
        _ -> False

pad2 :: Int -> String
pad2 n =
    let s = show (abs n)
     in case s of
            [d] -> ['0', d]
            _ -> s

readIntEither :: String -> Either String Int
readIntEither s =
    case readInt s of
        Just n -> Right n
        Nothing -> Left "Expected an integer."

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
        else
            if lookbackSec <= 0
                then Left "Lookback window must be > 0"
                else
                    -- Use ceiling so the effective history covers at least the requested duration.
                    let intervalI = toInteger intervalSec
                        lookbackI = toInteger lookbackSec
                        barsI = (lookbackI + intervalI - 1) `div` intervalI
                     in case integerToIntBounded barsI of
                            Just bars -> Right (max 1 bars)
                            Nothing -> Left "Lookback window too large for this interval."

inferPeriodsPerYear :: String -> Double
inferPeriodsPerYear interval =
    case parseIntervalSeconds interval of
        Just sec
            | sec > 0 ->
                let yearSec = 365 * 24 * 60 * 60 :: Int
                 in fromIntegral yearSec / fromIntegral sec
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

checkedIntProduct :: Integer -> Int -> Maybe Int
checkedIntProduct a b =
    integerToIntBounded (a * toInteger b)

readIntegerStrict :: String -> Maybe Integer
readIntegerStrict s =
    case reads s :: [(Integer, String)] of
        [(n, "")] -> Just n
        _ -> Nothing

integerToIntBounded :: Integer -> Maybe Int
integerToIntBounded n =
    if n < lo || n > hi
        then Nothing
        else Just (fromInteger n)
  where
    lo = toInteger (minBound :: Int)
    hi = toInteger (maxBound :: Int)

readInt :: String -> Maybe Int
readInt s =
    readIntegerStrict s >>= integerToIntBounded
