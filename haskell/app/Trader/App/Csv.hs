module Trader.App.Csv (
    CsvPriceSeries,
    loadCsvPriceSeries,
) where

import qualified Data.ByteString.Char8 as BS
import qualified Data.ByteString.Lazy as BL
import Data.Char (toLower)
import qualified Data.Csv as Csv
import qualified Data.HashMap.Strict as HM
import Data.Int (Int64)
import Data.List (isInfixOf, sortOn)
import qualified Data.Vector as V
import System.Directory (doesFileExist, getCurrentDirectory)
import Text.Read (readMaybe)

import Trader.App.Args (parseTimestampMs)
import Trader.Text (normalizeKey, trim)

type CsvPriceSeries =
    ( [Double]
    , Maybe [Double]
    , Maybe [Double]
    , Maybe [Double]
    , Maybe [Double]
    , Maybe [Int64]
    )

loadCsvPriceSeries ::
    FilePath ->
    String ->
    Maybe String ->
    Maybe String ->
    IO (Either String CsvPriceSeries)
loadCsvPriceSeries path closeCol mHighCol mLowCol = do
    exists <- doesFileExist path
    if not exists
        then do
            cwd <- getCurrentDirectory
            pure (Left ("CSV path not found: " ++ path ++ " (cwd: " ++ cwd ++ ")"))
        else do
            bs <- BL.readFile path
            case Csv.decodeByName bs of
                Left err -> pure (Left ("CSV decode failed (" ++ path ++ "): " ++ err))
                Right (hdr, rows) -> do
                    let hdrList = V.toList hdr
                        rowsList0 = V.toList rows
                        mTimeKey = csvTimeKey hdrList
                        mOpenKey = csvOpenKey hdrList
                        mVolumeKey = csvVolumeKey hdrList
                        rowsList = maybe rowsList0 (`sortCsvRowsByTime` rowsList0) mTimeKey
                        rowPairs = zip [1 :: Int ..] rowsList
                        seriesFor key = traverse (\(i, row) -> extractCellDoubleAt i key row) rowPairs
                        openTimesE =
                            case mTimeKey of
                                Nothing -> Right Nothing
                                Just tk -> Just <$> parseCsvTimes tk rowsList
                    pure $ do
                        closeKey <- resolveCsvColumnKey path "--price-column" hdrList closeCol
                        mHighKey <- traverse (resolveCsvColumnKey path "--high-column" hdrList) mHighCol
                        mLowKey <- traverse (resolveCsvColumnKey path "--low-column" hdrList) mLowCol
                        closeSeries <- seriesFor closeKey
                        openSeries <- traverse seriesFor mOpenKey
                        highSeries <- traverse seriesFor mHighKey
                        lowSeries <- traverse seriesFor mLowKey
                        volumeSeries <- traverse seriesFor mVolumeKey
                        openTimes <- openTimesE
                        pure (closeSeries, openSeries, highSeries, lowSeries, volumeSeries, openTimes)

resolveCsvColumnKey :: FilePath -> String -> [BS.ByteString] -> String -> Either String BS.ByteString
resolveCsvColumnKey path flagName hdrList raw =
    let wanted = trim raw
        wantedLower = map toLower wanted
        wantedNorm = normalizeKey wanted
        wantedBs = BS.pack wanted
        mKeyExact =
            if wantedBs `elem` hdrList then Just wantedBs else Nothing
        mKeyNorm =
            case filter (\h -> normalizeKey (BS.unpack h) == wantedNorm) hdrList of
                h : _ -> Just h
                [] -> Nothing
        mKey =
            case mKeyExact of
                Just k -> Just k
                Nothing ->
                    case filter (\h -> map toLower (BS.unpack h) == wantedLower) hdrList of
                        h : _ -> Just h
                        [] -> mKeyNorm
        available = BS.unpack (BS.intercalate (BS.pack ", ") hdrList)
        suggestions =
            let commonPrefixLen :: String -> String -> Int
                commonPrefixLen a b = length (takeWhile id (zipWith (==) a b))
                score hn =
                    let pref = commonPrefixLen wantedNorm hn
                        contains = if wantedNorm `isInfixOf` hn then 100 else 0
                     in contains + pref
                scored =
                    [ (negate s, h)
                    | h <- hdrList
                    , let hn = normalizeKey (BS.unpack h)
                    , let s = score hn
                    , s > 0
                    ]
             in take 5 (map snd (sortOn fst scored))
     in if null wanted
            then Left (flagName ++ " cannot be empty")
            else case mKey of
                Just k -> Right k
                Nothing ->
                    let hint =
                            if null suggestions
                                then ""
                                else " Suggestions: " ++ BS.unpack (BS.intercalate (BS.pack ", ") suggestions) ++ "."
                     in Left
                            ( "Column not found for "
                                ++ flagName
                                ++ ": "
                                ++ wanted
                                ++ " (file: "
                                ++ path
                                ++ "). Available columns: "
                                ++ available
                                ++ "."
                                ++ hint
                            )

extractCellDoubleAt :: Int -> BS.ByteString -> Csv.NamedRecord -> Either String Double
extractCellDoubleAt rowIndex key rec =
    case HM.lookup key rec of
        Nothing -> Left ("Column not found: " ++ BS.unpack key)
        Just raw ->
            let s = trim (BS.unpack raw)
             in case readMaybe s of
                    Just d ->
                        if isNaN d || isInfinite d
                            then
                                Left
                                    ( "Failed to parse finite value at row "
                                        ++ show rowIndex
                                        ++ " ("
                                        ++ BS.unpack key
                                        ++ "): "
                                        ++ s
                                    )
                            else Right d
                    Nothing ->
                        Left
                            ( "Failed to parse value at row "
                                ++ show rowIndex
                                ++ " ("
                                ++ BS.unpack key
                                ++ "): "
                                ++ s
                            )

csvTimeKey :: [BS.ByteString] -> Maybe BS.ByteString
csvTimeKey hdrList =
    let candidates =
            [ "openTimeMs"
            , "open_time_ms"
            , "openTime"
            , "open_time"
            , "timestamp"
            , "datetime"
            , "date"
            , "time"
            ]
     in firstJust [findHeaderKey hdrList c | c <- candidates]

csvOpenKey :: [BS.ByteString] -> Maybe BS.ByteString
csvOpenKey hdrList =
    let candidates =
            [ "open"
            , "openPrice"
            , "open_price"
            ]
     in firstJust [findHeaderKey hdrList c | c <- candidates]

csvVolumeKey :: [BS.ByteString] -> Maybe BS.ByteString
csvVolumeKey hdrList =
    let candidates =
            [ "volume"
            , "vol"
            , "baseVolume"
            , "base_volume"
            , "quoteVolume"
            , "quote_volume"
            ]
     in firstJust [findHeaderKey hdrList c | c <- candidates]

findHeaderKey :: [BS.ByteString] -> String -> Maybe BS.ByteString
findHeaderKey hdrList wanted =
    let w = normalizeKey wanted
        matches = filter (\h -> normalizeKey (BS.unpack h) == w) hdrList
     in case matches of
            h : _ -> Just h
            [] -> Nothing

sortCsvRowsByTime :: BS.ByteString -> [Csv.NamedRecord] -> [Csv.NamedRecord]
sortCsvRowsByTime timeKey rows =
    case traverse (lookupCell timeKey) rows of
        Nothing -> rows
        Just rawTimes ->
            let times = map (trim . BS.unpack) rawTimes
             in case traverse parseTimestampMs times of
                    Just ts ->
                        let pairs = zip ts rows
                         in map snd (sortOn fst pairs)
                    Nothing -> rows

parseCsvTimes :: BS.ByteString -> [Csv.NamedRecord] -> Either String [Int64]
parseCsvTimes timeKey rows =
    case traverse (lookupCell timeKey) rows of
        Nothing ->
            Left ("Time column missing: " ++ BS.unpack timeKey)
        Just rawTimes ->
            let times = map (trim . BS.unpack) rawTimes
                parseAt (idx, s) =
                    case parseTimestampMs s of
                        Just t -> Right t
                        Nothing ->
                            Left
                                ( "Failed to parse time value at row "
                                    ++ show idx
                                    ++ " ("
                                    ++ BS.unpack timeKey
                                    ++ "): "
                                    ++ s
                                )
             in traverse parseAt (zip [1 :: Int ..] times)

lookupCell :: BS.ByteString -> Csv.NamedRecord -> Maybe BS.ByteString
lookupCell = HM.lookup

firstJust :: [Maybe a] -> Maybe a
firstJust xs =
    case xs of
        [] -> Nothing
        y : ys ->
            case y of
                Just _ -> y
                Nothing -> firstJust ys
