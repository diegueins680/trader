{-# LANGUAGE OverloadedStrings #-}

module Main where

import Data.Aeson ((.:), (.:?), (.=))
import qualified Data.Aeson as Aeson
import qualified Data.ByteString.Lazy as BL
import Data.List (sortOn)
import qualified Data.Text as T
import Options.Applicative
import Trader.Formal.CloseTiming (ComboTimingStats (..), ComboTrade (..), TradeTimingSample (..), summarizeAllCombos, timingSamples)

data Args = Args
    { argInput :: FilePath
    , argOutput :: Maybe FilePath
    }

data ComboInput = ComboInput
    { ciComboId :: !String
    , ciPrices :: ![Double]
    , ciTrades :: ![ComboTrade]
    }

instance Aeson.FromJSON ComboInput where
    parseJSON = Aeson.withObject "ComboInput" $ \o -> do
        mComboId <- o .:? "comboId"
        let comboId = maybe "unknown" T.unpack mComboId
        prices <- o .: "prices"
        trades <- o .: "trades"
        pure ComboInput{ciComboId = comboId, ciPrices = prices, ciTrades = trades}

argsParser :: Parser Args
argsParser =
    Args
        <$> strOption (long "input" <> metavar "FILE" <> help "JSON file with combo prices/trades")
        <*> optional (strOption (long "output" <> metavar "FILE" <> help "Write report JSON to file"))

reportForCombo :: ComboInput -> [TradeTimingSample]
reportForCombo ci =
    let baseId = ciComboId ci
        prices = ciPrices ci
        trades = map (ensureCombo prices) (ciTrades ci)
     in timingSamples prices trades
  where
    ensureCombo prices tr =
        let comboFixed = if ctComboId tr == "unknown" then tr{ctComboId = baseId} else tr
            entryPx = ctEntryPrice comboFixed
            pxFixed =
                if entryPx > 0
                    then entryPx
                    else case safeAt prices (ctEntryIndex comboFixed) of
                        Just px -> px
                        Nothing ->
                            error
                                ( "Trade in combo '"
                                    ++ ctComboId comboFixed
                                    ++ "' at entryIndex="
                                    ++ show (ctEntryIndex comboFixed)
                                    ++ " has no valid entryPrice and index is out of range"
                                )
         in comboFixed{ctEntryPrice = pxFixed}

safeAt :: [a] -> Int -> Maybe a
safeAt xs i
    | i < 0 = Nothing
    | otherwise = go xs i
  where
    go [] _ = Nothing
    go (y : ys) k
        | k == 0 = Just y
        | otherwise = go ys (k - 1)

main :: IO ()
main = do
    args <- execParser (info (argsParser <**> helper) fullDesc)
    raw <- BL.readFile (argInput args)
    case Aeson.eitherDecode raw of
        Left err -> fail ("Invalid input JSON: " ++ err)
        Right combos -> do
            let samples = concatMap reportForCombo (combos :: [ComboInput])
                stats = sortOn ctsComboId (summarizeAllCombos samples)
                payload =
                    Aeson.object
                        [ "samples" .= samples
                        , "comboStats" .= stats
                        , "decisionIntegration" .=
                            Aeson.object
                                [ "rule" .= ("Close when the position age reaches the 50th-75th percentile of the tm/td ratio per combo, using a robust band [Q25, Q75] and an expected lift threshold > 0." :: String)
                                ]
                        ]
                encoded = Aeson.encode payload
            case argOutput args of
                Just out -> BL.writeFile out encoded
                Nothing -> BL.putStr encoded
