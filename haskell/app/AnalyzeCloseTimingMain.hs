{-# LANGUAGE OverloadedStrings #-}

module Main where

import Data.Aeson ((.!=), (.:), (.:?), (.=))
import qualified Data.Aeson as Aeson
import qualified Data.ByteString.Lazy as BL
import Data.List (sortOn)
import Data.Maybe (fromMaybe)
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
        comboId <- fmap (maybe "unknown" T.unpack) (o .:? "comboId")
        prices <- o .: "prices"
        trades <- o .: "trades"
        pure ComboInput{ciComboId = comboId, ciPrices = prices, ciTrades = trades}

instance Aeson.FromJSON ComboTrade where
    parseJSON = Aeson.withObject "ComboTrade" $ \o -> do
        comboId <- fmap T.unpack (o .:? "comboId" .!= "unknown")
        entryIndex <- o .: "entryIndex"
        exitIndex <- o .: "exitIndex"
        entryPrice <- o .:? "entryPrice" .!= 0
        side <- o .:? "side" .!= (1 :: Double)
        pure
            ComboTrade
                { ctComboId = comboId
                , ctEntryIndex = entryIndex
                , ctExitIndex = exitIndex
                , ctEntryPrice = entryPrice
                , ctSide = side
                }

instance Aeson.ToJSON TradeTimingSample where
    toJSON sample =
        Aeson.object
            [ "comboId" .= ttsComboId sample
            , "entryIndex" .= ttsEntryIndex sample
            , "exitIndex" .= ttsExitIndex sample
            , "optimalIndex" .= ttsOptimalIndex sample
            , "observedDuration" .= ttsObservedDuration sample
            , "optimalDuration" .= ttsOptimalDuration sample
            , "observedReturn" .= ttsObservedReturn sample
            , "optimalReturn" .= ttsOptimalReturn sample
            , "returnLift" .= ttsReturnLift sample
            ]

instance Aeson.ToJSON ComboTimingStats where
    toJSON stats =
        Aeson.object
            [ "comboId" .= ctsComboId stats
            , "sampleCount" .= ctsSampleCount stats
            , "medianRatio" .= ctsMedianRatio stats
            , "q25Ratio" .= ctsQ25Ratio stats
            , "q75Ratio" .= ctsQ75Ratio stats
            , "madRatio" .= ctsMadRatio stats
            , "meanLift" .= ctsMeanLift stats
            , "medianLift" .= ctsMedianLift stats
            ]

argsParser :: Parser Args
argsParser =
    Args
        <$> strOption (long "input" <> metavar "FILE" <> help "JSON file with combo prices/trades")
        <*> optional (strOption (long "output" <> metavar "FILE" <> help "Write report JSON to file"))

reportForCombo :: ComboInput -> [TradeTimingSample]
reportForCombo comboInput =
    let comboId = ciComboId comboInput
        prices = ciPrices comboInput
        trades = map (normalizeTrade comboId prices) (ciTrades comboInput)
     in timingSamples prices trades

normalizeTrade :: String -> [Double] -> ComboTrade -> ComboTrade
normalizeTrade comboId prices trade =
    let comboFixed =
            if ctComboId trade == "unknown"
                then trade{ctComboId = comboId}
                else trade
        entryPrice =
            if ctEntryPrice comboFixed > 0
                then ctEntryPrice comboFixed
                else fromMaybe 1 (safeAt prices (ctEntryIndex comboFixed))
     in comboFixed{ctEntryPrice = entryPrice}

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
                comboStats = sortOn ctsComboId (summarizeAllCombos samples)
                payload =
                    Aeson.object
                        [ "samples" .= samples
                        , "comboStats" .= comboStats
                        , "decisionIntegration"
                            .= Aeson.object
                                [ "rule" .= ("Close when position age reaches the combo's robust median-to-upper-quartile tm/(tc-ta) band and medianLift remains positive." :: String)
                                ]
                        ]
                encoded = Aeson.encode payload
            case argOutput args of
                Just outputPath -> BL.writeFile outputPath encoded
                Nothing -> BL.putStr encoded
