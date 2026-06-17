module Main (main) where

import Data.Char (isSpace)
import Options.Applicative
import System.Environment (getArgs, getProgName)
import System.Exit (ExitCode (..), exitSuccess, exitWith)
import System.IO (hPutStrLn, stderr)

import Trader.Optimizer.Merge (MergeArgs (..), runMerge)
import Trader.TopComboScoring (
    TopComboScoringConfig (..),
    defaultTopComboScoringConfig,
    validateTopComboScoringConfig,
 )

main :: IO ()
main = do
    argv <- getArgs
    progName <- getProgName
    args <- parseArgs progName argv
    case validateArgs args of
        Left err -> do
            hPutStrLn stderr err
            exitWith (ExitFailure 2)
        Right args' -> do
            code <- runMerge args'
            exitWith (if code == 0 then ExitSuccess else ExitFailure code)

parseArgs :: String -> [String] -> IO MergeArgs
parseArgs progName argv =
    handleParseResult $
        execParserPure
            (prefs showHelpOnError)
            (info (mergeArgsParser <**> helper) (fullDesc <> progDesc "Merge optimizer outputs into top-combos.json."))
            argv
  where
    handleParseResult result =
        case result of
            Success opts -> pure opts
            Failure failure -> do
                let (msg, _) = renderFailure failure progName
                hPutStrLn stderr (stripTrailingSpace msg)
                exitWith (ExitFailure 2)
            CompletionInvoked compl -> do
                msg <- execCompletion compl progName
                putStr msg
                exitSuccess

mergeArgsParser :: Parser MergeArgs
mergeArgsParser =
    MergeArgs
        <$> strOption (long "top-json" <> value "haskell/web/public/top-combos.json" <> metavar "PATH")
        <*> many (strOption (long "from-jsonl" <> metavar "PATH"))
        <*> many (strOption (long "from-top-json" <> metavar "PATH"))
        <*> strOption (long "out" <> value "" <> metavar "PATH")
        <*> option auto (long "max" <> value 200 <> metavar "INT")
        <*> optional (strOption (long "history-dir" <> metavar "PATH"))
        <*> scoringConfigParser
        <*> switch (long "copy-to-dist")

scoringConfigParser :: Parser TopComboScoringConfig
scoringConfigParser =
    TopComboScoringConfig
        <$> scoreDouble "score-live-blend-shrinkage-ops" tcscLiveBlendShrinkageOps "Pseudo-count of live operations used to shrink live annualized return into the backtest prior"
        <*> scoreInt "score-live-quarantine-min-operations" tcscLiveQuarantineMinOperations "Minimum completed live operations before a combo can be quarantined"
        <*> scoreDouble "score-live-quarantine-max-final-equity" tcscLiveQuarantineMaxFinalEquity "Final live equity ceiling that quarantines a combo after enough live operations"
        <*> scoreDouble "score-live-annualized-return-floor" tcscLiveAnnualizedReturnFloor "Floor applied to live annualized return before blending"
        <*> scoreDouble "score-live-annualized-return-ceiling" tcscLiveAnnualizedReturnCeiling "Ceiling applied to live annualized return before blending"
        <*> scoreDouble "score-validated-annualized-return-cap" tcscValidatedAnnualizedReturnCap "Cap applied to blended annualized return inside validatedScore"
        <*> scoreDouble "score-wf-deployable-multiplier" tcscDeployableWalkForwardMultiplier "validatedScore multiplier for walk-forward-deployable combos"
        <*> scoreDouble "score-wf-below-floor-multiplier" tcscBelowFloorWalkForwardMultiplier "validatedScore multiplier for combos with below-floor walk-forward Sharpe"
        <*> scoreDouble "score-wf-missing-multiplier" tcscMissingWalkForwardMultiplier "validatedScore multiplier for combos missing walk-forward Sharpe"
        <*> scoreDouble "score-drawdown-penalty-scale" tcscDrawdownPenaltyScale "Drawdown penalty scale used as 1/(1 + scale * drawdown)"
        <*> scoreDouble "score-equity-floor" tcscEquityFloor "Minimum final equity used before the log-equity term"
        <*> scoreDouble "score-equity-log-floor" tcscEquityLogFloor "Floor applied to the log-equity term in validatedScore"
        <*> scoreDouble "score-freshness-half-life-days" tcscFreshnessHalfLifeDays "Half-life in days for ranking by combo createdAtMs; 0 disables the freshness multiplier"
        <*> scoreDouble "score-freshness-floor-multiplier" tcscFreshnessFloorMultiplier "Lowest multiplier applied by the freshness rank adjustment"
  where
    def = defaultTopComboScoringConfig
    scoreDouble name getter helpText =
        option auto (long name <> value (getter def) <> showDefault <> help helpText)
    scoreInt name getter helpText =
        option auto (long name <> value (getter def) <> showDefault <> help helpText)

validateArgs :: MergeArgs -> Either String MergeArgs
validateArgs args = do
    let historyDir = normalizeMaybe (maHistoryDir args)
    scoringConfig <- validateTopComboScoringConfig (maScoringConfig args)
    pure args{maHistoryDir = historyDir, maScoringConfig = scoringConfig}

normalizeMaybe :: Maybe String -> Maybe String
normalizeMaybe value =
    case value of
        Nothing -> Nothing
        Just raw ->
            let trimmed = trim raw
             in if null trimmed then Nothing else Just trimmed

trim :: String -> String
trim = dropWhileEnd isSpace . dropWhile isSpace

dropWhileEnd :: (a -> Bool) -> [a] -> [a]
dropWhileEnd p = reverse . dropWhile p . reverse

stripTrailingSpace :: String -> String
stripTrailingSpace = reverse . dropWhile isSpace . reverse
