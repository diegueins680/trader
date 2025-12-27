module Main (main) where

import Data.Char (isSpace)
import Options.Applicative
import System.Environment (getArgs, getProgName)
import System.Exit (ExitCode (..), exitSuccess, exitWith)
import System.IO (hPutStrLn, stderr)

import Trader.Optimizer.Merge (MergeArgs (..), runMerge)

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
    <*> switch (long "copy-to-dist")

validateArgs :: MergeArgs -> Either String MergeArgs
validateArgs args = do
  let historyDir = normalizeMaybe (maHistoryDir args)
  pure args {maHistoryDir = historyDir}

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
