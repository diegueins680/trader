module Trader.App.Env (
    getBuildCommit,
    loadEnvFile,
    traderVersion,
) where

import Control.Applicative ((<|>))
import Control.Exception (IOException, SomeException, displayException, try)
import Control.Monad (forM_, when)
import Data.Char (isSpace)
import Data.List (isPrefixOf, stripPrefix)
import Data.Maybe (fromMaybe, isJust, mapMaybe)
import Data.Version (showVersion)
import qualified Paths_trader as Paths
import System.Directory (doesFileExist, findExecutable, getCurrentDirectory)
import System.Environment (lookupEnv, setEnv)
import System.Exit (ExitCode (..))
import System.FilePath (isAbsolute, takeDirectory, (</>))
import System.IO (hPutStrLn, stderr)
import System.Process (proc, readCreateProcessWithExitCode)

import Trader.Text (trim)

traderVersion :: String
traderVersion = showVersion Paths.version

getBuildCommit :: IO (Maybe String)
getBuildCommit = do
    envCommit <- resolveEnvBuildCommit
    case envCommit of
        Just commit -> pure (Just commit)
        Nothing -> do
            fileCommit <- resolveBuildCommitFile
            case fileCommit of
                Just commit -> pure (Just commit)
                Nothing -> resolveGitBuildCommit

resolveEnvBuildCommit :: IO (Maybe String)
resolveEnvBuildCommit = do
    let keys =
            [ "TRADER_GIT_COMMIT"
            , "TRADER_COMMIT"
            , "GIT_COMMIT"
            , "COMMIT_SHA"
            , "SOURCE_COMMIT"
            , "SOURCE_VERSION"
            , "GITHUB_SHA"
            ]
    firstNonEmpty <$> mapM lookupEnv keys

resolveBuildCommitFile :: IO (Maybe String)
resolveBuildCommitFile = do
    cwd <- getCurrentDirectory
    let candidates =
            [ cwd </> ".build-commit"
            , cwd </> "haskell" </> ".build-commit"
            , takeDirectory cwd </> ".build-commit"
            , takeDirectory cwd </> "haskell" </> ".build-commit"
            ]
    readFirstBuildCommitFile candidates

readFirstBuildCommitFile :: [FilePath] -> IO (Maybe String)
readFirstBuildCommitFile paths =
    case paths of
        [] -> pure Nothing
        path : rest -> do
            exists <- doesFileExist path
            if not exists
                then readFirstBuildCommitFile rest
                else do
                    contentsResult <- try (readFile path) :: IO (Either IOException String)
                    case contentsResult of
                        Left _ -> readFirstBuildCommitFile rest
                        Right contents ->
                            case nonEmptyTrimmed contents of
                                Just commit -> pure (Just commit)
                                Nothing -> readFirstBuildCommitFile rest

resolveGitBuildCommit :: IO (Maybe String)
resolveGitBuildCommit = do
    gitExe <- findExecutable "git"
    case gitExe of
        Nothing -> pure Nothing
        Just _ -> do
            (code, out, err) <- readCreateProcessWithExitCode (proc "git" ["rev-parse", "HEAD"]) ""
            case code of
                ExitSuccess -> pure (nonEmptyTrimmed out)
                ExitFailure _ -> pure Nothing

loadEnvFile :: IO (Either String FilePath) -> IO ()
loadEnvFile resolveRepoRoot = do
    mEnvFileRaw <- lookupEnv "TRADER_ENV_FILE"
    let envFileRaw = maybe ".env" trim mEnvFileRaw
        envFileName = trim envFileRaw
    if null envFileName
        then pure ()
        else do
            mPath <- resolveEnvFilePath resolveRepoRoot envFileName
            case mPath of
                Nothing -> do
                    when (isJust mEnvFileRaw) $
                        hPutStrLn stderr ("WARN: TRADER_ENV_FILE not found: " ++ envFileName)
                Just path -> do
                    contentsResult <- try (readFile path) :: IO (Either IOException String)
                    case contentsResult of
                        Left err ->
                            hPutStrLn stderr ("WARN: failed to read TRADER_ENV_FILE (" ++ path ++ "): " ++ displayException err)
                        Right contents -> do
                            let entries = parseEnvFile contents
                            forM_ entries $ \(key, value) -> do
                                existing <- lookupEnv key
                                case existing of
                                    Nothing -> setEnv key value
                                    Just _ -> pure ()

resolveEnvFilePath :: IO (Either String FilePath) -> FilePath -> IO (Maybe FilePath)
resolveEnvFilePath resolveRepoRoot raw = do
    let trimmed = trim raw
    if null trimmed
        then pure Nothing
        else
            if isAbsolute trimmed
                then do
                    exists <- doesFileExist trimmed
                    pure (if exists then Just trimmed else Nothing)
                else do
                    cwd <- getCurrentDirectory
                    let cwdPath = cwd </> trimmed
                    cwdExists <- doesFileExist cwdPath
                    if cwdExists
                        then pure (Just cwdPath)
                        else do
                            let parent = takeDirectory cwd
                                parentPath = parent </> trimmed
                            parentExists <- doesFileExist parentPath
                            if parentExists
                                then pure (Just parentPath)
                                else do
                                    gitExe <- findExecutable "git"
                                    case gitExe of
                                        Nothing -> pure Nothing
                                        Just _ -> do
                                            rootResult <- try resolveRepoRoot :: IO (Either SomeException (Either String FilePath))
                                            case rootResult of
                                                Left _ -> pure Nothing
                                                Right (Right root) -> do
                                                    let rootPath = root </> trimmed
                                                    rootExists <- doesFileExist rootPath
                                                    pure (if rootExists then Just rootPath else Nothing)
                                                Right (Left _) -> pure Nothing

parseEnvFile :: String -> [(String, String)]
parseEnvFile =
    mapMaybe parseEnvLine . lines

parseEnvLine :: String -> Maybe (String, String)
parseEnvLine raw =
    let stripped = trim raw
     in if null stripped || "#" `isPrefixOf` stripped
            then Nothing
            else
                let withoutExport = fromMaybe stripped (stripPrefix "export " stripped)
                    (keyRaw, rest) = break (== '=') withoutExport
                 in case rest of
                        '=' : valueRaw ->
                            let key = trim keyRaw
                                value = parseEnvValue valueRaw
                             in if null key then Nothing else Just (key, value)
                        _ -> Nothing

parseEnvValue :: String -> String
parseEnvValue raw =
    let trimmed = trim (stripInlineComment raw)
        stripWrappedQuote q s =
            case s of
                first : rest
                    | first == q ->
                        case reverse rest of
                            lastChar : innerRev
                                | lastChar == q -> Just (reverse innerRev)
                            _ -> Nothing
                _ -> Nothing
     in fromMaybe trimmed $
            (unescapeDoubleQuoted <$> stripWrappedQuote '"' trimmed)
                <|> stripWrappedQuote '\'' trimmed

stripInlineComment :: String -> String
stripInlineComment = go False False False []
  where
    go _ _ _ acc [] = reverse acc
    go inSingle inDouble escaped acc (c : cs)
        | escaped = go inSingle inDouble False (c : acc) cs
        | c == '\\' && inDouble = go inSingle inDouble True (c : acc) cs
        | c == '\'' && not inDouble = go (not inSingle) inDouble False (c : acc) cs
        | c == '"' && not inSingle = go inSingle (not inDouble) False (c : acc) cs
        | c == '#' && not inSingle && not inDouble =
            let prev = case acc of
                    [] -> Nothing
                    p : _ -> Just p
             in if maybe True isSpace prev then reverse acc else go inSingle inDouble False (c : acc) cs
        | otherwise = go inSingle inDouble False (c : acc) cs

unescapeDoubleQuoted :: String -> String
unescapeDoubleQuoted = go
  where
    go [] = []
    go ('\\' : 'n' : rest) = '\n' : go rest
    go ('\\' : 'r' : rest) = '\r' : go rest
    go ('\\' : 't' : rest) = '\t' : go rest
    go ('\\' : '"' : rest) = '"' : go rest
    go ('\\' : '\\' : rest) = '\\' : go rest
    go ('\\' : c : rest) = c : go rest
    go (c : rest) = c : go rest

firstNonEmpty :: [Maybe String] -> Maybe String
firstNonEmpty xs =
    case xs of
        [] -> Nothing
        y : ys ->
            case y >>= nonEmptyTrimmed of
                Just value -> Just value
                Nothing -> firstNonEmpty ys

nonEmptyTrimmed :: String -> Maybe String
nonEmptyTrimmed raw =
    let cleaned = trim raw
     in if null cleaned then Nothing else Just cleaned
