{-# LANGUAGE LambdaCase #-}

-- Parser-only adapter: no application code, splices or plugins are executed.
module Main (main) where

import Control.Monad (forM_, unless)
import Control.Monad.IO.Class (liftIO)
import qualified Data.ByteString as BS
import Data.List (intercalate, isPrefixOf, sort)
import qualified Distribution.Compiler as Compiler
import Distribution.PackageDescription (BuildType (Simple), buildInfo, buildType, cSources, condExecutables, cppOptions, cxxSources, defaultExtensions, hcOptions, hsSourceDirs, modulePath, otherModules, packageDescription)
import Distribution.PackageDescription.Parsec (parseGenericPackageDescriptionMaybe)
import Distribution.Pretty (prettyShow)
import Distribution.Types.CondTree (condTreeComponents, condTreeData)
import Distribution.Types.UnqualComponentName (unUnqualComponentName)
import Distribution.Utils.Path (getSymbolicPath)
import GHC (getSessionDynFlags, runGhc)
import GHC.Data.FastString (mkFastString)
import GHC.Data.StringBuffer (stringToStringBuffer)
import GHC.Driver.Config.Parser (initParserOpts)
import GHC.Driver.Ppr (showSDoc)
import GHC.Driver.Session (DynFlags, parseDynamicFilePragma)
import GHC.Hs (HsDecl (..), HsModule (..), ImportDecl (..))
import qualified GHC.Parser as Parser
import GHC.Parser.Header (getOptions)
import GHC.Parser.Lexer (ParseResult (..), initParserState, unP)
import GHC.Types.SrcLoc (mkRealSrcLoc, unLoc)
import GHC.Unit.Module.Name (moduleNameString)
import GHC.Utils.Outputable (ppr)
import System.Environment (getArgs)
import System.FilePath (takeDirectory, (</>))

allowedFlags :: [String]
allowedFlags = map ("-X" ++) ["ApplicativeDo", "BangPatterns", "DeriveGeneric", "DuplicateRecordFields", "FlexibleInstances", "LambdaCase", "NumericUnderscores", "OverloadedStrings", "PatternSynonyms", "RecordWildCards", "ScopedTypeVariables", "TupleSections"]

parseSource :: DynFlags -> FilePath -> IO ()
parseSource base path = do
    source <- readFile path
    unless (not (any (\line -> take 1 line == "#") (lines source))) (fail "preprocessing is outside the graph contract")
    let buffer = stringToStringBuffer source
        (_, options) = getOptions (initParserOpts base) buffer path
    unless (all ((`elem` allowedFlags) . unLoc) options) (fail "unsupported source options")
    (flags, leftovers, _) <- parseDynamicFilePragma base options
    unless (null leftovers) (fail "unparsed source options")
    case unP Parser.parseModule (initParserState (initParserOpts flags) buffer (mkRealSrcLoc (mkFastString path) 1 1)) of
        PFailed _ -> fail ("GHC parse failed: " ++ path)
        POk _ located -> do
            let parsed = unLoc located
                forbidden = \case
                    ForD{} -> True
                    SpliceD{} -> True
                    AnnD{} -> True
                    _ -> False
            unless (not (any (forbidden . unLoc) (hsmodDecls parsed))) (fail "unhandled generated/foreign code")
            let imports = map (unLoc) (hsmodImports parsed)
            -- The pretty-printed declaration retains package/source qualifiers.
            unless (all (\i -> not ('"' `elem` showSDoc flags (ppr i)) && not ("SOURCE" `elem` words (showSDoc flags (ppr i)))) imports) (fail "qualified/boot import")
            let name = maybe "Main" (moduleNameString . unLoc) (hsmodName parsed)
            putStrLn (intercalate "\t" ["M", path, name, intercalate "," (sort (map (moduleNameString . unLoc . ideclName) imports))])
            -- Selected decoder/source declarations are emitted by the real parser.
            if path == "haskell/app/Trader/Research/PolicyProposalV1.hs"
                then putStrLn (intercalate "\t" ["E", path, show (showSDoc flags (ppr (hsmodExports parsed)))])
                else pure ()
            forM_ (hsmodDecls parsed) $ \decl -> do
                let rendered = showSDoc flags (ppr (unLoc decl))
                if path == "haskell/app/Trader/Research/PolicyProposalV1.hs" || (path == "haskell/app/Main.hs" && any (`isPrefixOf` rendered) ["data PersistedLstmModel", "instance FromJSON PersistedLstmModel", "jsonOptions prefixLen", "loadPersistedLstmModel path"])
                    then putStrLn (intercalate "\t" ["D", path, show rendered])
                    else pure ()

parseCabal :: FilePath -> IO ()
parseCabal path = do
    bytes <- BS.readFile path
    case parseGenericPackageDescriptionMaybe bytes of
        Nothing -> fail "Cabal parse failed"
        Just package -> do
            unless (buildType (packageDescription package) == Simple) (fail "custom package build")
            forM_ (condExecutables package) $ \(name, tree) -> do
                unless (null (condTreeComponents tree)) (fail "conditional executable is outside contract")
                let executable = condTreeData tree
                    info = buildInfo executable
                    dirs = map getSymbolicPath (hsSourceDirs info)
                unless (all (`elem` ["-O0", "-threaded", "-rtsopts", "-with-rtsopts=-N"]) (hcOptions Compiler.GHC info) && null (cppOptions info) && null (defaultExtensions info) && null (cSources info) && null (cxxSources info)) (fail "unhandled component options")
                unless (dirs == ["app"]) (fail "unhandled executable source roots")
                putStrLn (intercalate "\t" ["R", unUnqualComponentName name, takeDirectory path </> "app" </> modulePath executable, intercalate "," (sort (map prettyShow (otherModules info)))])

main :: IO ()
main = do
    getArgs >>= \case
        libdir : cabalFile : files -> runGhc (Just libdir) $ do
            flags <- getSessionDynFlags
            liftIO $ do
                parseCabal cabalFile
                mapM_ (parseSource flags) files
        _ -> fail "expected libdir, cabal file and source files"
