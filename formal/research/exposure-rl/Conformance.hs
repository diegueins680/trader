-- Conformance driver: one fixture per input line, one result per output line.
module Main (main) where

import Data.Word (Word64)
import ExposureShieldV1 (
    Outcome (..),
    intentTarget,
    mkSnapshot,
    orderAuthorized,
    reserve,
    shield,
 )
import GHC.Float (castWord64ToDouble)
import Text.Read (readMaybe)

type ShieldInput = (Integer, Integer, Bool, Integer, Word64)

type ReserveInput = (Integer, [Integer], Integer, Integer, Int, Integer)

-- Output: kind, target after the decision, live-order authority.
render :: Integer -> Outcome -> String
render current outcome =
    case outcome of
        Permitted intent -> show (0 :: Int, intentTarget intent, orderAuthorized intent)
        Abstain -> show (1 :: Int, current, False)
        Rejected (Just intent) -> show (2 :: Int, intentTarget intent, orderAuthorized intent)
        Rejected Nothing -> show (2 :: Int, current, False)
        RecoveryRequired -> show (3 :: Int, current, False)

respond :: String -> String
respond ('S' : rest) =
    case readMaybe rest :: Maybe ShieldInput of
        Nothing -> "invalid_fixture"
        Just (exposure, drawdown, fresh, headroom, bits) ->
            render exposure (shield (mkSnapshot exposure drawdown fresh headroom) (castWord64ToDouble bits))
respond ('R' : rest) =
    case readMaybe rest :: Maybe ReserveInput of
        Nothing -> "invalid_fixture"
        Just (budget, held, version, seen, index, wanted) -> show (reserve budget held version seen index wanted)
respond _ = "invalid_fixture"

main :: IO ()
main = interact (unlines . map respond . lines)
