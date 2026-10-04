module Main (main) where

import Data.Word (Word64)
import GHC.Float (castDoubleToWord64, castWord64ToDouble)
import Text.Read (readMaybe)
import Trader.Research.PolicyProposalV1 (
    ProposalEvidence (..),
    ResearchMode (..),
    defaultResearchMode,
    orderAuthorized,
    proposalTarget,
    screenProposal,
 )

type Input = (Bool, Bool, Bool, Bool, Bool, Bool, Word64, Word64)

respond :: String -> String
respond "default" = show defaultResearchMode
respond line =
    case readMaybe line :: Maybe Input of
        Nothing -> "invalid_fixture"
        Just (enabled, valid, owned, artifact, support, risk, elapsed, target) ->
            let mode = if enabled then OfflineReplayV1 else Disabled
                evidence = ProposalEvidence valid owned artifact support risk (castWord64ToDouble elapsed)
             in case screenProposal mode evidence (castWord64ToDouble target) of
                    Nothing -> "absent"
                    Just proposal -> show (castDoubleToWord64 (proposalTarget proposal), orderAuthorized proposal)

main :: IO ()
main = interact (unlines . map respond . lines)
