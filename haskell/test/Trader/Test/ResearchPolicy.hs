module Trader.Test.ResearchPolicy (runResearchPolicyTests) where

import Control.Monad (forM_, unless)
import Trader.Research.PolicyProposalV1 (
    ProposalEvidence (..),
    ResearchMode (..),
    defaultResearchMode,
    orderAuthorized,
    proposalTarget,
    screenProposal,
 )

runResearchPolicyTests :: IO ()
runResearchPolicyTests = do
    assert "research defaults disabled" (defaultResearchMode == Disabled)
    forM_ cases $ \(mode, evidence, target) ->
        case screenProposal mode evidence target of
            Nothing ->
                assert "accepted research domain is exact" (not (admissible mode evidence target))
            Just proposal -> do
                assert "deterministic safety has precedence" (admissible mode evidence target)
                assert "proposal stays bounded" (abs (proposalTarget proposal) <= 0.25)
                assert "no policy proposal authorizes an order" (not (orderAuthorized proposal))
  where
    assert message ok = unless ok (ioError (userError message))
    finite x = not (isNaN x || isInfinite x)
    admissible mode e target =
        mode == OfflineReplayV1
            && observationValid e
            && positionOwned e
            && artifactCompatible e
            && actionSupported e
            && deterministicRiskAccepted e
            && finite (elapsedMilliseconds e)
            && elapsedMilliseconds e >= 0
            && elapsedMilliseconds e <= 20
            && finite target
            && target `elem` [-0.25, 0, 0.25]
    cases =
        [ (mode, ProposalEvidence valid owned artifact supported risk elapsed, target)
        | mode <- [Disabled, OfflineReplayV1]
        , valid <- [False, True]
        , owned <- [False, True]
        , artifact <- [False, True]
        , supported <- [False, True]
        , risk <- [False, True]
        , elapsed <- [-1, 0, 20, 21, 0 / 0, 1 / 0]
        , target <- [-1, -0.25, 0, 0.25, 1, 0 / 0, 1 / 0]
        ]
