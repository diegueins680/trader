{- | Isolated research proposal boundary. This module is not imported by any
production executable, exchange adapter, bot, or artifact loader. A value here
is never an order authorization. Deployment integration requires separate work.
-}
module Trader.Research.PolicyProposalV1 (
    ResearchMode (..),
    ProposalEvidence (..),
    ResearchProposal,
    defaultResearchMode,
    screenProposal,
    proposalTarget,
    orderAuthorized,
) where

data ResearchMode = Disabled | OfflineReplayV1
    deriving (Eq, Show)

data ProposalEvidence = ProposalEvidence
    { observationValid :: !Bool
    , positionOwned :: !Bool
    , artifactCompatible :: !Bool
    , actionSupported :: !Bool
    , deterministicRiskAccepted :: !Bool
    , elapsedMilliseconds :: !Double
    }
    deriving (Eq, Show)

-- Constructor intentionally private; no order/exchange identity is carried.
newtype ResearchProposal = ResearchProposal Double
    deriving (Eq, Show)

defaultResearchMode :: ResearchMode
defaultResearchMode = Disabled

screenProposal :: ResearchMode -> ProposalEvidence -> Double -> Maybe ResearchProposal
screenProposal mode evidence target
    | mode /= OfflineReplayV1 = Nothing
    | not (observationValid evidence && positionOwned evidence && artifactCompatible evidence && actionSupported evidence && deterministicRiskAccepted evidence) = Nothing
    | not (finite elapsed && elapsed >= 0 && elapsed <= 20) = Nothing
    | not (finite target && target `elem` [-0.25, 0, 0.25]) = Nothing
    | otherwise = Just (ResearchProposal target)
  where
    elapsed = elapsedMilliseconds evidence
    finite x = not (isNaN x || isInfinite x)

proposalTarget :: ResearchProposal -> Double
proposalTarget (ResearchProposal target) = target

-- | No research proposal, even an accepted one, carries order authority.
orderAuthorized :: ResearchProposal -> Bool
orderAuthorized _ = False
