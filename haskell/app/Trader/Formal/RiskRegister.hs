{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}

{- | Risk Register — Trader Firm

This module is the typed Haskell projection of @formal/risk-register.json@.
The formal verifier requires its IDs, severities, statuses, and ordering to
match the canonical registry and the operator-facing Markdown projection.
-}
module Trader.Formal.RiskRegister (
    RiskID (..),
    RiskSeverity (..),
    RiskStatus (..),
    RiskEntry (..),
    riskIdText,
    riskRegister,
    riskSeverityOf,
    riskStatusOf,
) where

import qualified Data.Text as T

-- | Unique risk identifier. 'riskIdText' is its canonical external form.
data RiskID
    = AUTOLOOP_DOWN_003
    | AUTOLOOP_RESET_2026_05_30
    | AUTOLOOP_SINGLETON_001
    | AUTOLOOP_STALL_001
    | BINARY_HANG_001
    | CIO_DEAFNESS_001
    | EXECUTION_DATASET_001
    | EXECUTION_MISSING_001
    | EXPECTANCY_INVALID_001
    | FEATURE_MISSINGNESS_001
    | GITHUB_502_001
    | KALMAN_NUMSTAB_001
    | LEVERAGE_INVALID_001
    | LEVERAGE_SANITY_001
    | LOSS_STREAK_LIMIT_INVALID_001
    | MARKET_DATA_TIMESTAMP_OVERFLOW_001
    | MAX_POSITION_GUARDRAIL_001
    | PREDICTOR_IDENTITY_001
    | RESEARCH_RECEIPT_001
    | RISK_LIMIT_001
    | RISK_LIMIT_NON_FINITE_001
    | RISK_METRIC_INVALID_001
    | SCHEMA_001
    | THRESHOLD_FACTOR_001
    | TRADE_LOG_GAP_001
    | TRADE_LOG_GAP_002
    | TRAILING_STOP_001
    | VOL_TARGET_001
    | VOL_TARGET_INVALID_001
    | ZERO_VIABLE_SIGNAL_001
    deriving (Bounded, Enum, Eq, Ord, Show)

-- | Impact if the risk materializes. Lifecycle is represented separately.
data RiskSeverity
    = LOW
    | MEDIUM
    | HIGH
    | CRITICAL
    deriving (Bounded, Enum, Eq, Ord, Show)

-- | Current lifecycle state of a risk.
data RiskStatus
    = OPEN
    | MITIGATED
    | CLOSED
    deriving (Bounded, Enum, Eq, Ord, Show)

-- | Single risk entry.
data RiskEntry = RiskEntry
    { reId :: !RiskID
    , reSeverity :: !RiskSeverity
    , reStatus :: !RiskStatus
    , reDescription :: !T.Text
    , reOwner :: !T.Text
    , reMitigation :: !T.Text
    }
    deriving (Eq, Show)

-- | Stable machine-readable ID shared by every risk-register projection.
riskIdText :: RiskID -> T.Text
riskIdText = \case
    AUTOLOOP_DOWN_003 -> "AUTOLOOP-DOWN-003"
    AUTOLOOP_RESET_2026_05_30 -> "AUTOLOOP-RESET-2026-05-30"
    AUTOLOOP_SINGLETON_001 -> "AUTOLOOP-SINGLETON-001"
    AUTOLOOP_STALL_001 -> "AUTOLOOP-STALL-001"
    BINARY_HANG_001 -> "BINARY-HANG-001"
    CIO_DEAFNESS_001 -> "CIO-DEAFNESS-001"
    EXECUTION_DATASET_001 -> "EXECUTION-DATASET-001"
    EXECUTION_MISSING_001 -> "EXECUTION-MISSING-001"
    EXPECTANCY_INVALID_001 -> "EXPECTANCY-INVALID-001"
    FEATURE_MISSINGNESS_001 -> "FEATURE-MISSINGNESS-001"
    GITHUB_502_001 -> "GITHUB-502-001"
    KALMAN_NUMSTAB_001 -> "KALMAN-NUMSTAB-001"
    LEVERAGE_INVALID_001 -> "LEVERAGE-INVALID-001"
    LEVERAGE_SANITY_001 -> "LEVERAGE-SANITY-001"
    LOSS_STREAK_LIMIT_INVALID_001 -> "LOSS-STREAK-LIMIT-INVALID-001"
    MARKET_DATA_TIMESTAMP_OVERFLOW_001 -> "MARKET-DATA-TIMESTAMP-OVERFLOW-001"
    MAX_POSITION_GUARDRAIL_001 -> "MAX-POSITION-GUARDRAIL-001"
    PREDICTOR_IDENTITY_001 -> "PREDICTOR-IDENTITY-001"
    RESEARCH_RECEIPT_001 -> "RESEARCH-RECEIPT-001"
    RISK_LIMIT_001 -> "RISK-LIMIT-001"
    RISK_LIMIT_NON_FINITE_001 -> "RISK-LIMIT-NON-FINITE-001"
    RISK_METRIC_INVALID_001 -> "RISK-METRIC-INVALID-001"
    SCHEMA_001 -> "SCHEMA-001"
    THRESHOLD_FACTOR_001 -> "THRESHOLD-FACTOR-001"
    TRADE_LOG_GAP_001 -> "TRADE-LOG-GAP-001"
    TRADE_LOG_GAP_002 -> "TRADE-LOG-GAP-002"
    TRAILING_STOP_001 -> "TRAILING-STOP-001"
    VOL_TARGET_001 -> "VOL-TARGET-001"
    VOL_TARGET_INVALID_001 -> "VOL-TARGET-INVALID-001"
    ZERO_VIABLE_SIGNAL_001 -> "ZERO-VIABLE-SIGNAL-001"

riskEntry :: RiskID -> RiskSeverity -> RiskStatus -> T.Text -> T.Text -> T.Text -> RiskEntry
riskEntry = RiskEntry

-- | Typed projection of the canonical risk register, sorted by canonical ID.
riskRegister :: [RiskEntry]
riskRegister =
    [ riskEntry
        AUTOLOOP_DOWN_003
        CRITICAL
        OPEN
        "Autoloop process was not alive at the last recorded operational review"
        "trader-firm-cto"
        "Restart or diagnose the supervisor and replace stale operational evidence with a fresh health witness"
    , riskEntry
        AUTOLOOP_RESET_2026_05_30
        CRITICAL
        OPEN
        "Autoloop cycle counter reset and broke continuity assumptions"
        "trader-firm-cto"
        "Establish and verify durable monotone cycle identity across supervisor restarts"
    , riskEntry
        AUTOLOOP_SINGLETON_001
        HIGH
        OPEN
        "Multiple autoloop instances may race on the same repository"
        "trader-firm-cto"
        "Enforce one process with a verified lock and stale-owner recovery"
    , riskEntry
        AUTOLOOP_STALL_001
        CRITICAL
        CLOSED
        "Autoloop stall detection depended on manual observation"
        "trader-firm-cto"
        "Heartbeat telemetry and a bounded stale-heartbeat alert are implemented"
    , riskEntry
        BINARY_HANG_001
        MEDIUM
        MITIGATED
        "The trader binary could hang while draining after a termination signal"
        "trader-firm-cto"
        "Bounded shutdown is implemented; close after a serve-mode PostgreSQL subprocess witness"
    , riskEntry
        CIO_DEAFNESS_001
        CRITICAL
        OPEN
        "The CIO reporting lane missed recorded deadlines"
        "trader-firm-ceo"
        "Obtain a current owner report and explicitly close or reassign the operational obligation"
    , riskEntry
        EXECUTION_DATASET_001
        MEDIUM
        OPEN
        "Backtest dataset generation is not fully reproducible"
        "trader-firm-data"
        "Seed randomness and record the source dataset hash in test output"
    , riskEntry
        EXECUTION_MISSING_001
        CRITICAL
        OPEN
        "The execution reporting lane missed recorded trade-log deadlines"
        "trader-firm-execution"
        "Obtain a current execution report and explicitly close or reassign the operational obligation"
    , riskEntry
        EXPECTANCY_INVALID_001
        CRITICAL
        CLOSED
        "Missing or non-finite expectancy could bypass a configured minimum"
        "trader-firm-risk"
        "specRiskHalt emits EXPECTANCY_INVALID and bounded verification covers the invariant"
    , riskEntry
        FEATURE_MISSINGNESS_001
        HIGH
        OPEN
        "Optional predictor features can encode unavailable evidence as the same numeric zero as an observed value"
        "trader-firm-research"
        "Preserve the new derivatives first-seen ledgers, define equivalent policies for remaining sources, and migrate production feature builders and artifacts under explicit compatibility versioning before promotion eligibility"
    , riskEntry
        GITHUB_502_001
        MEDIUM
        OPEN
        "Transient GitHub API failures can interrupt automation"
        "trader-firm-cto"
        "Bounded exponential retry is implemented; replace stale outage evidence with current operational validation"
    , riskEntry
        KALMAN_NUMSTAB_001
        MEDIUM
        OPEN
        "Kalman numerical instability could produce zero trades or hangs"
        "trader-firm-cto"
        "Covariance initialization and standard-deviation flooring are implemented; obtain current operational validation"
    , riskEntry
        LEVERAGE_INVALID_001
        CRITICAL
        CLOSED
        "Malformed leverage configuration could bypass position-size protection"
        "trader-firm-risk"
        "specRiskHalt emits LEVERAGE_INVALID and live futures leverage is capped"
    , riskEntry
        LEVERAGE_SANITY_001
        CRITICAL
        CLOSED
        "Corrupted or absurd venue leverage evidence could bypass size limits"
        "trader-firm-risk"
        "Venue leverage is sanity-checked at 125x and configuration validation rejects malformed values"
    , riskEntry
        LOSS_STREAK_LIMIT_INVALID_001
        CRITICAL
        CLOSED
        "A negative loss-streak limit could silently disable protection"
        "trader-firm-risk"
        "specRiskHalt rejects negative limits while preserving zero as the documented disabled boundary"
    , riskEntry
        MARKET_DATA_TIMESTAMP_OVERFLOW_001
        CRITICAL
        CLOSED
        "Timestamp arithmetic overflow could make stale or discontinuous evidence appear valid"
        "trader-firm-data"
        "Checked arithmetic fails closed in freshness, continuation, normalization, and continuity validation"
    , riskEntry
        MAX_POSITION_GUARDRAIL_001
        HIGH
        CLOSED
        "Malformed maximum-position configuration could silently disable every trade"
        "trader-firm-risk"
        "Checked simulation rejects non-positive or non-finite maximum-position configuration"
    , riskEntry
        PREDICTOR_IDENTITY_001
        HIGH
        MITIGATED
        "Legacy TCN, PatchTST, and Transformer identifiers can overstate the fidelity of lightweight proxy implementations"
        "trader-firm-research"
        "Preserve legacy semantics, expose accurate versioned implementation identities, and require a new model ID for any faithful architecture"
    , riskEntry
        RESEARCH_RECEIPT_001
        HIGH
        CLOSED
        "A metadata-only derivatives receipt could diverge from its frozen external archive or imply unauthorized outcome access"
        "trader-firm-research"
        "The schema-1 verifier binds the exact status and complete archive inventory while enforcing acquisition-only authority"
    , riskEntry
        RISK_LIMIT_001
        HIGH
        CLOSED
        "Daily, weekly, and drawdown limits were not enforced in the live loop"
        "trader-firm-risk"
        "Runtime spec-coupled invariant checks and guardrail regressions are implemented"
    , riskEntry
        RISK_LIMIT_NON_FINITE_001
        CRITICAL
        CLOSED
        "Non-finite risk limits could silently disable halt checks"
        "trader-firm-risk"
        "specRiskHalt emits RISK_LIMIT_NON_FINITE and bounded verification covers the invariant"
    , riskEntry
        RISK_METRIC_INVALID_001
        CRITICAL
        CLOSED
        "Malformed loss or drawdown evidence could bypass live halt checks"
        "trader-firm-risk"
        "specRiskHalt emits RISK_METRIC_INVALID before threshold comparisons"
    , riskEntry
        SCHEMA_001
        CRITICAL
        CLOSED
        "Live trade-log schema could drift from its declared contract"
        "trader-firm-cio"
        "The schema contract and executable validation are tracked"
    , riskEntry
        THRESHOLD_FACTOR_001
        MEDIUM
        OPEN
        "thresholdFactor may not be wired into simulation configuration"
        "trader-firm-research"
        "Confirm the research contract and add a simulation integration witness"
    , riskEntry
        TRADE_LOG_GAP_001
        HIGH
        CLOSED
        "Trade-log records lacked required exit and halt evidence"
        "trader-firm-cio"
        "The schema contract includes exit_reason and the trade-log implementation is tracked"
    , riskEntry
        TRADE_LOG_GAP_002
        MEDIUM
        OPEN
        "Trade logs lack a native snapshot of derived risk-state metrics"
        "trader-firm-cio"
        "Define whether schema or deterministic derivation owns the risk-state snapshot"
    , riskEntry
        TRAILING_STOP_001
        MEDIUM
        OPEN
        "A trailing-stop exit may re-enter on the same bar"
        "trader-firm-execution"
        "Add and verify a bar-level re-entry lock after trailing-stop exits"
    , riskEntry
        VOL_TARGET_001
        CRITICAL
        OPEN
        "A stateful volatility-target regression was reported in the Haskell test suite"
        "trader-firm-cto"
        "Reproduce against the current canonical Haskell wrapper and close or update this stale report"
    , riskEntry
        VOL_TARGET_INVALID_001
        CRITICAL
        CLOSED
        "Malformed volatility-target configuration could bypass scaling limits"
        "trader-firm-risk"
        "specRiskHalt emits VOL_TARGET_INVALID and bounded verification covers the invariant"
    , riskEntry
        ZERO_VIABLE_SIGNAL_001
        CRITICAL
        OPEN
        "No strategy signal had met the recorded long-sample viability threshold"
        "trader-firm-research"
        "Run and record a current long-dataset viability evaluation"
    ]

-- | Lookup severity by risk ID.
riskSeverityOf :: RiskID -> Maybe RiskSeverity
riskSeverityOf rid = reSeverity <$> lookupRisk rid

-- | Lookup lifecycle status by risk ID.
riskStatusOf :: RiskID -> Maybe RiskStatus
riskStatusOf rid = reStatus <$> lookupRisk rid

lookupRisk :: RiskID -> Maybe RiskEntry
lookupRisk rid = lookup rid (map (\entry -> (reId entry, entry)) riskRegister)
