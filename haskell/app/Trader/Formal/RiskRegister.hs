{-# LANGUAGE OverloadedStrings #-}

{- | Risk Register — Trader Firm

This module defines the canonical risk register for the trading system.
It is version-controlled and updated by the Risk Director.

Update rule: every severity change or new risk ID must be committed
to this file and cited in the Risk report.

Last updated: 2026-05-30 23:17 UTC
-}
module Trader.Formal.RiskRegister (
    RiskID (..),
    RiskSeverity (..),
    RiskEntry (..),
    riskRegister,
    riskSeverityOf,
) where

import qualified Data.Text as T

-- | Unique risk identifier.
data RiskID
    = KALMAN_NUMSTAB_001
    | GITHUB_502_001
    | ZERO_VIABLE_SIGNAL_001
    | TRADE_LOG_GAP_001
    | RISK_LIMIT_NON_FINITE_001
    | LEVERAGE_SANITY_001
    | AUTOLOOP_RESET_2026_05_30
    | EXPECTANCY_INVALID_001
    | VOL_TARGET_INVALID_001
    | LEVERAGE_INVALID_001
    | MARKET_DATA_TIMESTAMP_OVERFLOW_001
    | RISK_METRIC_INVALID_001
    | LOSS_STREAK_LIMIT_INVALID_001
    deriving (Eq, Show)

-- | Severity levels.
data RiskSeverity
    = LOW
    | AMBER
    | CRITICAL
    | CLOSED
    deriving (Eq, Show)

-- | Single risk entry.
data RiskEntry = RiskEntry
    { reId :: !RiskID
    , reSeverity :: !RiskSeverity
    , reDescription :: !T.Text
    , reOwner :: !T.Text
    , reMitigation :: !T.Text
    }
    deriving (Eq, Show)

-- | Canonical risk register.
riskRegister :: [RiskEntry]
riskRegister =
    [ RiskEntry
        { reId = KALMAN_NUMSTAB_001
        , reSeverity = LOW
        , reDescription = "Kalman filter numerical stability edge cases"
        , reOwner = "trader-firm-cto"
        , reMitigation = "CTO durability checks passed; monitored via autoloop metrics"
        }
    , RiskEntry
        { reId = GITHUB_502_001
        , reSeverity = LOW
        , reDescription = "GitHub API 502 Bad Gateway transient failures"
        , reOwner = "trader-firm-cto"
        , reMitigation = "CTO durability checks passed; retry with exponential backoff"
        }
    , RiskEntry
        { reId = ZERO_VIABLE_SIGNAL_001
        , reSeverity = AMBER
        , reDescription = "SMA cross signal viable but unconfirmed on long dataset"
        , reOwner = "trader-firm-research"
        , reMitigation = "Executable halt spec landed in Trader.Formal.Risk; awaiting long-dataset confirmation"
        }
    , RiskEntry
        { reId = TRADE_LOG_GAP_001
        , reSeverity = CLOSED
        , reDescription = "Trade-log NDJSON field-order and string-encoded decimal gaps"
        , reOwner = "trader-firm-execution"
        , reMitigation = "Field-order fix and string-encoded decimals landed by CTO/Execution"
        }
    , RiskEntry
        { reId = RISK_LIMIT_NON_FINITE_001
        , reSeverity = CRITICAL
        , reDescription = "Non-finite (NaN/Infinity) risk limits silently disable halt checks"
        , reOwner = "trader-firm-risk"
        , reMitigation = "Executable halt spec landed: specRiskHalt emits RISK_LIMIT_NON_FINITE; invariant proven in verifyFormalRisk"
        }
    , RiskEntry
        { reId = LEVERAGE_SANITY_001
        , reSeverity = CRITICAL
        , reDescription = "Corrupted or absurd leverage configs bypass position-size limits"
        , reOwner = "trader-firm-risk"
        , reMitigation = "futuresPositionRiskLeverageSane caps at 125x; CLI rejects >150x and non-finite"
        }
    , RiskEntry
        { reId = AUTOLOOP_RESET_2026_05_30
        , reSeverity = CRITICAL
        , reDescription = "Autoloop cycle reset (cycle 184 -> 1, gap ~2h10m, cycle 181 exitCode 1)"
        , reOwner = "trader-firm-cto"
        , reMitigation = "Root cause TBD; reset breaks continuity assumptions in readiness memo"
        }
    , RiskEntry
        { reId = EXPECTANCY_INVALID_001
        , reSeverity = CRITICAL
        , reDescription = "Missing or non-finite expectancy when min-expectancy is configured"
        , reOwner = "trader-firm-risk"
        , reMitigation = "Executable halt spec landed: specRiskHalt emits EXPECTANCY_INVALID; invariant proven in verifyFormalRisk"
        }
    , RiskEntry
        { reId = VOL_TARGET_INVALID_001
        , reSeverity = CRITICAL
        , reDescription = "Non-finite, negative, or >1000% vol-target configs bypass vol-scaling limits"
        , reOwner = "trader-firm-risk"
        , reMitigation = "Executable halt spec landed: specRiskHalt emits VOL_TARGET_INVALID; invariant proven in verifyFormalRisk"
        }
    , RiskEntry
        { reId = LEVERAGE_INVALID_001
        , reSeverity = CRITICAL
        , reDescription = "Non-finite, negative, or >150x leverage configs bypass position-size limits and allow catastrophic liquidation risk"
        , reOwner = "trader-firm-risk"
        , reMitigation = "Executable halt spec landed: specRiskHalt emits LEVERAGE_INVALID; invariant proven in verifyFormalRisk; futuresPositionRiskLeverageSane caps live trading at 125x"
        }
    , RiskEntry
        { reId = MARKET_DATA_TIMESTAMP_OVERFLOW_001
        , reSeverity = CLOSED
        , reDescription = "Market-data timestamp arithmetic can overflow Int64 and make stale or discontinuous evidence appear valid"
        , reOwner = "trader-firm-data"
        , reMitigation = "Checked timestamp arithmetic now fails closed in freshness, continuation, normalization, and continuity validation; bounded witnesses run in the formal verification suite"
        }
    , RiskEntry
        { reId = RISK_METRIC_INVALID_001
        , reSeverity = CLOSED
        , reDescription = "Negative or non-finite daily loss, weekly loss, or drawdown evidence can bypass live halt checks"
        , reOwner = "trader-firm-risk"
        , reMitigation = "specRiskHalt emits RISK_METRIC_INVALID before threshold comparisons; bounded witnesses run in the formal verification suite"
        }
    , RiskEntry
        { reId = LOSS_STREAK_LIMIT_INVALID_001
        , reSeverity = CLOSED
        , reDescription = "A negative loss-streak limit can silently disable loss-streak protection"
        , reOwner = "trader-firm-risk"
        , reMitigation = "specRiskHalt emits LOSS_STREAK_LIMIT_INVALID for negative limits while preserving zero as the documented disabled boundary"
        }
    ]

-- | Lookup severity by risk ID.
riskSeverityOf :: RiskID -> Maybe RiskSeverity
riskSeverityOf rid = reSeverity <$> lookup rid (map (\e -> (reId e, e)) riskRegister)
