import React from "react";
import type { BotPortfolioSelector } from "../lib/types";

const reasons: Record<string, string> = {
  "automatic-graduation-disabled": "Automatic review is disabled.",
  "graduation-start-time-invalid": "A valid review start time is required.",
  "reviewed-uuid-set-empty": "No strategies are assigned to this review.",
  "daily-observations-below-minimum": "More complete UTC evidence days are required.",
  "net-return-below-minimum": "Net return has not cleared the required minimum.",
  "max-drawdown-above-limit": "Drawdown exceeds the review limit.",
  "execution-attempts-below-minimum": "More order attempts are required.",
  "execution-reliability-below-minimum": "Order submission reliability is below the required level.",
  "status-reliability-below-minimum": "Healthy status coverage is below the required level.",
  "latest-status-unhealthy": "At least one strategy lacks a fresh, healthy live status.",
  "graduation-evidence-unavailable": "Review evidence is unavailable; check boundary baselines and operations persistence.",
  "strict-portfolio-selection-unavailable": "The enforcing selector has no valid selection.",
  "graduation-state-persist-failed": "The review decision could not be saved.",
};

function uuidSet(value: unknown): string[] | null {
  if (!Array.isArray(value) || value.length === 0 || value.length > 256) return null;
  if (!value.every((item) => typeof item === "string" && /^[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$/i.test(item.trim()))) return null;
  return [...new Set(value.map((item: string) => item.trim().toLowerCase()))].sort();
}

function utc(value: unknown): string {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value <= 0) return "unavailable";
  const date = new Date(value);
  return Number.isFinite(date.getTime()) ? date.toISOString().replace(".000Z", " UTC").replace("T", " ") : "unavailable";
}

export function PortfolioGraduationStatus({ selector }: { selector: BotPortfolioSelector }) {
  const graduation = selector.graduation;
  if (!graduation) return <span>Fleet review details unavailable.</span>;
  if (graduation.enabled === false) return <span>Automatic fleet review is disabled.</span>;
  if (graduation.enabled !== true) return <span>Fleet review details unavailable.</span>;
  const fleet = uuidSet(graduation.reviewedUuids);
  const review = graduation.review;
  const reviewed = uuidSet(review?.reviewedUuids);
  const matches = Boolean(
    fleet && reviewed && JSON.stringify(fleet) === JSON.stringify(reviewed)
    && typeof graduation.configVersion === "string" && graduation.configVersion.length > 0
    && review?.configVersion === graduation.configVersion,
  );
  const reviewTimeValid = typeof review?.reviewedAtMs === "number" && Number.isSafeInteger(review.reviewedAtMs)
    && review.reviewedAtMs > 0 && review.reviewedAtMs <= Date.now() + 60_000;
  const graduated = matches && reviewTimeValid && Array.isArray(review?.reasons) && review.reasons.length === 0
    && review.decision === "graduated" && selector.mode === "enforce";
  const dailyCount = matches && Number.isSafeInteger(review?.evidence?.dailyObservationCount) && review!.evidence.dailyObservationCount >= 0
    ? review!.evidence.dailyObservationCount : null;
  const requiredDays = graduation.requirements?.minimumDailyObservations;
  const dayProgress = dailyCount !== null && Number.isSafeInteger(requiredDays) && requiredDays > 0
    ? `${dailyCount}/${requiredDays} complete UTC days` : "Daily evidence count unavailable";
  const currentReasons = matches && Array.isArray(review?.reasons)
    ? [...new Set(review.reasons.filter((reason): reason is string => typeof reason === "string").map((reason) => Object.hasOwn(reasons, reason) ? reasons[reason] : "Another review requirement has not passed."))] : [];
  const maxBots = selector.configuration?.maxBots;
  const gross = selector.configuration?.maxGrossWeight;
  return (
    <div>
      <div>{graduated ? "Graduated" : "Pending review"} · {fleet ? `${fleet.length} reviewed strategies` : "Strategy set unavailable"} · {dayProgress}</div>
      <div>Window starts {utc(graduation.startedAtMs)}. Earliest review {utc(graduation.earliestReviewAtMs)}.</div>
      {Number.isSafeInteger(maxBots) && maxBots! >= 0 && typeof gross === "number" && Number.isFinite(gross) && gross >= 0 && gross <= 1 ? (
        <div>Enforcing selection is capped at {maxBots} bots and {(gross * 100).toFixed(1)}% gross weight.</div>
      ) : null}
      {!matches ? <div>{review ? "The saved review does not match the current strategy set or configuration. Fresh review evidence is required." : "No saved review is available for this fleet."}</div> : (
        <div>Last review {utc(review?.reviewedAtMs)}.</div>
      )}
      {!graduated && currentReasons.length > 0 ? <ul>{currentReasons.map((reason) => <li key={reason}>{reason}</li>)}</ul> : null}
    </div>
  );
}
