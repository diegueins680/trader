"""Versioned, disabled research target admission; no live-order capability.

Only a turnover-rejected preview is replaced. Both previews use the same
immutable starting session. Actual economic termination is never suppressed.
"""
from dataclasses import replace
from fractions import Fraction as F
import rl_composition as v1

VERSION = "rl-admission-continuation-v2"
OBSERVATION = v1.OBSERVATION


class Episode(v1.Episode):
    def _execute(self, bar, funding, applied, terminal, multiplier, fill):
        pair = super()._execute(bar, funding, applied, terminal, multiplier, fill)
        v1.require(pair is not None, "replay rejected (including rational resource limit)")
        rejected = pair[1].reason == "turnover_limit"
        final_target = applied
        if rejected:
            # No assignment to session, no debit and no time advancement from
            # the rejected preview. The exact kernel revalidates from scratch.
            final_target = F(0)
            pair = v1.advance_owned(self.session, bar, funding, final_target,
                                   terminal=terminal, multiplier=multiplier,
                                   fill=fill, enabled=True)
            v1.require(pair is not None and pair[1].reason != "turnover_limit",
                       "neutral admission failed")
        receipt = pair[1]
        self._admission = {
            "dueTarget": None if applied is None else float(applied),
            "finalTarget": None if final_target is None else float(final_target),
            "turnoverRejected": rejected,
            "unitsBefore": float(receipt.before.units),
            "unitsAfter": float(receipt.after.units),
            "priceBefore": float(receipt.before.price),
            "priceAfter": float(receipt.after.price),
            "fundingPerUnit": float(sum((mark * rate for mark, rate in funding), F(0))),
            "fillFraction": float(fill), "costMultiplier": float(multiplier),
        }
        return pair

    def step(self, action):
        result = super().step(action)
        row = self.rows[-1]
        row.update(self._admission)
        row["legacyReasonFlag"] = row["intervention"]
        row["supportChangedAction"] = row["permitted"] != row["proposal"]
        row["actionChanged"] = row["supportChangedAction"] or row["turnoverRejected"]
        row["schedulingDeferred"] = ":reserved" in row["reason"]
        row["newProposalIgnored"] = "new_proposal_ignored" in row["reason"]
        row["applied"] = row["finalTarget"]
        row["intervention"] = row["actionChanged"]
        if row["turnoverRejected"]:
            row["reason"] += ":turnover_rejected_zero_target"
        return result


def train(*args, **kwargs):
    fit = v1.train(*args, **kwargs)
    return replace(fit, metadata={**fit.metadata, "environment": VERSION})
