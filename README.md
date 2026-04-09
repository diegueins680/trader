Add one focused bullet in the ROI/tuning area, near the existing ROI objective description:
- ROI tuning now fails closed on malformed metrics: if annualized return, drawdown, tail loss, turnover, expectancy, payback, activity, or exposure inputs become non-finite or impossible during optimizer scoring, the candidate is normalized to conservative worst-case ROI scoring inputs so degenerate configurations cannot outrank valid ones.

Keep the rest of the README unchanged.