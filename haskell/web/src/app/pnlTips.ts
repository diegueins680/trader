import { TRADE_PNL_EPS } from "./appHelpers";
import { fmtNum } from "../lib/format";

const TRADE_PNL_EPS_LABEL = fmtNum(TRADE_PNL_EPS, 9);

export const ACCOUNT_TRADE_PNL_TIPS = {
  outcomes: [
    "Counts trades by exchange realized P&L (realizedPnl).",
    `Win/loss if |P&L| > ${TRADE_PNL_EPS_LABEL}; otherwise flat.`,
    "Win rate = wins / total trades (after filters).",
  ],
  avgWinLoss: [
    "Average realized P&L per winning/losing trade.",
    "Payoff ratio = avg win / |avg loss|.",
    "Fees are shown separately.",
  ],
  bestWorst: [
    "Largest positive/negative realized P&L in this set.",
    "Avg P&L = mean realized P&L across all trades.",
  ],
  totalPnl: [
    "Total P&L = sum of realized P&L across trades.",
    "Total win/loss = sum of positive/negative realized P&L.",
    "Profit factor = gross wins / |gross losses|.",
    "Fees are listed by asset (not subtracted from totals).",
  ],
  totals: [
    "Qty = sum of trade quantities.",
    "Quote = sum of quote quantities.",
    "Scope follows the current filters/symbol selection.",
  ],
};

export const BACKTEST_TRADE_PNL_TIPS = {
  outcomes: [
    "Counts trades by per-trade return from the backtest engine.",
    `Win/loss if |return| > ${TRADE_PNL_EPS_LABEL}; otherwise flat.`,
    "Returns are shown as % of equity per trade.",
  ],
  avgWinLoss: [
    "Average return per winning/losing trade.",
    "Payoff ratio = avg win / |avg loss|.",
    "Hold bars show average holding periods for wins/losses.",
  ],
  bestWorst: [
    "Largest positive/negative per-trade return.",
    "Avg return = mean return across all trades.",
  ],
  totals: [
    "Total win/loss = sum of per-trade returns (not compounded).",
    "Profit factor = gross wins / |gross losses|.",
  ],
};
