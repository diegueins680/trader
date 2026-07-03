import { TRADE_PNL_EPS } from "./appHelpers";
import { fmtNum } from "../lib/format";

const TRADE_PNL_EPS_LABEL = fmtNum(TRADE_PNL_EPS, 9);

export const ACCOUNT_TRADE_PNL_TIPS = {
  outcomes: [
    "Counts non-zero exchange realized P&L rows as closed outcomes.",
    `Win/loss if |P&L| > ${TRADE_PNL_EPS_LABEL}.`,
    "Zero-PNL fills are usually opens and are excluded from win rate.",
  ],
  avgWinLoss: [
    "Average realized P&L per winning/losing outcome.",
    "Payoff ratio = avg win / |avg loss|.",
    "Fees are shown separately.",
  ],
  bestWorst: [
    "Largest positive/negative realized P&L outcome in this set.",
    "Avg outcome P&L = mean realized P&L across scored outcomes.",
  ],
  totalPnl: [
    "Total P&L = sum of realized P&L across scored outcomes.",
    "Total win/loss = sum of positive/negative realized P&L.",
    "Profit factor = gross wins / |gross losses|.",
    "Fees are listed by asset (not subtracted from totals).",
  ],
  totals: [
    "Qty = sum of filtered fill quantities.",
    "Quote = sum of filtered fill quote quantities.",
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
