export type Market = "spot" | "margin" | "futures";
export type Method = "11" | "10" | "01";
export type Normalization = "none" | "minmax" | "standard" | "log";

export type DirectionLabel = "UP" | "DOWN" | null;

export type ApiError = { error: string };

export type ApiParams = {
  data?: string;
  priceColumn?: string;
  binanceSymbol?: string;
  market?: Market;
  interval?: string;
  bars?: number;
  lookbackWindow?: string;
  lookbackBars?: number;
  binanceTestnet?: boolean;
  normalization?: Normalization;
  hiddenSize?: number;
  epochs?: number;
  lr?: number;
  valRatio?: number;
  backtestRatio?: number;
  patience?: number;
  gradClip?: number;
  seed?: number;
  kalmanDt?: number;
  kalmanProcessVar?: number;
  kalmanMeasurementVar?: number;
  threshold?: number;
  method?: Method;
  optimizeOperations?: boolean;
  sweepThreshold?: boolean;
  fee?: number;
  periodsPerYear?: number;
  binanceLive?: boolean;
  orderQuote?: number;
  orderQuantity?: number;

  // Live bot (stateful) options
  botPollSeconds?: number;
  botOnlineEpochs?: number;
  botTrainBars?: number;
  botMaxPoints?: number;
  botTrade?: boolean;
};

export type LatestSignal = {
  method: Method;
  currentPrice: number;
  threshold: number;
  kalmanNext: number | null;
  kalmanDirection: DirectionLabel;
  lstmNext: number | null;
  lstmDirection: DirectionLabel;
  chosenDirection: DirectionLabel;
  action: string;
};

export type ApiOrderResult = {
  sent: boolean;
  mode?: string;
  side?: string;
  symbol?: string;
  quantity?: number;
  quoteQuantity?: number;
  response?: string;
  message: string;
};

export type ApiTradeResponse = {
  signal: LatestSignal;
  order: ApiOrderResult;
};

export type Trade = {
  entryIndex: number;
  exitIndex: number;
  entryEquity: number;
  exitEquity: number;
  return: number;
  holdingPeriods: number;
};

export type BacktestResponse = {
  split: {
    train: number;
    backtest: number;
    backtestRatio: number;
    backtestStartIndex: number;
  };
  method: Method;
  threshold: number;
  metrics: {
    finalEquity: number;
    totalReturn: number;
    annualizedReturn: number;
    annualizedVolatility: number;
    sharpe: number;
    maxDrawdown: number;
    tradeCount: number;
    roundTrips: number;
    winRate: number;
    profitFactor: number;
    avgTradeReturn: number;
    avgHoldingPeriods: number;
    exposure: number;
    agreementRate: number;
    turnover: number;
  };
  latestSignal: LatestSignal;
  equityCurve: number[];
  prices: number[];
  positions: number[];
  agreementOk: boolean[];
  trades: Trade[];
};

export type BotOperation = {
  index: number;
  side: "BUY" | "SELL";
  price: number;
};

export type BotOrderEvent = {
  index: number;
  opSide: "BUY" | "SELL";
  price: number;
  openTime: number;
  atMs: number;
  order: ApiOrderResult;
};

export type BotStatus =
  | { running: false }
  | {
      running: true;
      symbol: string;
      interval: string;
      market: Market;
      method: Method;
      threshold: number;
      startIndex: number;
      startedAtMs: number;
      updatedAtMs: number;
      prices: number[];
      openTimes: number[];
      kalmanPredNext: Array<number | null>;
      lstmPredNext: Array<number | null>;
      equityCurve: number[];
      positions: number[];
      operations: BotOperation[];
      orders: BotOrderEvent[];
      trades: Trade[];
      latestSignal: LatestSignal;
      lastOrder?: ApiOrderResult;
      error?: string;
    };
