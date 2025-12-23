export type Market = "spot" | "margin" | "futures";
export type Platform = "binance" | "kraken" | "poloniex";
export type Method = "11" | "10" | "01" | "blend";
export type Normalization = "none" | "minmax" | "standard" | "log";
export type Positioning = "long-flat" | "long-short";
export type IntrabarFill = "stop-first" | "take-profit-first";

export type DirectionLabel = "UP" | "DOWN" | null;

export type ApiError = { error: string; hint?: string | null };

export type ApiParams = {
  data?: string;
  priceColumn?: string;
  binanceSymbol?: string;
  platform?: Platform;
  market?: Market;
  interval?: string;
  bars?: number;
  lookbackWindow?: string;
  lookbackBars?: number;
  binanceTestnet?: boolean;
  binanceApiKey?: string;
  binanceApiSecret?: string;
  normalization?: Normalization;
  hiddenSize?: number;
  epochs?: number;
  lr?: number;
  valRatio?: number;
  backtestRatio?: number;
  tuneRatio?: number;
  tuneObjective?: string;
  tunePenaltyMaxDrawdown?: number;
  tunePenaltyTurnover?: number;
  minRoundTrips?: number;
  walkForwardFolds?: number;
  patience?: number;
  gradClip?: number;
  seed?: number;
  kalmanDt?: number;
  kalmanProcessVar?: number;
  kalmanMeasurementVar?: number;
  threshold?: number; // legacy (maps to open/close)
  openThreshold?: number;
  closeThreshold?: number;
  method?: Method;
  positioning?: Positioning;
  optimizeOperations?: boolean;
  sweepThreshold?: boolean;
  fee?: number;
  slippage?: number;
  spread?: number;
  intrabarFill?: IntrabarFill;
  stopLoss?: number;
  takeProfit?: number;
  trailingStop?: number;
  stopLossVolMult?: number;
  takeProfitVolMult?: number;
  trailingStopVolMult?: number;
  minHoldBars?: number;
  maxHoldBars?: number;
  cooldownBars?: number;
  maxDrawdown?: number;
  maxDailyLoss?: number;
  maxOrderErrors?: number;
  minEdge?: number;
  minSignalToNoise?: number;
  costAwareEdge?: boolean;
  edgeBuffer?: number;
  trendLookback?: number;
  maxPositionSize?: number;
  volTarget?: number;
  volLookback?: number;
  volEwmaAlpha?: number;
  volFloor?: number;
  volScaleMax?: number;
  maxVolatility?: number;
  blendWeight?: number;
  periodsPerYear?: number;
  binanceLive?: boolean;
  orderQuote?: number;
  orderQuantity?: number;
  orderQuoteFraction?: number;
  maxOrderQuote?: number;
  idempotencyKey?: string;
  tuneStressVolMult?: number;
  tuneStressShock?: number;
  tuneStressWeight?: number;

  // Confidence / gating (Kalman sensors + HMM/intervals)
  kalmanZMin?: number;
  kalmanZMax?: number;
  maxHighVolProb?: number;
  maxConformalWidth?: number;
  maxQuantileWidth?: number;
  confirmConformal?: boolean;
  confirmQuantiles?: boolean;
  confidenceSizing?: boolean;
  minPositionSize?: number;

  // Live bot (stateful) options
  botPollSeconds?: number;
  botOnlineEpochs?: number;
  botTrainBars?: number;
  botMaxPoints?: number;
  botTrade?: boolean;
  botAdoptExistingPosition?: boolean;
};

export type LatestSignal = {
  method: Method;
  currentPrice: number;
  threshold: number;
  openThreshold?: number;
  closeThreshold?: number;
  kalmanNext: number | null;
  kalmanReturn?: number | null;
  kalmanStd?: number | null;
  kalmanZ?: number | null;
  volatility?: number | null;
  regimes?: { trend: number; mr: number; highVol: number } | null;
  quantiles?: { q10: number; q50: number; q90: number; width: number } | null;
  conformalInterval?: { lo: number; hi: number; width: number } | null;
  confidence?: number | null;
  positionSize?: number | null;
  kalmanDirection: DirectionLabel;
  lstmNext: number | null;
  lstmDirection: DirectionLabel;
  chosenDirection: DirectionLabel;
  closeDirection?: DirectionLabel;
  action: string;
};

export type ApiOrderResult = {
  sent: boolean;
  mode?: string;
  side?: string;
  symbol?: string;
  quantity?: number;
  quoteQuantity?: number;
  orderId?: number;
  clientOrderId?: string;
  status?: string;
  executedQty?: number;
  cummulativeQuoteQty?: number;
  response?: string;
  message: string;
};

export type ApiTradeResponse = {
  signal: LatestSignal;
  order: ApiOrderResult;
};

export type BinanceProbe = {
  ok: boolean;
  skipped?: boolean;
  step: string;
  code?: number;
  msg?: string;
  summary: string;
};

export type BinanceKeysStatus = {
  market: Market;
  testnet: boolean;
  symbol?: string;
  hasApiKey: boolean;
  hasApiSecret: boolean;
  signed?: BinanceProbe;
  tradeTest?: BinanceProbe;
};

export type BinanceListenKeyResponse = {
  listenKey: string;
  market: Market;
  testnet: boolean;
  wsUrl: string;
  keepAliveMs: number;
};

export type BinanceListenKeyKeepAliveResponse = { ok: boolean; atMs: number };

export type BacktestMetrics = {
  finalEquity: number;
  totalReturn: number;
  annualizedReturn: number;
  annualizedVolatility: number;
  sharpe: number;
  maxDrawdown: number;
  tradeCount: number;
  positionChanges: number;
  roundTrips: number;
  winRate: number;
  grossProfit: number;
  grossLoss: number;
  profitFactor: number | null;
  avgTradeReturn: number;
  avgHoldingPeriods: number;
  exposure: number;
  agreementRate: number;
  turnover: number;
};

export type Trade = {
  entryIndex: number;
  exitIndex: number;
  entryEquity: number;
  exitEquity: number;
  return: number;
  holdingPeriods: number;
  exitReason?: string | null;
};

export type BacktestResponse = {
  split: {
    train: number;
    fit: number;
    tune: number;
    tuneRatio: number;
    tuneStartIndex: number;
    backtest: number;
    backtestRatio: number;
    backtestStartIndex: number;
  };
  method: Method;
  threshold: number;
  openThreshold?: number;
  closeThreshold?: number;
  minHoldBars?: number;
  maxHoldBars?: number | null;
  stopLossVolMult?: number;
  takeProfitVolMult?: number;
  trailingStopVolMult?: number;
  cooldownBars?: number;
  maxPositionSize?: number;
  minEdge?: number;
  minSignalToNoise?: number;
  costAwareEdge?: boolean;
  edgeBuffer?: number;
  trendLookback?: number;
  volTarget?: number | null;
  volLookback?: number;
  volEwmaAlpha?: number | null;
  volFloor?: number;
  volScaleMax?: number;
  maxVolatility?: number | null;
  blendWeight?: number;
  tuning?: {
    objective: string;
    penaltyMaxDrawdown: number;
    penaltyTurnover: number;
    stressVolMult: number;
    stressShock: number;
    stressWeight: number;
    minRoundTrips?: number;
    walkForwardFolds: number;
    tuneStats?: { folds: number; scores: number[]; meanScore: number; stdScore: number } | null;
    tuneMetrics?: BacktestMetrics | null;
  };
  costs?: {
    fee: number;
    slippage: number;
    spread: number;
    perSideCost: number;
    roundTripCost: number;
    breakEvenThreshold: number;
  };
  walkForward?: {
    foldCount: number;
    folds: { startIndex: number; endIndex: number; metrics: BacktestMetrics }[];
    summary: {
      finalEquityMean: number;
      finalEquityStd: number;
      annualizedReturnMean: number;
      annualizedReturnStd: number;
      sharpeMean: number;
      sharpeStd: number;
      maxDrawdownMean: number;
      maxDrawdownStd: number;
      turnoverMean: number;
      turnoverStd: number;
    };
  } | null;
  baselines?: { name: string; metrics: BacktestMetrics }[];
  metrics: BacktestMetrics;
  latestSignal: LatestSignal;
  equityCurve: number[];
  prices: number[];
  kalmanPredNext: Array<number | null>;
  lstmPredNext: Array<number | null>;
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

export type BotKline = {
  openTime: number;
  open: number;
  high: number;
  low: number;
  close: number;
};

export type BotStatusRunning = {
  running: true;
  symbol: string;
  interval: string;
  market: Market;
  method: Method;
  threshold: number;
  openThreshold?: number;
  closeThreshold?: number;
  settings?: { pollSeconds: number; onlineEpochs: number; trainBars: number; maxPoints: number; tradeEnabled: boolean };
  halted: boolean;
  peakEquity: number;
  dayStartEquity: number;
  consecutiveOrderErrors: number;
  cooldownLeft?: number;
  haltReason?: string;
  haltedAtMs?: number;
  startIndex: number;
  startedAtMs: number;
  updatedAtMs: number;
  polledAtMs?: number;
  pollLatencyMs?: number;
  fetchedKlines?: number;
  fetchedLastKline?: BotKline;
  lastBatchAtMs?: number;
  lastBatchSize?: number;
  lastBatchMs?: number;
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

export type BotStatusStopped = {
  running: false;
  starting?: boolean;
  symbol?: string;
  interval?: string;
  market?: Market;
  method?: Method;
  threshold?: number;
  openThreshold?: number;
  closeThreshold?: number;
  startedAtMs?: number;
  error?: string;
  snapshot?: BotStatusRunning;
  snapshotAtMs?: number;
};

export type BotStatus = BotStatusRunning | BotStatusStopped;
