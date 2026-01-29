export type Market = "spot" | "margin" | "futures";
export type Platform = "binance" | "coinbase" | "kraken" | "poloniex";
export type Method = "11" | "10" | "01" | "blend" | "router";
export type Normalization = "none" | "minmax" | "standard" | "log";
export type Positioning = "long-flat" | "long-short";
export type IntrabarFill = "stop-first" | "take-profit-first";

export type DirectionLabel = "UP" | "DOWN" | null;

export type ApiError = { error: string; hint?: string | null; errors?: Array<{ symbol: string; error: string }> };

export type ApiParams = {
  data?: string;
  priceColumn?: string;
  binanceSymbol?: string;
  botSymbols?: string[];
  platform?: Platform;
  market?: Market;
  interval?: string;
  bars?: number;
  lookbackWindow?: string;
  lookbackBars?: number;
  binanceTestnet?: boolean;
  binanceApiKey?: string;
  binanceApiSecret?: string;
  coinbaseApiKey?: string;
  coinbaseApiSecret?: string;
  coinbaseApiPassphrase?: string;
  tenantKey?: string;
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
  walkForwardEmbargoBars?: number;
  patience?: number;
  gradClip?: number;
  seed?: number;
  kalmanDt?: number;
  kalmanProcessVar?: number;
  kalmanMeasurementVar?: number;
  predictors?: string;
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
  rebalanceBars?: number;
  rebalanceThreshold?: number;
  rebalanceCostMult?: number;
  rebalanceGlobal?: boolean;
  rebalanceResetOnSignal?: boolean;
  fundingRate?: number;
  fundingBySide?: boolean;
  fundingOnOpen?: boolean;
  blendWeight?: number;
  routerLookback?: number;
  routerMinScore?: number;
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
  botProtectionOrders?: boolean;
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
  tenantKey?: string;
  signed?: BinanceProbe;
  tradeTest?: BinanceProbe;
};

export type CoinbaseKeysStatus = {
  hasApiKey: boolean;
  hasApiSecret: boolean;
  hasApiPassphrase: boolean;
  tenantKey?: string;
  signed?: BinanceProbe;
};

export type BinanceListenKeyResponse = {
  listenKey: string;
  market: Market;
  testnet: boolean;
  wsUrl: string;
  keepAliveMs: number;
};

export type BinanceListenKeyKeepAliveResponse = { ok: boolean; atMs: number };

export type BinanceTrade = {
  symbol: string;
  tradeId: number;
  orderId?: number | null;
  price: number;
  qty: number;
  quoteQty: number;
  commission?: number | null;
  commissionAsset?: string | null;
  time: number;
  isBuyer?: boolean | null;
  isMaker?: boolean | null;
  side?: string | null;
  positionSide?: string | null;
  realizedPnl?: number | null;
};

export type BinancePosition = {
  symbol: string;
  positionAmt: number;
  entryPrice: number;
  markPrice: number;
  unrealizedPnl: number;
  liquidationPrice?: number | null;
  breakEvenPrice?: number | null;
  leverage?: number | null;
  marginType?: string | null;
  positionSide?: string | null;
};

export type BinancePositionChart = {
  symbol: string;
  openTimes: number[];
  prices: number[];
};

export type ApiBinancePositionsRequest = {
  market?: Market;
  binanceTestnet?: boolean;
  binanceApiKey?: string;
  binanceApiSecret?: string;
  tenantKey?: string;
  interval?: string;
  limit?: number;
};

export type ApiBinancePositionsResponse = {
  market: Market;
  testnet: boolean;
  interval: string;
  limit: number;
  positions: BinancePosition[];
  charts: BinancePositionChart[];
  fetchedAtMs: number;
  accountUid?: number;
};

export type ApiBinanceTradesRequest = {
  market?: Market;
  binanceTestnet?: boolean;
  binanceApiKey?: string;
  binanceApiSecret?: string;
  tenantKey?: string;
  symbol?: string;
  symbols?: string[];
  limit?: number;
  startTimeMs?: number;
  endTimeMs?: number;
  fromId?: number;
};

export type ApiBinanceTradesResponse = {
  market: Market;
  testnet: boolean;
  symbols: string[];
  allSymbols: boolean;
  trades: BinanceTrade[];
  fetchedAtMs: number;
};

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
  rebalanceBars?: number;
  rebalanceThreshold?: number;
  rebalanceCostMult?: number;
  rebalanceGlobal?: boolean;
  rebalanceResetOnSignal?: boolean;
  fundingRate?: number;
  fundingBySide?: boolean;
  fundingOnOpen?: boolean;
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
    walkForwardEmbargoBars?: number;
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
  openTimes?: number[] | null;
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
  settings?: {
    pollSeconds: number;
    onlineEpochs: number;
    trainBars: number;
    maxPoints: number;
    tradeEnabled: boolean;
    protectionOrders?: boolean;
    adoptExistingPosition?: boolean;
  };
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
  startingReason?: string;
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

export type BotStatusSingle = BotStatusRunning | BotStatusStopped;

export type BotStatusMulti = {
  running: boolean;
  starting?: boolean;
  multi: true;
  bots: BotStatusSingle[];
  errors?: Array<{ symbol: string; error: string }>;
  snapshotAtMs?: number;
};

export type BotStatus = BotStatusSingle | BotStatusMulti;

export type BotStatusSnapshot = {
  savedAtMs: number;
  status: BotStatusSingle;
};

export type StateSyncPayload = {
  generatedAtMs?: number;
  botSnapshots?: BotStatusSnapshot[];
  topCombos?: unknown;
};

export type StateSyncImportResponse = {
  ok: boolean;
  atMs: number;
  botSnapshots?: {
    incoming: number;
    existing: number;
    merged: number;
    written: number;
    skipped: number;
  };
  topCombos?: {
    action: "replaced" | "kept" | "skipped";
    incomingGeneratedAtMs?: number;
    localGeneratedAtMs?: number;
  };
};

export type OpsOperation = {
  id: number;
  atMs: number;
  kind: string;
  params?: unknown;
  args?: unknown;
  result?: unknown;
  equity?: number;
};

export type OpsResponse = {
  enabled: boolean;
  hint?: string;
  latestId?: number;
  maxInMemory?: number;
  ops: OpsOperation[];
};

export type PerformanceCommitDelta = {
  gitCommitId: number;
  commitHash?: string;
  committedAtMs?: number;
  startAtMs?: number;
  endAtMs?: number;
  symbols?: number;
  combos?: number;
  rollups?: number;
  avgReturn?: number;
  medianReturn?: number;
  minReturn?: number;
  maxReturn?: number;
  avgDrawdown?: number;
  medianDrawdown?: number;
  worstDrawdown?: number;
  statusPoints?: number;
  orderCount?: number;
  samplePoints?: number;
  updatedAtMs?: number;
  prevCommitHash?: string;
  prevMedianReturn?: number;
  deltaMedianReturn?: number;
  prevMedianDrawdown?: number;
  deltaMedianDrawdown?: number;
  prevWorstDrawdown?: number;
  deltaWorstDrawdown?: number;
};

export type PerformanceComboDelta = {
  gitCommitId: number;
  commitHash?: string;
  committedAtMs?: number;
  symbol?: string;
  market?: string;
  interval?: string;
  comboUuid?: string;
  startAtMs?: number;
  endAtMs?: number;
  firstEquity?: number;
  lastEquity?: number;
  return?: number;
  maxDrawdown?: number;
  statusPoints?: number;
  orderCount?: number;
  samplePoints?: number;
  updatedAtMs?: number;
  prevCommitHash?: string;
  prevReturn?: number;
  deltaReturn?: number;
  prevMaxDrawdown?: number;
  deltaDrawdown?: number;
};

export type OpsPerformanceResponse = {
  enabled: boolean;
  ready: boolean;
  commitsReady: boolean;
  combosReady: boolean;
  hint?: string;
  updatedAtMs?: number;
  commits: PerformanceCommitDelta[];
  combos: PerformanceComboDelta[];
};

export type OptimizerSource = "binance" | "coinbase" | "kraken" | "poloniex" | "csv";

export type OptimizerRunRequest = {
  source?: OptimizerSource;
  binanceSymbol?: string;
  data?: string;
  priceColumn?: string;
  highColumn?: string;
  lowColumn?: string;
  intervals?: string;
  platforms?: string;
  lookbackWindow?: string;
  barsMin?: number;
  barsMax?: number;
  barsAutoProb?: number;
  barsDistribution?: "uniform" | "log";
  epochsMin?: number;
  epochsMax?: number;
  hiddenSizeMin?: number;
  hiddenSizeMax?: number;
  lrMin?: number;
  lrMax?: number;
  patienceMax?: number;
  gradClipMin?: number;
  gradClipMax?: number;
  pDisableGradClip?: number;
  trials?: number;
  timeoutSec?: number;
  seed?: number;
  seedTrials?: number;
  seedRatio?: number;
  survivorFraction?: number;
  perturbScaleDouble?: number;
  perturbScaleInt?: number;
  earlyStopNoImprove?: number;
  slippageMax?: number;
  spreadMax?: number;
  normalizations?: string;
  backtestRatio?: number;
  tuneRatio?: number;
  objective?: string;
  penaltyMaxDrawdown?: number;
  penaltyTurnover?: number;
  minAnnualizedReturn?: number;
  minCalmar?: number;
  maxTurnover?: number;
  minRoundTrips?: number;
  minWinRate?: number;
  minProfitFactor?: number;
  minExposure?: number;
  minSharpe?: number;
  minWalkForwardSharpeMean?: number;
  maxWalkForwardSharpeStd?: number;
  tuneObjective?: string;
  tunePenaltyMaxDrawdown?: number;
  tunePenaltyTurnover?: number;
  walkForwardFoldsMin?: number;
  walkForwardFoldsMax?: number;
  walkForwardEmbargoBarsMin?: number;
  walkForwardEmbargoBarsMax?: number;
  minHoldBarsMin?: number;
  minHoldBarsMax?: number;
  cooldownBarsMin?: number;
  cooldownBarsMax?: number;
  maxHoldBarsMin?: number;
  maxHoldBarsMax?: number;
  minEdgeMin?: number;
  minEdgeMax?: number;
  minSignalToNoiseMin?: number;
  minSignalToNoiseMax?: number;
  edgeBufferMin?: number;
  edgeBufferMax?: number;
  pCostAwareEdge?: number;
  trendLookbackMin?: number;
  trendLookbackMax?: number;
  rebalanceCostMultMin?: number;
  rebalanceCostMultMax?: number;
  stopMin?: number;
  stopMax?: number;
  tpMin?: number;
  tpMax?: number;
  trailMin?: number;
  trailMax?: number;
  methodWeightBlend?: number;
  blendWeightMin?: number;
  blendWeightMax?: number;
  disableLstmPersistence?: boolean;
  noSweepThreshold?: boolean;
} & Record<string, unknown>;

export type OptimizerRunResponse = {
  lastRecord: unknown;
  stdout: string;
  stderr: string;
};
