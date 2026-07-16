import type { Method as ContractMethod, Platform as ContractPlatform } from "../app/contracts";

export type Market = "spot" | "margin" | "futures";
export type Platform = ContractPlatform;
export type Method = ContractMethod;
export type Normalization = "none" | "minmax" | "standard" | "log";
export type Positioning = "long-flat" | "long-short";
export type IntrabarFill = "stop-first" | "take-profit-first";

export type DirectionLabel = "UP" | "DOWN" | null;
export type DecisionTraceStatus = "ok" | "warn" | "bad" | "skip";
export type DecisionTraceStage = {
  id: string;
  label: string;
  status: DecisionTraceStatus;
  detail: string;
};
export type DecisionTrace = {
  outcome: "operate" | "hold";
  summary: string;
  reason?: string | null;
  stages: DecisionTraceStage[];
};

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
  sensorVarianceEwmaAlpha?: number;
  kalmanSensorCorrelationInflation?: number;
  kalmanInnovationInflationThreshold?: number;
  kalmanInnovationInflationMax?: number;
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
  feeFixed?: number;
  feeMin?: number;
  slippageVolMult?: number;
  slippageImpact?: number;
  slippageImpactPower?: number;
  spreadVolMult?: number;
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
  maxOpenPositions?: number;
  maxOpenPerBase?: number;
  maxGrossExposure?: number;
  maxNetExposure?: number;
  maxExposurePerBase?: number;
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
  routerRegimeMinBars?: number;
  routerRegimeMinFraction?: number;
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
  predictionMarketHerd?: boolean;
  predictionMarketHerdMinProbability?: number;
  predictionMarketHerdMaxBoost?: number;
  predictionMarketHerdMinVolume?: number;
  predictionMarketHerdLimit?: number;
  predictionMarketHerdFreshTtlSec?: number;
  predictionMarketHerdStaleTtlSec?: number;
  predictionMarketHerdScoreBase?: number;
  predictionMarketHerdIntervalMatchBonus?: number;
  predictionMarketHerdTimeDecayBonus?: number;
  predictionMarketHerdPastEndPenalty?: number;
  predictionMarketHerdVolumeScoreWeight?: number;

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
  botNeuralGovernorEnabled?: boolean;
  botNeuralGovernorHiddenSize?: number;
  botNeuralGovernorLearningRate?: number;
  botNeuralGovernorL2?: number;
  botNeuralGovernorRewardClip?: number;
  botNeuralGovernorLossPenaltyScale?: number;
  botNeuralGovernorMinTrades?: number;
  botNeuralGovernorOpenScoreFloor?: number;
  botNeuralGovernorHoldScoreFloor?: number;
  botNeuralGovernorMinMultiplier?: number;
  botNeuralGovernorMaxMultiplier?: number;
  botNeuralGovernorInfluence?: number;
  botNeuralGovernorSeed?: number;
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
  sizingNext?: number | null;
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
  originIp?: string | null;
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
  egressIp?: string;
  tenantKey?: string;
  signed?: BinanceProbe;
  tradeTest?: BinanceProbe;
};

export type ApiRequestProgressStatus = {
  requestId: string;
  kind: string;
  currentPhase: string;
  lastCompletedPhase?: string | null;
  detail?: string | null;
  startedAtMs: number;
  updatedAtMs: number;
  completedAtMs?: number | null;
  completedOk?: boolean | null;
  error?: string | null;
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
  originIp?: string | null;
  executorIp?: string | null;
  originInstance?: string | null;
  method?: string | null;
  strategy?: string | null;
  decisionSummary?: string | null;
  decisionReason?: string | null;
  entryIp?: string | null;
  exitIp?: string | null;
  entryInstance?: string | null;
  exitInstance?: string | null;
  entryTime?: number | null;
  exitTime?: number | null;
  entryMethod?: string | null;
  exitMethod?: string | null;
  entryStrategy?: string | null;
  exitStrategy?: string | null;
  entryDecisionSummary?: string | null;
  exitDecisionSummary?: string | null;
  entryDecisionReason?: string | null;
  exitDecisionReason?: string | null;
  maxPnl?: number | null;
  maxPnlCloseTime?: number | null;
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
  /** Persisted system bot that opened this position, when known. */
  botId?: number | null;
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

export type ApiBinanceClosePositionRequest = {
  market?: Market;
  binanceTestnet?: boolean;
  binanceLive?: boolean;
  binanceApiKey?: string;
  binanceApiSecret?: string;
  tenantKey?: string;
  symbol: string;
  positionSide?: string;
  positionAmt?: number;
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
  stale?: boolean;
  source?: string;
  error?: string;
};

export type ApiBinanceTradesRequest = {
  market?: Market;
  binanceTestnet?: boolean;
  binanceApiKey?: string;
  binanceApiSecret?: string;
  tenantKey?: string;
  symbol?: string;
  symbols?: string[];
  interval?: string;
  limit?: number;
  startTimeMs?: number;
  endTimeMs?: number;
  fromId?: number;
  includeMaxPnl?: boolean;
};

export type ApiBinanceTradesResponse = {
  market: Market;
  testnet: boolean;
  interval?: string;
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
  sortino: number;
  calmar: number;
  downsideVolatility: number;
  var95: number;
  cvar95: number;
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
  entryHighVolProb?: number | null;
  exitReason?: string | null;
  entryIp?: string | null;
  exitIp?: string | null;
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
    feeFixed: number;
    feeMin: number;
    slippageVolMult: number;
    slippageImpact: number;
    slippageImpactPower: number;
    spreadVolMult: number;
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
  latestSignal?: LatestSignal | null;
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

export type BotMarketGovernor = {
  enabled: boolean;
  profile: string;
  entrySizeMultiplier: number;
  blockFreshEntries: boolean;
  reduceOnly: boolean;
  reason?: string | null;
  recommendedVolConfGate?: string | null;
  methodBias?: string[];
  inputs?: {
    marketDataStale?: boolean;
    volatility?: number | null;
    confidence?: number | null;
    trendProbability?: number | null;
    meanReversionProbability?: number | null;
    highVolProbability?: number | null;
    drawdown?: number | null;
    lossStreak?: number;
    rollingLoss?: number | null;
    capitalPreservationReason?: string | null;
  };
};

export type BotNeuralGovernorConfig = {
  enabled?: boolean;
  mode?: string;
  rolloutMode?: string;
  hiddenSize?: number;
  learningRate?: number;
  minTrades?: number;
  openScoreFloor?: number;
  holdScoreFloor?: number;
  minMultiplier?: number;
  maxMultiplier?: number;
  rewardClip?: number;
  lossPenaltyScale?: number;
  influence?: number;
};

export type BotNeuralGovernor = {
  enabled: boolean;
  mode?: string;
  rolloutMode?: string;
  enforced?: boolean;
  promoted?: boolean;
  rolledBack?: boolean;
  examples?: number;
  evaluationTrades?: number;
  ready?: boolean;
  score?: number;
  multiplier?: number;
  counterfactualAdvantage?: number;
  openBlockReason?: string | null;
  holdReason?: string | null;
  candidateOpenBlockReason?: string | null;
  candidateHoldReason?: string | null;
  reason?: string | null;
  lastReward?: number | null;
  config?: BotNeuralGovernorConfig;
  state?: Record<string, unknown>;
};

export type BotPortfolioSelector = {
  mode: "shadow" | "canary" | "enforce";
  selectionValid: boolean;
  evidenceAgeDays?: number | null;
  selection?: {
    generatedAtMs: number;
    validUntilMs: number;
    evidenceStartMs: number;
    evidenceEndMs: number;
    members: Array<{ uuid: string; symbol: string; weight: number }>;
    metrics: {
      annualizedReturnP10: number;
      annualizedReturnP50: number;
      annualizedReturnP90: number;
      maxDrawdownP95: number;
      averageCorrelation: number;
      switchingCost: number;
      pairedOutperformanceProbability: number;
    };
  } | null;
  configuration?: {
    maxBots: number;
    maxBotWeight: number;
    maxGrossWeight: number;
    maxDrawdown: number;
    minimumEvidenceDays: number;
    bootstrapSamples: number;
    bootstrapBlockDays: number;
    rotationImprovementFloor: number;
    rotationProbabilityFloor: number;
  };
};

export type BotStatusRunning = {
  running: true;
  live: boolean;
  tenantKey: string;
  symbol: string;
  interval: string;
  market: Market;
  method: Method;
  threshold: number;
  openThreshold?: number;
  closeThreshold?: number;
  settings: {
    pollSeconds: number;
    onlineEpochs: number;
    trainBars: number;
    maxPoints: number;
    neuralGovernor?: BotNeuralGovernorConfig;
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
  decisionTrace?: DecisionTrace;
  marketGovernor?: BotMarketGovernor;
  neuralGovernor?: BotNeuralGovernor;
  portfolioSelector?: BotPortfolioSelector;
  lastOrder?: ApiOrderResult;
  error?: string;
};

export type BotStatusStopped = {
  running: false;
  starting?: boolean;
  startingReason?: string;
  tenantKey?: string;
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
  portfolioSelector?: BotPortfolioSelector;
};

export type BotStatusSingle = BotStatusRunning | BotStatusStopped;

export type BotStatusMulti = {
  running: boolean;
  starting?: boolean;
  multi: true;
  bots: BotStatusSingle[];
  errors?: Array<{ symbol: string; error: string }>;
  queued?: Array<{ symbol: string; message: string }>;
  snapshotAtMs?: number;
  portfolioSelector?: BotPortfolioSelector;
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
    action: "replaced" | "kept" | "skipped" | "merged";
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
  serverId?: string | null;
  serverRole?: string | null;
  serverProvider?: string | null;
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
  survivorParentActivityFloor?: number;
  survivorParentAnnualizedReturnFloor?: number;
  survivorEdgeWeight?: number;
  survivorRankBias?: number;
  pLongShort?: number;
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
  methodWeight11?: number;
  methodWeight10?: number;
  methodWeight01?: number;
  methodWeightBlend?: number;
  methodWeightConfBlend?: number;
  methodWeightConfPick?: number;
  methodWeightConformalClip?: number;
  methodWeightCostPick?: number;
  methodWeightHarmonicBlend?: number;
  methodWeightDisagreementGuard?: number;
  methodWeightMedianBlend?: number;
  methodWeightNeutralGuard?: number;
  methodWeightRiskParityBlend?: number;
  methodWeightConsensusBoost?: number;
  methodWeightAnchorBlend?: number;
  methodWeightTensionGate?: number;
  methodWeightEntropyBlend?: number;
  methodWeightCoherenceGate?: number;
  methodWeightDivergenceGate?: number;
  methodWeightFractalBlend?: number;
  methodWeightPhaseCancel?: number;
  methodWeightSoftmaxBlend?: number;
  methodWeightSmoothSoftmaxBlend?: number;
  methodWeightHedgeBlend?: number;
  methodWeightNetSoftmaxBlend?: number;
  methodWeightEdgeBlend?: number;
  methodWeightEdgePick?: number;
  methodWeightGeoBlend?: number;
  methodWeightRegimeSwitch?: number;
  methodWeightBanditRouter?: number;
  methodWeightCrossSectionalMomentum?: number;
  blendWeightMin?: number;
  blendWeightMax?: number;
  routerRegimeMinBarsMin?: number;
  routerRegimeMinBarsMax?: number;
  routerRegimeMinFractionMin?: number;
  routerRegimeMinFractionMax?: number;
  correlationGuidanceJson?: string;
  disableLstmPersistence?: boolean;
  noSweepThreshold?: boolean;
} & Record<string, unknown>;

export type OptimizerRunResponse = {
  lastRecord: unknown;
  stdout: string;
  stderr: string;
};
