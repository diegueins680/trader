# Financial-market prediction evidence review (cutoff 2026-09-04)

## Decision summary

The literature supports several forecastable objects, but it does not support treating a better point forecast as a deployable trading edge. Volatility is more persistent than signed returns; medium-horizon trend and carry recur across asset classes; order-flow imbalance explains very short-horizon price changes; and cross-sectional crypto momentum, size, and attention/network variables have published evidence. The same literature also shows severe instability after publication, large multiple-testing losses, and rapid erosion after realistic costs.

For this repository, the strongest near-term research directions are therefore deliberately modest: a HAR-style volatility forecast used only for risk gating, a depth-normalized order-flow imbalance model contingent on obtaining lawful point-in-time book data, and a missingness-aware regularized distributional model with calibrated abstention. None has fresh confirmation evidence in this repository as of the cutoff. No candidate is approved for integration or promotion, and the current champion is unchanged.

The existing residual-momentum/funding and residual-reversal campaigns are unusually useful negative evidence. Their final holdout stayed sealed, yet the development paths failed economic, confidence, or drawdown gates. In particular, the apparent whole-development winner did not survive nested outer evaluation or cost/delay stress. That result is more decision-relevant than another architecture benchmark on the same contaminated period.

## Scope and method

This review covers publications and publicly available working papers that were available by 2026-09-04. The structured matrix contains 50 works spanning efficiency, econometrics, factors, machine learning, probabilistic forecasting, microstructure, crypto, execution, and statistical safeguards. Sources are primary publications, author repositories, or official proceedings pages. No paper PDF or restricted dataset is committed.

Implementation suitability uses a 100-point score:

| Dimension | Weight | Interpretation |
|---|---:|---|
| Methodological quality | 20 | Identification, validation, and statistical design |
| Independent replication | 15 | Evidence beyond the originating study |
| Leakage and data-snooping control | 10 | Point-in-time construction and multiplicity discipline |
| Transaction-cost realism | 10 | Fees, spread, impact, turnover, latency, and funding |
| Crypto-market relevance | 10 | Transferability to liquid crypto spot/perpetual markets |
| Repository fit | 10 | Incremental value over current code and data |
| Academic influence | 10 | Durable contribution rather than citation count alone |
| Reproducibility | 5 | Sufficient detail, code, and/or data access |
| Data/code licensing | 5 | Lawful use without restricted artifacts |
| Production feasibility | 5 | Deterministic CPU inference and operational simplicity |

A high score is not a promotion decision. A candidate must also have causally timestamped data, a frozen protocol, net out-of-sample improvement over the current champion under identical conditions, formal DSR/PBO evidence, robust cost/delay behavior, and safe failure semantics.

## What is predictably easier—and what is not

### Efficient markets and weak signed-return signals

[Fama (1970)](https://doi.org/10.1111/j.1540-6261.1970.tb00518.x) frames efficiency as prices reflecting information, while [Grossman and Stiglitz (1980)](https://doi.org/10.2307/1805228) explains why perfectly informative prices are internally inconsistent when information is costly. [Lo and MacKinlay (1988)](https://doi.org/10.1093/rfs/1.1.41) reject a strict random walk for weekly stock returns, but rejection is not equivalent to a stable net trading rule. Most directly, [Welch and Goyal (2008)](https://doi.org/10.1093/rfs/hhm014) show that prominent aggregate equity-premium predictors perform poorly and unstably out of sample.

The repository should assume signed expected returns are small, time-varying, and easy to overwhelm with costs. A model output is evidence only after it clears an executable edge floor; unavailable or invalid evidence must produce no position.

### Classical econometrics: favor stable state and risk estimates

ARIMA is a useful baseline for serial dependence, not a presumption of economic return predictability. ARCH/GARCH and stochastic-volatility models formalize conditional heteroskedasticity, while realized-volatility and HAR models exploit the much stronger persistence of variance. [Andersen et al. (2003)](https://doi.org/10.1111/1468-0262.00418) show that realized volatility supports tractable distributional forecasts; [Corsi (2009)](https://doi.org/10.1093/jjfinec/nbp001) obtains a competitive long-memory proxy from daily, weekly, and monthly realized-volatility components.

Cointegration, VAR/VECM, and lead–lag systems can be useful when the economic relation is stable and instruments are synchronized. In crypto, fragmented venues and changing contract mechanics make that stability an empirical question. Residual or spread signals need structural-break tests, point-in-time universe construction, borrow/funding treatment, and an explicit rule for relation failure.

[Hamilton (1989)](https://doi.org/10.2307/1912559), [Bai and Perron (1998)](https://doi.org/10.2307/2998540), and [Adams and MacKay (2007)](https://arxiv.org/abs/0710.3742) motivate regime and change-point layers. These are best used to reduce risk or abstain. Selecting the most profitable regime definition after observing outcomes simply moves the overfit problem one level up.

### Momentum, reversal, carry, and cross-sectional ranking

[Jegadeesh and Titman (1993)](https://doi.org/10.1111/j.1540-6261.1993.tb04702.x) and [De Bondt and Thaler (1985)](https://doi.org/10.1111/j.1540-6261.1985.tb05004.x) document momentum and long-horizon reversal in equities. [Moskowitz, Ooi, and Pedersen (2012)](https://doi.org/10.1016/j.jfineco.2011.11.003) find time-series momentum across 58 liquid futures, while [Koijen et al. (2018)](https://doi.org/10.1016/j.jfineco.2017.11.002) organize carry across asset classes. These effects are portfolio phenomena with crash, turnover, financing, and crowding risks—not universal next-bar laws.

Crypto-specific evidence is supportive but not decisive for this system. [Liu and Tsyvinski (2021)](https://doi.org/10.1093/rfs/hhaa113) report time-series momentum and attention/network predictors; [Liu, Tsyvinski, and Wu (2022)](https://doi.org/10.1111/jofi.13119) report market, size, and momentum factors plus ten characteristic sorts. The latter offers replication code, but its cross-sectional design, rebalance horizon, historical universe, and cost setting differ materially from a small Binance perpetual fleet.

The repository's `CrossSectionalMomentum` production component is not a contemporaneous cross-asset ranker: it produces a single-asset residual time-series signal against a market basket and modifies it with derivatives features. The offline research campaigns do implement true cross-sectional ranking. Their negative development results warn against inferring production value from the component name.

### Trees, regularized models, nearest neighbors, and combinations

Tree ensembles can capture interactions without the training and inference footprint of deep sequence models. [Friedman (2001)](https://doi.org/10.1214/aos/1013203451) provides the statistical gradient-boosting foundation; [Gu, Kelly, and Xiu (2020)](https://doi.org/10.1093/rfs/hhaa009) find value in nonlinear models for cross-sectional equity returns, with shallow structures often outperforming deeper ones. That study is economically informative but not a crypto-perpetual replication.

Nearest-neighbor methods are useful analogue baselines when distances are trained and normalized causally, but high-dimensional financial neighborhoods are unstable. Forecast combinations often beat brittle winner-take-all selection; [Bates and Granger (1969)](https://doi.org/10.1057/jors.1969.103) remains the basic motivation. Dynamic ensembles still need delayed-loss accounting, bounded adaptation, and a neutral fallback.

### Neural temporal architectures and foundation models

[Fischer and Krauss (2018)](https://doi.org/10.1016/j.ejor.2017.11.054) report an LSTM advantage on S&P 500 constituents, but the economic advantage weakens materially late in the sample and after costs. The generic TCN comparison of [Bai, Kolter, and Koltun (2018)](https://arxiv.org/abs/1803.01271), the Transformer of [Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762), and the Temporal Fusion Transformer of [Lim et al. (2021)](https://doi.org/10.1016/j.ijforecast.2021.03.012) establish architectures, not market alpha.

[PatchTST (Nie et al., 2023)](https://openreview.net/forum?id=Jbdc0vTOcol) contributes patch tokens and channel independence. The repository's `PatchTST` implementation does neither: it summarizes fixed trailing patches and fits ridge regression. Likewise, its `TCN` is ridge regression over hand-selected dilated return lags, and its `Transformer` is a softmax similarity-weighted average over stored targets. These are lightweight proxies with useful causal/fail-closed properties, but their names must not be cited as faithful reproductions.

[Chronos](https://arxiv.org/abs/2403.07815) and [TimesFM](https://openreview.net/forum?id=jn2iTJas6h) show broad zero-shot forecasting capability, while [GIFT-Eval](https://arxiv.org/abs/2410.10393) supplies a broad nonfinancial benchmark and non-leaking pretraining corpus. None establishes net crypto trading performance. Public TimesFM guidance for its early checkpoint also called for roughly 32 GB of memory, which is a poor fit for the repository's CPU-only, concurrent-symbol operational budget. Foundation models remain monitoring items, not production candidates.

### Probabilistic forecasts, quantiles, calibration, and abstention

The right target is a distribution of future net return, not an uncalibrated price point. Quantile loss exposes asymmetric tails; interval coverage and width reveal whether uncertainty is useful; and calibrated positive-return probabilities can support an explicit no-trade region. [Gibbs and Candès (2021)](https://proceedings.neurips.cc/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html) adapt conformal coverage under distribution shift, but long-run average coverage does not by itself guarantee conditional coverage in a given crypto regime.

The repository already contains linear quantile models and split/adaptive conformal logic. The gap is not another label: it is causal calibration evaluation, missingness-aware feature contracts, and proof that abstention improves net outcomes rather than merely reducing the number of losing trades after the fact.

### Microstructure, order flow, liquidity, and execution

[Kyle (1985)](https://doi.org/10.2307/1913210) formalizes informed trading and price impact. [Cont, Kukanov, and Stoikov (2014)](https://doi.org/10.1093/jjfinec/nbt003) find that order-flow imbalance is linearly related to short-horizon price changes with sensitivity inversely related to depth. [Sirignano and Cont (2019)](https://doi.org/10.1080/14697688.2019.1622295) and [Kolm, Turiel, and Westray (2023)](https://doi.org/10.1111/mafi.12413) show cross-asset regularities and the value of stationary order-flow features; the latter finds useful horizons on the order of only a few price changes.

Those horizons make latency, queue position, adverse selection, and impact first-order. The repository currently stores taker buy/sell aggregates and live book-risk evidence, not a licensed historical event-level L2 archive. Taker imbalance is not interchangeable with depth-normalized OFI. A faithful OFI prototype must wait for point-in-time book events and an executable fill model.

[Almgren and Chriss (2001)](https://doi.org/10.3905/jpm.2001.319105) makes the risk/impact trade-off explicit. The repository backtester models fees, spread/slippage proxies, funding, turnover, execution delay, and several risk gates, but it does not prove partial-fill, queue, or intrabar liquidation fidelity. Those omissions must remain explicit.

### Perpetual futures, basis, funding, options, and alternative data

[Makarov and Schoar (2020)](https://doi.org/10.1016/j.jfineco.2019.07.001) document persistent cross-exchange crypto price segmentation while emphasizing capital controls and settlement frictions. Funding and basis are therefore mechanisms, not free arbitrage. A 2026 peer-reviewed CEX/DEX funding study reports attractive scenario results, including leveraged cases, but the strongest outcome is selected from 60 scenarios and cross-venue execution/liquidation assumptions remain central ([Werapun et al., 2026](https://doi.org/10.1016/j.bcra.2025.100354)). A newer basis-trade risk decomposition uses synthetic DEX series for much of its analysis and is best treated as a risk framework, not efficacy evidence ([Werapun, 2026](https://doi.org/10.1007/s42521-026-00213-3)).

The September-cutoff SSRN paper [“Anatomy of Cryptocurrency Perpetual Futures Returns”](https://doi.org/10.2139/ssrn.6795783) reports 63 significant sorts from 170 predictors and a two-factor explanation. It is a one-page, non-peer-reviewed posting with an obvious multiplicity burden; it is monitored, not used to justify an implementation.

Options-implied variance and the variance risk premium can forecast or price tail exposure, but lawful historical crypto option surfaces are not presently available in the repository. On-chain, stablecoin, institutional-flow, macro, news, search, social, developer, governance, security-incident, and prediction-market data can be causally useful only with event time, availability time, revisions, coverage, and licensing. The existing external-data system covers many such families, but no family has passed an incremental economic ablation here. Missing observations must remain missing or force neutral/abstention behavior; an encoded numeric zero must never silently mean both “observed zero” and “unavailable.”

## Detailed technical assessments

### 1. Welch and Goyal (2008): aggregate return prediction

- **Assumption and target:** expanding or rolling historical relationships between macro/valuation predictors and the US equity premium can forecast the next premium.
- **Data and validation:** long US annual/monthly histories; genuine recursive out-of-sample comparisons against the historical-mean benchmark.
- **Reported effect:** most variables fail to beat the benchmark consistently; apparent in-sample relations are unstable.
- **Weakness/replication:** one aggregate market and low-frequency setting, but the negative finding has broad methodological influence.
- **Repository relevance:** establishes the correct baseline and burden of proof. Any crypto model must beat zero/mean/naive rules out of sample after costs. **Disposition: integrate as evaluation doctrine, not as a predictor.**

### 2. Andersen et al. (2003) and Corsi (2009): realized volatility and HAR

- **Assumption and target:** latent volatility is persistent and observed intraday squared returns provide a useful realized measure; heterogeneous horizons can be approximated linearly.
- **Data and validation:** liquid FX/equity realized measures; distributional and out-of-sample volatility comparisons. HAR uses daily, weekly, and monthly components.
- **Reported effect:** realized-volatility forecasts are substantially more tractable than signed returns; a small HAR model competes well with more elaborate long-memory specifications.
- **Weakness/replication:** microstructure noise, jumps, trading-hour definitions, and non-24/7 sampling matter. Bar-level squared returns are a noisier proxy than high-frequency realized variance.
- **Repository relevance:** high. It is cheap, interpretable, and can gate risk without claiming directional alpha. **Disposition: prototype prospectively as a disabled risk challenger.**

### 3. Hamilton (1989), Bai–Perron (1998), and Adams–MacKay (2007): regimes and breaks

- **Assumption and target:** data-generating parameters are piecewise stable or evolve through latent states.
- **Data and validation:** Hamilton estimates probabilistic latent regimes; Bai–Perron locates multiple structural changes; Bayesian online change-point detection updates run-length probabilities causally.
- **Reported effect:** regime/break models explain state-dependent dynamics in their original domains.
- **Weakness/replication:** labels are not intrinsically “trend,” “mean reversion,” or “high volatility”; post-hoc semantic labeling and regime-conditioned strategy selection can leak outcomes.
- **Repository relevance:** the HMM is structurally faithful but its economic state names are heuristic. Break probabilities are more defensible as an abstention input than as a direction. **Disposition: monitor; require prospective degradation evidence.**

### 4. Moskowitz, Ooi, and Pedersen (2012): time-series momentum

- **Assumption and target:** an instrument's own past excess return predicts its future excess return over monthly horizons.
- **Data and validation:** 58 equity-index, currency, commodity, and bond futures over decades; diversified portfolios and factor controls.
- **Reported effect:** positive 1–12 month persistence followed by longer-horizon partial reversal.
- **Weakness/replication:** strategy returns can be convex/crash-sensitive, and historical futures costs/capacity differ from crypto perpetuals. It says little about 1–6 bar intraday prediction.
- **Repository relevance:** supports simple trend baselines and a longer-horizon research prior, not the rejected residual campaign's exact construction. **Disposition: monitor; do not retry on contaminated data.**

### 5. Koijen et al. (2018): carry

- **Assumption and target:** observable forward/spot or yield relations proxy expected returns across assets.
- **Data and validation:** global currencies, commodities, equities, bonds, credit, and options; common-factor and recession analyses.
- **Reported effect:** carry is positive on average and comoves across asset classes, with exposure to global recession/liquidity risks.
- **Weakness/replication:** carry definitions are asset-specific; funding receipts are not the same as risk-free yield and can coincide with crowding or venue risk.
- **Repository relevance:** direct mechanism for the already preregistered prospective funding-carry path. **Disposition: continue the existing one-shot prospective campaign only; no early performance read.**

### 6. Gatev, Goetzmann, and Rouwenhorst (2006): pairs trading

- **Assumption and target:** historically close normalized price paths represent a temporary relative-value relation that reverts.
- **Data and validation:** US equities, formation/trading periods, simple distance selection, conservative return accounting for the era.
- **Reported effect:** excess returns in the studied period, declining in later evidence.
- **Weakness/replication:** data mining, delisting/borrow costs, relation breaks, and synchronization dominate implementation risk.
- **Repository relevance:** motivates residual baselines and explicit relation-failure guards. The repository's reversal variants failed mechanical/risk gates, so another spread rule is not justified now. **Disposition: reject for current data.**

### 7. Gu, Kelly, and Xiu (2020): machine learning in asset pricing

- **Assumption and target:** nonlinear functions of firm characteristics and macro interactions improve monthly cross-sectional excess-return forecasts.
- **Data and validation:** a large US equity panel with chronological train/validation/test design and portfolio sorts.
- **Reported effect:** trees and neural networks improve predictive and economic measures; shallow structures are competitive and momentum/liquidity/volatility features matter.
- **Weakness/replication:** monthly equities, survivorship/characteristic construction, portfolio breadth, and institutional-cost assumptions do not transfer automatically to crypto perps.
- **Repository relevance:** supports shallow nonlinear challengers and cross-sectional evaluation, not a claim that the repository's small stump booster is an exact reproduction. **Disposition: monitor/prototype only with point-in-time universe data.**

### 8. Fischer and Krauss (2018): LSTM equity prediction

- **Assumption and target:** recurrent nonlinearities learn daily cross-sectional return direction from lagged returns.
- **Data and validation:** rolling S&P 500 constituent data from 1992–2015; LSTM compared with random forest, deep net, and logistic regression; transaction costs included.
- **Reported effect:** LSTM leads the tested models over the full period, but the advantage and profitability deteriorate around the later sample.
- **Weakness/replication:** index constituent history, repeated tuning, simplified fills, and a structurally different market; late-sample decay is central.
- **Repository relevance:** the current LSTM is a real recurrent architecture, but raw-price training, limited normalization, simple splitting, and weak artifact provenance are material deviations. **Disposition: do not expand until validation/provenance gaps close.**

### 9. PatchTST (2023)

- **Assumption and target:** local semantic patches reduce attention cost and channel-independent processing improves multivariate long-horizon forecasting.
- **Data and validation:** standard electricity, traffic, weather, and related forecasting benchmarks; supervised train/validation/test comparisons.
- **Reported effect:** strong point-forecast benchmark results for long horizons.
- **Weakness/replication:** benchmark loss is not net financial value; later benchmark studies show rankings depend on datasets and protocols.
- **Repository relevance:** the namesake implementation lacks patch tokens, self-attention, channel independence, and neural optimization. **Disposition: reject faithful integration absent new economic evidence; relabel current semantics.**

### 10. Chronos, TimesFM, and GIFT-Eval

- **Assumption and target:** large heterogeneous pretraining can provide a transferable prior for probabilistic or point time-series forecasts.
- **Data and validation:** Chronos reports 42 datasets; TimesFM pretrains on roughly 100 billion points; GIFT-Eval spans 23 datasets and 144,000 series with a non-leaking pretraining resource.
- **Reported effect:** competitive zero-shot general forecasting, especially where local training data are scarce.
- **Weakness/replication:** limited financial-market representation, opaque pretraining overlap risk, large runtime/dependency footprint, and no demonstrated net trading edge.
- **Repository relevance:** poor production fit and no incremental mechanism beyond existing simple baselines. **Disposition: reject for integration; monitor CPU-small models and finance-specific independent tests.**

### 11. Cont, Kukanov, and Stoikov (2014): order-flow imbalance

- **Assumption and target:** net supply/demand changes at the best quotes, scaled by depth, move prices over short intervals.
- **Data and validation:** tick data for 50 US stocks; contemporaneous and short-horizon linear relations across intervals.
- **Reported effect:** OFI explains price changes more robustly than raw trade volume, and impact varies inversely with depth.
- **Weakness/replication:** much evidence is explanatory/contemporaneous rather than an executable forecast; equities' queue rules differ from crypto.
- **Repository relevance:** high mechanism fit, but required L2 event history is absent. Taker-volume ratios are not a faithful substitute. **Disposition: preregistered monitor, blocked on data.**

### 12. Kolm, Turiel, and Westray (2023): deep order-flow forecasting

- **Assumption and target:** stationary order-flow representations and cross-sectional training improve very-short-horizon mid-price forecasts.
- **Data and validation:** 115 Nasdaq stocks with LOBSTER data; rolling out-of-sample comparisons across architectures and horizons.
- **Reported effect:** stationary order-flow features outperform raw-book inputs; economically useful forecast horizons are extremely short.
- **Weakness/replication:** proprietary data, equity microstructure, forecast-to-fill gap, and capacity/adverse-selection risk.
- **Repository relevance:** reinforces the data and latency requirements that currently block OFI adoption. **Disposition: monitor.**

### 13. Liu and Tsyvinski (2021)

- **Assumption and target:** crypto-specific network growth, momentum, and investor attention forecast aggregate crypto returns.
- **Data and validation:** major cryptocurrencies and network/attention proxies over the early-to-maturing crypto era.
- **Reported effect:** strong time-series momentum and attention relationships; production-cost proxies are less informative.
- **Weakness/replication:** short history, expanding market composition, vendor/revision timing for attention/network data, and many tested predictors.
- **Repository relevance:** supports causal attention/network ablations, but the current external-data families lack an accepted incremental result. **Disposition: monitor.**

### 14. Liu, Tsyvinski, and Wu (2022)

- **Assumption and target:** cross-sectional crypto expected returns are summarized by market, size, momentum, and related characteristics.
- **Data and validation:** broad coin panel, long–short characteristic portfolios, factor spanning, and released replication code.
- **Reported effect:** ten characteristics have significant long–short returns explained by a three-factor model.
- **Weakness/replication:** investability, changing exchanges, delistings, spreads, and portfolio breadth are critical; factor explanation is not a timing model.
- **Repository relevance:** the strongest support for a true point-in-time cross-sectional ranker, but the ten-symbol fixed fleet is too narrow for a clean replication. **Disposition: continue data acquisition, not implementation.**

### 15. Makarov and Schoar (2020)

- **Assumption and target:** price differences across exchanges reflect segmented capital and settlement constraints.
- **Data and validation:** cross-exchange cryptocurrency prices, flows, and arbitrage spreads.
- **Reported effect:** large, persistent international and cross-venue dislocations, related to capital-flow frictions.
- **Weakness/replication:** quoted spreads are not freely capturable; transfer delay, custody, capital location, withdrawal, and counterparty risks bind.
- **Repository relevance:** justifies spot/perpetual and cross-exchange context but demands venue-specific execution accounting. **Disposition: monitor.**

### 16. Werapun et al. (2026) and Cao et al. (2026): newest perpetual evidence

- **Assumption and target:** cross-venue funding or log basis/price-volume characteristics predict total perpetual returns.
- **Data and validation:** the peer-reviewed funding paper studies several CEX/DEX venues and 60 scenarios; the SSRN posting describes 170 predictor sorts and 63 significant returns.
- **Reported effect:** selected funding scenarios are attractive; the working paper reports broad characteristic significance and a two-factor explanation.
- **Weakness/replication:** scenario selection, leverage, venue/fill/liquidation fidelity, one-page working-paper detail, and severe multiple-testing risk. Independent replication is not yet available.
- **Repository relevance:** informative for mechanism and risk decomposition, insufficient for adoption. **Disposition: monitor.**

### 17. White (2000) and Hansen (2005): data snooping

- **Assumption and target:** the best observed strategy must be evaluated relative to the entire searched family, not as if it were prespecified.
- **Data and validation:** bootstrap tests of whether the best model has superior predictive ability; SPA improves power against poor alternatives.
- **Reported effect:** apparent winners often lose significance after search is acknowledged.
- **Weakness/replication:** dependence/bootstrap choices matter and adaptive, undocumented experiments still escape the test family.
- **Repository relevance:** foundational. Every seed, configuration, and successor belongs in the registry. **Disposition: integrate as evaluation doctrine.**

### 18. Deflated Sharpe Ratio and Probability of Backtest Overfitting

- **Assumption and target:** Sharpe significance must account for non-normal returns and the number/correlation of trials; selection instability can be estimated by combinatorially symmetric cross-validation.
- **Data and validation:** analytical DSR and CSCV/PBO applied to complete trial-return matrices.
- **Reported effect:** conventional Sharpe ratios materially overstate confidence after selection.
- **Weakness/replication:** DSR/PBO cannot repair leakage, omitted trials, an invalid cost model, or a tiny number of independent decisions.
- **Repository relevance:** the research harness implements formal DSR and CSCV/PBO. The production optimizer's `deflatedSharpeProxy` and `pboProxy` are explicitly heuristics and must not be substituted for formal gates. **Disposition: retain canonical research thresholds (DSR probability ≥ 0.95; PBO ≤ 0.20).**

### 19. Gibbs and Candès (2021): adaptive conformal inference

- **Assumption and target:** online adjustment of the conformal miscoverage level can track distribution shift without a stationary exchangeable sequence.
- **Data and validation:** theoretical long-run coverage plus empirical nonfinancial shift examples.
- **Reported effect:** average coverage adapts under visible distribution changes.
- **Weakness/replication:** local conditional coverage, serially dependent financial losses, interval width, and action selection remain unresolved.
- **Repository relevance:** supports a bounded calibration challenger only if interval coverage, width, abstention, and net economic performance are all assessed. **Disposition: prototype prospectively with a missingness-aware shallow model.**

## Evidence against easy prediction

Three results set the prior for this work. [McLean and Pontiff (2016)](https://doi.org/10.1111/jofi.12365) find anomaly returns about 26% lower out of sample and 58% lower after publication. [Hou, Xue, and Zhang (2020)](https://doi.org/10.1093/rfs/hhy131) find that many published anomalies fail replication or stricter significance thresholds. Welch and Goyal find unstable aggregate timing. Together with the repository's failed nested campaign, these results favor small hypothesis families, prospective data, formal multiplicity correction, and negative decisions.

## Repository implications

1. Keep the current champion unchanged. No reviewed paper supplies repository-specific net OOS evidence.
2. Use volatility, calibration, and regime models first as risk/abstention tools; require a separate test before allowing them to drive direction.
3. Preserve the legacy `tcn`, `patch_tst`, and `transformer` identifiers, but describe them as version-1 proxies and accept accurate aliases. A faithful neural model would require a new semantic identifier and artifact version.
4. Treat external-feature availability as part of the feature schema. Do not let missing, stale, invalid, or non-finite evidence become an actionable numeric signal.
5. Keep the historical 1,227-return final holdout sealed and do not calculate performance for the prospective funding-carry campaign before its registered minimum evaluation time.
6. Do not retry residual-reversal exposure levels or invert failed signals on the already inspected development period.
7. Do not add a production Python service, GPU dependency, foundation-model checkpoint, paid feed, or restricted dataset for any current candidate.

The complete metadata and dispositions are in `paper-matrix.csv`; code-level findings are in `model-fidelity-audit.md` and `paper-to-repository-gap-matrix.md`.
