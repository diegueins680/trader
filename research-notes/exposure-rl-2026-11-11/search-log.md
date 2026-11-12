# Search log (2026-11-11)

All searches, fetches and API calls below were made on 2026-11-11. No paper
PDF was downloaded or stored. No third-party code was opened for reuse;
repository metadata (existence, licence field, archived flag) was read from the
GitHub REST API only.

## 1. Sources consulted

| Source | Use | Access |
|---|---|---|
| arXiv API (`export.arxiv.org/api/query?id_list=...`) | title, authors, latest version, dates, author comment, journal-ref | batched `id_list` queries; several batches hit "Rate exceeded" and were retried with back-off until all returned |
| arXiv abstract pages (`arxiv.org/abs/<id>`) | version history for ids fetched individually | web fetch |
| Crossref REST API (`api.crossref.org/works/<doi>`) | title, authors, venue, volume, pages, year for DOIs | one call per DOI |
| PMLR proceedings pages | venue, volume, pages | web fetch (v48 thomasa16, v48 wangf16, v80 pardo18a, v97 le19a) |
| GitHub REST API (`api.github.com/repos/<owner>/<repo>`) | repository existence, SPDX licence field, archived flag | unauthenticated, one call per repository |
| Web search | venue confirmation for works without DOI/arXiv id; 2024–2026 currency searches | queries listed in section 2 |
| This repository | prior results and prior matrices | `research-notes/sequential-control-2026-09-17/` (final-decision-memo, policy-cards, ope-and-support, cost-execution-and-robustness, environment-contract, paper-matrix, literature-update-2026-09-28), `research-notes/registrations/sequential-control-screen-v1.json`, `research-notes/market-prediction-2026-09-04/` (literature-review, paper-matrix) |

## 2. Exact web-search queries

Currency searches (2024–2026):

1. `reinforcement learning trading critique replication "simple" baselines outperform deep RL backtest overfitting 2025 arXiv`
2. `offline reinforcement learning 2025 2026 flow policy "Flow Q-learning" OR "value learning bottleneck" benchmark OGBench simple baselines`
3. `safe reinforcement learning shielding 2025 2026 action projection continuous action spaces policy gradient bias "action projection" OR "safety layer" arXiv`
4. `FinRL replication buy-and-hold deep reinforcement learning stock trading fails to outperform independent evaluation 2024 2025`
5. `deep reinforcement learning cryptocurrency perpetual futures trading 2025 2026 arXiv transaction costs funding rate out-of-sample evaluation`
6. `reinforcement learning in finance systematic review 2024 2025 critical "transaction costs" "overfitting" meta-analysis survey arXiv quantitative trading RL limitations`
7. `cryptocurrency time-series momentum reversal intraday perpetual futures 2024 2025 paper evidence after transaction costs`
8. `off-policy evaluation reinforcement learning 2025 fitted Q evaluation reliability model selection offline RL new results arXiv`
9. `Park Frans Levine Kumar "Is Value Learning Really the Main Bottleneck in Offline RL?" NeurIPS 2024`

Citation-verification searches:

10. `"Methods Matter: A Trading Agent with No Intelligence Routinely Outperforms AI-Based Traders" Cliff arXiv`
11. `Thomas Theocharous Ghavamzadeh "High Confidence Off-Policy Evaluation" AAAI 2015`
12. `Kapturowski "Recurrent Experience Replay in Distributed Reinforcement Learning" ICLR 2019 openreview`
13. `García Fernández "A Comprehensive Survey on Safe Reinforcement Learning" JMLR 2015 volume 16 pages`
14. `Precup Sutton Singh "Eligibility Traces for Off-Policy Policy Evaluation" ICML 2000 pages 759-766`
15. `Jansen Könighofer Junges Serban Bloem "Safe Reinforcement Learning Using Probabilistic Shields" CONCUR 2020 LIPIcs DOI`
16. `Ray Achiam Amodei "Benchmarking Safe Exploration in Deep Reinforcement Learning" 2019 Safety Gym PPO-Lagrangian`
17. `"The 37 Implementation Details of Proximal Policy Optimization" Huang Dossa Raffin Kanervisto Wang ICLR Blog Track 2022`
18. `Voloshin Le Jiang Yue "Empirical Study of Off-Policy Policy Evaluation for Reinforcement Learning" NeurIPS 2021 Datasets and Benchmarks track`
19. `"Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO" ICLR 2020 Engstrom Ilyas Santurkar Tsipras Janoos Rudolph Madry`
20. `Kumar Hong Singh Levine "When Should We Prefer Offline Reinforcement Learning Over Behavioral Cloning?" ICLR 2022 arXiv 2204.05618`
21. `Kostrikov Nair Levine "Offline Reinforcement Learning with Implicit Q-Learning" ICLR 2022 openreview; Chow Tamar Mannor Pavone "Risk-Sensitive and Robust Decision-Making: a CVaR Optimization Approach" NeurIPS 2015 proceedings`

Web fetches of arXiv abstract pages: 2201.04699, 2408.10932, 2502.08021,
2005.12729, 1712.00378, 2204.05618, 2406.09329, 2502.02538.

## 3. What the searches did not find

- No independent, cost-inclusive, out-of-sample replication of an RL exposure
  or position policy on crypto perpetuals (queries 1, 4, 5). Absence from
  these searches is weak evidence; such replications are rarely published.
- No 2025–2026 paper analysing policy-gradient bias under continuous action
  projection beyond the 2018–2023 works already mapped (query 3).
- Query 21 returned nothing for the CVaR paper's proceedings entry; its venue
  is recorded as "from memory".
- A Crossref lookup of `10.4230/LIPIcs.CONCUR.2020.3` returned "Resource not
  found"; the DOI was confirmed from the Dagstuhl DROPS record via query 15.

## 4. Leads seen but not reviewed (not cited as evidence anywhere)

These appeared in search results. They were not opened in full, are not in the
matrix, and support no claim in this review.

| Lead | Seen via | Note |
|---|---|---|
| arXiv:2512.23773 "FineFT: Efficient and Risk-Aware Ensemble Reinforcement Learning for Futures Trading" | query 5 | ensemble RL on perpetual-style futures; results not checked |
| arXiv:2602.11708 (crypto trend-following with adaptive portfolio construction) | query 7 | models taker fee, slippage and funding per search summary; unverified |
| arXiv:2608.21888 "Short-horizon mean reversion in cryptocurrency markets: a matched cross-market measurement" | query 7 | findings not retrieved |
| arXiv:2605.13435 "Q-Flow"; arXiv:2606.10613 (bootstrapped flow Q-learning, already in the 2026-09-17 matrix) | query 2 | 2026 successors to Flow Q-learning; robotics benchmarks |
| arXiv:2505.07078 (FINSABER; LLM investing strategies, KDD 2026 D&B per arXiv comment) | query 1; arXiv API | LLM agents, not RL; out of scope |
| Čekal, "Reinforcement Learning for Perpetual Futures Market Making", working paper 2026 | query 5 | market making; out of scope |
| Meng & Khushi (2019), Millea (2021) surveys of RL trading | query 6 | older critical surveys; not opened |
| arXiv:2210.10691 Kochdumper et al., action projection with zonotopes (IEEE OJ-CSYS 2023) | query 3; Crossref; arXiv API | metadata verified; not needed beyond [E08] |
| arXiv:2006.05990, 2305.09836, 2210.07105, 2005.01643, 2107.06277, 1804.06893, 2410.20092, 1806.08295, 1906.00949, 1910.00177, 1802.03042, 2004.06627 | arXiv API | metadata verified during candidate screening; excluded from the matrix to avoid padding |

## 5. Verification summary

- Works in the matrix: 75. All 75 had title and authors checked against at
  least one web source on 2026-11-11 (arXiv API or page, Crossref, a
  proceedings page, or web-search listings). None is included from memory
  alone.
- Fields that remain "from memory, not re-verified on 2026-11-11" although the
  work itself was verified: the venue of B02 (ICLR 2016), B08 (AAAI 2018), C03
  (NeurIPS 2015) and C07 (NeurIPS 2021); the last authors of D05 and E11
  (the reviewer's API listing was truncated at six names); the hosting
  repositories recorded for B14 and C09.
- Discrepancies and doubts found:
  - A10: the PMLR page gives pages 4045–4054; the arXiv journal-ref gives
    4042–4051.
  - F01: Crossref lists a second name (a discussant) as co-author; the work is
    cited as sole-authored, consistent with the earlier repository matrix.
  - F05: Crossref gives an issue date of 2016-09 and no volume or pages; the
    earlier repository matrix gives 2017. The year is kept as 2017 and flagged.
  - E02: the CONCUR 2020 title ("Using Probabilistic Shields") differs from
    the arXiv title ("via Probabilistic Shields").
  - E04: the TPAMI title ("Methods, Theories, and Applications") differs
    slightly from the arXiv title ("Methods, Theory and Applications").
  - D10: NeurIPS 2025 publication is reported by a search summary only; the
    arXiv page shows no journal reference.
  - E12: AAMAS 2026 acceptance is taken from this repository's 2026-09-28
    literature update, not re-verified.
  - N07: the ACM Computing Surveys DOI is taken from the arXiv page note and
    was not resolved through Crossref.
  - C02 and N04 are not peer-reviewed conference papers (an OpenAI technical
    report and an ICLR Blog Track post); B01, B14, C12, D05, E05, F03 and N05
    are preprints with no peer-reviewed venue verified.
  - C04: the OpenReview id comes from a search result; the forum page was not
    opened.
  - A11: the quoted return, funding share and information ratio come from a
    fetched summary of the arXiv abstract, not from the paper body.
- Licences: "NOASSERTION" from the GitHub API means a licence file exists that
  GitHub does not classify; it was not inspected. "No licence file detected"
  means the API returned no licence, in which case default copyright applies.
  No code from any repository was read or copied, so licence terms were not
  exercised.
- Content caveat: experimental details in the matrix (data, validation,
  statistical evidence) come from abstracts and the reviewer's prior reading
  unless the verification field says otherwise. No numerical result is quoted
  from a paper unless it appeared in a fetched abstract or page.

## 6. Per-paper record

Columns: matrix id; citation; version and publication status; identifier;
official code; licence; how verified.

| id | citation | version / status | arXiv or DOI | official code | licence | verification |
|---|---|---|---|---|---|---|
| A01 | Moody J.; Saffell M. Learning to trade via direct reinforcement. IEEE Trans. Neural Networks 12(4):875-889 | peer-reviewed journal (IEEE TNN) | 10.1109/72.935097 | none identified | n/a | Crossref API 2026-11-11 (title, authors, venue, year); content from abstract and prior reading, not re-read in full |
| A02 | Deng Y.; Bao F.; Kong Y.; Ren Z.; Dai Q. Deep Direct Reinforcement Learning for Financial Signal Representation and Trading. IEEE TNNLS 28(3):653-664 | peer-reviewed journal (IEEE TNNLS) | 10.1109/TNNLS.2016.2522401 | none identified | n/a | Crossref API 2026-11-11 (title, authors, venue, year); content from abstract and prior reading, not re-read in full |
| A03 | Garleanu N.; Pedersen L.H. Dynamic Trading with Predictable Returns and Transaction Costs. Journal of Finance 68(6):2309-2340 | peer-reviewed journal (JF) | 10.1111/jofi.12080 | none | n/a | Crossref API 2026-11-11 (title, authors, venue, year); content from abstract and prior reading, not re-read in full |
| A04 | Almgren R.; Chriss N. Optimal execution of portfolio transactions. Journal of Risk 3(2):5-39 | peer-reviewed journal (J. Risk); volume/issue per Crossref | 10.21314/JOR.2001.041 | none | n/a | Crossref API 2026-11-11 (title, authors, venue, year); already in market-prediction-2026-09-04 matrix; content from abstract and prior reading, not re-read in full |
| A05 | Nevmyvaka Y.; Feng Y.; Kearns M. Reinforcement learning for optimized trade execution. ICML 2006, pp. 673-680 | peer-reviewed conference (ICML) | 10.1145/1143844.1143929 | none identified | n/a | Crossref API 2026-11-11 (title, authors, venue, year); content from abstract and prior reading, not re-read in full |
| A06 | Mahadevan S. Average reward reinforcement learning: Foundations, algorithms, and empirical results. Machine Learning 22:159-195 | peer-reviewed journal (Machine Learning) | 10.1007/BF00114727 | none | n/a | Crossref API 2026-11-11 (title, authors, venue, year); content from abstract and prior reading, not re-read in full |
| A07 | Li L.; Chu W.; Langford J.; Schapire R.E. A Contextual-Bandit Approach to Personalized News Article Recommendation. WWW 2010 | peer-reviewed conference (WWW); arXiv v2 | arXiv:1003.0146; 10.1145/1772690.1772758 | none identified | n/a | arXiv API 2026-11-11 (title, authors, latest version); in sequential-control-2026-09-17 matrix; content from abstract and prior reading, not re-read in full |
| A08 | Zhang Z.; Zohren S.; Roberts S. Deep Reinforcement Learning for Trading. Journal of Financial Data Science 2(2):25-40 | peer-reviewed journal (JFDS); arXiv v1 | 10.3905/jfds.2020.1.030; arXiv:1911.10107 | none identified | n/a | Crossref API 2026-11-11 (title, authors, venue, year); arXiv API; content from abstract and prior reading, not re-read in full |
| A09 | Hambly B.; Xu R.; Yang H. Recent advances in reinforcement learning in finance. Mathematical Finance 33(3):437-503 | peer-reviewed journal (Math. Finance); arXiv v4 | 10.1111/mafi.12382; arXiv:2112.04553 | none | n/a | Crossref API 2026-11-11 (title, authors, venue, year); arXiv API; content from abstract and prior reading, not re-read in full |
| A10 | Pardo F.; Tavakoli A.; Levdik V.; Kormushev P. Time Limits in Reinforcement Learning. ICML 2018, PMLR 80 | peer-reviewed conference (ICML); arXiv v4 | arXiv:1712.00378 | project site per arXiv comment; repo unverified | unverified | arXiv abstract page fetched 2026-11-11 (title, authors, version history); PMLR page fetched (pages 4045-4054; arXiv journal-ref says 4042-4051 - discrepancy noted) |
| A11 | Borrageiro G.; Firoozye N.; Barucca P. The Recurrent Reinforcement Learning Crypto Agent. IEEE Access 10:38590-38599 | peer-reviewed journal (IEEE Access); arXiv v4 | 10.1109/ACCESS.2022.3166599; arXiv:2201.04699 | none identified | n/a | arXiv abstract page fetched 2026-11-11 (title, authors, version history) (abstract summary; DOI and journal ref from arXiv page) |
| B01 | Schulman J.; Wolski F.; Dhariwal P.; Radford A.; Klimov O. Proximal Policy Optimization Algorithms | preprint (arXiv v2); no peer-reviewed venue identified | arXiv:1707.06347 | https://github.com/openai/baselines | MIT | arXiv API 2026-11-11 (title, authors, latest version); code repo and licence via GitHub API 2026-11-11; content from abstract and prior reading, not re-read in full |
| B02 | Schulman J.; Moritz P.; Levine S.; Jordan M.; Abbeel P. High-Dimensional Continuous Control Using Generalized Advantage Estimation | ICLR 2016 (venue from memory, not re-verified); arXiv v6 | arXiv:1506.02438 | none identified | unverified | arXiv API 2026-11-11 (title, authors, latest version); content from abstract and prior reading, not re-read in full |
| B03 | Haarnoja T.; Zhou A.; Abbeel P.; Levine S. Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor. ICML 2018 | peer-reviewed conference (ICML, per arXiv comment); arXiv v2 | arXiv:1801.01290 | https://github.com/haarnoja/sac | custom/unrecognised by GitHub (NOASSERTION); not inspected | arXiv API 2026-11-11 (title, authors, latest version); code repo and licence via GitHub API 2026-11-11; content from abstract and prior reading, not re-read in full |
| B04 | Fujimoto S.; van Hoof H.; Meger D. Addressing Function Approximation Error in Actor-Critic Methods. ICML 2018 | peer-reviewed conference (ICML, per arXiv comment); arXiv v3 | arXiv:1802.09477 | https://github.com/sfujim/TD3 | MIT | arXiv API 2026-11-11 (title, authors, latest version); code repo and licence via GitHub API 2026-11-11; content from abstract and prior reading, not re-read in full |
| B05 | van Hasselt H.; Guez A.; Silver D. Deep Reinforcement Learning with Double Q-learning. AAAI 2016 | peer-reviewed conference (AAAI); arXiv v3 | arXiv:1509.06461; 10.1609/aaai.v30i1.10295 | none identified | n/a | arXiv API 2026-11-11 (title, authors, latest version); Crossref; content from abstract and prior reading, not re-read in full |
| B06 | Wang Z.; Schaul T.; Hessel M.; van Hasselt H.; Lanctot M.; de Freitas N. Dueling Network Architectures for Deep Reinforcement Learning. ICML 2016, PMLR 48:1995-2003 | peer-reviewed conference (ICML); arXiv v3 | arXiv:1511.06581 | none identified | n/a | arXiv API 2026-11-11 (title, authors, latest version); PMLR page fetched; content from abstract and prior reading, not re-read in full |
| B07 | Bellemare M.G.; Dabney W.; Munos R. A Distributional Perspective on Reinforcement Learning. ICML 2017 | peer-reviewed conference (ICML, per arXiv comment); arXiv v1 | arXiv:1707.06887 | none identified | n/a | arXiv API 2026-11-11 (title, authors, latest version); content from abstract and prior reading, not re-read in full |
| B08 | Dabney W.; Rowland M.; Bellemare M.G.; Munos R. Distributional Reinforcement Learning with Quantile Regression. AAAI 2018 | AAAI 2018 (venue from memory, not re-verified); arXiv v1 | arXiv:1710.10044 | none identified | n/a | arXiv API 2026-11-11 (title, authors, latest version); content from abstract and prior reading, not re-read in full |
| B09 | Dabney W.; Ostrovski G.; Silver D.; Munos R. Implicit Quantile Networks for Distributional Reinforcement Learning. ICML 2018 | peer-reviewed conference (ICML, per arXiv comment); arXiv v1 | arXiv:1806.06923 | none identified | n/a | arXiv API 2026-11-11 (title, authors, latest version); content from abstract and prior reading, not re-read in full |
| B10 | Kumar A.; Zhou A.; Tucker G.; Levine S. Conservative Q-Learning for Offline Reinforcement Learning. NeurIPS 2020 | peer-reviewed conference (NeurIPS; proceedings URL held in sequential-control-screen-v1 registration); arXiv v3 | arXiv:2006.04779 | https://github.com/aviralkumar2907/CQL | no licence file detected by GitHub API | arXiv API 2026-11-11 (title, authors, latest version); code repo and licence via GitHub API 2026-11-11; content from abstract and prior reading, not re-read in full |
| B11 | Kostrikov I.; Nair A.; Levine S. Offline Reinforcement Learning with Implicit Q-Learning. ICLR 2022 | peer-reviewed conference (ICLR); arXiv v1 | arXiv:2110.06169 | https://github.com/ikostrikov/implicit_q_learning | MIT | arXiv API 2026-11-11 (title, authors, latest version); ICLR venue via web search; code repo and licence via GitHub API 2026-11-11; content from abstract and prior reading, not re-read in full |
| B12 | Fujimoto S.; Gu S.S. A Minimalist Approach to Offline Reinforcement Learning. NeurIPS 2021 | peer-reviewed conference (NeurIPS, per arXiv comment); arXiv v2 | arXiv:2106.06860 | https://github.com/sfujim/TD3_BC | MIT | arXiv API 2026-11-11 (title, authors, latest version); code repo and licence via GitHub API 2026-11-11; content from abstract and prior reading, not re-read in full |
| B13 | Fujimoto S.; Meger D.; Precup D. Off-Policy Deep Reinforcement Learning without Exploration. ICML 2019 | peer-reviewed conference (ICML, per arXiv comment); arXiv v3 | arXiv:1812.02900 | https://github.com/sfujim/BCQ | MIT | arXiv API 2026-11-11 (title, authors, latest version); code repo and licence via GitHub API 2026-11-11; content from abstract and prior reading, not re-read in full |
| B14 | Nair A.; Gupta A.; Dalal M.; Levine S. AWAC: Accelerating Online Reinforcement Learning with Offline Datasets | preprint (arXiv v6); no peer-reviewed venue verified | arXiv:2006.09359 | https://github.com/rail-berkeley/rlkit (hosting repo; AWAC-specific path not verified) | MIT (rlkit) | arXiv API 2026-11-11 (title, authors, latest version); code repo and licence via GitHub API 2026-11-11; content from abstract and prior reading, not re-read in full |
| B15 | Kumar A.; Hong J.; Singh A.; Levine S. When Should We Prefer Offline Reinforcement Learning Over Behavioral Cloning? ICLR 2022 | peer-reviewed conference (ICLR, per arXiv comment); arXiv v1 | arXiv:2204.05618 | none identified | unverified | arXiv abstract page fetched 2026-11-11 (title, authors, version history); web search for venue and earlier title; content from abstract and prior reading, not re-read in full |
| B16 | Park S.; Frans K.; Levine S.; Kumar A. Is Value Learning Really the Main Bottleneck in Offline RL? NeurIPS 2024 | peer-reviewed conference (NeurIPS, per arXiv comment); arXiv v2 | arXiv:2406.09329 | none verified | unverified | arXiv abstract page fetched 2026-11-11 (title, authors, version history); web search; content from abstract and prior reading, not re-read in full |
| C01 | Achiam J.; Held D.; Tamar A.; Abbeel P. Constrained Policy Optimization. ICML 2017 | peer-reviewed conference (ICML, per arXiv comment); arXiv v1 | arXiv:1705.10528 | https://github.com/jachiam/cpo | no licence file detected by GitHub API | arXiv API 2026-11-11 (title, authors, latest version); code repo and licence via GitHub API 2026-11-11; content from abstract and prior reading, not re-read in full |
| C02 | Ray A.; Achiam J.; Amodei D. Benchmarking Safe Exploration in Deep Reinforcement Learning | technical report (OpenAI); not peer-reviewed | https://cdn.openai.com/safexp-short.pdf | https://github.com/openai/safety-starter-agents | MIT (archived) | web search 2026-11-11 (OpenAI page and PDF located; authors/date from search summary); code repo and licence via GitHub API 2026-11-11 |
| C03 | Chow Y.; Tamar A.; Mannor S.; Pavone M. Risk-Sensitive and Robust Decision-Making: a CVaR Optimization Approach | NeurIPS 2015 (venue from memory and prior repo matrix, not re-verified; arXiv comment says submitted); arXiv v1 | arXiv:1506.02188 | none identified | n/a | arXiv API 2026-11-11 (title, authors, latest version); content from abstract and prior reading, not re-read in full |
| C04 | Kapturowski S.; Ostrovski G.; Quan J.; Munos R.; Dabney W. Recurrent Experience Replay in Distributed Reinforcement Learning. ICLR 2019 | peer-reviewed conference (ICLR) | https://openreview.net/forum?id=r1lyTjAqYX (id from web search; forum page not opened) | none identified | n/a | web search 2026-11-11 (ICLR 2019 schedule and mlanthology entries) |
| C05 | Ni T.; Eysenbach B.; Salakhutdinov R. Recurrent Model-Free RL Can Be a Strong Baseline for Many POMDPs. ICML 2022 | peer-reviewed conference (ICML, per arXiv comment); arXiv v3 | arXiv:2110.05038 | https://github.com/twni2016/pomdp-baselines | MIT | arXiv API 2026-11-11 (title, authors, latest version); code repo and licence via GitHub API 2026-11-11; content from abstract and prior reading, not re-read in full |
| C06 | Iyengar G.N. Robust Dynamic Programming. Mathematics of Operations Research 30(2):257-280 | peer-reviewed journal (MOR) | 10.1287/moor.1040.0129 | none | n/a | Crossref API 2026-11-11 (title, authors, venue, year); content from abstract and prior reading, not re-read in full |
| C07 | Chen L.; Lu K.; Rajeswaran A.; Lee K.; Grover A.; Laskin M.; Abbeel P.; Srinivas A.; Mordatch I. Decision Transformer: Reinforcement Learning via Sequence Modeling. NeurIPS 2021 | NeurIPS 2021 (venue from memory and prior repo matrix, not re-verified); arXiv v2 | arXiv:2106.01345 | https://github.com/kzl/decision-transformer | MIT | arXiv API 2026-11-11 (title, authors, latest version); code repo and licence via GitHub API 2026-11-11; content from abstract and prior reading, not re-read in full |
| C08 | Yu T.; Thomas G.; Yu L.; Ermon S.; Zou J.; Levine S.; Finn C.; Ma T. MOPO: Model-based Offline Policy Optimization. NeurIPS 2020 | peer-reviewed conference (NeurIPS, per arXiv comment); arXiv v6 | arXiv:2005.13239 | https://github.com/tianheyu927/mopo | MIT | arXiv API 2026-11-11 (title, authors, latest version); code repo and licence via GitHub API 2026-11-11; content from abstract and prior reading, not re-read in full |
| C09 | Kidambi R.; Rajeswaran A.; Netrapalli P.; Joachims T. MOReL: Model-Based Offline Reinforcement Learning. NeurIPS 2020 | peer-reviewed conference (NeurIPS, per arXiv comment); arXiv v3 | arXiv:2005.05951 | https://github.com/aravindr93/mjrl (hosting repo per memory; MOReL path not verified) | Apache-2.0 (mjrl) | arXiv API 2026-11-11 (title, authors, latest version); code repo and licence via GitHub API 2026-11-11; content from abstract and prior reading, not re-read in full |
| C10 | Yu T.; Kumar A.; Rafailov R.; Rajeswaran A.; Levine S.; Finn C. COMBO: Conservative Offline Model-Based Policy Optimization. NeurIPS 2021 | peer-reviewed conference (NeurIPS, per arXiv comment); arXiv v2 | arXiv:2102.08363 | none verified | unverified | arXiv API 2026-11-11 (title, authors, latest version); content from abstract and prior reading, not re-read in full |
| C11 | Wang Z.; Hunt J.J.; Zhou M. Diffusion Policies as an Expressive Policy Class for Offline Reinforcement Learning. ICLR 2023 | peer-reviewed conference (ICLR, per arXiv comment); arXiv v3 | arXiv:2208.06193 | https://github.com/Zhendong-Wang/Diffusion-Policies-for-Offline-RL | Apache-2.0 | arXiv API 2026-11-11 (title, authors, latest version); code repo and licence via GitHub API 2026-11-11; content from abstract and prior reading, not re-read in full |
| C12 | Hansen-Estruch P.; Kostrikov I.; Janner M.; Kuba J.G.; Levine S. IDQL: Implicit Q-Learning as an Actor-Critic Method with Diffusion Policies | preprint (arXiv v2); no peer-reviewed venue verified | arXiv:2304.10573 | https://github.com/philippe-eecs/IDQL | no licence file detected by GitHub API | arXiv API 2026-11-11 (title, authors, latest version); code repo and licence via GitHub API 2026-11-11; content from abstract and prior reading, not re-read in full |
| C13 | Park S.; Li Q.; Levine S. Flow Q-Learning. ICML 2025, PMLR 267 | peer-reviewed conference (ICML); arXiv v2 | arXiv:2502.02538 | https://github.com/seohongpark/fql | MIT | arXiv abstract page fetched 2026-11-11 (title, authors, version history); PMLR v267 listing via web search; code repo and licence via GitHub API 2026-11-11 |
| D01 | Precup D.; Sutton R.S.; Singh S. Eligibility Traces for Off-Policy Policy Evaluation. ICML 2000, pp. 759-766 | peer-reviewed conference (ICML) | no DOI identified | none | n/a | web search 2026-11-11 (dblp, mlanthology and author bibliography listings agree on venue and pages) |
| D02 | Jiang N.; Li L. Doubly Robust Off-policy Value Evaluation for Reinforcement Learning. ICML 2016 | peer-reviewed conference (ICML, per arXiv comment); arXiv v3 | arXiv:1511.03722 | none identified | unverified | arXiv API 2026-11-11 (title, authors, latest version); content from abstract and prior reading, not re-read in full |
| D03 | Thomas P.S.; Brunskill E. Data-Efficient Off-Policy Policy Evaluation for Reinforcement Learning. ICML 2016, PMLR 48:2139-2148 | peer-reviewed conference (ICML); arXiv v1 | arXiv:1604.00923 | none identified | unverified | arXiv API 2026-11-11 (title, authors, latest version); PMLR page fetched; content from abstract and prior reading, not re-read in full |
| D04 | Le H.M.; Voloshin C.; Yue Y. Batch Policy Learning under Constraints. ICML 2019, PMLR 97:3703-3712 | peer-reviewed conference (ICML); arXiv v1 | arXiv:1903.08738 | none verified | unverified | arXiv API 2026-11-11 (title, authors, latest version); PMLR page fetched; content from abstract and prior reading, not re-read in full |
| D05 | Paine T.L.; Paduraru C.; Michi A.; Gulcehre C.; Zolna K.; Novikov A.; Wang Z.; de Freitas N. Hyperparameter Selection for Offline Reinforcement Learning | preprint (arXiv v1); no peer-reviewed venue verified | arXiv:2007.09055 | none identified | unverified | arXiv API 2026-11-11 (title, authors, latest version) (authors 7-8 from memory; API listing truncated at six); content from abstract and prior reading, not re-read in full |
| D06 | Thomas P.S.; Theocharous G.; Ghavamzadeh M. High-Confidence Off-Policy Evaluation. AAAI 2015 | peer-reviewed conference (AAAI) | 10.1609/aaai.v29i1.9541 | none identified | n/a | Crossref API 2026-11-11 (title, authors, venue, year); web search (AAAI page, author PDF) |
| D07 | Laroche R.; Trichelair P.; Tachet des Combes R. Safe Policy Improvement with Baseline Bootstrapping. ICML 2019 | peer-reviewed conference (ICML, per arXiv comment); arXiv v5 | arXiv:1712.06924 | https://github.com/RomainLaroche/SPIBB | MIT | arXiv API 2026-11-11 (title, authors, latest version); code repo and licence via GitHub API 2026-11-11; content from abstract and prior reading, not re-read in full |
| D08 | Fu J.; Norouzi M.; Nachum O.; Tucker G.; Wang Z.; Novikov A.; et al. Benchmarks for Deep Off-Policy Evaluation. ICLR 2021 | peer-reviewed conference (ICLR, per arXiv comment); arXiv v1 | arXiv:2103.16596 | https://github.com/google-research/deep_ope | Apache-2.0 (archived) | arXiv API 2026-11-11 (title, authors, latest version) (author list truncated at six by reviewer's script); code repo and licence via GitHub API 2026-11-11; content from abstract and prior reading, not re-read in full |
| D09 | Voloshin C.; Le H.M.; Jiang N.; Yue Y. Empirical Study of Off-Policy Policy Evaluation for Reinforcement Learning. NeurIPS 2021 Datasets and Benchmarks | peer-reviewed conference track (NeurIPS D&B); arXiv v3 | arXiv:1911.06854 | https://github.com/clvoloshin/COBS | no licence file detected by GitHub API | arXiv API 2026-11-11 (title, authors, latest version); NeurIPS D&B proceedings via web search; code repo and licence via GitHub API 2026-11-11; content from abstract and prior reading, not re-read in full |
| D10 | Liu P.; Zhao L.; Agarwal S.; Liu J.; Huang A.; Amortila P.; Jiang N. Model Selection for Off-policy Evaluation: New Algorithms and Experimental Protocol | arXiv v3 (2025-10-24); NeurIPS 2025 per web-search summary, not confirmed on arXiv page | arXiv:2502.08021 | none verified | unverified | arXiv abstract page fetched 2026-11-11 (title, authors, version history); web search |
| E01 | Alshiekh M.; Bloem R.; Ehlers R.; Konighofer B.; Niekum S.; Topcu U. Safe Reinforcement Learning via Shielding. AAAI 2018 | peer-reviewed conference (AAAI); arXiv v2 | 10.1609/aaai.v32i1.11797; arXiv:1708.08611 | none verified | unverified | Crossref API 2026-11-11 (title, authors, venue, year); arXiv API; content from abstract and prior reading, not re-read in full |
| E02 | Jansen N.; Konighofer B.; Junges S.; Serban A.; Bloem R. Safe Reinforcement Learning Using Probabilistic Shields. CONCUR 2020, LIPIcs 171, 3:1-3:16 (invited paper) | conference proceedings (CONCUR, invited paper); arXiv v2 under title 'Safe Reinforcement Learning via Probabilistic Shields' | 10.4230/LIPIcs.CONCUR.2020.3; arXiv:1807.06096 | none verified | unverified | web search 2026-11-11 (Dagstuhl DROPS record); arXiv API; Crossref lookup of DOI failed |
| E03 | Garcia J.; Fernandez F. A Comprehensive Survey on Safe Reinforcement Learning. JMLR 16(42):1437-1480 | peer-reviewed journal (JMLR) | https://jmlr.org/papers/v16/garcia15a.html | none | n/a | web search 2026-11-11 (JMLR page listing) |
| E04 | Gu S.; Yang L.; Du Y.; Chen G.; Walter F.; Wang J.; Knoll A. A Review of Safe Reinforcement Learning: Methods, Theories, and Applications. IEEE TPAMI 46(12):11216-11235 | peer-reviewed journal (IEEE TPAMI); arXiv v5 (title 'Methods, Theory and Applications') | 10.1109/TPAMI.2024.3457538; arXiv:2205.10330 | none verified | unverified | Crossref API 2026-11-11 (title, authors, venue, year); arXiv API |
| E05 | Dalal G.; Dvijotham K.; Vecerik M.; Hester T.; Paduraru C.; Tassa Y. Safe Exploration in Continuous Action Spaces | preprint (arXiv v1); no peer-reviewed venue identified | arXiv:1801.08757 | none identified | n/a | arXiv API 2026-11-11 (title, authors, latest version); web search; content from abstract and prior reading, not re-read in full |
| E06 | Huang S.; Ontanon S. A Closer Look at Invalid Action Masking in Policy Gradient Algorithms. FLAIRS 35 | peer-reviewed conference (FLAIRS); arXiv v3 | 10.32473/flairs.v35i.130584; arXiv:2006.14171 | https://github.com/vwxyzjn/invalid-action-masking | MIT | arXiv API 2026-11-11 (title, authors, latest version); code repo and licence via GitHub API 2026-11-11; content from abstract and prior reading, not re-read in full |
| E07 | Fujita Y.; Maeda S. Clipped Action Policy Gradient. ICML 2018 | peer-reviewed conference (ICML, per arXiv comment); arXiv v2 | arXiv:1802.07564 | https://github.com/pfnet-research/capg | MIT | arXiv API 2026-11-11 (title, authors, latest version); code repo and licence via GitHub API 2026-11-11; content from abstract and prior reading, not re-read in full |
| E08 | Krasowski H.; Thumm J.; Muller M.; Schafer L.; Wang X.; Althoff M. Provably Safe Reinforcement Learning: Conceptual Analysis, Survey, and Benchmarking. TMLR 2023 | peer-reviewed journal (TMLR, per arXiv journal-ref); arXiv v3 | arXiv:2205.06750 | none verified | unverified | arXiv API 2026-11-11 (title, authors, latest version); content from abstract and prior reading, not re-read in full |
| E09 | Katz G.; Barrett C.; Dill D.L.; Julian K.; Kochenderfer M.J. Reluplex: An Efficient SMT Solver for Verifying Deep Neural Networks. CAV 2017, LNCS pp. 97-117 | peer-reviewed conference (CAV); arXiv v2 (extended) | 10.1007/978-3-319-63387-9_5; arXiv:1702.01135 | none verified | unverified | Crossref API 2026-11-11 (title, authors, venue, year); arXiv API; content from abstract and prior reading, not re-read in full |
| E10 | Katz G.; Huang D.A.; Ibeling D.; Julian K.; Lazarus C.; Lim R.; Shah P.; et al. The Marabou Framework for Verification and Analysis of Deep Neural Networks. CAV 2019, LNCS pp. 443-452 | peer-reviewed conference (CAV); successor 'Marabou 2.0' arXiv:2401.14461 v2 (CAV 2024 per arXiv comment) | 10.1007/978-3-030-25540-4_26 | https://github.com/NeuralNetworkVerification/Marabou | custom/unrecognised by GitHub (NOASSERTION); not inspected | Crossref API 2026-11-11 (title, authors, venue, year) (author list truncated at seven by reviewer's script); arXiv API for 2.0; code repo and licence via GitHub API 2026-11-11 |
| E11 | Wang S.; Zhang H.; Xu K.; Lin X.; Jana S.; Hsieh C.-J.; Kolter J.Z. Beta-CROWN: Efficient Bound Propagation with Per-neuron Split Constraints for Complete and Incomplete Neural Network Robustness Verification. NeurIPS 2021 | peer-reviewed conference (NeurIPS, per arXiv comment); arXiv v2 | arXiv:2103.06624 | https://github.com/Verified-Intelligence/alpha-beta-CROWN | custom/unrecognised by GitHub (NOASSERTION); not inspected | arXiv API 2026-11-11 (title, authors, latest version) (seventh author from memory; API listing truncated at six); code repo and licence via GitHub API 2026-11-11; content from abstract and prior reading, not re-read in full |
| E12 | Galesloot M.F.L.; Rhemrev T.; Jansen N. Robust Probabilistic Shielding for Safe Offline Reinforcement Learning | arXiv v1 (2026-05-11); AAMAS 2026 per repository literature-update-2026-09-28, not re-verified | arXiv:2605.10293 | not established | unverified | arXiv API 2026-11-11 (title, authors, latest version); content from repository note of 2026-09-28 |
| F01 | Fama E.F. Efficient Capital Markets: A Review of Theory and Empirical Work. Journal of Finance 25(2):383-417 | peer-reviewed journal (JF) | 10.1111/j.1540-6261.1970.tb00518.x | none | n/a | Crossref API 2026-11-11 (title, authors, venue, year) (Crossref lists a discussant as co-author; Fama is sole author per the earlier repository matrix) |
| F02 | Liu Y.; Tsyvinski A.; Wu X. Common Risk Factors in Cryptocurrency. Journal of Finance 77(2):1133-1177 | peer-reviewed journal (JF) | 10.1111/jofi.13119 | none identified | n/a | Crossref API 2026-11-11 (title, authors, venue, year); in earlier repository matrices; content from abstract and prior reading, not re-read in full |
| F03 | He S.; Manela A.; Ross O.; von Wachter V. Fundamentals of Perpetual Futures | preprint (arXiv v7, updated 2026-09-17); peer-reviewed status unverified | arXiv:2212.06888 | none identified | n/a | arXiv API 2026-11-11 (title, authors, latest version); web search summary |
| F04 | Bailey D.H.; Lopez de Prado M. The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality. Journal of Portfolio Management 40(5):94-107 | peer-reviewed journal (JPM) | 10.3905/jpm.2014.40.5.094 | none identified | n/a | Crossref API 2026-11-11 (title, authors, venue, year) |
| F05 | Bailey D.H.; Borwein J.; Lopez de Prado M.; Zhu Q.J. The probability of backtest overfitting. Journal of Computational Finance | peer-reviewed journal (JCF); volume/pages not returned by Crossref; year per earlier repository matrix (Crossref issue date 2016-09) | 10.21314/JCF.2016.322 | none identified | n/a | Crossref API 2026-11-11 (title, authors, venue, year) |
| F06 | Harvey C.R.; Liu Y.; Zhu H. ... and the Cross-Section of Expected Returns. Review of Financial Studies 29(1):5-68 | peer-reviewed journal (RFS) | 10.1093/rfs/hhv059 | none | n/a | Crossref API 2026-11-11 (title, authors, venue, year); content from abstract and prior reading, not re-read in full |
| N01 | Henderson P.; Islam R.; Bachman P.; Pineau J.; Precup D.; Meger D. Deep Reinforcement Learning that Matters. AAAI 2018 | peer-reviewed conference (AAAI); arXiv v3 | 10.1609/aaai.v32i1.11694; arXiv:1709.06560 | none verified | unverified | Crossref API 2026-11-11 (title, authors, venue, year); arXiv API; content from abstract and prior reading, not re-read in full |
| N02 | Agarwal R.; Schwarzer M.; Castro P.S.; Courville A.; Bellemare M.G. Deep Reinforcement Learning at the Edge of the Statistical Precipice. NeurIPS 2021 | peer-reviewed conference (NeurIPS, Outstanding Paper per arXiv comment); arXiv v4 | arXiv:2108.13264 | https://github.com/google-research/rliable | Apache-2.0 (archived) | arXiv API 2026-11-11 (title, authors, latest version); code repo and licence via GitHub API 2026-11-11; content from abstract and prior reading, not re-read in full |
| N03 | Engstrom L.; Ilyas A.; Santurkar S.; Tsipras D.; Janoos F.; Rudolph L.; Madry A. Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO. ICLR 2020 | peer-reviewed conference (ICLR, per arXiv comment); arXiv v1 | arXiv:2005.12729 | https://github.com/MadryLab/implementation-matters | MIT | arXiv abstract page fetched 2026-11-11 (title, authors, version history); web search; code repo and licence via GitHub API 2026-11-11 |
| N04 | Huang S.; Dossa R.F.J.; Raffin A.; Kanervisto A.; Wang W. The 37 Implementation Details of Proximal Policy Optimization. ICLR Blog Track 2022 | ICLR Blog Track post (reviewed blog, not a conference paper) | https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/ | https://github.com/vwxyzjn/ppo-implementation-details | custom/unrecognised by GitHub (NOASSERTION); not inspected | web search 2026-11-11 (blog-track URL, authors, structure of the 37 details); code repo and licence via GitHub API 2026-11-11 |
| N05 | Gort B.J.D.; Liu X.-Y.; Sun X.; Gao J.; Chen S.; Wang C.D. Deep Reinforcement Learning for Cryptocurrency Trading: Practical Approach to Address Backtest Overfitting | preprint (arXiv v6, 2023-01-31); peer-reviewed venue unverified | arXiv:2209.05559 | unverified | unverified | arXiv API 2026-11-11 (title, authors, latest version); web search; content from abstract and prior reading, not re-read in full |
| N06 | Cliff D.; Rollins M. Methods Matter: A Trading Agent with No Intelligence Routinely Outperforms AI-Based Traders. IEEE SSCI 2020, pp. 392-399 | peer-reviewed conference (IEEE SSCI); arXiv:2011.14346 | 10.1109/SSCI47803.2020.9308172 | unverified | unverified | Crossref API 2026-11-11 (title, authors, venue, year); web search (arXiv id) |
| N07 | Pippas N.; Ludvig E.A.; Turkay C. The Evolution of Reinforcement Learning in Quantitative Finance: A Survey. ACM Computing Surveys (accepted 2025-04-18 per arXiv page) | peer-reviewed journal (ACM CSUR, per arXiv page note); arXiv v3 | 10.1145/3733714; arXiv:2408.10932 | none | n/a | arXiv abstract page fetched 2026-11-11 (title, authors, version history); web search; DOI not resolved via Crossref |
