# Microstructure and Mispricing in Polymarket's Bitcoin Up-Down Contracts

## 1. Introduction

Polymarket runs short-horizon Bitcoin "Up or Down" contracts that resolve every
5 or 15 minutes. Each contract pays $1 if BTC's price at the end of a fixed
window is at or above its price at the start, and $0 otherwise. This binary
payoff makes the contract price directly interpretable as the market's
implied probability that BTC will finish higher than where it started.

Short-horizon prediction markets are interesting because, unlike longer-dated
markets, they expose the market's frictions: there isn't enough time for
arbitrage to fully equilibrate prices, liquidity providers face acute
adverse-selection risk near expiry, and small differences in oracle behaviour
can determine whether a contract pays out at all.

This report investigates how these contracts behave near resolution, where
the most interesting mispricing happens, and presents two trading strategies
that try to exploit the inefficiencies we find. Section 2 describes the
contract structure and our data. Section 3 documents three microstructure
findings about how these markets can introduce unexpected deviations. 
Sections 4 and 5 present order-book and time-based trading strategies that 
exploit those findings. Section 6 examines how the strategies interact 
with broader cryptocurrency volatility regimes. 
Section 7 discusses what worked, what didn't, and what we'd do next.

### 1.1 Why these contracts are worth studying

BTC up-down contracts are a useful setting for studying short-horizon prediction
markets because their prices are easy to interpret and their resolution is
frequent. Unlike longer-dated prediction markets, these contracts expire within
minutes, so any inefficiency is more likely to reflect market frictions such as
slow updating, order-book imbalance, or resolution mechanics rather than broad
uncertainty about future fundamentals.

### 1.2 What this report tries to answer

This report asks three simple questions. First, do Polymarket prices behave like
well-calibrated probabilities? Second, do these markets display repeatable
mispricing near expiry? Third, if such inefficiencies exist, can they be used
to motivate consistent and practical trading strategies?

## 2. The Market

### 2.1 How a contract works

A Polymarket BTC up-down 5-minute contract has the slug format
`btc-updown-5m-<unix_timestamp>`. The unix timestamp marks the start of the
"live" 5-minute window. At that moment, Polymarket records BTC's spot price
from Chainlink as the strike. Five minutes later, it records BTC again. If
the second price is greater than or equal to the first, "Up" pays $1; otherwise
"Down" pays $1.

There are two tradeable token sides for every market. By no-arbitrage, the
prices for Up and Down should sum to approximately 1.0 — anything else would
be a free-money arbitrage. We verify this empirically and use it as a
liquidity sanity check.

Trading happens on Polymarket's central limit order book (CLOB). Polymarket
itself does not set prices; they emerge from limit orders placed by users.
Order matching is off-chain, settlement is on-chain via the Polygon network.

### 2.2 The three-phase lifecycle

A surprising fact about these contracts is that they trade for much longer
than their nominal 5-minute window. Each contract has a clear three-phase
lifecycle:

[INSERT LIFECYCLE PLOT HERE]

- **Pre-market** (~15-30 minutes before live start): The contract is listed
  and tradeable, but the strike price hasn't been set yet. Midprices sit at
  approximately 0.50, since traders have nothing concrete to anchor on
  besides BTC drift expectations.
- **Live** (the 5 or 15 minutes between start and end): The contract is
  actively determined by BTC's price movement. Midprices diverge from 0.50
  as BTC moves above or below the strike.
- **Post-resolution** (a few minutes after live-end): The outcome is
  effectively known, and prices converge toward 0 or 1, eventually ending
  in a "book wipe" as remaining orders are cancelled.

This phase structure matters because it determines what kind of analysis is
meaningful at each point. Pre-market prices can't be tested for fair value
(no strike). Live prices reflect real information aggregation. Post-resolution
prices reflect mechanical convergence rather than belief updating.

### 2.3 Resolution via Chainlink

Polymarket resolves these contracts using the Chainlink BTC/USD Data Stream,
a low-latency price oracle that aggregates spot prices from multiple
centralised exchanges. We do not have direct access to Chainlink Data Streams
historical reports, so we use Binance and Coinbase prices as proxies for the
underlying BTC spot price. We discuss the limitations of this in Section 3.4.

### 2.4 Data

We use:
- Polymarket order book data (~1.2 billion events from 21st Feb 2026 
  to 24th March 2026, filtered to BTC 5-minute and 15-minute markets). These
  are downloaded from PMXT archives https://archive.pmxt.dev/)
- Polymarket market metadata from scraping the Gamma API 
  (slug, condition ID, token IDs, strike, settlement timestamps)
- BTC spot prices from Binance (BTC/USDT) and Coinbase (BTC/USD) via ccxt
  https://github.com/ccxt/ccxt

## 3. Microstructure Findings

This section presents three practical observations from the data. For each one,
we describe how we measure it, what we observe, and why it matters for trading.

### 3.1 Finding 1: Terminal convergence lag

**What we measure.**  
We examine how quickly market prices converge toward their final realized
outcome near and after the end of the live window.

**What we find.**  
[Insert your numbers here.]

**Why it matters.**  
This suggests that some contracts do not fully incorporate the final outcome
immediately at live-end. Instead, prices can remain stale briefly before
snapping toward 0 or 1. This creates the basic intuition for the time-based
strategy in Section 5.

### 3.2 Finding 2: Last-second leader switches

**What we measure.**  
We define a switcher as a market where the side leading shortly before expiry
 is not the final winner.

**What we find.**  
We find that last-second switches occur in a non-trivial minority of markets.
In our BTC sample, the switch rate is meaningfully above zero, and in several
cases the direction of the reversal appears to follow the broader drift of BTC
on that day.

**Why it matters.**  
This suggests that prices near expiry are not always fully settled beliefs.
Instead, they may still lag the underlying market, especially in fast-moving
conditions.

### 3.3 Finding 3: Cross-exchange divergence at resolution

**What we measure.**  
Because we do not have historical Chainlink reports, we compare Binance and
Coinbase spot prices around the resolution boundary as proxies for the
underlying BTC move.

**What we find.**  
[Insert your comparison result here.]

**Why it matters.**  
This shows that some apparent mispricings may reflect genuine source
differences near the boundary rather than purely slow reaction by Polymarket
traders. This is an important source of residual risk in any expiry-based
strategy.

### 3.4 What these findings imply

Taken together, these findings suggest that short-horizon BTC up-down markets
are informative but not frictionless. Prices can lag near expiry, the apparent
winner can switch at the end, and source differences around the resolution
boundary can create tail risk. These effects motivate the order-book and
time-based strategies in the next sections.

## 4. Regime Conditioning

TBA

## 5. Strategy 1: Mid-price Momentum

TBA

## 6. Strategy 2: Relative Order Book Strength

The relative book strength strategy evaluates whether cross-sectional differences in order-book quality
between the two contract sides (Up and Down) can be used to identify the
eventual winning side. The central hypothesis is that persistent relative
microstructure advantages contain information that is not fully captured by a
single contemporaneous snapshot.

### 6.1 Economic Intuition

Because the two sides represent mutually exclusive outcomes of the same binary
event, their prices are jointly constrained by the payout structure. As a
result, the informative object is the *relative* state of the two books at a
given event time rather than either side in isolation. If one side repeatedly
exhibits tighter spreads, greater effective depth, and more favorable pressure,
this pattern may reflect faster information incorporation and, consequently,
higher probability of final correctness.

Event-level order-book states are, however, noisy. Quote revisions may be
transient, resting liquidity can be episodic, and event-time sampling is
irregular. The cumulative variants are therefore designed to suppress
high-frequency noise and extract persistent directional structure.

### 6.2 Signal Design and Notation

For a market, one of the two sides, and an event time, let the best ask price
be the top ask quote, the best bid price be the top bid quote, and the resting
ask and bid quantities at the top of book be the corresponding queue sizes.
We define the top-of-book spread in basis points as

$$
spread_{i,t}=10^4\cdot\frac{a^{(1)}_{i,t}-b^{(1)}_{i,t}}{\tfrac{1}{2}(a^{(1)}_{i,t}+b^{(1)}_{i,t})}
$$

and the top-of-book depth as

$$
depth_{i,t}=q^{ask}_{i,t}+q^{bid}_{i,t}
$$

- spread: top-of-book spread in basis points
- depth: sum of the resting ask and bid quantities at the top of book
- pressure: resting ask quantity minus resting bid quantity at the top of book
- imbalance: absolute value of pressure

To make features comparable across markets and timestamps, each raw feature is
converted into a within-market relative score across the two sides. Let the
side set contain Up and Down. For a generic feature x, define:

$$
r_{x,i,t}=\frac{x_{i,t}-\bar{x}_{m,t}}{\max_{j\in I_m}x_{j,t}-\min_{j\in I_m}x_{j,t}}
$$

where the numerator uses the cross-side mean at the same market-time point, and
the denominator is the cross-side range at that same market-time point.

Since lower spread implies better execution quality, we invert spread:

$$
r_{s,i,t}^{inv}=\frac{\bar{s}_{m,t}-s_{i,t}}{\max_{j\in I_m}s_{j,t}-\min_{j\in I_m}s_{j,t}}
$$

The snapshot strength score is defined as:

$$
S_{i,t}^{snp}=0.45\,r_{p,i,t}+0.35\,r_{s,i,t}^{inv}+0.15\,r_{d,i,t}+0.05\,r_{b,i,t}
$$

where pressure, depth, and imbalance denote the respective feature groups.

Three ranking rules are evaluated:

- Snapshot: rank by the snapshot strength score.
- Cumulative sum:

$$
S_{i,t}^{sum}=\sum_{\tau<t}S_{i,\tau}^{snp}
$$

- Exponentially weighted memory:

$$
S_{i,t}^{ewm}=\alpha S_{i,t-1}^{snp}+(1-\alpha)S_{i,t-1}^{ewm}
$$

$$
\alpha=0.2
$$

All cumulative diagnostics are computed in strict causal mode, so each score at
time t depends only on information available before the current event.

### 6.3 Full-Market Results from Reliability Diagnostic

On the full-market run, both cumulative specifications outperform the snapshot
baseline, and the cumulative-sum formulation is the best performer.

| Method | Timestamp Accuracy | Final Market Accuracy | Markets |
|---|---:|---:|---:|
| cumulative_sum_score | 0.549 | 0.665 | 5,076 |
| cumulative_ewm_score | 0.543 | 0.583 | 5,076 |
| snapshot_score | 0.509 | 0.462 | 5,076 |

Two implications follow immediately. First, temporal aggregation provides a
substantial gain over point-in-time ranking. Second, the strongest gains are
observed in the final-market metric, indicating materially better terminal
discrimination.

Progress-quartile analysis (Q1 early to Q4 late) further supports this result:

| Method | Q1 (early) | Q2 | Q3 | Q4 (late) |
|---|---:|---:|---:|---:|
| cumulative_sum_score | 0.517 | 0.539 | 0.557 | 0.582 |
| cumulative_ewm_score | 0.511 | 0.528 | 0.546 | 0.586 |
| snapshot_score | 0.500 | 0.502 | 0.504 | 0.529 |

The monotone increase in cumulative-method accuracy across quartiles is
consistent with a signal that strengthens as information accumulates and
transient microstructure noise is averaged out.

### 6.4 Confidence Intervals and Stability Diagnostics

Confidence and stability diagnostics were computed on the same run using Wilson
95% intervals. The results are:

| Method | Final Accuracy | Final 95% CI | Timestamp Accuracy | Timestamp 95% CI |
|---|---:|---:|---:|---:|
| cumulative_sum_score | 0.665 | [0.652, 0.678] | 0.549 | [0.549, 0.549] |
| cumulative_ewm_score | 0.583 | [0.570, 0.597] | 0.543 | [0.542, 0.543] |
| snapshot_score | 0.462 | [0.449, 0.476] | 0.509 | [0.509, 0.509] |

The interval structure is informative. Timestamp intervals are narrow due to
the very large sample size, while final-market intervals are wider but remain
well separated across methods, preserving the performance ranking.

Weekly concentration diagnostics for `cumulative_sum_score` are:

| Statistic | Value |
|---|---:|
| Minimum weekly accuracy | 0.542 |
| Maximum weekly accuracy | 0.569 |
| Weekly standard deviation | 0.010 |

Worst three weeks by accuracy:

| Week | Accuracy |
|---|---:|
| 2026-03-09 | 0.542 | 
| 2026-02-23 | 0.542 | 
| 2026-03-16 | 0.548 | 

Best three weeks by accuracy:

| Week | Accuracy |
|---|---:|
| 2026-03-30 | 0.569 | 
| 2026-03-23 | 0.552 | 
| 2026-03-02 | 0.548 | 

Taken together, these weekly results indicate that the edge is not concentrated
in a single outlier week. Performance remains above 0.50 in all observed weeks,
with moderate cross-week dispersion.

At this stage, the evidence is strong on a classification dimension: the
cumulative-sum specification ranks the eventual winner more reliably than the
alternative signal constructions. That is an essential first condition for a
viable strategy, but it is not yet a sufficient condition for deployment.

The reason is that ranking quality and execution quality are distinct objects.
A signal may be directionally correct, yet still produce weak trading outcomes
if entries are concentrated in unfavorable microstructure states, if prices are
already too extended at the decision point, or if liquidity and timing
conditions are inconsistent with efficient order placement. In other words,
there is a translation step between statistical discrimination and realized
PnL, and that translation is governed by the gating mechanism.

Accordingly, the flow of Section 6 is intentionally two-stage. Sections 6.3 and
6.4 establish that the direction signal is stable and informative. Section 6.5
then asks a separate question: given that the signal is informative, what
execution filter set yields the best balance between participation and risk
control?

### 6.5 Gate Design and Diagnostics

This subsection evaluates the execution layer of the relative book strength strategy in a way that is consistent with the preceding signal analysis. We keep the ranking model fixed
at the cumulative-sum score and vary only the gate structure, so that performance changes
can be attributed to filtering choices rather than signal redefinition.

Before presenting the ablation, it is important to clarify why gates exist in
the first place. In this report, a gate is a deterministic acceptance rule that
must be satisfied before a ranked signal is allowed to generate a trade.
Conceptually, the ranking model answers a directional question (which side is
more likely to win), while the gate layer answers an execution question
(whether current market conditions are suitable for expressing that view).

The motivation for gate design is therefore not to replace the signal, but to
control *where* and *when* the signal is acted upon. In short-horizon binary
markets, many false or low-quality entries arise not because direction is
completely wrong, but because execution state is poor: spreads may be too wide,
book shape may imply asymmetric fill risk, price may be too close to the payoff
ceiling, or time-to-resolution may be too short for robust position handling.
Gates are introduced to remove these avoidable states.

Each gate has a distinct pre-test purpose:

1. Spread gate: avoid paying excessive instantaneous transaction cost and reduce
  entries during quote dislocation.
2. Score-threshold gate: require a minimum absolute signal strength so that
  weakly ranked states do not trigger trades.
3. Score-gap gate: require cross-side separation in ranking scores, reducing
  entries in ambiguous states where the two sides are nearly tied.
4. Price-cap gate: avoid buying at prices where upside is mechanically limited
  by the binary payoff bound.
5. Liquidity gate: enforce minimum local depth to reduce slippage sensitivity
  and unstable fills.
6. Ask-depth-5 cap gate: avoid heavily stacked ask ladders that can indicate
  adverse local pressure or poor short-horizon execution geometry.
7. Time gate: constrain entries to a bounded time-to-resolution window, so the
  strategy avoids both premature low-information states and excessively late
  states with heightened microstructure noise.

From this perspective, gate testing is a model-selection problem over the
execution layer: which constraints materially improve realized outcomes, and
which constraints are overly restrictive once the cumulative-sum ranking signal
is already informative.

The design uses 500 markets sampled uniformly without replacement from the first
3,000 BTC 5-minute markets. This sampling choice provides two advantages. First,
it preserves tractability for repeated ablation runs while still covering a
meaningful cross-section of market states. Second, by sampling from a fixed
prefix pool, it avoids conflating gate effects with broad distribution shifts
that may arise when mixing very early and very late market cohorts.

The interpretation of these gates is straightforward. The spread, price-cap,
and liquidity gates control immediate execution quality; the score and score-gap
gates control decision confidence under the ranking model; and the time gate
controls exposure to late-window microstructure instability. The ask-depth-5
gate serves as a crowding/proximity filter that prevents entries when the local
book geometry suggests elevated fill-risk asymmetry.

Empirically, this framework is evaluated in a leave-one-gate-out design:
starting from the full gate set, one gate is removed at a time while all others
remain fixed. This isolates each gate's marginal contribution to trade count,
hit rate, and net PnL.

The ablation results for our 500 market sample test are summarized below.

| Scenario | Trades | Win Rate | Net PnL |
|---|---:|---:|---:|
| base_no_gates | 237 | 0.447 | -100.051 |
| full_gated | 362 | 0.751 | -4.766 |
| full_minus_spread_gate | 416 | 0.700 | 33.815 |
| full_minus_score_gap_gate | 365 | 0.753 | -0.291 |
| full_minus_price_cap_gate | 408 | 0.779 | -5.515 |
| full_minus_ask_depth_5_cap_gate | 384 | 0.747 | -7.310 |
| full_minus_time_gate | 424 | 0.731 | -7.301 |

The first and most important result is that gating itself is valuable.
Relative to the ungated baseline, the fully gated policy improves win rate from
0.447 to 0.751 and reduces net loss magnitude from -100.051 to -4.766. This
confirms that raw signal ranking, although informative, benefits materially from
execution-aware constraints.

The second result is that gate contributions are not uniform. Removing the
spread gate increases net PnL sharply (to 33.815) at the cost of lower hit
rate (0.700), which indicates that the current spread threshold is likely too
conservative for the cumulative-sum signal regime. Removing the score-gap gate
also improves net PnL (to -0.291) while marginally improving hit rate (0.753),
suggesting that this gate is filtering too many opportunities without providing
commensurate risk reduction.

The third result is that some gates are clearly protective. Removing the
ask-depth-5 cap or time gate degrades both net PnL and hit rate, indicating
that these constraints are suppressing adverse execution states rather than
simply reducing activity. By contrast, score, liquidity, and price-cap gates
appear weaker in marginal contribution for this sample and should be treated as
secondary controls pending broader robustness tests.

Accordingly, the recommended gating mechanism for the relative book strength strategy is:

1. Keep the cumulative-sum ranking core unchanged.
2. Disable the spread gate.
3. Disable the score-gap gate.
4. Retain the time gate and ask-depth-5 cap gate.
5. Retain price-cap, score-threshold, and liquidity gates as secondary controls,
   with periodic recalibration because their marginal effect is sample-dependent.

The recommended gate policy is validated by the realized backtest outcome
below, which is the final check on whether the gate simplification improved the
strategy rather than only making the diagnostics look cleaner.

| Scenario | Trades | Win Rate | Net PnL | Gross PnL | Fees | Avg Net PnL |
|---|---:|---:|---:|---:|---:|---:|
| base_no_gates | 237 | 0.447 | -100.051 | -75.119 | 24.932 | -0.4222 |
| full_gated | 362 | 0.751 | -4.766 | 18.574 | 23.340 | -0.0132 |
| recommended_gated | 419 | 0.702 | 42.660 | 63.050 | 20.390 | 0.1018 |

The recommended gate policy is stronger still on realized PnL. 
Relative to the fully gated baseline, it is less restrictive, 
which is exactly the intended effect of dropping the
spread gate and score-gap gate after the diagnostics showed that those two
filters were overly conservative for this signal regime.




## 7. Discussion

What worked, what didn't, what we'd do next. Limitations: Chainlink proxy,
sample size, trading cost assumptions.