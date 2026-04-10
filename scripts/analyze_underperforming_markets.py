#!/usr/bin/env python3
"""
Holistic analysis of underperforming markets.

Compares characteristics of markets in negative vs positive batches:
- Market symbol patterns
- Trade expectancy distributions
- Entry quality metrics
- Signal strength
- Market concentration and clustering
"""
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

WORKSPACE_ROOT = Path(__file__).parent.parent
ARTIFACTS = WORKSPACE_ROOT / "reports" / "artifacts"

metrics_csv = ARTIFACTS / "relative_book_degradation_market_metrics.csv"

print("=" * 100)
print("UNDERPERFORMING MARKETS: HOLISTIC ANALYSIS")
print("=" * 100)

# Load market metrics
df_metrics = pd.read_csv(metrics_csv)
df_oos = df_metrics[df_metrics["split"] == "oos_remaining"].copy()

# Reconstruct batch assignments
batch_market_counts = [72, 73, 86, 86, 81, 86, 80, 80, 84, 89, 57]
df_oos_sorted = df_oos.sort_values("market_id").reset_index(drop=True)

batch_assignments = []
cumulative = 0
for batch_num, count in enumerate(batch_market_counts, 1):
    for idx in range(cumulative, min(cumulative + count, len(df_oos_sorted))):
        batch_assignments.append(batch_num)
    cumulative += count

df_oos_sorted["batch"] = batch_assignments[:len(df_oos_sorted)]

# Define positive and negative batches
positive_batches = [1, 2, 3, 5, 11]
negative_batches = [4, 6, 7, 8, 9, 10]

df_positive = df_oos_sorted[df_oos_sorted["batch"].isin(positive_batches)]
df_negative = df_oos_sorted[df_oos_sorted["batch"].isin(negative_batches)]

print(f"\nPositive batches: {len(df_positive)} markets")
print(f"Negative batches: {len(df_negative)} markets")

# ============================================================================
# 1. STATISTICAL COMPARISON
# ============================================================================
print("\n" + "=" * 100)
print("1. STATISTICAL COMPARISON: POSITIVE vs NEGATIVE BATCHES")
print("=" * 100)

metrics_to_compare = ["trades", "hit_rate", "expectancy", "avg_slippage_bps", "gross_pnl", "net_pnl"]

comparison_table = []
for metric in metrics_to_compare:
    pos_mean = df_positive[metric].mean()
    pos_std = df_positive[metric].std()
    pos_median = df_positive[metric].median()

    neg_mean = df_negative[metric].mean()
    neg_std = df_negative[metric].std()
    neg_median = df_negative[metric].median()

    diff_mean = neg_mean - pos_mean
    diff_pct = (diff_mean / pos_mean * 100) if pos_mean != 0 else 0

    comparison_table.append({
        "Metric": metric,
        "Pos Mean": pos_mean,
        "Neg Mean": neg_mean,
        "Diff": diff_mean,
        "Diff %": diff_pct,
    })

df_comparison = pd.DataFrame(comparison_table)
print("\n" + df_comparison.to_string(index=False))

# ============================================================================
# 2. EXPECTANCY ANALYSIS
# ============================================================================
print("\n" + "=" * 100)
print("2. EXPECTANCY DISTRIBUTION")
print("=" * 100)

expectancy_bins = np.linspace(df_oos_sorted["expectancy"].min(), df_oos_sorted["expectancy"].max(), 11)
print("\nPositive batches expectancy distribution:")
pos_dist, _ = np.histogram(df_positive["expectancy"], bins=expectancy_bins)
for i, count in enumerate(pos_dist):
    bar = "█" * max(1, int(count / len(df_positive) * 50))
    pct = count / len(df_positive) * 100
    print(f"  [{expectancy_bins[i]:7.3f} to {expectancy_bins[i+1]:7.3f}): {bar} {pct:5.1f}% ({int(count)} mkts)")

print("\nNegative batches expectancy distribution:")
neg_dist, _ = np.histogram(df_negative["expectancy"], bins=expectancy_bins)
for i, count in enumerate(neg_dist):
    bar = "█" * max(1, int(count / len(df_negative) * 50))
    pct = count / len(df_negative) * 100
    print(f"  [{expectancy_bins[i]:7.3f} to {expectancy_bins[i+1]:7.3f}): {bar} {pct:5.1f}% ({int(count)} mkts)")

# Quantiles
print("\nExpectancy percentiles:")
print(f"  Positive: Q1={df_positive['expectancy'].quantile(0.25):.4f}, "
      f"Median={df_positive['expectancy'].median():.4f}, "
      f"Q3={df_positive['expectancy'].quantile(0.75):.4f}")
print(f"  Negative: Q1={df_negative['expectancy'].quantile(0.25):.4f}, "
      f"Median={df_negative['expectancy'].median():.4f}, "
      f"Q3={df_negative['expectancy'].quantile(0.75):.4f}")

# ============================================================================
# 3. MARKET SYMBOL ANALYSIS
# ============================================================================
print("\n" + "=" * 100)
print("3. MARKET SYMBOL & TOKEN PATTERN ANALYSIS")
print("=" * 100)

# Extract market symbol (first few chars or token identifiers from market_id)
# Polymarket IDs are hex, but we can group by first few chars (exchange routing info)
# Sample market IDs to understand pattern
print("\nSample market IDs (positive batches):")
for mid in df_positive["market_id"].head(3).values:
    print(f"  {mid}")

print("\nSample market IDs (negative batches):")
for mid in df_negative["market_id"].head(3).values:
    print(f"  {mid}")

# Group by first 4 chars to find exchange/token clusters
def extract_prefix(mid):
    return mid[:4] if isinstance(mid, str) else "unknown"

df_positive["market_prefix"] = df_positive["market_id"].apply(extract_prefix)
df_negative["market_prefix"] = df_negative["market_id"].apply(extract_prefix)

pos_prefixes = Counter(df_positive["market_prefix"])
neg_prefixes = Counter(df_negative["market_prefix"])

print("\nMarket prefix distribution (positive batches, top 10):")
for prefix, count in pos_prefixes.most_common(10):
    pct = count / len(df_positive) * 100
    print(f"  {prefix}: {count:3d} markets ({pct:5.1f}%)")

print("\nMarket prefix distribution (negative batches, top 10):")
for prefix, count in neg_prefixes.most_common(10):
    pct = count / len(df_negative) * 100
    print(f"  {prefix}: {count:3d} markets ({pct:5.1f}%)")

# ============================================================================
# 4. HIT RATE DISTRIBUTION
# ============================================================================
print("\n" + "=" * 100)
print("4. HIT RATE DISTRIBUTION (WIN RATE BY MARKET)")
print("=" * 100)

hit_rate_bins = np.arange(0, 1.1, 0.1)
print("\nPositive batches hit rate distribution:")
pos_hit_dist, _ = np.histogram(df_positive["hit_rate"], bins=hit_rate_bins)
for i, count in enumerate(pos_hit_dist):
    bar = "█" * max(1, int(count / len(df_positive) * 50))
    pct = count / len(df_positive) * 100
    print(f"  [{hit_rate_bins[i]*100:5.0f}% to {hit_rate_bins[i+1]*100:5.0f}%): {bar} {pct:5.1f}% ({int(count)} mkts)")

print("\nNegative batches hit rate distribution:")
neg_hit_dist, _ = np.histogram(df_negative["hit_rate"], bins=hit_rate_bins)
for i, count in enumerate(neg_hit_dist):
    bar = "█" * max(1, int(count / len(df_negative) * 50))
    pct = count / len(df_negative) * 100
    print(f"  [{hit_rate_bins[i]*100:5.0f}% to {hit_rate_bins[i+1]*100:5.0f}%): {bar} {pct:5.1f}% ({int(count)} mkts)")

# Markets with 0 or 1 trades
print("\nMarkets with single trade (high variance):")
print(f"  Positive batches: {(df_positive['trades'] == 1).sum()} / {len(df_positive)} markets "
      f"({(df_positive['trades'] == 1).sum() / len(df_positive) * 100:.1f}%)")
print(f"  Negative batches: {(df_negative['trades'] == 1).sum()} / {len(df_negative)} markets "
      f"({(df_negative['trades'] == 1).sum() / len(df_negative) * 100:.1f}%)")

# ============================================================================
# 5. TRADE COUNT ANALYSIS
# ============================================================================
print("\n" + "=" * 100)
print("5. TRADE COUNT DISTRIBUTION")
print("=" * 100)

print("\nPositive batches trade count:")
print(f"  Mean: {df_positive['trades'].mean():.2f}, Median: {df_positive['trades'].median():.0f}, "
      f"Std: {df_positive['trades'].std():.2f}")
print(f"  Range: {df_positive['trades'].min():.0f} - {df_positive['trades'].max():.0f}")

print("\nNegative batches trade count:")
print(f"  Mean: {df_negative['trades'].mean():.2f}, Median: {df_negative['trades'].median():.0f}, "
      f"Std: {df_negative['trades'].std():.2f}")
print(f"  Range: {df_negative['trades'].min():.0f} - {df_negative['trades'].max():.0f}")

# ============================================================================
# 6. PROFITABILITY PROFILE
# ============================================================================
print("\n" + "=" * 100)
print("6. PROFITABILITY PROFILE")
print("=" * 100)

# Markets that are profitable
profitable_pos = (df_positive["net_pnl"] > 0).sum()
profitable_neg = (df_negative["net_pnl"] > 0).sum()

print(f"\nPositive batches: {profitable_pos} / {len(df_positive)} markets profitable "
      f"({profitable_pos / len(df_positive) * 100:.1f}%)")
print(f"Negative batches: {profitable_neg} / {len(df_negative)} markets profitable "
      f"({profitable_neg / len(df_negative) * 100:.1f}%)")

# Breakdown of P&L
print("\nPositive batches PnL breakdown:")
print(f"  Total net: ${df_positive['net_pnl'].sum():.2f}")
print(f"  Avg per market: ${df_positive['net_pnl'].mean():.4f}")
print(f"  Avg per trade: ${(df_positive['net_pnl'].sum() / df_positive['trades'].sum()):.4f}")
print(f"  Avg winning market: ${df_positive[df_positive['net_pnl'] > 0]['net_pnl'].mean():.4f}")
print(f"  Avg losing market: ${df_positive[df_positive['net_pnl'] < 0]['net_pnl'].mean():.4f}")

print("\nNegative batches PnL breakdown:")
print(f"  Total net: ${df_negative['net_pnl'].sum():.2f}")
print(f"  Avg per market: ${df_negative['net_pnl'].mean():.4f}")
print(f"  Avg per trade: ${(df_negative['net_pnl'].sum() / df_negative['trades'].sum()):.4f}")
print(f"  Avg winning market: ${df_negative[df_negative['net_pnl'] > 0]['net_pnl'].mean():.4f}")
print(f"  Avg losing market: ${df_negative[df_negative['net_pnl'] < 0]['net_pnl'].mean():.4f}")

# ============================================================================
# 7. WORST PERFORMING MARKET DETAILS
# ============================================================================
print("\n" + "=" * 100)
print("7. WORST PERFORMING MARKETS (Bottom 20)")
print("=" * 100)

worst_20 = df_oos_sorted.nsmallest(20, "net_pnl")[["market_id", "batch", "trades", "hit_rate", "expectancy", "net_pnl", "gross_pnl"]]
print("\n" + worst_20.to_string(index=False, float_format=lambda x: f"{x:.4f}" if abs(x) < 1 else f"{x:.2f}"))

# ============================================================================
# 8. BEST PERFORMING MARKET DETAILS
# ============================================================================
print("\n" + "=" * 100)
print("8. BEST PERFORMING MARKETS (Top 20)")
print("=" * 100)

best_20 = df_oos_sorted.nlargest(20, "net_pnl")[["market_id", "batch", "trades", "hit_rate", "expectancy", "net_pnl", "gross_pnl"]]
print("\n" + best_20.to_string(index=False, float_format=lambda x: f"{x:.4f}" if abs(x) < 1 else f"{x:.2f}"))

# ============================================================================
# 9. BATCH COMPOSITION ANALYSIS
# ============================================================================
print("\n" + "=" * 100)
print("9. BATCH COMPOSITION: Trading Characteristics")
print("=" * 100)

for batch_num in [6, 8, 4, 10]:  # Top negative performers
    batch_data = df_oos_sorted[df_oos_sorted["batch"] == batch_num]
    print(f"\nBatch {batch_num} (Negative): {len(batch_data)} markets, {batch_data['trades'].sum()} trades, ${batch_data['net_pnl'].sum():.2f} net")
    print(f"  Hit rate: {batch_data['hit_rate'].mean()*100:.2f}% (std {batch_data['hit_rate'].std()*100:.2f}%)")
    print(f"  Expectancy: {batch_data['expectancy'].mean():.4f} (std {batch_data['expectancy'].std():.4f})")
    print(f"  Markets with 1 trade: {(batch_data['trades'] == 1).sum()} ({(batch_data['trades'] == 1).sum() / len(batch_data) * 100:.1f}%)")
    print(f"  Markets below 50% hit rate: {(batch_data['hit_rate'] < 0.5).sum()} ({(batch_data['hit_rate'] < 0.5).sum() / len(batch_data) * 100:.1f}%)")
    print(f"  Avg net per market: ${batch_data['net_pnl'].mean():.4f}")

print("\n" + "=" * 100)

# ============================================================================
# 10. CORRELATION ANALYSIS: WHAT PREDICTS UNDERPERFORMANCE?
# ============================================================================
print("\n" + "=" * 100)
print("10. CORRELATION ANALYSIS: PREDICTORS OF NET PnL")
print("=" * 100)

correlation_pairs = [
    ("hit_rate", "net_pnl"),
    ("expectancy", "net_pnl"),
    ("trades", "net_pnl"),
    ("gross_pnl", "net_pnl"),
    ("avg_slippage_bps", "net_pnl"),
]

for col1, col2 in correlation_pairs:
    corr = df_oos_sorted[col1].corr(df_oos_sorted[col2])
    print(f"  {col1:20s} vs {col2:10s}: r = {corr:+.4f}")

print("\n")
