#!/usr/bin/env python3
"""
Diagnose OOS batch performance variance.

Analyzes which batches performed worst and why by examining:
- Batch-level PnL aggregation
- Per-market metrics within worst batches
- Win rate, trade size, entry conditions
- Volatility and market conditions
"""
import json
from pathlib import Path

import pandas as pd

WORKSPACE_ROOT = Path(__file__).parent.parent
ARTIFACTS = WORKSPACE_ROOT / "reports" / "artifacts"

# Load market metrics from degradation diagnostics
metrics_csv = ARTIFACTS / "relative_book_degradation_market_metrics.csv"
diagnosis_json = ARTIFACTS / "relative_book_degradation_diagnosis.json"

print("=" * 80)
print("BATCH PERFORMANCE VARIANCE DIAGNOSIS")
print("=" * 80)

# Load market metrics
df_metrics = pd.read_csv(metrics_csv)
print(f"\nLoaded {len(df_metrics)} market rows from degradation_market_metrics.csv")

with open(diagnosis_json) as f:
    diagnosis = json.load(f)

# Reconstruct batch structure from logs
# Based on the actual batch logs, we have 11 batches with these market counts:
batch_market_counts = [72, 73, 86, 86, 81, 86, 80, 80, 84, 89, 57]
cumulative = 0
batch_ranges = []
for i, count in enumerate(batch_market_counts, 1):
    start = cumulative
    end = cumulative + count
    batch_ranges.append((i, start, end, count))
    cumulative += end - start

# Filter to OOS only
df_oos = df_metrics[df_metrics["split"] == "oos_remaining"].copy()
print(f"Filtered to {len(df_oos)} OOS markets")

# Sort by market_id to assign to batches
df_oos_sorted = df_oos.sort_values("market_id").reset_index(drop=True)

# Assign batch numbers
batch_assignments = []
for batch_num, start, end, count in batch_ranges:
    market_idxs = list(range(start, min(end, len(df_oos_sorted))))
    batch_assignments.extend([(batch_num, idx) for idx in market_idxs])

batch_map = {idx: batch_num for batch_num, idx in batch_assignments}
df_oos_sorted["batch"] = df_oos_sorted.index.map(batch_map)

print(f"\nAssigned {len(df_oos_sorted)} OOS markets to {len(batch_ranges)} batches")

# Compute batch-level aggregations
batch_stats = []
for batch_num, _, _, _ in batch_ranges:
    batch_data = df_oos_sorted[df_oos_sorted["batch"] == batch_num]
    if len(batch_data) == 0:
        continue

    stats = {
        "batch": batch_num,
        "market_count": len(batch_data),
        "total_trades": batch_data["trades"].sum(),
        "net_pnl": batch_data["net_pnl"].sum(),
        "gross_pnl": batch_data["gross_pnl"].sum(),
        "avg_hit_rate": batch_data["hit_rate"].mean(),
        "std_hit_rate": batch_data["hit_rate"].std(),
        "median_trades_per_market": batch_data["trades"].median(),
        "avg_slippage_bps": batch_data["avg_slippage_bps"].mean(),
        "avg_expectancy": batch_data["expectancy"].mean(),
    }
    batch_stats.append(stats)

df_batch = pd.DataFrame(batch_stats).sort_values("net_pnl")

print("\n" + "=" * 80)
print("BATCH-LEVEL PERFORMANCE RANKING (worst to best)")
print("=" * 80)
for idx, row in df_batch.iterrows():
    status = "🔴 LOSS" if row["net_pnl"] < 0 else "🟢 GAIN"
    print(f"\n{status} Batch {int(row['batch']):2d} | {int(row['market_count']):2d} mkts | {int(row['total_trades']):3d} trades")
    print(f"  Net PnL:        ${row['net_pnl']:8.2f}")
    print(f"  Gross PnL:      ${row['gross_pnl']:8.2f}")
    print(f"  Avg hit rate:   {row['avg_hit_rate']*100:5.2f}% (± {row['std_hit_rate']*100:4.2f}%)")
    print(f"  Avg slippage:   {row['avg_slippage_bps']:6.2f} bps")
    print(f"  Avg expectancy: {row['avg_expectancy']:6.4f}")

# Identify worst vs best
worst_batch = df_batch.iloc[0]
best_batch = df_batch.iloc[-1]

print("\n" + "=" * 80)
print("WORST BATCH DETAIL (Batch {})".format(int(worst_batch["batch"])))
print("=" * 80)
batch_worst_data = df_oos_sorted[df_oos_sorted["batch"] == worst_batch["batch"]].sort_values("net_pnl")
print("\nTop 5 losers in worst batch:")
for idx, row in batch_worst_data.head(5).iterrows():
    print(f"  {row['market_id'][:16]:16s} | trades={int(row['trades']):2d} | hit={row['hit_rate']*100:5.1f}% | "
          f"net=${row['net_pnl']:8.2f} | gross=${row['gross_pnl']:8.2f} | "
          f"slippage={row['avg_slippage_bps']:6.2f}bps | exp={row['expectancy']:6.4f}")

print("\nTop 5 gainers in worst batch:")
for idx, row in batch_worst_data.tail(5).iterrows():
    print(f"  {row['market_id'][:16]:16s} | trades={int(row['trades']):2d} | hit={row['hit_rate']*100:5.1f}% | "
          f"net=${row['net_pnl']:8.2f} | gross=${row['gross_pnl']:8.2f} | "
          f"slippage={row['avg_slippage_bps']:6.2f}bps | exp={row['expectancy']:6.4f}")

print("\n" + "=" * 80)
print("BEST BATCH DETAIL (Batch {})".format(int(best_batch["batch"])))
print("=" * 80)
batch_best_data = df_oos_sorted[df_oos_sorted["batch"] == best_batch["batch"]].sort_values("net_pnl")
print("\nTop 5 losers in best batch:")
for idx, row in batch_best_data.head(5).iterrows():
    print(f"  {row['market_id'][:16]:16s} | trades={int(row['trades']):2d} | hit={row['hit_rate']*100:5.1f}% | "
          f"net=${row['net_pnl']:8.2f} | gross=${row['gross_pnl']:8.2f} | "
          f"slippage={row['avg_slippage_bps']:6.2f}bps | exp={row['expectancy']:6.4f}")

print("\nTop 5 gainers in best batch:")
for idx, row in batch_best_data.tail(5).iterrows():
    print(f"  {row['market_id'][:16]:16s} | trades={int(row['trades']):2d} | hit={row['hit_rate']*100:5.1f}% | "
          f"net=${row['net_pnl']:8.2f} | gross=${row['gross_pnl']:8.2f} | "
          f"slippage={row['avg_slippage_bps']:6.2f}bps | exp={row['expectancy']:6.4f}")

# Comparative analysis
print("\n" + "=" * 80)
print("WORST vs BEST BATCH COMPARISON")
print("=" * 80)
comparison = pd.DataFrame({
    "Worst Batch": [worst_batch["batch"], worst_batch["market_count"], worst_batch["total_trades"],
                    worst_batch["net_pnl"], worst_batch["avg_hit_rate"]*100,
                    worst_batch["avg_slippage_bps"], worst_batch["avg_expectancy"]],
    "Best Batch": [best_batch["batch"], best_batch["market_count"], best_batch["total_trades"],
                   best_batch["net_pnl"], best_batch["avg_hit_rate"]*100,
                   best_batch["avg_slippage_bps"], best_batch["avg_expectancy"]],
}, index=["Batch #", "Markets", "Trades", "Net PnL ($)", "Avg Hit Rate (%)",
          "Slippage (bps)", "Avg Expectancy"])
print(comparison.to_string())

print("\n" + "=" * 80)
print("KEY INSIGHTS")
print("=" * 80)

# Compute deltas
pnl_delta = best_batch["net_pnl"] - worst_batch["net_pnl"]
hr_delta = (best_batch["avg_hit_rate"] - worst_batch["avg_hit_rate"]) * 100
slippage_delta = best_batch["avg_slippage_bps"] - worst_batch["avg_slippage_bps"]
exp_delta = best_batch["avg_expectancy"] - worst_batch["avg_expectancy"]

print(f"\nPnL delta (best - worst): ${pnl_delta:.2f}")
print(f"Hit rate delta:  {hr_delta:+.2f} pp")
print(f"Slippage delta: {slippage_delta:+.2f} bps")
print(f"Expectancy delta: {exp_delta:+.6f}")

# Identify which factors correlate with batch performance
print("\n" + "=" * 80)
print("FACTOR CORRELATION WITH BATCH NET PnL")
print("=" * 80)

correlations = {}
for col in ["avg_hit_rate", "avg_slippage_bps", "avg_expectancy"]:
    if col in df_batch.columns:
        corr = df_batch["net_pnl"].corr(df_batch[col])
        correlations[col] = corr

for factor, corr in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True):
    direction = "+" if corr > 0 else "-"
    print(f"  {factor:30s}: {direction} {abs(corr):.3f}")

print("\n" + "=" * 80)
print("NEGATIVE BATCH ANALYSIS")
print("=" * 80)

negative_batches = df_batch[df_batch["net_pnl"] < 0]
if len(negative_batches) > 0:
    print(f"\nFound {len(negative_batches)} batches with negative PnL:")
    for _, row in negative_batches.iterrows():
        loss_magnitude = abs(row["net_pnl"])
        trades = row["total_trades"]
        loss_per_trade = loss_magnitude / trades if trades > 0 else 0
        print(f"  Batch {int(row['batch']):2d}: ${row['net_pnl']:8.2f} ({loss_per_trade:6.2f}$/trade), "
              f"hit={row['avg_hit_rate']*100:5.2f}%, slippage={row['avg_slippage_bps']:6.2f}bps")

    avg_slippage_negative = negative_batches["avg_slippage_bps"].mean()
    avg_slippage_positive = df_batch[df_batch["net_pnl"] > 0]["avg_slippage_bps"].mean()
    avg_hr_negative = negative_batches["avg_hit_rate"].mean()
    avg_hr_positive = df_batch[df_batch["net_pnl"] > 0]["avg_hit_rate"].mean()

    print(f"\nNegative batch avg slippage: {avg_slippage_negative:.2f} bps")
    print(f"Positive batch avg slippage: {avg_slippage_positive:.2f} bps")
    print(f"Slippage deterioration: {avg_slippage_negative - avg_slippage_positive:+.2f} bps")

    print(f"\nNegative batch avg hit rate: {avg_hr_negative*100:.2f}%")
    print(f"Positive batch avg hit rate: {avg_hr_positive*100:.2f}%")
    print(f"Hit rate drop: {(avg_hr_negative - avg_hr_positive)*100:+.2f} pp")
else:
    print("\nNo batches with negative net PnL found.")

print("\n")

