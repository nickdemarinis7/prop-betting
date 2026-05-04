"""
Diagnostic pass over the strikeout model using cumulative validation data.

Answers four questions:
  1. Where exactly does the model lose to the naive K/9 x IP baseline?
  2. How large is the high-K over-projection bias, by bucket?
  3. Does the HIGH-confidence tier actually pick winners?
  4. At what probability threshold does betting turn profitable?

Run:  python diagnose_model_health.py
"""

import os
import glob
import pandas as pd
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------
# Load all prediction + validation pairs
# ---------------------------------------------------------------------
pred_files = sorted(
    glob.glob(os.path.join(BASE, 'predictions_strikeouts_*.csv'))
)
val_files = sorted(
    glob.glob(os.path.join(BASE, 'validation_results_*.csv'))
)

val_map = {
    os.path.basename(f).replace('validation_results_', '').replace('.csv', ''): f
    for f in val_files
}

rows = []
for pf in pred_files:
    base = os.path.basename(pf)
    ds = (
        base.replace('predictions_strikeouts_simplified_', '')
            .replace('predictions_strikeouts_', '')
            .replace('.csv', '')
    )
    if ds not in val_map:
        continue
    try:
        pred = pd.read_csv(pf)
        val = pd.read_csv(val_map[ds])
    except Exception:
        continue
    if 'pitcher' not in pred.columns or 'pitcher' not in val.columns:
        continue
    merged = pred.merge(
        val[['pitcher', 'actual']], on='pitcher', how='inner'
    )
    merged['date'] = ds
    # Prefer the _simplified_ file over legacy one for the same date
    if 'simplified' in base or ds not in {r['date'] for r in rows}:
        # drop any previously-inserted rows for this date that came from the
        # non-simplified file
        rows = [r for r in rows if r['date'] != ds]
        for _, r in merged.iterrows():
            rows.append(r.to_dict())

df = pd.DataFrame(rows)
if df.empty:
    print("No merged prediction/validation data found.")
    raise SystemExit(0)

print("=" * 80)
print(f"📋 Loaded {len(df)} pitcher-days across {df['date'].nunique()} dates")
print("=" * 80)

# ---------------------------------------------------------------------
# Derive the "baseline" (raw K/9 x expected IP, no adjustments)
# Use season_k9 blended with recent_k9 60/40 — this is what predict
# computes as `base_projection` before any multipliers. If base_projection
# column is present, use it; otherwise reconstruct.
# ---------------------------------------------------------------------
if 'base_projection' in df.columns and df['base_projection'].notna().any():
    df['baseline'] = df['base_projection']
else:
    # fallback: reconstruct from K/9 + expected_ip
    df['baseline'] = (
        ((df.get('season_k9', 0) * 0.6) + (df.get('recent_k9', 0) * 0.4))
        / 9.0
    ) * df.get('expected_ip', 0)

df['actual'] = pd.to_numeric(df['actual'], errors='coerce')
df['projection'] = pd.to_numeric(df['projection'], errors='coerce')
df['baseline'] = pd.to_numeric(df['baseline'], errors='coerce')
df = df.dropna(subset=['actual', 'projection', 'baseline'])

df['err_model'] = df['actual'] - df['projection']
df['err_base']  = df['actual'] - df['baseline']
df['abs_err_model'] = df['err_model'].abs()
df['abs_err_base']  = df['err_base'].abs()
df['model_beats_base'] = df['abs_err_model'] < df['abs_err_base']

print("\n" + "=" * 80)
print("1️⃣  MODEL vs BASELINE")
print("=" * 80)
overall_mae_model = df['abs_err_model'].mean()
overall_mae_base  = df['abs_err_base'].mean()
bias_model = df['err_model'].mean()
bias_base  = df['err_base'].mean()
win_rate = df['model_beats_base'].mean()

print(f"   Overall MAE  - model: {overall_mae_model:.2f} K  |  baseline: {overall_mae_base:.2f} K")
print(f"   Overall bias - model: {bias_model:+.2f} K  |  baseline: {bias_base:+.2f} K")
print(f"   Games where MODEL beats BASELINE: {win_rate:.1%}  ({(df['model_beats_base']).sum()}/{len(df)})")

# Per-bucket breakdown by projection range
print(f"\n   By projection bucket (model vs baseline MAE and bias):")
print(f"   {'range':<8} {'n':>4}  {'model MAE':>9}  {'base MAE':>9}  {'Δ MAE':>6}  {'model bias':>10}  {'base bias':>10}")
for lo, hi, label in [(0, 4, '0-4'), (4, 6, '4-6'), (6, 8, '6-8'), (8, 99, '8+')]:
    mask = (df['projection'] >= lo) & (df['projection'] < hi)
    sub = df[mask]
    if sub.empty:
        continue
    print(
        f"   {label:<8} {len(sub):>4}  "
        f"{sub['abs_err_model'].mean():>9.2f}  "
        f"{sub['abs_err_base'].mean():>9.2f}  "
        f"{(sub['abs_err_model'].mean() - sub['abs_err_base'].mean()):>+6.2f}  "
        f"{sub['err_model'].mean():>+10.2f}  "
        f"{sub['err_base'].mean():>+10.2f}"
    )

# ---------------------------------------------------------------------
# 2. High-K over-projection — how much shrinkage would fix it?
# ---------------------------------------------------------------------
print("\n" + "=" * 80)
print("2️⃣  HIGH-K OVER-PROJECTION — shrinkage experiment")
print("=" * 80)
print("   Shrink projection toward league mean (4.9 K) by factor S when proj >= 6:")
print("   new_proj = proj - S * (proj - 4.9)")
print(f"   {'S':>6} {'bucket 6-8':>12} {'bucket 8+':>12} {'overall MAE':>13} {'bias':>8}")

for s in [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]:
    adj = df['projection'].copy()
    hi_mask = adj >= 6.0
    adj.loc[hi_mask] = adj.loc[hi_mask] - s * (adj.loc[hi_mask] - 4.9)
    err = df['actual'] - adj
    mae_68 = err[(df['projection'] >= 6) & (df['projection'] < 8)].abs().mean()
    mae_8 = err[df['projection'] >= 8].abs().mean()
    mae_all = err.abs().mean()
    bias_all = err.mean()
    print(
        f"   {s:>6.2f} {mae_68:>12.2f} {mae_8:>12.2f} {mae_all:>13.2f} {bias_all:>+8.2f}"
    )

# ---------------------------------------------------------------------
# 3. HIGH-confidence audit
# ---------------------------------------------------------------------
if 'confidence' in df.columns:
    print("\n" + "=" * 80)
    print("3️⃣  CONFIDENCE TIER AUDIT")
    print("=" * 80)
    for tier in ['HIGH', 'MEDIUM', 'LOW']:
        sub = df[df['confidence'] == tier]
        if sub.empty:
            continue
        within2 = (sub['abs_err_model'] <= 2).mean()
        beats = sub['model_beats_base'].mean()
        print(
            f"   {tier:<6}  n={len(sub):>3}  "
            f"MAE={sub['abs_err_model'].mean():.2f}  "
            f"bias={sub['err_model'].mean():+.2f}  "
            f"within 2K={within2:.1%}  "
            f"beats baseline={beats:.1%}"
        )

    # Is HIGH actually worse than MEDIUM? And by what feature?
    high = df[df['confidence'] == 'HIGH']
    if not high.empty:
        high_big_miss = high[high['abs_err_model'] >= 3]
        print(
            f"\n   Of {len(high)} HIGH picks, {len(high_big_miss)} "
            f"({len(high_big_miss)/len(high):.0%}) missed by 3+ K"
        )
        if not high_big_miss.empty and 'projection' in high_big_miss.columns:
            print(
                f"   Avg projection on HIGH big-misses: "
                f"{high_big_miss['projection'].mean():.1f} K  "
                f"vs HIGH overall {high['projection'].mean():.1f} K"
            )

# ---------------------------------------------------------------------
# 4. P&L threshold sweep using calibrated probabilities if available
# ---------------------------------------------------------------------
print("\n" + "=" * 80)
print("4️⃣  P&L THRESHOLD SWEEP (assumes -110 juice when no odds in file)")
print("=" * 80)
# Find prob columns
prob_cols = sorted(
    [c for c in df.columns if c.startswith('prob_') and c.endswith('+')]
)
# derive threshold from column name: prob_5.5+ -> 5.5
def line_from_col(c):
    return float(c.replace('prob_', '').replace('+', ''))

print(f"   {'thresh':>7} {'bets':>5} {'hit%':>6} {'profit':>9} {'ROI':>8}")
for min_prob in [0.55, 0.60, 0.65, 0.70, 0.75]:
    total_bets = 0
    total_hits = 0
    net = 0.0
    for col in prob_cols:
        line = line_from_col(col)
        mask = (df[col] >= min_prob)
        sub = df[mask]
        if sub.empty:
            continue
        total_bets += len(sub)
        hits = (sub['actual'] >= np.ceil(line))  # >= next integer
        total_hits += int(hits.sum())
        # -110 juice: win +$0.909, loss -$1.00
        net += hits.sum() * 0.909 - (~hits).sum() * 1.0
    if total_bets == 0:
        continue
    roi = net / total_bets
    print(
        f"   {min_prob:>7.2f} {total_bets:>5d} "
        f"{total_hits/total_bets:>6.1%} "
        f"{net:>+9.2f}u {roi:>+8.1%}"
    )

# Also evaluate per-line to see which ladder tiers leak
print(f"\n   Per-line performance at prob >= 0.60:")
print(f"   {'line':>5} {'bets':>5} {'hit%':>6} {'net':>8} {'ROI':>7}")
for col in prob_cols:
    line = line_from_col(col)
    mask = (df[col] >= 0.60)
    sub = df[mask]
    if sub.empty:
        continue
    hits = (sub['actual'] >= np.ceil(line))
    net = hits.sum() * 0.909 - (~hits).sum() * 1.0
    print(
        f"   {line:>5.1f} {len(sub):>5d} "
        f"{hits.mean():>6.1%} "
        f"{net:>+8.2f}u {(net/len(sub)):>+7.1%}"
    )

print("\n" + "=" * 80)
print("Done.")
print("=" * 80)
