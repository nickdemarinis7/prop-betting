"""
Strikeout model calibration backtest.

Uses validation_cumulative.csv (all historical predictions + actuals) to:
  1. Assess point-projection accuracy (MAE, bias, bucket breakdowns).
  2. Assess probability calibration at each ladder rung (reliability diagram).
  3. Compute Brier score and log-loss per rung.
  4. Simulate betting P&L at various thresholds.
  5. Diagnose high-K shrinkage effectiveness.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

HERE = Path(__file__).parent
CUM = HERE / "validation_cumulative.csv"

RUNGS = [3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
ODDS_AMERICAN = -115   # assumed standard book price for over
DECIMAL = 1 + 100/115  # ≈ 1.870

def american_to_decimal(odds):
    return 1 + (100/abs(odds) if odds < 0 else odds/100)

def implied_prob(odds):
    d = american_to_decimal(odds)
    return 1/d

def main():
    df = pd.read_csv(CUM)
    df['date'] = pd.to_datetime(df['date'].astype(str), format='%Y%m%d', errors='coerce')
    df = df.dropna(subset=['projection', 'actual']).copy()
    n = len(df)

    print("="*78)
    print(f"STRIKEOUT MODEL CALIBRATION BACKTEST  —  {n} predictions")
    if df['date'].notna().any():
        print(f"Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    print("="*78)

    # ---- 1. POINT PROJECTION ACCURACY ----
    df['error'] = df['projection'] - df['actual']
    df['abs_error'] = df['error'].abs()
    mae = df['abs_error'].mean()
    bias = df['error'].mean()
    rmse = np.sqrt((df['error']**2).mean())
    within_1 = (df['abs_error'] <= 1).mean()
    within_2 = (df['abs_error'] <= 2).mean()

    print("\n[1] POINT PROJECTION ACCURACY")
    print(f"   MAE            {mae:.2f} K")
    print(f"   Bias (proj-act){bias:+.2f} K  {'(OVER-proj)' if bias>0 else '(UNDER-proj)'}")
    print(f"   RMSE           {rmse:.2f}")
    print(f"   within ±1 K    {within_1:.1%}")
    print(f"   within ±2 K    {within_2:.1%}")

    # ---- 2. BUCKETED ACCURACY (low/mid/high projection) ----
    print("\n[2] ACCURACY BY PROJECTION BUCKET")
    buckets = [
        ("low    <5.0",   df[df['projection'] < 5.0]),
        ("mid  5.0-6.5",  df[(df['projection'] >= 5.0) & (df['projection'] < 6.5)]),
        ("high 6.5-7.5",  df[(df['projection'] >= 6.5) & (df['projection'] < 7.5)]),
        ("ace   ≥7.5",    df[df['projection'] >= 7.5]),
    ]
    print(f"   {'bucket':<14} {'n':>4} {'MAE':>6} {'bias':>7}")
    for name, sub in buckets:
        if len(sub):
            print(f"   {name:<14} {len(sub):>4} {sub['abs_error'].mean():>6.2f} {sub['error'].mean():>+7.2f}")

    # ---- 3. PROBABILITY CALIBRATION PER RUNG ----
    print("\n[3] PROBABILITY CALIBRATION (per ladder rung, over X+0.5)")
    print(f"   {'rung':>4} {'n':>4} {'avg_pred':>9} {'actual':>7} {'diff':>7} {'brier':>6} {'logloss':>7}")
    rung_stats = []
    for rung in RUNGS:
        prob_col = f'prob_{rung}'
        hit_col  = f'hit_{rung}'
        if prob_col not in df.columns: 
            continue
        sub = df.dropna(subset=[prob_col, hit_col])
        if len(sub) < 5: 
            continue
        p = sub[prob_col].clip(1e-4, 1-1e-4).values
        y = sub[hit_col].astype(int).values
        avg_pred = p.mean()
        actual   = y.mean()
        brier    = np.mean((p - y)**2)
        logloss  = -np.mean(y*np.log(p) + (1-y)*np.log(1-p))
        rung_stats.append((rung, len(sub), avg_pred, actual, brier, logloss))
        print(f"   {rung:>4} {len(sub):>4} {avg_pred:>9.3f} {actual:>7.3f} {avg_pred-actual:>+7.3f} {brier:>6.3f} {logloss:>7.3f}")

    # ---- 4. RELIABILITY DIAGRAM (most-liquid rungs) ----
    print("\n[4] RELIABILITY (predicted-prob decile → actual hit rate)")
    for rung in [5.5, 6.5]:
        prob_col = f'prob_{rung}'
        hit_col  = f'hit_{rung}'
        if prob_col not in df.columns:
            continue
        sub = df.dropna(subset=[prob_col, hit_col]).copy()
        if len(sub) < 30:
            continue
        print(f"\n   Rung {rung} (n={len(sub)})")
        print(f"   {'pred range':<14} {'n':>4} {'avg_pred':>9} {'actual':>7} {'diff':>7}")
        bins = [0, 0.3, 0.45, 0.55, 0.65, 0.75, 0.85, 1.0]
        sub['bin'] = pd.cut(sub[prob_col], bins=bins, include_lowest=True)
        for interval, g in sub.groupby('bin', observed=True):
            if len(g) == 0: 
                continue
            print(f"   {str(interval):<14} {len(g):>4} {g[prob_col].mean():>9.3f} {g[hit_col].mean():>7.3f} {g[prob_col].mean()-g[hit_col].mean():>+7.3f}")

    # ---- 5. P&L SIMULATION AT THRESHOLDS ----
    print("\n[5] P&L SIMULATION  (flat 1u bets, -115 assumed, over X+0.5)")
    print(f"   {'thresh':>6} {'rung':>4} {'bets':>4} {'wins':>4} {'hit%':>6} {'ROI':>7} {'P&L(u)':>8}")
    best = None
    for thresh in [0.55, 0.60, 0.62, 0.65, 0.68, 0.70, 0.75]:
        for rung in RUNGS:
            prob_col = f'prob_{rung}'
            hit_col  = f'hit_{rung}'
            if prob_col not in df.columns:
                continue
            sub = df.dropna(subset=[prob_col, hit_col])
            picks = sub[sub[prob_col] >= thresh]
            if len(picks) < 5:
                continue
            wins = int(picks[hit_col].sum())
            losses = len(picks) - wins
            pnl = wins * (DECIMAL - 1) - losses * 1.0
            roi = pnl / len(picks)
            if best is None or pnl > best[-1]:
                best = (thresh, rung, len(picks), wins, pnl)
            print(f"   {thresh:>6.2f} {rung:>4} {len(picks):>4} {wins:>4} {wins/len(picks):>6.1%} {roi:>+7.1%} {pnl:>+8.2f}")
    if best:
        print(f"\n   ★ Best: thresh={best[0]} rung={best[1]} n={best[2]} wins={best[3]} P&L={best[4]:+.2f}u")

    # ---- 6. CURRENT DEPLOYED THRESHOLD CHECK ----
    print("\n[6] DEPLOYED BETTING FILTER  (prob ≥ 0.65, any rung)")
    flagged = []
    for rung in RUNGS:
        prob_col = f'prob_{rung}'
        hit_col  = f'hit_{rung}'
        if prob_col not in df.columns:
            continue
        sub = df.dropna(subset=[prob_col, hit_col])
        picks = sub[sub[prob_col] >= 0.65]
        for _, r in picks.iterrows():
            flagged.append({'rung': rung, 'prob': r[prob_col], 'hit': r[hit_col]})
    if flagged:
        fd = pd.DataFrame(flagged)
        wins = int(fd['hit'].sum())
        pnl = wins * (DECIMAL - 1) - (len(fd) - wins) * 1.0
        print(f"   Total qualifying bets: {len(fd)}")
        print(f"   Wins:                  {wins}   Hit%: {wins/len(fd):.1%}")
        print(f"   P&L:                   {pnl:+.2f}u   ROI: {pnl/len(fd):+.1%}")
        print(f"   Avg predicted prob:    {fd['prob'].mean():.3f}")
        print(f"   Calibration gap:       {fd['prob'].mean() - wins/len(fd):+.3f}")
    else:
        print("   (no qualifying bets)")

    # ---- 7. HIGH-K SHRINKAGE AUDIT ----
    print("\n[7] HIGH-K SHRINKAGE AUDIT  (projection ≥ 7.0)")
    high = df[df['projection'] >= 7.0]
    if len(high):
        print(f"   n={len(high)}  proj_mean={high['projection'].mean():.2f}  actual_mean={high['actual'].mean():.2f}")
        print(f"   bias={high['error'].mean():+.2f}  MAE={high['abs_error'].mean():.2f}")
        if high['error'].mean() > 0.5:
            print(f"   ⚠️  Still over-projecting aces by {high['error'].mean():.2f} K — consider stronger shrinkage")
        else:
            print(f"   ✓ Shrinkage appears healthy")

    print("\n" + "="*78)


if __name__ == "__main__":
    main()
