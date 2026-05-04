"""
Home run model calibration backtest.

Uses validation_results_*.csv to assess:
  1. Raw vs calibrated probability accuracy.
  2. Calibration by bucket (raw vs calibrated).
  3. Brier score improvement.
  4. YES-pick P&L impact.
"""
import pandas as pd
import numpy as np
import glob
from pathlib import Path

HERE = Path(__file__).parent

def load_all_validation():
    files = sorted(glob.glob(str(HERE / 'validation_results_*.csv')))
    dfs = []
    for f in files:
        d = pd.read_csv(f)
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def main():
    df = load_all_validation()
    if df.empty:
        print("No validation data found.")
        return

    df = df.dropna(subset=['projection']).copy()
    df['hit'] = (df['actual_hr'] > 0).astype(int)
    n = len(df)

    print("=" * 70)
    print(f"HR CALIBRATION BACKTEST — {n} batter-days")
    print("=" * 70)

    # Load the calibrator
    try:
        from prob_calibrator import ProbabilityCalibrator
        cal = ProbabilityCalibrator()
        cal.load()
        if cal.is_fitted:
            df['cal_prob'] = df['projection'].apply(cal.calibrate)
        else:
            df['cal_prob'] = df['projection']
            print("  ⚠️ Calibrator not fitted, using raw probs")
    except Exception as e:
        df['cal_prob'] = df['projection']
        print(f"  ⚠️ Could not load calibrator: {e}")

    raw = df['projection'].values
    cal = df['cal_prob'].values
    y = df['hit'].values

    # 1. Overall
    raw_brier = np.mean((raw - y) ** 2)
    cal_brier = np.mean((cal - y) ** 2)
    baseline_brier = np.mean((y.mean() - y) ** 2)
    raw_bias = raw.mean() - y.mean()
    cal_bias = cal.mean() - y.mean()

    print(f"\n[1] OVERALL")
    print(f"   Actual HR rate: {y.mean():.1%}")
    print(f"   Raw avg prob:   {raw.mean():.1%}  bias={raw_bias:+.2%}")
    print(f"   Cal avg prob:   {cal.mean():.1%}  bias={cal_bias:+.2%}")
    print(f"   Brier (raw):    {raw_brier:.4f}")
    print(f"   Brier (cal):    {cal_brier:.4f}  {'✅ improved' if cal_brier < raw_brier else '⚠️ worse'}")
    print(f"   Brier (naive):  {baseline_brier:.4f}")

    # 2. Calibration buckets
    print(f"\n[2] CALIBRATION BY BUCKET (raw → calibrated)")
    bins = [0, 0.05, 0.08, 0.12, 0.16, 0.20, 1.0]
    df['bin'] = pd.cut(df['projection'], bins=bins, include_lowest=True)
    print(f"   {'bucket':<14} {'n':>4} {'raw_avg':>8} {'cal_avg':>8} {'actual':>7} {'raw_gap':>8} {'cal_gap':>8}")
    for interval, g in df.groupby('bin', observed=True):
        if len(g) == 0:
            continue
        raw_avg = g['projection'].mean()
        cal_avg = g['cal_prob'].mean()
        act = g['hit'].mean()
        print(f"   {str(interval):<14} {len(g):>4} {raw_avg:>8.1%} {cal_avg:>8.1%} {act:>7.1%} "
              f"{raw_avg-act:>+8.1%} {cal_avg-act:>+8.1%}")

    # 3. YES-pick impact
    yes = df[df.get('recommended_side', pd.Series(dtype=str)) == 'YES'].copy()
    if len(yes) and 'book_odds' in yes.columns:
        print(f"\n[3] YES-PICK CALIBRATION")
        print(f"   n={len(yes)}")
        print(f"   Raw avg proj:  {yes['projection'].mean():.1%}")
        print(f"   Cal avg proj:  {yes['cal_prob'].mean():.1%}")
        print(f"   Actual HR rate: {yes['hit'].mean():.1%}")

        # P&L comparison
        for label, prob_col in [('raw', 'projection'), ('calibrated', 'cal_prob')]:
            pnl = 0.0
            bets = 0
            for _, r in yes.iterrows():
                # Re-check if this pick still qualifies under calibrated prob
                if label == 'calibrated':
                    our_p = r['cal_prob']
                    if pd.notna(r.get('book_prob')):
                        if our_p - r['book_prob'] < 0.04:
                            continue  # no longer qualifies
                bets += 1
                if r['hit']:
                    pnl += r['book_odds'] / 100.0
                else:
                    pnl -= 1.0
            if bets > 0:
                print(f"   P&L ({label:>10}): {bets} bets → {pnl:+.2f}u  ROI={pnl/bets:+.1%}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
