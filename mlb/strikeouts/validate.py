"""
⚾ MLB Strikeout Prediction Validation
Validates predictions against actual game results

Usage:
    python validate.py                   # Validate yesterday
    python validate.py --date 20260418   # Validate specific date
    python validate.py --cumulative      # Show cumulative stats across all dates
"""

import sys
import os
import glob
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests
import argparse

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ======================================================================
# Fetch actuals from MLB API
# ======================================================================
def fetch_actual_strikeouts(date_str, predictions_df, quiet=False):
    """Fetch actual K counts from MLB boxscores for a given date."""
    validate_date = datetime.strptime(date_str, '%Y%m%d')

    # Fetch schedule once
    schedule_url = "https://statsapi.mlb.com/api/v1/schedule"
    params = {'sportId': 1, 'date': validate_date.strftime('%Y-%m-%d')}
    resp = requests.get(schedule_url, params=params, timeout=10)
    if resp.status_code != 200 or not resp.json().get('dates'):
        if not quiet:
            print(f"   ⚠️  No games found on {date_str}")
        return []

    games = resp.json()['dates'][0].get('games', [])

    # Build team-abbrev cache once
    team_cache = {}
    for game in games:
        for side in ('away', 'home'):
            tid = game['teams'][side]['team']['id']
            if tid not in team_cache:
                tr = requests.get(f"https://statsapi.mlb.com/api/v1/teams/{tid}", timeout=5)
                if tr.status_code == 200:
                    team_cache[tid] = tr.json()['teams'][0]['abbreviation']

    results = []

    for _, row in predictions_df.iterrows():
        pitcher_name = row['pitcher']
        team = row['team']
        projection = row['projection']

        actual_k = None
        for game in games:
            away_abbr = team_cache.get(game['teams']['away']['team']['id'])
            home_abbr = team_cache.get(game['teams']['home']['team']['id'])

            if team not in (away_abbr, home_abbr):
                continue

            game_pk = game['gamePk']
            box_resp = requests.get(
                f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore", timeout=10
            )
            if box_resp.status_code != 200:
                continue

            box = box_resp.json()
            for side in ('away', 'home'):
                pitcher_ids = box['teams'][side].get('pitchers', [])
                players_data = box['teams'][side].get('players', {})
                for pid in pitcher_ids:
                    pkey = f'ID{pid}'
                    if pkey not in players_data:
                        continue
                    full_name = players_data[pkey]['person']['fullName']
                    if pitcher_name.split()[-1].lower() in full_name.lower():
                        stats = players_data[pkey].get('stats', {}).get('pitching', {})
                        actual_k = stats.get('strikeOuts', 0)
                        break
                if actual_k is not None:
                    break
            if actual_k is not None:
                break

        if actual_k is None:
            if not quiet:
                print(f"   ⚠️  {pitcher_name} — not found / did not start")
            continue

        diff = round(actual_k - projection, 1)

        result = {
            'date': date_str,
            'pitcher': pitcher_name,
            'team': team,
            'opponent': row.get('opponent', ''),
            'projection': projection,
            'actual': int(actual_k),
            'error': diff,
            'abs_error': abs(diff),
            'season_k9': row.get('season_k9', 0),
            'recent_k9': row.get('recent_k9', 0),
            'expected_ip': row.get('expected_ip', 0),
            'opponent_k_rate': row.get('opponent_k_rate', 0),
            'std_dev': row.get('std_dev', 0),
            'confidence': row.get('confidence', ''),
            'red_flags': row.get('red_flags', 'None'),
            'ml_correction': row.get('ml_correction', 0),
        }

        # Ladder hit flags + probs for P&L
        for line in [3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]:
            prob_col = f'prob_{line}+'
            result[f'hit_{line}'] = 1 if actual_k >= line else 0
            result[f'prob_{line}'] = row.get(prob_col, 0)

        results.append(result)

        if not quiet:
            icon = "✅" if abs(diff) <= 2 else "⚠️" if abs(diff) <= 3 else "❌"
            print(f"   {icon} {pitcher_name:25s} Proj:{projection:5.1f} Act:{actual_k:2d} Err:{diff:+5.1f}")

    return results


# ======================================================================
# Single-date validation
# ======================================================================
def validate_date(date_str, quiet=False):
    """Validate predictions for one date. Returns DataFrame or None.

    Accepts either predictions_strikeouts_YYYYMMDD.csv or
    predictions_strikeouts_simplified_YYYYMMDD.csv (preferring the
    simplified version when both exist, since that's our current output).
    """
    simplified_file = f"predictions_strikeouts_simplified_{date_str}.csv"
    original_file = f"predictions_strikeouts_{date_str}.csv"
    if os.path.exists(simplified_file):
        predictions_file = simplified_file
    elif os.path.exists(original_file):
        predictions_file = original_file
    else:
        if not quiet:
            print(f"❌ File not found: {original_file} or {simplified_file}")
        return None

    predictions_df = pd.read_csv(predictions_file)
    if not quiet:
        print(f"\n📅 Validating: {date_str} ({len(predictions_df)} pitchers)")

    results = fetch_actual_strikeouts(date_str, predictions_df, quiet=quiet)
    if not results:
        return None

    return pd.DataFrame(results)


# ======================================================================
# Display helpers
# ======================================================================
def print_summary(df, label=""):
    """Print comprehensive accuracy summary."""
    print("=" * 80)
    print(f"📊 VALIDATION RESULTS{' — ' + label if label else ''}")
    print("=" * 80)

    n = len(df)
    mae = df['abs_error'].mean()
    rmse = np.sqrt((df['error'] ** 2).mean())
    bias = df['error'].mean()
    median_err = df['error'].median()

    print(f"\n🎯 Overall ({n} pitchers):")
    print(f"   MAE:    {mae:.2f} K")
    print(f"   RMSE:   {rmse:.2f} K")
    print(f"   Bias:   {bias:+.2f} K ({'over-projecting' if bias < 0 else 'under-projecting'})")
    print(f"   Median: {median_err:+.2f} K")

    # Accuracy buckets
    w1 = (df['abs_error'] <= 1).mean()
    w2 = (df['abs_error'] <= 2).mean()
    w3 = (df['abs_error'] <= 3).mean()
    print(f"\n📏 Accuracy:")
    print(f"   Within 1 K: {w1:.0%} | Within 2 K: {w2:.0%} | Within 3 K: {w3:.0%}")

    # Baseline comparison: what if we just used (season_k9/9 * expected_ip)?
    if 'season_k9' in df.columns and 'expected_ip' in df.columns:
        valid = df[(df['season_k9'] > 0) & (df['expected_ip'] > 0)]
        if len(valid) > 0:
            baseline_proj = (valid['season_k9'] / 9) * valid['expected_ip']
            baseline_mae = (valid['actual'] - baseline_proj).abs().mean()
            model_mae_subset = valid['abs_error'].mean()
            improvement = baseline_mae - model_mae_subset
            pct = (improvement / baseline_mae * 100) if baseline_mae > 0 else 0
            icon = "✅" if improvement > 0 else "⚠️"
            print(f"\n📐 Baseline Comparison (raw K/9 × IP):")
            print(f"   Baseline MAE: {baseline_mae:.2f} K")
            print(f"   Model MAE:    {model_mae_subset:.2f} K")
            print(f"   {icon} Model is {abs(improvement):.2f} K {'better' if improvement > 0 else 'worse'} ({abs(pct):.0f}%)")

    # By projection bucket
    print(f"\n📊 By Projection Range:")
    for lo, hi, label in [(0, 4, '0-4 K'), (4, 6, '4-6 K'), (6, 8, '6-8 K'), (8, 12, '8+ K')]:
        subset = df[(df['projection'] >= lo) & (df['projection'] < hi)]
        if len(subset) > 0:
            print(f"   {label:6s}: n={len(subset):3d}  MAE={subset['abs_error'].mean():.2f}  Bias={subset['error'].mean():+.2f}")

    # By confidence (if available)
    if 'confidence' in df.columns and df['confidence'].notna().any():
        print(f"\n🎯 By Confidence:")
        for conf in ['HIGH', 'MEDIUM', 'LOW']:
            subset = df[df['confidence'] == conf]
            if len(subset) > 0:
                print(f"   {conf:6s}: n={len(subset):3d}  MAE={subset['abs_error'].mean():.2f}  Bias={subset['error'].mean():+.2f}")

    # Ladder performance
    print(f"\n🎰 Ladder Hit Rates:")
    for line in [4.5, 5.5, 6.5, 7.5, 8.5]:
        hit_col = f'hit_{line}'
        prob_col = f'prob_{line}'
        if hit_col in df.columns and prob_col in df.columns:
            valid = df[df[prob_col] > 0.01]
            if len(valid) > 0:
                actual_rate = valid[hit_col].mean() * 100
                expected_rate = valid[prob_col].mean() * 100
                gap = actual_rate - expected_rate
                icon = "✅" if abs(gap) < 10 else ("📈" if gap > 0 else "📉")
                print(f"   {line}+ K: {actual_rate:5.1f}% actual vs {expected_rate:5.1f}% expected {icon} {gap:+.1f}pp")

    # P&L simulation (simple: bet 1u on every line where model prob > implied prob of -110)
    total_wagered = 0.0
    total_returned = 0.0
    for line in [4.5, 5.5, 6.5, 7.5]:
        hit_col = f'hit_{line}'
        prob_col = f'prob_{line}'
        if hit_col not in df.columns or prob_col not in df.columns:
            continue
        # Only bet when our prob > 55% (implied -122)
        bettable = df[df[prob_col] > 0.55]
        for _, row in bettable.iterrows():
            total_wagered += 1.0
            if row[hit_col]:
                total_returned += 1.0 + (100 / 110)  # Standard -110 payout

    if total_wagered > 0:
        net = total_returned - total_wagered
        roi = (net / total_wagered) * 100
        icon = "✅" if net >= 0 else "❌"
        print(f"\n💰 P&L Simulation (1u on lines where model prob >55%, -110 juice):")
        print(f"   Bets: {int(total_wagered)} | Returned: {total_returned:.1f}u | {icon} Net: {net:+.1f}u (ROI: {roi:+.1f}%)")

    print()


def print_individual(df):
    """Print individual pitcher results."""
    print("=" * 80)
    print("🎯 INDIVIDUAL RESULTS")
    print("=" * 80)

    sorted_df = df.sort_values('abs_error', ascending=False)

    for _, row in sorted_df.iterrows():
        err = row['error']
        icon = "✅" if row['abs_error'] <= 2 else "⚠️" if row['abs_error'] <= 3 else "❌"
        conf = row.get('confidence', '')
        conf_icon = '🟢' if conf == 'HIGH' else '🟡' if conf == 'MEDIUM' else '🔴' if conf == 'LOW' else ''
        flags = row.get('red_flags', 'None')
        flag_str = f" ⚠️ {flags}" if flags != 'None' and pd.notna(flags) else ''

        print(f"{icon} {row['pitcher']:25s} | Proj:{row['projection']:5.1f} | Act:{row['actual']:2d} | Err:{err:+5.1f} {conf_icon}{flag_str}")

        # Ladder hits
        ladder = []
        for line in [4.5, 5.5, 6.5, 7.5, 8.5]:
            hit_col = f'hit_{line}'
            prob_col = f'prob_{line}'
            if hit_col in row and prob_col in row and row[prob_col] > 0.10:
                h = "✅" if row[hit_col] else "❌"
                ladder.append(f"{line}+ {h} ({row[prob_col]:.0%})")
        if ladder:
            print(f"   Ladder: {' | '.join(ladder)}")
        print()


# ======================================================================
# Main
# ======================================================================
parser = argparse.ArgumentParser(description="Validate MLB strikeout predictions")
parser.add_argument('--date', type=str, help='Date to validate (YYYYMMDD)', default=None)
parser.add_argument('--cumulative', action='store_true', help='Cumulative stats across all dates')
args = parser.parse_args()

print("=" * 80)
print("⚾ MLB STRIKEOUT PREDICTION VALIDATION")
print("=" * 80)

if args.cumulative:
    # ------------------------------------------------------------------
    # Cumulative mode
    # ------------------------------------------------------------------
    # Match both simplified and original filenames, de-duped by date so
    # each day validates exactly once (preferring simplified).
    all_files = glob.glob('predictions_strikeouts_*.csv')
    # Exclude _rerun files — they may contain live/in-game odds
    all_files = [f for f in all_files if '_rerun' not in f]
    by_date = {}
    for pf in all_files:
        base = os.path.basename(pf)
        ds = base.replace('predictions_strikeouts_simplified_', '').replace(
            'predictions_strikeouts_', '').replace('.csv', '')
        if ds not in by_date or 'simplified' in base:
            by_date[ds] = pf
    pred_files = [by_date[d] for d in sorted(by_date)]
    if not pred_files:
        print("\n❌ No prediction files found")
        sys.exit(1)

    all_results = []
    dates_done = []

    for pf in pred_files:
        base = os.path.basename(pf)
        ds = base.replace('predictions_strikeouts_simplified_', '').replace(
            'predictions_strikeouts_', '').replace('.csv', '')

        # Reuse existing validation CSV if present
        val_file = f"validation_results_{ds}.csv"
        if os.path.exists(val_file):
            try:
                existing = pd.read_csv(val_file)
                # Ensure expected columns exist
                if 'abs_error' in existing.columns:
                    if 'date' not in existing.columns:
                        existing['date'] = ds
                    all_results.append(existing)
                    dates_done.append(ds)
                    continue
            except Exception:
                pass

        # Fetch live
        result = validate_date(ds, quiet=True)
        if result is not None:
            result.to_csv(val_file, index=False)
            all_results.append(result)
            dates_done.append(ds)

    if not all_results:
        print("\n❌ No validation data")
        sys.exit(1)

    combined = pd.concat(all_results, ignore_index=True)
    print(f"\n📅 Dates: {len(dates_done)} ({dates_done[0]} → {dates_done[-1]})")
    print_summary(combined, label="CUMULATIVE")
    print_individual(combined)

    combined.to_csv("validation_cumulative.csv", index=False)
    print("=" * 80)
    print("✅ Saved to: validation_cumulative.csv")
    print("=" * 80)

else:
    # ------------------------------------------------------------------
    # Single-date mode
    # ------------------------------------------------------------------
    if args.date:
        date_str = args.date
    else:
        yesterday = datetime.now() - timedelta(days=1)
        date_str = yesterday.strftime('%Y%m%d')

    result_df = validate_date(date_str)
    if result_df is None:
        print(f"\nAvailable prediction files:")
        for f in sorted(glob.glob('predictions_strikeouts_*.csv')):
            print(f"   • {os.path.basename(f)}")
        sys.exit(1)

    print_summary(result_df, label=date_str)
    print_individual(result_df)

    output_file = f"validation_results_{date_str}.csv"
    result_df.to_csv(output_file, index=False)
    print("=" * 80)
    print(f"✅ Saved to: {output_file}")
    print("=" * 80)
