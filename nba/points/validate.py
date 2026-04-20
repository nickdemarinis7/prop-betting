"""
🏀 NBA Points Prediction Validation
Compares predictions to actual game results

Usage:
    python validate.py                   # Validate yesterday
    python validate.py --date 20260418   # Validate specific date
    python validate.py --cumulative      # Show cumulative stats across all dates
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import glob
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from shared.scrapers.gamelog import GameLogScraper
from nba_api.stats.static import players
import argparse

# ======================================================================
# Constants
# ======================================================================
PLAYOFF_START_MMDD = (4, 14)


def is_playoff_date(date_str):
    """Check if a date falls in the playoff window."""
    dt = datetime.strptime(date_str, '%Y%m%d')
    return dt.month > PLAYOFF_START_MMDD[0] or (
        dt.month == PLAYOFF_START_MMDD[0] and dt.day >= PLAYOFF_START_MMDD[1]
    )


def simulate_pnl(row):
    """
    Simulate P&L for a single player's ladder bets.
    Assumes standard -110 juice when no Min_Odds column is present.
    Returns dict with units_wagered, units_won, net_pnl.
    """
    units_wagered = 0.0
    units_won = 0.0

    for threshold, units_col, hit_col, odds_col in [
        (15, 'Units_15+', 'Hit_15+', 'Min_Odds_15+'),
        (20, 'Units_20+', 'Hit_20+', 'Min_Odds_20+'),
        (25, 'Units_25+', 'Hit_25+', 'Min_Odds_25+'),
    ]:
        units = row.get(units_col, 0)
        if pd.isna(units) or units <= 0:
            continue

        units_wagered += units

        if row.get(hit_col, False):
            odds = row.get(odds_col, -110)
            if pd.isna(odds) or odds == 0:
                odds = -110
            odds = float(odds)

            if odds < 0:
                payout = units * (100 / abs(odds))
            else:
                payout = units * (odds / 100)
            units_won += units + payout

    net = units_won - units_wagered
    return {
        'units_wagered': round(units_wagered, 2),
        'units_won': round(units_won, 2),
        'net_pnl': round(net, 2),
    }


# ======================================================================
# Fetch + validate a single date
# ======================================================================
def validate_date(date_str, gamelog_scraper, all_players, quiet=False):
    """Validate predictions for a single date. Returns results DataFrame or None."""
    predictions_file = f"predictions_production_{date_str}.csv"

    if not os.path.exists(predictions_file):
        if not quiet:
            print(f"❌ File not found: {predictions_file}")
        return None

    predictions = pd.read_csv(predictions_file)
    playoff = is_playoff_date(date_str)

    if not quiet:
        mode = "🏆 PLAYOFFS" if playoff else "📅 Regular Season"
        print(f"\n{mode} — Validating: {date_str}")
        print(f"📁 {predictions_file} ({len(predictions)} predictions)")

    results = []

    for _, row in predictions.iterrows():
        player_name = row['Player']
        projection = row['Proj']

        player_match = [p for p in all_players if p['full_name'] == player_name]
        if not player_match:
            continue

        player_id = player_match[0]['id']
        recent_games = gamelog_scraper.get_recent_games(player_id, n_games=3)

        if recent_games.empty:
            continue

        # Find game matching the validation date
        validate_dt = pd.to_datetime(date_str, format='%Y%m%d').date()
        matched_game = None
        for _, g in recent_games.iterrows():
            game_dt = pd.to_datetime(g['GAME_DATE']).date()
            if game_dt == validate_dt:
                matched_game = g
                break

        if matched_game is None:
            if not quiet:
                last_date = recent_games['GAME_DATE'].iloc[0]
                print(f"   ⚠️  {player_name} did not play on {date_str} (last: {last_date}) — skipping")
            continue

        actual_points = int(matched_game['PTS'])
        game_date = matched_game['GAME_DATE']

        diff = round(actual_points - projection, 1)
        l10 = row.get('L10', projection)
        baseline_diff = round(actual_points - l10, 1) if pd.notna(l10) else None

        result = {
            'Date': date_str,
            'Player': player_name,
            'Type': row.get('Type', 'TOP PLAY'),
            'Projection': projection,
            'L10': l10,
            'Actual': actual_points,
            'Diff': diff,
            'Baseline_Diff': baseline_diff,
            'Hit_15+': actual_points >= 15,
            'Hit_20+': actual_points >= 20,
            'Hit_25+': actual_points >= 25,
            'Hit_30+': actual_points >= 30,
            'Prob_15+': row.get('15+%', 0),
            'Prob_20+': row.get('20+%', 0),
            'Prob_25+': row.get('25+%', 0),
            'Ladder_Value': row.get('Ladder_Value', 0),
            'Tier': row.get('Tier', 0),
            'Conf': row.get('Conf', ''),
            'Red_Flags': row.get('Red_Flags', 'None'),
            'Is_Playoff': playoff,
            'Game_Date': game_date,
        }

        # Carry over unit columns for P&L
        for col in ['Units_15+', 'Units_20+', 'Units_25+',
                     'Min_Odds_15+', 'Min_Odds_20+', 'Min_Odds_25+']:
            result[col] = row.get(col, 0)

        results.append(result)

    if not results:
        if not quiet:
            print("   ❌ No results to validate")
        return None

    return pd.DataFrame(results)


# ======================================================================
# Display helpers
# ======================================================================
def print_summary(results_df, label=""):
    """Print comprehensive accuracy summary."""
    top_plays = results_df[results_df['Type'] == 'TOP PLAY']
    fades = results_df[results_df['Type'] == 'FADE']
    n_playoff = results_df['Is_Playoff'].sum()
    n_regular = len(results_df) - n_playoff

    print("=" * 80)
    print(f"📊 RESULTS SUMMARY{' — ' + label if label else ''}")
    print("=" * 80)
    print(f"\nPlayers validated: {len(results_df)} ({n_regular} reg season, {n_playoff} playoff)")

    # --- TOP PLAYS ---
    if len(top_plays) > 0:
        print(f"\nTOP PLAYS ({len(top_plays)} players):")
        print("-" * 60)

        for thresh, prob_col, hit_col in [('15+', 'Prob_15+', 'Hit_15+'),
                                           ('20+', 'Prob_20+', 'Hit_20+'),
                                           ('25+', 'Prob_25+', 'Hit_25+')]:
            actual_rate = top_plays[hit_col].mean() * 100
            expected_rate = top_plays[prob_col].mean()
            gap = actual_rate - expected_rate
            gap_icon = "✅" if abs(gap) < 10 else ("📈" if gap > 0 else "📉")
            print(f"   {thresh} PTS Hit Rate:  {actual_rate:5.1f}% (Expected: {expected_rate:.0f}%) {gap_icon} {gap:+.1f}pp")

        # Model MAE vs baseline (L10) MAE
        model_mae = top_plays['Diff'].abs().mean()
        rmse = np.sqrt((top_plays['Diff'] ** 2).mean())
        bias = top_plays['Diff'].mean()
        print(f"\n   Model MAE:  {model_mae:.2f} pts")
        print(f"   RMSE:       {rmse:.2f} pts")
        bias_dir = "over-projecting" if bias < 0 else "under-projecting"
        print(f"   Bias:       {bias:+.2f} pts ({bias_dir})")

        if 'Baseline_Diff' in top_plays.columns:
            baseline_valid = top_plays['Baseline_Diff'].dropna()
            if len(baseline_valid) > 0:
                baseline_mae = baseline_valid.abs().mean()
                improvement = baseline_mae - model_mae
                pct = (improvement / baseline_mae * 100) if baseline_mae > 0 else 0
                icon = "✅" if improvement > 0 else "⚠️"
                print(f"\n   Baseline MAE: {baseline_mae:.2f} pts (just using L10 avg)")
                print(f"   {icon} Model is {abs(improvement):.2f} pts {'better' if improvement > 0 else 'worse'} ({abs(pct):.0f}%)")

        # Accuracy buckets
        w3 = (top_plays['Diff'].abs() <= 3).mean()
        w5 = (top_plays['Diff'].abs() <= 5).mean()
        w8 = (top_plays['Diff'].abs() <= 8).mean()
        print(f"\n📏 Accuracy:")
        print(f"   Within 3 pts: {w3:.0%} | Within 5 pts: {w5:.0%} | Within 8 pts: {w8:.0%}")

    # --- By projection bucket ---
    print(f"\n📊 By Projection Range:")
    for lo, hi, lbl in [(0, 15, '<15'), (15, 22, '15-22'), (22, 30, '22-30'), (30, 50, '30+')]:
        subset = results_df[(results_df['Projection'] >= lo) & (results_df['Projection'] < hi)]
        if len(subset) > 0:
            print(f"   {lbl:5s}: n={len(subset):3d}  MAE={subset['Diff'].abs().mean():.2f}  Bias={subset['Diff'].mean():+.2f}")

    # --- By confidence ---
    if 'Conf' in results_df.columns and results_df['Conf'].notna().any():
        print(f"\n🎯 By Confidence:")
        for conf in ['HIGH', 'MEDIUM', 'LOW']:
            subset = results_df[results_df['Conf'] == conf]
            if len(subset) > 0:
                print(f"   {conf:6s}: n={len(subset):3d}  MAE={subset['Diff'].abs().mean():.2f}  Bias={subset['Diff'].mean():+.2f}")

    # --- By tier ---
    if 'Tier' in results_df.columns:
        print(f"\n🏅 By Tier:")
        for tier in [1, 2, 3]:
            subset = results_df[results_df['Tier'] == tier]
            if len(subset) > 0:
                print(f"   Tier {tier}: n={len(subset):3d}  MAE={subset['Diff'].abs().mean():.2f}  Bias={subset['Diff'].mean():+.2f}")

    # --- FADES ---
    if len(fades) > 0:
        print(f"\nFADES ({len(fades)} players):")
        print("-" * 60)
        fade_15_rate = fades['Hit_15+'].mean() * 100
        fade_20_rate = fades['Hit_20+'].mean() * 100
        print(f"   15+ PTS Hit Rate: {fade_15_rate:.0f}% (lower is better — we're fading)")
        print(f"   20+ PTS Hit Rate: {fade_20_rate:.0f}%")

    # --- P&L ---
    pnl_data = results_df.apply(simulate_pnl, axis=1, result_type='expand')
    total_wagered = pnl_data['units_wagered'].sum()
    total_won = pnl_data['units_won'].sum()
    net = pnl_data['net_pnl'].sum()

    if total_wagered > 0:
        roi = (net / total_wagered) * 100
        print(f"\n💰 P&L SIMULATION (based on unit recommendations):")
        print("-" * 60)
        print(f"   Units Wagered:  {total_wagered:.2f}u")
        print(f"   Units Returned: {total_won:.2f}u")
        icon = "✅" if net >= 0 else "❌"
        print(f"   {icon} Net P&L:      {net:+.2f}u (ROI: {roi:+.1f}%)")

    print()


def print_individual(results_df):
    """Print individual player results."""
    print("=" * 80)
    print("🎯 INDIVIDUAL RESULTS")
    print("=" * 80)
    print()

    sorted_df = results_df.sort_values('Ladder_Value', ascending=False)

    for _, row in sorted_df.iterrows():
        player = row['Player']
        proj = row['Projection']
        actual = row['Actual']
        diff = row['Diff']
        playoff_tag = " 🏆" if row.get('Is_Playoff', False) else ""

        if abs(diff) <= 3.0:
            icon = "✅"
        elif abs(diff) <= 5.0:
            icon = "⚠️"
        else:
            icon = "❌"

        conf = row.get('Conf', '')
        conf_icon = '🟢' if conf == 'HIGH' else '🟡' if conf == 'MEDIUM' else '🔴' if conf == 'LOW' else ''
        flags = row.get('Red_Flags', 'None')
        flag_str = f" ⚠️ {flags}" if flags not in ('None', '') and pd.notna(flags) else ''

        print(f"{icon} {player:25} | Proj: {proj:5.1f} | Actual: {actual:2d} | Diff: {diff:+6.1f} {conf_icon}{playoff_tag}{flag_str}")

        ladder = []
        for thresh, hit_col, prob_col in [('15+', 'Hit_15+', 'Prob_15+'),
                                           ('20+', 'Hit_20+', 'Prob_20+'),
                                           ('25+', 'Hit_25+', 'Prob_25+')]:
            prob = row.get(prob_col, 0)
            hit = row.get(hit_col, False)
            if hit:
                ladder.append(f"{thresh} ✅ ({prob:.0f}%)")
            elif prob > 5:
                ladder.append(f"{thresh} ❌ ({prob:.0f}%)")
        print(f"   Ladder: {' | '.join(ladder)}")

        # P&L for this player
        pnl = simulate_pnl(row)
        if pnl['units_wagered'] > 0:
            pnl_icon = "✅" if pnl['net_pnl'] >= 0 else "❌"
            print(f"   💰 {pnl['units_wagered']:.2f}u wagered → {pnl_icon} {pnl['net_pnl']:+.2f}u")
        print()


# ======================================================================
# Main
# ======================================================================
parser = argparse.ArgumentParser(description="Validate NBA points predictions")
parser.add_argument('--date', type=str, help='Date to validate (YYYYMMDD)', default=None)
parser.add_argument('--cumulative', action='store_true', help='Show cumulative stats across all dates')
args = parser.parse_args()

print("=" * 80)
print("📊 NBA POINTS VALIDATION — How Did We Do?")
print("=" * 80)

gamelog_scraper = GameLogScraper()
all_players_list = players.get_players()

if args.cumulative:
    # ------------------------------------------------------------------
    # Cumulative mode
    # ------------------------------------------------------------------
    pred_files = sorted(glob.glob('predictions_production_*.csv'))
    if not pred_files:
        print("\n❌ No prediction files found")
        sys.exit(1)

    all_results = []
    dates_processed = []

    for pf in pred_files:
        ds = os.path.basename(pf).replace('predictions_production_', '').replace('.csv', '')

        # Reuse existing validation CSV if present
        val_file = f"validation_results_{ds}.csv"
        if os.path.exists(val_file):
            try:
                existing = pd.read_csv(val_file)
                if 'Diff' in existing.columns:
                    if 'Is_Playoff' not in existing.columns:
                        existing['Is_Playoff'] = is_playoff_date(ds)
                    if 'Date' not in existing.columns:
                        existing['Date'] = ds
                    all_results.append(existing)
                    dates_processed.append(ds)
                    continue
            except Exception:
                pass

        # Fetch live
        result = validate_date(ds, gamelog_scraper, all_players_list, quiet=True)
        if result is not None:
            result.to_csv(val_file, index=False)
            all_results.append(result)
            dates_processed.append(ds)

    if not all_results:
        print("\n❌ No validation data available")
        sys.exit(1)

    combined = pd.concat(all_results, ignore_index=True)

    print(f"\n📅 Dates: {len(dates_processed)} ({dates_processed[0]} → {dates_processed[-1]})")
    print_summary(combined, label="CUMULATIVE")
    print_individual(combined)

    output_file = "validation_cumulative.csv"
    combined.to_csv(output_file, index=False)
    print("=" * 80)
    print(f"✅ Saved cumulative results to: {output_file}")
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

    print(f"\n🔄 Fetching actual game results for {date_str}...")

    results_df = validate_date(date_str, gamelog_scraper, all_players_list)

    if results_df is None:
        print(f"\nAvailable prediction files:")
        for f in sorted(glob.glob('predictions_production_*.csv')):
            print(f"   • {os.path.basename(f)}")
        sys.exit(1)

    print_summary(results_df, label=date_str)
    print_individual(results_df)

    output_file = f"validation_results_{date_str}.csv"
    results_df.to_csv(output_file, index=False)
    print("=" * 80)
    print(f"✅ Saved detailed results to: {output_file}")
    print("=" * 80)
