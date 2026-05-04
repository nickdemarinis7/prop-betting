"""
Validate Picks - Check how yesterday's predictions performed
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
PLAYOFF_START_MMDD = (4, 14)  # NBA playoffs typically start mid-April


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
        (5, 'Units_5+', 'Hit_5+', 'Min_Odds_5+'),
        (7, 'Units_7+', 'Hit_7+', 'Min_Odds_7+'),
        (10, 'Units_10+', 'Hit_10+', 'Min_Odds_10+'),
    ]:
        units = row.get(units_col, 0)
        if pd.isna(units) or units <= 0:
            continue

        units_wagered += units

        if row.get(hit_col, False):
            # Use min odds from prediction as proxy for the line we'd bet
            odds = row.get(odds_col, -110)
            if pd.isna(odds) or odds == 0:
                odds = -110
            odds = float(odds)

            if odds < 0:
                payout = units * (100 / abs(odds))
            else:
                payout = units * (odds / 100)
            units_won += units + payout  # Return of stake + profit
        # else: loss — units_won stays 0 for this leg

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
        recent_games = gamelog_scraper.get_recent_games(player_id, n_games=1)

        if recent_games.empty:
            continue

        actual_assists = recent_games['AST'].iloc[0]
        game_date = recent_games['GAME_DATE'].iloc[0]

        # Verify game was on the validation date
        game_dt = pd.to_datetime(game_date)
        validate_dt = pd.to_datetime(date_str, format='%Y%m%d')
        if game_dt.date() != validate_dt.date():
            if not quiet:
                print(f"   ⚠️  {player_name} did not play on {date_str} (last game: {game_date}) — skipping")
            continue

        diff = round(actual_assists - projection, 1)
        l10 = row.get('L10', projection)  # Baseline: L10 average
        baseline_diff = round(actual_assists - l10, 1) if pd.notna(l10) else None

        result = {
            'Date': date_str,
            'Player': player_name,
            'Type': row['Type'],
            'Projection': projection,
            'L10': l10,
            'Actual': int(actual_assists),
            'Diff': diff,
            'Baseline_Diff': baseline_diff,
            'Hit_5+': actual_assists >= 5,
            'Hit_7+': actual_assists >= 7,
            'Hit_10+': actual_assists >= 10,
            'Prob_5+': row.get('5+%', 0),
            'Prob_7+': row.get('7+%', 0),
            'Prob_10+': row.get('10+%', 0),
            'Ladder_Value': row.get('Ladder_Value', 0),
            'Is_Playoff': playoff,
            'Game_Date': game_date,
        }

        # Carry over unit columns for P&L
        for col in ['Units_5+', 'Units_7+', 'Units_10+', 'Min_Odds_5+', 'Min_Odds_7+', 'Min_Odds_10+']:
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
    """Print accuracy summary for a results DataFrame."""
    top_plays = results_df[results_df['Type'] == 'TOP PLAY']
    fades = results_df[results_df['Type'] == 'FADE']
    n_playoff = results_df['Is_Playoff'].sum()
    n_regular = len(results_df) - n_playoff

    print("=" * 80)
    print(f"📊 RESULTS SUMMARY{' — ' + label if label else ''}")
    print("=" * 80)
    print(f"\nPlayers validated: {len(results_df)} ({n_regular} reg season, {n_playoff} playoff)")

    if len(top_plays) > 0:
        print(f"\nTOP PLAYS ({len(top_plays)} players):")
        print("-" * 60)

        for thresh, prob_col, hit_col in [('5+', 'Prob_5+', 'Hit_5+'),
                                           ('7+', 'Prob_7+', 'Hit_7+'),
                                           ('10+', 'Prob_10+', 'Hit_10+')]:
            actual_rate = top_plays[hit_col].mean() * 100
            expected_rate = top_plays[prob_col].mean()
            gap = actual_rate - expected_rate
            gap_icon = "✅" if abs(gap) < 8 else ("📈" if gap > 0 else "📉")
            print(f"   {thresh} AST Hit Rate:  {actual_rate:5.1f}% (Expected: {expected_rate:.0f}%) {gap_icon} {gap:+.1f}pp")

        # Model MAE vs baseline (L10 naive) MAE
        model_mae = top_plays['Diff'].abs().mean()
        print(f"\n   Model MAE:    {model_mae:.2f} assists")

        if 'Baseline_Diff' in top_plays.columns:
            baseline_valid = top_plays['Baseline_Diff'].dropna()
            if len(baseline_valid) > 0:
                baseline_mae = baseline_valid.abs().mean()
                improvement = baseline_mae - model_mae
                pct = (improvement / baseline_mae * 100) if baseline_mae > 0 else 0
                icon = "✅" if improvement > 0 else "⚠️"
                print(f"   Baseline MAE: {baseline_mae:.2f} assists (just using L10 avg)")
                print(f"   {icon} Model is {abs(improvement):.2f} assists {'better' if improvement > 0 else 'worse'} ({abs(pct):.0f}%)")

        # Bias
        bias = top_plays['Diff'].mean()
        bias_dir = "over-projecting" if bias < 0 else "under-projecting"
        print(f"\n   Bias: {bias:+.2f} (model is {bias_dir})")

    if len(fades) > 0:
        print(f"\nFADES ({len(fades)} players):")
        print("-" * 60)
        fade_5_rate = fades['Hit_5+'].mean() * 100
        print(f"   5+ AST Hit Rate: {fade_5_rate:.0f}% (lower is better — we're fading these)")

    # P&L
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
        playoff_tag = " 🏆" if row['Is_Playoff'] else ""

        if abs(diff) <= 1.5:
            icon = "✅"
        elif abs(diff) <= 3.0:
            icon = "⚠️"
        else:
            icon = "❌"

        print(f"{icon} {player:25} | Proj: {proj:4.1f} | Actual: {actual:2d} | Diff: {diff:+5.1f}{playoff_tag}")

        ladder = []
        for thresh, hit_col, prob_col in [('5+', 'Hit_5+', 'Prob_5+'),
                                           ('7+', 'Hit_7+', 'Prob_7+'),
                                           ('10+', 'Hit_10+', 'Prob_10+')]:
            prob = row[prob_col]
            hit = row[hit_col]
            if hit:
                ladder.append(f"{thresh} ✅ ({prob:.0f}%)")
            elif prob > 5:
                ladder.append(f"{thresh} ❌ ({prob:.0f}%)")
        print(f"   Ladder: {' | '.join(ladder)}")

        # P&L for this player
        pnl = simulate_pnl(row)
        if pnl['units_wagered'] > 0:
            icon = "✅" if pnl['net_pnl'] >= 0 else "❌"
            print(f"   💰 {pnl['units_wagered']:.2f}u wagered → {icon} {pnl['net_pnl']:+.2f}u")
        print()


# ======================================================================
# Main
# ======================================================================
parser = argparse.ArgumentParser(description="Validate NBA assists predictions")
parser.add_argument('--date', type=str, help='Date to validate (YYYYMMDD)', default=None)
parser.add_argument('--cumulative', action='store_true', help='Show cumulative stats across all dates')
args = parser.parse_args()

print("=" * 80)
print("📊 PICKS VALIDATION — How Did We Do?")
print("=" * 80)

gamelog_scraper = GameLogScraper()
all_players_list = players.get_players()

if args.cumulative:
    # ------------------------------------------------------------------
    # Cumulative mode: load all existing validation CSVs + validate remaining
    # ------------------------------------------------------------------
    pred_files = sorted(glob.glob('predictions_production_*.csv'))
    if not pred_files:
        print("\n❌ No prediction files found")
        sys.exit(1)

    all_results = []
    dates_processed = []

    for pf in pred_files:
        ds = os.path.basename(pf).replace('predictions_production_', '').replace('.csv', '')

        # Check if we already have a validation CSV
        val_file = f"validation_results_{ds}.csv"
        if os.path.exists(val_file):
            try:
                existing = pd.read_csv(val_file)
                if 'Is_Playoff' not in existing.columns:
                    existing['Is_Playoff'] = is_playoff_date(ds)
                all_results.append(existing)
                dates_processed.append(ds)
                continue
            except Exception:
                pass

        # Fetch live data for this date
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
