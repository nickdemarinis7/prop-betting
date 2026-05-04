"""
🏀 NBA Assists Prediction Validation
Compares predictions to actual game results.

Usage:
    python validate.py                   # Validate yesterday
    python validate.py --date 20260427   # Validate specific date
    python validate.py --cumulative      # Cumulative stats across all dates

Reads predictions_assists_YYYYMMDD.csv (current format) — falls back to
legacy predictions_production_YYYYMMDD.csv where it can.

P&L is computed from the recommended_side / book_odds / kelly_units
columns the predictor now writes. If those columns are missing (legacy
files), we fall back to a flat-1u/-110 simulation against any prob_X+
columns that are present.
"""

import sys
import os
import glob
import argparse
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', category=FutureWarning)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from shared.scrapers.gamelog import GameLogScraper
from nba_api.stats.static import players


# ======================================================================
# Helpers
# ======================================================================
PLAYOFF_START_MMDD = (4, 14)


def is_playoff_date(date_str):
    dt = datetime.strptime(date_str, '%Y%m%d')
    return dt.month > PLAYOFF_START_MMDD[0] or (
        dt.month == PLAYOFF_START_MMDD[0] and dt.day >= PLAYOFF_START_MMDD[1]
    )


def _normalize_columns(predictions):
    """Return a copy with both legacy (capitalized) and new (snake_case)
    column names available, so downstream code can read whichever exists.
    """
    df = predictions.copy()
    legacy_to_new = {
        'Player': 'player_name',
        'Team': 'team',
        'Opp': 'opponent',
        'Home': 'is_home',
        'Proj': 'projection',
        'L10': 'ast_last10_avg',
        'Conf': 'confidence',
        'Red_Flags': 'red_flags',
        '5+%': 'prob_5+',
        '7+%': 'prob_7+',
        '10+%': 'prob_10+',
    }
    for legacy, new in legacy_to_new.items():
        if legacy in df.columns and new not in df.columns:
            df[new] = df[legacy]
        if new in df.columns and legacy not in df.columns:
            df[legacy] = df[new]
    return df


def _amer_to_decimal(odds):
    if odds is None or pd.isna(odds):
        return None
    o = float(odds)
    return 1 + (o / 100) if o > 0 else 1 + (100 / abs(o))


def simulate_pnl(row):
    """Simulate P&L for a single validated row using the predictor's
    recommended bet (recommended_side / book_odds / kelly_units).

    Falls back to legacy ladder simulation if those columns are absent.
    """
    rec = row.get('Recommended_Side', row.get('recommended_side', ''))
    odds = row.get('Book_Odds', row.get('book_odds'))
    line = row.get('Book_Line', row.get('book_line'))
    kelly = row.get('Kelly_Units', row.get('kelly_units'))

    if rec in ('OVER', 'UNDER') and pd.notna(odds) and pd.notna(line):
        stake = float(kelly) if pd.notna(kelly) and float(kelly) > 0 else 1.0
        decimal = _amer_to_decimal(odds)
        if decimal is None:
            return {'units_wagered': 0.0, 'units_won': 0.0, 'net_pnl': 0.0}
        actual = row['Actual']
        # OVER hits when actual > line (sportsbooks always set .5 lines)
        hit = (actual > line) if rec == 'OVER' else (actual < line)
        won = stake * decimal if hit else 0.0
        return {
            'units_wagered': round(stake, 2),
            'units_won': round(won, 2),
            'net_pnl': round(won - stake, 2),
        }

    # ---- Legacy fallback: bet on prob_X+ ladder rungs at -110 flat 1u
    units_w = 0.0
    units_won = 0.0
    for thresh, prob_col, hit_col in [
        (5, 'prob_5+', 'Hit_5+'),
        (7, 'prob_7+', 'Hit_7+'),
        (10, 'prob_10+', 'Hit_10+'),
    ]:
        prob = row.get(prob_col)
        if pd.isna(prob) or float(prob) < 55:  # legacy probs are 0-100
            # prob may be 0-1 instead — handle both
            if pd.isna(prob) or float(prob) < 0.55:
                continue
        units_w += 1.0
        if row.get(hit_col, False):
            units_won += 1.0 + (100 / 110)
    return {
        'units_wagered': round(units_w, 2),
        'units_won': round(units_won, 2),
        'net_pnl': round(units_won - units_w, 2),
    }


# ======================================================================
# Validate one date
# ======================================================================
def validate_date(date_str, gamelog_scraper, all_players, quiet=False):
    """Validate predictions for a single date. Returns DataFrame or None."""
    new_file = f"predictions_assists_{date_str}.csv"
    legacy_file = f"predictions_production_{date_str}.csv"
    if os.path.exists(new_file):
        predictions_file = new_file
    elif os.path.exists(legacy_file):
        predictions_file = legacy_file
    else:
        if not quiet:
            print(f"❌ File not found: {new_file} or {legacy_file}")
        return None

    predictions = pd.read_csv(predictions_file)
    predictions = _normalize_columns(predictions)
    playoff = is_playoff_date(date_str)

    if not quiet:
        mode = "🏆 PLAYOFFS" if playoff else "📅 Regular Season"
        print(f"\n{mode} — Validating: {date_str}")
        print(f"📁 {predictions_file} ({len(predictions)} predictions)")

    name_to_id = {p['full_name']: p['id'] for p in all_players}
    validate_dt = pd.to_datetime(date_str, format='%Y%m%d').date()

    results = []
    for _, row in predictions.iterrows():
        player_name = row.get('player_name')
        if pd.isna(player_name) or not player_name:
            continue
        projection = row.get('projection')
        if pd.isna(projection):
            continue

        player_id = name_to_id.get(player_name)
        if player_id is None:
            # Try last-name fuzzy match
            last = str(player_name).split()[-1].lower()
            match = next(
                (p for p in all_players
                 if p['full_name'].split()[-1].lower() == last
                 and p['full_name'].split()[0][0].lower() == str(player_name)[0].lower()),
                None,
            )
            if match is None:
                continue
            player_id = match['id']

        recent_games = gamelog_scraper.get_recent_games(player_id, n_games=3)
        if recent_games.empty:
            continue

        matched = None
        for _, g in recent_games.iterrows():
            if pd.to_datetime(g['GAME_DATE']).date() == validate_dt:
                matched = g
                break
        if matched is None:
            if not quiet:
                last_date = recent_games['GAME_DATE'].iloc[0]
                print(f"   ⚠️  {player_name} did not play on {date_str} "
                      f"(last: {last_date}) — skipping")
            continue

        actual = int(matched['AST'])
        diff = round(actual - float(projection), 1)
        l10 = row.get('ast_last10_avg', projection)
        try:
            baseline_diff = round(actual - float(l10), 1)
        except (TypeError, ValueError):
            baseline_diff = None

        result = {
            'Date': date_str,
            'Is_Playoff': playoff,
            'Player': player_name,
            'Team': row.get('team', ''),
            'Opponent': row.get('opponent', ''),
            'Projection': float(projection),
            'L10': l10,
            'Actual': actual,
            'Diff': diff,
            'Baseline_Diff': baseline_diff,
            'Confidence': row.get('confidence', ''),
            'Red_Flags': row.get('red_flags', 'None'),
            'Recommended_Side': row.get('recommended_side', ''),
            'Book_Line': row.get('book_line'),
            'Book_Odds': row.get('book_odds'),
            'Our_Prob': row.get('our_prob'),
            'Book_Prob': row.get('book_prob'),
            'Edge_PP': row.get('edge_pp'),
            'EV': row.get('ev'),
            'Kelly_Units': row.get('kelly_units'),
            'Bookmaker': row.get('bookmaker', ''),
            'Hit_3+': actual >= 3,
            'Hit_5+': actual >= 5,
            'Hit_7+': actual >= 7,
            'Hit_10+': actual >= 10,
        }
        # Carry whatever ladder probs exist
        for col in ['prob_3+', 'prob_5+', 'prob_7+', 'prob_10+']:
            if col in row.index:
                result[col] = row.get(col)
        results.append(result)

        if not quiet:
            icon = "✅" if abs(diff) <= 3 else "⚠️" if abs(diff) <= 5 else "❌"
            print(f"   {icon} {player_name:25s} Proj:{float(projection):5.1f} "
                  f"Act:{actual:2d} AST  Diff:{diff:+5.1f}")

    if not results:
        if not quiet:
            print("   ❌ No results to validate")
        return None
    return pd.DataFrame(results)


# ======================================================================
# Display
# ======================================================================
def print_summary(df, label=""):
    print("=" * 80)
    print(f"📊 NBA POINTS VALIDATION{' — ' + label if label else ''}")
    print("=" * 80)

    n = len(df)
    n_playoff = int(df['Is_Playoff'].sum()) if 'Is_Playoff' in df.columns else 0
    print(f"\nPlayers validated: {n} ({n - n_playoff} reg season, {n_playoff} playoff)")

    # Overall accuracy
    mae = df['Diff'].abs().mean()
    rmse = np.sqrt((df['Diff'] ** 2).mean())
    bias = df['Diff'].mean()
    print(f"\n🎯 Overall:")
    print(f"   MAE:    {mae:.2f} AST")
    print(f"   RMSE:   {rmse:.2f} AST")
    print(f"   Bias:   {bias:+.2f} AST ({'over' if bias < 0 else 'under'}-projecting)")

    w1 = (df['Diff'].abs() <= 1).mean()
    w2 = (df['Diff'].abs() <= 2).mean()
    w3 = (df['Diff'].abs() <= 3).mean()
    print(f"\n📏 Accuracy:")
    print(f"   Within 1 AST: {w1:.0%} | Within 2 AST: {w2:.0%} | Within 3 AST: {w3:.0%}")

    # Baseline (L10)
    if 'Baseline_Diff' in df.columns:
        valid = df['Baseline_Diff'].dropna()
        if len(valid) > 0:
            base_mae = valid.abs().mean()
            improvement = base_mae - mae
            pct = (improvement / base_mae * 100) if base_mae > 0 else 0
            icon = "✅" if improvement > 0 else "⚠️"
            print(f"\n📐 Baseline Comparison (just using L10 avg):")
            print(f"   Baseline MAE: {base_mae:.2f} AST")
            print(f"   Model MAE:    {mae:.2f} AST")
            print(f"   {icon} Model is {abs(improvement):.2f} AST "
                  f"{'better' if improvement > 0 else 'worse'} ({abs(pct):.0f}%)")

    # By projection bucket
    print(f"\n📊 By Projection Range:")
    for lo, hi, lbl in [(0, 3, '<3'), (3, 5, '3-5'),
                         (5, 8, '5-8'), (8, 20, '8+')]:
        sub = df[(df['Projection'] >= lo) & (df['Projection'] < hi)]
        if len(sub) > 0:
            print(f"   {lbl:5s}: n={len(sub):3d}  "
                  f"MAE={sub['Diff'].abs().mean():.2f}  "
                  f"Bias={sub['Diff'].mean():+.2f}")

    # By confidence
    if 'Confidence' in df.columns and df['Confidence'].notna().any():
        print(f"\n🎯 By Confidence:")
        for conf in ['HIGH', 'MEDIUM', 'LOW']:
            sub = df[df['Confidence'] == conf]
            if len(sub) > 0:
                print(f"   {conf:6s}: n={len(sub):3d}  "
                      f"MAE={sub['Diff'].abs().mean():.2f}  "
                      f"Bias={sub['Diff'].mean():+.2f}")

    # Recommended-side hit rates (the actual betting decisions)
    if 'Recommended_Side' in df.columns:
        plays = df[df['Recommended_Side'].isin(['OVER', 'UNDER'])
                   & df['Book_Line'].notna()]
        if not plays.empty:
            wins = (
                ((plays['Recommended_Side'] == 'OVER')
                 & (plays['Actual'] > plays['Book_Line']))
                | ((plays['Recommended_Side'] == 'UNDER')
                   & (plays['Actual'] < plays['Book_Line']))
            )
            print(f"\n📈 Recommended Plays:")
            print(f"   {len(plays)} bets  ({(plays['Recommended_Side'] == 'OVER').sum()} OVER, "
                  f"{(plays['Recommended_Side'] == 'UNDER').sum()} UNDER)")
            print(f"   Hit rate: {wins.mean():.1%}  "
                  f"(avg our_prob: {plays['Our_Prob'].mean():.1%})")
            calib_gap = wins.mean() - plays['Our_Prob'].mean()
            icon = "✅" if abs(calib_gap) < 0.08 else ("📈" if calib_gap > 0 else "📉")
            print(f"   Calibration: {calib_gap:+.1%} {icon} "
                  f"(actual vs predicted hit rate)")

    # Calibration buckets on our_prob
    if 'Our_Prob' in df.columns and df['Our_Prob'].notna().any():
        plays = df[df['Recommended_Side'].isin(['OVER', 'UNDER'])
                   & df['Our_Prob'].notna()
                   & df['Book_Line'].notna()].copy()
        if not plays.empty:
            plays['hit'] = (
                ((plays['Recommended_Side'] == 'OVER')
                 & (plays['Actual'] > plays['Book_Line']))
                | ((plays['Recommended_Side'] == 'UNDER')
                   & (plays['Actual'] < plays['Book_Line']))
            ).astype(int)
            print(f"\n📐 Calibration by our_prob bucket:")
            print(f"   {'bucket':<13} {'n':>4}  {'avg prob':>9}  "
                  f"{'actual':>8}  {'gap':>7}")
            for lo, hi in [(0.50, 0.60), (0.60, 0.70),
                           (0.70, 0.80), (0.80, 1.00)]:
                sub = plays[(plays['Our_Prob'] >= lo)
                            & (plays['Our_Prob'] < hi)]
                if sub.empty:
                    continue
                avg_p = sub['Our_Prob'].mean()
                actual = sub['hit'].mean()
                gap = actual - avg_p
                icon = "✅" if abs(gap) < 0.08 else ("📈" if gap > 0 else "📉")
                print(f"   {lo:.2f}-{hi:.2f}    {len(sub):>4}  "
                      f"{avg_p:>9.1%}  {actual:>8.1%}  {gap:>+7.1%} {icon}")

    # P&L
    pnl = df.apply(simulate_pnl, axis=1, result_type='expand')
    total_w = pnl['units_wagered'].sum()
    total_won = pnl['units_won'].sum()
    net = pnl['net_pnl'].sum()
    if total_w > 0:
        roi = (net / total_w) * 100
        icon = "✅" if net >= 0 else "❌"
        print(f"\n💰 P&L SIMULATION (kelly stakes at actual book odds):")
        print(f"   Units wagered:  {total_w:.2f}u")
        print(f"   Units returned: {total_won:.2f}u")
        print(f"   {icon} Net P&L:      {net:+.2f}u (ROI: {roi:+.1f}%)")
    print()


def print_individual(df):
    print("=" * 80)
    print("🎯 INDIVIDUAL RESULTS")
    print("=" * 80)
    print()

    sorted_df = df.sort_values('Diff', key=lambda x: x.abs(), ascending=False)

    for _, row in sorted_df.iterrows():
        proj = row['Projection']
        actual = int(row['Actual'])
        diff = row['Diff']
        playoff_tag = " 🏆" if row.get('Is_Playoff', False) else ""
        if abs(diff) <= 3:
            icon = "✅"
        elif abs(diff) <= 5:
            icon = "⚠️"
        else:
            icon = "❌"
        conf = row.get('Confidence', '')
        conf_icon = (
            '🟢' if conf == 'HIGH' else '🟡' if conf == 'MEDIUM'
            else '🔴' if conf == 'LOW' else ''
        )
        flags = row.get('Red_Flags', 'None')
        flag_str = (
            f" ⚠️ {flags}" if flags not in ('None', '') and pd.notna(flags) else ''
        )
        print(f"{icon} {row['Player']:25s} | Proj:{proj:5.1f} | "
              f"Act:{actual:2d} AST | Diff:{diff:+6.1f} {conf_icon}{playoff_tag}{flag_str}")

        # Bet outcome
        rec = row.get('Recommended_Side', '')
        if rec in ('OVER', 'UNDER') and pd.notna(row.get('Book_Line')):
            line = row['Book_Line']
            odds = row.get('Book_Odds')
            won = (
                (rec == 'OVER' and actual > line)
                or (rec == 'UNDER' and actual < line)
            )
            bet_icon = "✅" if won else "❌"
            kelly = row.get('Kelly_Units')
            stake = (
                f" {float(kelly):.2f}u" if pd.notna(kelly) and float(kelly) > 0
                else ''
            )
            odds_str = (
                f" @ {int(odds):+d}" if pd.notna(odds) else ''
            )
            print(f"   {bet_icon} {rec} {line}{odds_str}{stake}")
        print()


# ======================================================================
# Main
# ======================================================================
parser = argparse.ArgumentParser(description="Validate NBA assists predictions")
parser.add_argument('--date', type=str, default=None,
                    help='Date to validate (YYYYMMDD)')
parser.add_argument('--cumulative', action='store_true',
                    help='Cumulative stats across all dates')
args = parser.parse_args()

print("=" * 80)
print("🏀 NBA ASSISTS VALIDATION — How Did We Do?")
print("=" * 80)

gamelog_scraper = GameLogScraper()
all_players_list = players.get_players()

if args.cumulative:
    all_files = (
        glob.glob('predictions_assists_*.csv')
        + glob.glob('predictions_production_*.csv')
    )
    by_date = {}
    for pf in all_files:
        base = os.path.basename(pf)
        ds = (
            base.replace('predictions_assists_', '')
                .replace('predictions_production_', '')
                .replace('.csv', '')
        )
        # Prefer the new filename when both exist
        if ds not in by_date or 'predictions_assists_' in base:
            by_date[ds] = pf
    pred_files = [by_date[d] for d in sorted(by_date)]
    if not pred_files:
        print("\n❌ No prediction files found")
        sys.exit(1)

    all_results = []
    for pf in pred_files:
        base = os.path.basename(pf)
        ds = (
            base.replace('predictions_assists_', '')
                .replace('predictions_production_', '')
                .replace('.csv', '')
        )
        val_file = f"validation_results_{ds}.csv"
        if os.path.exists(val_file):
            try:
                cached = pd.read_csv(val_file)
                # Migrate cached schema to new column names if needed
                if 'Conf' in cached.columns and 'Confidence' not in cached.columns:
                    cached['Confidence'] = cached['Conf']
                if 'projection' in cached.columns and 'Projection' not in cached.columns:
                    cached['Projection'] = cached['projection']
                all_results.append(cached)
                print(f"   ↪ Reused cached validation for {ds} ({len(cached)} rows)")
                continue
            except Exception:
                pass
        df = validate_date(ds, gamelog_scraper, all_players_list, quiet=True)
        if df is None:
            continue
        df.to_csv(val_file, index=False)
        all_results.append(df)
        print(f"   ✓ Validated {ds} ({len(df)} players)")

    if not all_results:
        print("\n❌ Nothing to summarize")
        sys.exit(1)

    combined = pd.concat(all_results, ignore_index=True)
    print_summary(combined, label="CUMULATIVE")
    print(f"\n✅ {len(combined)} player-days across "
          f"{combined['Date'].nunique()} dates")
else:
    if args.date:
        date_str = args.date
    else:
        date_str = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')

    print(f"\n🔄 Fetching actual game results for {date_str}...")
    df = validate_date(date_str, gamelog_scraper, all_players_list)
    if df is None:
        print(f"\nAvailable prediction files:")
        for f in sorted(
            glob.glob('predictions_assists_*.csv')
            + glob.glob('predictions_production_*.csv')
        ):
            print(f"   • {os.path.basename(f)}")
        sys.exit(1)

    print_summary(df, label=date_str)
    print_individual(df)

    out = f"validation_results_{date_str}.csv"
    df.to_csv(out, index=False)
    print("=" * 80)
    print(f"✅ Saved detailed results to: {out}")
