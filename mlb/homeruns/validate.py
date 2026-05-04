"""
⚾ MLB Home Run Prediction Validation

Validates HR predictions against actual game results.

Usage:
    python validate.py                   # Validate yesterday
    python validate.py --date 20260427   # Validate specific date
    python validate.py --cumulative      # Cumulative stats across all dates

The HR prediction is a probability (e.g. 0.135 means a 13.5% chance the
batter goes deep tonight). We evaluate with binary-outcome metrics:
  - Brier score: mean((proj - hit)^2)
  - Log loss
  - Calibration buckets (predicted % vs actual hit rate)
  - P&L simulation on rows where recommended_side == 'OVER'
"""

import sys
import os
import glob
import argparse
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import requests
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
BOXSCORE_URL = "https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"
TEAM_URL = "https://statsapi.mlb.com/api/v1/teams/{team_id}"


# ======================================================================
# Fetch actual HR per batter for a given date
# ======================================================================
def fetch_actual_hrs(date_str, quiet=False):
    """Return dict mapping (normalized_name, team_abbr) -> int HR count
    for every batter who played on the given date.

    Also returns a flat name->HR map as fallback when team abbrev mismatches.
    """
    validate_date = datetime.strptime(date_str, '%Y%m%d')
    resp = requests.get(
        SCHEDULE_URL,
        params={'sportId': 1, 'date': validate_date.strftime('%Y-%m-%d')},
        timeout=10,
    )
    if resp.status_code != 200 or not resp.json().get('dates'):
        if not quiet:
            print(f"   ⚠️  No games found on {date_str}")
        return {}, {}

    games = resp.json()['dates'][0].get('games', [])

    # Map team_id -> abbreviation
    team_cache = {}
    for game in games:
        for side in ('away', 'home'):
            tid = game['teams'][side]['team']['id']
            if tid in team_cache:
                continue
            tr = requests.get(TEAM_URL.format(team_id=tid), timeout=5)
            if tr.status_code == 200:
                team_cache[tid] = tr.json()['teams'][0]['abbreviation']

    # (name_lower, team) -> hr ; name_lower -> hr (fallback)
    keyed = {}
    flat = {}
    for game in games:
        gpk = game['gamePk']
        # Skip not-yet-final games
        status = game.get('status', {}).get('detailedState', '')
        if status not in ('Final', 'Game Over', 'Completed Early'):
            # Still try the boxscore — sometimes the status flag lags
            pass
        br = requests.get(BOXSCORE_URL.format(game_pk=gpk), timeout=10)
        if br.status_code != 200:
            continue
        box = br.json()
        for side in ('away', 'home'):
            team_id = game['teams'][side]['team']['id']
            team_abbr = team_cache.get(team_id, '')
            players = box['teams'][side].get('players', {})
            for pkey, pdata in players.items():
                full_name = pdata.get('person', {}).get('fullName', '')
                batting = pdata.get('stats', {}).get('batting', {}) or {}
                # Skip players who didn't bat
                if not batting:
                    continue
                hr = int(batting.get('homeRuns', 0) or 0)
                # Only record players who actually had a PA
                pa = int(batting.get('plateAppearances', 0) or 0)
                ab = int(batting.get('atBats', 0) or 0)
                if pa == 0 and ab == 0:
                    continue
                key = (full_name.lower(), team_abbr)
                keyed[key] = max(keyed.get(key, 0), hr)
                flat[full_name.lower()] = max(flat.get(full_name.lower(), 0), hr)
    return keyed, flat


def lookup_actual_hr(name, team, keyed, flat):
    """Best-effort lookup of HR count for a player on the validation date.
    Returns int HR (0 if found and didn't HR) or None if player not found.
    """
    name_lower = name.lower()
    key = (name_lower, team)
    if key in keyed:
        return keyed[key]
    if name_lower in flat:
        return flat[name_lower]
    # Try last-name + first initial fallback
    parts = name.split()
    if len(parts) >= 2:
        last = parts[-1].lower()
        first_initial = parts[0][0].lower()
        for full_name_lower, hr in flat.items():
            fp = full_name_lower.split()
            if len(fp) >= 2 and fp[-1] == last and fp[0][0] == first_initial:
                return hr
    return None


# ======================================================================
# Single-date validation
# ======================================================================
def validate_date(date_str, quiet=False):
    """Validate predictions for one date. Returns DataFrame or None."""
    predictions_file = f"predictions_homeruns_{date_str}.csv"
    if not os.path.exists(predictions_file):
        if not quiet:
            print(f"❌ File not found: {predictions_file}")
        return None

    pred_df = pd.read_csv(predictions_file)
    if not quiet:
        print(f"\n📅 Validating: {date_str} ({len(pred_df)} batters)")

    keyed, flat = fetch_actual_hrs(date_str, quiet=quiet)
    if not keyed and not flat:
        return None

    rows = []
    not_found = []
    for _, p in pred_df.iterrows():
        name = p['player_name']
        team = p['team']
        hr = lookup_actual_hr(name, team, keyed, flat)
        if hr is None:
            not_found.append(name)
            continue

        hit = 1 if hr >= 1 else 0
        proj = float(p['projection']) if pd.notna(p['projection']) else 0.0
        result = {
            'date': date_str,
            'player_name': name,
            'team': team,
            'opponent': p.get('opponent', ''),
            'is_home': p.get('is_home', 0),
            'projection': proj,
            'actual_hr': int(hr),
            'hit': hit,
            'confidence': p.get('confidence', ''),
            'red_flags': p.get('red_flags', 'None'),
            # Book / edge fields (may be NaN if no odds matched)
            'book_odds': p.get('book_odds'),
            'book_prob': p.get('book_prob'),
            'edge_pp': p.get('edge_pp'),
            'ev': p.get('ev'),
            'kelly_units': p.get('kelly_units'),
            'recommended_side': p.get('recommended_side', ''),
            'bookmaker': p.get('bookmaker', ''),
            'pitcher_name': p.get('pitcher_name', ''),
            'pitcher_hand': p.get('pitcher_hand', ''),
            'park_factor': p.get('park_factor', 1.0),
        }
        rows.append(result)

    if not rows:
        if not quiet:
            print("   ⚠️  No batter results matched")
        return None

    df = pd.DataFrame(rows)

    if not quiet and not_found:
        print(f"   ⚠️  {len(not_found)} batters not found "
              f"(likely benched / DNP): {', '.join(not_found[:5])}"
              f"{'...' if len(not_found) > 5 else ''}")
        # Quick visual: top hits
        hits = df[df['hit'] == 1].sort_values('projection', ascending=False)
        if not hits.empty:
            print(f"\n   💣 {len(hits)} batters homered:")
            for _, r in hits.head(15).iterrows():
                print(f"      ✅ {r['player_name']:25s} ({r['team']}) "
                      f"proj {r['projection']:.3f} → {r['actual_hr']} HR")

    return df


# ======================================================================
# Display helpers
# ======================================================================
def print_summary(df, label=""):
    print("=" * 80)
    print(f"📊 HR VALIDATION{' — ' + label if label else ''}")
    print("=" * 80)

    n = len(df)
    base_rate = df['hit'].mean()           # actual P(HR) in this sample
    avg_proj = df['projection'].mean()
    bias = avg_proj - base_rate
    brier = ((df['projection'] - df['hit']) ** 2).mean()
    # Log loss with safe clipping
    p = df['projection'].clip(1e-4, 1 - 1e-4)
    log_loss = -(df['hit'] * np.log(p) + (1 - df['hit']) * np.log(1 - p)).mean()
    # Compare against the constant "always predict base rate" baseline
    baseline_brier = base_rate * (1 - base_rate)

    print(f"\n🎯 Overall ({n} batters):")
    print(f"   Actual HR rate:    {base_rate:.1%}  ({df['hit'].sum()}/{n})")
    print(f"   Avg projection:    {avg_proj:.1%}")
    print(f"   Calibration bias:  {bias:+.2%}  "
          f"({'over' if bias > 0 else 'under'}-projecting)")
    print(f"   Brier score:       {brier:.4f}  (baseline {baseline_brier:.4f})")
    print(f"   Log loss:          {log_loss:.4f}")

    # Calibration by projection bucket
    print(f"\n📐 Calibration by projection bucket:")
    print(f"   {'bucket':<13} {'n':>4}  {'avg proj':>9}  {'actual':>8}  {'gap':>7}")
    edges = [(0.00, 0.05), (0.05, 0.08), (0.08, 0.12),
             (0.12, 0.16), (0.16, 0.20), (0.20, 1.00)]
    for lo, hi in edges:
        sub = df[(df['projection'] >= lo) & (df['projection'] < hi)]
        if sub.empty:
            continue
        a_proj = sub['projection'].mean()
        actual = sub['hit'].mean()
        gap = actual - a_proj
        icon = "✅" if abs(gap) < 0.03 else ("📈" if gap > 0 else "📉")
        print(f"   {lo:.2f}-{hi:.2f}    {len(sub):>4}  "
              f"{a_proj:>9.1%}  {actual:>8.1%}  {gap:>+7.1%} {icon}")

    # By confidence tier
    if df['confidence'].notna().any():
        print(f"\n🎯 By confidence:")
        print(f"   {'tier':<7} {'n':>4}  {'avg proj':>9}  {'actual':>8}  {'brier':>7}")
        for tier in ['HIGH', 'MEDIUM', 'LOW']:
            sub = df[df['confidence'] == tier]
            if sub.empty:
                continue
            br = ((sub['projection'] - sub['hit']) ** 2).mean()
            print(f"   {tier:<7} {len(sub):>4}  "
                  f"{sub['projection'].mean():>9.1%}  "
                  f"{sub['hit'].mean():>8.1%}  {br:>7.4f}")

    # ----------------------------------------------------------------
    # P&L simulation on recommended OVER plays (HR YES @ book odds)
    # ----------------------------------------------------------------
    bets = df[(df['recommended_side'] == 'OVER') & df['book_odds'].notna()].copy()
    if not bets.empty:
        bets['stake'] = bets['kelly_units'].fillna(1.0).clip(lower=0)
        # Treat 0-stake "PASS"-equivalents as 1u flat fallback
        bets.loc[bets['stake'] == 0, 'stake'] = 1.0

        # Decimal odds from American
        def _to_decimal(odds):
            if pd.isna(odds):
                return None
            o = float(odds)
            return 1 + (o / 100) if o > 0 else 1 + (100 / abs(o))

        bets['decimal_odds'] = bets['book_odds'].apply(_to_decimal)
        bets['payout'] = np.where(
            bets['hit'] == 1,
            bets['stake'] * (bets['decimal_odds'] - 1),  # win
            -bets['stake'],                              # loss
        )
        net_units = bets['payout'].sum()
        total_stake = bets['stake'].sum()
        roi = (net_units / total_stake) if total_stake > 0 else 0
        hits = int(bets['hit'].sum())
        icon = "✅" if net_units >= 0 else "❌"
        print(f"\n💰 P&L on recommended OVER plays "
              f"(stake = kelly_units, fallback 1u):")
        print(f"   Bets: {len(bets)}  Hits: {hits}  Hit rate: {hits/len(bets):.1%}")
        print(f"   Total staked: {total_stake:.2f}u  Net: {net_units:+.2f}u  "
              f"ROI: {roi:+.1%} {icon}")

        # Flat 1u variant for easier comparison
        flat_payout = np.where(
            bets['hit'] == 1, bets['decimal_odds'] - 1, -1.0,
        )
        flat_net = flat_payout.sum()
        print(f"   Flat 1u/bet: Net {flat_net:+.2f}u  "
              f"ROI {flat_net/len(bets):+.1%}")

    # ----------------------------------------------------------------
    # Threshold sweep on book_prob edge — what would various filters yield?
    # ----------------------------------------------------------------
    edged = df[df['book_odds'].notna() & df['projection'].notna()].copy()
    if not edged.empty:
        # Reconstruct decimal payout for *all* candidate bets, not just
        # the ones the predictor flagged.
        def _to_decimal(odds):
            if pd.isna(odds):
                return None
            o = float(odds)
            return 1 + (o / 100) if o > 0 else 1 + (100 / abs(o))
        edged['decimal_odds'] = edged['book_odds'].apply(_to_decimal)

        print(f"\n📈 Threshold sweep "
              f"(every batter with odds, flat 1u, +EV based on raw projection):")
        print(f"   {'min EV':>8} {'bets':>5} {'hit%':>6} {'net':>9} {'ROI':>8}")
        for min_ev in [0.00, 0.05, 0.10, 0.15, 0.20, 0.30]:
            # EV per 1u: proj * (decimal-1) - (1-proj) * 1
            ev_arr = (
                edged['projection'] * (edged['decimal_odds'] - 1)
                - (1 - edged['projection']) * 1.0
            )
            keep = edged[ev_arr >= min_ev]
            if keep.empty:
                continue
            payout = np.where(keep['hit'] == 1,
                              keep['decimal_odds'] - 1, -1.0)
            net = payout.sum()
            print(f"   {min_ev:>+8.2f} {len(keep):>5d} "
                  f"{keep['hit'].mean():>6.1%} "
                  f"{net:>+9.2f}u {net/len(keep):>+8.1%}")

    print()


def print_individual(df, max_rows=20):
    print("=" * 80)
    print("🎯 TOP HIT/MISS RESULTS")
    print("=" * 80)
    # Show biggest hits first (highest projection that DID HR), then the worst misses
    hits = df[df['hit'] == 1].sort_values('projection', ascending=False)
    misses = df[(df['hit'] == 0) & (df['recommended_side'] == 'OVER')] \
        .sort_values('projection', ascending=False)

    if not hits.empty:
        print("\n💣 Confirmed HRs (highest projection first):")
        for _, r in hits.head(max_rows).iterrows():
            side = r.get('recommended_side', '')
            tag = "📈 PICK" if side == 'OVER' else "       "
            odds = (
                f"{int(r['book_odds']):+d} ({r.get('bookmaker', '')})"
                if pd.notna(r.get('book_odds')) else 'no odds'
            )
            print(f"   ✅ {tag} {r['player_name']:25s} "
                  f"proj {r['projection']:.3f} | {odds}")

    if not misses.empty:
        print("\n❌ Recommended OVER plays that missed:")
        for _, r in misses.head(max_rows).iterrows():
            odds = (
                f"{int(r['book_odds']):+d}"
                if pd.notna(r.get('book_odds')) else ''
            )
            print(f"   ❌ {r['player_name']:25s} "
                  f"proj {r['projection']:.3f} | {odds}")
    print()


# ======================================================================
# Main
# ======================================================================
parser = argparse.ArgumentParser(description="Validate MLB HR predictions")
parser.add_argument('--date', type=str, default=None,
                    help='Date to validate (YYYYMMDD)')
parser.add_argument('--cumulative', action='store_true',
                    help='Cumulative stats across all dates')
args = parser.parse_args()

print("=" * 80)
print("⚾ MLB HOME RUN PREDICTION VALIDATION")
print("=" * 80)

if args.cumulative:
    pred_files = sorted(glob.glob('predictions_homeruns_*.csv'))
    # Exclude _rerun files — they may contain live/in-game odds
    pred_files = [f for f in pred_files if '_rerun' not in f]
    if not pred_files:
        print("\n❌ No prediction files found")
        sys.exit(1)

    all_results = []
    for pf in pred_files:
        ds = (
            os.path.basename(pf)
            .replace('predictions_homeruns_', '')
            .replace('.csv', '')
        )
        # Reuse cached validation if present
        val_file = f"validation_results_{ds}.csv"
        if os.path.exists(val_file):
            try:
                cached = pd.read_csv(val_file)
                all_results.append(cached)
                print(f"   ↪ Reused cached validation for {ds} "
                      f"({len(cached)} rows)")
                continue
            except Exception:
                pass
        df = validate_date(ds, quiet=True)
        if df is None:
            continue
        df.to_csv(val_file, index=False)
        all_results.append(df)
        print(f"   ✓ Validated {ds} ({len(df)} batters)")

    if not all_results:
        print("\n❌ Nothing to summarize")
        sys.exit(1)

    combined = pd.concat(all_results, ignore_index=True)
    print_summary(combined, label="CUMULATIVE")
    print(f"\n✅ {len(combined)} batter-days across "
          f"{combined['date'].nunique()} dates")
else:
    if args.date:
        date_str = args.date
    else:
        # Default to yesterday
        date_str = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')

    df = validate_date(date_str)
    if df is None:
        print(f"\nAvailable prediction files:")
        for f in sorted(glob.glob('predictions_homeruns_*.csv')):
            print(f"   • {os.path.basename(f)}")
        sys.exit(1)

    print_summary(df, label=date_str)
    print_individual(df)

    out = f"validation_results_{date_str}.csv"
    df.to_csv(out, index=False)
    print(f"✅ Saved to: {out}")
