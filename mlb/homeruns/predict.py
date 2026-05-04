"""
⚾ MLB HOME RUN PREDICTIONS
Identify hitters with best edge on hitting home runs today
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os
import warnings
from dotenv import load_dotenv
import unicodedata
import requests

# Suppress warnings
warnings.filterwarnings('ignore')

# Empirically reasonable league constants for normalization.
# Sources: 2024-2025 MLB season aggregates.
LEAGUE_AVG_HR9 = 1.20      # HR per 9 IP
LEAGUE_AVG_HR_PA = 0.030   # HR per plate appearance
LEAGUE_AVG_ISO = 0.150     # ISO (SLG - AVG)

# Handedness HR boost: same-side advantage. e.g. LHB vs RHP gains ~7% HR rate.
# Cross-handed batter facing same-handed pitcher loses ~5%.
HANDEDNESS_HR_FACTOR = {
    ('L', 'R'): 1.07,  # LHB vs RHP
    ('R', 'L'): 1.07,  # RHB vs LHP
    ('L', 'L'): 0.93,  # LHB vs LHP (tougher)
    ('R', 'R'): 0.97,  # RHB vs RHP (slight)
    ('S', 'R'): 1.03,  # Switch vs RHP
    ('S', 'L'): 1.03,  # Switch vs LHP
}

# Bayesian shrinkage "prior weights" — think of it as: "how many league-average
# observations is the prior worth?". Larger prior = more shrinkage of small
# samples toward the league mean.
#   - 150 PA: a batter with 75 PA is weighted 50/50 with league prior
#   - 30 IP : a pitcher with 30 IP is 50/50 weighted (≈5 starts of 6 IP each)
BATTER_PA_PRIOR = 150.0
PITCHER_IP_PRIOR = 30.0


def shrink_rate(observed_rate, observed_count, prior_rate, prior_count):
    """Empirical-Bayes shrinkage of a per-trial rate toward a prior.
    
    Returns: (prior_count * prior_rate + observed_count * observed_rate) /
             (prior_count + observed_count)
    
    Equivalent to a Beta-Binomial posterior mean when prior is Beta(α, β)
    with α = prior_rate * prior_count and β = (1-prior_rate) * prior_count.
    """
    if observed_count <= 0:
        return prior_rate
    return (
        (prior_count * prior_rate + observed_count * observed_rate)
        / (prior_count + observed_count)
    )


def iso_implied_hr_pa(iso, league_iso=LEAGUE_AVG_ISO,
                     league_hr_pa=LEAGUE_AVG_HR_PA):
    """Estimate HR/PA from ISO using the rule of thumb that HR/PA scales
    roughly linearly with ISO around the league average.
    
    A hitter with 0.250 ISO (well above league avg 0.150) is ~67% above
    league HR rate → 5.0% HR/PA.
    """
    if iso is None or pd.isna(iso) or iso <= 0:
        return league_hr_pa
    return league_hr_pa * (iso / league_iso)


# Helper function to fetch batter game logs
def fetch_batter_game_logs(player_id, season=2026, limit=10):
    """Fetch recent game logs for a batter"""
    url = f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats"
    params = {
        'stats': 'gameLog',
        'group': 'hitting',
        'season': season,
        'limit': limit
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        games = []
        for stat_group in data.get('stats', []):
            for split in stat_group.get('splits', []):
                stat = split.get('stat', {})
                games.append({
                    'date': split.get('date'),
                    'HR': stat.get('homeRuns', 0),
                    'PA': stat.get('plateAppearances', 0),
                    'AB': stat.get('atBats', 0),
                    'H': stat.get('hits', 0)
                })
        
        return pd.DataFrame(games)
    except Exception as e:
        return pd.DataFrame()

load_dotenv()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from core.name_utils import normalize_name, filter_by_name
from mlb.shared.scrapers.mlb_schedule import MLBScheduleScraper
from mlb.shared.scrapers.batter_stats import BatterStatsScraper
from mlb.shared.scrapers.pitcher_stats import PitcherStatsScraper
from mlb.shared.scrapers.mlb_lineups import MLBLineupScraper
from mlb.shared.scrapers.rotochamp_lineups import RotoChampLineupScraper
from mlb.shared.scrapers.baseball_savant import BaseballSavantScraper
from mlb.shared.scrapers.odds_api import (
    OddsAPIScraper,
    calculate_implied_probability,
    calculate_expected_value,
)
from mlb.shared.features.park_factors import get_park_hr_factor

print("=" * 80)
print("⚾ MLB HOME RUN PREDICTIONS")
print("=" * 80)

# Initialize scrapers
schedule_scraper = MLBScheduleScraper()
batter_scraper = BatterStatsScraper()
pitcher_scraper = PitcherStatsScraper()
lineup_scraper = MLBLineupScraper()
roto_lineup_scraper = RotoChampLineupScraper()
savant_scraper = BaseballSavantScraper()
odds_scraper = OddsAPIScraper()

# Get target date
target_date = sys.argv[1] if len(sys.argv) > 1 else None

if target_date:
    print(f"\n📅 Fetching games for {target_date}...")
else:
    print("\n📅 Fetching today's games...")

games_df = schedule_scraper.get_todays_games(date=target_date)

if games_df.empty:
    print("   ❌ No games found")
    sys.exit(1)

print(f"   ✓ Found {len(games_df)} games today")

# Fetch home run odds
print("\n📡 Fetching home run odds...")
hr_odds_df = odds_scraper.get_all_home_run_odds()
if not hr_odds_df.empty:
    print(f"   ✓ Found odds for {len(hr_odds_df)} batter/line combinations")
else:
    print("   ⚠️  No home run odds found")

# Collect all starting lineups
print("\n📊 Fetching starting lineups...")
all_batters = []

for _, game in games_df.iterrows():
    game_id = game.get('game_id')
    
    # Try to get actual lineups first
    for team_type in ['home', 'away']:
        team = game[f'{team_type}_team'] if f'{team_type}_team' in game else (game['home_team'] if team_type == 'home' else game['away_team'])
        opponent = game['away_team'] if team_type == 'home' else game['home_team']
        
        lineup = None
        if game_id:
            lineup = lineup_scraper.get_lineup_for_team(game_id, team_type=team_type)
        
        # Fallback to projected lineup
        if not lineup or len(lineup) < 8:
            lineup = roto_lineup_scraper.get_projected_lineup(team)
        
        if lineup and len(lineup) >= 8:
            for batter in lineup:
                # Lineup scrapers return list of strings (player names)
                if isinstance(batter, str):
                    all_batters.append({
                        'player_name': batter,
                        'player_id': None,
                        'team': team,
                        'opponent': opponent,
                        'is_home': 1 if team_type == 'home' else 0,
                        'game_id': game_id
                    })
                elif isinstance(batter, dict):
                    all_batters.append({
                        'player_name': batter.get('name', batter.get('player_name', 'Unknown')),
                        'player_id': batter.get('player_id'),
                        'team': team,
                        'opponent': opponent,
                        'is_home': 1 if team_type == 'home' else 0,
                        'game_id': game_id
                    })

print(f"   ✓ Found {len(all_batters)} batters in starting lineups")

# Remove duplicates (keep first occurrence)
seen = set()
unique_batters = []
for b in all_batters:
    key = b['player_name']
    if key and key not in seen:
        seen.add(key)
        unique_batters.append(b)

all_batters = unique_batters
print(f"   ✓ {len(all_batters)} unique batters after deduping")

# Helper function to fetch batter season stats from MLB API
def fetch_batter_season_stats(season, min_pa=50):
    """Fetch season batting stats for all qualified batters"""
    url = "https://statsapi.mlb.com/api/v1/stats"
    params = {
        'stats': 'season',
        'group': 'hitting',
        'season': season,
        'sportId': 1,
        'limit': 500
    }
    
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        batters = []
        for stat_group in data.get('stats', []):
            for split in stat_group.get('splits', []):
                player = split.get('player', {})
                stats = split.get('stat', {})
                
                pa = int(stats.get('plateAppearances', 0) or 0)
                if pa >= min_pa:
                    avg = float(stats.get('avg', 0) or 0)
                    slg = float(stats.get('slg', 0) or 0)
                    # batSide is on the player object (R/L/S)
                    bat_side = (player.get('batSide') or {}).get('code', 'R')
                    batters.append({
                        'player_id': player.get('id'),
                        'player_name': player.get('fullName'),
                        'bats': bat_side,
                        'HR': int(stats.get('homeRuns', 0) or 0),
                        'AB': int(stats.get('atBats', 0) or 0),
                        'PA': pa,
                        'AVG': avg,
                        'SLG': slg,
                        'ISO': slg - avg
                    })
        
        return pd.DataFrame(batters)
    except Exception as e:
        print(f"   ⚠️  Error fetching {season} stats: {e}")
        return pd.DataFrame()

# Fetch batter stats
print("\n📊 Fetching batter stats...")
stats_2025 = fetch_batter_season_stats(2025, min_pa=50)
stats_2026 = fetch_batter_season_stats(2026, min_pa=10)

print(f"   ✓ Loaded {len(stats_2025)} batters from 2025")
print(f"   ✓ Loaded {len(stats_2026)} batters from 2026")

# Fetch pitcher HR rates (HR/9)
print("\n📊 Fetching pitcher home run rates...")
pitcher_stats_2025 = pitcher_scraper.get_season_stats(season=2025, min_starts=10)
pitcher_stats_2026 = pitcher_scraper.get_season_stats(season=2026, min_starts=2)

# Combine pitcher stats from both seasons (prefer 2026, fallback 2025).
# get_season_stats() now provides 'HR9' (HR per 9 IP) and 'hand' columns.
all_pitcher_stats = (
    pd.concat([pitcher_stats_2025, pitcher_stats_2026])
    .drop_duplicates('pitcher_id', keep='last')
    .reset_index(drop=True)
)
# Lookup tables: by id and by normalized name
pitcher_hr9_by_id = {}
pitcher_hand_by_id = {}
pitcher_lookup_by_name = {}
for _, p in all_pitcher_stats.iterrows():
    pid = p.get('pitcher_id')
    hr9 = float(p.get('HR9', 0) or 0)
    if pid is None:
        continue
    if hr9 > 0:
        pitcher_hr9_by_id[pid] = hr9
    pitcher_hand_by_id[pid] = p.get('hand', 'R')
    name = p.get('pitcher_name')
    if name:
        pitcher_lookup_by_name[normalize_name(name)] = {
            'id': pid,
            'hr9': hr9 if hr9 > 0 else LEAGUE_AVG_HR9,
            'hand': p.get('hand', 'R'),
        }

print(
    f"   ✓ Loaded HR/9 for {len(pitcher_hr9_by_id)} pitchers "
    f"({len(all_pitcher_stats)} total in stats)"
)


def lookup_pitcher_info(pitcher_id, pitcher_name):
    """Resolve opposing pitcher's HR/9 and hand.

    Tries pitcher_id first, then full-name match, then last-name match.
    Returns (hr9, hand) where hr9 falls back to LEAGUE_AVG_HR9 and hand to 'R'.
    """
    if pitcher_id and pitcher_id in pitcher_hr9_by_id:
        return (
            pitcher_hr9_by_id[pitcher_id],
            pitcher_hand_by_id.get(pitcher_id, 'R'),
        )
    if pitcher_name:
        norm = normalize_name(pitcher_name)
        # Exact full-name match
        if norm in pitcher_lookup_by_name:
            info = pitcher_lookup_by_name[norm]
            return info['hr9'], info['hand']
        # Last-name fallback (handles 'M. Perez' etc.)
        last = norm.split()[-1] if norm else ''
        if last:
            for key, info in pitcher_lookup_by_name.items():
                if last in key.split():
                    return info['hr9'], info['hand']
    return LEAGUE_AVG_HR9, 'R'

# GENERATE PREDICTIONS
predictions = []

# Load HR probability calibrator (isotonic regression trained on validation data).
# Maps raw P(1+ HR) to empirically calibrated probability that corrects the
# S-curve under-projection at extremes (<5% and >20%).
try:
    from prob_calibrator import ProbabilityCalibrator as HRCalibrator
    hr_calibrator = HRCalibrator()
    if not hr_calibrator.load():
        hr_calibrator = None
except Exception:
    hr_calibrator = None

print("\n🎯 Generating predictions...")
print("=" * 80)

for batter in all_batters:
    player_name = batter['player_name']
    
    try:
        # Look up 2026 + 2025 stats for the batter. We pool both seasons so
        # small 2026 samples get reinforced by last year's data instead of
        # being thrown away.
        stats_26 = stats_2026[stats_2026['player_name'].str.contains(player_name, case=False, na=False)]
        stats_25 = stats_2025[stats_2025['player_name'].str.contains(player_name, case=False, na=False)]
        
        if stats_26.empty and stats_25.empty:
            continue
        
        # Pool counts across both years. 2026 weighted ~2x because it's the
        # current season and reflects current power level.
        hr_26 = int(stats_26.iloc[0].get('HR', 0)) if not stats_26.empty else 0
        pa_26 = int(stats_26.iloc[0].get('PA', 0)) if not stats_26.empty else 0
        ab_26 = int(stats_26.iloc[0].get('AB', 0)) if not stats_26.empty else 0
        iso_26 = float(stats_26.iloc[0].get('ISO', 0) or 0) if not stats_26.empty else 0
        
        hr_25 = int(stats_25.iloc[0].get('HR', 0)) if not stats_25.empty else 0
        pa_25 = int(stats_25.iloc[0].get('PA', 0)) if not stats_25.empty else 0
        iso_25 = float(stats_25.iloc[0].get('ISO', 0) or 0) if not stats_25.empty else 0
        
        # Season stats source: prefer 2026 (current year context: bats hand,
        # player_id, etc.). Fall back to 2025 if no 2026 line.
        season_stats = stats_26.iloc[0] if not stats_26.empty else stats_25.iloc[0]
        player_id = season_stats.get('player_id')
        weight_2026 = 1.0 if not stats_26.empty else 0.0
        
        # Pooled HR rate: 2x weight on 2026 PA (current year is more recent),
        # 1x on 2025. Equivalent to: HR_total / weighted_PA_total.
        weighted_hr = 2.0 * hr_26 + 1.0 * hr_25
        weighted_pa = 2.0 * pa_26 + 1.0 * pa_25
        raw_hr_per_pa = weighted_hr / weighted_pa if weighted_pa > 0 else LEAGUE_AVG_HR_PA
        
        # Pooled ISO: same 2x/1x weighting.
        weighted_iso_num = 2.0 * iso_26 * pa_26 + 1.0 * iso_25 * pa_25
        if weighted_pa > 0:
            iso = weighted_iso_num / weighted_pa
        else:
            iso = iso_26 if iso_26 > 0 else iso_25
        
        # Total observed PA (un-weighted, used as the "n" for shrinkage)
        pa = pa_26 + pa_25
        ab = ab_26  # Used only for hr_per_ab below; OK to be 2026-only
        hr = hr_26 + hr_25  # Total HR for display
        hr_per_ab = hr / ab if ab > 0 else 0
        
        # ISO-implied HR rate: regresses small-sample HR/PA toward a power
        # signal that's more stable than HR count itself.
        iso_hr_pa = iso_implied_hr_pa(iso)
        # 60/40 blend of raw HR/PA and ISO-implied gives weight to both
        # the actual HR luck and the underlying power.
        blended_season_hr_pa = 0.6 * raw_hr_per_pa + 0.4 * iso_hr_pa
        
        # Bayesian shrinkage of season HR rate toward league mean.
        # Effect: a 30-PA hitter is mostly league-avg; a 600-PA hitter trusts
        # their own number.
        hr_per_pa = shrink_rate(
            blended_season_hr_pa, pa, LEAGUE_AVG_HR_PA, BATTER_PA_PRIOR
        )
        
        # Get recent game logs for trend (only if we have player_id)
        recent_hr_rate = hr_per_pa
        recent_pa_count = 0
        game_logs = pd.DataFrame()
        if player_id:
            game_logs = fetch_batter_game_logs(player_id, season=2026)
            if not game_logs.empty and len(game_logs) >= 5:
                recent_games = game_logs.head(10)
                recent_hr = recent_games['HR'].sum() if 'HR' in recent_games.columns else 0
                recent_pa_count = (
                    int(recent_games['PA'].sum())
                    if 'PA' in recent_games.columns
                    else len(recent_games) * 4
                )
                raw_recent_rate = (
                    recent_hr / recent_pa_count if recent_pa_count > 0 else hr_per_pa
                )
                # Shrink recent rate toward season rate (not league avg) since
                # season already encodes the player's true power level.
                recent_hr_rate = shrink_rate(
                    raw_recent_rate, recent_pa_count,
                    hr_per_pa, BATTER_PA_PRIOR / 2,
                )
        
        # Resolve opposing starting pitcher
        opponent_team = batter['opponent']
        opposing_pitcher_id = None
        opposing_pitcher_name = None
        for _, game in games_df.iterrows():
            if game.get('home_team') == opponent_team:
                opposing_pitcher_id = game.get('home_pitcher_id')
                opposing_pitcher_name = game.get('home_pitcher')
                break
            elif game.get('away_team') == opponent_team:
                opposing_pitcher_id = game.get('away_pitcher_id')
                opposing_pitcher_name = game.get('away_pitcher')
                break
        
        pitcher_hr9_raw, pitcher_hand = lookup_pitcher_info(
            opposing_pitcher_id, opposing_pitcher_name
        )
        # Shrink pitcher HR/9 toward league average. We don't have IP here
        # for that pitcher specifically, so estimate IP from games started
        # in the all_pitcher_stats table by matching id/name.
        pitcher_ip = 0.0
        if opposing_pitcher_id is not None:
            row = all_pitcher_stats[
                all_pitcher_stats['pitcher_id'] == opposing_pitcher_id
            ]
            if not row.empty:
                pitcher_ip = float(row.iloc[0].get('IP', 0) or 0)
        if pitcher_ip == 0 and opposing_pitcher_name:
            norm = normalize_name(opposing_pitcher_name)
            for _, prow in all_pitcher_stats.iterrows():
                if normalize_name(prow.get('pitcher_name', '')) == norm:
                    pitcher_ip = float(prow.get('IP', 0) or 0)
                    break
        pitcher_hr9 = shrink_rate(
            pitcher_hr9_raw, pitcher_ip, LEAGUE_AVG_HR9, PITCHER_IP_PRIOR
        )
        
        # Park factor: HOME team's stadium is in play.
        park_team = batter['team'] if batter['is_home'] else opponent_team
        park_factor = get_park_hr_factor(park_team)
        
        # Handedness factor (LHB vs RHP gets a small boost, etc.)
        bats = season_stats.get('bats', 'R') if isinstance(season_stats, pd.Series) else season_stats.get('bats', 'R')
        # If batter is switch ('S'), code lookup handles both vs L and vs R.
        hand_factor = HANDEDNESS_HR_FACTOR.get(
            (bats, pitcher_hand), 1.0
        )
        
        # Calculate expected plate appearances
        expected_pa = 4.2 if batter['is_home'] else 4.0  # Home teams get slightly more PA
        
        # Blend rates (season vs recent)
        blended_hr_rate = (hr_per_pa * 0.6) + (recent_hr_rate * 0.4)
        
        # Adjust for pitcher HR rate. Now that pitcher_hr9 is shrunk toward
        # the league mean, we can use a tighter cap (0.8-1.3) without losing
        # signal on real outliers.
        pitcher_factor = pitcher_hr9 / LEAGUE_AVG_HR9 if pitcher_hr9 > 0 else 1.0
        pitcher_factor = min(max(pitcher_factor, 0.8), 1.3)
        
        # Stack adjustments: pitcher × park × handedness
        adjusted_hr_rate = (
            blended_hr_rate * pitcher_factor * park_factor * hand_factor
        )
        # HR/PA cannot meaningfully exceed ~15% even for elite power hitters
        # in extreme parks; clamp to keep the per-PA math sane.
        adjusted_hr_rate = min(max(adjusted_hr_rate, 0.0), 0.15)
        park_adjusted_rate = adjusted_hr_rate  # name kept for clarity below
        
        # Book market is "1+ HR" (Over 0.5). So we need P(at least one HR over
        # `expected_pa` plate appearances), NOT expected HR count. Treating
        # each PA as independent Bernoulli(park_adjusted_rate):
        #     P(1+ HR) = 1 - (1 - p)^PA
        hr_projection = 1.0 - (1.0 - park_adjusted_rate) ** expected_pa
        # Realistic ceiling: even Aaron Judge at Coors maxes around ~40-45%.
        hr_projection = min(hr_projection, 0.50)

        # Apply isotonic calibration if available
        raw_hr_projection = hr_projection
        if hr_calibrator is not None and hr_calibrator.is_fitted:
            hr_projection = hr_calibrator.calibrate(hr_projection)
            hr_projection = float(hr_projection)
        
        # Confidence assessment
        red_flags = []
        
        # Low sample size
        if pa < 100:
            red_flags.append(f'Small sample ({pa} PA)')
        
        # Cold streak (no HR in last 10 games)
        if not game_logs.empty and len(game_logs) >= 10:
            last_10_hr = game_logs.head(10)['HR'].sum() if 'HR' in game_logs.columns else 0
            if last_10_hr == 0:
                red_flags.append('No HR in last 10 games')
        
        # Confidence tier
        confidence = 'HIGH'
        if len(red_flags) >= 2 or pa < 50:
            confidence = 'LOW'
        elif len(red_flags) == 1 or hr < 3:
            confidence = 'MEDIUM'
        
        # Store prediction
        pred = {
            'player_name': player_name,
            'team': batter['team'],
            'opponent': batter['opponent'],
            'is_home': batter['is_home'],
            'bats': bats,
            'projection': round(hr_projection, 3),
            'hr': hr,
            'pa': pa,
            'iso': round(float(iso or 0), 3),
            'raw_hr_per_pa': round(raw_hr_per_pa, 4),
            'hr_per_pa': round(hr_per_pa, 4),
            'recent_hr_rate': round(recent_hr_rate, 4),
            'pitcher_name': opposing_pitcher_name or '',
            'pitcher_hand': pitcher_hand,
            'pitcher_hr9_raw': round(pitcher_hr9_raw, 2),
            'pitcher_hr9': round(pitcher_hr9, 2),
            'park_factor': round(park_factor, 2),
            'hand_factor': round(hand_factor, 2),
            'confidence': confidence,
            'red_flags': '; '.join(red_flags) if red_flags else 'None',
            'book_implied': 'N/A',
            'model_vs_book': 0,
        }
        
        predictions.append(pred)
        
    except Exception as e:
        continue

# SAVE AND DISPLAY
if predictions:
    df = pd.DataFrame(predictions)

    # ------------------------------------------------------------------
    # EDGE ANALYSIS: match book odds, compute book implied prob, edge, EV.
    # ------------------------------------------------------------------
    df['book_odds'] = float('nan')
    df['book_prob'] = float('nan')
    df['edge_pp'] = 0.0
    df['ev'] = 0.0
    df['kelly_units'] = 0.0
    df['bookmaker'] = ''
    df['recommended_side'] = ''

    # ¼-Kelly sizing — capped at 1.0u per HR bet (lower than K's because the
    # variance is much higher: a single HR coin flip vs continuous K total).
    HR_KELLY_FRACTION = 0.25
    HR_MAX_KELLY = 1.0

    def hr_kelly_units(our_p, american_odds):
        if american_odds < 0:
            decimal = 1 + (100 / abs(american_odds))
        else:
            decimal = 1 + (american_odds / 100)
        b = decimal - 1
        if b <= 0:
            return 0.0
        kelly = (our_p * b - (1 - our_p)) / b
        if kelly <= 0:
            return 0.0
        units = kelly * HR_KELLY_FRACTION * 10
        return round(min(units, HR_MAX_KELLY), 2)

    have_hr_odds = (
        not hr_odds_df.empty
        and 'batter' in hr_odds_df.columns
        and 'odds' in hr_odds_df.columns
    )

    for idx, row in df.iterrows():
        if not have_hr_odds:
            break
        batter_odds = filter_by_name(hr_odds_df, 'batter', row['player_name'])
        if batter_odds.empty:
            last = row['player_name'].split()[-1]
            batter_odds = hr_odds_df[
                hr_odds_df['batter'].str.contains(last, case=False, na=False)
            ]
        if batter_odds.empty:
            continue

        # Pick best Yes odds: longest +EV opportunity = highest payout for Yes
        # bets, but use abs().idxmin() to land on the most-traded line.
        best_odds = batter_odds.loc[batter_odds['odds'].abs().idxmin()]
        odds_val = int(best_odds['odds'])
        book_prob = calculate_implied_probability(odds_val)
        our_prob = float(row['projection'])
        ev_yes = calculate_expected_value(our_prob, odds_val)

        df.at[idx, 'book_implied'] = f"Yes @ {odds_val:+d} ({best_odds['bookmaker']})"
        df.at[idx, 'book_odds'] = odds_val
        df.at[idx, 'book_prob'] = round(book_prob, 3)
        df.at[idx, 'bookmaker'] = best_odds['bookmaker']
        df.at[idx, 'edge_pp'] = round((our_prob - book_prob) * 100, 1)
        df.at[idx, 'ev'] = round(ev_yes, 3)
        df.at[idx, 'kelly_units'] = hr_kelly_units(our_prob, odds_val)

        # HR Yes-only book (no "No" side typically priced), so we only
        # recommend YES with edge or PASS otherwise. Threshold tighter than K's
        # because HR markets have wider spreads.
        if ev_yes >= 0.10 and (our_prob - book_prob) >= 0.04:
            df.at[idx, 'recommended_side'] = 'YES'
        else:
            df.at[idx, 'recommended_side'] = 'PASS'

        # model_vs_book stays as raw probability delta for backward compat
        df.at[idx, 'model_vs_book'] = round(our_prob - book_prob, 3)

    # ------------------------------------------------------------------
    # Save full CSV (sort by EV)
    # ------------------------------------------------------------------
    confidence_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
    df['confidence_rank'] = df['confidence'].map(confidence_order)
    df = df.sort_values(['ev', 'confidence_rank'], ascending=[False, True])

    if target_date:
        date_str = target_date.replace('-', '')
    else:
        date_str = datetime.now().strftime('%Y%m%d')
    filename = f"predictions_homeruns_{date_str}.csv"
    df_output = df.drop(columns=['confidence_rank'])

    # Guard: if the pre-game predictions file already exists, don't overwrite
    # it — the original has the correct pre-game odds. Re-runs (potentially
    # during live games) save to a separate file so odds integrity is preserved.
    if os.path.exists(filename):
        rerun_file = f"predictions_homeruns_{date_str}_rerun.csv"
        df_output.to_csv(rerun_file, index=False)
        print(f"\n   ⚠️  Pre-game file already exists: {filename}")
        print(f"   📁 Re-run saved to: {rerun_file} (odds may reflect live lines)")
    else:
        df_output.to_csv(filename, index=False)

    print("\n" + "=" * 80)
    print(f"✅ Predictions saved to: {filename}")
    print("=" * 80)

    # ------------------------------------------------------------------
    # TOP PLAYS — strongest EV picks with confidence filter
    # ------------------------------------------------------------------
    plays = df[
        (df['recommended_side'] == 'YES') &
        (df['confidence'].isin(['HIGH', 'MEDIUM']))
    ].copy()

    SUSPICIOUS_EV = 0.40  # HR markets are noisier; flag huge edges

    print(f"\n🎯 TOP HR PLAYS — {len(plays)} qualifying bets today")
    print("=" * 80)
    if plays.empty:
        print("   No qualifying plays (no edges ≥+10% EV with HIGH/MEDIUM confidence)")
    else:
        for i, (_, p) in enumerate(plays.head(7).iterrows(), 1):
            conf_icon = '🟢' if p['confidence'] == 'HIGH' else '🟡'
            verify = " ⚠️ VERIFY LINE" if p['ev'] > SUSPICIOUS_EV else ""
            stake = float(p.get('kelly_units', 0))
            print(
                f"\n{i}. {conf_icon} 💥 YES — {p['player_name']} "
                f"({p['team']} vs {p['opponent']}){verify}"
            )
            print(
                f"   STAKE {stake:.2f}u  |  {p['bookmaker']} {int(p['book_odds']):+d}  |  "
                f"Our {p['projection']:.1%} vs Book {p['book_prob']:.1%}  |  "
                f"Edge {p['edge_pp']:+.1f}pp  |  EV {p['ev']:+.1%}"
            )
            print(
                f"   Season: {p['hr']} HR in {p['pa']} PA ({p['hr_per_pa']:.1%}) | "
                f"Recent: {p['recent_hr_rate']:.1%}"
            )
            print(
                f"   {p.get('bats','?')}HB vs {p.get('pitcher_name','?')} ({p.get('pitcher_hand','?')}HP) | "
                f"Pitcher HR/9: {p['pitcher_hr9']:.2f} | "
                f"Park: {p['park_factor']:.2f}x | "
                f"Hand: {p.get('hand_factor', 1.0):.2f}x"
            )
            if p['red_flags'] != 'None':
                print(f"   ⚠️  {p['red_flags']}")

    # ------------------------------------------------------------------
    # Full projection list
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("📊 ALL PROJECTIONS")
    print("=" * 80)
    for _, row in df.head(25).iterrows():
        conf_icon = '🟢' if row['confidence'] == 'HIGH' else '🟡' if row['confidence'] == 'MEDIUM' else '🔴'
        side = row.get('recommended_side', '')
        side_str = ' 💥 YES' if side == 'YES' else (' ⚪ PASS' if side == 'PASS' else '')
        print(
            f"\n{conf_icon} {row['player_name']:25s} "
            f"({row['team']} vs {row['opponent']}){side_str}"
        )
        print(f"   HR Projection: {row['projection']:.1%} | Book: {row.get('book_implied', 'N/A')}")
        if pd.notna(row.get('book_prob')):
            print(
                f"   Edge: {row['edge_pp']:+.1f}pp | EV: {row['ev']:+.1%}"
            )
        print(
            f"   Season: {row['hr']} HR / {row['pa']} PA "
            f"({row['hr_per_pa']:.1%}) | "
            f"Pitcher HR/9: {row['pitcher_hr9']:.2f} | "
            f"Park: {row['park_factor']:.2f}x"
        )
        if row['red_flags'] != 'None':
            print(f"   ⚠️  {row['red_flags']}")

    print("\n" + "=" * 80)
    print("💡 EV ≥ +10% with edge ≥ +4pp qualifies as a TOP PLAY.")
    print("=" * 80)

else:
    print("\n⚠️  No predictions generated")
