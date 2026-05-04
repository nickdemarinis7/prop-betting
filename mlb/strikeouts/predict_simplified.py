"""
⚾ MLB STRIKEOUT PREDICTIONS V3 - SIMPLIFIED & DATA-DRIVEN
Core principle: Only keep what validation proves works
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os
import warnings
from dotenv import load_dotenv
from scipy import stats
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score

# Suppress pandas warnings
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Load environment variables
load_dotenv()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from core.name_utils import normalize_name, filter_by_name
from mlb.shared.scrapers.mlb_schedule import MLBScheduleScraper
from mlb.shared.scrapers.pitcher_stats import PitcherStatsScraper
from mlb.shared.scrapers.baseball_savant import BaseballSavantScraper
from mlb.shared.features.pitcher_context import PitcherContextAnalyzer
from mlb.shared.scrapers.rotochamp_lineups import RotoChampLineupScraper
from mlb.shared.scrapers.mlb_lineups import MLBLineupScraper
from mlb.shared.scrapers.batter_stats import BatterStatsScraper
from mlb.shared.scrapers.odds_api import (
    OddsAPIScraper,
    calculate_implied_probability,
    calculate_expected_value,
)

# Local module — ML corrector lives next to this script
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
from ml_corrector import StrikeoutMLCorrector
from prob_calibrator import ProbabilityCalibrator

print("=" * 80)
print("⚾ MLB STRIKEOUT PREDICTIONS V3 - SIMPLIFIED MODEL")
print("=" * 80)

# ----------------------------------------------------------------------
# Empirically calibrated parameters (from validation_results_*.csv).
# See /tmp/check_calibration.py for the analysis used to derive these.
# ----------------------------------------------------------------------
# Residual std from 281 historical predictions: 2.45 K. We use it as the
# floor for the per-pitcher std_dev so prob_X+ columns are calibrated to
# real-world hit rates, not to per-pitcher 5-game variance.
CALIBRATED_RESIDUAL_STD = 2.5

# Initialize scrapers
schedule_scraper = MLBScheduleScraper()
stats_scraper = PitcherStatsScraper()
savant_scraper = BaseballSavantScraper()
context_analyzer = PitcherContextAnalyzer()
lineup_scraper = RotoChampLineupScraper()
mlb_lineup_scraper = MLBLineupScraper()
batter_scraper = BatterStatsScraper()
odds_scraper = OddsAPIScraper()

# Initialize ML residual corrector. Trained on validation history.
ml_corrector = StrikeoutMLCorrector()
if not ml_corrector.load():
    print("   ⚠️  No trained ML corrector found, attempting to train...")
    if not ml_corrector.train():
        print("   ⚠️  ML corrector unavailable; falling back to no-correction.")
        ml_corrector = None

# Initialize probability calibrator. Maps raw normal-CDF probabilities to
# empirically calibrated hit rates (isotonic regression on validation data).
prob_calibrator = ProbabilityCalibrator()
if not prob_calibrator.load():
    print("   ⚠️  No prob calibrator found, attempting to train...")
    if not prob_calibrator.train():
        print("   ⚠️  Calibrator unavailable; using raw probabilities.")
        prob_calibrator = None

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

# Fetch odds data
print("\n📡 Fetching sportsbook odds...")
odds_df = odds_scraper.get_all_strikeout_odds()
if not odds_df.empty:
    print(f"   ✓ Found odds for {len(odds_df)} pitcher/line combinations")
else:
    print("   ⚠️  No odds found")

# Collect starting pitchers
starters = []
for _, game in games_df.iterrows():
    if pd.notna(game.get('away_pitcher')):
        starters.append({
            'pitcher_name': game['away_pitcher'],
            'pitcher_id': game.get('away_pitcher_id'),
            'team': game['away_team'],
            'team_id': game.get('away_team_id'),
            'opponent': game['home_team'],
            'opponent_id': game.get('home_team_id'),
            'is_home': 0,
            'ballpark': game.get('venue', 'Unknown'),
            'game_time': game.get('game_time', '')
        })
    
    if pd.notna(game.get('home_pitcher')):
        starters.append({
            'pitcher_name': game['home_pitcher'],
            'pitcher_id': game.get('home_pitcher_id'),
            'team': game['home_team'],
            'team_id': game.get('home_team_id'),
            'opponent': game['away_team'],
            'opponent_id': game.get('away_team_id'),
            'is_home': 1,
            'ballpark': game.get('venue', 'Unknown'),
            'game_time': game.get('game_time', '')
        })

print(f"   ✓ Found {len(starters)} starting pitchers")

# Fetch season stats with OPTIMAL BLEND
print("\n📊 Fetching season stats...")

# PHASE 1: Data blending optimization
# Test different weights: 2026 is current season but small sample
# 2025 is full season but older data
# Current: 70% 2026 / 30% 2025
# Test alternatives: 80/20, 60/40, or use game count weighted

stats_2025 = stats_scraper.get_season_stats(season=2025, min_starts=10)
stats_2026 = stats_scraper.get_season_stats(season=2026, min_starts=2)

print(f"   ✓ Loaded {len(stats_2025)} pitchers from 2025")
print(f"   ✓ Loaded {len(stats_2026)} pitchers from 2026")

# SMART BLENDING: Weight by games pitched in 2026
# If pitcher has 10+ games in 2026, trust it more
# If only 2-3 games, weight 2025 more heavily
all_season_stats = []
for pitcher_id in set(stats_2025['pitcher_id'].tolist() + stats_2026['pitcher_id'].tolist()):
    stats_25 = stats_2025[stats_2025['pitcher_id'] == pitcher_id]
    stats_26 = stats_2026[stats_2026['pitcher_id'] == pitcher_id]
    
    if not stats_26.empty and not stats_25.empty:
        # Game-count weighted blend
        games_26 = stats_26.iloc[0].get('games', 5)
        
        # Weight formula: more 2026 games = more weight to 2026
        # 2 games = 50/50 blend
        # 10 games = 80/20 blend
        # 20 games = 90/10 blend
        weight_26 = min(0.5 + (games_26 / 25), 0.9)  # Cap at 90%
        weight_25 = 1 - weight_26
        
        blended = stats_26.iloc[0].copy()
        blended['K9'] = (stats_26.iloc[0]['K9'] * weight_26) + (stats_25.iloc[0]['K9'] * weight_25)
        blended['IP'] = stats_26.iloc[0]['IP']  # Use 2026 IP
        blended['weight_2026'] = weight_26  # Track for validation
        all_season_stats.append(blended)
    elif not stats_26.empty:
        row_copy = stats_26.iloc[0].copy()
        row_copy['weight_2026'] = 1.0
        all_season_stats.append(row_copy)
    else:
        row_copy = stats_25.iloc[0].copy()
        row_copy['weight_2026'] = 0.0
        all_season_stats.append(row_copy)

all_season_stats = pd.DataFrame(all_season_stats)
print(f"   ✓ Combined: {len(all_season_stats)} total pitchers")
print(f"   📊 Avg 2026 weight: {all_season_stats['weight_2026'].mean():.1%}")

# SIMPLIFIED PREDICTION LOGIC
predictions = []

print("\n🎯 Generating predictions...")
print("=" * 80)

for starter in starters:
    pitcher_name = starter['pitcher_name']
    pitcher_id = starter['pitcher_id']
    
    if not pitcher_id:
        print(f"\n⚠️  Skipping {pitcher_name} (no pitcher ID)")
        continue
    
    print(f"\n📊 {pitcher_name} ({starter['team']} vs {starter['opponent']})")
    
    try:
        # 1. BASE K/9 (game-count weighted blend from above)
        season_stats = all_season_stats[all_season_stats['pitcher_id'] == pitcher_id]
        
        if season_stats.empty:
            print(f"   ⚠️  No season stats available")
            continue
        
        season_k9 = season_stats.iloc[0].get('K9', 0)
        # Get weight_2026 - handle NaN case
        weight_val = season_stats.iloc[0].get('weight_2026')
        if pd.isna(weight_val) or weight_val is None:
            # Default based on what data we have
            has_2026 = pitcher_id in stats_2026['pitcher_id'].values
            has_2025 = pitcher_id in stats_2025['pitcher_id'].values
            if has_2026 and not has_2025:
                weight_2026 = 1.0
            elif has_2025 and not has_2026:
                weight_2026 = 0.0
            else:
                weight_2026 = 0.7  # Default blend
        else:
            weight_2026 = float(weight_val)
        
        # 2. RECENT FORM (last 5 starts) - COLD STREAK DETECTION ONLY
        game_logs = stats_scraper.get_game_logs(pitcher_id, season=2026)
        
        if game_logs.empty or len(game_logs) < 2:
            print(f"   ⚠️  Insufficient game logs")
            continue
        
        recent_games = game_logs.head(5)
        recent_k = recent_games['SO'].values
        recent_ip = recent_games['IP'].values
        
        total_k = recent_k.sum()
        total_ip = recent_ip.sum()
        recent_k9 = (total_k / total_ip) * 9 if total_ip > 0 else season_k9
        
        # COLD STREAK: Only adjust if pitcher is struggling significantly
        if recent_k9 < season_k9 - 1.5:  # 1.5+ K/9 drop (more lenient than current)
            # Weight recent more when pitcher is cold
            base_k9 = (season_k9 * 0.3) + (recent_k9 * 0.7)
            print(f"   ⚠️  Cold streak detected (Season: {season_k9:.2f}, Recent: {recent_k9:.2f})")
        else:
            # Normal: 60/40 season/recent blend
            base_k9 = (season_k9 * 0.6) + (recent_k9 * 0.4)
        
        print(f"   Base K/9: {base_k9:.2f} (Season: {season_k9:.2f}, Recent: {recent_k9:.2f})")
        
        # 3. EXPECTED IP (from historical patterns + context)
        _game_time_str = starter.get('game_time', '')
        try:
            _game_dt = datetime.fromisoformat(_game_time_str.replace('Z', '+00:00'))
            is_day_game = _game_dt.hour < 21
        except:
            is_day_game = False
        
        context = context_analyzer.get_full_context(
            pitcher_id,
            game_date=datetime.now(),
            is_day_game=is_day_game
        )
        
        expected_ip = context['expected_ip']
        is_short_rest = context.get('is_short_rest', False)
        
        print(f"   Expected IP: {expected_ip:.1f}")
        
        # 4. BASE PROJECTION
        base_projection = (base_k9 / 9) * expected_ip
        
        # 5. OPPONENT ADJUSTMENT (single, clean adjustment)
        opponent_team_id = starter.get('opponent_id')
        pitcher_hand = season_stats.iloc[0].get('hand', 'R')
        
        # Try actual lineup first, fall back to projected
        game_id = None
        for _, game in games_df.iterrows():
            if game['away_team'] == starter['team'] or game['home_team'] == starter['team']:
                game_id = game.get('game_id')
                break
        
        actual_lineup = None
        if game_id:
            team_type = 'home' if starter['is_home'] else 'away'
            actual_lineup = mlb_lineup_scraper.get_lineup_for_team(game_id, team_type=team_type)
        
        if actual_lineup and len(actual_lineup) >= 8:
            lineup = actual_lineup
            opponent_k_rate = batter_scraper.calculate_lineup_k_rate(lineup, vs_hand=pitcher_hand, season=2026)
            print(f"   ✓ Actual lineup: {opponent_k_rate:.1%} K rate vs {pitcher_hand}HP")
        else:
            lineup = lineup_scraper.get_projected_lineup(starter['opponent'])
            if lineup and len(lineup) >= 8:
                opponent_k_rate = batter_scraper.calculate_lineup_k_rate(lineup, vs_hand=pitcher_hand, season=2026)
                print(f"   ⚠️  Projected lineup: {opponent_k_rate:.1%} K rate vs {pitcher_hand}HP")
            else:
                opponent_k_rate = savant_scraper.get_team_k_rate_vs_hand(starter['opponent'], vs_hand=pitcher_hand)
                print(f"   ⚠️  Team-level K rate: {opponent_k_rate:.1%}")
        
        # Single opponent multiplier (shrunk toward league average)
        league_avg_k_rate = 0.23
        shrunk_k_rate = (opponent_k_rate * 0.75) + (league_avg_k_rate * 0.25)
        opponent_multiplier = min(shrunk_k_rate / league_avg_k_rate, 1.15)
        
        # 6. CONTEXT ADJUSTMENTS (simplified to 2 factors)
        # Home/away
        home_multiplier = 1.02 if starter['is_home'] else 0.98
        
        # Short rest (only if < 4 days)
        rest_multiplier = 0.92 if is_short_rest else 1.0
        if is_short_rest:
            print(f"   ⚠️  Short rest detected")
        
        # 7. ML CORRECTION — trained residual corrector replaces the heuristic.
        # The model learns systematic biases from validation history
        # (e.g. we project +0.45 K too high on average) and adjusts.
        features_for_ml = {
            'season_k9': season_k9,
            'recent_k9': recent_k9,
            'expected_ip': expected_ip,
            'opponent_k_rate': opponent_k_rate,
            'is_home': starter['is_home'],
            'is_day_game': is_day_game,
            'is_short_rest': is_short_rest,
        }
        
        # 8. FINAL PROJECTION
        # Build heuristic projection first; corrector applies on top.
        heuristic_projection = (
            base_projection * opponent_multiplier * home_multiplier * rest_multiplier
        )
        
        if ml_corrector is not None and ml_corrector.is_trained:
            ml_correction = ml_corrector.predict_correction(features_for_ml)
            # The bundled corrector was trained on the OLD predict.py outputs
            # which already included aggressive heuristic adjustments. Feeding
            # it our cleaner heuristic projection here risks over-correction,
            # so we cap tighter than the corrector's internal ±1.5 clip until
            # it is re-trained on simplified outputs.
            ml_correction = float(np.clip(ml_correction, -0.7, 0.7))
            if abs(ml_correction) > 0.1:
                print(f"   🤖 ML correction: {ml_correction:+.2f} K")
        else:
            ml_correction = 0.0
        
        final_projection = heuristic_projection + ml_correction
        final_projection = max(final_projection, 0.5)

        # 8b. PER-PITCHER ANCHOR SHRINKAGE
        # The heuristic (recency blend × opp mult × home/rest × ML corr)
        # overshoots each pitcher's own season baseline by +1.21 K on
        # average across 304 validated preds. Shrinking toward a flat
        # league mean (4.9) fits backtest best but collapses true aces
        # to 5.5 K, flipping every over-pick to under.
        #
        # Per-pitcher anchor = season_k9/9 × expected_ip preserves ace
        # identity (deGrom anchor ≈ 6.7, journeyman anchor ≈ 3.5).
        # Shrinkage is applied only when the heuristic is meaningfully
        # above the anchor (> 0.5 K gap) — protects cases where recency
        # legitimately elevates the projection.
        #
        # Tuning (_anchor_tune.py, n=304):
        #   factor 0.60, gap_min 0.5
        #   → overall MAE 2.05, bias +0.55
        #   → deGrom 7.5 stays at 7.0 (still OVER book 6.5)
        #   → mid-tier inflated 7.5 with anchor 4.44 shrinks to 5.7
        # Accepts slightly worse raw MAE vs flat-league approach in
        # exchange for realistic per-pitcher behavior.
        ANCHOR_SHRINK = 0.60
        ANCHOR_GAP_MIN = 0.5
        own_anchor = (season_k9 / 9.0) * expected_ip
        pre_shrink_projection = final_projection
        if final_projection > own_anchor + ANCHOR_GAP_MIN:
            final_projection = (
                final_projection
                - ANCHOR_SHRINK * (final_projection - own_anchor)
            )
            if abs(final_projection - pre_shrink_projection) > 0.1:
                print(
                    f"   📉 Anchor shrink: {pre_shrink_projection:.1f} → "
                    f"{final_projection:.1f} (anchor={own_anchor:.1f}, "
                    f"season K/9={season_k9:.1f})"
                )

        print(f"   Projection: {final_projection:.1f} K")
        
        # 9. CONFIDENCE ASSESSMENT (informational only, doesn't change projection)
        red_flags = []
        
        # Track blowup rate (games with ≤2 K)
        blowup_games = (recent_k <= 2).sum()
        blowup_rate = blowup_games / len(recent_k) if len(recent_k) > 0 else 0
        if blowup_rate >= 0.4:  # 40% or more blowups
            red_flags.append(f'High blowup rate ({blowup_games}/{len(recent_k)})')
        
        # ERA check (informational) - calculate from ER and IP if ERA column not present
        if 'ERA' in recent_games.columns:
            recent_era = recent_games['ERA'].mean()
        elif 'ER' in recent_games.columns and 'IP' in recent_games.columns:
            # Calculate ERA: (ER / IP) * 9
            total_er = recent_games['ER'].sum()
            total_ip = recent_games['IP'].sum()
            recent_era = (total_er / total_ip) * 9 if total_ip > 0 else 0
        else:
            recent_era = 0
            
        if recent_era > 5.0:
            red_flags.append(f'High ERA ({recent_era:.1f})')
        
        # Volatility check
        if len(recent_k) >= 3:
            k_std = recent_k.std()
            if k_std > 4.0:
                red_flags.append(f'High variance (σ={k_std:.1f})')
        
        # Confidence tier
        # Diagnostic found old heuristic was INVERTED: HIGH picks had worse
        # MAE (2.43) than LOW (1.44). Old rule rewarded aces with clean form
        # — exactly the pitchers we over-project most.
        #
        # New rule: confidence reflects PROJECTION ACCURACY, not pitcher
        # quality. Projections of 4-6 K are most accurate (MAE 1.90, near-zero
        # bias). Projections >=6 K are much less accurate (MAE 3.08).
        if final_projection >= 7.0:
            confidence = 'LOW'       # high-K projections historically unreliable
        elif final_projection >= 6.0 or len(red_flags) >= 2 or blowup_rate >= 0.3:
            confidence = 'MEDIUM'
        else:
            confidence = 'HIGH'      # 4-6 K band is our best-calibrated zone
        # Any red flag demotes HIGH → MEDIUM
        if confidence == 'HIGH' and len(red_flags) >= 1:
            confidence = 'MEDIUM'
        
        if red_flags:
            print(f"   Red flags: {', '.join(red_flags)}")
        
        # 10. PROBABILITY DISTRIBUTION
        # Step 1: raw prob from normal CDF using empirical residual std.
        # Step 2: feed through isotonic calibrator so the prob actually
        #         matches the historical hit rate (was 4-12pp overconfident).
        std_dev = CALIBRATED_RESIDUAL_STD
        probabilities = {}
        for line in [3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5]:
            raw_prob = 1 - stats.norm.cdf(line, final_projection, std_dev)
            if prob_calibrator is not None and prob_calibrator.is_fitted:
                cal_prob = prob_calibrator.calibrate(raw_prob)
            else:
                cal_prob = raw_prob
            probabilities[f'prob_{line}+'] = round(float(cal_prob), 3)
        
        # 11. STORE PREDICTION
        pred = {
            'pitcher': pitcher_name,
            'team': starter['team'],
            'opponent': starter['opponent'],
            'is_home': starter['is_home'],
            'projection': round(final_projection, 1),
            'base_projection': round(base_projection, 1),
            'season_k9': round(season_k9, 2),
            'recent_k9': round(recent_k9, 2),
            'expected_ip': round(expected_ip, 1),
            'opponent_k_rate': round(opponent_k_rate, 3),
            'opponent_multiplier': round(opponent_multiplier, 3),
            'ml_correction': round(ml_correction, 2),
            'confidence': confidence,
            'red_flags': '; '.join(red_flags) if red_flags else 'None',
            'blowup_rate': round(blowup_rate, 2),
            'weight_2026': round(weight_2026, 2),
            'recent_era': round(recent_era, 2),
            'std_dev': round(std_dev, 2),
            'book_implied': 'N/A',
            'model_vs_book': 0,  # Will be calculated after odds lookup
            **probabilities,
        }
        
        predictions.append(pred)
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        continue

# SAVE AND DISPLAY
if predictions:
    df = pd.DataFrame(predictions)
    df = df.sort_values('projection', ascending=False)

    # ------------------------------------------------------------------
    # EDGE ANALYSIS: for each pitcher, find the best (Over or Under) bet
    # at the most-traded line and compute EV-based edge metrics.
    # ------------------------------------------------------------------
    df['book_line'] = float('nan')
    df['book_odds'] = float('nan')
    df['recommended_side'] = ''
    df['our_prob'] = float('nan')
    df['book_prob'] = float('nan')
    df['edge_pp'] = 0.0   # edge in percentage points (our - book)
    df['ev'] = 0.0        # expected value per 1 unit risked
    df['kelly_units'] = 0.0  # ¼-Kelly stake (capped at 1.5u)
    df['bookmaker'] = ''
    
    def _our_prob_at(row, line, side):
        """Return our model's calibrated probability for OVER / UNDER at
        any book line. Computed on-demand from the row's projection +
        std_dev so we can support lines outside the precomputed PROB_LINES
        (e.g. 2.5, 11.5).
        """
        if pd.isna(row.get('projection')) or pd.isna(row.get('std_dev')):
            return None
        raw_over = 1.0 - stats.norm.cdf(
            line, float(row['projection']), float(row['std_dev'])
        )
        if prob_calibrator is not None and prob_calibrator.is_fitted:
            cal_over = float(prob_calibrator.calibrate(raw_over))
        else:
            cal_over = raw_over
        return cal_over if side == 'Over' else 1.0 - cal_over

    # ¼-Kelly unit sizing — capped to 1.5u per leg.
    KELLY_FRACTION = 0.25
    MAX_KELLY_UNITS = 1.5
    
    def kelly_units(our_p, american_odds):
        """Fractional Kelly sizing in betting units. Returns 0 if -EV."""
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
        # Scale so 1 full-Kelly ≈ 2.5u, ¼-Kelly ≈ ~0.6u typical.
        units = kelly * KELLY_FRACTION * 10
        return round(min(units, MAX_KELLY_UNITS), 2)

    # Skip edge analysis entirely if we have no odds (e.g. missing API key)
    have_odds = (
        not odds_df.empty
        and 'pitcher' in odds_df.columns
        and 'odds' in odds_df.columns
        and 'over_under' in odds_df.columns
    )

    # Thresholds for what qualifies as a TOP play / alternate rung
    # P&L sweep on n=40 simplified preds: min_prob 0.55–0.60 loses -14% ROI;
    # 0.65 flips to +4% ROI, 0.70 to +27%. We require BOTH EV and absolute
    # probability thresholds so low-probability +EV coin flips don't qualify.
    MIN_EV_PLAY = 0.05        # main-line bet must clear +5% EV
    MIN_EDGE_PP_PLAY = 0.04   # and +4pp probability edge
    MIN_PROB_PLAY = 0.65      # AND our calibrated prob must be ≥ 65%
                              # (diagnostic sweep: 0.60 → -14% ROI,
                              #  0.65 → +4% ROI, 0.70 → +27% ROI)
    MIN_EV_ALT = 0.05         # alternate rung EV threshold
    MIN_PROB_ALT = 0.60       # alt rungs slightly less strict
    
    # alternates_by_pitcher: pitcher_name -> list of dict rungs
    alternates_by_pitcher = {}

    for idx, row in df.iterrows():
        if not have_odds:
            break
        pitcher_odds = filter_by_name(odds_df, 'pitcher', row['pitcher'])
        if pitcher_odds.empty:
            # Last-name fallback (handles abbreviated names like "M. Perez")
            last = row['pitcher'].split()[-1]
            pitcher_odds = odds_df[
                odds_df['pitcher'].str.contains(last, case=False, na=False)
            ]
        if pitcher_odds.empty:
            continue
        
        # Scan EVERY (line, side, book) combo for this pitcher. Keep best EV
        # per (line, side) — this is the alternate-line ladder set.
        candidate_rungs = []
        seen = {}  # (line, side) -> best ev rung dict
        for _, lr in pitcher_odds.iterrows():
            line_val = float(lr['line'])
            side = lr['over_under']
            our_p = _our_prob_at(row, line_val, side)
            if our_p is None:
                continue
            book_p = calculate_implied_probability(lr['odds'])
            ev = calculate_expected_value(our_p, lr['odds'])
            rung = {
                'line': line_val,
                'side': side.upper(),
                'odds': int(lr['odds']),
                'bookmaker': lr['bookmaker'],
                'our_prob': round(our_p, 3),
                'book_prob': round(book_p, 3),
                'edge_pp': round((our_p - book_p) * 100, 1),
                'ev': round(ev, 3),
                'kelly_units': kelly_units(our_p, lr['odds']),
            }
            key = (line_val, side)
            if key not in seen or rung['ev'] > seen[key]['ev']:
                seen[key] = rung
        candidate_rungs = list(seen.values())
        if not candidate_rungs:
            continue
        
        # Main pick: highest EV at the most-traded line (Over closest to even)
        over_only = pitcher_odds[pitcher_odds['over_under'] == 'Over']
        if over_only.empty:
            continue
        main_idx = over_only['odds'].abs().idxmin()
        main_line = float(over_only.loc[main_idx, 'line'])
        df.at[idx, 'book_line'] = main_line
        df.at[idx, 'book_implied'] = (
            f"{main_line} @ {int(over_only.loc[main_idx, 'odds']):+d} "
            f"({over_only.loc[main_idx, 'bookmaker']})"
        )
        # Best of (Over, Under) at the main line
        main_rungs = [r for r in candidate_rungs if r['line'] == main_line]
        if not main_rungs:
            continue
        main_pick = max(main_rungs, key=lambda r: r['ev'])
        df.at[idx, 'book_odds'] = main_pick['odds']
        df.at[idx, 'bookmaker'] = main_pick['bookmaker']
        df.at[idx, 'our_prob'] = main_pick['our_prob']
        df.at[idx, 'book_prob'] = main_pick['book_prob']
        df.at[idx, 'edge_pp'] = main_pick['edge_pp']
        df.at[idx, 'ev'] = main_pick['ev']
        df.at[idx, 'kelly_units'] = main_pick['kelly_units']
        if (
            main_pick['ev'] >= MIN_EV_PLAY
            and (main_pick['our_prob'] - main_pick['book_prob']) >= MIN_EDGE_PP_PLAY
            and main_pick['our_prob'] >= MIN_PROB_PLAY
        ):
            df.at[idx, 'recommended_side'] = main_pick['side']
        else:
            df.at[idx, 'recommended_side'] = 'PASS'
        df.at[idx, 'model_vs_book'] = round(row['projection'] - main_line, 1)
        
        # Alternate rungs: any other +EV side at non-main lines, same direction
        # as the main pick. We only ladder in one direction so the legs
        # actually correlate (avoids buying both Over 4.5 AND Under 6.5).
        alts = [
            r for r in candidate_rungs
            if r['line'] != main_line
            and r['side'] == main_pick['side']
            and r['ev'] >= MIN_EV_ALT
            and r['our_prob'] >= MIN_PROB_ALT
        ]
        alts.sort(key=lambda r: r['ev'], reverse=True)
        if alts:
            alternates_by_pitcher[row['pitcher']] = alts

    # ------------------------------------------------------------------
    # Save full CSV (sort by EV so best plays surface first)
    # ------------------------------------------------------------------
    confidence_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
    df['confidence_rank'] = df['confidence'].map(confidence_order)
    df = df.sort_values(['ev', 'confidence_rank'], ascending=[False, True])

    if target_date:
        date_str = target_date.replace('-', '')
    else:
        date_str = datetime.now().strftime('%Y%m%d')
    filename = f"predictions_strikeouts_simplified_{date_str}.csv"
    df_output = df.drop(columns=['confidence_rank'])

    # Guard: if the pre-game predictions file already exists, don't overwrite
    # it — the original has the correct pre-game odds. Re-runs (potentially
    # during live games) save to a separate file so odds integrity is preserved.
    if os.path.exists(filename):
        rerun_file = f"predictions_strikeouts_simplified_{date_str}_rerun.csv"
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
        (df['recommended_side'].isin(['OVER', 'UNDER'])) &
        (df['confidence'].isin(['HIGH', 'MEDIUM']))
    ].copy()

    SUSPICIOUS_EV = 0.25  # Edges this large often mean a stale line; verify

    print(f"\n🎯 TOP PLAYS — {len(plays)} qualifying bets today")
    print("=" * 80)
    if plays.empty:
        print("   No qualifying plays (no edges ≥+5% EV with HIGH/MEDIUM confidence)")
    else:
        # Per-pitcher exposure cap: scale total Kelly across main + alts so we
        # never put more than MAX_PITCHER_EXPOSURE total units on one pitcher.
        MAX_PITCHER_EXPOSURE = 1.5
        
        for i, (_, p) in enumerate(plays.head(7).iterrows(), 1):
            conf_icon = '🟢' if p['confidence'] == 'HIGH' else '🟡'
            side_icon = '📈' if p['recommended_side'] == 'OVER' else '📉'
            verify = " ⚠️ VERIFY LINE" if p['ev'] > SUSPICIOUS_EV else ""
            
            # Combine main pick + same-direction alternates into one ladder
            main_rung = {
                'line': p['book_line'],
                'side': p['recommended_side'],
                'odds': int(p['book_odds']),
                'bookmaker': p['bookmaker'],
                'our_prob': p['our_prob'],
                'book_prob': p['book_prob'],
                'edge_pp': p['edge_pp'],
                'ev': p['ev'],
                'kelly_units': float(p.get('kelly_units', 0)),
                'is_main': True,
            }
            alts = alternates_by_pitcher.get(p['pitcher'], [])
            ladder = [main_rung] + [
                {**a, 'is_main': False} for a in alts
            ]
            # Scale stakes so total exposure ≤ MAX_PITCHER_EXPOSURE
            total_units = sum(r['kelly_units'] for r in ladder)
            if total_units > MAX_PITCHER_EXPOSURE and total_units > 0:
                scale = MAX_PITCHER_EXPOSURE / total_units
                for r in ladder:
                    r['kelly_units'] = round(r['kelly_units'] * scale, 2)
            
            print(
                f"\n{i}. {conf_icon} {side_icon} {p['recommended_side']} — "
                f"{p['pitcher']} ({p['team']} vs {p['opponent']}){verify}"
            )
            print(
                f"   Projection: {p['projection']:.1f} K | "
                f"K/9 S/R: {p['season_k9']:.1f}/{p['recent_k9']:.1f} | "
                f"Exp IP: {p['expected_ip']:.1f}"
            )
            if p['red_flags'] != 'None':
                print(f"   ⚠️  {p['red_flags']}")
            
            # Print ladder rungs (main first, then alternates by EV desc)
            print(f"   Ladder ({len(ladder)} rung{'s' if len(ladder) > 1 else ''}, "
                  f"total {sum(r['kelly_units'] for r in ladder):.2f}u):")
            for r in ladder:
                tag = 'MAIN' if r['is_main'] else 'ALT '
                print(
                    f"     • {tag} {r['kelly_units']:.2f}u  "
                    f"{p['recommended_side']} {r['line']} @ {r['odds']:+d} "
                    f"({r['bookmaker']})  "
                    f"Our {r['our_prob']:.1%} | EV {r['ev']:+.1%}"
                )

    # ------------------------------------------------------------------
    # Full projection list (for reference, sorted by EV desc)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("📊 ALL PROJECTIONS")
    print("=" * 80)
    for _, row in df.iterrows():
        conf_icon = '🟢' if row['confidence'] == 'HIGH' else '🟡' if row['confidence'] == 'MEDIUM' else '🔴'
        side = row.get('recommended_side', '')
        side_str = ''
        if side in ('OVER', 'UNDER'):
            side_icon = '📈' if side == 'OVER' else '📉'
            side_str = f" {side_icon} {side} {row['book_line']}"
        elif side == 'PASS':
            side_str = " ⚪ PASS"
        print(f"\n{conf_icon} {row['pitcher']:25s} ({row['team']} vs {row['opponent']}){side_str}")
        print(f"   Projection:   {row['projection']:.1f} K | Book: {row.get('book_implied', 'N/A')}")
        if pd.notna(row.get('our_prob')):
            print(
                f"   Our: {row['our_prob']:.1%} | Book: {row['book_prob']:.1%} "
                f"| Edge: {row['edge_pp']:+.1f}pp | EV: {row['ev']:+.1%}"
            )
        print(
            f"   K/9 S/R: {row['season_k9']:.2f}/{row['recent_k9']:.2f} | "
            f"Exp IP: {row['expected_ip']:.1f} | "
            f"Opp K%: {row['opponent_k_rate']:.1%}"
        )
        if row['red_flags'] != 'None':
            print(f"   ⚠️  {row['red_flags']}")

    print("\n" + "=" * 80)
    print("💡 EV ≥ +5% with edge ≥ +4pp qualifies as a TOP PLAY.")
    print("=" * 80)
else:
    print("\n⚠️  No predictions generated")
