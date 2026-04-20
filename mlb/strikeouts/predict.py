"""
MLB Strikeout Predictions V2 - Simplified & Improved
Focus on what actually matters for strikeout prediction
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from mlb.shared.scrapers.mlb_schedule import MLBScheduleScraper
from mlb.shared.scrapers.pitcher_stats import PitcherStatsScraper
from mlb.shared.scrapers.baseball_savant import BaseballSavantScraper
from mlb.shared.features.pitcher_context import PitcherContextAnalyzer
from mlb.shared.scrapers.rotochamp_lineups import RotoChampLineupScraper
from mlb.shared.scrapers.mlb_lineups import MLBLineupScraper
from mlb.shared.scrapers.batter_stats import BatterStatsScraper
from scipy import stats as scipy_stats

try:
    from ml_corrector import StrikeoutMLCorrector
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False


print("=" * 80)
print("⚾ MLB STRIKEOUT PREDICTIONS V2 - SIMPLIFIED MODEL")
print("=" * 80)

# Initialize scrapers
schedule_scraper = MLBScheduleScraper()
stats_scraper = PitcherStatsScraper()
savant_scraper = BaseballSavantScraper()
context_analyzer = PitcherContextAnalyzer()
lineup_scraper = RotoChampLineupScraper()
mlb_lineup_scraper = MLBLineupScraper()
batter_scraper = BatterStatsScraper()

# Initialize ML corrector (loads saved model or trains from validation data)
ml_corrector = None
if ML_AVAILABLE:
    ml_corrector = StrikeoutMLCorrector()
    if not ml_corrector.load():
        print("   ⚠️  No saved ML model found, attempting to train...")
        if not ml_corrector.train():
            ml_corrector = None

# Get games for specified date (or today)
import sys
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

# Collect all starting pitchers
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

# Fetch all season stats once (more efficient)
print("\n📊 Fetching season stats...")

# Get 2025 full season data (for baseline)
stats_2025 = stats_scraper.get_season_stats(season=2025, min_starts=10)
print(f"   ✓ Loaded {len(stats_2025)} pitchers from 2025")

# Get 2026 early season data (lower threshold since season just started)
stats_2026 = stats_scraper.get_season_stats(season=2026, min_starts=2)
print(f"   ✓ Loaded {len(stats_2026)} pitchers from 2026")

# Combine: blend 2025 & 2026 data (70% 2026, 30% 2025)
all_season_stats = []
for pitcher_id in set(stats_2025['pitcher_id'].tolist() + stats_2026['pitcher_id'].tolist()):
    stats_25 = stats_2025[stats_2025['pitcher_id'] == pitcher_id]
    stats_26 = stats_2026[stats_2026['pitcher_id'] == pitcher_id]
    
    if not stats_26.empty and not stats_25.empty:
        # Pitcher in both - blend the K9
        blended = stats_26.iloc[0].copy()
        blended['K9'] = (stats_26.iloc[0]['K9'] * 0.7) + (stats_25.iloc[0]['K9'] * 0.3)
        blended['IP'] = stats_26.iloc[0]['IP']  # Use 2026 IP
        all_season_stats.append(blended)
    elif not stats_26.empty:
        # Only 2026 data
        all_season_stats.append(stats_26.iloc[0])
    else:
        # Only 2025 data
        all_season_stats.append(stats_25.iloc[0])

all_season_stats = pd.DataFrame(all_season_stats)
print(f"   ✓ Combined: {len(all_season_stats)} total pitchers (blended 2025+2026)")

# Process each pitcher
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
        # 1. GET PITCHER BASELINE STATS
        season_stats = all_season_stats[all_season_stats['pitcher_id'] == pitcher_id]
        
        if season_stats.empty:
            print(f"   ⚠️  No season stats available")
            continue
        
        season_k9 = season_stats.iloc[0].get('K9', 0)
        season_ip = season_stats.iloc[0].get('IP', 0)
        
        # Get recent form (last 5 starts) - use 2026 data
        game_logs = stats_scraper.get_game_logs(pitcher_id, season=2026)
        
        if game_logs.empty or len(game_logs) < 2:
            print(f"   ⚠️  Insufficient game logs (need at least 2 starts)")
            continue
        
        recent_games = game_logs.head(5)
        recent_k = recent_games['SO'].values
        recent_ip = recent_games['IP'].values
        
        # Calculate recent K/9
        total_k = recent_k.sum()
        total_ip = recent_ip.sum()
        recent_k9 = (total_k / total_ip) * 9 if total_ip > 0 else season_k9
        
        # PHASE 1 FIX #3: Cold streak penalty - weight recent more when pitcher is struggling
        if recent_k9 < season_k9 - 1.0:
            # Pitcher is cold - weight recent form more heavily
            base_k9 = (season_k9 * 0.4) + (recent_k9 * 0.6)
            print(f"   ⚠️  Cold streak detected - weighting recent form more")
        else:
            # Normal weighting (60% season, 40% recent)
            base_k9 = (season_k9 * 0.6) + (recent_k9 * 0.4)
        
        # PHASE 1 FIX #2: Regress elite K/9 pitchers toward mean
        if base_k9 > 12.0:
            original_k9 = base_k9
            base_k9 = (base_k9 * 0.7) + (11.0 * 0.3)
            print(f"   ⚠️  Elite K/9 regression: {original_k9:.2f} → {base_k9:.2f}")
        
        print(f"   Season K/9: {season_k9:.2f} | Recent K/9: {recent_k9:.2f} | Base: {base_k9:.2f}")
        
        # 2. GET PITCHER CONTEXT
        is_day_game = 'AM' in starter.get('game_time', '') or '12:' in starter.get('game_time', '') or '1:' in starter.get('game_time', '')
        
        context = context_analyzer.get_full_context(
            pitcher_id,
            game_date=datetime.now(),
            is_day_game=is_day_game
        )
        
        expected_ip = context['expected_ip']
        is_short_rest = context.get('is_short_rest', False)
        
        print(f"   Expected IP: {expected_ip:.1f} | Short Rest: {is_short_rest} | Day Game: {is_day_game}")
        
        # Don't use day/night K/9 for now - it's causing issues with small sample sizes
        # Just use the base K/9
        adjusted_k9 = base_k9
        
        # 3. BASE PROJECTION (K/9 × Expected IP)
        base_projection = (adjusted_k9 / 9) * expected_ip
        
        print(f"   Base Projection: {base_projection:.2f} K")
        
        # 4. OPPONENT LINEUP ADJUSTMENT
        opponent_team_id = starter.get('opponent_id')
        pitcher_hand = season_stats.iloc[0].get('hand', 'R')
        
        # Try to get actual lineup first (if game has started or lineup posted)
        game_id = None
        for _, game in games_df.iterrows():
            if (game['away_team'] == starter['team'] or game['home_team'] == starter['team']):
                game_id = game.get('game_id')
                break
        
        actual_lineup = None
        if game_id:
            # Determine if we need away or home lineup
            team_type = 'home' if starter['is_home'] else 'away'
            actual_lineup = mlb_lineup_scraper.get_lineup_for_team(game_id, team_type=team_type)
        
        if actual_lineup and len(actual_lineup) >= 8:
            print(f"   ✓ Using ACTUAL lineup ({len(actual_lineup)} batters)")
            lineup = actual_lineup
            # Calculate lineup-specific K%
            opponent_k_rate = batter_scraper.calculate_lineup_k_rate(
                lineup, 
                vs_hand=pitcher_hand,
                season=2026
            )
            print(f"   ✓ Lineup-specific K%: {opponent_k_rate:.1%} (vs {pitcher_hand}HP)")
        else:
            # Fall back to projected lineup
            lineup = lineup_scraper.get_projected_lineup(starter['opponent'])
            if lineup and len(lineup) >= 8:
                print(f"   ⚠️  Using projected lineup ({len(lineup)} batters)")
                # Try to calculate K% for projected lineup
                opponent_k_rate = batter_scraper.calculate_lineup_k_rate(
                    lineup,
                    vs_hand=pitcher_hand,
                    season=2026
                )
                print(f"   ⚠️  Projected lineup K%: {opponent_k_rate:.1%} (vs {pitcher_hand}HP)")
            else:
                # Fall back to team-level K%
                print(f"   ⚠️  No lineup available, using team K%")
                opponent_k_rate = savant_scraper.get_team_k_rate_vs_hand(
                    starter['opponent'],
                    vs_hand=pitcher_hand
                )
        
        league_avg_k_rate = 0.23
        # Shrink opponent K% toward league average to reduce noise from small samples
        shrunk_k_rate = (opponent_k_rate * 0.75) + (league_avg_k_rate * 0.25)
        opponent_multiplier = min(shrunk_k_rate / league_avg_k_rate, 1.20)
        
        print(f"   Opponent K%: {opponent_k_rate:.1%} (vs {pitcher_hand}HP) | Multiplier: {opponent_multiplier:.3f}")
        
        # 5. OTHER ADJUSTMENTS
        home_multiplier = 1.05 if starter['is_home'] else 0.95
        
        # Short rest penalty
        rest_multiplier = 0.90 if is_short_rest else 1.0
        
        # 6. FINAL PROJECTION
        final_projection = base_projection * opponent_multiplier * home_multiplier * rest_multiplier
        
        print(f"   Final Projection: {final_projection:.2f} K")
        
        # 7. STD DEV (used for probabilities after all corrections)
        std_dev = max(recent_k.std(), 2.0) if len(recent_k) > 2 else 3.0
        
        # 8. EARLY EXIT RISK CHECKS
        red_flags = []
        
        # Track blowup rate (games with ≤2 K in recent starts)
        blowup_games = (recent_k <= 2).sum()
        blowup_rate = blowup_games / len(recent_k) if len(recent_k) > 0 else 0
        if blowup_rate >= 0.3:
            red_flags.append(f'Blowup risk ({blowup_games}/{len(recent_k)} starts ≤2K)')
        
        if len(recent_games) >= 3:
            recent_er = recent_games.head(3)['ER'].sum()
            recent_ip_3 = recent_games.head(3)['IP'].sum()
            recent_era = recent_games['ERA'].mean() if 'ERA' in recent_games.columns else 0
            
            # Check 1: Blowout risk (ERA >4.5)
            if recent_era > 4.5:
                max_k = 5.5
                if final_projection > max_k:
                    print(f"   ⚠️  Blowout Risk (Recent ERA: {recent_era:.2f}) - Capping at {max_k} K")
                    final_projection = min(final_projection, 5.5)
                    red_flags.append(f'High ERA ({recent_era:.1f})')
            
            # Check 2: High volatility (inconsistent performances)
            if len(recent_k) >= 3:
                k_std = recent_k.std()
                if k_std > 3.0 and final_projection > 7.0:
                    original = final_projection
                    final_projection *= 0.90
                    print(f"   ⚠️  High Volatility (σ={k_std:.1f}) - Reducing: {original:.1f} → {final_projection:.1f} K")
                    red_flags.append(f'High variance (σ={k_std:.1f})')
            
            # Check 3: Recent short outings (avg IP <5.0 in last 3)
            avg_recent_ip = recent_ip[:3].mean()
            if avg_recent_ip < 5.0 and final_projection > 6.0:
                original = final_projection
                final_projection *= 0.92
                print(f"   ⚠️  Short Recent Outings (avg {avg_recent_ip:.1f} IP) - Reducing: {original:.1f} → {final_projection:.1f} K")
                red_flags.append(f'Short outings (avg {avg_recent_ip:.1f} IP)')
            
            # Check 4: Elite K pitcher safety cap
            if base_k9 > 12.0 and final_projection > 9.0:
                original = final_projection
                final_projection = min(final_projection, 9.0)
                if original > final_projection:
                    print(f"   ⚠️  Elite K Safety Cap - Limiting: {original:.1f} → {final_projection:.1f} K")
        
        # Check 5: Global cap — no projection above 8.5 (even elite pitchers rarely avg above this)
        if final_projection > 8.5:
            original = final_projection
            final_projection = 8.5
            print(f"   ⚠️  Global Cap: {original:.1f} → {final_projection:.1f} K")
            red_flags.append(f'Capped from {original:.1f}')
        
        # 9. ML CORRECTION (if model available) — tighter cap of ±1.5
        ml_correction = 0.0
        if ml_corrector and ml_corrector.is_trained:
            ml_correction = ml_corrector.predict_correction({
                'season_k9': season_k9,
                'recent_k9': recent_k9,
                'expected_ip': expected_ip,
                'opponent_k_rate': opponent_k_rate,
                'is_home': starter['is_home'],
                'is_day_game': is_day_game,
                'is_short_rest': is_short_rest
            })
            ml_correction = float(np.clip(ml_correction, -1.5, 1.5))  # Tighter cap
            if abs(ml_correction) > 0.1:
                original = final_projection
                final_projection += ml_correction
                final_projection = max(final_projection, 0.5)
                # Re-apply global cap after ML correction
                final_projection = min(final_projection, 8.5)
                print(f"   🤖 ML Correction: {ml_correction:+.2f} K → {final_projection:.2f} K")
        
        # 10. CALCULATE PROBABILITIES (after all corrections)
        probabilities = {}
        for line in [3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5]:
            prob = 1 - scipy_stats.norm.cdf(line, final_projection, std_dev)
            probabilities[f'prob_{line}+'] = prob
            
            if prob > 0.10:
                print(f"   {line}+ K: {prob:.1%}")
        
        # 11. CONFIDENCE TIER
        confidence = 'HIGH'
        if len(red_flags) >= 2:
            confidence = 'LOW'
        elif len(red_flags) == 1 or blowup_rate >= 0.2:
            confidence = 'MEDIUM'
        elif std_dev > 2.8:
            confidence = 'MEDIUM'
        
        print(f"   Confidence: {confidence} | Red Flags: {', '.join(red_flags) if red_flags else 'None'}")
        
        # Store prediction
        pred = {
            'pitcher': pitcher_name,
            'team': starter['team'],
            'opponent': starter['opponent'],
            'is_home': starter['is_home'],
            'projection': round(final_projection, 1),
            'season_k9': round(season_k9, 2),
            'recent_k9': round(recent_k9, 2),
            'expected_ip': round(expected_ip, 1),
            'opponent_k_rate': round(opponent_k_rate, 3),
            'std_dev': round(std_dev, 2),
            'is_day_game': is_day_game,
            'is_short_rest': is_short_rest,
            'ml_correction': round(ml_correction, 2),
            'confidence': confidence,
            'red_flags': '; '.join(red_flags) if red_flags else 'None',
            'blowup_rate': round(blowup_rate, 2),
            **{k: round(v, 3) for k, v in probabilities.items()}
        }
        
        predictions.append(pred)
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        continue

# Save predictions
if predictions:
    df = pd.DataFrame(predictions)
    df = df.sort_values('projection', ascending=False)
    
    if target_date:
        date_str = target_date.replace('-', '')
    else:
        date_str = datetime.now().strftime('%Y%m%d')
    
    filename = f"predictions_strikeouts_{date_str}.csv"
    df.to_csv(filename, index=False)
    
    print("\n" + "=" * 80)
    print(f"✅ Predictions saved to: {filename}")
    print("=" * 80)
    
    print("\n📊 TOP PROJECTIONS:")
    for _, row in df.head(10).iterrows():
        conf_icon = '🟢' if row['confidence'] == 'HIGH' else '🟡' if row['confidence'] == 'MEDIUM' else '🔴'
        flags = f" ⚠️ {row['red_flags']}" if row['red_flags'] != 'None' else ''
        print(f"   {conf_icon} {row['pitcher']:25s} {row['projection']:.1f} K  ({row['team']} vs {row['opponent']}){flags}")
else:
    print("\n⚠️  No predictions generated")
