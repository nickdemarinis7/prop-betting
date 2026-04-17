"""
Validate Picks - Check how yesterday's predictions performed
Compares predictions to actual game results
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from datetime import datetime, timedelta
from shared.scrapers.gamelog import GameLogScraper
from nba_api.stats.static import players
import argparse
import numpy as np

print("=" * 80)
print("📊 PICKS VALIDATION - How Did We Do?")
print("=" * 80)

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--date', type=str, help='Date to validate (YYYYMMDD)', default=None)
args = parser.parse_args()

# Determine which predictions file to validate
if args.date:
    predictions_file = f"predictions_production_{args.date}.csv"
    date_str = args.date
else:
    # Default to yesterday
    yesterday = datetime.now() - timedelta(days=1)
    date_str = yesterday.strftime('%Y%m%d')
    predictions_file = f"predictions_production_{date_str}.csv"

print(f"\n📅 Validating predictions from: {date_str}")
print(f"📁 Loading: {predictions_file}")

# Load predictions
try:
    predictions = pd.read_csv(predictions_file)
    print(f"✓ Loaded {len(predictions)} predictions")
except FileNotFoundError:
    print(f"❌ File not found: {predictions_file}")
    print(f"\nAvailable prediction files:")
    import os
    for f in os.listdir('.'):
        if f.startswith('predictions_production_') and f.endswith('.csv'):
            print(f"   • {f}")
    sys.exit(1)

print()

# Initialize scraper
gamelog_scraper = GameLogScraper()
all_players = players.get_players()

# Get actual results for each player
results = []

print("🔄 Fetching actual game results...")
print()

for idx, row in predictions.iterrows():
    player_name = row['Player']
    projection = row['Proj']
    
    # Find player ID
    player_match = [p for p in all_players if p['full_name'] == player_name]
    if not player_match:
        print(f"⚠️  Could not find player: {player_name}")
        continue
    
    player_id = player_match[0]['id']
    
    # Get most recent game
    recent_games = gamelog_scraper.get_recent_games(player_id, n_games=1)
    
    if recent_games.empty:
        print(f"⚠️  No recent game found for: {player_name}")
        continue
    
    actual_points = recent_games['PTS'].iloc[0]
    game_date = recent_games['GAME_DATE'].iloc[0]
    
    # Calculate results
    hit_5 = actual_points >= 5
    hit_7 = actual_points >= 7
    hit_10 = actual_points >= 10
    
    diff = actual_points - projection
    
    results.append({
        'Player': player_name,
        'Type': row['Type'],
        'Projection': projection,
        'Actual': actual_points,
        'Diff': diff,
        'Hit_15+': hit_5,
        'Hit_20+': hit_7,
        'Hit_25+': hit_10,
        'Prob_15+': row['15+%'],
        'Prob_20+': row['20+%'],
        'Prob_25+': row['25+%'],
        'Ladder_Value': row['Ladder_Value'],
        'Game_Date': game_date
    })

if not results:
    print("❌ No results to validate")
    sys.exit(1)

results_df = pd.DataFrame(results)

print("=" * 80)
print("📊 RESULTS SUMMARY")
print("=" * 80)
print()

# Overall accuracy
top_plays = results_df[results_df['Type'] == 'TOP PLAY']
fades = results_df[results_df['Type'] == 'FADE']

print(f"TOP PLAYS ({len(top_plays)} players):")
print("-" * 80)

# Calculate hit rates
if len(top_plays) > 0:
    hit_rate_5 = (top_plays['Hit_15+'].sum() / len(top_plays)) * 100
    hit_rate_7 = (top_plays['Hit_20+'].sum() / len(top_plays)) * 100
    hit_rate_10 = (top_plays['Hit_25+'].sum() / len(top_plays)) * 100
    
    avg_prob_5 = top_plays['Prob_15+'].mean()
    avg_prob_7 = top_plays['Prob_20+'].mean()
    avg_prob_10 = top_plays['Prob_25+'].mean()
    
    print(f"15+ PTS Hit Rate:  {hit_rate_5:.0f}% (Expected: {avg_prob_5:.0f}%)")
    print(f"20+ PTS Hit Rate:  {hit_rate_7:.0f}% (Expected: {avg_prob_7:.0f}%)")
    print(f"25+ PTS Hit Rate: {hit_rate_10:.0f}% (Expected: {avg_prob_10:.0f}%)")
    print()
    
    # Projection accuracy
    mae = abs(top_plays['Diff']).mean()
    print(f"Mean Absolute Error: {mae:.2f} points")
    print()

# Show individual results
print("=" * 80)
print("🎯 INDIVIDUAL RESULTS")
print("=" * 80)
print()

# Sort by ladder value
results_df = results_df.sort_values('Ladder_Value', ascending=False)

for _, row in results_df.iterrows():
    player = row['Player']
    proj = row['Projection']
    actual = row['Actual']
    diff = row['Diff']
    
    # Result indicator
    if abs(diff) <= 1.5:
        result_icon = "✅"  # Very accurate
    elif abs(diff) <= 3.0:
        result_icon = "⚠️"   # Acceptable
    else:
        result_icon = "❌"  # Miss
    
    print(f"{result_icon} {player:25} | Proj: {proj:4.1f} | Actual: {actual:2.0f} | Diff: {diff:+5.1f}")
    
    # Show ladder results
    ladder_results = []
    if row['Hit_15+']:
        ladder_results.append(f"15+ ✅ ({row['Prob_15+']}%)")
    else:
        ladder_results.append(f"15+ ❌ ({row['Prob_15+']}%)")
    
    if row['Hit_20+']:
        ladder_results.append(f"20+ ✅ ({row['Prob_20+']}%)")
    else:
        ladder_results.append(f"20+ ❌ ({row['Prob_20+']}%)")
    
    if row['Hit_25+']:
        ladder_results.append(f"25+ ✅ ({row['Prob_25+']}%)")
    elif row['Prob_25+'] > 5:  # Only show if probability was meaningful
        ladder_results.append(f"25+ ❌ ({row['Prob_25+']}%)")
    
    print(f"   Ladder: {' | '.join(ladder_results)}")
    print()

# Save results
output_file = f"validation_results_{date_str}.csv"
results_df.to_csv(output_file, index=False)
print("=" * 80)
print(f"✅ Saved detailed results to: {output_file}")
print("=" * 80)
