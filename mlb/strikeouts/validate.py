"""
⚾ MLB Strikeout Prediction Validation
Validates predictions against actual game results
"""

import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mlb.shared.scrapers.pitcher_stats import PitcherStatsScraper

print("=" * 80)
print("⚾ MLB STRIKEOUT PREDICTION VALIDATION")
print("=" * 80)

# Get date to validate (default to yesterday)
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--date', type=str, help='Date to validate (YYYYMMDD)', default=None)
args = parser.parse_args()

if args.date:
    validate_date = datetime.strptime(args.date, '%Y%m%d')
    date_str = args.date
else:
    validate_date = datetime.now() - timedelta(days=1)
    date_str = validate_date.strftime('%Y%m%d')

print(f"\n📅 Validating predictions for: {validate_date.strftime('%B %d, %Y')}")

# Load predictions file
predictions_file = f"predictions_strikeouts_{date_str}.csv"
    
if not os.path.exists(predictions_file):
    print(f"\n❌ No predictions file found for {date_str}")
    print("   Available files:")
    for f in os.listdir('.'):
        if f.startswith('predictions_strikeouts') and f.endswith('.csv'):
            print(f"      {f}")
    sys.exit(1)

print(f"   ✓ Loading predictions from: {predictions_file}")
predictions_df = pd.read_csv(predictions_file)

print(f"   ✓ Found {len(predictions_df)} predictions")

# Initialize scraper
scraper = PitcherStatsScraper()

# Get actual results for each pitcher
print(f"\n📊 Fetching actual game results...")

results = []
errors = []

for idx, row in predictions_df.iterrows():
    pitcher_name = row['pitcher']
    team = row['team']
    projection = row['projection']
    
    print(f"\n   Checking {pitcher_name} ({team})...")
    
    # Get actual game results from MLB API directly
    try:
        import requests
        
        # Get schedule for validation date
        schedule_url = f"https://statsapi.mlb.com/api/v1/schedule"
        params = {
            'sportId': 1,
            'date': validate_date.strftime('%Y-%m-%d')
        }
        
        schedule_response = requests.get(schedule_url, params=params)
        if schedule_response.status_code != 200:
            print(f"      ⚠️  Could not fetch schedule")
            continue
        
        schedule_data = schedule_response.json()
        
        if not schedule_data.get('dates'):
            print(f"      ℹ️  No games on {validate_date.strftime('%Y-%m-%d')}")
            continue
        
        games = schedule_data['dates'][0].get('games', [])
        
        # Find game for this team
        actual_k = None
        for game in games:
            # Get team IDs and fetch abbreviations
            away_team_id = game['teams']['away']['team']['id']
            home_team_id = game['teams']['home']['team']['id']
            
            # Fetch team details to get abbreviations
            away_team_response = requests.get(f"https://statsapi.mlb.com/api/v1/teams/{away_team_id}")
            home_team_response = requests.get(f"https://statsapi.mlb.com/api/v1/teams/{home_team_id}")
            
            away_team = away_team_response.json()['teams'][0]['abbreviation'] if away_team_response.status_code == 200 else None
            home_team = home_team_response.json()['teams'][0]['abbreviation'] if home_team_response.status_code == 200 else None
            
            if not away_team or not home_team:
                continue
            
            if team not in [away_team, home_team]:
                continue
            
            # Get box score for pitcher stats
            game_pk = game['gamePk']
            boxscore_url = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"
            
            box_response = requests.get(boxscore_url)
            if box_response.status_code != 200:
                continue
            
            box_data = box_response.json()
            
            # Check both teams' pitchers
            for team_side in ['away', 'home']:
                pitchers = box_data['teams'][team_side].get('pitchers', [])
                players = box_data['teams'][team_side].get('players', {})
                
                for pitcher_id in pitchers:
                    player_key = f'ID{pitcher_id}'
                    if player_key not in players:
                        continue
                    
                    player = players[player_key]
                    player_name_full = player['person']['fullName']
                    
                    # Match pitcher name (check last name)
                    if pitcher_name.split()[-1].lower() in player_name_full.lower():
                        stats = player.get('stats', {}).get('pitching', {})
                        actual_k = stats.get('strikeOuts', 0)
                        break
                
                if actual_k is not None:
                    break
            
            if actual_k is not None:
                break
        
        if actual_k is None:
            print(f"      ℹ️  Pitcher did not start or game not found")
            continue
        print(f"      ✓ Actual: {actual_k} K's | Projected: {projection:.1f} K's")
        
        # Calculate error
        error = actual_k - projection
        abs_error = abs(error)
        
        # Check ladder results
        ladder_results = {}
        for threshold in [4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5]:
            col_name = f'prob_{threshold:.1f}+'
            if col_name in row:
                prob = row[col_name]
                hit = 1 if actual_k >= threshold else 0
                ladder_results[f'{threshold:.1f}+'] = {
                    'prob': prob,
                    'hit': hit,
                    'actual': actual_k
                }
        
        results.append({
            'pitcher': pitcher_name,
            'team': team,
            'projection': projection,
            'actual': actual_k,
            'error': error,
            'abs_error': abs_error,
            'ladder_results': ladder_results
        })
        
    except Exception as e:
        print(f"      ❌ Error: {e}")
        errors.append(pitcher_name)

# Create results dataframe
if not results:
    print("\n❌ No results to validate")
    sys.exit(1)

results_df = pd.DataFrame(results)

print("\n" + "=" * 80)
print("📊 VALIDATION RESULTS")
print("=" * 80)

# Overall metrics
mae = results_df['abs_error'].mean()
rmse = np.sqrt((results_df['error'] ** 2).mean())
mean_error = results_df['error'].mean()
median_error = results_df['error'].median()

print(f"\n🎯 Overall Performance:")
print(f"   Predictions validated: {len(results_df)}")
print(f"   Mean Absolute Error (MAE): {mae:.2f} strikeouts")
print(f"   Root Mean Squared Error (RMSE): {rmse:.2f} strikeouts")
print(f"   Mean Error (bias): {mean_error:+.2f} strikeouts")
print(f"   Median Error: {median_error:+.2f} strikeouts")

# Accuracy buckets
within_1 = (results_df['abs_error'] <= 1).sum()
within_2 = (results_df['abs_error'] <= 2).sum()
within_3 = (results_df['abs_error'] <= 3).sum()

print(f"\n📏 Accuracy Distribution:")
print(f"   Within 1 K: {within_1}/{len(results_df)} ({within_1/len(results_df)*100:.1f}%)")
print(f"   Within 2 K: {within_2}/{len(results_df)} ({within_2/len(results_df)*100:.1f}%)")
print(f"   Within 3 K: {within_3}/{len(results_df)} ({within_3/len(results_df)*100:.1f}%)")

# Ladder performance
print(f"\n🎯 Ladder Betting Performance:")

all_ladder_results = []
for _, row in results_df.iterrows():
    for threshold, data in row['ladder_results'].items():
        all_ladder_results.append({
            'threshold': threshold,
            'prob': data['prob'],
            'hit': data['hit'],
            'actual': data['actual']
        })

if all_ladder_results:
    ladder_df = pd.DataFrame(all_ladder_results)
    
    for threshold in sorted(ladder_df['threshold'].unique()):
        threshold_data = ladder_df[ladder_df['threshold'] == threshold]
        hit_rate = threshold_data['hit'].mean()
        avg_prob = threshold_data['prob'].mean()
        count = len(threshold_data)
        
        print(f"   {threshold:>5s}: {hit_rate:>5.1%} hit rate | {avg_prob:>5.1%} avg prob | {count} bets")

# Individual results
print(f"\n📋 Individual Results:")
print("-" * 80)

results_df_sorted = results_df.sort_values('abs_error', ascending=False)

for idx, row in results_df_sorted.head(10).iterrows():
    error_icon = "✅" if row['abs_error'] <= 2 else "⚠️" if row['abs_error'] <= 3 else "❌"
    print(f"{error_icon} {row['pitcher']:25s} | Proj: {row['projection']:4.1f} | Actual: {row['actual']:2.0f} | Error: {row['error']:+4.1f}")

if len(results_df) > 10:
    print(f"   ... and {len(results_df) - 10} more")

# Save validation results
output_file = f"validation_results_{date_str}.csv"
results_df.drop('ladder_results', axis=1).to_csv(output_file, index=False)
print(f"\n✅ Validation results saved to: {output_file}")

print("\n" + "=" * 80)
print("💡 Insights:")

if mean_error > 1:
    print("   ⚠️  Model is over-projecting (predicting too high)")
elif mean_error < -1:
    print("   ⚠️  Model is under-projecting (predicting too low)")
else:
    print("   ✅ Model bias is minimal")

if mae > 2.5:
    print("   ⚠️  High error rate - consider model improvements")
elif mae < 1.5:
    print("   ✅ Excellent accuracy!")
else:
    print("   ✅ Good accuracy")

print("=" * 80)
