"""
NBA Assists Ladder Strategy with Sportsbook Odds
Find the best value bets by comparing model projections to actual odds
"""

import pandas as pd
import numpy as np
import sys
import os
import requests
from datetime import datetime
from dotenv import load_dotenv
from scipy import stats as scipy_stats

# Load .env from project root
load_dotenv(os.path.join(os.path.dirname(__file__), '../../.env'))

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# Use shared OddsAPIScraper rather than a local duplicate.
from mlb.shared.scrapers.odds_api import (
    OddsAPIScraper,
    calculate_implied_probability,
    calculate_expected_value,
)


class _DeprecatedNBAOddsAPI:
    """Deprecated: kept for back-compat only. Use OddsAPIScraper."""
    
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('ODDS_API_KEY')
        self.base_url = "https://api.the-odds-api.com/v4"
        self.session = requests.Session()
    
    def get_all_assists_odds(self):
        """Get player assists odds for all today's NBA games"""
        if not self.api_key:
            print("   ⚠️  No API key - cannot fetch odds")
            return pd.DataFrame()
        
        try:
            # Step 1: Get all NBA events
            events_url = f"{self.base_url}/sports/basketball_nba/events"
            params = {'apiKey': self.api_key}
            
            events_response = self.session.get(events_url, params=params, timeout=10)
            events_response.raise_for_status()
            events = events_response.json()
            
            if not events:
                print("   ⚠️  No upcoming NBA games found")
                return pd.DataFrame()
            
            print(f"   📅 Found {len(events)} NBA games")
            
            # Step 2: Get assists odds for each event
            all_odds = []
            
            for i, event in enumerate(events[:15], 1):  # Limit to save API calls
                event_id = event['id']
                
                # Get odds for player assists
                odds_url = f"{self.base_url}/sports/basketball_nba/events/{event_id}/odds"
                params = {
                    'apiKey': self.api_key,
                    'regions': 'us',
                    'markets': 'player_assists',
                    'oddsFormat': 'american'
                }
                
                try:
                    odds_response = self.session.get(odds_url, params=params, timeout=10)
                    odds_response.raise_for_status()
                    
                    if i == 1:
                        remaining = odds_response.headers.get('x-requests-remaining')
                        if remaining:
                            print(f"   📊 Odds API requests remaining: {remaining}")
                    
                    odds_data = odds_response.json()
                    game_odds = self._parse_assists_odds(odds_data)
                    
                    if not game_odds.empty:
                        all_odds.append(game_odds)
                        print(f"   ✅ Game {i}: Found {len(game_odds)} assists odds")
                    
                except Exception as e:
                    print(f"   ⚠️  Game {i}: {str(e)[:50]}")
                    continue
            
            if all_odds:
                combined = pd.concat(all_odds, ignore_index=True)
                print(f"   ✅ Total: {len(combined)} assists odds from {len(all_odds)} games")
                return combined
            else:
                print("   ⚠️  No assists odds available")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"   ❌ Error fetching assists odds: {e}")
            return pd.DataFrame()
    
    def _parse_assists_odds(self, game_data):
        """Parse assists odds from API response"""
        odds_list = []
        
        if not game_data or 'bookmakers' not in game_data:
            return pd.DataFrame()
        
        for bookmaker in game_data['bookmakers']:
            book_name = bookmaker['title']
            
            for market in bookmaker.get('markets', []):
                if market['key'] != 'player_assists':
                    continue
                
                for outcome in market.get('outcomes', []):
                    player_name = outcome.get('description', '')
                    line = outcome.get('point')
                    odds = outcome.get('price')
                    over_under = outcome.get('name')
                    
                    odds_list.append({
                        'player': player_name,
                        'bookmaker': book_name,
                        'line': line,
                        'over_under': over_under,
                        'odds': odds
                    })
        
        return pd.DataFrame(odds_list)


# calculate_implied_probability and calculate_expected_value are imported from
# mlb.shared.scrapers.odds_api at the top of this file.


def analyze_nba_assists_value(predictions_file, api_key=None, min_odds=-200):
    """
    Find best NBA assists value bets
    
    Args:
        predictions_file: Path to predictions CSV
        api_key: The Odds API key
        min_odds: Minimum odds to consider (default -200)
    """
    print("=" * 80)
    print("🏀 NBA ASSISTS LADDER STRATEGY - WITH SPORTSBOOK ODDS")
    print("=" * 80)
    
    # Load predictions
    try:
        predictions = pd.read_csv(predictions_file)
        print(f"\n📊 Loaded {len(predictions)} player projections")
    except Exception as e:
        print(f"\n❌ Error loading predictions: {e}")
        return
    
    # Get sportsbook odds
    print("\n📡 Fetching sportsbook odds...")
    odds_api = OddsAPIScraper(api_key=api_key)
    odds_df = odds_api.get_all_nba_assists_odds()
    
    if odds_df.empty:
        print("\n⚠️  No sportsbook odds available")
        print("   Set ODDS_API_KEY environment variable")
        return
    
    # Find value bets
    value_bets = []
    near_misses = []  # Track close calls for debug
    matched_players = 0
    
    for _, pred in predictions.iterrows():
        player = pred['Player']
        projection = pred['Proj']
        std_dev = pred.get('StdDev', 2.5)  # Use player's actual std_dev
        calibrated_std = std_dev * 1.5  # Match the calibration in predict.py
        
        # Get odds for this player — try last name match
        last_name = player.split()[-1]
        player_odds = odds_df[
            odds_df['player'].str.contains(last_name, case=False, na=False)
        ]
        
        # If multiple matches (common last name), try full name
        if len(player_odds) > 20:
            first_name = player.split()[0]
            player_odds = player_odds[
                player_odds['player'].str.contains(first_name, case=False, na=False)
            ]
        
        if player_odds.empty:
            continue
        
        matched_players += 1
        
        # Get all unique lines the book offers for this player
        available_lines = player_odds['line'].dropna().unique()
        is_fade = pred.get('Type', '') == 'FADE'
        
        for line in sorted(available_lines):
            # Calculate our OVER probability dynamically for ANY line
            z_score = (line - projection) / calibrated_std if calibrated_std > 0 else 0
            over_prob = 1 - scipy_stats.norm.cdf(z_score)
            under_prob = 1 - over_prob
            
            # Cap 10+ AST probability (same as predict.py)
            if line >= 9.5:
                over_prob = min(over_prob, 0.25)
                under_prob = max(under_prob, 0.75)
            
            # --- Check OVER side ---
            if over_prob >= 0.05:
                over_odds = player_odds[
                    (player_odds['line'] == line) &
                    (player_odds['over_under'] == 'Over')
                ]
                if not over_odds.empty:
                    best_row = over_odds.loc[over_odds['odds'].idxmax()]
                    best_odds = best_row['odds']
                    implied = calculate_implied_probability(best_odds)
                    edge = over_prob - implied
                    ev = calculate_expected_value(over_prob, best_odds)
                    
                    bet_info = {
                        'player': pred['Player'], 'team': pred.get('Team', ''),
                        'opponent': pred.get('Opp', ''), 'projection': projection,
                        'line': line, 'side': 'OVER', 'our_prob': over_prob,
                        'book_odds': best_odds, 'bookmaker': best_row['bookmaker'],
                        'implied_prob': implied, 'edge': edge, 'ev': ev,
                        'tier': pred.get('Tier', ''),
                        'confidence': 'High' if edge > 0.10 else 'Medium'
                    }
                    if edge > 0.03 and best_odds >= min_odds:
                        value_bets.append(bet_info)
                    elif edge > 0.00 and best_odds >= min_odds:
                        near_misses.append(bet_info)
            
            # --- Check UNDER side (especially valuable for FADE plays) ---
            if under_prob >= 0.15:
                under_odds = player_odds[
                    (player_odds['line'] == line) &
                    (player_odds['over_under'] == 'Under')
                ]
                if not under_odds.empty:
                    best_row = under_odds.loc[under_odds['odds'].idxmax()]
                    best_odds = best_row['odds']
                    implied = calculate_implied_probability(best_odds)
                    edge = under_prob - implied
                    ev = calculate_expected_value(under_prob, best_odds)
                    
                    bet_info = {
                        'player': pred['Player'], 'team': pred.get('Team', ''),
                        'opponent': pred.get('Opp', ''), 'projection': projection,
                        'line': line, 'side': 'UNDER', 'our_prob': under_prob,
                        'book_odds': best_odds, 'bookmaker': best_row['bookmaker'],
                        'implied_prob': implied, 'edge': edge, 'ev': ev,
                        'tier': pred.get('Tier', ''),
                        'confidence': 'High' if edge > 0.10 else 'Medium'
                    }
                    if edge > 0.03 and best_odds >= min_odds:
                        value_bets.append(bet_info)
                    elif edge > 0.00 and best_odds >= min_odds:
                        near_misses.append(bet_info)
    
    print(f"\n🔍 Matched {matched_players}/{len(predictions)} players to sportsbook odds")
    
    if near_misses:
        print(f"\n📋 Near misses ({len(near_misses)} bets with 0-3% edge):")
        near_df = pd.DataFrame(near_misses).sort_values('edge', ascending=False)
        for _, bet in near_df.head(10).iterrows():
            side = bet.get('side', 'OVER')
            side_label = 'O' if side == 'OVER' else 'U'
            print(f"   {bet['player']:25s} {side_label}{bet['line']} | Our: {bet['our_prob']:.0%} vs Book: {bet['implied_prob']:.0%} | Edge: {bet['edge']:+.1%} @ {int(bet['book_odds']):+d} ({bet['bookmaker']})")
    
    if not value_bets:
        print("\n⚠️  No value bets found")
        return
    
    # Convert to DataFrame and sort by edge
    value_df = pd.DataFrame(value_bets).sort_values('edge', ascending=False)
    
    # Display results
    print("\n" + "=" * 80)
    print("🔥 TOP VALUE BETS (Sorted by Edge)")
    print("=" * 80)
    print(f"\nFilters: Edge >3% | Odds better than {min_odds}")
    print("These are bets where our model gives significantly higher probability")
    print("than the sportsbook's implied probability.\n")
    
    for i, (_, bet) in enumerate(value_df.head(15).iterrows(), 1):
        side = bet.get('side', 'OVER')
        side_icon = "📈" if side == 'OVER' else "📉"
        print(f"{i:2d}. {side_icon} {bet['player']:25s} {side} {bet['line']} Assists")
        print(f"    {bet['team']} vs {bet['opponent']} | Projection: {bet['projection']:.1f} AST")
        print(f"    📊 Our Probability: {bet['our_prob']:.1%}")
        print(f"    📖 {bet['bookmaker']}: {bet['book_odds']:+d} (Implied: {bet['implied_prob']:.1%})")
        print(f"    💰 EDGE: {bet['edge']:.1%} | EV: {bet['ev']:+.1%} | {bet['confidence']} Confidence")
        print()
    
    # Ladder opportunities
    print("=" * 80)
    print("🪜 LADDER OPPORTUNITIES")
    print("=" * 80)
    
    player_groups = value_df.groupby('player')
    
    ladder_count = 0
    for player, group in player_groups:
        if len(group) >= 2:
            ladder_count += 1
            
            print(f"\n{ladder_count}. {player}")
            print(f"   Projection: {group.iloc[0]['projection']:.1f} AST")
            print(f"   Ladder Strategy:")
            
            total_units = 0
            for _, bet in group.iterrows():
                units = 1.0 if bet['edge'] > 0.15 else 0.5
                total_units += units
                side = bet.get('side', 'OVER')
                print(f"   • {units:.1f}u on {side} {bet['line']} @ {bet['book_odds']:+d} ({bet['bookmaker']})")
                print(f"     Edge: {bet['edge']:.1%} | Our Prob: {bet['our_prob']:.1%}")
            
            print(f"   📊 Total Risk: {total_units:.1f} units")
            print()
    
    if ladder_count == 0:
        print("\n   No multi-line ladder opportunities")
        print("   Consider single bets from recommendations above")
    
    # Summary
    print("=" * 80)
    print("📈 SUMMARY")
    print("=" * 80)
    
    print(f"\nTotal Value Bets Found: {len(value_df)}")
    print(f"Average Edge: {value_df['edge'].mean():.1%}")
    print(f"Average EV: {value_df['ev'].mean():+.1%}")
    print(f"High Confidence Bets (>15% edge): {len(value_df[value_df['edge'] > 0.15])}")
    print(f"Medium Confidence Bets (5-15% edge): {len(value_df[(value_df['edge'] >= 0.05) & (value_df['edge'] <= 0.15)])}")


if __name__ == "__main__":
    from datetime import datetime
    
    # Get API key
    api_key = os.getenv('ODDS_API_KEY')
    
    if len(sys.argv) > 1:
        predictions_file = sys.argv[1]
    else:
        # Try to find most recent predictions file
        today = datetime.now().strftime('%Y%m%d')
        predictions_file = f"predictions_{today}.csv"
        
        if not os.path.exists(predictions_file):
            # Find most recent file
            import glob
            files = glob.glob("predictions_*.csv")
            if files:
                predictions_file = max(files, key=os.path.getctime)
                print(f"Using most recent predictions: {predictions_file}")
    
    if not os.path.exists(predictions_file):
        print(f"❌ Predictions file not found: {predictions_file}")
        print(f"   Run predict.py first to generate predictions")
        sys.exit(1)
    
    analyze_nba_assists_value(predictions_file, api_key=api_key, min_odds=-300)
