"""
NBA Assists Ladder Strategy with Sportsbook Odds
Find the best value bets by comparing model projections to actual odds
"""

import pandas as pd
import sys
import os
import requests
from datetime import datetime

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))


class NBAOddsAPI:
    """Fetch NBA player props odds from The Odds API"""
    
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


def calculate_implied_probability(american_odds):
    """Convert American odds to implied probability"""
    if american_odds < 0:
        return abs(american_odds) / (abs(american_odds) + 100)
    else:
        return 100 / (american_odds + 100)


def calculate_expected_value(our_prob, american_odds):
    """Calculate expected value of a bet"""
    if american_odds < 0:
        decimal_odds = 1 + (100 / abs(american_odds))
    else:
        decimal_odds = 1 + (american_odds / 100)
    
    ev = our_prob * decimal_odds - 1
    return ev


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
    odds_api = NBAOddsAPI(api_key=api_key)
    odds_df = odds_api.get_all_assists_odds()
    
    if odds_df.empty:
        print("\n⚠️  No sportsbook odds available")
        print("   Set ODDS_API_KEY environment variable")
        return
    
    # Find value bets
    value_bets = []
    
    for _, pred in predictions.iterrows():
        player = pred['Player']
        
        # Get odds for this player
        player_odds = odds_df[
            odds_df['player'].str.contains(player.split()[-1], case=False, na=False)
        ]
        
        if player_odds.empty:
            continue
        
        # Check common assist lines (4.5, 5.5, 6.5, 7.5, 8.5, 9.5)
        for line in [4.5, 5.5, 6.5, 7.5, 8.5, 9.5]:
            # Get our probability for this line
            # Map to column names: 5+%, 7+%, 10+%
            if line == 4.5:
                prob_col = '5+%'
            elif line == 6.5:
                prob_col = '7+%'
            elif line == 9.5:
                prob_col = '10+%'
            else:
                # Interpolate for other lines
                continue
            
            if prob_col not in pred or pd.isna(pred[prob_col]):
                continue
            
            our_prob = pred[prob_col] / 100  # Convert from percentage
            
            # Get best OVER odds for this line
            line_odds = player_odds[
                (player_odds['line'] == line) &
                (player_odds['over_under'] == 'Over')
            ]
            
            if line_odds.empty:
                continue
            
            # Find best odds across bookmakers
            best_odds_row = line_odds.loc[line_odds['odds'].idxmax()]
            best_odds = best_odds_row['odds']
            bookmaker = best_odds_row['bookmaker']
            
            # Calculate edge
            implied_prob = calculate_implied_probability(best_odds)
            ev = calculate_expected_value(our_prob, best_odds)
            edge = our_prob - implied_prob
            
            # Filter: edge >5% and odds better than min_odds
            if edge > 0.05 and best_odds > min_odds:
                value_bets.append({
                    'player': pred['Player'],
                    'team': pred.get('Team', ''),
                    'opponent': pred.get('Opp', ''),
                    'projection': pred['Proj'],
                    'line': line,
                    'our_prob': our_prob,
                    'book_odds': best_odds,
                    'bookmaker': bookmaker,
                    'implied_prob': implied_prob,
                    'edge': edge,
                    'ev': ev,
                    'tier': pred.get('Tier', ''),
                    'confidence': 'High' if edge > 0.15 else 'Medium'
                })
    
    if not value_bets:
        print("\n⚠️  No value bets found")
        return
    
    # Convert to DataFrame and sort by edge
    value_df = pd.DataFrame(value_bets).sort_values('edge', ascending=False)
    
    # Display results
    print("\n" + "=" * 80)
    print("🔥 TOP VALUE BETS (Sorted by Edge)")
    print("=" * 80)
    print(f"\nFilters: Edge >5% | Odds better than {min_odds}")
    print("These are bets where our model gives significantly higher probability")
    print("than the sportsbook's implied probability.\n")
    
    for i, (_, bet) in enumerate(value_df.head(15).iterrows(), 1):
        print(f"{i:2d}. {bet['player']:25s} OVER {bet['line']} Assists")
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
                print(f"   • {units:.1f}u on OVER {bet['line']} @ {bet['book_odds']:+d} ({bet['bookmaker']})")
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
    
    analyze_nba_assists_value(predictions_file, api_key=api_key, min_odds=-200)
