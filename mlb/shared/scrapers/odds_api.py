"""
The Odds API Scraper
Fetch sportsbook odds for MLB pitcher strikeouts

API Docs: https://the-odds-api.com/
Free tier: 500 requests/month
"""

import requests
import pandas as pd
from datetime import datetime
import os


class OddsAPIScraper:
    """Scrape sportsbook odds from The Odds API"""
    
    def __init__(self, api_key=None):
        """
        Initialize The Odds API scraper
        
        Args:
            api_key: The Odds API key (get free key at the-odds-api.com)
                    If not provided, will look for ODDS_API_KEY env variable
        """
        self.api_key = api_key or os.getenv('ODDS_API_KEY')
        if not self.api_key:
            print("⚠️  No Odds API key found. Set ODDS_API_KEY environment variable")
            print("   Get a free key at: https://the-odds-api.com/")
        
        self.base_url = "https://api.the-odds-api.com/v4"
        self.session = requests.Session()
    
    def get_mlb_games(self):
        """
        Get list of upcoming MLB games
        
        Returns:
            List of game IDs and matchups
        """
        if not self.api_key:
            return []
        
        try:
            url = f"{self.base_url}/sports/baseball_mlb/odds"
            params = {
                'apiKey': self.api_key,
                'regions': 'us',
                'markets': 'h2h',  # Just to get game list
                'oddsFormat': 'american'
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            games = response.json()
            
            # Check remaining quota
            remaining = response.headers.get('x-requests-remaining')
            if remaining:
                print(f"   📊 Odds API requests remaining: {remaining}")
            
            return games
            
        except Exception as e:
            print(f"   ❌ Error fetching MLB games: {e}")
            return []
    
    def get_pitcher_strikeout_odds(self, game_id=None):
        """
        Get pitcher strikeout odds for MLB games
        
        Args:
            game_id: Specific game ID (optional, gets all if None)
        
        Returns:
            DataFrame with pitcher strikeout odds
        """
        if not self.api_key:
            print("   ⚠️  No API key - cannot fetch odds")
            return pd.DataFrame()
        
        try:
            # Get events (games) first
            events_url = f"{self.base_url}/sports/baseball_mlb/events"
            params = {'apiKey': self.api_key}
            
            events_response = self.session.get(events_url, params=params, timeout=10)
            events_response.raise_for_status()
            events = events_response.json()
            
            if not events:
                print("   ⚠️  No upcoming MLB games found")
                return pd.DataFrame()
            
            # Get odds for player props (pitcher strikeouts)
            odds_url = f"{self.base_url}/sports/baseball_mlb/events/{game_id or events[0]['id']}/odds"
            params = {
                'apiKey': self.api_key,
                'regions': 'us',
                'markets': 'pitcher_strikeouts',
                'oddsFormat': 'american'
            }
            
            odds_response = self.session.get(odds_url, params=params, timeout=10)
            odds_response.raise_for_status()
            
            # Check remaining quota
            remaining = odds_response.headers.get('x-requests-remaining')
            if remaining:
                print(f"   📊 Odds API requests remaining: {remaining}")
            
            odds_data = odds_response.json()
            
            # Parse odds into DataFrame
            return self._parse_strikeout_odds(odds_data)
            
        except Exception as e:
            print(f"   ❌ Error fetching strikeout odds: {e}")
            return pd.DataFrame()
    
    def get_all_strikeout_odds(self):
        """
        Get pitcher strikeout odds for all today's MLB games
        
        Returns:
            DataFrame with all pitcher strikeout odds
        """
        if not self.api_key:
            print("   ⚠️  No API key - using sample data")
            return self._get_sample_odds()
        
        try:
            # Step 1: Get all events (games)
            events_url = f"{self.base_url}/sports/baseball_mlb/events"
            params = {'apiKey': self.api_key}
            
            events_response = self.session.get(events_url, params=params, timeout=10)
            events_response.raise_for_status()
            events = events_response.json()
            
            if not events:
                print("   ⚠️  No upcoming MLB games found")
                return pd.DataFrame()
            
            print(f"   📅 Found {len(events)} MLB games")
            
            # Step 2: Get strikeout odds for each event
            all_odds = []
            
            for i, event in enumerate(events[:10], 1):  # Limit to first 10 games to save API calls
                event_id = event['id']
                
                # Get odds for this specific event with pitcher_strikeouts market
                odds_url = f"{self.base_url}/sports/baseball_mlb/events/{event_id}/odds"
                params = {
                    'apiKey': self.api_key,
                    'regions': 'us',
                    'markets': 'pitcher_strikeouts',
                    'oddsFormat': 'american'
                }
                
                try:
                    odds_response = self.session.get(odds_url, params=params, timeout=10)
                    odds_response.raise_for_status()
                    
                    # Check remaining quota
                    if i == 1:  # Only print once
                        remaining = odds_response.headers.get('x-requests-remaining')
                        if remaining:
                            print(f"   📊 Odds API requests remaining: {remaining}")
                    
                    odds_data = odds_response.json()
                    
                    # Parse odds for this game
                    game_odds = self._parse_strikeout_odds(odds_data)
                    if not game_odds.empty:
                        all_odds.append(game_odds)
                        print(f"   ✅ Game {i}: Found {len(game_odds)} strikeout odds")
                    
                except Exception as e:
                    print(f"   ⚠️  Game {i}: {str(e)[:50]}")
                    continue
            
            if all_odds:
                combined = pd.concat(all_odds, ignore_index=True)
                print(f"   ✅ Total: {len(combined)} strikeout odds from {len(all_odds)} games")
                return combined
            else:
                print("   ⚠️  No strikeout odds available")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"   ❌ Error fetching strikeout odds: {e}")
            return pd.DataFrame()
    
    def _parse_strikeout_odds(self, game_data):
        """Parse strikeout odds from API response"""
        odds_list = []
        
        if not game_data or 'bookmakers' not in game_data:
            return pd.DataFrame()
        
        home_team = game_data.get('home_team', 'Unknown')
        away_team = game_data.get('away_team', 'Unknown')
        
        for bookmaker in game_data['bookmakers']:
            book_name = bookmaker['title']
            
            for market in bookmaker.get('markets', []):
                if market['key'] != 'pitcher_strikeouts':
                    continue
                
                for outcome in market.get('outcomes', []):
                    pitcher_name = outcome.get('description', '')
                    line = outcome.get('point')
                    odds = outcome.get('price')
                    over_under = outcome.get('name')  # 'Over' or 'Under'
                    
                    odds_list.append({
                        'pitcher': pitcher_name,
                        'home_team': home_team,
                        'away_team': away_team,
                        'bookmaker': book_name,
                        'line': line,
                        'over_under': over_under,
                        'odds': odds
                    })
        
        return pd.DataFrame(odds_list)
    
    def get_best_odds(self, pitcher_name, line, over_under='Over'):
        """
        Get best available odds for a specific pitcher/line
        
        Args:
            pitcher_name: Pitcher name
            line: K line (e.g., 6.5)
            over_under: 'Over' or 'Under'
        
        Returns:
            Dict with best odds and bookmaker
        """
        all_odds = self.get_all_strikeout_odds()
        
        if all_odds.empty:
            return None
        
        # Filter to specific pitcher and line
        filtered = all_odds[
            (all_odds['pitcher'].str.contains(pitcher_name, case=False, na=False)) &
            (all_odds['line'] == line) &
            (all_odds['over_under'] == over_under)
        ]
        
        if filtered.empty:
            return None
        
        # Find best odds (highest for Over, lowest for Under)
        if over_under == 'Over':
            best = filtered.loc[filtered['odds'].idxmax()]
        else:
            best = filtered.loc[filtered['odds'].idxmin()]
        
        return {
            'bookmaker': best['bookmaker'],
            'odds': best['odds'],
            'line': best['line']
        }
    
    def _get_sample_odds(self):
        """Sample odds data for testing without API key"""
        return pd.DataFrame([
            {'pitcher': 'Jacob Misiorowski', 'line': 6.5, 'over_under': 'Over', 'odds': -110, 'bookmaker': 'DraftKings'},
            {'pitcher': 'Jacob Misiorowski', 'line': 7.5, 'over_under': 'Over', 'odds': +120, 'bookmaker': 'DraftKings'},
            {'pitcher': 'Bryan Woo', 'line': 6.5, 'over_under': 'Over', 'odds': -115, 'bookmaker': 'FanDuel'},
            {'pitcher': 'Bryan Woo', 'line': 7.5, 'over_under': 'Over', 'odds': +110, 'bookmaker': 'FanDuel'},
        ])


def calculate_implied_probability(american_odds):
    """
    Convert American odds to implied probability
    
    Args:
        american_odds: American odds (e.g., -110, +120)
    
    Returns:
        Implied probability (0-1)
    """
    if american_odds < 0:
        return abs(american_odds) / (abs(american_odds) + 100)
    else:
        return 100 / (american_odds + 100)


def calculate_expected_value(our_prob, american_odds):
    """
    Calculate expected value of a bet
    
    Args:
        our_prob: Our model's probability (0-1)
        american_odds: Sportsbook odds
    
    Returns:
        Expected value as percentage
    """
    if american_odds < 0:
        decimal_odds = 1 + (100 / abs(american_odds))
    else:
        decimal_odds = 1 + (american_odds / 100)
    
    ev = our_prob * decimal_odds - 1
    return ev


if __name__ == "__main__":
    # Test the scraper
    scraper = OddsAPIScraper()
    
    print("Testing Odds API Scraper...")
    print("=" * 60)
    
    # Try to get odds
    odds = scraper.get_all_strikeout_odds()
    
    if not odds.empty:
        print(f"\n✅ Found {len(odds)} odds entries")
        print("\nSample odds:")
        print(odds.head(10))
    else:
        print("\n⚠️  No odds data available (need API key)")
        print("Set ODDS_API_KEY environment variable")
        print("Get free key at: https://the-odds-api.com/")
