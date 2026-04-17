"""
MLB Actual Lineup Scraper
Get actual game-day lineups from MLB Stats API
"""

import requests
import pandas as pd
from datetime import datetime


class MLBLineupScraper:
    """Scrape actual lineups from MLB Stats API"""
    
    def __init__(self):
        self.base_url = "https://statsapi.mlb.com/api/v1.1"
        self.session = requests.Session()
    
    def get_actual_lineup(self, game_id):
        """
        Get actual starting lineup for a game
        
        Args:
            game_id: MLB game ID
            
        Returns:
            dict with 'away' and 'home' lineups (list of player names)
        """
        try:
            url = f"{self.base_url}/game/{game_id}/feed/live"
            
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            lineups = {
                'away': [],
                'home': []
            }
            
            # Get lineups from boxscore
            boxscore = data.get('liveData', {}).get('boxscore', {})
            
            # Away lineup
            away_batters = boxscore.get('teams', {}).get('away', {}).get('batters', [])
            away_players = boxscore.get('teams', {}).get('away', {}).get('players', {})
            
            for batter_id in away_batters[:9]:  # First 9 are starters
                player_key = f'ID{batter_id}'
                if player_key in away_players:
                    player_name = away_players[player_key].get('person', {}).get('fullName')
                    if player_name:
                        lineups['away'].append(player_name)
            
            # Home lineup
            home_batters = boxscore.get('teams', {}).get('home', {}).get('batters', [])
            home_players = boxscore.get('teams', {}).get('home', {}).get('players', {})
            
            for batter_id in home_batters[:9]:  # First 9 are starters
                player_key = f'ID{batter_id}'
                if player_key in home_players:
                    player_name = home_players[player_key].get('person', {}).get('fullName')
                    if player_name:
                        lineups['home'].append(player_name)
            
            if lineups['away'] or lineups['home']:
                return lineups
            else:
                return None
                
        except Exception as e:
            # Lineup not available yet
            return None
    
    def get_lineup_for_team(self, game_id, team_type='away'):
        """
        Get lineup for specific team
        
        Args:
            game_id: MLB game ID
            team_type: 'away' or 'home'
            
        Returns:
            List of player names in batting order
        """
        lineups = self.get_actual_lineup(game_id)
        
        if lineups:
            return lineups.get(team_type, [])
        else:
            return []


if __name__ == "__main__":
    # Test with a recent game
    scraper = MLBLineupScraper()
    
    # Example game ID (you'd get this from schedule)
    test_game_id = 746729  # Example
    
    print("Testing MLB Lineup Scraper")
    print("=" * 60)
    
    lineups = scraper.get_actual_lineup(test_game_id)
    
    if lineups:
        print("\nAway Lineup:")
        for i, player in enumerate(lineups['away'], 1):
            print(f"  {i}. {player}")
        
        print("\nHome Lineup:")
        for i, player in enumerate(lineups['home'], 1):
            print(f"  {i}. {player}")
    else:
        print("\nLineup not available yet")
