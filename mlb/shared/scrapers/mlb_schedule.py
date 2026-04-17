"""
MLB Schedule Scraper
Fetches today's games and probable starters
"""

import pandas as pd
import requests
from datetime import datetime


class MLBScheduleScraper:
    """Scrape MLB schedule and probable starters"""
    
    def __init__(self):
        self.base_url = "https://statsapi.mlb.com/api/v1"
    
    def get_todays_games(self, date=None):
        """
        Get today's MLB games from MLB Stats API
        
        Args:
            date: Date string (YYYY-MM-DD), defaults to today
            
        Returns:
            DataFrame with game info
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            url = f"{self.base_url}/schedule"
            params = {
                'sportId': 1,  # MLB
                'date': date,
                'hydrate': 'probablePitcher,team'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            games = []
            for date_data in data.get('dates', []):
                for game in date_data.get('games', []):
                    # Skip non-regular season games
                    if game.get('gameType') != 'R':
                        continue
                    
                    away_team = game['teams']['away']['team']
                    home_team = game['teams']['home']['team']
                    
                    # Get probable pitchers
                    away_pitcher_info = game['teams']['away'].get('probablePitcher', {})
                    home_pitcher_info = game['teams']['home'].get('probablePitcher', {})
                    
                    games.append({
                        'game_id': game['gamePk'],
                        'game_time': game.get('gameDate', ''),
                        'away_team': away_team.get('abbreviation', 'UNK'),
                        'away_team_id': away_team.get('id'),
                        'home_team': home_team.get('abbreviation', 'UNK'),
                        'home_team_id': home_team.get('id'),
                        'away_pitcher': away_pitcher_info.get('fullName'),
                        'away_pitcher_id': away_pitcher_info.get('id'),
                        'home_pitcher': home_pitcher_info.get('fullName'),
                        'home_pitcher_id': home_pitcher_info.get('id'),
                        'venue': game.get('venue', {}).get('name', 'Unknown')
                    })
            
            if games:
                return pd.DataFrame(games)
            else:
                print(f"   ⚠️  No games found for {date}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"   ❌ Error fetching schedule: {e}")
            print(f"   Using fallback sample data...")
            return self._get_sample_schedule()
    
    def get_probable_starters(self, game_id):
        """
        Get probable starters for a specific game
        
        Args:
            game_id: MLB game ID
            
        Returns:
            Dict with away and home probable pitchers
        """
        # Placeholder
        return {
            'away_pitcher': 'Sample Pitcher',
            'away_pitcher_id': 12345,
            'home_pitcher': 'Sample Pitcher 2',
            'home_pitcher_id': 67890
        }
    
    def get_team_schedule(self, team_abbr, n_games=10):
        """
        Get recent/upcoming schedule for a team
        
        Args:
            team_abbr: Team abbreviation (e.g., 'NYY')
            n_games: Number of games to fetch
            
        Returns:
            DataFrame with team schedule
        """
        # Placeholder
        sample_schedule = []
        for i in range(n_games):
            game = {
                'date': f'2026-04-{20-i:02d}',
                'opponent': ['BOS', 'TB', 'TOR', 'BAL', 'CLE'][i % 5],
                'is_home': i % 2,
                'result': 'W' if i % 3 else 'L',
                'score': f'{4+i%3}-{2+i%2}'
            }
            sample_schedule.append(game)
        
        return pd.DataFrame(sample_schedule)
    
    def _get_sample_schedule(self):
        """Fallback sample schedule for testing"""
        sample_games = [
            {
                'game_id': 1,
                'game_time': '19:10',
                'away_team': 'NYY',
                'home_team': 'BOS',
                'away_pitcher': 'Gerrit Cole',
                'away_pitcher_id': 543037,
                'home_pitcher': 'Chris Sale',
                'home_pitcher_id': 519242,
                'venue': 'Fenway Park'
            },
            {
                'game_id': 2,
                'game_time': '20:10',
                'away_team': 'LAD',
                'home_team': 'SF',
                'away_pitcher': 'Tyler Glasnow',
                'away_pitcher_id': 607192,
                'home_pitcher': 'Logan Webb',
                'home_pitcher_id': 657277,
                'venue': 'Oracle Park'
            },
            {
                'game_id': 3,
                'game_time': '19:40',
                'away_team': 'HOU',
                'home_team': 'TEX',
                'away_pitcher': 'Framber Valdez',
                'away_pitcher_id': 664285,
                'home_pitcher': 'Nathan Eovaldi',
                'home_pitcher_id': 543135,
                'venue': 'Globe Life Field'
            }
        ]
        return pd.DataFrame(sample_games)


# Example usage
if __name__ == "__main__":
    scraper = MLBScheduleScraper()
    
    # Get today's games
    games = scraper.get_todays_games()
    print("Today's games:")
    print(games)
    
    # Get probable starters
    starters = scraper.get_probable_starters(game_id=1)
    print(f"\nProbable starters:")
    print(starters)
