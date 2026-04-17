"""
NBA.com data scraper for player statistics and game schedules
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import NBA_HEADERS, NBA_SCHEDULE_URL, NBA_PLAYER_STATS_URL, NBA_TRACKING_URL


class NBAStatsScraper:
    """Scraper for NBA.com statistics"""
    
    def __init__(self):
        self.headers = NBA_HEADERS
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def get_todays_games(self, game_date=None):
        """
        Fetch today's NBA game schedule
        
        Args:
            game_date: Date string in format 'YYYY-MM-DD' (defaults to today)
        
        Returns:
            DataFrame with game information
        """
        if game_date is None:
            game_date = datetime.now().strftime('%Y-%m-%d')
        
        params = {
            'GameDate': game_date,
            'LeagueID': '00',
            'DayOffset': '0'
        }
        
        try:
            response = self.session.get(NBA_SCHEDULE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            games = []
            if 'resultSets' in data and len(data['resultSets']) > 0:
                headers = data['resultSets'][0]['headers']
                rows = data['resultSets'][0]['rowSet']
                
                for row in rows:
                    game_dict = dict(zip(headers, row))
                    games.append(game_dict)
            
            return pd.DataFrame(games)
        
        except Exception as e:
            print(f"Error fetching today's games: {e}")
            return pd.DataFrame()
    
    def get_playing_teams(self, game_date=None):
        """
        Get list of team IDs playing today
        
        Returns:
            List of team IDs
        """
        games_df = self.get_todays_games(game_date)
        
        if games_df.empty:
            return []
        
        teams = []
        if 'HOME_TEAM_ID' in games_df.columns and 'VISITOR_TEAM_ID' in games_df.columns:
            teams.extend(games_df['HOME_TEAM_ID'].tolist())
            teams.extend(games_df['VISITOR_TEAM_ID'].tolist())
        
        return list(set(teams))
    
    def get_player_stats(self, season='2025-26', season_type='Regular Season'):
        """
        Fetch player statistics including traditional stats
        
        Args:
            season: NBA season (e.g., '2025-26')
            season_type: 'Regular Season' or 'Playoffs'
        
        Returns:
            DataFrame with player statistics
        """
        params = {
            'College': '',
            'Conference': '',
            'Country': '',
            'DateFrom': '',
            'DateTo': '',
            'Division': '',
            'DraftPick': '',
            'DraftYear': '',
            'GameScope': '',
            'GameSegment': '',
            'Height': '',
            'LastNGames': '0',
            'LeagueID': '00',
            'Location': '',
            'MeasureType': 'Base',
            'Month': '0',
            'OpponentTeamID': '0',
            'Outcome': '',
            'PORound': '0',
            'PaceAdjust': 'N',
            'PerMode': 'PerGame',
            'Period': '0',
            'PlayerExperience': '',
            'PlayerPosition': '',
            'PlusMinus': 'N',
            'Rank': 'N',
            'Season': season,
            'SeasonSegment': '',
            'SeasonType': season_type,
            'ShotClockRange': '',
            'StarterBench': '',
            'TeamID': '0',
            'TwoWay': '0',
            'VsConference': '',
            'VsDivision': '',
            'Weight': ''
        }
        
        try:
            time.sleep(0.6)  # Rate limiting
            response = self.session.get(NBA_PLAYER_STATS_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if 'resultSets' in data and len(data['resultSets']) > 0:
                headers = data['resultSets'][0]['headers']
                rows = data['resultSets'][0]['rowSet']
                df = pd.DataFrame(rows, columns=headers)
                return df
            
            return pd.DataFrame()
        
        except Exception as e:
            print(f"Error fetching player stats: {e}")
            return pd.DataFrame()
    
    def get_tracking_stats(self, pt_measure_type='Passing', season='2025-26'):
        """
        Fetch player tracking statistics (includes Potential Assists)
        
        Args:
            pt_measure_type: Type of tracking stat ('Passing', 'Defense', etc.)
            season: NBA season
        
        Returns:
            DataFrame with tracking statistics
        """
        params = {
            'College': '',
            'Conference': '',
            'Country': '',
            'DateFrom': '',
            'DateTo': '',
            'Division': '',
            'DraftPick': '',
            'DraftYear': '',
            'GameScope': '',
            'Height': '',
            'LastNGames': '0',
            'LeagueID': '00',
            'Location': '',
            'Month': '0',
            'OpponentTeamID': '0',
            'Outcome': '',
            'PORound': '0',
            'PerMode': 'PerGame',
            'PlayerExperience': '',
            'PlayerOrTeam': 'Player',
            'PlayerPosition': '',
            'PtMeasureType': pt_measure_type,
            'Season': season,
            'SeasonSegment': '',
            'SeasonType': 'Regular Season',
            'StarterBench': '',
            'TeamID': '0',
            'VsConference': '',
            'VsDivision': '',
            'Weight': ''
        }
        
        try:
            time.sleep(0.6)  # Rate limiting
            response = self.session.get(NBA_TRACKING_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if 'resultSets' in data and len(data['resultSets']) > 0:
                headers = data['resultSets'][0]['headers']
                rows = data['resultSets'][0]['rowSet']
                df = pd.DataFrame(rows, columns=headers)
                return df
            
            return pd.DataFrame()
        
        except Exception as e:
            print(f"Error fetching tracking stats: {e}")
            return pd.DataFrame()
    
    def get_combined_player_data(self, season='2025-26'):
        """
        Combine traditional stats with tracking stats (including Potential Assists)
        
        Returns:
            DataFrame with combined player data
        """
        print("Fetching player statistics...")
        player_stats = self.get_player_stats(season=season)
        
        print("Fetching tracking statistics (Potential Assists)...")
        tracking_stats = self.get_tracking_stats(pt_measure_type='Passing', season=season)
        
        if player_stats.empty or tracking_stats.empty:
            print("Warning: Could not fetch complete data")
            return pd.DataFrame()
        
        # Merge on PLAYER_ID
        combined = player_stats.merge(
            tracking_stats[['PLAYER_ID', 'POTENTIAL_AST', 'AST_TO_PASS_PCT', 'AST_TO_PASS_PCT_ADJ']],
            on='PLAYER_ID',
            how='left',
            suffixes=('', '_tracking')
        )
        
        return combined


if __name__ == "__main__":
    # Test the scraper
    scraper = NBAStatsScraper()
    
    print("Testing NBA Stats Scraper")
    print("=" * 50)
    
    # Test today's games
    print("\nFetching today's games...")
    games = scraper.get_todays_games()
    print(f"Found {len(games)} games today")
    
    # Test player stats
    print("\nFetching player statistics...")
    stats = scraper.get_combined_player_data()
    
    if not stats.empty:
        print(f"Retrieved stats for {len(stats)} players")
        print("\nTop 10 players by Potential Assists:")
        top_potential = stats.nlargest(10, 'POTENTIAL_AST')[
            ['PLAYER_NAME', 'TEAM_ABBREVIATION', 'AST', 'POTENTIAL_AST', 'GP']
        ]
        print(top_potential.to_string(index=False))
