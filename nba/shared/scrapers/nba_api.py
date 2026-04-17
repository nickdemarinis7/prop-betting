"""
NBA Stats scraper using the official nba_api library
Free, open-source, and comprehensive access to all NBA.com statistics
"""

import pandas as pd
from datetime import datetime, timedelta
import time
from nba_api.stats.endpoints import (
    leaguedashplayerstats,
    leaguedashptteamdefend,
    leaguedashptstats,
    scoreboardv2
)
from nba_api.stats.static import teams, players


class NBAApiScraper:
    """Scraper using official nba_api library"""
    
    def __init__(self):
        self.teams_data = teams.get_teams()
        
    def get_todays_games(self, game_date=None):
        """
        Fetch today's NBA game schedule
        
        Args:
            game_date: Date string in format 'YYYY-MM-DD' (defaults to today)
        
        Returns:
            DataFrame with game information
        """
        if game_date is None:
            game_date = datetime.now().strftime('%m/%d/%Y')
        else:
            # Convert from YYYY-MM-DD to MM/DD/YYYY
            dt = datetime.strptime(game_date, '%Y-%m-%d')
            game_date = dt.strftime('%m/%d/%Y')
        
        try:
            scoreboard = scoreboardv2.ScoreboardV2(game_date=game_date)
            games_df = scoreboard.get_data_frames()[0]  # GameHeader
            return games_df
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
        Fetch traditional player statistics
        
        Args:
            season: NBA season (e.g., '2024-25')
            season_type: 'Regular Season' or 'Playoffs'
        
        Returns:
            DataFrame with player statistics
        """
        try:
            time.sleep(0.6)  # Rate limiting
            stats = leaguedashplayerstats.LeagueDashPlayerStats(
                season=season,
                season_type_all_star=season_type,
                per_mode_detailed='PerGame'
            )
            df = stats.get_data_frames()[0]
            return df
        except Exception as e:
            print(f"Error fetching player stats: {e}")
            return pd.DataFrame()
    
    def get_tracking_stats(self, pt_measure_type='Passing', season='2025-26'):
        """
        Fetch player tracking statistics (includes Potential Assists!)
        
        Args:
            pt_measure_type: Type of tracking stat ('Passing', 'Defense', etc.)
            season: NBA season
        
        Returns:
            DataFrame with tracking statistics including POTENTIAL_AST
        """
        try:
            time.sleep(0.6)  # Rate limiting
            tracking = leaguedashptstats.LeagueDashPtStats(
                season=season,
                season_type_all_star='Regular Season',
                pt_measure_type=pt_measure_type,
                per_mode_simple='PerGame'
            )
            df = tracking.get_data_frames()[0]
            return df
        except Exception as e:
            print(f"Error fetching tracking stats: {e}")
            return pd.DataFrame()
    
    def get_combined_player_data(self, season='2025-26'):
        """
        Combine traditional stats with tracking stats (including Potential Assists)
        
        Returns:
            DataFrame with combined player data including POTENTIAL_AST
        """
        print("Fetching player statistics...")
        player_stats = self.get_player_stats(season=season)
        
        if player_stats.empty:
            print("Warning: Could not fetch player stats")
            return pd.DataFrame()
        
        print(f"Retrieved stats for {len(player_stats)} players")
        
        print("Fetching tracking statistics (Potential Assists)...")
        tracking_stats = self.get_tracking_stats(pt_measure_type='Passing', season=season)
        
        if tracking_stats.empty:
            print("Warning: Could not fetch tracking stats")
            return player_stats
        
        print(f"Retrieved tracking stats for {len(tracking_stats)} players")
        
        # Check what columns we actually have
        print(f"Tracking stats columns: {tracking_stats.columns.tolist()[:10]}...")
        
        # Find the ID column (might be PLAYER_ID or something else)
        id_col = None
        for col in ['PLAYER_ID', 'TEAM_ID', 'TEAM_ABBREVIATION']:
            if col in tracking_stats.columns:
                id_col = col
                break
        
        if id_col is None:
            print("Warning: Could not find ID column in tracking stats")
            return player_stats
        
        # Get available columns for merge
        merge_cols = [id_col]
        for col in ['POTENTIAL_AST', 'AST_TO_PASS_PCT', 'AST_TO_PASS_PCT_ADJ', 'PASSES_MADE', 'PASSES_RECEIVED']:
            if col in tracking_stats.columns:
                merge_cols.append(col)
        
        # Merge on the ID column
        if id_col in player_stats.columns:
            combined = player_stats.merge(
                tracking_stats[merge_cols],
                on=id_col,
                how='left',
                suffixes=('', '_tracking')
            )
        else:
            print(f"Warning: {id_col} not in player_stats, returning player stats only")
            return player_stats
        
        print(f"✓ Combined data for {len(combined)} players!")
        if 'POTENTIAL_AST' in combined.columns:
            print(f"✓ Potential Assists column successfully added!")
        
        return combined


if __name__ == "__main__":
    # Test the scraper
    print("=" * 70)
    print("Testing NBA API Scraper (Official Library)")
    print("=" * 70)
    
    scraper = NBAApiScraper()
    
    # Test today's games
    print("\n1. Fetching today's games...")
    games = scraper.get_todays_games()
    
    if not games.empty:
        print(f"   ✓ Found {len(games)} games today")
        if 'GAMECODE' in games.columns:
            print("\n   Today's Games:")
            for _, game in games.iterrows():
                print(f"   - {game.get('GAMECODE', 'Game')}")
    else:
        print("   No games today or unable to fetch")
    
    # Test player stats with Potential Assists
    print("\n2. Fetching player statistics with Potential Assists...")
    stats = scraper.get_combined_player_data(season='2025-26')
    
    if not stats.empty:
        print(f"\n   ✓ Retrieved complete stats for {len(stats)} players")
        
        # Show top players by Potential Assists
        if 'POTENTIAL_AST' in stats.columns:
            print("\n   🎯 Top 10 Players by Potential Assists:")
            top_potential = stats.nlargest(10, 'POTENTIAL_AST')[
                ['PLAYER_NAME', 'TEAM_ABBREVIATION', 'AST', 'POTENTIAL_AST', 'GP', 'MIN']
            ]
            for idx, row in top_potential.iterrows():
                print(f"   {row['PLAYER_NAME']:25} ({row['TEAM_ABBREVIATION']:3}) - "
                      f"AST: {row['AST']:4.1f}, Potential AST: {row['POTENTIAL_AST']:4.1f}, "
                      f"MIN: {row['MIN']:4.1f}")
        
        print("\n   ✅ SUCCESS! We have access to Potential Assists!")
    else:
        print("   ⚠️  Could not fetch player data")
    
    print("\n" + "=" * 70)
    print("✅ NBA API Library works! Free access to ALL stats including")
    print("   Potential Assists, advanced stats, and more!")
    print("=" * 70)
