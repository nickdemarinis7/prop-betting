"""
Feature engineering for NBA assists prediction model
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FEATURES, MIN_GAMES_PLAYED


class FeatureEngine:
    """Generate features for assists prediction"""
    
    def __init__(self):
        self.feature_names = FEATURES
    
    def calculate_advanced_metrics(self, df):
        """
        Calculate advanced metrics from basic stats
        
        Args:
            df: DataFrame with player statistics
        
        Returns:
            DataFrame with additional calculated features
        """
        df = df.copy()
        
        # Usage Rate (estimate based on FGA, FTA, TOV, MIN, TEAM_MIN)
        if all(col in df.columns for col in ['FGA', 'FTA', 'TOV', 'MIN']):
            df['usage_rate'] = ((df['FGA'] + 0.44 * df['FTA'] + df['TOV']) / df['MIN']) * 100
        else:
            df['usage_rate'] = 0
        
        # Assists per game
        if 'AST' in df.columns:
            df['assists_per_game'] = df['AST']
        else:
            df['assists_per_game'] = 0
        
        # Minutes per game
        if 'MIN' in df.columns:
            df['minutes_per_game'] = df['MIN']
        else:
            df['minutes_per_game'] = 0
        
        # Games played
        if 'GP' in df.columns:
            df['games_played'] = df['GP']
        else:
            df['games_played'] = 0
        
        # Potential assists (already in tracking data)
        if 'POTENTIAL_AST' in df.columns:
            df['potential_assists'] = df['POTENTIAL_AST']
        else:
            df['potential_assists'] = 0
        
        # Assist to pass percentage
        if 'AST_TO_PASS_PCT' in df.columns:
            df['assist_to_pass_pct'] = df['AST_TO_PASS_PCT']
        else:
            df['assist_to_pass_pct'] = 0
        
        # Pace (estimate - would need team data for accurate calculation)
        df['pace'] = 100.0  # Placeholder - should be team-specific
        
        # Home/Away (placeholder - needs game context)
        df['home_away'] = 1  # 1 for home, 0 for away
        
        # Days rest (placeholder - needs schedule data)
        df['days_rest'] = 1
        
        # Opponent defensive rating (placeholder - needs matchup data)
        df['opponent_defensive_rating'] = 110.0
        
        # Team assists per game (placeholder - needs team data)
        df['team_assists_per_game'] = 25.0
        
        return df
    
    def filter_active_players(self, df, min_games=MIN_GAMES_PLAYED):
        """
        Filter for players with minimum games played
        
        Args:
            df: DataFrame with player statistics
            min_games: Minimum games played threshold
        
        Returns:
            Filtered DataFrame
        """
        if 'games_played' in df.columns:
            return df[df['games_played'] >= min_games].copy()
        return df
    
    def prepare_features(self, df):
        """
        Prepare feature matrix for model
        
        Args:
            df: DataFrame with player statistics
        
        Returns:
            DataFrame with only feature columns
        """
        df = self.calculate_advanced_metrics(df)
        df = self.filter_active_players(df)
        
        # Ensure all required features exist
        for feature in self.feature_names:
            if feature not in df.columns:
                df[feature] = 0
        
        return df
    
    def create_target_variable(self, df, target_col='AST'):
        """
        Create target variable (actual assists)
        
        Args:
            df: DataFrame with player statistics
            target_col: Column name for target variable
        
        Returns:
            Series with target values
        """
        if target_col in df.columns:
            return df[target_col]
        return pd.Series([0] * len(df))
    
    def add_matchup_context(self, df, games_df):
        """
        Add matchup-specific context (home/away, opponent, etc.)
        
        Args:
            df: Player statistics DataFrame
            games_df: Today's games DataFrame
        
        Returns:
            DataFrame with matchup context
        """
        # This would be enhanced with actual game matchup data
        # For now, return df as-is
        return df
    
    def get_recent_form(self, player_id, last_n_games=5):
        """
        Get player's recent performance (last N games)
        
        Args:
            player_id: NBA player ID
            last_n_games: Number of recent games to analyze
        
        Returns:
            Dictionary with recent performance metrics
        """
        # Placeholder - would need game log data
        return {
            'recent_assists_avg': 0,
            'recent_potential_assists_avg': 0,
            'trending_up': False
        }


if __name__ == "__main__":
    # Test feature engineering
    print("Testing Feature Engineering")
    print("=" * 50)
    
    # Create sample data
    sample_data = pd.DataFrame({
        'PLAYER_ID': [1, 2, 3],
        'PLAYER_NAME': ['Player A', 'Player B', 'Player C'],
        'AST': [8.5, 6.2, 10.1],
        'POTENTIAL_AST': [12.3, 9.1, 14.5],
        'MIN': [35.0, 28.0, 38.0],
        'GP': [45, 50, 42],
        'FGA': [15.0, 12.0, 18.0],
        'FTA': [5.0, 3.0, 6.0],
        'TOV': [3.0, 2.5, 4.0],
        'AST_TO_PASS_PCT': [0.45, 0.38, 0.52]
    })
    
    engine = FeatureEngine()
    processed = engine.prepare_features(sample_data)
    
    print("\nProcessed Features:")
    print(processed[['PLAYER_NAME'] + engine.feature_names].to_string(index=False))
