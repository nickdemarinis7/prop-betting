"""
Pace analysis for predicting game tempo and assist opportunities
Faster pace = more possessions = more assists
"""

import pandas as pd
import time
from nba_api.stats.endpoints import leaguedashteamstats


class PaceAnalyzer:
    """Analyze team pace and predict game tempo"""
    
    def __init__(self):
        self.team_pace_cache = {}
    
    def get_team_pace_stats(self, season='2025-26'):
        """
        Get pace statistics for all teams
        
        Returns:
            DataFrame with team pace metrics
        """
        try:
            time.sleep(0.6)
            
            team_stats = leaguedashteamstats.LeagueDashTeamStats(
                season=season,
                season_type_all_star='Regular Season',
                measure_type_detailed_defense='Base',
                per_mode_detailed='PerGame'
            )
            
            df = team_stats.get_data_frames()[0]
            
            # If PACE column doesn't exist, estimate it from possessions
            if 'PACE' not in df.columns and 'GP' in df.columns:
                # Estimate pace from games played and other stats
                # Pace ≈ possessions per 48 minutes
                # Rough estimate: (FGA + 0.44*FTA + TOV - OREB) per game
                if all(col in df.columns for col in ['FGA', 'FTA', 'TOV', 'OREB']):
                    df['PACE'] = df['FGA'] + (0.44 * df['FTA']) + df['TOV'] - df['OREB']
                else:
                    df['PACE'] = 100.0  # Default
            
            return df
            
        except Exception as e:
            print(f"Error fetching pace stats: {e}")
            return pd.DataFrame()
    
    def get_team_pace(self, team_id, season='2025-26'):
        """
        Get a team's pace (possessions per 48 minutes)
        
        Args:
            team_id: NBA team ID
            season: Season string
        
        Returns:
            Float: Pace value (typically 95-105)
        """
        cache_key = f"{team_id}_{season}"
        if cache_key in self.team_pace_cache:
            return self.team_pace_cache[cache_key]
        
        all_stats = self.get_team_pace_stats(season)
        
        if all_stats.empty:
            return 100.0  # League average
        
        team_stats = all_stats[all_stats['TEAM_ID'] == team_id]
        
        if team_stats.empty:
            return 100.0
        
        pace = team_stats.iloc[0].get('PACE', 100.0)
        
        self.team_pace_cache[cache_key] = pace
        return pace
    
    def calculate_game_pace(self, team1_id, team2_id, season='2025-26'):
        """
        Predict the pace of a game between two teams
        
        Args:
            team1_id: First team ID
            team2_id: Second team ID
            season: Season string
        
        Returns:
            Dictionary with pace prediction
        """
        team1_pace = self.get_team_pace(team1_id, season)
        team2_pace = self.get_team_pace(team2_id, season)
        
        # Game pace is typically the average of both teams' paces
        # But faster teams tend to dictate pace more
        # Use weighted average: 60% faster team, 40% slower team
        
        faster_pace = max(team1_pace, team2_pace)
        slower_pace = min(team1_pace, team2_pace)
        
        predicted_pace = (faster_pace * 0.6) + (slower_pace * 0.4)
        
        # Calculate pace factor (relative to league average)
        league_avg_pace = 100.0
        pace_factor = predicted_pace / league_avg_pace
        
        return {
            'team1_pace': team1_pace,
            'team2_pace': team2_pace,
            'predicted_pace': predicted_pace,
            'pace_factor': pace_factor,
            'pace_category': self._categorize_pace(predicted_pace)
        }
    
    def _categorize_pace(self, pace):
        """Categorize pace as slow, average, or fast"""
        if pace < 97:
            return 'SLOW'
        elif pace > 103:
            return 'FAST'
        else:
            return 'AVERAGE'
    
    def calculate_pace_boost(self, predicted_pace):
        """
        Calculate assist boost based on game pace
        
        Args:
            predicted_pace: Predicted game pace
        
        Returns:
            Float: Multiplier (e.g., 1.10 = 10% boost)
        """
        league_avg = 100.0
        
        # For every 5 points above/below average pace:
        # +5 pace = +5% assists
        # -5 pace = -5% assists
        
        pace_diff = predicted_pace - league_avg
        boost = 1.0 + (pace_diff / 100.0)
        
        # Cap between 0.85 and 1.15 (±15%)
        boost = max(0.85, min(1.15, boost))
        
        return boost
    
    def get_all_team_paces(self, season='2025-26'):
        """
        Get pace rankings for all teams
        
        Returns:
            DataFrame sorted by pace
        """
        all_stats = self.get_team_pace_stats(season)
        
        if all_stats.empty:
            return pd.DataFrame()
        
        # Select relevant columns
        cols = ['TEAM_ID', 'TEAM_NAME', 'PACE', 'OFF_RATING', 'DEF_RATING']
        available_cols = [col for col in cols if col in all_stats.columns]
        
        result = all_stats[available_cols].copy()
        
        if 'PACE' in result.columns:
            result = result.sort_values('PACE', ascending=False)
            result['PACE_RANK'] = range(1, len(result) + 1)
        
        return result


if __name__ == "__main__":
    print("=" * 70)
    print("PACE ANALYZER - TEST")
    print("=" * 70)
    
    analyzer = PaceAnalyzer()
    
    # Test 1: Get all team paces
    print("\n1. Getting Team Pace Rankings...")
    all_paces = analyzer.get_all_team_paces(season='2025-26')
    
    if not all_paces.empty:
        print(f"\n   Fastest Teams:")
        display_cols = ['TEAM_NAME']
        if 'PACE' in all_paces.columns:
            display_cols.append('PACE')
        if 'PACE_RANK' in all_paces.columns:
            display_cols.append('PACE_RANK')
        print(all_paces.head(5)[display_cols].to_string(index=False))
        
        print(f"\n   Slowest Teams:")
        print(all_paces.tail(5)[display_cols].to_string(index=False))
    else:
        print("   ⚠️  Could not fetch pace data")
    
    # Test 2: Predict game pace
    print("\n2. Predicting Game Pace...")
    
    # Example: Lakers (1610612747) vs Warriors (1610612744)
    lakers_id = 1610612747
    warriors_id = 1610612744
    
    game_pace = analyzer.calculate_game_pace(lakers_id, warriors_id, season='2025-26')
    
    print(f"\n   Lakers pace: {game_pace['team1_pace']:.1f}")
    print(f"   Warriors pace: {game_pace['team2_pace']:.1f}")
    print(f"   Predicted game pace: {game_pace['predicted_pace']:.1f}")
    print(f"   Pace category: {game_pace['pace_category']}")
    print(f"   Pace factor: {game_pace['pace_factor']:.2f}x")
    
    # Test 3: Calculate pace boost
    print("\n3. Pace Boost Examples:")
    
    test_paces = [95, 100, 105, 110]
    for pace in test_paces:
        boost = analyzer.calculate_pace_boost(pace)
        category = analyzer._categorize_pace(pace)
        print(f"   Pace {pace} ({category:7}): {boost:.2f}x boost ({(boost-1)*100:+.0f}%)")
    
    print("\n" + "=" * 70)
    print("✅ Pace analysis ready!")
    print("   Fast games = more possessions = more assists")
    print("=" * 70)
