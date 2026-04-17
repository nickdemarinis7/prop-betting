"""
Opponent defense analysis for assists prediction
Calculates how well teams defend against assists
"""

import pandas as pd
import time
from nba_api.stats.endpoints import (
    leaguedashteamstats,
    teamgamelog,
    leaguedashptteamdefend
)


class OpponentDefenseAnalyzer:
    """Analyze opponent defensive strength against assists"""
    
    def __init__(self):
        self.team_defense_cache = {}
    
    def get_team_defensive_stats(self, season='2025-26'):
        """
        Get defensive statistics for all teams
        
        Returns:
            DataFrame with team defensive metrics
        """
        try:
            time.sleep(0.6)
            
            # Get team defensive stats
            team_stats = leaguedashteamstats.LeagueDashTeamStats(
                season=season,
                season_type_all_star='Regular Season',
                measure_type_detailed_defense='Opponent',
                per_mode_detailed='PerGame'
            )
            
            df = team_stats.get_data_frames()[0]
            return df
            
        except Exception as e:
            print(f"Error fetching team defensive stats: {e}")
            return pd.DataFrame()
    
    def calculate_assists_allowed(self, team_id, season='2025-26', last_n_games=10):
        """
        Calculate how many assists a team allows per game
        
        Args:
            team_id: NBA team ID
            season: Season string
            last_n_games: Not used - we use season stats instead
        
        Returns:
            Dictionary with defensive metrics
        """
        # Check cache first
        cache_key = f"{team_id}_{season}"
        if cache_key in self.team_defense_cache:
            return self.team_defense_cache[cache_key]
        
        # Get league-wide defensive stats which includes OPP_AST
        all_team_stats = self.get_team_defensive_stats(season)
        
        if all_team_stats.empty:
            return self._default_defense_metrics()
        
        # Find this team's defensive stats
        team_stats = all_team_stats[all_team_stats['TEAM_ID'] == team_id]
        
        if team_stats.empty:
            return self._default_defense_metrics()
        
        team_row = team_stats.iloc[0]
        
        # Extract defensive metrics
        opp_ast = team_row.get('OPP_AST', 25.0)
        opp_pts = team_row.get('OPP_PTS', 110.0)
        def_rating = team_row.get('DEF_RATING', 110.0)
        
        metrics = {
            'opp_ast_allowed': opp_ast,
            'opp_pts_allowed': opp_pts,
            'opp_tov_forced': 14.0,  # Not in this endpoint
            'def_rating': def_rating,
            'pace': 100.0,  # Use league average
            'games_analyzed': team_row.get('GP', 0),
            'ast_rate_allowed': (opp_ast / 100.0) * 100  # Per 100 possessions estimate
        }
        
        # Rank defense (lower assists allowed = better defense)
        # Normalize to 0-100 scale where 100 = worst defense (allows most assists)
        league_avg_ast = 25.0
        metrics['def_strength'] = (metrics['opp_ast_allowed'] / league_avg_ast) * 100
        
        # Cache the result
        self.team_defense_cache[cache_key] = metrics
        
        return metrics
    
    def _default_defense_metrics(self):
        """Return default/average defensive metrics"""
        return {
            'opp_ast_allowed': 25.0,
            'opp_pts_allowed': 110.0,
            'opp_tov_forced': 14.0,
            'def_rating': 110.0,
            'pace': 100.0,
            'ast_rate_allowed': 25.0,
            'def_strength': 100.0,
            'games_analyzed': 0
        }
    
    def get_matchup_factor(self, opponent_team_id, season='2025-26'):
        """
        Get a matchup factor for assists prediction
        
        Returns:
            Float between 0.7 and 1.3 representing difficulty
            - 0.7-0.9: Tough defense (expect fewer assists)
            - 0.9-1.1: Average defense
            - 1.1-1.3: Weak defense (expect more assists)
        """
        metrics = self.calculate_assists_allowed(opponent_team_id, season)
        
        # Convert def_strength to a multiplier
        # def_strength of 100 = league average = 1.0 multiplier
        # def_strength of 80 = good defense = 0.8 multiplier
        # def_strength of 120 = bad defense = 1.2 multiplier
        
        multiplier = metrics['def_strength'] / 100.0
        
        # Clamp between 0.7 and 1.3 to avoid extreme adjustments
        multiplier = max(0.7, min(1.3, multiplier))
        
        return multiplier
    
    def get_all_team_defenses(self, season='2025-26'):
        """
        Get defensive rankings for all teams
        
        Returns:
            DataFrame with team defensive rankings
        """
        team_stats = self.get_team_defensive_stats(season)
        
        if team_stats.empty:
            return pd.DataFrame()
        
        # Calculate assists allowed ranking
        if 'OPP_AST' in team_stats.columns:
            team_stats['AST_ALLOWED_RANK'] = team_stats['OPP_AST'].rank(ascending=True)
        
        # Select relevant columns
        cols = ['TEAM_ID', 'TEAM_NAME', 'OPP_AST', 'DEF_RATING', 'AST_ALLOWED_RANK']
        available_cols = [col for col in cols if col in team_stats.columns]
        
        return team_stats[available_cols].sort_values('OPP_AST', ascending=True)


if __name__ == "__main__":
    print("=" * 70)
    print("Testing Opponent Defense Analyzer")
    print("=" * 70)
    
    analyzer = OpponentDefenseAnalyzer()
    
    # Test with a specific team (e.g., Lakers - ID: 1610612747)
    print("\n1. Testing defensive metrics for Lakers...")
    team_id = 1610612747
    
    metrics = analyzer.calculate_assists_allowed(team_id, season='2025-26', last_n_games=10)
    
    print(f"\n   Lakers Defensive Metrics (Last 10 Games):")
    print(f"   - Assists Allowed: {metrics['opp_ast_allowed']:.1f} per game")
    print(f"   - Points Allowed: {metrics['opp_pts_allowed']:.1f} per game")
    print(f"   - Defensive Rating: {metrics['def_rating']:.1f}")
    print(f"   - Pace: {metrics['pace']:.1f}")
    print(f"   - Defense Strength: {metrics['def_strength']:.1f} (100 = average)")
    
    # Get matchup factor
    print(f"\n2. Testing matchup factor...")
    factor = analyzer.get_matchup_factor(team_id, season='2025-26')
    print(f"   Matchup Factor: {factor:.2f}")
    
    if factor < 0.9:
        print(f"   → Tough matchup (expect fewer assists)")
    elif factor > 1.1:
        print(f"   → Favorable matchup (expect more assists)")
    else:
        print(f"   → Average matchup")
    
    # Get all team defenses
    print(f"\n3. Getting all team defensive rankings...")
    all_defenses = analyzer.get_all_team_defenses(season='2025-26')
    
    if not all_defenses.empty:
        print(f"\n   Top 5 Best Defenses (Fewest Assists Allowed):")
        print(all_defenses.head(5).to_string(index=False))
        
        print(f"\n   Top 5 Worst Defenses (Most Assists Allowed):")
        print(all_defenses.tail(5).to_string(index=False))
    
    print("\n" + "=" * 70)
    print("✅ Opponent defense analysis ready!")
    print("=" * 70)
