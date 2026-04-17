"""
Fatigue analysis for back-to-back games and rest days
Players perform worse on back-to-backs and with less rest
"""

import pandas as pd
from datetime import datetime, timedelta
from nba_api.stats.endpoints import teamgamelog


class FatigueAnalyzer:
    """Analyze rest days and back-to-back impact on performance"""
    
    def __init__(self):
        self.team_schedule_cache = {}
    
    def get_team_recent_games(self, team_id, season='2025-26', n_games=5):
        """
        Get a team's recent games to check schedule
        
        Args:
            team_id: NBA team ID
            season: Season string
            n_games: Number of recent games to fetch
        
        Returns:
            DataFrame with recent games
        """
        cache_key = f"{team_id}_{season}"
        if cache_key in self.team_schedule_cache:
            return self.team_schedule_cache[cache_key]
        
        try:
            import time
            time.sleep(0.6)
            
            gamelog = teamgamelog.TeamGameLog(
                team_id=team_id,
                season=season,
                season_type_all_star='Regular Season'
            )
            
            df = gamelog.get_data_frames()[0]
            
            if not df.empty:
                df = df.head(n_games)
                self.team_schedule_cache[cache_key] = df
            
            return df
            
        except Exception as e:
            print(f"Error fetching team schedule: {e}")
            return pd.DataFrame()
    
    def calculate_days_rest(self, team_id, game_date=None, season='2025-26'):
        """
        Calculate days of rest before tonight's game
        
        Args:
            team_id: NBA team ID
            game_date: Date of the game (default: today)
            season: Season string
        
        Returns:
            Integer: Days of rest (0 = back-to-back)
        """
        if game_date is None:
            game_date = datetime.now()
        elif isinstance(game_date, str):
            game_date = datetime.strptime(game_date, '%Y-%m-%d')
        
        recent_games = self.get_team_recent_games(team_id, season, n_games=3)
        
        if recent_games.empty:
            return 2  # Assume normal rest
        
        # Get most recent game
        if 'GAME_DATE' in recent_games.columns:
            recent_games['GAME_DATE'] = pd.to_datetime(recent_games['GAME_DATE'])
            recent_games = recent_games.sort_values('GAME_DATE', ascending=False)
            
            last_game_date = recent_games.iloc[0]['GAME_DATE']
            
            # Calculate days between
            days_rest = (game_date - last_game_date).days - 1
            
            # Ensure non-negative
            days_rest = max(0, days_rest)
            
            return days_rest
        
        return 2  # Default
    
    def is_back_to_back(self, team_id, game_date=None, season='2025-26'):
        """
        Check if team is playing back-to-back
        
        Returns:
            Boolean: True if back-to-back
        """
        days_rest = self.calculate_days_rest(team_id, game_date, season)
        return days_rest == 0
    
    def calculate_fatigue_factor(self, days_rest):
        """
        Calculate performance adjustment based on rest
        
        Args:
            days_rest: Number of days since last game
        
        Returns:
            Float: Multiplier (e.g., 0.92 = 8% decrease)
        """
        # Research shows:
        # - Back-to-back (0 days): ~8-10% performance decrease
        # - 1 day rest: ~3-5% decrease
        # - 2+ days rest: Normal performance
        # - 3+ days rest: Slight increase (fresh legs)
        
        fatigue_map = {
            0: 0.92,   # Back-to-back: -8%
            1: 0.96,   # 1 day rest: -4%
            2: 1.00,   # 2 days: Normal
            3: 1.02,   # 3 days: +2% (fresh)
            4: 1.02,   # 4+ days: +2%
        }
        
        if days_rest >= 4:
            return fatigue_map[4]
        
        return fatigue_map.get(days_rest, 1.0)
    
    def get_rest_category(self, days_rest):
        """Categorize rest level"""
        if days_rest == 0:
            return 'BACK-TO-BACK'
        elif days_rest == 1:
            return 'SHORT REST'
        elif days_rest == 2:
            return 'NORMAL REST'
        else:
            return 'EXTRA REST'
    
    def analyze_matchup_fatigue(self, team1_id, team2_id, game_date=None, season='2025-26'):
        """
        Analyze fatigue for both teams in a matchup
        
        Returns:
            Dictionary with fatigue analysis
        """
        team1_rest = self.calculate_days_rest(team1_id, game_date, season)
        team2_rest = self.calculate_days_rest(team2_id, game_date, season)
        
        team1_factor = self.calculate_fatigue_factor(team1_rest)
        team2_factor = self.calculate_fatigue_factor(team2_rest)
        
        # Determine advantage
        if team1_rest > team2_rest:
            advantage = 'TEAM1'
        elif team2_rest > team1_rest:
            advantage = 'TEAM2'
        else:
            advantage = 'EVEN'
        
        return {
            'team1_days_rest': team1_rest,
            'team2_days_rest': team2_rest,
            'team1_fatigue_factor': team1_factor,
            'team2_fatigue_factor': team2_factor,
            'team1_category': self.get_rest_category(team1_rest),
            'team2_category': self.get_rest_category(team2_rest),
            'rest_advantage': advantage
        }


if __name__ == "__main__":
    print("=" * 70)
    print("FATIGUE ANALYZER - TEST")
    print("=" * 70)
    
    analyzer = FatigueAnalyzer()
    
    # Test with a known team
    print("\n1. Testing Fatigue Factors:")
    
    for days in range(5):
        factor = analyzer.calculate_fatigue_factor(days)
        category = analyzer.get_rest_category(days)
        impact = (factor - 1) * 100
        
        print(f"   {days} days rest ({category:15}): {factor:.2f}x ({impact:+.0f}%)")
    
    # Test 2: Check specific team
    print("\n2. Checking Lakers Schedule:")
    lakers_id = 1610612747
    
    days_rest = analyzer.calculate_days_rest(lakers_id, season='2025-26')
    is_b2b = analyzer.is_back_to_back(lakers_id, season='2025-26')
    factor = analyzer.calculate_fatigue_factor(days_rest)
    
    print(f"   Days of rest: {days_rest}")
    print(f"   Back-to-back: {is_b2b}")
    print(f"   Fatigue factor: {factor:.2f}x")
    
    if is_b2b:
        print(f"   ⚠️  ALERT: Lakers playing back-to-back!")
        print(f"   Expected ~8% performance decrease")
    
    # Test 3: Matchup analysis
    print("\n3. Matchup Fatigue Analysis:")
    warriors_id = 1610612744
    
    matchup = analyzer.analyze_matchup_fatigue(lakers_id, warriors_id, season='2025-26')
    
    print(f"   Lakers: {matchup['team1_days_rest']} days ({matchup['team1_category']})")
    print(f"   Warriors: {matchup['team2_days_rest']} days ({matchup['team2_category']})")
    print(f"   Rest advantage: {matchup['rest_advantage']}")
    
    print("\n" + "=" * 70)
    print("✅ Fatigue analysis ready!")
    print("   Back-to-backs = -8% performance")
    print("=" * 70)
