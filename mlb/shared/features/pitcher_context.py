"""
Pitcher Context Features
Day/night splits, expected IP, short rest detection
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from shared.scrapers.pitcher_stats import PitcherStatsScraper


class PitcherContextAnalyzer:
    """Analyze pitcher context: day/night, expected IP, rest"""
    
    def __init__(self):
        self.stats_scraper = PitcherStatsScraper()
        self.current_season = datetime.now().year
    
    def _get_game_logs_with_fallback(self, pitcher_id, season=None):
        """Get game logs, trying current season first then falling back"""
        if season is None:
            season = self.current_season
        
        game_logs = self.stats_scraper.get_game_logs(pitcher_id, season=season)
        
        if game_logs.empty and season > 2024:
            game_logs = self.stats_scraper.get_game_logs(pitcher_id, season=season - 1)
        
        if game_logs.empty and season - 1 > 2024:
            game_logs = self.stats_scraper.get_game_logs(pitcher_id, season=season - 2)
        
        return game_logs
    
    def get_day_night_splits(self, pitcher_id):
        """
        Get pitcher's K/9 in day vs night games
        
        Args:
            pitcher_id: MLB player ID
            
        Returns:
            dict with day_k9 and night_k9
        """
        try:
            # Get game logs
            game_logs = self._get_game_logs_with_fallback(pitcher_id)
            
            if game_logs.empty:
                return {'day_k9': None, 'night_k9': None}
            
            # Determine day/night from game time
            # MLB API doesn't always have this, so we'll use a heuristic:
            # Games before 5pm local time = day games
            
            # For now, calculate from all games (we'll enhance this later)
            day_games = []
            night_games = []
            
            for _, game in game_logs.iterrows():
                # Try to determine from game time if available
                # For now, split games randomly as placeholder
                # TODO: Get actual day/night indicator from API
                
                k9 = self._calculate_k9(game)
                
                # Placeholder: assume 30% are day games
                if np.random.random() < 0.3:
                    day_games.append(k9)
                else:
                    night_games.append(k9)
            
            day_k9 = np.mean(day_games) if day_games else None
            night_k9 = np.mean(night_games) if night_games else None
            
            return {
                'day_k9': day_k9,
                'night_k9': night_k9,
                'day_games': len(day_games),
                'night_games': len(night_games)
            }
            
        except Exception as e:
            print(f"   ⚠️  Error getting day/night splits: {e}")
            return {'day_k9': None, 'night_k9': None}
    
    def get_expected_ip(self, pitcher_id, n_games=10):
        """
        Calculate expected innings pitched based on recent starts
        
        Args:
            pitcher_id: MLB player ID
            n_games: Number of recent games to analyze
            
        Returns:
            float: Expected IP for next start
        """
        try:
            game_logs = self._get_game_logs_with_fallback(pitcher_id)
            
            if game_logs.empty:
                return 5.0  # Default
            
            # Get recent starts (filter out relief appearances)
            recent_starts = game_logs.head(n_games)
            recent_starts = recent_starts[recent_starts['IP'] >= 3.0]  # Only starts
            
            if recent_starts.empty:
                return 5.0
            
            # Calculate average IP
            avg_ip = recent_starts['IP'].median()
            
            # Cap at reasonable maximum
            expected_ip = min(avg_ip, 6.5)
            
            return expected_ip
            
        except Exception as e:
            print(f"   ⚠️  Error calculating expected IP: {e}")
            return 5.0
    
    def check_short_rest(self, pitcher_id, game_date=None):
        """
        Check if pitcher is on short rest
        
        Args:
            pitcher_id: MLB player ID
            game_date: Date of upcoming game (default: today)
            
        Returns:
            dict with days_rest and is_short_rest
        """
        try:
            if game_date is None:
                game_date = datetime.now()
            elif isinstance(game_date, str):
                game_date = datetime.strptime(game_date, '%Y-%m-%d')
            
            game_logs = self._get_game_logs_with_fallback(pitcher_id)
            
            if game_logs.empty:
                return {'days_rest': None, 'is_short_rest': False}
            
            # Get most recent start
            last_start = game_logs.iloc[0]
            last_start_date = pd.to_datetime(last_start['game_date'])
            
            # Calculate days of rest
            days_rest = (game_date - last_start_date).days
            
            # Short rest = less than 4 days
            is_short_rest = days_rest < 4
            
            return {
                'days_rest': days_rest,
                'is_short_rest': is_short_rest,
                'last_start_date': last_start_date.strftime('%Y-%m-%d'),
                'last_start_ip': last_start.get('IP', 0),
                'last_start_pitches': last_start.get('Pitches', 0)
            }
            
        except Exception as e:
            print(f"   ⚠️  Error checking short rest: {e}")
            return {'days_rest': None, 'is_short_rest': False}
    
    def get_recent_workload(self, pitcher_id, n_games=4):
        """
        Analyze recent pitch count workload
        
        Args:
            pitcher_id: MLB player ID
            n_games: Number of recent games to analyze
            
        Returns:
            dict with workload metrics
        """
        try:
            game_logs = self._get_game_logs_with_fallback(pitcher_id)
            
            if game_logs.empty:
                return {'high_workload': False}
            
            recent_games = game_logs.head(n_games)
            
            # Get pitch counts (if available)
            if 'Pitches' in recent_games.columns:
                pitch_counts = recent_games['Pitches'].values
                
                # Check for high workload (>110 pitches in last 2 starts)
                recent_high = any(pc > 110 for pc in pitch_counts[:2])
                
                return {
                    'high_workload': recent_high,
                    'avg_pitches': pitch_counts.mean(),
                    'max_recent_pitches': pitch_counts.max(),
                    'pitch_counts': list(pitch_counts)
                }
            else:
                return {'high_workload': False}
                
        except Exception as e:
            print(f"   ⚠️  Error analyzing workload: {e}")
            return {'high_workload': False}
    
    def _calculate_k9(self, game_row):
        """Calculate K/9 for a single game"""
        try:
            k = game_row.get('SO', 0)
            ip = game_row.get('IP', 0)
            
            if ip == 0:
                return 0
            
            return (k / ip) * 9
        except:
            return 0
    
    def get_full_context(self, pitcher_id, game_date=None, is_day_game=False):
        """
        Get all context features for a pitcher
        
        Args:
            pitcher_id: MLB player ID
            game_date: Date of game
            is_day_game: Whether it's a day game
            
        Returns:
            dict with all context features
        """
        context = {}
        
        # Expected IP
        context['expected_ip'] = self.get_expected_ip(pitcher_id)
        
        # Short rest
        rest_info = self.check_short_rest(pitcher_id, game_date)
        context.update(rest_info)
        
        # Workload
        workload_info = self.get_recent_workload(pitcher_id)
        context.update(workload_info)
        
        # Day/night splits
        splits = self.get_day_night_splits(pitcher_id)
        context['day_k9'] = splits['day_k9']
        context['night_k9'] = splits['night_k9']
        
        # Determine which K/9 to use
        if is_day_game and splits['day_k9']:
            context['context_k9'] = splits['day_k9']
        elif not is_day_game and splits['night_k9']:
            context['context_k9'] = splits['night_k9']
        else:
            context['context_k9'] = None  # Use season average
        
        return context


if __name__ == "__main__":
    # Test the analyzer
    analyzer = PitcherContextAnalyzer()
    
    # Test with a known pitcher (example ID)
    test_pitcher_id = 543037  # Justin Verlander
    
    print("Testing Pitcher Context Analyzer")
    print("=" * 60)
    
    context = analyzer.get_full_context(test_pitcher_id, is_day_game=True)
    
    print(f"\nExpected IP: {context['expected_ip']:.1f}")
    print(f"Days Rest: {context.get('days_rest', 'N/A')}")
    print(f"Short Rest: {context.get('is_short_rest', False)}")
    print(f"High Workload: {context.get('high_workload', False)}")
    print(f"Day K/9: {context.get('day_k9', 'N/A')}")
    print(f"Night K/9: {context.get('night_k9', 'N/A')}")
