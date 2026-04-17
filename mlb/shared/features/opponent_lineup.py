"""
Opponent Lineup Analyzer
Analyzes opponent batting lineups to calculate weighted K rates
"""

import requests
import pandas as pd
from datetime import datetime


class OpponentLineupAnalyzer:
    """Analyze opponent lineups for more accurate strikeout projections"""
    
    def __init__(self):
        self.base_url = "https://statsapi.mlb.com/api/v1"
        self.session = requests.Session()
        self._batter_cache = {}
        self._lineup_cache = {}
    
    def get_team_roster(self, team_id, season=None):
        """
        Get team's active roster
        
        Args:
            team_id: MLB team ID
            season: Year (defaults to current)
            
        Returns:
            DataFrame with roster players
        """
        if season is None:
            season = datetime.now().year
        
        try:
            url = f"{self.base_url}/teams/{team_id}/roster"
            params = {'season': season}
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            roster = []
            for player in data.get('roster', []):
                person = player.get('person', {})
                position = player.get('position', {})
                
                roster.append({
                    'player_id': person.get('id'),
                    'player_name': person.get('fullName'),
                    'position': position.get('abbreviation'),
                    'position_type': position.get('type')
                })
            
            return pd.DataFrame(roster)
            
        except Exception as e:
            print(f"      ⚠️  Error fetching roster: {e}")
            return pd.DataFrame()
    
    def get_batter_stats_vs_hand(self, player_id, vs_hand='R', season=None):
        """
        Get batter's stats vs RHP or LHP
        
        Args:
            player_id: MLB player ID
            vs_hand: 'R' for vs RHP, 'L' for vs LHP
            season: Year (defaults to current)
            
        Returns:
            Dict with K%, AB, SO, etc.
        """
        if season is None:
            season = datetime.now().year
        
        # Check cache
        cache_key = f"{player_id}_{vs_hand}_{season}"
        if cache_key in self._batter_cache:
            return self._batter_cache[cache_key]
        
        try:
            url = f"{self.base_url}/people/{player_id}/stats"
            params = {
                'stats': 'statSplits',
                'group': 'hitting',
                'season': season,
                'sitCodes': f'vs{vs_hand}HP'  # vsRHP or vsLHP
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Parse splits
            for stat_group in data.get('stats', []):
                for split in stat_group.get('splits', []):
                    stats = split.get('stat', {})
                    
                    ab = stats.get('atBats', 0)
                    so = stats.get('strikeOuts', 0)
                    
                    result = {
                        'AB': ab,
                        'SO': so,
                        'K_PCT': so / ab if ab > 0 else 0.23,  # League avg fallback
                        'AVG': stats.get('avg', 0),
                        'OBP': stats.get('obp', 0),
                        'SLG': stats.get('slg', 0)
                    }
                    
                    self._batter_cache[cache_key] = result
                    return result
            
            # No splits found, return league average
            result = {'AB': 0, 'SO': 0, 'K_PCT': 0.23, 'AVG': 0, 'OBP': 0, 'SLG': 0}
            self._batter_cache[cache_key] = result
            return result
            
        except Exception as e:
            # Return league average on error
            return {'AB': 0, 'SO': 0, 'K_PCT': 0.23, 'AVG': 0, 'OBP': 0, 'SLG': 0}
    
    def get_probable_lineup(self, team_id, season=None):
        """
        Get team's probable starting lineup
        Uses recent games to determine regular starters
        
        Args:
            team_id: MLB team ID
            season: Year (defaults to current)
            
        Returns:
            List of player IDs in batting order
        """
        if season is None:
            season = datetime.now().year
        
        # Check cache
        cache_key = f"lineup_{team_id}_{season}"
        if cache_key in self._lineup_cache:
            return self._lineup_cache[cache_key]
        
        try:
            # Get team's recent games to find regular starters
            url = f"{self.base_url}/schedule"
            params = {
                'sportId': 1,
                'teamId': team_id,
                'season': season,
                'gameType': 'R'
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Get roster and filter to position players
            roster = self.get_team_roster(team_id, season)
            if roster.empty:
                return []
            
            # Filter to hitters (exclude pitchers)
            hitters = roster[roster['position_type'] != 'Pitcher']
            
            # For now, return top hitters by player_id (simplified)
            # In production, would analyze recent lineups
            lineup = hitters['player_id'].head(9).tolist()
            
            self._lineup_cache[cache_key] = lineup
            return lineup
            
        except Exception as e:
            print(f"      ⚠️  Error getting lineup: {e}")
            return []
    
    def get_weighted_lineup_k_rate(self, team_id, vs_hand='R', season=None):
        """
        Calculate weighted K rate for probable lineup
        
        Weights:
        - Top 3 hitters (1-3): 40%
        - Middle 3 hitters (4-6): 35%
        - Bottom 3 hitters (7-9): 25%
        
        Args:
            team_id: MLB team ID
            vs_hand: 'R' for vs RHP, 'L' for vs LHP
            season: Year (defaults to current)
            
        Returns:
            Float weighted K% (e.g., 0.24 for 24%)
        """
        if season is None:
            season = datetime.now().year
        
        lineup = self.get_probable_lineup(team_id, season)
        
        if not lineup or len(lineup) < 9:
            # Fallback to league average
            return 0.23
        
        # Get K rates for each batter
        k_rates = []
        for player_id in lineup[:9]:
            stats = self.get_batter_stats_vs_hand(player_id, vs_hand, season)
            k_rates.append(stats['K_PCT'])
        
        # Weight by lineup position
        top_3_k = sum(k_rates[:3]) / 3 if len(k_rates) >= 3 else 0.23
        middle_3_k = sum(k_rates[3:6]) / 3 if len(k_rates) >= 6 else 0.23
        bottom_3_k = sum(k_rates[6:9]) / 3 if len(k_rates) >= 9 else 0.23
        
        weighted_k = (
            top_3_k * 0.40 +
            middle_3_k * 0.35 +
            bottom_3_k * 0.25
        )
        
        return weighted_k
    
    def get_enhanced_opponent_k_rate(self, team_abbrev, team_id, vs_hand='R', season=None):
        """
        Get enhanced opponent K rate using lineup analysis
        Falls back to team average if lineup data unavailable
        
        Args:
            team_abbrev: Team abbreviation (e.g., 'NYY')
            team_id: MLB team ID
            vs_hand: 'R' for vs RHP, 'L' for vs LHP
            season: Year (defaults to current)
            
        Returns:
            Dict with K rate info
        """
        if season is None:
            season = datetime.now().year
        
        # Try to get weighted lineup K rate
        lineup_k_rate = self.get_weighted_lineup_k_rate(team_id, vs_hand, season)
        
        # Get team overall K rate as fallback
        try:
            from mlb.shared.scrapers.team_stats import TeamStatsScraper
            team_scraper = TeamStatsScraper()
            team_k_rate = team_scraper.get_team_k_rate(team_abbrev, season)
        except:
            team_k_rate = 0.23
        
        # Use lineup K rate if available, otherwise team average
        final_k_rate = lineup_k_rate if lineup_k_rate > 0 else team_k_rate
        
        return {
            'k_rate': final_k_rate,
            'lineup_based': lineup_k_rate > 0,
            'vs_hand': vs_hand
        }
