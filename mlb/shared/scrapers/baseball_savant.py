"""
Baseball Savant Scraper
Fetch advanced batting stats including K% vs RHP/LHP
"""

import requests
import pandas as pd
from datetime import datetime
import time


class BaseballSavantScraper:
    """Scrape Baseball Savant for advanced batting statistics"""
    
    def __init__(self):
        self.base_url = "https://baseballsavant.mlb.com"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        self._cache = {}
    
    def get_team_k_rate_vs_hand(self, team_abbrev, vs_hand='R', season=None):
        """
        Get team K% vs RHP or LHP from Baseball Savant
        
        Args:
            team_abbrev: Team abbreviation (e.g., 'NYY', 'BOS')
            vs_hand: 'R' for vs RHP, 'L' for vs LHP
            season: Year (defaults to current)
            
        Returns:
            Float K% (e.g., 0.24 for 24%)
        """
        if season is None:
            season = datetime.now().year
        
        # Check cache
        cache_key = f"{team_abbrev}_{vs_hand}_{season}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            # Baseball Savant CSV export endpoint
            url = f"{self.base_url}/leaderboard/custom"
            
            params = {
                'year': season,
                'type': 'batter',
                'filter': '',
                'sort': 4,
                'sortDir': 'desc',
                'min': 'q',  # Qualified batters
                'selections': 'k_percent,',
                'chart': 'false',
                'x': 'k_percent',
                'y': 'k_percent',
                'r': 'no',
                'chartType': 'beeswarm',
                'csv': 'true'
            }
            
            # Add pitcher hand filter
            if vs_hand == 'R':
                params['filter'] = 'pitcher_hand=R'
            elif vs_hand == 'L':
                params['filter'] = 'pitcher_hand=L'
            
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            # Parse CSV
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
            
            # Filter to team - try multiple column names
            team_data = pd.DataFrame()
            
            for col in ['team_name_alt', 'team_name', 'team', 'batting_team']:
                if col in df.columns:
                    team_data = df[df[col].str.contains(team_abbrev, case=False, na=False)]
                    if not team_data.empty:
                        break
            
            if not team_data.empty:
                # Calculate team average K%
                k_rate = team_data['k_percent'].mean() / 100  # Convert from % to decimal
                self._cache[cache_key] = k_rate
                return k_rate
            else:
                # Fallback to league average
                return 0.23
                
        except Exception as e:
            print(f"      ⚠️  Baseball Savant error: {e}")
            return 0.23  # League average fallback
    
    def get_batter_k_rate_vs_hand(self, player_name, vs_hand='R', season=None):
        """
        Get individual batter K% vs RHP or LHP
        
        Args:
            player_name: Player name (e.g., 'Aaron Judge')
            vs_hand: 'R' for vs RHP, 'L' for vs LHP
            season: Year (defaults to current)
            
        Returns:
            Float K% (e.g., 0.18 for 18%)
        """
        if season is None:
            season = datetime.now().year
        
        # Check cache
        cache_key = f"{player_name}_{vs_hand}_{season}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            url = f"{self.base_url}/leaderboard/custom"
            
            params = {
                'year': season,
                'type': 'batter',
                'filter': '',
                'sort': 4,
                'sortDir': 'desc',
                'min': '10',  # Min 10 PA
                'selections': 'k_percent,',
                'chart': 'false',
                'csv': 'true'
            }
            
            # Add pitcher hand filter
            if vs_hand == 'R':
                params['filter'] = 'pitcher_hand=R'
            elif vs_hand == 'L':
                params['filter'] = 'pitcher_hand=L'
            
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            # Parse CSV
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
            
            # Find player (try different name formats)
            last_name = player_name.split()[-1]
            player_data = df[df['last_name'].str.contains(last_name, case=False, na=False)]
            
            if not player_data.empty:
                k_rate = player_data.iloc[0]['k_percent'] / 100
                self._cache[cache_key] = k_rate
                return k_rate
            else:
                return 0.23  # League average fallback
                
        except Exception as e:
            return 0.23
    
    def get_team_lineup_k_rates(self, team_abbrev, vs_hand='R', season=None):
        """
        Get K rates for team's regular lineup vs specific hand
        
        HYBRID APPROACH:
        1. Get team roster from MLB API (player IDs)
        2. Get K% data from Baseball Savant (by player ID)
        3. Calculate weighted team K%
        
        Args:
            team_abbrev: Team abbreviation
            vs_hand: 'R' for vs RHP, 'L' for vs LHP
            season: Year (defaults to current)
            
        Returns:
            Dict with lineup K rates and weighted average
        """
        if season is None:
            season = datetime.now().year
        
        try:
            # Step 1: Get ALL K% data from Baseball Savant (league-wide)
            url = f"{self.base_url}/leaderboard/custom"
            
            params = {
                'year': season,
                'type': 'batter',
                'filter': '',
                'sort': 4,
                'sortDir': 'desc',
                'min': '10',  # Min 10 PA
                'selections': 'k_percent,',
                'chart': 'false',
                'csv': 'true'
            }
            
            # Add pitcher hand filter
            if vs_hand == 'R':
                params['filter'] = 'pitcher_hand=R'
            elif vs_hand == 'L':
                params['filter'] = 'pitcher_hand=L'
            
            response = self.session.get(url, params=params, timeout=20)
            response.raise_for_status()
            
            # Parse CSV
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
            
            if df.empty or 'k_percent' not in df.columns:
                return {'weighted_k_rate': 0.23, 'source': 'empty_data'}
            
            # Step 2: Get team roster from MLB API
            team_id = self._get_team_id(team_abbrev)
            if not team_id:
                # Fallback to league average
                league_k = df['k_percent'].mean() / 100
                return {'weighted_k_rate': league_k, 'source': f'league_avg_vs_{vs_hand}HP'}
            
            roster_url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster"
            roster_response = self.session.get(roster_url, params={'season': season}, timeout=10)
            roster_response.raise_for_status()
            roster_data = roster_response.json()
            
            # Get position players (exclude pitchers)
            position_players = []
            for player in roster_data.get('roster', []):
                if player.get('position', {}).get('type') != 'Pitcher':
                    position_players.append(player['person']['id'])
            
            if not position_players:
                league_k = df['k_percent'].mean() / 100
                return {'weighted_k_rate': league_k, 'source': 'no_roster_data'}
            
            # Step 3: Match roster players with Baseball Savant K% data
            team_k_rates = []
            for player_id in position_players:
                player_data = df[df['player_id'] == player_id]
                if not player_data.empty:
                    team_k_rates.append(player_data.iloc[0]['k_percent'] / 100)
            
            if len(team_k_rates) >= 3:
                # Calculate weighted average (simple average for now)
                weighted_k = sum(team_k_rates) / len(team_k_rates)
                
                return {
                    'weighted_k_rate': weighted_k,
                    'source': 'baseball_savant',
                    'players_found': len(team_k_rates),
                    'vs_hand': vs_hand
                }
            else:
                # Not enough data, use league average
                league_k = df['k_percent'].mean() / 100
                return {'weighted_k_rate': league_k, 'source': f'insufficient_data_{len(team_k_rates)}_players'}
                
        except Exception as e:
            print(f"      ⚠️  Baseball Savant error: {str(e)[:100]}")
            return {'weighted_k_rate': 0.23, 'source': 'error_fallback'}
    
    def _get_team_id(self, team_abbrev):
        """Get MLB team ID from abbreviation"""
        team_map = {
            'ATH': 133, 'PIT': 134, 'SD': 135, 'SEA': 136, 'SF': 137,
            'MIL': 158, 'LAA': 108, 'AZ': 109, 'BAL': 110, 'BOS': 111,
            'CHC': 112, 'CIN': 113, 'CLE': 114, 'COL': 115, 'DET': 116,
            'HOU': 117, 'KC': 118, 'LAD': 119, 'WSH': 120, 'NYM': 121,
            'PHI': 143, 'STL': 138, 'TB': 139, 'TEX': 140, 'TOR': 141,
            'MIN': 142, 'CWS': 145, 'MIA': 146, 'NYY': 147, 'ATL': 144
        }
        return team_map.get(team_abbrev)
    
    def clear_cache(self):
        """Clear the cache (useful for daily updates)"""
        self._cache = {}
