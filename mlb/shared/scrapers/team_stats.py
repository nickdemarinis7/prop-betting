"""
Team Statistics Scraper
Fetch team-level batting statistics including strikeout rates
"""

import requests
import pandas as pd
from datetime import datetime


class TeamStatsScraper:
    """Scrape team batting statistics from MLB Stats API"""
    
    def __init__(self):
        self.base_url = "https://statsapi.mlb.com/api/v1"
        self.session = requests.Session()
    
    def get_team_batting_stats(self, season=None):
        """
        Get team batting statistics including K rates
        
        Args:
            season: Year (defaults to current year)
            
        Returns:
            DataFrame with team batting stats
        """
        if season is None:
            season = datetime.now().year
        
        try:
            url = f"{self.base_url}/stats"
            params = {
                'stats': 'season',
                'season': season,
                'group': 'hitting',
                'gameType': 'R',
                'sportId': 1,
                'limit': 30
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            teams = []
            for stat_group in data.get('stats', []):
                for split in stat_group.get('splits', []):
                    team_info = split.get('team', {})
                    stats = split.get('stat', {})
                    
                    plate_appearances = stats.get('plateAppearances', 1)
                    strikeouts = stats.get('strikeOuts', 0)
                    
                    teams.append({
                        'team_id': team_info.get('id'),
                        'team': team_info.get('abbreviation', 'UNK'),
                        'team_name': team_info.get('name', 'Unknown'),
                        'PA': plate_appearances,
                        'SO': strikeouts,
                        'K_PCT': strikeouts / plate_appearances if plate_appearances > 0 else 0.22,
                        'AVG': stats.get('avg', 0),
                        'OBP': stats.get('obp', 0),
                        'SLG': stats.get('slg', 0),
                        'OPS': stats.get('ops', 0),
                        'HR': stats.get('homeRuns', 0),
                        'BB': stats.get('baseOnBalls', 0)
                    })
            
            if teams:
                df = pd.DataFrame(teams)
                print(f"   ✓ Found {len(df)} teams with batting stats")
                return df
            else:
                print(f"   ⚠️  No team batting data found for {season}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"   ❌ Error fetching team stats: {e}")
            return self._get_fallback_team_stats()
    
    def get_team_k_rate(self, team_abbrev, season=None):
        """
        Get strikeout rate for a specific team
        
        Args:
            team_abbrev: Team abbreviation (e.g., 'NYY', 'BOS')
            season: Year (defaults to current year)
            
        Returns:
            Float K% (e.g., 0.23 for 23%)
        """
        team_stats = self.get_team_batting_stats(season=season)
        
        if team_stats.empty:
            return 0.22  # League average fallback
        
        team_row = team_stats[team_stats['team'] == team_abbrev]
        
        if team_row.empty:
            return 0.22  # League average fallback
        
        return team_row.iloc[0]['K_PCT']
    
    def get_team_splits_vs_handedness(self, team_abbrev, vs_hand='R', season=None):
        """
        Get team batting stats vs RHP or LHP
        
        Args:
            team_abbrev: Team abbreviation
            vs_hand: 'R' for vs RHP, 'L' for vs LHP
            season: Year (defaults to current year)
            
        Returns:
            Dict with splits stats
        """
        if season is None:
            season = datetime.now().year
        
        try:
            # Get team ID first
            team_stats = self.get_team_batting_stats(season=season)
            if team_stats.empty:
                return {'K_PCT': 0.22}
            
            team_row = team_stats[team_stats['team'] == team_abbrev]
            if team_row.empty:
                return {'K_PCT': 0.22}
            
            team_id = team_row.iloc[0]['team_id']
            
            # Fetch splits data
            url = f"{self.base_url}/stats"
            params = {
                'stats': 'vsLefty' if vs_hand == 'L' else 'vsRighty',
                'season': season,
                'group': 'hitting',
                'gameType': 'R',
                'sportId': 1,
                'teamId': team_id
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            for stat_group in data.get('stats', []):
                for split in stat_group.get('splits', []):
                    stats = split.get('stat', {})
                    
                    plate_appearances = stats.get('plateAppearances', 1)
                    strikeouts = stats.get('strikeOuts', 0)
                    
                    return {
                        'K_PCT': strikeouts / plate_appearances if plate_appearances > 0 else 0.22,
                        'AVG': stats.get('avg', 0),
                        'OBP': stats.get('obp', 0),
                        'SLG': stats.get('slg', 0)
                    }
            
            return {'K_PCT': 0.22}
            
        except Exception as e:
            print(f"   ⚠️  Error fetching splits for {team_abbrev}: {e}")
            return {'K_PCT': 0.22}
    
    def _get_fallback_team_stats(self):
        """Fallback team stats with league averages"""
        teams = [
            'AZ', 'ATL', 'BAL', 'BOS', 'CHC', 'CWS', 'CIN', 'CLE', 'COL', 'DET',
            'HOU', 'KC', 'LAA', 'LAD', 'MIA', 'MIL', 'MIN', 'NYM', 'NYY', 'ATH',
            'PHI', 'PIT', 'SD', 'SF', 'SEA', 'STL', 'TB', 'TEX', 'TOR', 'WSH'
        ]
        
        data = []
        for team in teams:
            data.append({
                'team_id': 0,
                'team': team,
                'team_name': team,
                'PA': 6000,
                'SO': 1320,
                'K_PCT': 0.22,  # League average
                'AVG': 0.250,
                'OBP': 0.320,
                'SLG': 0.420,
                'OPS': 0.740,
                'HR': 180,
                'BB': 500
            })
        
        return pd.DataFrame(data)


# Example usage
if __name__ == "__main__":
    scraper = TeamStatsScraper()
    
    # Get all team stats
    teams = scraper.get_team_batting_stats(season=2025)
    print("\nTeam K% Rankings:")
    print(teams[['team', 'K_PCT']].sort_values('K_PCT', ascending=False).head(10))
    
    # Get specific team K rate
    yankees_k_rate = scraper.get_team_k_rate('NYY', season=2025)
    print(f"\nYankees K%: {yankees_k_rate:.1%}")
    
    # Get splits vs RHP
    splits = scraper.get_team_splits_vs_handedness('NYY', vs_hand='R', season=2025)
    print(f"\nYankees vs RHP K%: {splits['K_PCT']:.1%}")
