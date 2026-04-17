"""
Pitcher Statistics Scraper
Fetches pitcher game logs and season stats from Baseball Reference / MLB Stats API
"""

import pandas as pd
import requests
from datetime import datetime
import time


class PitcherStatsScraper:
    """Scrape pitcher statistics for strikeout predictions"""
    
    def __init__(self):
        self.base_url = "https://statsapi.mlb.com/api/v1"
        self.session = requests.Session()
    
    def get_season_stats(self, season=2026, min_starts=5):
        """
        Get season statistics for all pitchers from MLB Stats API
        
        Args:
            season: Year (default 2026)
            min_starts: Minimum starts to include
            
        Returns:
            DataFrame with pitcher season stats
        """
        print(f"   Fetching {season} pitcher stats from MLB API...")
        
        try:
            # FIXED: Get all teams' rosters, then fetch individual pitcher stats
            # The generic /stats endpoint was missing some teams (like STL)
            teams_url = f"{self.base_url}/teams"
            teams_response = self.session.get(teams_url, params={'sportId': 1, 'season': season}, timeout=10)
            teams_response.raise_for_status()
            teams_data = teams_response.json()
            
            all_pitchers = []
            
            # For each team, get their roster and then individual pitcher stats
            for team in teams_data.get('teams', []):
                team_id = team.get('id')
                team_abbr = team.get('abbreviation', 'UNK')
                
                try:
                    # Get team roster
                    roster_url = f"{self.base_url}/teams/{team_id}/roster"
                    roster_response = self.session.get(roster_url, params={'rosterType': 'active', 'season': season}, timeout=10)
                    roster_response.raise_for_status()
                    roster_data = roster_response.json()
                    
                    # For each pitcher on the roster, get their stats
                    for player in roster_data.get('roster', []):
                        position_type = player.get('position', {}).get('type')
                        # Include both Pitchers and Two-Way Players (like Ohtani)
                        if position_type in ['Pitcher', 'Two-Way Player']:
                            person = player.get('person', {})
                            pitcher_id = person.get('id')
                            pitcher_name = person.get('fullName', 'Unknown')
                            
                            # Get this pitcher's season stats
                            stats_url = f"{self.base_url}/people/{pitcher_id}/stats"
                            stats_params = {
                                'stats': 'season',
                                'season': season,
                                'group': 'pitching',
                                'gameType': 'R'
                            }
                            
                            try:
                                stats_response = self.session.get(stats_url, params=stats_params, timeout=10)
                                stats_response.raise_for_status()
                                stats_data = stats_response.json()
                                
                                # Parse stats
                                for stat_group in stats_data.get('stats', []):
                                    for split in stat_group.get('splits', []):
                                        stats = split.get('stat', {})
                                        
                                        gs = stats.get('gamesStarted', 0)
                                        if gs >= min_starts:
                                            ip = float(stats.get('inningsPitched', '0'))
                                            so = stats.get('strikeOuts', 0)
                                            
                                            # Calculate K/9 and K%
                                            k9 = (so / ip * 9) if ip > 0 else 0
                                            batters_faced = stats.get('battersFaced', 1)
                                            k_pct = so / batters_faced if batters_faced > 0 else 0
                                            
                                            # Get pitcher handedness
                                            pitch_hand = person.get('pitchHand', {}).get('code', 'R')
                                            
                                            all_pitchers.append({
                                                'pitcher_id': pitcher_id,
                                                'pitcher_name': pitcher_name,
                                                'team': team_abbr,
                                                'hand': pitch_hand,
                                                'GS': gs,
                                                'IP': ip,
                                                'SO': so,
                                                'K9': k9,
                                                'K_PCT': k_pct,
                                                'BB': stats.get('baseOnBalls', 0),
                                                'ERA': stats.get('era', 0),
                                                'WHIP': stats.get('whip', 0)
                                            })
                            except:
                                # Skip pitchers with no stats
                                continue
                except:
                    # Skip teams that error
                    continue
            
            if all_pitchers:
                print(f"   ✓ Found {len(all_pitchers)} pitchers with {min_starts}+ starts")
                return pd.DataFrame(all_pitchers)
            else:
                print(f"   ⚠️  No pitcher data found for {season}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"   ❌ Error fetching season stats: {e}")
            print(f"   Using fallback sample data...")
            # Fallback to sample data
            return self._get_sample_season_stats(min_starts)
    
    def get_game_logs(self, pitcher_id, n_games=20, season=None):
        """
        Get recent game logs for a pitcher from MLB Stats API
        
        Args:
            pitcher_id: MLB pitcher ID
            n_games: Number of recent games to fetch
            season: Specific season year (defaults to current year)
            
        Returns:
            DataFrame with game-by-game stats
        """
        try:
            from datetime import datetime
            if season is None:
                season = datetime.now().year
            current_year = season
            
            # Ensure pitcher_id is an integer (no decimals)
            try:
                pitcher_id = int(float(pitcher_id))
            except (ValueError, TypeError):
                print(f"   ⚠️  Invalid pitcher ID: {pitcher_id}")
                return pd.DataFrame()
            
            url = f"{self.base_url}/people/{pitcher_id}/stats"
            params = {
                'stats': 'gameLog',
                'season': current_year,
                'group': 'pitching',
                'gameType': 'R',
                'sportId': 1
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            games = []
            for stat_group in data.get('stats', []):
                for split in stat_group.get('splits', [])[:n_games]:
                    stats = split.get('stat', {})
                    game_info = split.get('game', {})
                    opponent = split.get('opponent', {})
                    
                    ip = float(stats.get('inningsPitched', '0'))
                    so = stats.get('strikeOuts', 0)
                    pitches = stats.get('numberOfPitches', 1)
                    
                    # Calculate K/9 and K%
                    k9 = (so / ip * 9) if ip > 0 else 0
                    k_pct = so / pitches if pitches > 0 else 0
                    
                    games.append({
                        'game_date': split.get('date'),
                        'opponent': opponent.get('abbreviation', 'UNK'),
                        'is_home': 1 if split.get('isHome') else 0,
                        'IP': ip,
                        'SO': so,
                        'BB': stats.get('baseOnBalls', 0),
                        'H': stats.get('hits', 0),
                        'ER': stats.get('earnedRuns', 0),
                        'pitches': pitches,
                        'K9': k9,
                        'K_PCT': k_pct,
                        'ballpark': game_info.get('venue', {}).get('name', 'Unknown'),
                        'opp_k_rate': 0.22  # Would fetch from team stats
                    })
            
            if games:
                return pd.DataFrame(games)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            print(f"   ⚠️  Error fetching game logs for pitcher {pitcher_id}: {e}")
            return self._get_sample_game_logs(n_games)
    
    def get_pitcher_by_name(self, pitcher_name):
        """
        Look up pitcher by name
        
        Args:
            pitcher_name: Full name of pitcher
            
        Returns:
            Dict with pitcher info or None
        """
        # Placeholder - would search MLB API
        # For now, return sample pitcher data
        sample_pitcher = {
            'pitcher_id': 12345,
            'pitcher_name': pitcher_name,
            'team': 'NYY',
            'throws': 'R',  # Right-handed
            'K9': 9.5,
            'K_PCT': 0.27,
            'ERA': 3.25,
            'WHIP': 1.08
        }
        
        return sample_pitcher
    
    def get_pitcher_splits(self, pitcher_id, split_type='home_away'):
        """
        Get pitcher splits (home/away, vs LHB/RHB, etc.)
        
        Args:
            pitcher_id: MLB pitcher ID
            split_type: Type of split ('home_away', 'handedness', 'month')
            
        Returns:
            DataFrame with split stats
        """
        # Placeholder for splits data
        if split_type == 'home_away':
            splits = {
                'split': ['Home', 'Away'],
                'GS': [8, 7],
                'SO': [55, 50],
                'K9': [10.2, 9.8],
                'ERA': [2.95, 3.45]
            }
        else:
            splits = {
                'split': ['vs LHB', 'vs RHB'],
                'PA': [120, 180],
                'K': [35, 70],
                'K_PCT': [0.29, 0.39]
            }
        
        return pd.DataFrame(splits)
    
    def calculate_rolling_features(self, game_logs, windows=[3, 5, 10]):
        """
        Calculate rolling averages from game logs
        
        Args:
            game_logs: DataFrame of recent games
            windows: List of rolling window sizes
            
        Returns:
            Dict of rolling features
        """
        features = {}
        
        for window in windows:
            if len(game_logs) >= window:
                recent = game_logs.head(window)
                features[f'k_last_{window}'] = recent['SO'].mean()
                features[f'k9_last_{window}'] = recent['K9'].mean()
                features[f'ip_last_{window}'] = recent['IP'].mean()
                features[f'bb_last_{window}'] = recent['BB'].mean()
        
        # Trend (improving or declining)
        if len(game_logs) >= 10:
            first_half = game_logs.iloc[5:10]['SO'].mean()
            second_half = game_logs.iloc[0:5]['SO'].mean()
            features['k_trend'] = second_half - first_half
        
        # Consistency (std dev)
        if len(game_logs) >= 5:
            features['k_std'] = game_logs.head(10)['SO'].std()
        
        return features
    
    def _get_sample_season_stats(self, min_starts=5):
        """Fallback sample data for testing"""
        sample_data = {
            'pitcher_id': [1, 2, 3],
            'pitcher_name': ['Sample Pitcher 1', 'Sample Pitcher 2', 'Sample Pitcher 3'],
            'team': ['NYY', 'LAD', 'HOU'],
            'hand': ['R', 'L', 'R'],
            'GS': [15, 18, 12],
            'IP': [95.1, 110.2, 75.0],
            'SO': [105, 128, 88],
            'K9': [9.9, 10.4, 10.6],
            'K_PCT': [0.28, 0.30, 0.29],
            'BB': [25, 30, 22],
            'ERA': [3.15, 2.85, 3.45],
            'WHIP': [1.05, 0.98, 1.12]
        }
        df = pd.DataFrame(sample_data)
        return df[df['GS'] >= min_starts]
    
    def _get_sample_game_logs(self, n_games=20):
        """Fallback sample game logs for testing"""
        sample_games = []
        for i in range(min(n_games, 10)):
            game = {
                'game_date': f'2026-04-{20-i:02d}',
                'opponent': ['BOS', 'TB', 'TOR', 'BAL', 'CLE'][i % 5],
                'is_home': i % 2,
                'IP': 6.0 + (i % 3) * 0.5,
                'SO': 5 + (i % 4),
                'BB': 2 + (i % 3),
                'H': 4 + (i % 5),
                'ER': 2 + (i % 4),
                'pitches': 95 + (i % 15),
                'K9': 7.5 + (i % 3) * 0.5,
                'K_PCT': 0.24 + (i % 5) * 0.01,
                'ballpark': 'Yankee Stadium',
                'opp_k_rate': 0.22
            }
            sample_games.append(game)
        return pd.DataFrame(sample_games)


# Example usage
if __name__ == "__main__":
    scraper = PitcherStatsScraper()
    
    # Get season stats
    pitchers = scraper.get_season_stats(season=2026)
    print(f"Found {len(pitchers)} pitchers")
    print(pitchers.head())
    
    # Get game logs for a pitcher
    game_logs = scraper.get_game_logs(pitcher_id=1, n_games=10)
    print(f"\nGame logs:")
    print(game_logs.head())
    
    # Calculate rolling features
    features = scraper.calculate_rolling_features(game_logs)
    print(f"\nRolling features:")
    for key, value in features.items():
        print(f"  {key}: {value:.2f}")
