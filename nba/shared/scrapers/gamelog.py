"""
Game-by-game scraper for building proper predictive models
Fetches individual game logs instead of season averages
"""

import pandas as pd
from datetime import datetime, timedelta
import time
from nba_api.stats.endpoints import (
    playergamelog,
    leaguegamefinder,
    teamgamelog,
    scoreboardv2
)
from nba_api.stats.static import teams, players


class GameLogScraper:
    """Scraper for individual game logs to build predictive features"""
    
    def __init__(self):
        self.teams_data = teams.get_teams()
        self.current_season = '2025-26'
    
    def get_todays_games(self, game_date=None):
        """Get today's games with team matchups"""
        if game_date is None:
            game_date = datetime.now().strftime('%m/%d/%Y')
        else:
            dt = datetime.strptime(game_date, '%Y-%m-%d')
            game_date = dt.strftime('%m/%d/%Y')
        
        try:
            scoreboard = scoreboardv2.ScoreboardV2(game_date=game_date)
            games_df = scoreboard.get_data_frames()[0]
            return games_df
        except Exception as e:
            print(f"Error fetching games: {e}")
            return pd.DataFrame()
    
    def get_player_game_logs(self, player_id, season='2025-26', season_type='Regular Season'):
        """
        Get individual game logs for a player
        
        Returns:
            DataFrame with each row being one game
        """
        try:
            time.sleep(0.6)  # Rate limiting
            gamelog = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=season,
                season_type_all_star=season_type
            )
            df = gamelog.get_data_frames()[0]
            return df
        except Exception as e:
            print(f"Error fetching game log for player {player_id}: {e}")
            return pd.DataFrame()
    
    def get_recent_games(self, player_id, n_games=10):
        """Get player's last N games (regular season + playoffs)"""
        reg = self.get_player_game_logs(player_id, season=self.current_season, season_type='Regular Season')
        playoffs = self.get_player_game_logs(player_id, season=self.current_season, season_type='Playoffs')
        
        # Combine both, preferring the most recent games regardless of type
        gamelog = pd.concat([reg, playoffs], ignore_index=True) if not playoffs.empty else reg
        
        if gamelog.empty:
            return pd.DataFrame()
        
        # Convert GAME_DATE to datetime for proper sorting
        if 'GAME_DATE' in gamelog.columns:
            gamelog['GAME_DATE_PARSED'] = pd.to_datetime(gamelog['GAME_DATE'])
            gamelog = gamelog.sort_values('GAME_DATE_PARSED', ascending=False).head(n_games)
            gamelog = gamelog.drop('GAME_DATE_PARSED', axis=1)  # Clean up temp column
        else:
            # Fallback if no date column
            gamelog = gamelog.head(n_games)
        
        return gamelog
    
    def calculate_rolling_features(self, gamelog_df, windows=[5, 10]):
        """
        Calculate rolling averages and trends from game logs
        
        Args:
            gamelog_df: DataFrame with individual games
            windows: List of window sizes for rolling averages
        
        Returns:
            Dictionary of features
        """
        if gamelog_df.empty or len(gamelog_df) < 3:
            return None
        
        # Sort by date (oldest first for rolling calculations)
        df = gamelog_df.sort_values('GAME_DATE', ascending=True).copy()
        
        features = {}
        
        # Recent averages for different windows
        for window in windows:
            if len(df) >= window:
                recent = df.tail(window)
                features[f'ast_last_{window}'] = recent['AST'].mean()
                features[f'min_last_{window}'] = recent['MIN'].mean()
                features[f'usg_last_{window}'] = recent['USG_PCT'].mean() if 'USG_PCT' in recent.columns else 0
                features[f'pts_last_{window}'] = recent['PTS'].mean()
                features[f'tov_last_{window}'] = recent['TOV'].mean()
            else:
                features[f'ast_last_{window}'] = df['AST'].mean()
                features[f'min_last_{window}'] = df['MIN'].mean()
                features[f'usg_last_{window}'] = 0
                features[f'pts_last_{window}'] = df['PTS'].mean()
                features[f'tov_last_{window}'] = df['TOV'].mean()
        
        # Trend (is player improving or declining?) - for both AST and PTS
        if len(df) >= 10:
            first_half_ast = df.head(5)['AST'].mean()
            second_half_ast = df.tail(5)['AST'].mean()
            features['ast_trend'] = second_half_ast - first_half_ast
            
            first_half_pts = df.head(5)['PTS'].mean()
            second_half_pts = df.tail(5)['PTS'].mean()
            features['pts_trend'] = second_half_pts - first_half_pts
        else:
            features['ast_trend'] = 0
            features['pts_trend'] = 0
        
        # Consistency (standard deviation) - for both AST and PTS
        features['ast_std'] = df['AST'].std()
        features['ast_consistency'] = 1 / (df['AST'].std() + 0.1)  # Higher = more consistent
        features['pts_std'] = df['PTS'].std()
        features['pts_consistency'] = 1 / (df['PTS'].std() + 0.1)
        
        # Home/Away splits - for both AST and PTS
        if 'MATCHUP' in df.columns:
            home_games = df[df['MATCHUP'].str.contains('vs.', na=False)]
            away_games = df[df['MATCHUP'].str.contains('@', na=False)]
            
            features['ast_home_avg'] = home_games['AST'].mean() if len(home_games) > 0 else df['AST'].mean()
            features['ast_away_avg'] = away_games['AST'].mean() if len(away_games) > 0 else df['AST'].mean()
            features['pts_home_avg'] = home_games['PTS'].mean() if len(home_games) > 0 else df['PTS'].mean()
            features['pts_away_avg'] = away_games['PTS'].mean() if len(away_games) > 0 else df['PTS'].mean()
        else:
            features['ast_home_avg'] = df['AST'].mean()
            features['ast_away_avg'] = df['AST'].mean()
            features['pts_home_avg'] = df['PTS'].mean()
            features['pts_away_avg'] = df['PTS'].mean()
        
        # Days since last game (rest factor)
        if 'GAME_DATE' in df.columns and len(df) >= 2:
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
            df = df.sort_values('GAME_DATE', ascending=False)
            last_game = df.iloc[0]['GAME_DATE']
            second_last = df.iloc[1]['GAME_DATE']
            features['days_rest'] = (last_game - second_last).days
        else:
            features['days_rest'] = 1
        
        # Recent high/low - for both AST and PTS
        features['ast_recent_high'] = df.tail(10)['AST'].max() if len(df) >= 10 else df['AST'].max()
        features['ast_recent_low'] = df.tail(10)['AST'].min() if len(df) >= 10 else df['AST'].min()
        features['pts_recent_high'] = df.tail(10)['PTS'].max() if len(df) >= 10 else df['PTS'].max()
        features['pts_recent_low'] = df.tail(10)['PTS'].min() if len(df) >= 10 else df['PTS'].min()
        
        # ADVANCED FEATURES
        
        # 1. Weighted recent performance (emphasize most recent games) - for both AST and PTS
        if len(df) >= 10:
            # Assists
            l3_avg_ast = df.tail(3)['AST'].mean()
            l5_avg_ast = df.tail(5)['AST'].mean()
            l10_avg_ast = df.tail(10)['AST'].mean()
            
            features['ast_weighted_recent'] = (l3_avg_ast * 0.5) + (l5_avg_ast * 0.3) + (l10_avg_ast * 0.2)
            features['ast_momentum'] = (l3_avg_ast - l10_avg_ast) / max(l10_avg_ast, 1.0) if l10_avg_ast > 0 else 0
            
            # Points
            l3_avg_pts = df.tail(3)['PTS'].mean()
            l5_avg_pts = df.tail(5)['PTS'].mean()
            l10_avg_pts = df.tail(10)['PTS'].mean()
            
            features['pts_weighted_recent'] = (l3_avg_pts * 0.5) + (l5_avg_pts * 0.3) + (l10_avg_pts * 0.2)
            features['pts_momentum'] = (l3_avg_pts - l10_avg_pts) / max(l10_avg_pts, 1.0) if l10_avg_pts > 0 else 0
        else:
            features['ast_weighted_recent'] = df['AST'].mean()
            features['ast_momentum'] = 0
            features['pts_weighted_recent'] = df['PTS'].mean()
            features['pts_momentum'] = 0
        
        # 2. Usage rate (if available)
        if 'USG_PCT' in df.columns:
            features['usage_rate'] = df.tail(10)['USG_PCT'].mean() if len(df) >= 10 else df['USG_PCT'].mean()
        else:
            features['usage_rate'] = 20.0  # Default
        
        # 3. Minutes share (% of game played)
        avg_minutes = df.tail(10)['MIN'].mean() if len(df) >= 10 else df['MIN'].mean()
        features['minutes_share'] = min(avg_minutes / 48.0, 1.0)
        
        # 4. Potential assists/points (estimate if not available)
        # Typically ~1.3x actual assists
        features['potential_assists'] = features.get('ast_last_10', df['AST'].mean()) * 1.3
        # For points, estimate based on usage and minutes
        features['potential_points'] = features.get('pts_last_10', df['PTS'].mean()) * 1.2
        
        return features
    
    def get_team_pace(self, team_id, season='2025-26', last_n_games=10):
        """Get team's recent pace"""
        try:
            time.sleep(0.6)
            teamlog = teamgamelog.TeamGameLog(
                team_id=team_id,
                season=season,
                season_type_all_star='Regular Season'
            )
            df = teamlog.get_data_frames()[0]
            
            if df.empty:
                return 100.0
            
            # Calculate pace from recent games
            recent = df.head(last_n_games)
            # Pace estimate: possessions per 48 minutes
            # Simplified: (FGA + 0.44*FTA + TOV - OREB) per game
            if all(col in recent.columns for col in ['FGA', 'FTA', 'TOV', 'OREB']):
                pace = (recent['FGA'] + 0.44 * recent['FTA'] + recent['TOV'] - recent['OREB']).mean()
                return pace
            
            return 100.0
        except Exception as e:
            return 100.0
    
    def build_training_data(self, player_ids, season='2025-26', min_games=15):
        """
        Build training dataset from game logs
        
        For each game, create features from previous games to predict that game's assists
        
        Args:
            player_ids: List of player IDs to include
            season: NBA season
            min_games: Minimum games played to include player
        
        Returns:
            DataFrame with features (X) and target (y)
        """
        all_data = []
        
        print(f"Building training data from game logs...")
        print(f"Processing {len(player_ids)} players...")
        
        for i, player_id in enumerate(player_ids):
            if i % 50 == 0:
                print(f"  Progress: {i}/{len(player_ids)} players")
            
            # Get regular season + playoff games for this player
            reg = self.get_player_game_logs(player_id, season=season, season_type='Regular Season')
            playoffs = self.get_player_game_logs(player_id, season=season, season_type='Playoffs')
            
            # Tag source before combining
            if not reg.empty:
                reg['_is_playoff'] = 0
            if not playoffs.empty:
                playoffs['_is_playoff'] = 1
            
            gamelog = pd.concat([reg, playoffs], ignore_index=True) if not playoffs.empty else reg
            
            if gamelog.empty or len(gamelog) < min_games:
                continue
            
            # Sort by date
            gamelog['_date_parsed'] = pd.to_datetime(gamelog['GAME_DATE'])
            gamelog = gamelog.sort_values('_date_parsed', ascending=True)
            
            # For each game (except first 10), use previous games as features
            for idx in range(10, len(gamelog)):
                # Features: previous games
                previous_games = gamelog.iloc[:idx]
                
                # Target: this game's assists
                target_game = gamelog.iloc[idx]
                target_assists = target_game['AST']
                
                # Calculate features from previous games
                features = self.calculate_rolling_features(previous_games.tail(20))
                
                if features is None:
                    continue
                
                # Add player and game context
                features['player_id'] = player_id
                features['game_date'] = target_game['GAME_DATE']
                features['is_home'] = 1 if 'vs.' in str(target_game.get('MATCHUP', '')) else 0
                features['is_playoff'] = int(target_game.get('_is_playoff', 0))
                features['target_assists'] = target_assists
                
                all_data.append(features)
        
        print(f"  ✓ Created {len(all_data)} training samples")
        
        if not all_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_data)
        return df


if __name__ == "__main__":
    print("=" * 70)
    print("Testing Game Log Scraper")
    print("=" * 70)
    
    scraper = GameLogScraper()
    
    # Test with a known player (e.g., Trae Young - ID: 1629027)
    print("\n1. Testing individual game log fetch...")
    player_id = 1629027  # Trae Young
    
    gamelog = scraper.get_recent_games(player_id, n_games=10)
    
    if not gamelog.empty:
        print(f"   ✓ Retrieved {len(gamelog)} recent games")
        print(f"\n   Last 5 games:")
        recent = gamelog.head(5)[['GAME_DATE', 'MATCHUP', 'MIN', 'PTS', 'AST', 'REB']]
        print(recent.to_string(index=False))
        
        # Test feature calculation
        print(f"\n2. Testing rolling feature calculation...")
        features = scraper.calculate_rolling_features(gamelog)
        
        if features:
            print(f"   ✓ Calculated {len(features)} features")
            print(f"\n   Sample features:")
            for key, value in list(features.items())[:10]:
                print(f"   {key:20} = {value:.2f}")
    else:
        print("   Could not fetch game log")
    
    print("\n" + "=" * 70)
