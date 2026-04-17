"""
Usage boost calculator for players when teammates are injured
Adjusts projections based on increased opportunity
"""

import pandas as pd
import numpy as np


class UsageBoostCalculator:
    """Calculate usage boost when key teammates are injured"""
    
    # Position hierarchy - who benefits most when each position is out
    POSITION_HIERARCHY = {
        'PG': {'PG': 1.30, 'SG': 1.15, 'SF': 1.05, 'PF': 1.00, 'C': 1.00},  # Backup PG gets huge boost
        'SG': {'SG': 1.25, 'PG': 1.10, 'SF': 1.10, 'PF': 1.00, 'C': 1.00},
        'SF': {'SF': 1.25, 'SG': 1.10, 'PG': 1.10, 'PF': 1.05, 'C': 1.00},
        'PF': {'PF': 1.25, 'SF': 1.10, 'C': 1.10, 'SG': 1.05, 'PG': 1.00},
        'C': {'C': 1.25, 'PF': 1.10, 'SF': 1.05, 'SG': 1.00, 'PG': 1.00}
    }
    
    # Star player impact - when a star is out, everyone benefits
    STAR_USAGE_THRESHOLDS = {
        'superstar': 30.0,  # Usage rate > 30% = superstar
        'star': 25.0,       # Usage rate > 25% = star
        'starter': 20.0     # Usage rate > 20% = key starter
    }
    
    def __init__(self):
        self.team_injuries = {}
    
    def categorize_player(self, usage_rate, assists_per_game):
        """
        Categorize player importance
        
        Args:
            usage_rate: Player's usage percentage
            assists_per_game: Assists per game
        
        Returns:
            String: 'superstar', 'star', 'starter', or 'bench'
        """
        if usage_rate >= self.STAR_USAGE_THRESHOLDS['superstar'] or assists_per_game >= 8.0:
            return 'superstar'
        elif usage_rate >= self.STAR_USAGE_THRESHOLDS['star'] or assists_per_game >= 6.0:
            return 'star'
        elif usage_rate >= self.STAR_USAGE_THRESHOLDS['starter'] or assists_per_game >= 4.0:
            return 'starter'
        else:
            return 'bench'
    
    def calculate_position_boost(self, player_position, injured_position):
        """
        Calculate boost based on position overlap
        
        Args:
            player_position: Position of player getting boost
            injured_position: Position of injured player
        
        Returns:
            Float: Multiplier (e.g., 1.25 = 25% boost)
        """
        if injured_position not in self.POSITION_HIERARCHY:
            return 1.0
        
        return self.POSITION_HIERARCHY[injured_position].get(player_position, 1.0)
    
    def calculate_star_boost(self, injured_player_category, team_size=12):
        """
        Calculate boost when a star player is out
        
        Args:
            injured_player_category: 'superstar', 'star', 'starter', or 'bench'
            team_size: Number of active players
        
        Returns:
            Float: Multiplier for remaining players
        """
        # When a star is out, their usage/assists get distributed
        boosts = {
            'superstar': 1.20,  # 20% boost to team when superstar out
            'star': 1.12,       # 12% boost when star out
            'starter': 1.05,    # 5% boost when starter out
            'bench': 1.00       # No boost when bench player out
        }
        
        return boosts.get(injured_player_category, 1.0)
    
    def calculate_playmaker_boost(self, injured_assists_avg, player_assists_avg):
        """
        Special boost for playmakers when another playmaker is out
        
        Args:
            injured_assists_avg: Injured player's assists per game
            player_assists_avg: Current player's assists per game
        
        Returns:
            Float: Additional multiplier
        """
        # If a high-assist player is out, other playmakers benefit more
        if injured_assists_avg >= 7.0:  # Primary playmaker out
            if player_assists_avg >= 4.0:  # Secondary playmaker
                return 1.25  # 25% boost
            elif player_assists_avg >= 2.0:  # Tertiary playmaker
                return 1.15  # 15% boost
            else:
                return 1.05  # 5% boost
        elif injured_assists_avg >= 4.0:  # Secondary playmaker out
            if player_assists_avg >= 3.0:
                return 1.15
            else:
                return 1.05
        
        return 1.0
    
    def get_team_injuries(self, team_id, all_players_df, injury_data):
        """
        Get all injured players for a team
        
        Args:
            team_id: NBA team ID
            all_players_df: DataFrame with all players
            injury_data: DataFrame with injury information
        
        Returns:
            List of injured player dictionaries
        """
        if injury_data.empty:
            return []
        
        team_players = all_players_df[all_players_df['TEAM_ID'] == team_id]
        
        injured = []
        
        for _, player in team_players.iterrows():
            player_name = player['PLAYER_NAME']
            
            # Try multiple name matching strategies
            # 1. Full name match
            player_injuries = injury_data[
                injury_data['player_name'].str.contains(player_name, case=False, na=False, regex=False)
            ]
            
            # 2. If no match, try last name only
            if player_injuries.empty and ' ' in player_name:
                last_name = player_name.split()[-1]
                player_injuries = injury_data[
                    injury_data['player_name'].str.contains(last_name, case=False, na=False, regex=False)
                ]
            
            # 3. If still no match, try first + last initial
            if player_injuries.empty and ' ' in player_name:
                parts = player_name.split()
                first_name = parts[0]
                player_injuries = injury_data[
                    injury_data['player_name'].str.contains(first_name, case=False, na=False, regex=False)
                ]
            
            if not player_injuries.empty:
                status = str(player_injuries.iloc[0].get('status', ''))
                
                # Check for OUT status (various formats)
                is_out = any(keyword in status.upper() for keyword in ['OUT', 'DNP', 'SUSPEND'])
                
                if is_out:
                    # Get position - try multiple fields
                    position = 'G'  # Default
                    if 'POSITION' in player:
                        position = player['POSITION']
                    elif hasattr(player, 'get'):
                        position = player.get('POSITION', 'G')
                    
                    # Extract first position if multiple (e.g., "PG-SG" -> "PG")
                    if '-' in str(position):
                        position = position.split('-')[0]
                    
                    # Normalize position to standard format
                    position_map = {
                        'G': 'PG', 'F': 'SF', 'C': 'C',
                        'GUARD': 'PG', 'FORWARD': 'SF', 'CENTER': 'C'
                    }
                    position = position_map.get(position.upper(), position)
                    
                    usage_rate = 20.0
                    if 'USG_PCT' in player and player['USG_PCT'] is not None:
                        usage_rate = float(player['USG_PCT']) * 100
                    
                    assists_avg = float(player.get('AST', 0))
                    
                    injured.append({
                        'name': player_name,
                        'position': position,
                        'assists_avg': assists_avg,
                        'usage_rate': usage_rate,
                        'category': self.categorize_player(usage_rate, assists_avg)
                    })
        
        return injured
    
    def calculate_total_boost(self, player, team_injuries):
        """
        Calculate total usage boost for a player
        
        Args:
            player: Dictionary or Series with player info
            team_injuries: List of injured teammates
        
        Returns:
            Dictionary with boost info
        """
        if not team_injuries:
            return {
                'total_boost': 1.0,
                'boost_breakdown': [],
                'injured_teammates': 0
            }
        
        player_position = player.get('POSITION', 'G')
        player_assists = player.get('AST', 0)
        
        boosts = []
        total_multiplier = 1.0
        
        for injured in team_injuries:
            # Position-based boost
            position_boost = self.calculate_position_boost(
                player_position, 
                injured['position']
            )
            
            # Star player boost
            star_boost = self.calculate_star_boost(injured['category'])
            
            # Playmaker boost
            playmaker_boost = self.calculate_playmaker_boost(
                injured['assists_avg'],
                player_assists
            )
            
            # Combine boosts (multiplicative, but capped)
            injury_boost = position_boost * star_boost * playmaker_boost
            
            # Don't let single injury boost too much
            injury_boost = min(injury_boost, 1.40)  # Cap at 40% per injury
            
            boosts.append({
                'injured_player': injured['name'],
                'boost': injury_boost,
                'reason': f"{injured['category']} {injured['position']} out"
            })
            
            # Accumulate total boost (but with diminishing returns)
            total_multiplier *= injury_boost
        
        # Cap total boost at 1.75 (75% max increase)
        total_multiplier = min(total_multiplier, 1.75)
        
        return {
            'total_boost': total_multiplier,
            'boost_breakdown': boosts,
            'injured_teammates': len(team_injuries)
        }


def apply_usage_boosts(predictions_df, all_players_df, injury_data):
    """
    Apply usage boosts to all predictions based on injuries
    
    Args:
        predictions_df: DataFrame with predictions
        all_players_df: DataFrame with all player stats
        injury_data: DataFrame with injury information
    
    Returns:
        DataFrame with boosted predictions
    """
    calculator = UsageBoostCalculator()
    
    # Get injuries by team
    team_injuries_map = {}
    
    for team_id in predictions_df['TEAM_ID'].unique():
        team_injuries_map[team_id] = calculator.get_team_injuries(
            team_id, all_players_df, injury_data
        )
    
    # Apply boosts
    boosted_predictions = []
    
    for _, player in predictions_df.iterrows():
        team_id = player['TEAM_ID']
        team_injuries = team_injuries_map.get(team_id, [])
        
        # Calculate boost
        boost_info = calculator.calculate_total_boost(player, team_injuries)
        
        # Apply boost to projection
        original_projection = player['projected_assists']
        boosted_projection = original_projection * boost_info['total_boost']
        
        player_dict = player.to_dict()
        player_dict['projected_assists'] = boosted_projection
        player_dict['original_projection'] = original_projection
        player_dict['usage_boost'] = boost_info['total_boost']
        player_dict['injured_teammates'] = boost_info['injured_teammates']
        player_dict['boost_breakdown'] = boost_info['boost_breakdown']
        
        boosted_predictions.append(player_dict)
    
    return pd.DataFrame(boosted_predictions)


if __name__ == "__main__":
    print("=" * 70)
    print("USAGE BOOST CALCULATOR - TEST")
    print("=" * 70)
    
    calculator = UsageBoostCalculator()
    
    # Test 1: Position-based boost
    print("\n1. Position-Based Boost:")
    print("   When starting PG is out:")
    print(f"   - Backup PG boost: {calculator.calculate_position_boost('PG', 'PG'):.2f}x (30%)")
    print(f"   - SG boost: {calculator.calculate_position_boost('SG', 'PG'):.2f}x (15%)")
    print(f"   - SF boost: {calculator.calculate_position_boost('SF', 'PG'):.2f}x (5%)")
    
    # Test 2: Star player boost
    print("\n2. Star Player Impact:")
    print(f"   - Superstar out: {calculator.calculate_star_boost('superstar'):.2f}x (20%)")
    print(f"   - Star out: {calculator.calculate_star_boost('star'):.2f}x (12%)")
    print(f"   - Starter out: {calculator.calculate_star_boost('starter'):.2f}x (5%)")
    
    # Test 3: Playmaker boost
    print("\n3. Playmaker Boost:")
    print("   When 8 APG player is out:")
    print(f"   - Secondary playmaker (5 APG): {calculator.calculate_playmaker_boost(8.0, 5.0):.2f}x (25%)")
    print(f"   - Tertiary playmaker (3 APG): {calculator.calculate_playmaker_boost(8.0, 3.0):.2f}x (15%)")
    
    # Test 4: Real scenario
    print("\n4. Real Scenario Example:")
    print("   Scenario: Luka Doncic (PG, 10 AST, superstar) is OUT")
    print("   Impact on Kyrie Irving (PG, 6 AST):")
    
    # Simulate Kyrie's boost
    position_boost = calculator.calculate_position_boost('PG', 'PG')
    star_boost = calculator.calculate_star_boost('superstar')
    playmaker_boost = calculator.calculate_playmaker_boost(10.0, 6.0)
    
    total = position_boost * star_boost * playmaker_boost
    total = min(total, 1.40)  # Cap
    
    print(f"   - Position boost: {position_boost:.2f}x")
    print(f"   - Star boost: {star_boost:.2f}x")
    print(f"   - Playmaker boost: {playmaker_boost:.2f}x")
    print(f"   - TOTAL BOOST: {total:.2f}x ({(total-1)*100:.0f}% increase)")
    print(f"\n   If Kyrie projected 6.5 AST normally:")
    print(f"   → With Luka out: {6.5 * total:.1f} AST")
    
    print("\n" + "=" * 70)
    print("✅ Usage boost system ready!")
    print("=" * 70)
