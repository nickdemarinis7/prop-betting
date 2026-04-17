"""
Batter Statistics Scraper
Get individual batter K% vs pitcher handedness from Baseball Savant
"""

import requests
import pandas as pd
from datetime import datetime


class BatterStatsScraper:
    """Scrape batter statistics from Baseball Savant"""
    
    def __init__(self):
        self.base_url = "https://baseballsavant.mlb.com"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        self._cache = {}
    
    def get_batter_k_rate_vs_hand(self, batter_name, vs_hand='R', season=2026):
        """
        Get batter's K% vs RHP or LHP
        
        Args:
            batter_name: Full name of batter
            vs_hand: 'R' or 'L' for pitcher handedness
            season: Year
            
        Returns:
            float: K% (0.0 to 1.0)
        """
        cache_key = f"{batter_name}_{vs_hand}_{season}"
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            # For now, use a simplified approach via MLB Stats API
            # Baseball Savant requires more complex scraping
            
            # Try to get player ID from name
            player_id = self._get_player_id_from_name(batter_name)
            
            if not player_id:
                # Default to league average
                return 0.23
            
            # Get splits from MLB Stats API
            url = f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats"
            params = {
                'stats': 'statSplits',
                'group': 'hitting',
                'sitCodes': 'vr' if vs_hand == 'R' else 'vl',
                'season': season
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Parse K%
            splits = data.get('stats', [])
            if splits and len(splits) > 0:
                split_data = splits[0].get('splits', [])
                if split_data and len(split_data) > 0:
                    stats = split_data[0].get('stat', {})
                    
                    strikeouts = stats.get('strikeOuts', 0)
                    at_bats = stats.get('atBats', 0)
                    
                    if at_bats > 0:
                        k_rate = strikeouts / at_bats
                        self._cache[cache_key] = k_rate
                        return k_rate
            
            # No 2026 data - try 2025 as fallback
            if season == 2026:
                return self.get_batter_k_rate_vs_hand(batter_name, vs_hand, 2025)
            else:
                # Default to league average if no data
                return 0.23
            
        except Exception as e:
            # Return league average on error
            return 0.23
    
    def _get_player_id_from_name(self, player_name):
        """
        Get MLB player ID from name
        
        Args:
            player_name: Full name
            
        Returns:
            int: Player ID or None
        """
        try:
            # Search for player
            url = "https://statsapi.mlb.com/api/v1/people/search"
            params = {
                'names': player_name,
                'sportId': 1  # MLB
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            people = data.get('people', [])
            if people and len(people) > 0:
                return people[0].get('id')
            
            return None
            
        except Exception as e:
            return None
    
    def calculate_lineup_k_rate(self, lineup, vs_hand='R', season=2026):
        """
        Calculate weighted K% for a lineup
        
        Args:
            lineup: List of player names in batting order
            vs_hand: Pitcher handedness
            season: Year
            
        Returns:
            float: Weighted lineup K%
        """
        if not lineup or len(lineup) == 0:
            return 0.23  # League average
        
        # Batting order weights (top of order faces pitcher more)
        # Based on typical PA distribution in a game
        weights = [1.20, 1.15, 1.10, 1.05, 1.00, 0.95, 0.90, 0.85, 0.80]
        
        k_rates = []
        found_count = 0
        for i, batter in enumerate(lineup[:9]):  # Only first 9
            k_rate = self.get_batter_k_rate_vs_hand(batter, vs_hand, season)
            if k_rate != 0.23:  # Not default
                found_count += 1
            weight = weights[i] if i < len(weights) else 0.80
            k_rates.append(k_rate * weight)
        
        # Debug: print how many batters we found stats for
        if found_count < len(lineup[:9]) / 2:
            print(f"      ⚠️  Only found K% for {found_count}/{len(lineup[:9])} batters, using defaults")
        
        if k_rates:
            weighted_k_rate = sum(k_rates) / sum(weights[:len(k_rates)])
            return weighted_k_rate
        else:
            return 0.23


if __name__ == "__main__":
    # Test the scraper
    scraper = BatterStatsScraper()
    
    print("Testing Batter Stats Scraper")
    print("=" * 60)
    
    # Test individual batter
    test_batters = ['Aaron Judge', 'Juan Soto', 'Shohei Ohtani']
    
    for batter in test_batters:
        k_rate_vs_rhp = scraper.get_batter_k_rate_vs_hand(batter, 'R')
        k_rate_vs_lhp = scraper.get_batter_k_rate_vs_hand(batter, 'L')
        
        print(f"\n{batter}:")
        print(f"  vs RHP: {k_rate_vs_rhp:.1%}")
        print(f"  vs LHP: {k_rate_vs_lhp:.1%}")
    
    # Test lineup
    print("\n" + "=" * 60)
    print("Testing Lineup K%:")
    
    test_lineup = ['Aaron Judge', 'Juan Soto', 'Giancarlo Stanton', 
                   'Anthony Volpe', 'Gleyber Torres', 'Anthony Rizzo',
                   'DJ LeMahieu', 'Austin Wells', 'Oswaldo Cabrera']
    
    lineup_k_rate = scraper.calculate_lineup_k_rate(test_lineup, 'R')
    print(f"\nLineup K% vs RHP: {lineup_k_rate:.1%}")
