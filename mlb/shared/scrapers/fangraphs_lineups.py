"""
Fangraphs Lineup Scraper
Get projected lineups from Fangraphs depth charts
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime


class FangraphsLineupScraper:
    """Scrape projected lineups from Fangraphs"""
    
    def __init__(self):
        self.base_url = "https://www.fangraphs.com/api/depth-charts/roster-resource"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        
        # Team abbreviation mapping (MLB API -> Fangraphs)
        self.team_mapping = {
            'AZ': 'diamondbacks', 'ATL': 'braves', 'BAL': 'orioles', 'BOS': 'red-sox',
            'CHC': 'cubs', 'CWS': 'white-sox', 'CIN': 'reds', 'CLE': 'guardians',
            'COL': 'rockies', 'DET': 'tigers', 'HOU': 'astros', 'KC': 'royals',
            'LAA': 'angels', 'LAD': 'dodgers', 'MIA': 'marlins', 'MIL': 'brewers',
            'MIN': 'twins', 'NYM': 'mets', 'NYY': 'yankees', 'OAK': 'athletics',
            'PHI': 'phillies', 'PIT': 'pirates', 'SD': 'padres', 'SF': 'giants',
            'SEA': 'mariners', 'STL': 'cardinals', 'TB': 'rays', 'TEX': 'rangers',
            'TOR': 'blue-jays', 'WSH': 'nationals',
            # Alternate abbreviations
            'ARI': 'diamondbacks', 'CHW': 'white-sox', 'CWS': 'white-sox',
            'KCR': 'royals', 'SDP': 'padres', 'SFG': 'giants', 'TBR': 'rays',
            'WSN': 'nationals'
        }
    
    def get_projected_lineup(self, team_abbrev):
        """
        Get projected starting lineup for a team
        
        Args:
            team_abbrev: Team abbreviation (e.g., 'NYY', 'BOS')
            
        Returns:
            List of player names in batting order (1-9)
        """
        try:
            # Convert team abbreviation
            team_name = self.team_mapping.get(team_abbrev.upper())
            if not team_name:
                print(f"   ⚠️  Unknown team: {team_abbrev}")
                return []
            
            # Fetch depth chart data
            url = f"https://www.fangraphs.com/roster-resource/depth-charts/{team_name}"
            
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find the batting order section
            lineup = []
            
            # Look for depth chart table
            tables = soup.find_all('table', class_='depth-chart')
            
            if not tables:
                # Try alternate parsing
                lineup = self._parse_alternate_format(soup)
            else:
                # Parse standard depth chart
                for table in tables:
                    rows = table.find_all('tr')
                    for row in rows:
                        position = row.find('td', class_='position')
                        if position and position.text.strip() in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
                            player_cell = row.find('td', class_='player-name')
                            if player_cell:
                                player_name = player_cell.text.strip()
                                lineup.append(player_name)
            
            if lineup:
                print(f"   ✓ Found projected lineup for {team_abbrev}: {len(lineup)} players")
                return lineup[:9]  # Return top 9
            else:
                print(f"   ⚠️  No lineup found for {team_abbrev}")
                return []
                
        except Exception as e:
            print(f"   ⚠️  Error fetching lineup for {team_abbrev}: {e}")
            return []
    
    def _parse_alternate_format(self, soup):
        """Try alternate parsing method"""
        lineup = []
        
        # Look for any batting order indicators
        for elem in soup.find_all(['div', 'span', 'td']):
            text = elem.text.strip()
            # Look for patterns like "1. Player Name" or "CF Player Name"
            if text and any(pos in text for pos in ['1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.']):
                # Extract player name
                parts = text.split()
                if len(parts) >= 2:
                    player = ' '.join(parts[1:])
                    lineup.append(player)
        
        return lineup
    
    def get_lineup_with_stats(self, team_abbrev, vs_hand='R'):
        """
        Get projected lineup with K% stats vs pitcher handedness
        
        Args:
            team_abbrev: Team abbreviation
            vs_hand: Pitcher handedness ('R' or 'L')
            
        Returns:
            DataFrame with player names and K% vs hand
        """
        lineup = self.get_projected_lineup(team_abbrev)
        
        if not lineup:
            return pd.DataFrame()
        
        # Would integrate with Baseball Savant here to get K% for each player
        # For now, return just the names
        return pd.DataFrame({
            'player': lineup,
            'order': range(1, len(lineup) + 1)
        })


if __name__ == "__main__":
    # Test the scraper
    scraper = FangraphsLineupScraper()
    
    test_teams = ['NYY', 'BOS', 'LAD', 'HOU']
    
    for team in test_teams:
        print(f"\n{team}:")
        lineup = scraper.get_projected_lineup(team)
        for i, player in enumerate(lineup, 1):
            print(f"  {i}. {player}")
