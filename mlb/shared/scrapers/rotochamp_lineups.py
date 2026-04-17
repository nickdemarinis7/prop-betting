"""
RotoChamp Lineup Scraper
Get projected lineups from RotoChamp depth charts
"""

import requests
from bs4 import BeautifulSoup


class RotoChampLineupScraper:
    """Scrape projected lineups from RotoChamp"""
    
    def __init__(self):
        self.base_url = "https://www.rotochamp.com/baseball/TeamPage.aspx"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        
        # Team abbreviation mapping (MLB API -> RotoChamp)
        self.team_mapping = {
            'AZ': 'ARI', 'ATL': 'ATL', 'BAL': 'BAL', 'BOS': 'BOS',
            'CHC': 'CHC', 'CWS': 'CWS', 'CIN': 'CIN', 'CLE': 'CLE',
            'COL': 'COL', 'DET': 'DET', 'HOU': 'HOU', 'KC': 'KAN',
            'LAA': 'LAA', 'LAD': 'LAD', 'MIA': 'MIA', 'MIL': 'MIL',
            'MIN': 'MIN', 'NYM': 'NYM', 'NYY': 'NYY', 'ATH': 'OAK',
            'PHI': 'PHI', 'PIT': 'PIT', 'SD': 'SD', 'SF': 'SF',
            'SEA': 'SEA', 'STL': 'STL', 'TB': 'TB', 'TEX': 'TEX',
            'TOR': 'TOR', 'WSH': 'WAS',
            # Alternate abbreviations
            'ARI': 'ARI', 'CHW': 'CWS', 'KCR': 'KAN', 'OAK': 'OAK',
            'SDP': 'SD', 'SFG': 'SF', 'TBR': 'TB', 'WSN': 'WAS'
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
            team_id = self.team_mapping.get(team_abbrev.upper())
            if not team_id:
                return []
            
            # Fetch team page
            url = f"{self.base_url}?TeamID={team_id}"
            
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find the projected lineup table
            lineup_table = soup.find('table', id='MainContent_gridProjectedLineup')
            
            if not lineup_table:
                return []
            
            lineup = []
            rows = lineup_table.find_all('tr')
            
            # Skip header row, get first 9 batters
            for row in rows[1:10]:
                cells = row.find_all('td')
                if cells and len(cells) >= 2:
                    # Second cell contains player name link
                    player_link = cells[1].find('a')
                    if player_link:
                        player_name = player_link.text.strip()
                        lineup.append(player_name)
            
            return lineup
                
        except Exception as e:
            return []


if __name__ == "__main__":
    # Test the scraper
    scraper = RotoChampLineupScraper()
    
    test_teams = ['NYY', 'BOS', 'LAD', 'NYM', 'CLE']
    
    for team in test_teams:
        print(f"\n{team}:")
        lineup = scraper.get_projected_lineup(team)
        if lineup:
            for i, player in enumerate(lineup, 1):
                print(f"  {i}. {player}")
        else:
            print("  ❌ No lineup found")
