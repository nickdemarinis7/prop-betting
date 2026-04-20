"""
Player availability and injury status tracking
Pulls data from multiple sources to determine who's playing tonight
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from datetime import datetime


class PlayerAvailabilityTracker:
    """Track player injuries, starting lineups, and availability"""
    
    def __init__(self):
        self.injury_cache = {}
        self.lineup_cache = {}
    
    def get_rotowire_injuries(self):
        """
        Scrape injury reports from RotoWire
        
        Returns:
            DataFrame with player injury information
        """
        url = "https://www.rotowire.com/basketball/nba-lineups.php"
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            injuries = []
            
            # RotoWire structure - look for injury indicators
            # This is a simplified version - actual scraping would need to be more robust
            injury_divs = soup.find_all('div', class_='lineup__player')
            
            for div in injury_divs:
                try:
                    player_name = div.find('a', class_='lineup__player-name')
                    status = div.find('span', class_='lineup__inj')
                    
                    if player_name and status:
                        injuries.append({
                            'player_name': player_name.text.strip(),
                            'status': status.text.strip(),
                            'source': 'RotoWire'
                        })
                except:
                    continue
            
            return pd.DataFrame(injuries)
            
        except Exception as e:
            print(f"Error scraping RotoWire: {e}")
            return pd.DataFrame()
    
    def get_espn_injuries(self):
        """
        Scrape injury reports from ESPN
        
        Returns:
            DataFrame with injury information and proper status interpretation
        """
        url = "https://www.espn.com/nba/injuries"
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            injuries = []
            
            # ESPN injury table structure
            tables = soup.find_all('table', class_='Table')
            
            for table in tables:
                team_name = None
                team_header = table.find_previous('div', class_='Table__Title')
                if team_header:
                    team_name = team_header.text.strip()
                
                rows = table.find_all('tr')[1:]  # Skip header
                
                for row in rows:
                    try:
                        cols = row.find_all('td')
                        if len(cols) >= 3:
                            player_name = cols[0].text.strip()
                            position = cols[1].text.strip()
                            return_date = cols[2].text.strip()
                            
                            # ESPN has 5 columns: NAME, POS, EST. RETURN DATE, STATUS, COMMENT
                            espn_status = cols[3].text.strip() if len(cols) >= 4 else ''
                            comment = cols[4].text.strip() if len(cols) >= 5 else ''
                            comment_lower = comment.lower()
                            
                            # Determine real injury status from comment keywords
                            # Comment contains the most accurate/recent info
                            if any(kw in comment_lower for kw in ["ruled out", "won't play", "will not play", "is out ", "has been ruled out", "sidelined"]):
                                injury_status = 'OUT'
                            elif 'doubtful' in comment_lower:
                                injury_status = 'DOUBTFUL'
                            elif 'questionable' in comment_lower:
                                injury_status = 'QUESTIONABLE'
                            elif 'probable' in comment_lower or 'expected to play' in comment_lower:
                                injury_status = 'PROBABLE'
                            elif espn_status.lower() == 'out':
                                injury_status = 'OUT'
                            elif espn_status.lower() == 'day-to-day':
                                # Day-To-Day without further detail — treat as questionable
                                injury_status = 'QUESTIONABLE'
                            else:
                                injury_status = 'QUESTIONABLE'
                            
                            injuries.append({
                                'player_name': player_name,
                                'team': team_name,
                                'position': position,
                                'return_date': return_date,
                                'status': injury_status,
                                'espn_status': espn_status,
                                'comment': comment,
                                'injury': comment,
                                'source': 'ESPN'
                            })
                    except:
                        continue
            
            return pd.DataFrame(injuries)
            
        except Exception as e:
            print(f"Error scraping ESPN: {e}")
            return pd.DataFrame()
    
    def check_player_status(self, player_name):
        """
        Check if a specific player is available to play
        
        Args:
            player_name: Full player name
        
        Returns:
            Dictionary with availability info
        """
        # Check cache first
        if player_name in self.injury_cache:
            cached = self.injury_cache[player_name]
            # Cache for 1 hour
            if (datetime.now() - cached['timestamp']).seconds < 3600:
                return cached['data']
        
        # Get injury data from multiple sources
        rotowire_injuries = self.get_rotowire_injuries()
        espn_injuries = self.get_espn_injuries()
        
        # Combine sources
        all_injuries = pd.concat([rotowire_injuries, espn_injuries], ignore_index=True)
        
        # Normalize player name for better matching (remove accents, handle special chars)
        import unicodedata
        def normalize_name(name):
            """Remove accents and normalize for matching"""
            # Normalize unicode characters (é -> e, ć -> c, etc.)
            normalized = unicodedata.normalize('NFD', name)
            # Remove accent marks
            ascii_name = ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')
            return ascii_name.lower()
        
        # Normalize the search name
        search_name_normalized = normalize_name(player_name)
        
        # Try exact match first, then normalized match
        player_injuries = all_injuries[
            all_injuries['player_name'].str.contains(player_name, case=False, na=False)
        ]
        
        # If no match, try normalized matching
        if player_injuries.empty and not all_injuries.empty:
            all_injuries['normalized_name'] = all_injuries['player_name'].apply(normalize_name)
            player_injuries = all_injuries[
                all_injuries['normalized_name'].str.contains(search_name_normalized, case=False, na=False)
            ]
        
        if player_injuries.empty:
            status = {
                'available': True,
                'status': 'Active',
                'injury': None,
                'confidence': 'Medium'  # No news might mean healthy
            }
        else:
            # Get most recent/severe status
            latest = player_injuries.iloc[0]
            
            # Determine availability based on status
            unavailable_statuses = ['OUT', 'DOUBTFUL', 'DNP']
            questionable_statuses = ['QUESTIONABLE', 'PROBABLE', 'GTD']
            
            status_text = latest.get('status', '').upper()
            
            if any(s in status_text for s in unavailable_statuses):
                available = False
                confidence = 'High'
            elif any(s in status_text for s in questionable_statuses):
                available = True  # Assume playing unless ruled out
                confidence = 'Low'
            else:
                available = True
                confidence = 'Medium'
            
            status = {
                'available': available,
                'status': latest.get('status', 'Unknown'),
                'injury': latest.get('injury', None),
                'confidence': confidence
            }
        
        # Cache the result
        self.injury_cache[player_name] = {
            'data': status,
            'timestamp': datetime.now()
        }
        
        return status
    
    def get_starting_lineups(self, game_date=None):
        """
        Get projected starting lineups for today's games
        
        Returns:
            Dictionary mapping team to starting 5
        """
        # This would scrape from RotoWire or similar
        # For now, return empty dict (placeholder)
        
        print("⚠️  Starting lineup data requires web scraping")
        print("   Recommended sources:")
        print("   - RotoWire: https://www.rotowire.com/basketball/nba-lineups.php")
        print("   - FantasyLabs: https://www.fantasylabs.com/nba/lineups/")
        print("   - DailyRoto: https://www.dailyroto.com/nba-lineups")
        
        return {}
    
    def get_minutes_projection_adjustment(self, player_name, team_injuries):
        """
        Adjust minutes projection based on teammate injuries
        
        Args:
            player_name: Player to adjust
            team_injuries: List of injured teammates
        
        Returns:
            Float: Minutes multiplier (e.g., 1.15 = 15% more minutes)
        """
        # If key teammates are out, this player gets more minutes/usage
        
        # Simple heuristic: each injured starter = +5% minutes for remaining players
        injured_starters = len([inj for inj in team_injuries if inj.get('is_starter', False)])
        
        multiplier = 1.0 + (injured_starters * 0.05)
        
        return min(multiplier, 1.25)  # Cap at 25% increase


def create_availability_report(players_df):
    """
    Create an availability report for a list of players
    
    Args:
        players_df: DataFrame with player names
    
    Returns:
        DataFrame with availability status added
    """
    tracker = PlayerAvailabilityTracker()
    
    print("Checking player availability...")
    print("(This may take a moment...)\n")
    
    # Get injury data once
    rotowire = tracker.get_rotowire_injuries()
    espn = tracker.get_espn_injuries()
    
    all_injuries = pd.concat([rotowire, espn], ignore_index=True)
    
    if all_injuries.empty:
        print("⚠️  Could not fetch injury data from web sources")
        print("   All players assumed available")
        players_df['availability'] = 'ACTIVE'
        players_df['injury_status'] = None
        return players_df
    
    print(f"✓ Found {len(all_injuries)} injury reports\n")
    
    # Check each player
    availability = []
    
    for _, player in players_df.iterrows():
        player_name = player.get('PLAYER_NAME', player.get('player_name', ''))
        
        # Look for injuries
        player_injuries = all_injuries[
            all_injuries['player_name'].str.contains(player_name, case=False, na=False)
        ]
        
        if player_injuries.empty:
            availability.append({
                'status': 'ACTIVE',
                'injury': None
            })
        else:
            latest = player_injuries.iloc[0]
            availability.append({
                'status': latest.get('status', 'UNKNOWN'),
                'injury': latest.get('injury', None)
            })
    
    players_df['availability'] = [a['status'] for a in availability]
    players_df['injury_status'] = [a['injury'] for a in availability]
    
    # Show summary
    out_count = len(players_df[players_df['availability'].str.contains('OUT', case=False, na=False)])
    questionable_count = len(players_df[players_df['availability'].str.contains('QUESTIONABLE', case=False, na=False)])
    
    print(f"📊 Availability Summary:")
    print(f"   OUT: {out_count} players")
    print(f"   QUESTIONABLE: {questionable_count} players")
    print(f"   ACTIVE: {len(players_df) - out_count - questionable_count} players")
    
    return players_df


if __name__ == "__main__":
    print("=" * 70)
    print("PLAYER AVAILABILITY TRACKER - TEST")
    print("=" * 70)
    
    tracker = PlayerAvailabilityTracker()
    
    # Test 1: Get injury reports
    print("\n1. Fetching Injury Reports from ESPN...")
    espn_injuries = tracker.get_espn_injuries()
    
    if not espn_injuries.empty:
        print(f"   ✓ Found {len(espn_injuries)} injured players")
        print(f"\n   Sample injuries:")
        print(espn_injuries.head(10).to_string(index=False))
    else:
        print("   ⚠️  Could not fetch injury data")
        print("   This is normal - web scraping can be blocked")
    
    # Test 2: Check specific player
    print("\n2. Testing Player Status Check...")
    test_players = ["LeBron James", "Stephen Curry", "Giannis Antetokounmpo"]
    
    for player in test_players:
        status = tracker.check_player_status(player)
        print(f"\n   {player}:")
        print(f"   - Available: {status['available']}")
        print(f"   - Status: {status['status']}")
        print(f"   - Injury: {status['injury']}")
    
    print("\n" + "=" * 70)
    print("NOTE: Web scraping may be blocked by some sites")
    print("For production, consider using a paid API like SportsDataIO")
    print("=" * 70)
