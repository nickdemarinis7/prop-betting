"""
Configuration settings for NBA Assists Projection System
"""

import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# NBA.com API endpoints
NBA_STATS_BASE_URL = "https://stats.nba.com/stats"
NBA_SCHEDULE_URL = f"{NBA_STATS_BASE_URL}/scoreboardv2"
NBA_PLAYER_STATS_URL = f"{NBA_STATS_BASE_URL}/leaguedashplayerstats"
NBA_TRACKING_URL = f"{NBA_STATS_BASE_URL}/leaguedashptstats"

# Headers for NBA.com requests (required to avoid blocking)
NBA_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Origin': 'https://www.nba.com',
    'Referer': 'https://www.nba.com/',
    'Connection': 'keep-alive',
    'x-nba-stats-origin': 'stats',
    'x-nba-stats-token': 'true'
}

# Model parameters
FEATURES = [
    'potential_assists',
    'assists_per_game',
    'usage_rate',
    'pace',
    'minutes_per_game',
    'games_played',
    'opponent_defensive_rating',
    'home_away',
    'days_rest',
    'team_assists_per_game'
]

# Prediction settings
MIN_GAMES_PLAYED = 5  # Minimum games to include player in predictions
LOOKBACK_DAYS = 30    # Days of historical data to consider

# Output settings
TOP_N_PLAYERS = 20    # Number of top projected players to display
