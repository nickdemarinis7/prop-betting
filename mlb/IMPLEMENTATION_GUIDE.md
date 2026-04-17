# 🚀 MLB Strikeout Model - Implementation Guide

## Quick Start: From Placeholder to Production

This guide shows how to integrate real MLB data into your strikeout prediction system.

---

## 📋 **Current Status**

✅ **Working:** Model architecture, feature engineering, predictions  
⚠️ **Placeholder:** Data sources (using sample data)  
🔜 **Needed:** Real MLB API integration

---

## 🔌 **Step 1: MLB Stats API Integration**

### **Get API Access:**
```bash
# MLB Stats API is free and public!
# No API key required
# Base URL: https://statsapi.mlb.com/api/v1
```

### **Update `pitcher_stats.py`:**

Replace the `get_season_stats()` method:

```python
def get_season_stats(self, season=2026, min_starts=5):
    """Fetch real pitcher stats from MLB API"""
    url = f"{self.base_url}/stats"
    params = {
        'stats': 'season',
        'season': season,
        'group': 'pitching',
        'gameType': 'R',  # Regular season
        'limit': 200
    }
    
    response = self.session.get(url, params=params)
    data = response.json()
    
    pitchers = []
    for split in data.get('stats', [{}])[0].get('splits', []):
        pitcher = split['player']
        stats = split['stat']
        
        if stats.get('gamesStarted', 0) >= min_starts:
            pitchers.append({
                'pitcher_id': pitcher['id'],
                'pitcher_name': pitcher['fullName'],
                'team': pitcher.get('currentTeam', {}).get('abbreviation', 'UNK'),
                'GS': stats.get('gamesStarted', 0),
                'IP': stats.get('inningsPitched', 0),
                'SO': stats.get('strikeOuts', 0),
                'K9': stats.get('strikeoutsPer9Inn', 0),
                'K_PCT': stats.get('strikeoutsPer9Inn', 0) / 9 * 0.27,  # Approximate
                'BB': stats.get('baseOnBalls', 0),
                'ERA': stats.get('era', 0),
                'WHIP': stats.get('whip', 0)
            })
    
    return pd.DataFrame(pitchers)
```

Replace the `get_game_logs()` method:

```python
def get_game_logs(self, pitcher_id, n_games=20):
    """Fetch real game logs from MLB API"""
    url = f"{self.base_url}/people/{pitcher_id}/stats"
    params = {
        'stats': 'gameLog',
        'season': 2026,
        'group': 'pitching',
        'gameType': 'R'
    }
    
    response = self.session.get(url, params=params)
    data = response.json()
    
    games = []
    for split in data.get('stats', [{}])[0].get('splits', [])[:n_games]:
        game = split['stat']
        game_data = split.get('game', {})
        opponent = split.get('opponent', {})
        
        games.append({
            'game_date': split.get('date'),
            'opponent': opponent.get('abbreviation', 'UNK'),
            'is_home': 1 if split.get('isHome') else 0,
            'IP': game.get('inningsPitched', 0),
            'SO': game.get('strikeOuts', 0),
            'BB': game.get('baseOnBalls', 0),
            'H': game.get('hits', 0),
            'ER': game.get('earnedRuns', 0),
            'pitches': game.get('numberOfPitches', 0),
            'K9': game.get('strikeoutsPer9Inn', 0),
            'K_PCT': game.get('strikeOuts', 0) / max(game.get('numberOfPitches', 1), 1),
            'ballpark': game_data.get('venue', {}).get('name', 'Unknown'),
            'opp_k_rate': 0.22  # Would fetch from team stats
        })
    
    return pd.DataFrame(games)
```

---

## 📅 **Step 2: Schedule Integration**

### **Update `mlb_schedule.py`:**

```python
def get_todays_games(self, date=None):
    """Fetch real schedule from MLB API"""
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')
    
    url = f"{self.base_url}/schedule"
    params = {
        'sportId': 1,  # MLB
        'date': date,
        'hydrate': 'probablePitcher'
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    games = []
    for date_data in data.get('dates', []):
        for game in date_data.get('games', []):
            away_team = game['teams']['away']['team']
            home_team = game['teams']['home']['team']
            
            # Get probable pitchers
            away_pitcher = game['teams']['away'].get('probablePitcher', {})
            home_pitcher = game['teams']['home'].get('probablePitcher', {})
            
            games.append({
                'game_id': game['gamePk'],
                'game_time': game['gameDate'],
                'away_team': away_team['abbreviation'],
                'home_team': home_team['abbreviation'],
                'away_pitcher': away_pitcher.get('fullName'),
                'away_pitcher_id': away_pitcher.get('id'),
                'home_pitcher': home_pitcher.get('fullName'),
                'home_pitcher_id': home_pitcher.get('id'),
                'venue': game['venue']['name']
            })
    
    return pd.DataFrame(games)
```

---

## 🌤️ **Step 3: Weather Data (Optional but Recommended)**

### **Add Weather API:**

```bash
# Sign up for free API key at openweathermap.org
pip install requests
```

### **Create `weather.py`:**

```python
import requests

class WeatherAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.openweathermap.org/data/2.5"
    
    def get_ballpark_weather(self, lat, lon):
        """Get weather at ballpark location"""
        url = f"{self.base_url}/weather"
        params = {
            'lat': lat,
            'lon': lon,
            'appid': self.api_key,
            'units': 'imperial'
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        return {
            'temp': data['main']['temp'],
            'wind_speed': data['wind']['speed'],
            'wind_dir': data['wind']['deg'],
            'humidity': data['main']['humidity']
        }
```

### **Ballpark Coordinates:**

```python
BALLPARK_COORDS = {
    'Fenway Park': (42.3467, -71.0972),
    'Yankee Stadium': (40.8296, -73.9262),
    'Dodger Stadium': (34.0739, -118.2400),
    # ... add all 30 parks
}
```

---

## 👨‍⚖️ **Step 4: Umpire Data (Advanced)**

### **Create `umpire.py`:**

```python
class UmpireDatabase:
    """Track umpire strike zone tendencies"""
    
    def __init__(self):
        # Would scrape from Baseball Savant or similar
        self.umpire_factors = {
            'Angel Hernandez': 0.95,  # Tight zone
            'Joe West': 1.02,  # Wide zone
            'CB Bucknor': 0.97,
            # ... add more umpires
        }
    
    def get_umpire_k_factor(self, umpire_name):
        """Get K factor for umpire (1.0 = average)"""
        return self.umpire_factors.get(umpire_name, 1.00)
```

---

## 🎯 **Step 5: Enhanced Features**

### **Add to `predict.py`:**

```python
# After basic features, add:

# Weather impact
if weather_available:
    weather = weather_api.get_ballpark_weather(ballpark_coords)
    features['wind_speed'] = weather['wind_speed']
    features['temperature'] = weather['temp']
    
    # Wind adjustment (>15 mph affects swings)
    if weather['wind_speed'] > 15:
        features['high_wind'] = 1
    else:
        features['high_wind'] = 0

# Umpire impact
if umpire_known:
    features['umpire_k_factor'] = umpire_db.get_umpire_k_factor(umpire_name)

# Opponent lineup quality
features['opp_lineup_k_rate'] = get_opponent_lineup_k_rate(opponent_team, pitcher_hand)
```

---

## 📊 **Step 6: Validation**

### **Create `validate.py`:**

```python
"""
Validate predictions against actual results
"""

import pandas as pd
from datetime import datetime, timedelta

# Load yesterday's predictions
yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
predictions = pd.read_csv(f'predictions_strikeouts_{yesterday}.csv')

# Fetch actual results from MLB API
actual_results = []
for _, pred in predictions.iterrows():
    pitcher_id = pred['pitcher_id']
    actual_k = get_actual_strikeouts(pitcher_id, yesterday)
    
    actual_results.append({
        'pitcher': pred['pitcher'],
        'predicted': pred['projection'],
        'actual': actual_k,
        'error': abs(pred['projection'] - actual_k),
        'hit_4.5': actual_k > 4.5,
        'hit_5.5': actual_k > 5.5,
        'hit_6.5': actual_k > 6.5,
        'hit_7.5': actual_k > 7.5
    })

results_df = pd.DataFrame(actual_results)

# Calculate metrics
mae = results_df['error'].mean()
print(f"MAE: {mae:.2f} strikeouts")

# Hit rates by line
for line in [4.5, 5.5, 6.5, 7.5]:
    hit_rate = results_df[f'hit_{line}'].mean()
    print(f"{line}+ K's hit rate: {hit_rate:.1%}")
```

---

## 🔄 **Step 7: Automated Daily Runs**

### **Create `run_daily.sh`:**

```bash
#!/bin/bash

# Daily MLB strikeout predictions
cd /path/to/mlb/strikeouts

# Run predictions
python predict.py

# Archive yesterday's results
yesterday=$(date -d "yesterday" +%Y%m%d)
if [ -f "predictions_strikeouts_${yesterday}.csv" ]; then
    python validate.py
    mkdir -p archive
    mv predictions_strikeouts_${yesterday}.csv archive/
fi

# Send notification (optional)
# mail -s "MLB Predictions Ready" you@email.com < predictions_strikeouts_$(date +%Y%m%d).csv
```

### **Set up cron job:**

```bash
# Run daily at 11 AM (before games)
crontab -e

# Add this line:
0 11 * * * /path/to/run_daily.sh
```

---

## 📈 **Step 8: ROI Tracking**

### **Reuse NBA ROI Tracker:**

```python
from nba.points.roi_tracker import ROITracker

tracker = ROITracker(tracking_file='mlb_roi_tracking.json')

# After placing bets
tracker.add_bet('20260411', 'Gerrit Cole', '6.5+', 0.75, -120)

# After games complete
tracker.update_bet_result(0, 'win', 7)  # Cole got 7 K's

# View performance
tracker.print_summary()
```

---

## ✅ **Testing Checklist**

Before going live:

- [ ] Test MLB API connection
- [ ] Verify pitcher data accuracy
- [ ] Check schedule scraping
- [ ] Validate ballpark factors
- [ ] Test weather integration (if added)
- [ ] Run backtest on last season
- [ ] Verify probability calculations
- [ ] Test CSV output format
- [ ] Set up automated runs
- [ ] Initialize ROI tracker

---

## 🎯 **Go-Live Checklist**

Day 1:
- [ ] Run predictions for today's games
- [ ] Manually verify probable starters
- [ ] Check weather conditions
- [ ] Review model confidence levels
- [ ] Place bets on high-confidence plays (65%+)
- [ ] Track all bets in ROI tracker

Day 2:
- [ ] Run validation on Day 1 results
- [ ] Calculate actual MAE
- [ ] Review hit rates by line
- [ ] Adjust strategy if needed

Week 1:
- [ ] Daily predictions and validation
- [ ] Track ROI and win rate
- [ ] Identify any systematic errors
- [ ] Refine features if needed

Month 1:
- [ ] Comprehensive performance review
- [ ] Model retraining with new data
- [ ] Feature importance analysis
- [ ] Strategy optimization

---

## 💡 **Pro Tips**

1. **Start Small:** Bet 0.25-0.5 units first week
2. **Track Everything:** Use ROI tracker religiously
3. **Verify Starters:** Always confirm probable pitcher is starting
4. **Check Weather:** Wind >15 mph = skip or reduce bet
5. **Know Umpires:** Some have tight zones (fewer K's)
6. **Avoid Variance:** Skip inconsistent pitchers
7. **Target Domes:** More predictable conditions
8. **Line Shopping:** Compare odds across books
9. **Bet Early:** Lines move as sharp money comes in
10. **Review Weekly:** Adjust strategy based on results

---

## 🚨 **Common Issues**

### **Issue: No probable starters announced**
**Solution:** Wait until ~2-3 hours before game time

### **Issue: Pitcher scratched last minute**
**Solution:** Set up alerts, check lineups 1 hour before

### **Issue: Model predictions too similar**
**Solution:** Add more variance features (weather, umpire, opponent lineup)

### **Issue: Low hit rates on certain lines**
**Solution:** Recalibrate probabilities, adjust std dev

### **Issue: High variance in results**
**Solution:** Increase sample size, add more features

---

## 📚 **Resources**

### **MLB Stats API:**
- Docs: https://statsapi.mlb.com/docs/
- Swagger: https://statsapi.mlb.com/api/v1/swagger.json

### **Baseball Savant:**
- Statcast: https://baseballsavant.mlb.com/statcast_search
- Umpire Scorecards: https://umpscorecards.com/

### **Weather:**
- OpenWeatherMap: https://openweathermap.org/api

### **Community:**
- r/sportsbook (Reddit)
- r/MLBbetting (Reddit)

---

## 🎓 **Next Level**

Once comfortable with basic model:

1. **Ensemble Models:** Combine XGBoost + LightGBM + CatBoost
2. **Neural Networks:** Try LSTM for time series
3. **Pitch-Level Data:** Analyze pitch mix, velocity
4. **Opponent Lineup:** Model specific batter matchups
5. **Live Betting:** In-game K predictions
6. **Multiple Markets:** Expand to H+R+RBI, F5 totals

---

**You're ready to go from placeholder to production!** 🚀⚾💰
