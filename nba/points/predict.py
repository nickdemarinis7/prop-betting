"""
🏀 NBA POINTS PREDICTION - PRODUCTION SYSTEM
Streamlined for reliability and accuracy

Focus on what works:
1. Game-by-game ML predictions (recent form, home/away, trends)
2. Opponent defensive strength
3. Pace matchup analysis
4. Injury alerts (manual review)

No over-engineering. Just the edge.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shared.scrapers.gamelog import GameLogScraper
from shared.scrapers.nba_api import NBAApiScraper
from shared.features.opponent_defense import OpponentDefenseAnalyzer
from shared.utils.injuries import PlayerAvailabilityTracker
from shared.features.pace_analysis import PaceAnalyzer
from shared.features.usage_boost import UsageBoostCalculator
from shared.features.fatigue_analysis import FatigueAnalyzer
from shared.utils.betting_math import (
    calculate_probability,
    prob_to_american_odds,
    recommend_units
)
from nba_api.stats.static import teams
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Note: Probability and odds functions now imported from betting_math module

print("=" * 80)
print("🏀 NBA POINTS PREDICTION - PRODUCTION SYSTEM")
print("   Focused. Reliable. Profitable.")
print("=" * 80)

# Detect playoff period
from datetime import datetime
_today = datetime.now()
is_playoff = _today.month > 4 or (_today.month == 4 and _today.day >= 14)
if is_playoff:
    print("🏆 PLAYOFF MODE ACTIVE")

# Initialize core components
base_scraper = NBAApiScraper()
gamelog_scraper = GameLogScraper()
defense_analyzer = OpponentDefenseAnalyzer()
pace_analyzer = PaceAnalyzer()
availability_tracker = PlayerAvailabilityTracker()
usage_boost_calc = UsageBoostCalculator()
fatigue_analyzer = FatigueAnalyzer()

# Step 1: Get tonight's games
print("\n📅 Tonight's Schedule...")
try:
    games = base_scraper.get_todays_games()
    
    if games.empty:
        print("   No games tonight")
        sys.exit(0)
    
    print(f"   ✓ {len(games)} games")
except Exception as e:
    print(f"   ❌ Error fetching games: {e}")
    sys.exit(1)

# Create team mappings
all_teams = teams.get_teams()
team_id_to_abbr = {team['id']: team['abbreviation'] for team in all_teams}
team_id_to_name = {team['id']: team['full_name'] for team in all_teams}

matchups = {}
for _, game in games.iterrows():
    home_team = game['HOME_TEAM_ID']
    away_team = game['VISITOR_TEAM_ID']
    matchups[home_team] = away_team
    matchups[away_team] = home_team

# Step 2: Quick injury check and filtering
print("\n🏥 Injury Check...")
try:
    espn_injuries = availability_tracker.get_espn_injuries()
except Exception as e:
    print(f"   ⚠️  Error fetching injuries: {e}")
    espn_injuries = pd.DataFrame()

injury_alerts = []
out_player_names = []  # Track names of OUT players to filter

if not espn_injuries.empty:
    playing_teams = list(matchups.keys())
    tonight_injuries = espn_injuries[
        espn_injuries['team'].isin([team_id_to_name.get(tid, '') for tid in playing_teams])
    ]
    
    if not tonight_injuries.empty:
        # Filter OUT and DOUBTFUL players (unlikely to play)
        out_players = tonight_injuries[
            tonight_injuries['status'].str.contains('OUT|DOUBTFUL', case=False, na=False, regex=True)
        ]
        if not out_players.empty:
            print(f"   ⚠️  {len(out_players)} players OUT/DOUBTFUL - will be filtered from predictions")
            
            # Normalize names for filtering (handle accents)
            import unicodedata
            def normalize_name(name):
                normalized = unicodedata.normalize('NFD', name)
                return ''.join(c for c in normalized if unicodedata.category(c) != 'Mn').lower()
            
            for _, inj in out_players.iterrows():
                player_name = inj['player_name']
                injury_alerts.append(f"{player_name} ({inj.get('team', 'UNK')[:3]})")
                out_player_names.append(normalize_name(player_name))
                
                # Show first 10 OUT players
                if len(injury_alerts) <= 10:
                    print(f"      • {player_name} ({inj.get('team', 'UNK')[:15]})")
            
            if len(out_players) > 10:
                print(f"      ... and {len(out_players) - 10} more")
        else:
            print(f"   ✓ No major injuries")
    else:
        print(f"   ✓ No injuries affecting tonight's games")
else:
    print(f"   ⚠️  Could not fetch injury data")

# Step 3: Train ML model
print("\n🤖 Training Model...")
playing_teams_list = list(matchups.keys())

try:
    all_stats = base_scraper.get_combined_player_data(season='2025-26')
    tonight_players = all_stats[all_stats['TEAM_ID'].isin(playing_teams_list)].copy()
except Exception as e:
    print(f"   ❌ Error fetching player data: {e}")
    sys.exit(1)

# Basic filters for potential players
# We'll do active player check AFTER predictions to save time
tonight_players = tonight_players[
    (tonight_players['MIN'] > 15) & 
    (tonight_players['GP'] > 10)
].copy()

print(f"\n🔍 {len(tonight_players)} potential players (15+ MPG, 10+ GP)")

# PHASE 3: Expand training data to more players for better model
print("\n📊 Expanding Training Dataset...")
top_scorers = all_stats[
    (all_stats['MIN'] > 20) &  # Regular rotation players
    (all_stats['GP'] > 15) &   # Enough games played
    (all_stats['PTS'] > 8)     # Meaningful scorers
].nlargest(250, 'PTS')  # Top 250 scorers (up from 150)

print(f"   ✓ Training on {len(top_scorers)} players (expanded from 150)")

# Calculate blowout risk for each game
print("\n⚖️  Analyzing Competitive Balance...")
game_competitiveness = {}

for _, game in games.iterrows():
    home_id = game['HOME_TEAM_ID']
    away_id = game['VISITOR_TEAM_ID']
    
    # Get team records (W-L percentage)
    home_team_stats = all_stats[all_stats['TEAM_ID'] == home_id]
    away_team_stats = all_stats[all_stats['TEAM_ID'] == away_id]
    
    if not home_team_stats.empty and not away_team_stats.empty:
        # Calculate win percentage (using GP and team performance)
        home_wins = home_team_stats['W'].iloc[0] if 'W' in home_team_stats.columns else 41
        home_games = home_team_stats['GP'].iloc[0] if 'GP' in home_team_stats.columns else 82
        away_wins = away_team_stats['W'].iloc[0] if 'W' in away_team_stats.columns else 41
        away_games = away_team_stats['GP'].iloc[0] if 'GP' in away_team_stats.columns else 82
        
        home_win_pct = home_wins / max(home_games, 1)
        away_win_pct = away_wins / max(away_games, 1)
        
        # Calculate differential (in wins over 82 games)
        win_diff = abs(home_win_pct - away_win_pct) * 82
        
        # Determine blowout risk
        if win_diff >= 20:  # 20+ game difference
            risk = 'HIGH'
        elif win_diff >= 12:  # 12-19 game difference
            risk = 'MEDIUM'
        else:
            risk = 'LOW'
        
        favored_team = team_id_to_abbr.get(home_id if home_win_pct > away_win_pct else away_id, 'UNK')
        
        game_competitiveness[home_id] = {
            'risk': risk,
            'win_diff': win_diff,
            'favored': favored_team,
            'home_win_pct': home_win_pct,
            'away_win_pct': away_win_pct
        }
        game_competitiveness[away_id] = game_competitiveness[home_id]

blowout_games = sum(1 for g in game_competitiveness.values() if g['risk'] in ['HIGH', 'MEDIUM'])
if blowout_games > 0:
    print(f"   ⚠️  {blowout_games} games with blowout risk")
else:
    print(f"   ✓ All games expected to be competitive")

# Build training data - POINTS SPECIFIC (PHASE 3: Use expanded dataset)
# Use top_scorers from above (250 players)
top_players = top_scorers['PLAYER_ID'].tolist()

# Build points-specific training data
print("Building training data from game logs...")
print(f"Processing {len(top_players)} players...")

all_training_data = []
for i, player_id in enumerate(top_players):
    if i % 50 == 0:
        print(f"  Progress: {i}/{len(top_players)} players")
    
    # Get all games for this player
    gamelog = gamelog_scraper.get_player_game_logs(player_id, season='2025-26')
    
    if gamelog.empty or len(gamelog) < 20:
        continue
    
    # Sort by date
    gamelog = gamelog.sort_values('GAME_DATE', ascending=True)
    
    # For each game (except first 10), use previous games as features
    for idx in range(10, len(gamelog)):
        previous_games = gamelog.iloc[:idx]
        target_game = gamelog.iloc[idx]
        
        # TARGET: This game's POINTS (not assists!)
        target_points = target_game['PTS']
        
        # Calculate features from previous games
        features = gamelog_scraper.calculate_rolling_features(previous_games.tail(20))
        
        if features is None:
            continue
        
        # Add player and game context
        features['player_id'] = player_id
        features['game_date'] = target_game['GAME_DATE']
        features['is_home'] = 1 if 'vs.' in str(target_game.get('MATCHUP', '')) else 0
        features['target_points'] = target_points  # POINTS target!
        
        all_training_data.append(features)

print(f"  ✓ Created {len(all_training_data)} training samples")

training_data = pd.DataFrame(all_training_data) if all_training_data else pd.DataFrame()

# Enhanced feature set with advanced metrics - POINTS SPECIFIC
feature_cols = [
    # Recent performance
    'pts_last_5', 'pts_last_10', 'min_last_5', 'min_last_10',
    'tov_last_5', 'tov_last_10',
    
    # Trends and consistency
    'pts_trend', 'pts_std', 'pts_consistency',
    'pts_recent_high', 'pts_recent_low',
    
    # Home/away context
    'pts_home_avg', 'pts_away_avg', 'is_home',
    
    # Opponent factors
    'opp_def_strength', 'opp_pts_allowed',
    
    # Rest/fatigue (NEW)
    'days_rest',
    'is_back_to_back',     # 1 if B2B, 0 otherwise
    
    # ADVANCED: Player usage/role
    'usage_rate',          # % of team possessions used
    'potential_points',    # Expected points based on usage
    'minutes_share',       # % of game played
    
    # ADVANCED: Weighted recent performance
    'pts_weighted_recent', # Weighted: L3(50%) + L5(30%) + L10(20%)
    'pts_momentum',        # (L3 - L10) / L10 - getting better/worse
    
    # ADVANCED: Teammate context (ENHANCED)
    'teammates_out',       # Number of rotation players injured
    'usage_boost_multiplier', # Calculated boost from injuries (NEW)
    
    # ADVANCED: Opponent advanced metrics
    'opp_pace',           # Opponent's pace
    'opp_turnovers_forced' # Opponent's defensive pressure
]

for col in feature_cols:
    if col not in training_data.columns:
        # Set appropriate defaults for each feature type
        if 'opp_def_strength' in col:
            training_data[col] = 100.0
        elif 'opp_pts_allowed' in col or 'opp_pace' in col:
            training_data[col] = 25.0
        elif 'opp_turnovers_forced' in col:
            training_data[col] = 10.0
        elif 'usage_rate' in col:
            training_data[col] = 20.0
        elif 'minutes_share' in col:
            training_data[col] = 0.5
        elif 'potential_points' in col:
            training_data[col] = 5.0
        elif 'teammates_out' in col:
            training_data[col] = 0
        elif 'usage_boost_multiplier' in col:
            training_data[col] = 1.0  # No boost by default
        elif 'is_back_to_back' in col:
            training_data[col] = 0
        else:
            training_data[col] = 0

# Extract features and ensure they're numeric
X = training_data[feature_cols].fillna(0).copy()

# Convert all columns to numeric, coercing errors to NaN then filling with 0
X = X.apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)

# Convert to numpy array to avoid pandas DataFrame issues with XGBoost
X = X.values

y = training_data['target_points'].astype(float).values

# Create sample weights - weight recent games more heavily
sample_weights = np.ones(len(training_data))
if 'game_number' in training_data.columns:
    # Games in last 10 get 1.5x weight
    recent_mask = training_data['game_number'].values <= 10
    sample_weights[recent_mask] = 1.5

# Check data quality
print(f"   Training samples: {len(X)}")
print(f"   Target mean: {y.mean():.2f}, std: {y.std():.2f}")
print(f"   Target range: {y.min():.0f} - {y.max():.0f}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Get corresponding weights for training set
_, _, weights_train, _ = train_test_split(
    X, sample_weights, test_size=0.2, random_state=42
)

# PHASE 3: Optimized XGBoost parameters (tuned for better performance)
model = xgb.XGBRegressor(
    n_estimators=300,        # More trees for better learning (up from 200)
    max_depth=5,             # Slightly shallower to prevent overfitting (down from 6)
    learning_rate=0.03,      # Slower learning for better generalization (down from 0.04)
    min_child_weight=5,      # More regularization (up from 3)
    gamma=0.2,               # Stronger regularization (up from 0.1)
    subsample=0.8,           # Prevent overfitting (down from 0.85)
    colsample_bytree=0.8,    # Prevent overfitting (down from 0.85)
    reg_alpha=0.1,           # L1 regularization (NEW)
    reg_lambda=1.0,          # L2 regularization (NEW)
    random_state=42
)

# PHASE 3: Train with early stopping to prevent overfitting
model.fit(
    X_train, y_train, 
    sample_weight=weights_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

test_pred = model.predict(X_test)
test_mae = mean_absolute_error(y_test, test_pred)
test_r2 = r2_score(y_test, test_pred)

# PHASE 3: Cross-validation for more robust performance estimate
print(f"   ✓ Model ready (MAE: {test_mae:.2f}, R²: {test_r2:.1%})")
print("   ⏳ Running 5-fold cross-validation...")

cv_scores = cross_val_score(
    model, X, y, 
    cv=5, 
    scoring='neg_mean_absolute_error',
    n_jobs=-1
)
cv_mae = -cv_scores.mean()
cv_std = cv_scores.std()

print(f"   ✓ Cross-Val MAE: {cv_mae:.2f} ± {cv_std:.2f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n   Top 10 Most Important Features:")
for idx, row in feature_importance.head(10).iterrows():
    print(f"      {row['feature']:25s} {row['importance']:.3f}")

# Step 4: Generate predictions
print("\n🎯 Generating Predictions...")

# Create set of OUT/DOUBTFUL players for quick lookup
out_players = set()
if not espn_injuries.empty:
    out_list = espn_injuries[
        espn_injuries['status'].str.contains('OUT|DOUBTFUL', case=False, na=False, regex=True)
    ]['player_name'].tolist()
    out_players = set(out_list)
    print(f"   ⚠️  Filtering out {len(out_players)} injured/doubtful players")

predictions = []
filtered_count = 0

for _, player in tonight_players.iterrows():
    # Skip if player is OUT
    player_name = player['PLAYER_NAME']
    if player_name in out_players:
        filtered_count += 1
        continue
    player_id = player['PLAYER_ID']
    player_team = player['TEAM_ID']
    opponent_team = matchups.get(player_team)
    
    if opponent_team is None:
        continue
    
    recent_games = gamelog_scraper.get_recent_games(player_id, n_games=20)
    
    if recent_games.empty or len(recent_games) < 5:
        continue
    
    # Check if player has played recently (within last 14 days)
    # This filters out waived/inactive players
    if 'GAME_DATE' in recent_games.columns:
        most_recent_game = pd.to_datetime(recent_games['GAME_DATE'].iloc[0])
        days_since_lpts_game = (pd.Timestamp.now() - most_recent_game).days
        
        if days_since_lpts_game > 14:
            filtered_count += 1
            continue  # Skip inactive players
    
    # Filter OUT players (injuries)
    if out_player_names:
        import unicodedata
        def normalize_name(name):
            normalized = unicodedata.normalize('NFD', name)
            return ''.join(c for c in normalized if unicodedata.category(c) != 'Mn').lower()
        
        player_name_normalized = normalize_name(player['PLAYER_NAME'])
        if player_name_normalized in out_player_names:
            filtered_count += 1
            continue  # Skip injured OUT players
    
    features = gamelog_scraper.calculate_rolling_features(recent_games)
    
    if features is None:
        continue
    
    is_home = 0
    for _, game in games.iterrows():
        if game['HOME_TEAM_ID'] == player_team:
            is_home = 1
            break
    
    features['is_home'] = is_home
    
    # Opponent defense
    # Use assists_allowed as proxy for overall defense (returns def_strength rating)
    opp_defense = defense_analyzer.calculate_assists_allowed(opponent_team, season='2025-26')
    features['opp_def_strength'] = opp_defense['def_strength']
    features['opp_pts_allowed'] = opp_defense['opp_pts_allowed']
    
    # ADVANCED FEATURES - Calculate new metrics
    
    # 1. Player usage/role metrics
    features['usage_rate'] = player.get('USG_PCT', 20.0)  # Usage percentage
    features['potential_points'] = player.get('POTENTIAL_PTS', features.get('pts_last_10', 0) * 1.2)
    features['minutes_share'] = min(player.get('MIN', 25) / 48.0, 1.0)  # % of game
    
    # 2. Weighted recent performance (already calculated in gamelog scraper)
    # Just ensure it exists
    if 'pts_weighted_recent' not in features:
        if 'PTS' in recent_games.columns and len(recent_games) >= 10:
            l3_avg = recent_games['PTS'].head(3).mean()
            l5_avg = recent_games['PTS'].head(5).mean()
            l10_avg = recent_games['PTS'].head(10).mean()
            
            features['pts_weighted_recent'] = (l3_avg * 0.5) + (l5_avg * 0.3) + (l10_avg * 0.2)
            features['pts_momentum'] = (l3_avg - l10_avg) / max(l10_avg, 1.0) if l10_avg > 0 else 0
        else:
            features['pts_weighted_recent'] = features.get('pts_last_5', 0)
            features['pts_momentum'] = 0
    
    # 3. FATIGUE ANALYSIS - Rest days and back-to-back impact
    try:
        days_rest = fatigue_analyzer.calculate_days_rest(player_team, season='2025-26')
        is_b2b = fatigue_analyzer.is_back_to_back(player_team, season='2025-26')
        
        # Override days_rest from gamelog if we have team-level data
        features['days_rest'] = days_rest
        features['is_back_to_back'] = 1 if is_b2b else 0
    except:
        # Use player-level days_rest from gamelog scraper if team data fails
        features['is_back_to_back'] = 1 if features.get('days_rest', 1) == 0 else 0
    
    # 4. USAGE BOOST - Calculate boost from injured teammates
    # Get injured teammates for this team
    team_injuries_list = usage_boost_calc.get_team_injuries(
        player_team, 
        all_stats, 
        espn_injuries
    )
    
    # Calculate total boost for this player
    boost_info = usage_boost_calc.calculate_total_boost(player, team_injuries_list)
    features['teammates_out'] = boost_info['injured_teammates']
    features['usage_boost_multiplier'] = boost_info['total_boost']
    
    # 5. Opponent advanced metrics
    try:
        opp_pace_data = pace_analyzer.get_team_pace(opponent_team, season='2025-26')
        features['opp_pace'] = opp_pace_data.get('pace', 100.0)
    except:
        features['opp_pace'] = 100.0
    
    # Estimate opponent turnovers forced (proxy for defensive pressure)
    features['opp_turnovers_forced'] = opp_defense.get('def_strength', 100.0) / 10.0
    
    # Base ML prediction
    feature_vector = [features.get(col, 0 if 'opp' not in col else 100.0 if 'strength' in col else 25.0) 
                     for col in feature_cols]
    
    base_prediction = model.predict([feature_vector])[0]
    
    # Apply only proven adjustments with safety caps
    final_prediction = base_prediction
    
    # 1. Opponent defense (proven impact) - Cap at ±15%
    defense_factor = opp_defense['def_strength'] / 100.0
    defense_factor = max(0.85, min(1.15, defense_factor))  # Safety cap
    final_prediction *= defense_factor
    
    # 2. Pace (proven impact) - Cap at ±5%
    game_pace = pace_analyzer.calculate_game_pace(player_team, opponent_team, season='2025-26')
    pace_boost = pace_analyzer.calculate_pace_boost(game_pace['predicted_pace'])
    pace_boost = max(0.95, min(1.05, pace_boost))  # Safety cap
    final_prediction *= pace_boost
    
    # 3. USAGE BOOST (NEW) - Apply boost from injured teammates
    # Cap at 1.5x (50% max increase) to prevent over-projection
    usage_boost = features.get('usage_boost_multiplier', 1.0)
    usage_boost = min(usage_boost, 1.5)  # Safety cap
    final_prediction *= usage_boost
    
    # 4. FATIGUE PENALTY (NEW) - Reduce projection for back-to-back games
    if features.get('is_back_to_back', 0) == 1:
        # Players typically score 5-8% fewer points on B2B
        fatigue_penalty = 0.93  # 7% reduction
        final_prediction *= fatigue_penalty
    elif features.get('days_rest', 1) >= 3:
        # Well-rested players get slight boost (2-3%)
        rest_bonus = 1.025
        final_prediction *= rest_bonus
    
    # 5. Sanity check - Don't project more than 1.5x recent average
    l10_avg = features.get('pts_last_10', player['PTS'])
    max_reasonable = max(l10_avg * 1.5, 8.0)  # At least 8.0 for low-point players
    final_prediction = min(final_prediction, max_reasonable)
    
    # 5b. Global cap - no player above 42 points (even elite scorers rarely avg above this)
    if final_prediction > 42.0:
        final_prediction = 42.0
    
    opponent_name = team_id_to_abbr.get(opponent_team, 'UNK')
    
    # Get blowout risk for this game
    game_comp = game_competitiveness.get(player_team, {})
    blowout_risk = game_comp.get('risk', 'LOW')
    is_favored = game_comp.get('favored', '') == player['TEAM_ABBREVIATION']
    
    # Calculate confidence - Aligned with Play Quality Score
    confidence_score = 0
    
    # 1. Minutes (starter vs bench)
    if player['MIN'] > 30:
        confidence_score += 3
    elif player['MIN'] > 25:
        confidence_score += 2
    elif player['MIN'] > 20:
        confidence_score += 1
    else:
        confidence_score -= 1
    
    # 2. Consistency (low variance = reliable)
    pts_std = min(features.get('pts_std', 999), 7.0)
    if pts_std < 4.0:
        confidence_score += 2
    elif pts_std < 5.5:
        confidence_score += 1
    elif pts_std > 6.5:
        confidence_score -= 2
    
    # 3. Matchup quality (soft matchups are GOOD for confidence)
    if defense_factor > 1.08:  # Soft defense
        confidence_score += 2
    elif defense_factor > 1.03:  # Favorable
        confidence_score += 1
    elif defense_factor < 0.92:  # Tough defense
        confidence_score -= 1
    
    # 4. Recent form
    recent_trend = features.get('pts_trend', 0)
    if recent_trend > 0.5:  # Hot
        confidence_score += 1
    elif recent_trend < -0.5:  # Cold
        confidence_score -= 1
    
    # 5. Blowout risk penalty
    if blowout_risk == 'HIGH' and is_favored:
        confidence_score -= 3  # Major penalty
    elif blowout_risk == 'MEDIUM' and is_favored:
        confidence_score -= 1
    
    # Convert score to confidence level
    if confidence_score >= 5:
        confidence = 'HIGH'
    elif confidence_score >= 2:
        confidence = 'MEDIUM'
    else:
        confidence = 'LOW'
    
    # Calculate ladder probabilities for points thresholds (10-40)
    std_dev = min(features.get('pts_std', 5.0), 7.0)  # Cap std_dev at 7.0
    prob_10 = calculate_probability(final_prediction, std_dev, 10)
    prob_15 = calculate_probability(final_prediction, std_dev, 15)
    prob_20 = calculate_probability(final_prediction, std_dev, 20)
    prob_25 = calculate_probability(final_prediction, std_dev, 25)
    prob_30 = calculate_probability(final_prediction, std_dev, 30)
    prob_35 = calculate_probability(final_prediction, std_dev, 35)
    prob_40 = calculate_probability(final_prediction, std_dev, 40)
    
    predictions.append({
        'PLAYER_ID': player_id,
        'player_name': player['PLAYER_NAME'],
        'team': player['TEAM_ABBREVIATION'],
        'opponent': opponent_name,
        'base_ml_prediction': base_prediction,  # What the ML model actually predicted
        'final_projection': final_prediction,   # After all adjustments
        'season_avg': player['PTS'],
        'lpts_5_avg': features.get('pts_last_5', 0),
        'lpts_10_avg': features.get('pts_last_10', 0),
        'is_home': is_home,
        'opp_pts_allowed': opp_defense['opp_pts_allowed'],
        'defense_factor': defense_factor,
        'pace_boost': pace_boost,
        'predicted_pace': game_pace['predicted_pace'],
        'recent_trend': features.get('pts_trend', 0),
        'pts_momentum': features.get('pts_momentum', 0),  # L3 vs L10 for current form
        'consistency': features.get('pts_std', 0),
        'confidence': confidence,
        'blowout_risk': blowout_risk,
        'is_favored': is_favored,
        # NEW: Usage boost and fatigue
        'usage_boost': usage_boost,
        'injured_teammates': features.get('teammates_out', 0),
        'days_rest': features.get('days_rest', 1),
        'is_back_to_back': features.get('is_back_to_back', 0),
        # Ladder probabilities (10-40 points)
        'prob_10+': prob_10,
        'prob_15+': prob_15,
        'prob_20+': prob_20,
        'prob_25+': prob_25,
        'prob_30+': prob_30,
        'prob_35+': prob_35,
        'prob_40+': prob_40
    })

print(f"   ✓ {len(predictions)} players analyzed ({filtered_count} filtered: injury/inactive)")

# Step 5: Display results
results_df = pd.DataFrame(predictions).sort_values('final_projection', ascending=False)

# Analyze and create smart picks
print("\n" + "=" * 80)
print("⭐ SMART PICKS - TODAY'S BEST BETS")
print("=" * 80)
print()
print("   Based on projection, matchup, form, injuries, and blowout risk:")
print()

# Calculate unified Play Quality Score (0-100)
def calculate_play_quality(row):
    score = 50  # Start at neutral
    reasons = []
    warnings = []
    
    # 1. PROJECTION RELIABILITY - Compare to L10, not season
    # This captures recent role/form better than season average
    l10_avg = row.get('lpts_10_avg', row['season_avg'])
    projection_ratio = row['final_projection'] / max(l10_avg, 1.0)
    
    if 0.85 <= projection_ratio <= 1.4:  # Within 40% of recent average
        score += 15
        reasons.append("Reliable projection")
    elif 1.4 < projection_ratio <= 2.0:  # Elevated but possible
        score += 5
        warnings.append("⚠️ Above recent average")
    elif projection_ratio > 2.5:  # Very high variance
        score -= 10
        warnings.append("⚠️ High variance projection")
    
    # 2. CONFIDENCE LEVEL
    if row['confidence'] == 'HIGH':
        score += 15
        reasons.append("High confidence")
    elif row['confidence'] == 'MEDIUM':
        score += 5
    elif row['confidence'] == 'LOW':
        score -= 10
        warnings.append("Low confidence")
    
    # 3. MATCHUP QUALITY
    if row['defense_factor'] > 1.10:
        score += 12
        reasons.append("Soft matchup")
    elif row['defense_factor'] > 1.05:
        score += 6
        reasons.append("Favorable matchup")
    elif row['defense_factor'] < 0.92:
        score -= 8
        warnings.append("Tough defense")
    
    # 4. RECENT FORM - Use momentum (L3 vs L10) for more current assessment
    momentum = row.get('pts_momentum', row.get('recent_trend', 0))
    
    if momentum > 0.3:  # L3 is 30%+ better than L10
        score += 12
        reasons.append("Very hot")
    elif momentum > 0.1:  # L3 is 10%+ better than L10
        score += 6
        reasons.append("Trending up")
    elif momentum < -0.2:  # L3 is 20%+ worse than L10
        score -= 8
        warnings.append("Cold streak")
    elif momentum < -0.1:  # L3 is 10%+ worse than L10
        score -= 4
        warnings.append("Cooling off")
    
    # 5. PACE ADVANTAGE
    if row['pace_boost'] > 1.04:
        score += 8
        reasons.append("Fast pace")
    elif row['pace_boost'] > 1.02:
        score += 4
    
    # 6. USAGE BOOST (ENHANCED)
    injured_teammates = row.get('injured_teammates', 0)
    usage_boost = row.get('usage_boost', 1.0)
    
    if usage_boost >= 1.30:  # 30%+ boost
        score += 15
        reasons.append(f"Major usage boost ({injured_teammates} out, {usage_boost:.2f}x)")
    elif usage_boost >= 1.15:  # 15%+ boost
        score += 10
        reasons.append(f"Usage boost ({injured_teammates} out, {usage_boost:.2f}x)")
    elif usage_boost >= 1.08:  # 8%+ boost
        score += 5
        reasons.append(f"Slight usage boost ({injured_teammates} out)")
    
    # 7. FATIGUE FACTOR (NEW)
    is_b2b = row.get('is_back_to_back', 0)
    days_rest = row.get('days_rest', 1)
    
    if is_b2b == 1:
        score -= 10
        warnings.append("Back-to-back game (fatigue)")
    elif days_rest >= 3:
        score += 5
        reasons.append(f"Well-rested ({days_rest} days)")
    
    # 8. BLOWOUT RISK
    if row.get('blowout_risk') == 'HIGH' and row.get('is_favored'):
        score -= 20
        warnings.append("⚠️ BLOWOUT RISK")
    elif row.get('blowout_risk') == 'MEDIUM' and row.get('is_favored'):
        score -= 8
        warnings.append("May sit early")
    
    # 9. CONSISTENCY BONUS
    if row['consistency'] < 4.0:
        score += 6
        reasons.append("Very consistent")
    elif row['consistency'] < 5.5:
        score += 3
    
    # Combine reasons and warnings
    all_reasons = reasons + warnings
    
    # Clamp score to 0-100
    final_score = max(0, min(100, score))
    
    return final_score, all_reasons

results_df['play_quality'] = results_df.apply(lambda row: calculate_play_quality(row)[0], axis=1)
results_df['reasons'] = results_df.apply(lambda row: calculate_play_quality(row)[1], axis=1)

# Calculate hit probabilities for ladder betting
from scipy import stats

def calculate_hit_probability(projection, std_dev, threshold):
    """Calculate probability of hitting threshold using normal distribution"""
    if std_dev > 0:
        z_score = (threshold - 0.5 - projection) / std_dev
        prob = 1 - stats.norm.cdf(z_score)
    else:
        prob = 1.0 if projection >= threshold else 0.0
    return prob

# Add probability columns
for threshold in [3, 5, 7, 10]:
    results_df[f'prob_{threshold}+'] = results_df.apply(
        lambda row: calculate_hit_probability(row['final_projection'], row['consistency'], threshold),
        axis=1
    )

# Calculate Ladder Value Score - prioritizes best multi-threshold opportunities
def calculate_ladder_value(row):
    """
    Score based on ladder betting potential
    Higher score = better for ladder betting
    """
    score = 0
    
    # 1. Multi-threshold value (40 points max)
    # Reward players with good probabilities at MULTIPLE thresholds
    prob_15 = row['prob_15+']
    prob_20 = row['prob_20+']
    prob_25 = row['prob_25+']
    
    # 5+ PTS probability (0-15 points)
    if prob_15 > 0.70:
        score += 15
    elif prob_15 > 0.50:
        score += 12
    elif prob_15 > 0.35:
        score += 8
    elif prob_15 > 0.20:
        score += 4
    
    # 7+ PTS probability (0-15 points)
    if prob_20 > 0.40:
        score += 15
    elif prob_20 > 0.25:
        score += 12
    elif prob_20 > 0.15:
        score += 8
    elif prob_20 > 0.08:
        score += 4
    
    # 10+ PTS probability (0-10 points) - bonus for high upside
    if prob_25 > 0.10:
        score += 10
    elif prob_25 > 0.05:
        score += 6
    elif prob_25 > 0.02:
        score += 3
    
    # 2. Projection reliability (20 points max)
    l10_avg = row['lpts_10_avg']
    projection = row['final_projection']
    ratio = projection / l10_avg if l10_avg > 0 else 1.0
    
    if 0.9 <= ratio <= 1.2:
        score += 20  # Very reliable
    elif 0.8 <= ratio <= 1.35:
        score += 15  # Reliable
    elif 0.7 <= ratio <= 1.5:
        score += 10  # Acceptable
    elif ratio > 1.5:
        score -= 10  # Too wild
    
    # 3. Consistency (15 points max)
    std_dev = min(row['consistency'], 7.0)
    if std_dev < 4.0:
        score += 15  # Very consistent
    elif std_dev < 5.0:
        score += 12  # Consistent
    elif std_dev < 6.0:
        score += 8   # Acceptable
    elif std_dev > 6.5:
        score -= 5   # Too volatile
    
    # 4. Matchup quality (15 points max)
    defense_factor = row['defense_factor']
    if defense_factor > 1.10:
        score += 15  # Soft matchup
    elif defense_factor > 1.05:
        score += 10  # Favorable
    elif defense_factor > 1.00:
        score += 5   # Slight advantage
    elif defense_factor < 0.92:
        score -= 10  # Tough matchup
    
    # 5. Usage boost (10 points max)
    teammates_out = row.get('injured_teammates', 0)
    if teammates_out >= 4:
        score += 10
    elif teammates_out >= 2:
        score += 6
    
    # 6. Blowout risk penalty (-15 points)
    if row['blowout_risk'] == 'HIGH':
        score -= 15
    elif row['blowout_risk'] == 'MEDIUM':
        score -= 5
    
    # 7. Confidence bonus (5 points max)
    if row['confidence'] == 'HIGH':
        score += 5
    elif row['confidence'] == 'MEDIUM':
        score += 2
    
    # 8. Momentum penalty (NEW - based on validation results)
    momentum = row.get('pts_momentum', 0)
    if momentum < -0.3:
        score -= 15  # Very cold - trending down significantly
    elif momentum < -0.2:
        score -= 10  # Cooling off - recent decline
    elif momentum < -0.1:
        score -= 5   # Slight cooling
    
    return max(0, min(100, score))

results_df['ladder_value'] = results_df.apply(calculate_ladder_value, axis=1)

# Get top 10 based on LADDER VALUE (best for multi-threshold betting)
smart_picks = results_df.nlargest(10, 'ladder_value')

# Categorize picks into confidence tiers
def get_confidence_tier(row):
    """Determine confidence tier based on ladder value, momentum, and variance"""
    ladder_value = row['ladder_value']
    momentum = row.get('pts_momentum', 0)
    std_dev = min(row['consistency'], 7.0)  # Use capped std_dev
    
    # Tier 1: Safest bets (relaxed thresholds)
    if ladder_value >= 65 and momentum >= -0.05 and std_dev < 6.0:
        return 1
    # Tier 3: Higher risk (strong negative signals)
    elif momentum < -0.25 or std_dev > 7.0 or ladder_value < 45:
        return 3
    # Tier 2: Good value (everything else)
    else:
        return 2

smart_picks['tier'] = smart_picks.apply(get_confidence_tier, axis=1)

print()
print("=" * 80)
print("💰 TOP BETTING OPPORTUNITIES - LADDER BETTING GUIDE")
print("=" * 80)
print()

# Display by tier
for tier_num in [1, 2, 3]:
    tier_picks = smart_picks[smart_picks['tier'] == tier_num]
    
    if tier_picks.empty:
        continue
    
    # Tier header
    if tier_num == 1:
        print("🟢 TIER 1: SAFEST BETS")
        print("   (High ladder value + positive momentum + low variance)")
    elif tier_num == 2:
        print("🟡 TIER 2: GOOD VALUE")
        print("   (Solid opportunities with minor concerns)")
    else:
        print("🟠 TIER 3: HIGHER RISK")
        print("   (Negative momentum, high variance, or other red flags)")
    
    print("=" * 80)
    print()
    
    for i, (_, row) in enumerate(tier_picks.iterrows(), 1):
        location = "🏠" if row['is_home'] == 1 else "✈️"
        
        # Quality indicator
        quality = row['play_quality']
        if quality >= 75:
            quality_icon = "🟢"
        elif quality >= 60:
            quality_icon = "🟡"
        else:
            quality_icon = "🔴"
        
        print(f"#{i} {row['player_name']} ({row['team']}) {location} vs {row['opponent']}")
        print("-" * 80)
        
        # Projection and comparison
        projection = row['final_projection']
        l10_avg = row['lpts_10_avg']
        std_dev = row['consistency']
        ratio = projection / l10_avg if l10_avg > 0 else 1.0
        
        print(f"📊 PROJECTION: {projection:.1f} PTS  (L10: {l10_avg:.1f}, Std: {std_dev:.1f} | {ratio:.2f}x)")
        print()
        
        # Ladder probabilities - the key info for betting
        prob_10 = row['prob_10+']
        prob_15 = row['prob_15+']
        prob_20 = row['prob_20+']
        prob_25 = row['prob_25+']
        prob_30 = row['prob_30+']
        prob_35 = row['prob_35+']
        prob_40 = row['prob_40+']
        
        print(f"🎯 LADDER PROBABILITIES:")
        print(f"   10+ PTS: {prob_10:>5.0%}  {'🔥' if prob_10 > 0.90 else '✅' if prob_10 > 0.75 else ''}")
        print(f"   15+ PTS: {prob_15:>5.0%}  {'🔥' if prob_15 > 0.80 else '✅' if prob_15 > 0.60 else ''}")
        print(f"   20+ PTS: {prob_20:>5.0%}  {'🔥' if prob_20 > 0.60 else '✅' if prob_20 > 0.40 else ''}")
        print(f"   25+ PTS: {prob_25:>5.0%}  {'🔥' if prob_25 > 0.40 else '✅' if prob_25 > 0.20 else ''}")
        print(f"   30+ PTS: {prob_30:>5.0%}  {'🔥' if prob_30 > 0.25 else '✅' if prob_30 > 0.10 else ''}")
        print(f"   35+ PTS: {prob_35:>5.0%}  {'🔥' if prob_35 > 0.15 else '✅' if prob_35 > 0.05 else ''}")
        print(f"   40+ PTS: {prob_40:>5.0%}  {'🔥' if prob_40 > 0.10 else '✅' if prob_40 > 0.03 else ''}")
        print()
        
        # Calculate implied odds needed for +EV
        def prob_to_american_odds(prob):
            """Convert probability to American odds"""
            if prob >= 0.5:
                return int(-100 * prob / (1 - prob))
            else:
                return int(100 * (1 - prob) / prob)
        
        print(f"💵 MINIMUM ODDS FOR +EV:")
        if prob_15 > 0.01:
            min_odds_15 = prob_to_american_odds(prob_15)
            print(f"   15+ PTS: {min_odds_15:+5d} or better")
        if prob_20 > 0.01:
            min_odds_20 = prob_to_american_odds(prob_20)
            print(f"   20+ PTS: {min_odds_20:+5d} or better")
        if prob_25 > 0.01:
            min_odds_25 = prob_to_american_odds(prob_25)
            print(f"   25+ PTS: {min_odds_25:+5d} or better")
        if prob_30 > 0.01:
            min_odds_30 = prob_to_american_odds(prob_30)
            print(f"   30+ PTS: {min_odds_30:+5d} or better")
        print()
        
        # Historical hit rate from actual recent games
        player_id = row['PLAYER_ID']
        recent_games = gamelog_scraper.get_recent_games(player_id, n_games=10)
        
        if not recent_games.empty and 'PTS' in recent_games.columns:
            points = recent_games['PTS'].values
            hist_15 = (points >= 15).sum()
            hist_20 = (points >= 20).sum()
            hist_25 = (points >= 25).sum()
            
            print(f"📈 ACTUAL HIT RATE (Last 10 games):")
            print(f"   15+ PTS: {hist_15}/10 games ({hist_15*10}%)")
            if hist_20 > 0 or prob_20 > 0.15:
                print(f"   20+ PTS: {hist_20}/10 games ({hist_20*10}%)")
            if hist_25 > 0 or prob_25 > 0.05:
                print(f"   25+ PTS: {hist_25}/10 games ({hist_25*10}%)")
            if prob_30 > 0.02:
                hist_30 = (points >= 30).sum()
                print(f"   30+ PTS: {hist_30}/10 games ({hist_30*10}%)")
            print()
        
        # Context - why these numbers
        print(f"📝 CONTEXT:")
        
        # Key factors
        factors = []
        
        # Red flags - NEW!
        red_flags = []
        
        # Matchup
        defense_factor = row['defense_factor']
        if defense_factor > 1.08:
            factors.append(f"Soft matchup ({defense_factor:.2f}x)")
        elif defense_factor > 1.03:
            factors.append(f"Favorable matchup ({defense_factor:.2f}x)")
        elif defense_factor < 0.92:
            factors.append(f"Tough matchup ({defense_factor:.2f}x)")
            red_flags.append(f"Tough defense ({defense_factor:.2f}x)")
        
        # Usage boost
        teammates_out = row.get('injured_teammates', 0)
        if teammates_out >= 4:
            factors.append(f"High usage boost ({int(teammates_out)} teammates out)")
        elif teammates_out >= 2:
            factors.append(f"Usage boost ({int(teammates_out)} teammates out)")
        
        # Form - with red flag for negative momentum
        momentum = row.get('pts_momentum', 0)
        if momentum > 0.2:
            factors.append(f"Hot streak (momentum: {momentum:+.1f})")
        elif momentum < -0.2:
            factors.append(f"Cooling off (momentum: {momentum:+.1f})")
            red_flags.append(f"Negative momentum ({momentum:+.1f}) - Player trending down")
        
        # Consistency
        std_dev = row['consistency']
        if std_dev < 2.0:
            factors.append(f"Very consistent (σ: {std_dev:.1f})")
        elif std_dev > 6.5:
            factors.append(f"High variance (σ: {std_dev:.1f})")
            red_flags.append(f"High variance (σ: {std_dev:.1f}) - Unpredictable")
        
        # Pace
        pace_boost = row['pace_boost']
        if pace_boost > 1.03:
            factors.append(f"Fast pace ({pace_boost:.3f}x)")
        
        # Blowout risk
        if row['blowout_risk'] == 'HIGH':
            factors.append("⚠️ BLOWOUT RISK")
            red_flags.append("HIGH blowout risk - May sit in 4th quarter")
        elif row['blowout_risk'] == 'MEDIUM':
            red_flags.append("MEDIUM blowout risk - Monitor game flow")
        
        for factor in factors:
            print(f"   • {factor}")
        
        # Show red flags if any
        if red_flags:
            print()
            print(f"⚠️ RED FLAGS:")
            for flag in red_flags:
                print(f"   • {flag}")
        
        print()
        
        # Recommend unit sizing based on tier, probability, and red flags
        def recommend_units(prob, tier, has_red_flags):
            """Recommend unit size from: 1.5, 1.25, 1, 0.75, 0.5, 0.25, 0.1"""
            # Base units on probability
            if prob >= 0.80:
                base_units = 1.5
            elif prob >= 0.70:
                base_units = 1.25
            elif prob >= 0.60:
                base_units = 1.0
            elif prob >= 0.50:
                base_units = 0.75
            elif prob >= 0.40:
                base_units = 0.5
            elif prob >= 0.25:
                base_units = 0.25
            else:
                base_units = 0.1
            
            # Adjust for tier
            if tier == 3:  # Higher risk
                base_units = max(0.1, base_units - 0.5)
            elif tier == 1:  # Safest
                base_units = min(1.5, base_units + 0.25)
            
            # Reduce if red flags
            if has_red_flags:
                base_units = max(0.1, base_units - 0.25)
            
            # Round to nearest valid unit size
            valid_units = [1.5, 1.25, 1.0, 0.75, 0.5, 0.25, 0.1]
            return min(valid_units, key=lambda x: abs(x - base_units))
        
        # Calculate unit recommendations
        has_flags = len(red_flags) > 0
        tier_num = row['tier']
        units_5 = recommend_units(prob_15, tier_num, has_flags) if prob_15 > 0.01 else 0
        units_7 = recommend_units(prob_20, tier_num, has_flags) if prob_20 > 0.15 else 0
        units_10 = recommend_units(prob_25, tier_num, has_flags) if prob_25 > 0.05 else 0
        
        print(f"💰 RECOMMENDED UNITS:")
        if units_5 > 0:
            print(f"   15+ PTS: {units_5:.2f}u")
        if units_7 > 0:
            print(f"   20+ PTS: {units_7:.2f}u")
        if units_10 > 0:
            print(f"   25+ PTS: {units_10:.2f}u")
        print()
        
        # Show both scores
        ladder_value = row['ladder_value']
        print(f"💎 Ladder Value: {ladder_value:.0f}/100  |  Quality: {quality:.0f}/100 {quality_icon}")
        print()
        
        if i < len(tier_picks):
            print()
    
    # Add spacing between tiers
    print()

print("=" * 80)
print()

# Show fade list (players to avoid)
print("=" * 80)
print("🚫 FADE LIST - AVOID THESE PLAYERS")
print("=" * 80)
print()
print("   High-risk plays with negative factors:")
print()

# Get fade candidates: Low quality OR high projection with warnings
fade_candidates = results_df[
    (results_df['play_quality'] < 45) |
    ((results_df['blowout_risk'] == 'HIGH') & (results_df['is_favored'] == True))
].nsmallest(5, 'play_quality')

if not fade_candidates.empty:
    for i, (_, row) in enumerate(fade_candidates.iterrows(), 1):
        location = "🏠" if row['is_home'] == 1 else "✈️"
        quality = row['play_quality']
        
        print(f"   {i}. {row['player_name']:25} ({row['team']:3}) {location} vs {row['opponent']:3}")
        print(f"      Projection: {row['final_projection']:.1f} PTS | Season: {row['season_avg']:.1f} | L10: {row['lpts_10_avg']:.1f}")
        print(f"      Quality: {quality:.0f}/100 🔴 | {', '.join(row['reasons'])}")
        print()
else:
    print("   No major fade candidates tonight - all top players look good!")
    print()

print("=" * 80)
print()

print("\n" + "=" * 80)
print("🏆 TOP 20 PROJECTED POINTS")
print("=" * 80)
print()

for i, (_, row) in enumerate(results_df.head(20).iterrows(), 1):
    # Icons
    location = "🏠" if row['is_home'] == 1 else "✈️"
    
    # Matchup quality
    if row['defense_factor'] < 0.92:
        matchup = "🔒 TOUGH"
    elif row['defense_factor'] > 1.08:
        matchup = "🎯 SOFT"
    else:
        matchup = "➡️  AVG"
    
    # Pace
    if row['pace_boost'] > 1.03:
        pace = "⚡ FPTS"
    elif row['pace_boost'] < 0.97:
        pace = "🐌 SLOW"
    else:
        pace = ""
    
    # Trend
    if row['recent_trend'] > 0.5:
        trend = "📈 HOT"
    elif row['recent_trend'] < -0.5:
        trend = "📉 COLD"
    else:
        trend = ""
    
    # Confidence
    conf_icon = "🟢" if row['confidence'] == 'HIGH' else "🟡" if row['confidence'] == 'MEDIUM' else "🔴"
    
    # Blowout risk warning
    blowout_warning = ""
    if row.get('blowout_risk') == 'HIGH' and row.get('is_favored'):
        blowout_warning = " ⚠️ BLOWOUT RISK"
    elif row.get('blowout_risk') == 'MEDIUM' and row.get('is_favored'):
        blowout_warning = " ⚠️ May sit early"
    
    print(f"{i:2}. {row['player_name']:25} ({row['team']:3}) {location} vs {row['opponent']:3}{blowout_warning}")
    print(f"    Projection: {row['final_projection']:4.1f} PTS {conf_icon}")
    print(f"    Season: {row['season_avg']:4.1f} | L5: {row['lpts_5_avg']:4.1f} | L10: {row['lpts_10_avg']:4.1f}")
    print(f"    Matchup: {matchup} (Opp allows {row['opp_pts_allowed']:.1f}) {pace} {trend}")
    
    # Show breakdown if adjustments are significant
    if abs(row['defense_factor'] - 1.0) > 0.05 or abs(row['pace_boost'] - 1.0) > 0.03:
        print(f"    Adjustments: Defense {row['defense_factor']:.2f}x | Pace {row['pace_boost']:.2f}x")
    
    print()

# Show injury alerts and usage boost candidates
if injury_alerts:
    print("=" * 80)
    print("⚠️  INJURY ALERTS - REVIEW MANUALLY")
    print("=" * 80)
    print()
    print("   The following players are OUT. Consider:")
    print("   • Who benefits from increased usage?")
    print("   • Are any projected players affected?")
    print()
    for alert in injury_alerts[:10]:
        print(f"   • {alert}")
    print()
    
    # Identify usage boost candidates
    print("=" * 80)
    print("🚀 USAGE BOOST CANDIDATES")
    print("=" * 80)
    print()
    print("   Teammates of injured players who may see increased opportunity:")
    print()
    
    # Get injured players by team
    if not espn_injuries.empty:
        injured_by_team = {}
        out_injuries = espn_injuries[espn_injuries['status'] == 'OUT']
        
        for _, inj in out_injuries.iterrows():
            team = inj.get('team', '')
            player = inj.get('player_name', '')
            
            # Only track teams playing tonight
            team_abbr = None
            for tid, tname in team_id_to_name.items():
                if team in tname and tid in matchups:
                    team_abbr = team_id_to_abbr.get(tid)
                    break
            
            if team_abbr:
                if team_abbr not in injured_by_team:
                    injured_by_team[team_abbr] = []
                injured_by_team[team_abbr].append(player)
        
        # Show candidates from results
        if injured_by_team:
            for team_abbr, injured_players in injured_by_team.items():
                # Find teammates in our projections
                teammates = results_df[results_df['team'] == team_abbr].head(5)
                
                if not teammates.empty:
                    print(f"   {team_abbr} ({len(injured_players)} OUT: {', '.join(injured_players[:2])}{'...' if len(injured_players) > 2 else ''})")
                    
                    for _, tm in teammates.iterrows():
                        boost_indicator = ""
                        # Highlight if they're a guard/playmaker
                        if tm['season_avg'] >= 3.0:  # Has some playmaking ability
                            boost_indicator = " ⭐"
                        
                        print(f"      → {tm['player_name']:25} Proj: {tm['final_projection']:4.1f} (Season: {tm['season_avg']:4.1f}){boost_indicator}")
                    print()
        else:
            print("   No significant injuries affecting tonight's teams")
    print()

# Summary
print("=" * 80)
print("📊 SUMMARY")
print("=" * 80)
print()
print(f"Model Performance:")
print(f"  • MAE: {test_mae:.2f} points (typical error)")
print(f"  • R²: {test_r2:.1%} (variance explained)")
print()
print(f"Tonight's Analysis:")
print(f"  • {len(games)} games")
print(f"  • {len(results_df)} players projected")
print(f"  • {len(results_df[results_df['confidence']=='HIGH'])} high-confidence plays")
print()
print(f"Edge Factors:")
print(f"  ✓ Recent form (L5, L10 weighted heavily)")
print(f"  ✓ Opponent defense (±15% impact)")
print(f"  ✓ Pace matchup (±5% impact)")
print(f"  ✓ Home/away splits")
print()

# Add usage boost indicators to dataframe
if not espn_injuries.empty:
    # Build injured teammates map
    injured_by_team = {}
    out_injuries = espn_injuries[espn_injuries['status'] == 'OUT']
    
    for _, inj in out_injuries.iterrows():
        team = inj.get('team', '')
        player = inj.get('player_name', '')
        
        # Find team abbreviation
        team_abbr = None
        for tid, tname in team_id_to_name.items():
            if team in tname and tid in matchups:
                team_abbr = team_id_to_abbr.get(tid)
                break
        
        if team_abbr:
            if team_abbr not in injured_by_team:
                injured_by_team[team_abbr] = []
            injured_by_team[team_abbr].append(player)
    
    # Add columns
    results_df['injured_teammates'] = results_df['team'].apply(
        lambda t: len(injured_by_team.get(t, []))
    )
    results_df['is_playmaker'] = results_df['season_avg'] >= 3.0
    results_df['usage_boost_candidate'] = (
        (results_df['injured_teammates'] >= 2) & 
        (results_df['is_playmaker'])
    )
else:
    results_df['injured_teammates'] = 0
    results_df['is_playmaker'] = results_df['season_avg'] >= 3.0
    results_df['usage_boost_candidate'] = False

# Create focused CSV with only Smart Picks and Fade List
smart_picks_df = smart_picks.copy()
smart_picks_df['recommendation'] = 'TOP PLAY'

fade_list_df = fade_candidates.copy()
fade_list_df['recommendation'] = 'FADE'

# Combine
focused_df = pd.concat([smart_picks_df, fade_list_df], ignore_index=True)

# Prepare comprehensive CSV for ladder betting
# Calculate minimum odds needed for +EV
def prob_to_american_odds(prob):
    """Convert probability to American odds"""
    if prob >= 0.5:
        return int(-100 * prob / (1 - prob))
    else:
        return int(100 * (1 - prob) / prob)

# Add calculated columns
for col in ['prob_25+', 'prob_15+', 'prob_20+', 'prob_25+']:
    if col in focused_df.columns:
        # Probability as percentage
        focused_df[f'{col}_pct'] = (focused_df[col] * 100).round(0).astype(int)
        
        # Minimum odds for +EV
        threshold = col.replace('prob_', '').replace('+', '')
        focused_df[f'min_odds_{threshold}'] = focused_df[col].apply(
            lambda p: prob_to_american_odds(p) if p > 0.01 else None
        )

# Round numeric columns
for col in ['final_projection', 'season_avg', 'lpts_10_avg']:
    if col in focused_df.columns:
        focused_df[col] = focused_df[col].round(1)

if 'defense_factor' in focused_df.columns:
    focused_df['defense_factor'] = focused_df['defense_factor'].round(2)

if 'consistency' in focused_df.columns:
    focused_df['std_dev'] = focused_df['consistency'].round(1)

# Calculate projection ratio
if 'final_projection' in focused_df.columns and 'lpts_10_avg' in focused_df.columns:
    focused_df['proj_ratio'] = (focused_df['final_projection'] / focused_df['lpts_10_avg']).round(2)

# Create clear matchup description
if 'defense_factor' in focused_df.columns:
    def matchup_label(df):
        if df > 1.10:
            return 'SOFT'
        elif df > 1.05:
            return 'FAVORABLE'
        elif df < 0.92:
            return 'TOUGH'
        else:
            return 'AVERAGE'
    focused_df['matchup'] = focused_df['defense_factor'].apply(matchup_label)

# Add tier classification
def get_tier_for_csv(row):
    """Determine confidence tier for CSV"""
    ladder_value = row.get('ladder_value', 0)
    momentum = row.get('pts_momentum', 0)
    std_dev = min(row.get('consistency', 5.0), 7.0)
    
    if ladder_value >= 65 and momentum >= -0.05 and std_dev < 6.0:
        return 1
    elif momentum < -0.25 or std_dev > 7.0 or ladder_value < 45:
        return 3
    else:
        return 2

focused_df['tier'] = focused_df.apply(get_tier_for_csv, axis=1)

# Generate red flags summary
def generate_red_flags(row):
    """Generate comma-separated red flags for CSV"""
    flags = []
    
    # Negative momentum
    momentum = row.get('pts_momentum', 0)
    if momentum < -0.2:
        flags.append(f"Cooling ({momentum:+.1f})")
    
    # High variance (NBA points std_devs are naturally higher)
    std_dev = row.get('consistency', 0)
    if std_dev > 6.5:
        flags.append(f"High variance ({std_dev:.1f})")
    
    # Tough matchup
    defense_factor = row.get('defense_factor', 1.0)
    if defense_factor < 0.92:
        flags.append(f"Tough defense ({defense_factor:.2f})")
    
    # Blowout risk
    blowout = row.get('blowout_risk', '')
    if blowout == 'HIGH':
        flags.append("High blowout risk")
    elif blowout == 'MEDIUM':
        flags.append("Med blowout risk")
    
    return '; '.join(flags) if flags else 'None'

focused_df['red_flags'] = focused_df.apply(generate_red_flags, axis=1)

# Add unit sizing recommendations
def calculate_unit_size(prob, tier, has_red_flags):
    """Calculate recommended unit size"""
    # Base units on probability
    if prob >= 0.80:
        base_units = 1.5
    elif prob >= 0.70:
        base_units = 1.25
    elif prob >= 0.60:
        base_units = 1.0
    elif prob >= 0.50:
        base_units = 0.75
    elif prob >= 0.40:
        base_units = 0.5
    elif prob >= 0.25:
        base_units = 0.25
    else:
        base_units = 0.1
    
    # Adjust for tier
    if tier == 3:  # Higher risk
        base_units = max(0.1, base_units - 0.5)
    elif tier == 1:  # Safest
        base_units = min(1.5, base_units + 0.25)
    
    # Reduce if red flags
    if has_red_flags:
        base_units = max(0.1, base_units - 0.25)
    
    # Round to nearest valid unit size
    valid_units = [1.5, 1.25, 1.0, 0.75, 0.5, 0.25, 0.1]
    return min(valid_units, key=lambda x: abs(x - base_units))

# Calculate units for each threshold
focused_df['units_5'] = focused_df.apply(
    lambda row: calculate_unit_size(row['prob_15+'], row['tier'], row['red_flags'] != 'None') 
    if row.get('prob_15+', 0) > 0.01 else 0, 
    axis=1
)
focused_df['units_7'] = focused_df.apply(
    lambda row: calculate_unit_size(row['prob_20+'], row['tier'], row['red_flags'] != 'None') 
    if row.get('prob_20+', 0) > 0.15 else 0, 
    axis=1
)
focused_df['units_10'] = focused_df.apply(
    lambda row: calculate_unit_size(row['prob_25+'], row['tier'], row['red_flags'] != 'None') 
    if row.get('prob_25+', 0) > 0.05 else 0, 
    axis=1
)

# Select and order columns for ladder betting workflow
csv_columns = [
    # Identity
    'recommendation',
    'player_name', 
    'team', 
    'opponent',
    'is_home',
    
    # Projection
    'final_projection',
    'lpts_10_avg',
    'proj_ratio',
    
    # Ladder Probabilities (%)
    'prob_15+_pct',
    'prob_20+_pct',
    'prob_25+_pct',
    
    # Minimum Odds for +EV
    'min_odds_5',
    'min_odds_7',
    'min_odds_10',
    
    # Recommended Units
    'units_5',
    'units_7',
    'units_10',
    
    # Scores
    'ladder_value',
    'tier',
    'play_quality',
    'confidence',
    
    # Context
    'matchup',
    'defense_factor',
    'injured_teammates',
    'std_dev',
    'blowout_risk',
    'red_flags',
    
    # Additional
    'season_avg'
]

# Only include columns that exist
csv_columns = [col for col in csv_columns if col in focused_df.columns]

# Rename columns for maximum clarity
column_rename = {
    'recommendation': 'Type',
    'player_name': 'Player',
    'team': 'Team',
    'opponent': 'Opp',
    'is_home': 'Home',
    'final_projection': 'Proj',
    'lpts_10_avg': 'L10',
    'proj_ratio': 'Ratio',
    'prob_15+_pct': '15+%',
    'prob_20+_pct': '20+%',
    'prob_25+_pct': '25+%',
    'min_odds_5': 'Min_Odds_15+',
    'min_odds_7': 'Min_Odds_20+',
    'min_odds_10': 'Min_Odds_25+',
    'units_5': 'Units_15+',
    'units_7': 'Units_20+',
    'units_10': 'Units_25+',
    'ladder_value': 'Ladder_Value',
    'tier': 'Tier',
    'play_quality': 'Quality',
    'confidence': 'Conf',
    'matchup': 'Matchup',
    'defense_factor': 'Def_Factor',
    'injured_teammates': 'Tmts_Out',
    'std_dev': 'StdDev',
    'blowout_risk': 'Blowout',
    'red_flags': 'Red_Flags',
    'season_avg': 'Season'
}

# Save ladder betting focused CSV
output_file = f"predictions_production_{pd.Timestamp.now().strftime('%Y%m%d')}.csv"
focused_df[csv_columns].rename(columns=column_rename).to_csv(output_file, index=False)
print(f"✅ Saved to: {output_file}")
print(f"   ({len(smart_picks_df)} Top Plays + {len(fade_list_df)} Fade candidates)")
print()
print("💡 Pro Tips:")
print("   • Focus on HIGH confidence plays (🟢)")
print("   • Fade TOUGH matchups (🔒) unless player is hot (📈)")
print("   • Target SOFT matchups (🎯) + FPTS pace (⚡)")
print("   • Review injury alerts before betting")
print()
print("=" * 80)
