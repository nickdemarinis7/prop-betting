"""
🏀 NBA ASSISTS PREDICTION - PRODUCTION SYSTEM
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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core.name_utils import normalize_name, filter_by_name
from shared.scrapers.gamelog import GameLogScraper
from shared.scrapers.nba_api import NBAApiScraper
from shared.features.opponent_defense import OpponentDefenseAnalyzer
from shared.utils.injuries import PlayerAvailabilityTracker
from shared.features.pace_analysis import PaceAnalyzer
from mlb.shared.scrapers.odds_api import (
    OddsAPIScraper,
    calculate_implied_probability,
    calculate_expected_value,
)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
from prob_calibrator import ProbabilityCalibrator

from nba_api.stats.static import teams
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("🏀 NBA ASSISTS PREDICTION - PRODUCTION SYSTEM")
print("   Focused. Reliable. Profitable.")
print("=" * 80)

# Initialize core components
base_scraper = NBAApiScraper()
gamelog_scraper = GameLogScraper()
defense_analyzer = OpponentDefenseAnalyzer()
pace_analyzer = PaceAnalyzer()
availability_tracker = PlayerAvailabilityTracker()

# Initialize probability calibrator (Platt sigmoid on validation history).
# Validation showed model is over-confident at high probabilities; calibrator
# corrects this. We no longer multiply std_dev by 1.5x.
prob_calibrator = ProbabilityCalibrator()
if not prob_calibrator.load():
    print("   ⚠️  No prob calibrator found, attempting to train...")
    if not prob_calibrator.train():
        prob_calibrator = None

# Step 1: Get tonight's games
print("\n📅 Tonight's Schedule...")
games = base_scraper.get_todays_games()

if games.empty:
    print("   No games tonight")
    sys.exit(0)

print(f"   ✓ {len(games)} games")

# Detect playoff games (NBA regular season typically ends ~April 13)
from datetime import datetime as _dt
_today = _dt.now()
is_playoff_today = 1 if (_today.month > 4 or (_today.month == 4 and _today.day > 13)) else 0
if is_playoff_today:
    print("   🏆 PLAYOFF MODE — tighter defense, higher variance expected")

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
espn_injuries = availability_tracker.get_espn_injuries()

injury_alerts = []
out_player_names = []  # Track names of OUT players to filter

if not espn_injuries.empty:
    playing_teams = list(matchups.keys())
    tonight_injuries = espn_injuries[
        espn_injuries['team'].isin([team_id_to_name.get(tid, '') for tid in playing_teams])
    ]
    
    if not tonight_injuries.empty:
        out_players = tonight_injuries[tonight_injuries['status'].str.contains('OUT', case=False, na=False)]
        if not out_players.empty:
            print(f"   ⚠️  {len(out_players)} players OUT - will be filtered from predictions")
            
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
all_stats = base_scraper.get_combined_player_data(season='2025-26')
tonight_players = all_stats[all_stats['TEAM_ID'].isin(playing_teams_list)].copy()

# Basic filters for potential players
# We'll do active player check AFTER predictions to save time
tonight_players = tonight_players[
    (tonight_players['MIN'] > 15) & 
    (tonight_players['GP'] > 10)
].copy()

print(f"\n🔍 {len(tonight_players)} potential players (15+ MPG, 10+ GP)")

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

# Build training data - IMPROVED
# Use more players but with higher quality threshold
top_players = all_stats[
    (all_stats['MIN'] >= 18) &  # Rotation players only
    (all_stats['GP'] >= 20)     # Sufficient sample size
].nlargest(150, 'AST')['PLAYER_ID'].tolist()

training_data = gamelog_scraper.build_training_data(
    player_ids=top_players,
    season='2025-26',
    min_games=20  # Higher quality threshold
)

# Enhanced feature set with advanced metrics
feature_cols = [
    # Recent performance
    'ast_last_5', 'ast_last_10', 'min_last_5', 'min_last_10',
    'pts_last_5', 'pts_last_10', 'tov_last_5', 'tov_last_10',
    
    # Trends and consistency
    'ast_trend', 'ast_std', 'ast_consistency',
    'ast_recent_high', 'ast_recent_low',
    
    # Home/away context
    'ast_home_avg', 'ast_away_avg', 'is_home',
    
    # Opponent factors
    'opp_def_strength', 'opp_ast_allowed',
    
    # Rest/fatigue
    'days_rest',
    
    # ADVANCED: Player usage/role (NEW)
    'usage_rate',          # % of team possessions used
    'potential_assists',   # Tracking data - passes that should be assists
    'minutes_share',       # % of game played
    
    # ADVANCED: Weighted recent performance (NEW)
    'ast_weighted_recent', # Weighted: L3(50%) + L5(30%) + L10(20%)
    'ast_momentum',        # (L3 - L10) / L10 - getting better/worse
    
    # ADVANCED: Teammate context (NEW)
    'teammates_out',       # Number of rotation players injured
    
    # ADVANCED: Opponent advanced metrics (NEW)
    'opp_pace',           # Opponent's pace
    'opp_turnovers_forced', # Opponent's defensive pressure
    
    # Game type
    'is_playoff',          # Playoff games have different dynamics
]

for col in feature_cols:
    if col not in training_data.columns:
        # Set appropriate defaults for each feature type
        if 'opp_def_strength' in col:
            training_data[col] = 100.0
        elif 'opp_ast_allowed' in col or 'opp_pace' in col:
            training_data[col] = 25.0
        elif 'opp_turnovers_forced' in col:
            training_data[col] = 10.0
        elif 'usage_rate' in col:
            training_data[col] = 20.0
        elif 'minutes_share' in col:
            training_data[col] = 0.5
        elif 'potential_assists' in col:
            training_data[col] = 5.0
        elif 'teammates_out' in col:
            training_data[col] = 0
        else:
            training_data[col] = 0

X = training_data[feature_cols].fillna(0)
y = training_data['target_assists']

# Create sample weights - weight recent games more heavily
sample_weights = np.ones(len(training_data))
if 'game_number' in training_data.columns:
    # Games in last 10 get 1.5x weight
    recent_mask = training_data['game_number'] <= 10
    sample_weights[recent_mask] = 1.5

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Get corresponding weights for training set
_, _, weights_train, _ = train_test_split(
    X, sample_weights, test_size=0.2, random_state=42
)

# Optimized XGBoost parameters for assists prediction
model = xgb.XGBRegressor(
    n_estimators=200,        # More trees for better learning
    max_depth=6,            # Slightly deeper for complex patterns
    learning_rate=0.04,     # Slower, more careful learning
    min_child_weight=3,     # Prevent overfitting on small samples
    gamma=0.1,              # Regularization
    subsample=0.85,         # Use more data per tree
    colsample_bytree=0.85,  # Use more features per tree
    random_state=42
)

model.fit(X_train, y_train, sample_weight=weights_train)
test_pred = model.predict(X_test)
test_mae = mean_absolute_error(y_test, test_pred)
test_r2 = r2_score(y_test, test_pred)

print(f"   ✓ Model ready (MAE: {test_mae:.2f}, R²: {test_r2:.1%})")

# Step 4: Generate predictions
print("\n🎯 Generating Predictions...")

# Create set of OUT/DOUBTFUL players for quick lookup (tonight's teams only)
out_players = set()
if not espn_injuries.empty:
    playing_team_names = [team_id_to_name.get(tid, '') for tid in matchups.keys()]
    tonight_out = espn_injuries[
        (espn_injuries['status'].str.contains('OUT|DOUBTFUL', case=False, na=False, regex=True)) &
        (espn_injuries['team'].isin(playing_team_names))
    ]
    out_players = set(tonight_out['player_name'].tolist())
    if out_players:
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
    
    # Check if player has played recently
    # In playoffs, games are every 2-3 days; use 7-day window
    max_inactive_days = 7 if is_playoff_today else 14
    if 'GAME_DATE' in recent_games.columns:
        most_recent_game = pd.to_datetime(recent_games['GAME_DATE'].iloc[0])
        days_since_last_game = (pd.Timestamp.now() - most_recent_game).days
        
        if days_since_last_game > max_inactive_days:
            filtered_count += 1
            continue  # Skip inactive players
    
    # Filter DNPs: if player had 0 assists AND played < 5 min in last game, skip
    if 'AST' in recent_games.columns and 'MIN' in recent_games.columns:
        last_ast = recent_games['AST'].iloc[0]
        last_min = recent_games['MIN'].iloc[0]
        if last_ast == 0 and last_min < 5:
            filtered_count += 1
            continue  # Likely DNP or injury exit
    
    # Filter OUT players (injuries)
    if out_player_names:
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
    opp_defense = defense_analyzer.calculate_assists_allowed(opponent_team, season='2025-26')
    features['opp_def_strength'] = opp_defense['def_strength']
    features['opp_ast_allowed'] = opp_defense['opp_ast_allowed']
    
    # ADVANCED FEATURES - Calculate new metrics
    
    # 1. Player usage/role metrics
    features['usage_rate'] = player.get('USG_PCT', 20.0)  # Usage percentage
    features['potential_assists'] = player.get('POTENTIAL_AST', features.get('ast_last_10', 0) * 1.3)
    features['minutes_share'] = min(player.get('MIN', 25) / 48.0, 1.0)  # % of game
    
    # 2. Weighted recent performance (emphasize recent games)
    if 'AST' in recent_games.columns and len(recent_games) >= 10:
        l3_avg = recent_games['AST'].head(3).mean()
        l5_avg = recent_games['AST'].head(5).mean()
        l10_avg = recent_games['AST'].head(10).mean()
        
        features['ast_weighted_recent'] = (l3_avg * 0.5) + (l5_avg * 0.3) + (l10_avg * 0.2)
        features['ast_momentum'] = (l3_avg - l10_avg) / max(l10_avg, 1.0) if l10_avg > 0 else 0
    else:
        features['ast_weighted_recent'] = features.get('ast_last_5', 0)
        features['ast_momentum'] = 0
    
    # 3. Teammate context (injuries)
    team_injuries = espn_injuries[espn_injuries['team'] == player['TEAM_ABBREVIATION']]
    teammates_out = len(team_injuries[team_injuries['status'] == 'OUT'])
    features['teammates_out'] = teammates_out
    
    # 4. Opponent advanced metrics
    try:
        opp_pace_data = pace_analyzer.get_team_pace(opponent_team, season='2025-26')
        features['opp_pace'] = opp_pace_data.get('pace', 100.0)
    except:
        features['opp_pace'] = 100.0
    
    # Estimate opponent turnovers forced (proxy for defensive pressure)
    features['opp_turnovers_forced'] = opp_defense.get('def_strength', 100.0) / 10.0
    
    # Game type
    features['is_playoff'] = is_playoff_today
    
    # Base ML prediction
    feature_vector = [features.get(col, 0 if 'opp' not in col else 100.0 if 'strength' in col else 25.0) 
                     for col in feature_cols]
    
    base_prediction = model.predict([feature_vector])[0]
    
    # Apply only proven adjustments with safety caps
    final_prediction = base_prediction
    
    # 1. Opponent defense (proven impact) - Cap at ±12%
    defense_factor = opp_defense['def_strength'] / 100.0
    defense_factor = max(0.88, min(1.12, defense_factor))  # Tighter safety cap
    final_prediction *= defense_factor
    
    # 2. Pace (proven impact) - Cap at ±4%
    game_pace = pace_analyzer.calculate_game_pace(player_team, opponent_team, season='2025-26')
    pace_boost = pace_analyzer.calculate_pace_boost(game_pace['predicted_pace'])
    pace_boost = max(0.96, min(1.04, pace_boost))  # Tighter safety cap
    final_prediction *= pace_boost
    
    # 3. PLAYOFF PENALTY — validation shows playoffs reduce assists ~15%
    #    Tighter defense, slower pace, more half-court sets
    if is_playoff_today:
        final_prediction *= 0.85
    
    # 4. L10 anchor blend — validation (155 samples) shows the GBM loses to
    #    a simple L10 average by 20%.  8+ AST projections over-shoot by 4.2 AST.
    #    Heavily anchor to L10; let the model contribute matchup signal only.
    l10_avg = features.get('ast_last_10', player['AST'])
    final_prediction = (final_prediction * 0.35) + (l10_avg * 0.65)
    
    # 5. Sanity check - Don't project more than 1.2x recent average
    max_reasonable = max(l10_avg * 1.2, 3.0)  # At least 3.0 for low-assist players
    final_prediction = min(final_prediction, max_reasonable)
    
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
    ast_std = features.get('ast_std', 999)
    if ast_std < 2.0:
        confidence_score += 2
    elif ast_std < 2.8:
        confidence_score += 1
    elif ast_std > 3.5:
        confidence_score -= 2
    
    # 3. Matchup quality (soft matchups are GOOD for confidence)
    if defense_factor > 1.08:  # Soft defense
        confidence_score += 2
    elif defense_factor > 1.03:  # Favorable
        confidence_score += 1
    elif defense_factor < 0.92:  # Tough defense
        confidence_score -= 1
    
    # 4. Recent form
    recent_trend = features.get('ast_trend', 0)
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
    
    predictions.append({
        'PLAYER_ID': player_id,
        'player_name': player['PLAYER_NAME'],
        'team': player['TEAM_ABBREVIATION'],
        'opponent': opponent_name,
        'base_ml_prediction': base_prediction,  # What the ML model actually predicted
        'final_projection': final_prediction,   # After defense/pace adjustments
        'season_avg': player['AST'],
        'last_5_avg': features.get('ast_last_5', 0),
        'last_10_avg': features.get('ast_last_10', 0),
        'is_home': is_home,
        'opp_ast_allowed': opp_defense['opp_ast_allowed'],
        'defense_factor': defense_factor,
        'pace_boost': pace_boost,
        'predicted_pace': game_pace['predicted_pace'],
        'recent_trend': features.get('ast_trend', 0),
        'ast_momentum': features.get('ast_momentum', 0),  # L3 vs L10 for current form
        'consistency': features.get('ast_std', 0),
        'confidence': confidence,
        'blowout_risk': blowout_risk,
        'is_favored': is_favored
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
    l10_avg = row.get('last_10_avg', row['season_avg'])
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
    momentum = row.get('ast_momentum', row.get('recent_trend', 0))
    
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
    
    # 6. USAGE BOOST
    injured_teammates = row.get('injured_teammates', 0)
    if injured_teammates >= 4 and row.get('is_playmaker', False):
        score += 12
        reasons.append(f"Major usage boost ({injured_teammates} out)")
    elif injured_teammates >= 2 and row.get('is_playmaker', False):
        score += 6
        reasons.append(f"Usage boost ({injured_teammates} out)")
    
    # 7. BLOWOUT RISK
    if row.get('blowout_risk') == 'HIGH' and row.get('is_favored'):
        score -= 20
        warnings.append("⚠️ BLOWOUT RISK")
    elif row.get('blowout_risk') == 'MEDIUM' and row.get('is_favored'):
        score -= 8
        warnings.append("May sit early")
    
    # 8. CONSISTENCY BONUS
    if row['consistency'] < 2.0:
        score += 6
        reasons.append("Very consistent")
    elif row['consistency'] < 2.8:
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
    """P(actual >= threshold) under Normal(projection, std_dev), then
    pass through the empirical Platt-sigmoid calibrator. The calibrator
    handles asymmetric over-confidence at high probs (validation showed
    raw 86% → actual 65%) so we no longer apply ad-hoc inflation/caps.
    """
    if std_dev <= 0:
        raw = 1.0 if projection >= threshold else 0.0
    else:
        # Bound std to avoid both pinpoint overconfidence and lottery widths
        bounded_std = max(min(std_dev, 4.0), 1.5)
        z_score = (threshold - projection) / bounded_std
        raw = 1.0 - stats.norm.cdf(z_score)
    if prob_calibrator is not None and prob_calibrator.is_fitted:
        return float(prob_calibrator.calibrate(raw))
    return raw

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
    prob_5 = row['prob_5+']
    prob_7 = row['prob_7+']
    prob_10 = row['prob_10+']
    
    # 5+ AST probability (0-15 points)
    if prob_5 > 0.70:
        score += 15
    elif prob_5 > 0.50:
        score += 12
    elif prob_5 > 0.35:
        score += 8
    elif prob_5 > 0.20:
        score += 4
    
    # 7+ AST probability (0-15 points)
    if prob_7 > 0.40:
        score += 15
    elif prob_7 > 0.25:
        score += 12
    elif prob_7 > 0.15:
        score += 8
    elif prob_7 > 0.08:
        score += 4
    
    # 10+ AST probability (0-10 points) - bonus for high upside
    if prob_10 > 0.10:
        score += 10
    elif prob_10 > 0.05:
        score += 6
    elif prob_10 > 0.02:
        score += 3
    
    # 2. Projection reliability (20 points max)
    l10_avg = row['last_10_avg']
    projection = row['final_projection']
    ratio = projection / l10_avg if l10_avg > 0 else 1.0
    
    if 0.9 <= ratio <= 1.4:
        score += 20  # Very reliable
    elif 0.8 <= ratio <= 1.6:
        score += 15  # Reliable
    elif 0.7 <= ratio <= 1.8:
        score += 10  # Acceptable
    elif ratio > 2.5:
        score -= 10  # Too wild
    
    # 3. Consistency (15 points max)
    std_dev = row['consistency']
    if std_dev < 2.0:
        score += 15  # Very consistent
    elif std_dev < 2.5:
        score += 12  # Consistent
    elif std_dev < 3.0:
        score += 8   # Acceptable
    elif std_dev > 4.0:
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
    momentum = row.get('ast_momentum', 0)
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
    momentum = row.get('ast_momentum', 0)
    std_dev = row['consistency']
    
    # Tier 1: Safest bets
    if ladder_value >= 75 and momentum >= 0 and std_dev < 2.5:
        return 1
    # Tier 3: Higher risk (negative momentum or very high variance)
    elif momentum < -0.2 or std_dev > 3.5 or ladder_value < 65:
        return 3
    # Tier 2: Good value (everything else)
    else:
        return 2

smart_picks['tier'] = smart_picks.apply(get_confidence_tier, axis=1)

# ---------------------------------------------------------------------
# BOOK ODDS + EV-BASED EDGE ANALYSIS (calibrated probs, Kelly, ladder)
# ---------------------------------------------------------------------
print("\n📊 Fetching book odds for assists...")
odds_scraper = OddsAPIScraper()
odds_df = odds_scraper.get_all_nba_assists_odds() if hasattr(odds_scraper, 'get_all_nba_assists_odds') else pd.DataFrame()
if odds_df is None:
    odds_df = pd.DataFrame()

results_df['book_line'] = float('nan')
results_df['book_odds'] = float('nan')
results_df['bookmaker'] = ''
results_df['recommended_side'] = ''
results_df['our_prob'] = float('nan')
results_df['book_prob'] = float('nan')
results_df['edge_pp'] = 0.0
results_df['ev'] = 0.0
results_df['kelly_units'] = 0.0

NBA_KELLY_FRACTION = 0.25
NBA_MAX_KELLY = 1.5

def _ast_kelly(our_p, american_odds):
    if american_odds < 0:
        decimal = 1 + (100 / abs(american_odds))
    else:
        decimal = 1 + (american_odds / 100)
    b = decimal - 1
    if b <= 0:
        return 0.0
    kelly = (our_p * b - (1 - our_p)) / b
    if kelly <= 0:
        return 0.0
    return round(min(kelly * NBA_KELLY_FRACTION * 10, NBA_MAX_KELLY), 2)

def _our_prob_at(line, side, projection, std_dev):
    raw_over = 1.0 - stats.norm.cdf(line, projection, std_dev)
    if prob_calibrator is not None and prob_calibrator.is_fitted:
        cal_over = float(prob_calibrator.calibrate(raw_over))
    else:
        cal_over = raw_over
    return cal_over if side == 'Over' else 1.0 - cal_over

MIN_EV_PLAY = 0.05
MIN_EDGE_PP_PLAY = 0.04
alternates_by_player = {}

if not odds_df.empty:
    for idx, row in results_df.iterrows():
        player_name = row['player_name']
        proj = float(row['final_projection'])
        std_for_player = max(min(float(row.get('consistency', 2.5) or 2.5), 4.0), 1.5)
        
        player_odds = filter_by_name(odds_df, 'player', player_name)
        if player_odds.empty:
            continue
        
        seen = {}
        for _, lr in player_odds.iterrows():
            line_val = float(lr['line'])
            side = lr['over_under']
            our_p = _our_prob_at(line_val, side, proj, std_for_player)
            book_p = calculate_implied_probability(int(lr['odds']))
            ev = calculate_expected_value(our_p, int(lr['odds']))
            rung = {
                'line': line_val,
                'side': side.upper(),
                'odds': int(lr['odds']),
                'bookmaker': lr['bookmaker'],
                'our_prob': round(our_p, 3),
                'book_prob': round(book_p, 3),
                'edge_pp': round((our_p - book_p) * 100, 1),
                'ev': round(ev, 3),
                'kelly_units': _ast_kelly(our_p, int(lr['odds'])),
            }
            key = (line_val, side)
            if key not in seen or rung['ev'] > seen[key]['ev']:
                seen[key] = rung
        candidate_rungs = list(seen.values())
        if not candidate_rungs:
            continue
        
        over_rows = player_odds[player_odds['over_under'] == 'Over']
        if over_rows.empty:
            continue
        main_line = float(over_rows.loc[over_rows['odds'].abs().idxmin(), 'line'])
        main_rungs = [r for r in candidate_rungs if r['line'] == main_line]
        if not main_rungs:
            continue
        main_pick = max(main_rungs, key=lambda r: r['ev'])
        
        results_df.at[idx, 'book_line'] = main_line
        results_df.at[idx, 'book_odds'] = main_pick['odds']
        results_df.at[idx, 'bookmaker'] = main_pick['bookmaker']
        results_df.at[idx, 'our_prob'] = main_pick['our_prob']
        results_df.at[idx, 'book_prob'] = main_pick['book_prob']
        results_df.at[idx, 'edge_pp'] = main_pick['edge_pp']
        results_df.at[idx, 'ev'] = main_pick['ev']
        results_df.at[idx, 'kelly_units'] = main_pick['kelly_units']
        if (
            main_pick['ev'] >= MIN_EV_PLAY
            and (main_pick['our_prob'] - main_pick['book_prob']) >= MIN_EDGE_PP_PLAY
        ):
            results_df.at[idx, 'recommended_side'] = main_pick['side']
        else:
            results_df.at[idx, 'recommended_side'] = 'PASS'
        
        alts = [
            r for r in candidate_rungs
            if r['line'] != main_line
            and r['side'] == main_pick['side']
            and r['ev'] >= MIN_EV_PLAY
        ]
        alts.sort(key=lambda r: r['ev'], reverse=True)
        if alts:
            alternates_by_player[player_name] = alts
    
    matched = (results_df['book_line'].notna()).sum()
    print(f"   ✓ Matched {matched} players with book odds")
else:
    print("   ⚠️  No book odds available for assists")

# Refresh smart_picks with the newly-added book/EV columns so the
# PROJECTIONS display can show them.
_pick_names = smart_picks['player_name'].tolist()
smart_picks = results_df[results_df['player_name'].isin(_pick_names)].copy()
smart_picks['_ord'] = smart_picks['player_name'].map({n: i for i, n in enumerate(_pick_names)})
smart_picks = smart_picks.sort_values('_ord').drop(columns=['_ord'])

print()
print("=" * 80)
print("📊 PROJECTIONS")
print("=" * 80)
print()

for i, (_, row) in enumerate(smart_picks.iterrows(), 1):
    location = "🏠" if row['is_home'] == 1 else "✈️"
    
    print(f"{i}. {row['player_name']} ({row['team']}) {location} vs {row['opponent']}")
    print("-" * 80)
    
    # Projection
    projection = row['final_projection']
    l10_avg = row['last_10_avg']
    season_avg = row['season_avg']
    
    print(f"   Our Projection: {projection:.1f} AST")
    print(f"   Season Avg: {season_avg:.1f} | L10 Avg: {l10_avg:.1f}")
    
    if pd.notna(row.get('book_line')):
        side = row.get('recommended_side', '')
        side_icon = "🟢 OVER" if side == 'OVER' else "🔴 UNDER" if side == 'UNDER' else "⚪ PASS"
        print(
            f"   Book: {row['book_line']:.1f} @ {int(row['book_odds']):+d} ({row['bookmaker']})  |  "
            f"Our {row['our_prob']:.1%} vs Book {row['book_prob']:.1%}  |  "
            f"EV {row['ev']:+.1%}  |  Stake {row['kelly_units']:.2f}u  {side_icon}"
        )
    print()
    
    # Reasons
    print(f"   Reasons:")
    print(f"     - Momentum: {row.get('ast_momentum', 0):+.2f}")
    print(f"     - Consistency (std dev): {row['consistency']:.2f}")
    print(f"     - Matchup: {row.get('matchup_quality', 'N/A')}")
    print(f"     - Pace: {row.get('pace_factor', 'N/A')}")
    print()

print("=" * 80)
print()

# ---------------------------------------------------------------------
# TOP PLAYS — EV-based recommendations + Kelly stakes + ladder rungs
# ---------------------------------------------------------------------
if not odds_df.empty and results_df['book_line'].notna().any():
    SUSPICIOUS_EV = 0.25
    plays = results_df[
        (results_df['recommended_side'].isin(['OVER', 'UNDER']))
        & (results_df['confidence'].isin(['HIGH', 'MEDIUM']))
    ].copy()
    plays = plays.sort_values('ev', ascending=False)

    print(f"\n🎯 TOP PLAYS — {len(plays)} qualifying assists bets today")
    print("=" * 80)
    if plays.empty:
        print("   No qualifying plays (no edges ≥+5% EV with HIGH/MEDIUM confidence)")
    else:
        MAX_PLAYER_EXPOSURE = 1.5
        for i, (_, p) in enumerate(plays.head(7).iterrows(), 1):
            conf_icon = '🟢' if p['confidence'] == 'HIGH' else '🟡'
            side_icon = '📈' if p['recommended_side'] == 'OVER' else '📉'
            verify = " ⚠️ VERIFY LINE" if p['ev'] > SUSPICIOUS_EV else ""

            main_rung = {
                'line': p['book_line'],
                'side': p['recommended_side'],
                'odds': int(p['book_odds']),
                'bookmaker': p['bookmaker'],
                'our_prob': p['our_prob'],
                'book_prob': p['book_prob'],
                'edge_pp': p['edge_pp'],
                'ev': p['ev'],
                'kelly_units': float(p.get('kelly_units', 0)),
                'is_main': True,
            }
            alts = alternates_by_player.get(p['player_name'], [])
            ladder = [main_rung] + [{**a, 'is_main': False} for a in alts]
            total_units = sum(r['kelly_units'] for r in ladder)
            if total_units > MAX_PLAYER_EXPOSURE and total_units > 0:
                scale = MAX_PLAYER_EXPOSURE / total_units
                for r in ladder:
                    r['kelly_units'] = round(r['kelly_units'] * scale, 2)

            print(
                f"\n{i}. {conf_icon} {side_icon} {p['recommended_side']} — "
                f"{p['player_name']} ({p['team']} vs {p['opponent']}){verify}"
            )
            print(
                f"   Projection: {p['final_projection']:.1f} AST | "
                f"L10: {p['last_10_avg']:.1f} | Season: {p['season_avg']:.1f}"
            )
            print(f"   Ladder ({len(ladder)} rung{'s' if len(ladder) > 1 else ''}, "
                  f"total {sum(r['kelly_units'] for r in ladder):.2f}u):")
            for r in ladder:
                tag = 'MAIN' if r['is_main'] else 'ALT '
                print(
                    f"     • {tag} {r['kelly_units']:.2f}u  "
                    f"{p['recommended_side']} {r['line']} @ {r['odds']:+d} "
                    f"({r['bookmaker']})  Our {r['our_prob']:.1%} | EV {r['ev']:+.1%}"
                )
    print("\n" + "=" * 80)
    print("💡 EV ≥ +5% with edge ≥ +4pp qualifies as a TOP PLAY.")
    print("=" * 80)
    print()

# ---------------------------------------------------------------------
# SAVE PREDICTIONS TO CSV — for downstream validate.py
# ---------------------------------------------------------------------
from datetime import datetime as _dt
date_str = _dt.now().strftime('%Y%m%d')
filename = f"predictions_assists_{date_str}.csv"

csv_columns = [
    'player_name', 'team', 'opponent', 'is_home',
    'final_projection', 'season_avg', 'last_10_avg',
    'consistency', 'confidence',
    'book_line', 'book_odds', 'bookmaker',
    'recommended_side', 'our_prob', 'book_prob',
    'edge_pp', 'ev', 'kelly_units',
    'play_quality', 'red_flags',
    'prob_3+', 'prob_5+', 'prob_7+', 'prob_10+',
]
available = [c for c in csv_columns if c in smart_picks.columns]
out_df = smart_picks[available].copy().rename(columns={
    'final_projection': 'projection',
    'season_avg': 'ast_season_avg',
    'last_10_avg': 'ast_last10_avg',
})
out_df.to_csv(filename, index=False)
print(f"✅ Predictions saved to: {filename}  ({len(out_df)} rows)")
