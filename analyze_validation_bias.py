import pandas as pd
import glob
import os
from collections import defaultdict

# Analyze NBA assists validation
print("=" * 80)
print("NBA ASSISTS VALIDATION ANALYSIS")
print("=" * 80)

assists_files = sorted(glob.glob("/Users/nickdemarinis/CascadeProjects/windsurf-project/nba/assists/validation_results_*.csv"))
assists_data = []

for file in assists_files:
    try:
        df = pd.read_csv(file)
        assists_data.append(df)
    except Exception as e:
        print(f"Error reading {file}: {e}")

if assists_data:
    assists_all = pd.concat(assists_data, ignore_index=True)
    
    print(f"\nTotal records: {len(assists_all)}")
    print(f"TOP PLAY count: {len(assists_all[assists_all['Type'] == 'TOP PLAY'])}")
    print(f"FADE count: {len(assists_all[assists_all['Type'] == 'FADE'])}")
    
    # Analyze TOP PLAY hit rates vs probabilities
    top_play = assists_all[assists_all['Type'] == 'TOP PLAY'].copy()
    
    # Fix probability columns (stored as integers, need to divide by 100)
    for threshold in ['5+', '7+', '10+']:
        prob_col = f'Prob_{threshold}'
        if prob_col in top_play.columns:
            top_play[prob_col] = top_play[prob_col] / 100
    
    print("\n" + "=" * 80)
    print("TOP PLAY: Actual Hit Rate vs Predicted Probability")
    print("=" * 80)
    
    for threshold in ['5+', '7+', '10+']:
        hit_col = f'Hit_{threshold}'
        prob_col = f'Prob_{threshold}'
        
        actual_hit_rate = top_play[hit_col].mean()
        avg_prob = top_play[prob_col].mean()
        diff = actual_hit_rate - avg_prob
        
        print(f"\n{threshold}:")
        print(f"  Actual Hit Rate: {actual_hit_rate:.1%}")
        print(f"  Avg Predicted Prob: {avg_prob:.1%}")
        print(f"  Difference: {diff:+.1%}")
        
        if diff < -0.05:
            print(f"  ⚠️  UNDERPERFORMING by {abs(diff):.1%}")
        elif diff > 0.05:
            print(f"  ✅ OVERPERFORMING by {diff:.1%}")
    
    # Analyze FADE hit rates vs probabilities
    fade = assists_all[assists_all['Type'] == 'FADE'].copy()
    
    # Fix probability columns (stored as integers, need to divide by 100)
    for threshold in ['5+', '7+', '10+']:
        prob_col = f'Prob_{threshold}'
        if prob_col in fade.columns:
            fade[prob_col] = fade[prob_col] / 100
    
    print("\n" + "=" * 80)
    print("FADE: Actual Hit Rate vs Predicted Probability")
    print("=" * 80)
    
    for threshold in ['5+', '7+', '10+']:
        hit_col = f'Hit_{threshold}'
        prob_col = f'Prob_{threshold}'
        
        if prob_col in fade.columns:
            actual_hit_rate = fade[hit_col].mean()
            avg_prob = fade[prob_col].mean()
            diff = actual_hit_rate - avg_prob
            
            print(f"\n{threshold}:")
            print(f"  Actual Hit Rate: {actual_hit_rate:.1%}")
            print(f"  Avg Predicted Prob: {avg_prob:.1%}")
            print(f"  Difference: {diff:+.1%}")
            
            if diff < -0.05:
                print(f"  ⚠️  UNDERPERFORMING by {abs(diff):.1%}")
            elif diff > 0.05:
                print(f"  ✅ OVERPERFORMING by {diff:.1%}")

# Analyze NBA points validation
print("\n" + "=" * 80)
print("NBA POINTS VALIDATION ANALYSIS")
print("=" * 80)

points_files = sorted(glob.glob("/Users/nickdemarinis/CascadeProjects/windsurf-project/nba/points/validation_results_*.csv"))
points_data = []

for file in points_files:
    try:
        df = pd.read_csv(file)
        points_data.append(df)
    except Exception as e:
        print(f"Error reading {file}: {e}")

if points_data:
    points_all = pd.concat(points_data, ignore_index=True)
    
    print(f"\nTotal records: {len(points_all)}")
    print(f"TOP PLAY count: {len(points_all[points_all['Type'] == 'TOP PLAY'])}")
    print(f"FADE count: {len(points_all[points_all['Type'] == 'FADE'])}")
    
    # Analyze TOP PLAY hit rates vs probabilities
    top_play = points_all[points_all['Type'] == 'TOP PLAY'].copy()
    
    # Fix probability columns (stored as integers, need to divide by 100)
    for threshold in ['15+', '20+', '25+']:
        prob_col = f'Prob_{threshold}'
        if prob_col in top_play.columns:
            top_play[prob_col] = top_play[prob_col] / 100
    
    print("\n" + "=" * 80)
    print("TOP PLAY: Actual Hit Rate vs Predicted Probability")
    print("=" * 80)
    
    for threshold in ['15+', '20+', '25+']:
        hit_col = f'Hit_{threshold}'
        prob_col = f'Prob_{threshold}'
        
        actual_hit_rate = top_play[hit_col].mean()
        avg_prob = top_play[prob_col].mean()
        diff = actual_hit_rate - avg_prob
        
        print(f"\n{threshold}:")
        print(f"  Actual Hit Rate: {actual_hit_rate:.1%}")
        print(f"  Avg Predicted Prob: {avg_prob:.1%}")
        print(f"  Difference: {diff:+.1%}")
        
        if diff < -0.05:
            print(f"  ⚠️  UNDERPERFORMING by {abs(diff):.1%}")
        elif diff > 0.05:
            print(f"  ✅ OVERPERFORMING by {diff:.1%}")
    
    # Analyze FADE hit rates vs probabilities
    fade = points_all[points_all['Type'] == 'FADE'].copy()
    
    # Fix probability columns (stored as integers, need to divide by 100)
    for threshold in ['15+', '20+', '25+']:
        prob_col = f'Prob_{threshold}'
        if prob_col in fade.columns:
            fade[prob_col] = fade[prob_col] / 100
    
    print("\n" + "=" * 80)
    print("FADE: Actual Hit Rate vs Predicted Probability")
    print("=" * 80)
    
    for threshold in ['15+', '20+', '25+']:
        hit_col = f'Hit_{threshold}'
        prob_col = f'Prob_{threshold}'
        
        if prob_col in fade.columns and len(fade) > 0:
            actual_hit_rate = fade[hit_col].mean()
            avg_prob = fade[prob_col].mean()
            diff = actual_hit_rate - avg_prob
            
            print(f"\n{threshold}:")
            print(f"  Actual Hit Rate: {actual_hit_rate:.1%}")
            print(f"  Avg Predicted Prob: {avg_prob:.1%}")
            print(f"  Difference: {diff:+.1%}")
            
            if diff < -0.05:
                print(f"  ⚠️  UNDERPERFORMING by {abs(diff):.1%}")
            elif diff > 0.05:
                print(f"  ✅ OVERPERFORMING by {diff:.1%}")

# Analyze MLB strikeouts validation
print("\n" + "=" * 80)
print("MLB STRIKEOUTS VALIDATION ANALYSIS")
print("=" * 80)

mlb_files = sorted(glob.glob("/Users/nickdemarinis/CascadeProjects/windsurf-project/mlb/strikeouts/validation_results_*.csv"))
mlb_data = []

for file in mlb_files:
    try:
        df = pd.read_csv(file)
        mlb_data.append(df)
    except Exception as e:
        print(f"Error reading {file}: {e}")

if mlb_data:
    mlb_all = pd.concat(mlb_data, ignore_index=True)
    
    print(f"\nTotal records: {len(mlb_all)}")
    
    # Analyze hit rates vs probabilities for all pitchers
    print("\n" + "=" * 80)
    print("ALL PITCHERS: Actual Hit Rate vs Predicted Probability")
    print("=" * 80)
    
    for threshold in [3.5, 4.5, 5.5, 6.5, 7.5]:
        hit_col = f'hit_{threshold}'
        prob_col = f'prob_{threshold}'
        
        actual_hit_rate = mlb_all[hit_col].mean()
        avg_prob = mlb_all[prob_col].mean()
        diff = actual_hit_rate - avg_prob
        
        print(f"\n{threshold}:")
        print(f"  Actual Hit Rate: {actual_hit_rate:.1%}")
        print(f"  Avg Predicted Prob: {avg_prob:.1%}")
        print(f"  Difference: {diff:+.1%}")
        
        if diff < -0.05:
            print(f"  ⚠️  UNDERPERFORMING by {abs(diff):.1%}")
        elif diff > 0.05:
            print(f"  ✅ OVERPERFORMING by {diff:.1%}")
    
    # Analyze by confidence level
    print("\n" + "=" * 80)
    print("BY CONFIDENCE LEVEL: Actual Hit Rate vs Predicted Probability")
    print("=" * 80)
    
    for conf in ['HIGH', 'MEDIUM', 'LOW']:
        if conf in mlb_all['confidence'].values:
            conf_data = mlb_all[mlb_all['confidence'] == conf]
            print(f"\n{conf} confidence (n={len(conf_data)}):")
            
            for threshold in [4.5, 5.5, 6.5]:
                hit_col = f'hit_{threshold}'
                prob_col = f'prob_{threshold}'
                
                actual_hit_rate = conf_data[hit_col].mean()
                avg_prob = conf_data[prob_col].mean()
                diff = actual_hit_rate - avg_prob
                
                print(f"  {threshold}: Actual {actual_hit_rate:.1%} vs Predicted {avg_prob:.1%} (Diff: {diff:+.1%})")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
