#!/usr/bin/env python3
"""Compare Phase 2 vs Phase 3 results"""

import pandas as pd
import os
from datetime import datetime

# Check for today's predictions
today = datetime.now().strftime('%Y%m%d')
csv_file = f'predictions_production_{today}.csv'

if not os.path.exists(csv_file):
    print(f"No predictions file found for today: {csv_file}")
    exit(1)

df = pd.read_csv(csv_file)

print("=" * 80)
print("PHASE 2 vs PHASE 3 COMPARISON")
print("=" * 80)

print(f"\nFile: {csv_file}")
print(f"Total Predictions: {len(df)}")

# Tier distribution
print("\nTIER DISTRIBUTION:")
top_plays = df[df['Type'] == 'TOP PLAY']
for tier in [1, 2, 3]:
    count = len(top_plays[top_plays['Tier'] == tier])
    pct = (count / len(top_plays) * 100) if len(top_plays) > 0 else 0
    print(f"   Tier {tier}: {count:2d} plays ({pct:5.1f}%)")

# Quality metrics
print("\nQUALITY METRICS:")
print(f"   Average Quality Score: {top_plays['Quality'].mean():.1f}/100")
print(f"   Average Ladder Value:  {top_plays['Ladder_Value'].mean():.1f}/100")

# High confidence plays
high_conf = top_plays[(top_plays['Quality'] >= 85) & (top_plays['Ladder_Value'] >= 75)]
print(f"   High Confidence (Q≥85, L≥75): {len(high_conf)} plays")

# Projection accuracy indicators
print("\nPROJECTION ANALYSIS:")
print(f"   Average Projection Ratio: {top_plays['Ratio'].mean():.2f}x")
print(f"   Average Std Dev: {top_plays['StdDev'].mean():.1f}")

# Usage boost analysis
usage_boost_plays = top_plays[top_plays['Tmts_Out'] >= 3]
print(f"\nUSAGE BOOST OPPORTUNITIES:")
print(f"   Players with 3+ teammates out: {len(usage_boost_plays)}")
if len(usage_boost_plays) > 0:
    print(f"   Average projection for these: {usage_boost_plays['Proj'].mean():.1f} pts")
    print(f"   Average boost ratio: {usage_boost_plays['Ratio'].mean():.2f}x")

# Red flags
print("\nRED FLAGS:")
clean_plays = top_plays[top_plays['Red_Flags'] == 'None']
flagged_plays = top_plays[top_plays['Red_Flags'] != 'None']
print(f"   Clean plays: {len(clean_plays)}")
print(f"   Flagged plays: {len(flagged_plays)}")

print("\n" + "=" * 80)

# If we have Phase 2 results, compare
phase2_file = 'predictions_production_phase2.csv'
if os.path.exists(phase2_file):
    print("\nCOMPARISON TO PHASE 2:")
    df2 = pd.read_csv(phase2_file)
    top2 = df2[df2['Type'] == 'TOP PLAY']
    
    print(f"   Quality Score: {top2['Quality'].mean():.1f} → {top_plays['Quality'].mean():.1f} ({top_plays['Quality'].mean() - top2['Quality'].mean():+.1f})")
    print(f"   Ladder Value:  {top2['Ladder_Value'].mean():.1f} → {top_plays['Ladder_Value'].mean():.1f} ({top_plays['Ladder_Value'].mean() - top2['Ladder_Value'].mean():+.1f})")
    print(f"   Avg Std Dev:   {top2['StdDev'].mean():.1f} → {top_plays['StdDev'].mean():.1f} ({top_plays['StdDev'].mean() - top2['StdDev'].mean():+.1f})")
    
    print("\n" + "=" * 80)
