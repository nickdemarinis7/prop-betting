#!/usr/bin/env python3
"""Analyze Phase 2 improvements in predictions"""

import pandas as pd
from collections import Counter

# Load predictions
df = pd.read_csv('predictions_production_20260409.csv')

print("=" * 80)
print("PHASE 2 ENHANCEMENTS - IMPACT ANALYSIS")
print("=" * 80)

print(f"\nTotal Predictions: {len(df)}")
print(f"   Top Plays: {len(df[df['Type'] == 'TOP PLAY'])}")
print(f"   Fade List: {len(df[df['Type'] == 'FADE'])}")

print("\nUSAGE BOOST ANALYSIS:")
top_plays = df[df['Type'] == 'TOP PLAY']
print(f"   Players with 2+ teammates out: {len(top_plays[top_plays['Tmts_Out'] >= 2])}")
print(f"   Players with 3+ teammates out: {len(top_plays[top_plays['Tmts_Out'] >= 3])}")
print(f"   Players with 5+ teammates out: {len(top_plays[top_plays['Tmts_Out'] >= 5])}")

if len(top_plays[top_plays['Tmts_Out'] >= 3]) > 0:
    print("\n   Top Usage Boost Candidates:")
    boost_plays = top_plays[top_plays['Tmts_Out'] >= 3].sort_values('Tmts_Out', ascending=False).head(5)
    for _, row in boost_plays.iterrows():
        player = row['Player']
        team = row['Team']
        tmts_out = int(row['Tmts_Out'])
        proj = row['Proj']
        l10 = row['L10']
        ratio = row['Ratio']
        print(f"   {player:20s} ({team}) - {tmts_out} out, Proj: {proj:.1f} (L10: {l10:.1f}, {ratio:.2f}x)")

print("\nPROJECTION QUALITY:")
print(f"   Average Ladder Value: {top_plays['Ladder_Value'].mean():.1f}/100")
print(f"   Average Quality Score: {top_plays['Quality'].mean():.1f}/100")
print(f"   Tier 1 plays: {len(top_plays[top_plays['Tier'] == 1])}")
print(f"   Tier 2 plays: {len(top_plays[top_plays['Tier'] == 2])}")
print(f"   Tier 3 plays: {len(top_plays[top_plays['Tier'] == 3])}")

print("\nHIGH-CONFIDENCE OPPORTUNITIES:")
high_conf = top_plays[(top_plays['Quality'] >= 80) & (top_plays['Ladder_Value'] >= 70)]
if len(high_conf) > 0:
    print(f"   Found {len(high_conf)} high-quality plays:")
    for _, row in high_conf.head(5).iterrows():
        player = row['Player']
        proj = row['Proj']
        matchup = row['Matchup']
        quality = row['Quality']
        ladder = row['Ladder_Value']
        flags = row['Red_Flags'] if row['Red_Flags'] != 'None' else 'Clean'
        print(f"   {player:20s} {proj:5.1f} pts ({matchup:9s}, Q:{quality:.0f}, L:{ladder:.0f}) - {flags}")
else:
    print("   No plays meeting criteria (Quality 80+, Ladder 70+)")

print("\nRED FLAGS SUMMARY:")
all_flags = []
for flags in df['Red_Flags']:
    if flags != 'None':
        all_flags.extend([f.strip() for f in str(flags).split(';')])

flag_counts = Counter(all_flags)
for flag, count in flag_counts.most_common(5):
    print(f"   {flag}: {count} players")

print("\n" + "=" * 80)
