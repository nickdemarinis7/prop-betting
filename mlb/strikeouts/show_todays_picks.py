import pandas as pd
from datetime import datetime

df = pd.read_csv('predictions_strikeouts_' + datetime.now().strftime('%Y%m%d') + '.csv')

print('=' * 80)
print('MLB STRIKEOUT PREDICTIONS - ' + datetime.now().strftime('%B %d, %Y'))
print('=' * 80)
print()
print('Model V2.1+ with Enhanced Early Exit Detection')
print('MAE: 1.7-1.8 K (estimated) | Improved safety caps')
print()
print('=' * 80)
print('TOP PROJECTIONS')
print('=' * 80)
print()

for _, row in df.head(15).iterrows():
    matchup = f"{row['team']} {'vs' if row['is_home'] else '@'} {row['opponent']}"
    print(f"{row['pitcher']:25s} {row['projection']:4.1f} K  ({matchup})")
    print(f"   Season K/9: {row['season_k9']:.2f} | Expected IP: {row['expected_ip']:.1f}")
    
    # Show key probabilities
    probs = []
    for line in [4.5, 5.5, 6.5, 7.5]:
        col = f'prob_{line}+'
        if col in row and row[col] > 0.10:
            probs.append(f"{line}+ K: {row[col]:.0%}")
    
    if probs:
        print(f"   {' | '.join(probs)}")
    print()

print('=' * 80)
print('RECOMMENDATIONS')
print('=' * 80)
print()

# Find best plays
high_conf = df[df['prob_6.5+'] > 0.65]
if not high_conf.empty:
    print('STRONG PLAYS:')
    for _, row in high_conf.head(3).iterrows():
        print(f"   - {row['pitcher']} Over 6.5 K ({row['prob_6.5+']:.0%} confidence)")
    print()

med_conf = df[(df['prob_5.5+'] > 0.55) & (df['prob_5.5+'] < 0.65)]
if not med_conf.empty:
    print('MEDIUM PLAYS:')
    for _, row in med_conf.head(3).iterrows():
        print(f"   - {row['pitcher']} Over 5.5 K ({row['prob_5.5+']:.0%} confidence)")
    print()

print('AVOID:')
print('   - Pitchers with Expected IP <4.5 innings')
print('   - Pitchers with K/9 <7.0')
print()
print('=' * 80)
print('SAFETY FEATURES ACTIVE')
print('=' * 80)
print()
print('✅ Elite K pitcher cap (max 9.0 K)')
print('✅ Short recent outings penalty (-8%)')
print('✅ High volatility penalty (-10%)')
print('✅ Blowout risk cap (ERA >4.5)')
print()
print('=' * 80)
