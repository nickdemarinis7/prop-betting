"""
⚾ MLB Model A/B Comparison
Runs both current and simplified models for side-by-side comparison
"""

import pandas as pd
import subprocess
import sys
from datetime import datetime

print("=" * 80)
print("⚾ MLB MODEL A/B COMPARISON")
print("=" * 80)

target_date = sys.argv[1] if len(sys.argv) > 1 else None
date_str = target_date if target_date else datetime.now().strftime('%Y%m%d')

print(f"\n📅 Running comparison for: {date_str}")

# Run current model
print("\n" + "=" * 80)
print("🏃 RUNNING CURRENT MODEL (predict.py)")
print("=" * 80)
result_current = subprocess.run(
    ['python', 'predict.py'] + ([target_date] if target_date else []),
    capture_output=True,
    text=True
)

# Check if it ran successfully
if result_current.returncode != 0:
    print(f"❌ Current model failed:\n{result_current.stderr}")
else:
    print("✅ Current model completed")
    # Show last 30 lines of output
    lines = result_current.stdout.split('\n')
    print("\n".join(lines[-30:]))

# Run simplified model
print("\n" + "=" * 80)
print("🏃 RUNNING SIMPLIFIED MODEL (predict_simplified.py)")
print("=" * 80)
result_simplified = subprocess.run(
    ['python', 'predict_simplified.py'] + ([target_date] if target_date else []),
    capture_output=True,
    text=True
)

if result_simplified.returncode != 0:
    print(f"❌ Simplified model failed:\n{result_simplified.stderr}")
else:
    print("✅ Simplified model completed")
    lines = result_simplified.stdout.split('\n')
    print("\n".join(lines[-30:]))

# Load and compare outputs
print("\n" + "=" * 80)
print("📊 COMPARING PREDICTIONS")
print("=" * 80)

try:
    current_csv = f"predictions_strikeouts_{date_str.replace('-', '')}.csv"
    simplified_csv = f"predictions_strikeouts_simplified_{date_str.replace('-', '')}.csv"
    
    current_df = pd.read_csv(current_csv)
    simplified_df = pd.read_csv(simplified_csv)
    
    # Merge on pitcher name
    comparison = pd.merge(
        current_df[['pitcher', 'projection', 'confidence']],
        simplified_df[['pitcher', 'projection', 'confidence']],
        on='pitcher',
        suffixes=('_current', '_simplified')
    )
    
    comparison['diff'] = comparison['projection_simplified'] - comparison['projection_current']
    comparison['diff_pct'] = (comparison['diff'] / comparison['projection_current'] * 100).round(1)
    
    print(f"\nFound {len(comparison)} common pitchers")
    print(f"\nProjection Differences (Simplified - Current):")
    print(f"   Mean diff: {comparison['diff'].mean():+.2f} K")
    print(f"   Std dev:   {comparison['diff'].std():.2f} K")
    print(f"   Max diff:  {comparison['diff'].abs().max():.2f} K")
    
    # Show pitchers with >1K difference
    big_diffs = comparison[comparison['diff'].abs() > 1.0].sort_values('diff', ascending=False)
    
    if not big_diffs.empty:
        print(f"\n🔍 Pitchers with >1K difference:")
        for _, row in big_diffs.iterrows():
            direction = "⬆️" if row['diff'] > 0 else "⬇️"
            print(f"   {direction} {row['pitcher']:25s} "
                  f"Current: {row['projection_current']:4.1f}K | "
                  f"Simplified: {row['projection_simplified']:4.1f}K "
                  f"({row['diff']:+.1f}K)")
    
    # Save comparison
    comparison_file = f"model_comparison_{date_str.replace('-', '')}.csv"
    comparison.to_csv(comparison_file, index=False)
    print(f"\n✅ Comparison saved to: {comparison_file}")
    
    print("\n" + "=" * 80)
    print("📝 VALIDATION INSTRUCTIONS:")
    print("=" * 80)
    print("1. Tomorrow, run: python validate.py --date YYYYMMDD")
    print("2. Compare MAE between models:")
    print("   - Current:    Look at validation_results_YYYYMMDD.csv")
    print("   - Simplified: Look at validation_results_simplified_YYYYMMDD.csv")
    print("3. If simplified performs better, we'll make it the default")
    print("\n" + "=" * 80)
    
except FileNotFoundError as e:
    print(f"\n⚠️  Could not load CSV files: {e}")
    print("   Models may have failed or saved to different filenames")
