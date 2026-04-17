#!/usr/bin/env python3
"""
Calibration Analysis for Points Predictions
Checks if predicted probabilities match actual outcomes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from shared.scrapers.gamelog import GameLogScraper


def load_historical_predictions(days_back=7):
    """Load prediction files from the last N days"""
    predictions = []
    
    for i in range(days_back):
        date = datetime.now() - timedelta(days=i)
        date_str = date.strftime('%Y%m%d')
        filename = f'predictions_production_{date_str}.csv'
        
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            df['prediction_date'] = date_str
            predictions.append(df)
            print(f"   ✓ Loaded {filename} ({len(df)} predictions)")
        else:
            print(f"   ⚠️  Missing {filename}")
    
    if not predictions:
        return pd.DataFrame()
    
    return pd.concat(predictions, ignore_index=True)


def get_actual_results(predictions_df):
    """Fetch actual game results for predicted players"""
    scraper = GameLogScraper()
    results = []
    
    print("\n📊 Fetching actual game results...")
    
    for _, pred in predictions_df.iterrows():
        player_name = pred['Player']
        pred_date = pred['prediction_date']
        
        # Get player's games around this date
        try:
            # This is a simplified version - you'd need player_id lookup
            # For now, we'll create a placeholder
            results.append({
                'player_name': player_name,
                'prediction_date': pred_date,
                'predicted_pts': pred['Proj'],
                'actual_pts': None,  # Would fetch from game logs
                'prob_15+': pred.get('5+%', 0) / 100,
                'prob_20+': pred.get('7+%', 0) / 100,
                'prob_25+': pred.get('10+%', 0) / 100,
                'hit_15+': None,
                'hit_20+': None,
                'hit_25+': None
            })
        except Exception as e:
            print(f"   ⚠️  Error fetching {player_name}: {e}")
            continue
    
    return pd.DataFrame(results)


def calculate_calibration(results_df, prob_col, hit_col, n_bins=10):
    """Calculate calibration for a specific threshold"""
    
    # Remove rows with missing data
    valid_data = results_df[[prob_col, hit_col]].dropna()
    
    if len(valid_data) == 0:
        return None, None
    
    # Create probability bins
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    predicted_probs = []
    actual_rates = []
    counts = []
    
    for i in range(n_bins):
        bin_mask = (valid_data[prob_col] >= bins[i]) & (valid_data[prob_col] < bins[i+1])
        bin_data = valid_data[bin_mask]
        
        if len(bin_data) > 0:
            predicted_probs.append(bin_data[prob_col].mean())
            actual_rates.append(bin_data[hit_col].mean())
            counts.append(len(bin_data))
    
    return predicted_probs, actual_rates, counts


def plot_calibration_curve(predicted_probs, actual_rates, threshold_name):
    """Plot calibration curve"""
    plt.figure(figsize=(8, 8))
    
    # Perfect calibration line
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
    
    # Actual calibration
    if predicted_probs and actual_rates:
        plt.plot(predicted_probs, actual_rates, 'bo-', 
                label='Model Calibration', linewidth=2, markersize=8)
    
    plt.xlabel('Predicted Probability', fontsize=12)
    plt.ylabel('Actual Hit Rate', fontsize=12)
    plt.title(f'Calibration Curve - {threshold_name}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    # Save plot
    filename = f'calibration_{threshold_name.lower().replace("+", "plus")}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved {filename}")
    plt.close()


def calculate_brier_score(predicted_probs, actual_outcomes):
    """Calculate Brier score (lower is better, 0 = perfect)"""
    valid_mask = ~(pd.isna(predicted_probs) | pd.isna(actual_outcomes))
    
    if valid_mask.sum() == 0:
        return None
    
    pred = predicted_probs[valid_mask]
    actual = actual_outcomes[valid_mask]
    
    return np.mean((pred - actual) ** 2)


def main():
    print("=" * 80)
    print("📊 CALIBRATION ANALYSIS")
    print("=" * 80)
    
    # Load historical predictions
    print("\n📁 Loading historical predictions...")
    predictions = load_historical_predictions(days_back=7)
    
    if predictions.empty:
        print("\n❌ No historical predictions found!")
        print("   Run predictions for several days first, then run calibration analysis.")
        return
    
    print(f"\n✓ Loaded {len(predictions)} total predictions from {predictions['prediction_date'].nunique()} days")
    
    # Get actual results
    results = get_actual_results(predictions)
    
    print("\n" + "=" * 80)
    print("⚠️  NOTE: Actual results fetching not fully implemented")
    print("=" * 80)
    print("\nTo complete calibration analysis, you need to:")
    print("1. Implement player_id lookup from player names")
    print("2. Fetch actual game logs for prediction dates")
    print("3. Match predictions to actual results")
    print("4. Calculate hit rates for each threshold")
    print("\nFor now, this is a framework. Run validate.py for actual backtesting.")
    
    # Placeholder calibration analysis
    print("\n" + "=" * 80)
    print("📈 CALIBRATION FRAMEWORK READY")
    print("=" * 80)
    print("\nOnce actual results are available, this will show:")
    print("   • Calibration curves for each threshold (15+, 20+, 25+)")
    print("   • Brier scores (prediction accuracy)")
    print("   • Over/under-confidence analysis")
    print("   • Recommended probability adjustments")
    
    print("\n💡 TIP: Use validate.py for immediate backtesting of yesterday's predictions")
    print("=" * 80)


if __name__ == "__main__":
    main()
