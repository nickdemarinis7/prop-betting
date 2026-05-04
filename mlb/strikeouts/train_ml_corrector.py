"""
🤖 MLB Strikeout ML Corrector Training
Trains model on historical predictions vs actuals to learn optimal corrections
"""

import pandas as pd
import numpy as np
import sys
import os
import glob
from datetime import datetime, timedelta
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import pickle

print("=" * 80)
print("🤖 TRAINING ML CORRECTOR")
print("=" * 80)

# Find all validation results
validation_files = glob.glob("validation_results_*.csv")
prediction_files = glob.glob("predictions_strikeouts_simplified_*.csv")

if not validation_files:
    print("\n❌ No validation files found!")
    print("   Run validate.py first to generate validation_results_*.csv")
    sys.exit(1)

print(f"\n📊 Found {len(validation_files)} validation files")

# Load and combine all validation data
all_validations = []
for vf in validation_files:
    try:
        df = pd.read_csv(vf)
        df['validation_file'] = vf
        all_validations.append(df)
        print(f"   ✓ Loaded {vf}: {len(df)} rows")
    except Exception as e:
        print(f"   ⚠️  Error loading {vf}: {e}")

if not all_validations:
    print("\n❌ No valid validation data found")
    sys.exit(1)

combined = pd.concat(all_validations, ignore_index=True)
print(f"\n📊 Total validation samples: {len(combined)}")

# Calculate error (what we want the model to predict and correct)
combined['error'] = combined['actual'] - combined['projection']
combined['abs_error'] = combined['error'].abs()

print(f"\n📈 Current Performance:")
print(f"   MAE: {combined['abs_error'].mean():.2f} K")
print(f"   Bias: {combined['error'].mean():+.2f} K")
print(f"   R²: {1 - (combined['error'].var() / combined['actual'].var()):.1%}")

# Prepare features for ML model
# These are the inputs the corrector will have at prediction time
feature_cols = [
    'season_k9', 'recent_k9', 'expected_ip', 'opponent_k_rate',
    'opponent_multiplier', 'is_home', 'weight_2026', 'blowup_rate',
    'recent_era', 'base_projection'
]

# Check which columns are available
available_features = [c for c in feature_cols if c in combined.columns]
missing_features = [c for c in feature_cols if c not in combined.columns]

if missing_features:
    print(f"\n⚠️  Missing features in validation data: {missing_features}")
    print(f"   Available features: {available_features}")

if not available_features:
    print("\n❌ No features available for training!")
    print("   Need predictions_strikeouts_simplified_*.csv format")
    sys.exit(1)

# Prepare training data
X = combined[available_features].fillna(0)
y = combined['error']  # Target: predict the error (so we can correct for it)

print(f"\n🎯 Training ML Corrector to predict error...")
print(f"   Features: {available_features}")
print(f"   Target: error (actual - projection)")
print(f"   Samples: {len(X)}")

# Split for evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

train_mae = mean_absolute_error(y_train, train_pred)
test_mae = mean_absolute_error(y_test, test_pred)

print(f"\n📊 Model Performance:")
print(f"   Train MAE: {train_mae:.2f} K")
print(f"   Test MAE:  {test_mae:.2f} K")

# Simulate: What if we applied these corrections?
# Corrected projection = original projection + predicted error
corrected_test = combined.iloc[X_test.index]['projection'] + test_pred
corrected_mae = mean_absolute_error(
    combined.iloc[X_test.index]['actual'], 
    corrected_test
)
original_mae = combined.iloc[X_test.index]['abs_error'].mean()

print(f"\n💡 Impact Simulation:")
print(f"   Original MAE: {original_mae:.2f} K")
print(f"   Corrected MAE: {corrected_mae:.2f} K")
print(f"   Improvement: {original_mae - corrected_mae:+.2f} K ({(original_mae - corrected_mae)/original_mae:+.1%})")

# Feature importance
importance = pd.DataFrame({
    'feature': available_features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n🔍 Feature Importance:")
for _, row in importance.iterrows():
    print(f"   {row['feature']:20s} {row['importance']:.3f}")

# Save model
model_data = {
    'model': model,
    'features': available_features,
    'train_date': datetime.now().isoformat(),
    'samples': len(X),
    'test_mae': test_mae,
    'impact': original_mae - corrected_mae
}

output_file = "ml_corrector_trained.pkl"
with open(output_file, 'wb') as f:
    pickle.dump(model_data, f)

print(f"\n✅ Model saved to: {output_file}")
print(f"   Update predict_simplified.py to use this model for corrections")
