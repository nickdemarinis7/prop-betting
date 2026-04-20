"""
ML Residual Corrector for Strikeout Predictions
Learns systematic biases in the heuristic model from historical validation data.

Usage:
    - Standalone: python ml_corrector.py          (train/retrain)
    - Integrated: imported by predict.py automatically
"""

import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime

try:
    import joblib
    from xgboost import XGBRegressor
    ML_DEPS_AVAILABLE = True
except ImportError:
    ML_DEPS_AVAILABLE = False

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ml_model_corrector.joblib')
MIN_TRAINING_SAMPLES = 30


class StrikeoutMLCorrector:
    """Learns residual corrections on top of the heuristic strikeout model."""

    FEATURE_COLS = ['season_k9', 'recent_k9', 'expected_ip', 'opponent_k_rate',
                    'is_home', 'is_day_game', 'is_short_rest']

    def __init__(self):
        self.model = None
        self.is_trained = False

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def load_training_data(self):
        """Load and join prediction + validation CSVs from the strikeouts directory."""
        base_dir = os.path.dirname(os.path.abspath(__file__))

        pred_files = sorted(glob.glob(os.path.join(base_dir, 'predictions_strikeouts_*.csv')))
        val_files = sorted(glob.glob(os.path.join(base_dir, 'validation_results_*.csv')))

        # Map date -> validation file
        val_dates = {}
        for f in val_files:
            date_str = os.path.basename(f).replace('validation_results_', '').replace('.csv', '')
            val_dates[date_str] = f

        all_data = []
        for pred_file in pred_files:
            date_str = os.path.basename(pred_file).replace('predictions_strikeouts_', '').replace('.csv', '')

            if date_str not in val_dates:
                continue

            try:
                pred_df = pd.read_csv(pred_file)
                val_df = pd.read_csv(val_dates[date_str])

                # Skip files that lack required feature columns
                missing_cols = [c for c in self.FEATURE_COLS if c not in pred_df.columns]
                if missing_cols:
                    continue

                merged = pred_df.merge(val_df[['pitcher', 'actual']], on='pitcher', how='inner')
                merged['date'] = date_str
                all_data.append(merged)
            except Exception as e:
                print(f"   ⚠️  Error loading {date_str}: {e}")
                continue

        if not all_data:
            return pd.DataFrame()

        return pd.concat(all_data, ignore_index=True)

    # ------------------------------------------------------------------
    # Feature prep
    # ------------------------------------------------------------------
    def _prepare_features(self, df):
        """Prepare feature matrix from a dataframe."""
        X = df[self.FEATURE_COLS].copy()
        for col in ['is_home', 'is_day_game', 'is_short_rest']:
            X[col] = X[col].astype(int)
        return X

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self):
        """Train the residual correction model on historical data."""
        if not ML_DEPS_AVAILABLE:
            print("   ⚠️  ML Corrector: xgboost or joblib not installed. Skipping.")
            return False

        data = self.load_training_data()

        if len(data) < MIN_TRAINING_SAMPLES:
            print(f"   ⚠️  ML Corrector: Only {len(data)} samples (need {MIN_TRAINING_SAMPLES}). Skipping.")
            return False

        X = self._prepare_features(data)

        # Target: residual (actual - heuristic projection)
        y = data['actual'] - data['projection']

        self.model = XGBRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=1.0,
            reg_lambda=2.0,
            random_state=42
        )

        self.model.fit(X, y)
        self.is_trained = True

        train_preds = self.model.predict(X)
        train_mae = float(np.mean(np.abs(y - train_preds)))

        joblib.dump({
            'model': self.model,
            'feature_cols': self.FEATURE_COLS,
            'n_samples': len(data),
            'trained_date': datetime.now().isoformat(),
            'train_mae': train_mae,
            'mean_residual': float(y.mean()),
        }, MODEL_PATH)

        print(f"   ✅ ML Corrector trained on {len(data)} samples")
        print(f"      Training residual MAE: {train_mae:.2f} K")
        print(f"      Mean residual (bias): {y.mean():+.2f} K")

        return True

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load(self):
        """Load a previously trained model from disk."""
        if not ML_DEPS_AVAILABLE:
            return False

        if not os.path.exists(MODEL_PATH):
            return False

        try:
            saved = joblib.load(MODEL_PATH)
            self.model = saved['model']
            self.is_trained = True
            print(f"   ✅ ML Corrector loaded ({saved['n_samples']} samples, trained {saved['trained_date'][:10]})")
            return True
        except Exception as e:
            print(f"   ⚠️  Error loading ML model: {e}")
            return False

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict_correction(self, features_dict):
        """
        Predict residual correction for a single pitcher.

        Args:
            features_dict: dict with keys matching FEATURE_COLS

        Returns:
            float: correction to add to the heuristic projection
        """
        if not self.is_trained:
            return 0.0

        try:
            X = pd.DataFrame([features_dict])[self.FEATURE_COLS]
            for col in ['is_home', 'is_day_game', 'is_short_rest']:
                X[col] = X[col].astype(int)

            correction = self.model.predict(X)[0]

            # Cap correction to prevent wild swings
            correction = float(np.clip(correction, -1.5, 1.5))
            return correction
        except Exception as e:
            print(f"   ⚠️  ML prediction error: {e}")
            return 0.0


# ======================================================================
# Standalone training script
# ======================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🤖 ML Strikeout Corrector - Training")
    print("=" * 60)

    corrector = StrikeoutMLCorrector()

    # Load training data
    data = corrector.load_training_data()
    print(f"\n📊 Training data: {len(data)} samples")

    if not data.empty:
        residuals = data['actual'] - data['projection']
        print(f"   Mean residual: {residuals.mean():+.2f}")
        print(f"   Std residual:  {residuals.std():.2f}")
        print(f"   Dates covered: {sorted(data['date'].unique())}")

    # Train
    success = corrector.train()

    if success:
        print("\n✅ Model saved. It will be used automatically by predict.py")
    else:
        print("\n⚠️  Training skipped (not enough data yet)")
