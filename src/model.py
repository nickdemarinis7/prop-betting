"""
Machine learning model for NBA assists prediction
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import pickle
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FEATURES, MODELS_DIR


class AssistsPredictor:
    """ML model for predicting player assists"""
    
    def __init__(self, model_type='xgboost'):
        """
        Initialize predictor
        
        Args:
            model_type: 'xgboost', 'random_forest', or 'gradient_boosting'
        """
        self.model_type = model_type
        self.model = None
        self.feature_names = FEATURES
        self.is_trained = False
        
        if model_type == 'xgboost':
            self.model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
        elif model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train(self, X, y, test_size=0.2):
        """
        Train the model
        
        Args:
            X: Feature matrix (DataFrame)
            y: Target variable (Series or array)
            test_size: Proportion of data for testing
        
        Returns:
            Dictionary with training metrics
        """
        # Ensure we have the right features
        X_features = X[self.feature_names].copy()
        
        # Handle missing values
        X_features = X_features.fillna(0)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_features, y, test_size=test_size, random_state=42
        )
        
        # Train model
        print(f"Training {self.model_type} model...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        metrics = {
            'train_mae': mean_absolute_error(y_train, train_pred),
            'test_mae': mean_absolute_error(y_test, test_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
            'train_r2': r2_score(y_train, train_pred),
            'test_r2': r2_score(y_test, test_pred)
        }
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_features, y, cv=5, 
            scoring='neg_mean_absolute_error'
        )
        metrics['cv_mae'] = -cv_scores.mean()
        metrics['cv_mae_std'] = cv_scores.std()
        
        return metrics
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Feature matrix (DataFrame)
        
        Returns:
            Array of predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_features = X[self.feature_names].copy()
        X_features = X_features.fillna(0)
        
        return self.model.predict(X_features)
    
    def predict_with_confidence(self, X):
        """
        Make predictions with confidence intervals (for tree-based models)
        
        Args:
            X: Feature matrix (DataFrame)
        
        Returns:
            DataFrame with predictions and confidence intervals
        """
        predictions = self.predict(X)
        
        # For tree-based models, use individual tree predictions for uncertainty
        if hasattr(self.model, 'estimators_'):
            tree_predictions = np.array([
                tree.predict(X[self.feature_names].fillna(0)) 
                for tree in self.model.estimators_
            ])
            std = tree_predictions.std(axis=0)
            lower_bound = predictions - 1.96 * std
            upper_bound = predictions + 1.96 * std
        else:
            # Simple estimate for non-ensemble models
            std = predictions * 0.15  # 15% uncertainty
            lower_bound = predictions - std
            upper_bound = predictions + std
        
        return pd.DataFrame({
            'prediction': predictions,
            'lower_bound': np.maximum(lower_bound, 0),  # Can't be negative
            'upper_bound': upper_bound,
            'std': std
        })
    
    def get_feature_importance(self):
        """
        Get feature importance scores
        
        Returns:
            DataFrame with feature importance
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        if hasattr(self.model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance
        else:
            return pd.DataFrame()
    
    def save_model(self, filepath=None):
        """
        Save trained model to disk
        
        Args:
            filepath: Path to save model (defaults to MODELS_DIR)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        if filepath is None:
            filepath = MODELS_DIR / f'assists_model_{self.model_type}.pkl'
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath=None):
        """
        Load trained model from disk
        
        Args:
            filepath: Path to load model from
        """
        if filepath is None:
            filepath = MODELS_DIR / f'assists_model_{self.model_type}.pkl'
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        
        self.is_trained = True
        print(f"Model loaded from {filepath}")


class EnsemblePredictor:
    """Ensemble of multiple models for robust predictions"""
    
    def __init__(self):
        self.models = {
            'xgboost': AssistsPredictor('xgboost'),
            'random_forest': AssistsPredictor('random_forest'),
            'gradient_boosting': AssistsPredictor('gradient_boosting')
        }
        self.weights = {'xgboost': 0.5, 'random_forest': 0.25, 'gradient_boosting': 0.25}
    
    def train_all(self, X, y):
        """Train all models in the ensemble"""
        results = {}
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            metrics = model.train(X, y)
            results[name] = metrics
            print(f"{name} - Test MAE: {metrics['test_mae']:.3f}, Test R²: {metrics['test_r2']:.3f}")
        return results
    
    def predict(self, X):
        """Make weighted ensemble predictions"""
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(X)
        
        # Weighted average
        ensemble_pred = sum(
            predictions[name] * self.weights[name] 
            for name in self.models.keys()
        )
        return ensemble_pred


if __name__ == "__main__":
    # Test the model
    print("Testing Assists Prediction Model")
    print("=" * 50)
    
    # Create sample training data
    np.random.seed(42)
    n_samples = 200
    
    sample_data = pd.DataFrame({
        'potential_assists': np.random.uniform(5, 20, n_samples),
        'assists_per_game': np.random.uniform(2, 12, n_samples),
        'usage_rate': np.random.uniform(15, 35, n_samples),
        'pace': np.random.uniform(95, 105, n_samples),
        'minutes_per_game': np.random.uniform(20, 38, n_samples),
        'games_played': np.random.randint(10, 70, n_samples),
        'opponent_defensive_rating': np.random.uniform(105, 115, n_samples),
        'home_away': np.random.randint(0, 2, n_samples),
        'days_rest': np.random.randint(0, 4, n_samples),
        'team_assists_per_game': np.random.uniform(20, 30, n_samples)
    })
    
    # Create synthetic target (assists) with some correlation to features
    sample_target = (
        0.6 * sample_data['potential_assists'] + 
        0.3 * sample_data['assists_per_game'] +
        np.random.normal(0, 1, n_samples)
    )
    
    # Train model
    predictor = AssistsPredictor('xgboost')
    metrics = predictor.train(sample_data, sample_target)
    
    print("\nModel Performance:")
    print(f"Test MAE: {metrics['test_mae']:.3f}")
    print(f"Test RMSE: {metrics['test_rmse']:.3f}")
    print(f"Test R²: {metrics['test_r2']:.3f}")
    print(f"CV MAE: {metrics['cv_mae']:.3f} (+/- {metrics['cv_mae_std']:.3f})")
    
    print("\nFeature Importance:")
    print(predictor.get_feature_importance().to_string(index=False))
