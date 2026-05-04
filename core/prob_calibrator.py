"""
Generic probability calibrator using isotonic regression.

Maps raw predicted probabilities (e.g. from a normal CDF) to empirically
calibrated hit rates derived from validation history.

Each sport/market keeps its own joblib file so calibrations don't bleed
across markets (assists shape ≠ points shape ≠ strikeouts shape).
"""

import os
import glob
import joblib
import numpy as np
import pandas as pd

try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class ProbabilityCalibrator:
    """Fits one isotonic regression mapping raw_prob → calibrated_prob.

    Pools all (line, prob, hit) tuples across validation files. We pool
    because we want a single calibration curve over the full prob range,
    and most validation has fewer than ~1500 samples.
    """

    def __init__(self, model_path, validation_glob, prob_cols, hit_cols,
                 prob_scale=1.0, min_samples=100, method='isotonic',
                 file_filter=None):
        """
        Args:
            model_path: Where to save/load the joblib model
            validation_glob: Glob pattern for validation CSVs
            prob_cols: List of probability column names (e.g. ['prob_3.5+'])
            hit_cols:  Matching list of hit columns (e.g. ['hit_3.5'])
            prob_scale: Multiplier if probs are stored as percent (use 0.01)
            min_samples: Minimum (prob, hit) pairs required to train
            method: 'isotonic' (flexible, can over-fit small data) or
                    'platt' (sigmoid, much more robust to <1000 samples)
            file_filter: Optional callable(filepath) -> bool.  Return True to
                         include the file, False to skip it.  Useful for
                         excluding pre-playoff / outlier dates.
        """
        self.model_path = model_path
        self.validation_glob = validation_glob
        self.prob_cols = prob_cols
        self.hit_cols = hit_cols
        self.prob_scale = prob_scale
        self.min_samples = min_samples
        self.method = method
        self.file_filter = file_filter
        self.model = None
        self.is_fitted = False

    def load_training_data(self):
        files = sorted(glob.glob(self.validation_glob))
        rows = []
        for f in files:
            if self.file_filter is not None and not self.file_filter(f):
                continue
            try:
                df = pd.read_csv(f)
            except Exception:
                continue
            for pcol, hcol in zip(self.prob_cols, self.hit_cols):
                if pcol not in df.columns or hcol not in df.columns:
                    continue
                sub = df[df[pcol].notna() & df[hcol].notna()]
                for _, r in sub.iterrows():
                    p = float(r[pcol]) * self.prob_scale
                    h = int(bool(r[hcol]))
                    rows.append({'prob': p, 'hit': h})
        return pd.DataFrame(rows)

    def train(self):
        if not SKLEARN_AVAILABLE:
            print("   ⚠️  sklearn not installed; calibrator disabled.")
            return False
        data = self.load_training_data()
        if len(data) < self.min_samples:
            print(f"   ⚠️  Calibrator: only {len(data)} samples "
                  f"(need {self.min_samples}).")
            return False

        x = data['prob'].values
        y = data['hit'].values

        if self.method == 'platt':
            # Platt scaling: logistic regression on the raw probability.
            # Much smoother than isotonic and far less prone to over-fitting
            # on small validation sets (<500 samples).
            self.model = LogisticRegression(C=1.0, solver='lbfgs')
            self.model.fit(x.reshape(-1, 1), y)
        else:
            self.model = IsotonicRegression(
                out_of_bounds='clip', y_min=0.0, y_max=1.0, increasing=True,
            )
            self.model.fit(x, y)
        self.is_fitted = True

        raw_mean = x.mean()
        cal_mean = self._predict(x).mean()
        actual_mean = y.mean()
        print(f"   ✅ Prob calibrator fitted ({self.method}) on {len(data)} samples")
        print(f"      Raw mean prob:   {raw_mean:.1%}")
        print(f"      Calibrated:      {cal_mean:.1%}")
        print(f"      Actual hit rate: {actual_mean:.1%}")

        joblib.dump({
            'model': self.model,
            'method': self.method,
            'n_samples': len(data),
        }, self.model_path)
        return True

    def load(self):
        if not SKLEARN_AVAILABLE or not os.path.exists(self.model_path):
            return False
        try:
            saved = joblib.load(self.model_path)
            self.model = saved['model']
            # Backward-compat: older joblibs didn't store method
            self.method = saved.get('method', 'isotonic')
            self.is_fitted = True
            print(
                f"   ✅ Prob calibrator loaded "
                f"({saved['n_samples']} samples, {self.method}) "
                f"from {os.path.basename(self.model_path)}"
            )
            return True
        except Exception as e:
            print(f"   ⚠️  Error loading calibrator: {e}")
            return False

    def _predict(self, arr):
        if self.method == 'platt':
            # LogisticRegression: predict_proba returns [P(0), P(1)]
            return self.model.predict_proba(arr.reshape(-1, 1))[:, 1]
        return self.model.predict(arr)

    def calibrate(self, raw_prob):
        if not self.is_fitted:
            return raw_prob
        arr = np.atleast_1d(raw_prob).astype(float)
        out = self._predict(arr)
        return float(out[0]) if np.isscalar(raw_prob) else out
