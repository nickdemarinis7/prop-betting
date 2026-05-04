"""
MLB home run probability calibrator.

Thin wrapper around core.prob_calibrator with HR-specific column names.
HR validation files store a single probability ('projection') and binary
outcome ('hit'), so only one (prob, hit) pair per row.
"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from core.prob_calibrator import ProbabilityCalibrator as _Generic

CALIBRATOR_PATH = os.path.join(_THIS_DIR, 'prob_calibrator.joblib')
VALIDATION_GLOB = os.path.join(_THIS_DIR, 'validation_results_*.csv')


class ProbabilityCalibrator(_Generic):
    def __init__(self):
        super().__init__(
            model_path=CALIBRATOR_PATH,
            validation_glob=VALIDATION_GLOB,
            prob_cols=['projection'],
            hit_cols=['hit'],
            prob_scale=1.0,
            min_samples=100,
            # isotonic: 1002 samples is enough for a flexible fit and
            # captures the S-curve under-projection at extremes (<5%, >20%).
            method='isotonic',
        )


if __name__ == "__main__":
    print("=" * 60)
    print("📐 MLB Home Run Probability Calibrator - Training")
    print("=" * 60)
    cal = ProbabilityCalibrator()
    cal.train()
