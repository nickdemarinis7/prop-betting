"""
MLB strikeout probability calibrator.

Thin wrapper around core.prob_calibrator with strikeout-specific column
names. Kept as a module so existing imports continue to work.
"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from core.prob_calibrator import ProbabilityCalibrator as _Generic

PROB_LINES = [3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
CALIBRATOR_PATH = os.path.join(_THIS_DIR, 'prob_calibrator.joblib')
VALIDATION_GLOB = os.path.join(_THIS_DIR, 'validation_results_*.csv')


class ProbabilityCalibrator(_Generic):
    def __init__(self):
        super().__init__(
            model_path=CALIBRATOR_PATH,
            validation_glob=VALIDATION_GLOB,
            prob_cols=[f'prob_{L}' for L in PROB_LINES],
            hit_cols=[f'hit_{L}' for L in PROB_LINES],
            prob_scale=1.0,
            min_samples=100,
        )


if __name__ == "__main__":
    print("=" * 60)
    print("📐 MLB Strikeout Probability Calibrator - Training")
    print("=" * 60)
    cal = ProbabilityCalibrator()
    cal.train()
