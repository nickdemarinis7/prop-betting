"""
NBA assists probability calibrator.

Validation files store probabilities as percentages (0-100), so we use
prob_scale=0.01 to normalize before fitting isotonic regression.

Playoff-only filter: excludes early-season dates (before Apr 12) that
had extreme outliers and different game dynamics.
"""

import os
import sys
import re

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from core.prob_calibrator import ProbabilityCalibrator as _Generic

# Validation only stored 5+ / 7+ / 10+ historically
VAL_THRESHOLDS = [5, 7, 10]

CALIBRATOR_PATH = os.path.join(_THIS_DIR, 'prob_calibrator.joblib')
VALIDATION_GLOB = os.path.join(_THIS_DIR, 'validation_results_*.csv')

# Playoff-only: exclude dates before Apr 12 (regular-season outliers).
PLAYOFF_START = '20260412'

def _playoff_only(filepath):
    m = re.search(r'(\d{8})', os.path.basename(filepath))
    if not m:
        return True
    return m.group(1) >= PLAYOFF_START


class ProbabilityCalibrator(_Generic):
    def __init__(self):
        super().__init__(
            model_path=CALIBRATOR_PATH,
            validation_glob=VALIDATION_GLOB,
            prob_cols=[f'Prob_{T}+' for T in VAL_THRESHOLDS],
            hit_cols=[f'Hit_{T}+' for T in VAL_THRESHOLDS],
            prob_scale=0.01,
            min_samples=50,
            method='isotonic',
            file_filter=_playoff_only,
        )


if __name__ == "__main__":
    print("=" * 60)
    print("📐 NBA Assists Probability Calibrator - Training")
    print("=" * 60)
    cal = ProbabilityCalibrator()
    cal.train()
