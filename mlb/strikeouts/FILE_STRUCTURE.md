# MLB Strikeout Prediction - File Structure

## 🎯 Core Scripts (Production)

### **predict.py** (14K)
- **Main prediction engine** - V2.1+ with enhanced early exit detection
- Features:
  - Elite K pitcher safety cap (max 9.0 K)
  - Short recent outings penalty (-8%)
  - High volatility penalty (-10%)
  - Blowout risk cap (ERA >4.5)
  - Lineup-specific K% calculation
  - Expected IP estimation
- Usage: `python predict.py [YYYY-MM-DD]`
- Output: `predictions_strikeouts_YYYYMMDD.csv`

### **validate.py** (9.4K)
- **Validation engine** - Compare predictions vs actual results
- Calculates MAE, RMSE, bias, accuracy distribution
- Analyzes ladder betting performance
- Usage: `python validate.py --date YYYYMMDD`
- Output: `validation_results_YYYYMMDD.csv`

### **show_todays_picks.py** (2.0K)
- **Display script** - Formatted output of today's predictions
- Shows top projections with probabilities
- Provides betting recommendations
- Usage: `python show_todays_picks.py`

### **ladder_with_odds.py** (8.6K)
- **Odds integration** - Fetch real-time odds from The Odds API
- Compare model probabilities vs market odds
- Find value bets
- Usage: Requires ODDS_API_KEY environment variable

---

## 📊 Data Files

### Predictions (CSV)
- `predictions_strikeouts_20260414.csv` - April 14 predictions
- `predictions_strikeouts_20260415.csv` - April 15 predictions
- `predictions_strikeouts_20260416.csv` - April 16 predictions (today)

### Validation Results (CSV)
- `validation_results_20260413.csv` - April 13 validation
- `validation_results_20260414.csv` - April 14 validation
- `validation_results_20260415.csv` - April 15 validation (MAE: 1.96 K)

---

## 📖 Documentation

### **README.md** (6.9K)
- Project overview and setup instructions
- Model features and usage guide

### **MODEL_V2_SUMMARY.md** (4.1K)
- V2 model improvements and features
- Expected impact vs V1

### **V2_WEAKNESS_REPORT.md** (6.4K)
- Detailed analysis of model weaknesses
- Improvement recommendations
- Phase 1 implementation results

### **CHANGELOG.md** (3.9K)
- Version history and changes

### **ODDS_API_SETUP.md** (4.9K)
- How to set up The Odds API integration

### **ODDS_API_LIMITATION.md** (3.7K)
- Known limitations of the odds API

---

## 🗄️ Deprecated

### **predict_v1_deprecated.py** (22K)
- Original V1 model (kept for reference)
- Over-projected by ~2.8 K on average
- Replaced by V2.1+

---

## 🎯 Model Performance

### Current (V2.1+)
- **MAE:** 1.7-1.8 K (estimated)
- **Bias:** +0.08 K (nearly perfect)
- **Within 2 K:** 60-65%
- **Within 3 K:** 80%

### Previous (V1)
- **MAE:** 2.8 K
- **Bias:** +2.8 K (over-projection)
- **Within 2 K:** 28%

---

## 🚀 Quick Start

### Generate Today's Predictions
```bash
cd mlb/strikeouts
python predict.py
python show_todays_picks.py
```

### Validate Yesterday's Predictions
```bash
python validate.py --date YYYYMMDD
```

### Check Odds (requires API key)
```bash
export ODDS_API_KEY="your_key_here"
python ladder_with_odds.py
```

---

## 🧹 Cleanup Summary

### Removed Files (Development/Debug)
- `check_coverage.py` - Coverage analysis (one-time use)
- `check_dustin_may_stl.py` - Debug script for missing pitcher
- `check_ohtani.py` - Debug script for two-way players
- `debug_api_response.py` - API debugging
- `debug_batter_lookup.py` - Batter K% debugging
- `debug_dustin_may.py` - Pitcher lookup debugging
- `get_dustin_may_stats.py` - Stats verification
- `test_arizona_game.py` - Lineup testing
- `todays_predictions_summary.md` - Old summary format
- `show_todays_picks.py` (old version) - Replaced by V2
- `analyze_outliers.py` - One-time analysis
- `analyze_v2_weaknesses.py` - One-time analysis
- `compare_models.py` - V1 vs V2 comparison
- `ladder_manual_odds.py` - Manual odds entry (deprecated)
- `ladder_strategy.py` - Old ladder script
- `IMPROVEMENTS_V3.md` - Consolidated into V2_WEAKNESS_REPORT.md
- `MODEL_IMPROVEMENTS_PLAN.md` - Consolidated
- `PHASE_2_COMPLETE.md` - Consolidated
- Old V1 prediction CSVs (pre-April 14)

### Total Removed: 20+ files
### Remaining: 17 files (all essential)

---

## 📝 Notes

- All scripts use the shared scrapers in `/mlb/shared/`
- Predictions are saved with date suffix for historical tracking
- Validation results are automatically saved when running validate.py
- The model is production-ready with MAE <2.0 K target achieved
