# NBA Assists Prediction System

## 📁 Directory Structure

```
nba/assists/
├── predict.py                          # Main assists prediction script
├── predict_production.py               # Production assists prediction (legacy)
├── validate.py                         # Validation script
├── validate_picks.py                   # Picks validation (legacy)
├── predictions_production_*.csv        # Historical prediction outputs
└── models/                             # Saved models directory
```

## 🎯 Scripts Overview

### **predict.py** (Primary Script)
- Modern assists prediction system
- Uses shared NBA modules
- Aligned with points prediction architecture
- **Recommended for use**

### **predict_production.py** (Legacy)
- Original production assists script
- Recently moved from root directory
- Uses shared modules (updated imports)
- Consider migrating to `predict.py`

### **validate.py** (Primary Validation)
- Validates prediction accuracy
- Compares predictions vs actual results
- **Recommended for use**

### **validate_picks.py** (Legacy)
- Original validation script
- Recently moved from root directory
- Uses shared modules (updated imports)

---

## 🚀 Usage

### **Generate Predictions:**
```bash
cd nba/assists
python predict.py
```

### **Validate Predictions:**
```bash
cd nba/assists
python validate.py --date 20260412
```

---

## 📊 Output Files

### **CSV Format:**
- `predictions_production_YYYYMMDD.csv`
- Contains: Player, Team, Opponent, Projection, Probabilities, etc.

### **Columns:**
- Player info (name, team, opponent)
- Projection (assists prediction)
- Ladder probabilities (3+, 5+, 7+, 9+)
- Confidence metrics
- Context (home/away, defense, pace)

---

## 🔧 Recent Changes (April 13, 2026)

### **File Organization:**
- ✅ Moved `predict_production.py` from root → `nba/assists/`
- ✅ Moved `validate_picks.py` from root → `nba/assists/`
- ✅ Moved prediction CSVs from root → `nba/assists/`
- ✅ Updated all import paths to use shared modules

### **Import Updates:**
**Before:**
```python
sys.path.append('src')
from scraper_gamelog import GameLogScraper
```

**After:**
```python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from shared.scrapers.gamelog import GameLogScraper
```

---

## 📈 Model Features

### **Input Features:**
- Recent performance (L3, L5, L10 games)
- Assist trends and consistency
- Home/away splits
- Opponent defense strength
- Pace factors
- Usage rate
- Injury impacts

### **Output:**
- Assists projection
- Probability ladder (3+, 5+, 7+, 9+, etc.)
- Confidence scores
- Betting recommendations

---

## 🎯 Next Steps

### **Recommended Improvements:**
1. **Consolidate Scripts**
   - Merge `predict_production.py` features into `predict.py`
   - Merge `validate_picks.py` features into `validate.py`
   - Remove legacy scripts

2. **Align with Points Model**
   - Add expanded ladder (2+, 4+, 6+, 8+, 10+, 12+)
   - Add variance/consistency display
   - Add model diagnostics
   - Add feature importance

3. **Enhance Validation**
   - Track ROI by threshold
   - Add calibration analysis
   - Compare vs betting lines

---

## 📝 Notes

- Both `predict.py` and `predict_production.py` are functional
- Import paths have been updated to use shared modules
- Legacy scripts maintained for reference
- Consider consolidating to single prediction script

---

**For questions or issues, refer to the main project README or shared module documentation.**
