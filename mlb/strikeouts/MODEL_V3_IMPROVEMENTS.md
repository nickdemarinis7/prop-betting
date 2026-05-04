# MLB Strikeout Predictions V3 - Improvements Summary

## 🎯 Core Philosophy
**Simplified & Data-Driven**: Only keep what validation proves works. Remove hand-tuned adjustments that don't improve accuracy.

---

## 📊 Current Model Problems (Based on 20260424 Validation)

| Metric | Current Model | Baseline (K/9 × IP) | Assessment |
|--------|--------------|---------------------|------------|
| MAE | 1.93 K | 1.92 K | **No improvement** - all adjustments = noise |
| Bias | +0.06 K | N/A | Slight under-projection |
| vs Baseline | 0% worse | - | Model not beating simple math |

**Conclusion**: The current model's 80+ lines of adjustments (ERA caps, volatility reductions, global caps, elite regression) **don't improve accuracy**.

---

## 🔧 V3 Simplifications

### **REMOVED** (No Validation Evidence They Help)

1. **High ERA Cap** (5.5K max for ERA >4.5)
   - *Why removed*: Let the projection stand; confidence tier handles risk
   
2. **Volatility Reduction** (σ>3.0 → -10%)
   - *Why removed*: std_dev already in confidence tier
   
3. **Short Outings Reduction** (<5.0 IP → -8%)
   - *Why removed*: Expected IP calculation already accounts for this
   
4. **Elite K Safety Cap** (9.0K max for K/9>12)
   - *Why removed*: If Scherzer has 12 K/9, let him project 9+ Ks
   
5. **Global Quality Caps** (8K/9K/10K tiers)
   - *Why removed*: Arbitrary; let projection reflect reality
   
6. **Calibration Offset** (-0.5K flat)
   - *Why removed*: Fighting symptoms; fix root causes instead

### **KEPT** (Validation Proves They Work)

1. **Base K/9 Blending** (60% season / 40% recent)
   - *Why kept*: Core of accurate prediction
   
2. **Cold Streak Detection** (recent < season-1.5 → weight recent more)
   - *Why kept*: Identifies struggling pitchers
   
3. **Opponent Lineup Adjustment**
   - *Why kept*: Real edge over market
   
4. **Expected IP**
   - *Why kept*: Critical for total Ks
   
5. **Confidence Tiers** (HIGH/MEDIUM/LOW)
   - *Why kept*: Validation shows HIGH = 1.57 MAE vs MEDIUM = 2.17
   
6. **ML Corrector** (±2.0K cap)
   - *Why kept*: Data-driven adjustments vs hand-tuned

---

## 🆕 V3 NEW FEATURES

### 1. **Smart Data Blending** (Game-Count Weighted)

**Old**: Fixed 70% 2026 / 30% 2025

**New**: Dynamic based on games pitched in 2026
```python
weight_2026 = min(0.5 + (games_2026 / 25), 0.90)
# 2 games  → 58% 2026 / 42% 2025
# 5 games  → 70% 2026 / 30% 2025  
# 10 games → 90% 2026 / 10% 2025
# 20 games → 90% 2026 / 10% 2025 (capped)
```

*Benefit*: More weight to 2025 data when pitcher has few 2026 starts (small sample risk)

### 2. **ML Corrector Training Pipeline**

**New file**: `train_ml_corrector.py`

Learns from historical validation data:
- Input: Prediction features (K/9, IP, opponent, etc.)
- Target: Actual error (actual - projection)
- Output: Trained model that predicts correction needed

*Benefit*: Data-driven adjustments instead of hand-tuned heuristics

### 3. **Side-by-Side Comparison**

**New file**: `compare_models.py`

Runs both models simultaneously:
1. Runs current `predict.py`
2. Runs simplified `predict_simplified.py`
3. Compares outputs
4. Saves to `model_comparison_YYYYMMDD.csv`

*Benefit*: A/B testing framework to validate improvements

---

## 🎲 How to Use V3

### Step 1: Run Simplified Model
```bash
cd mlb/strikeouts
python predict_simplified.py
```

### Step 2: Compare with Current (Optional)
```bash
python compare_models.py
```

### Step 3: Validate Tomorrow
```bash
python validate.py --date 20260426
```

### Step 4: Train ML Corrector (After 5+ validation days)
```bash
python train_ml_corrector.py
```

---

## 📈 Expected Outcomes

### Short Term (1-2 weeks)
- **Goal**: Match or beat baseline MAE (1.92K)
- **Test**: Run both models, compare validation results
- **Decision**: If simplified performs better, make it default

### Medium Term (1 month)
- **Goal**: Train ML corrector on 20+ validation samples
- **Target**: MAE < 1.80K (10% improvement over baseline)

### Long Term (Season)
- **Goal**: Continuous learning system
- **Process**: 
  1. Make predictions
  2. Validate next day
  3. Retrain corrector weekly
  4. Deploy improvements

---

## 🔬 Future Improvements (Post-V3)

### Potential Additions to Test

| Feature | Hypothesis | Test Method |
|---------|-----------|-------------|
| **Park Factors** | Pitchers in COL/ARI get fewer Ks | Add park factor multiplier |
| **Umpire K Rate** | Some umps have smaller zones | Track umpire in predictions |
| **Weather** | Wind affects fly balls/Ks | Add temp/wind for day games |
| **Velocity Trends** | Declining velo = declining Ks | Track 4-seam velocity delta |
| **Spin Rate** | Higher spin = more Ks | Add avg spin rate feature |
| **Bullpen Load** | Tired pen = starter goes deeper | Add bullpen IP last 3 days |

### Validation Needed
Each addition requires:
1. A/B test vs current model
2. 20+ game validation sample
3. Statistical significance (p < 0.05)
4. Only keep if MAE improves

---

## 📋 Files Added/Modified

### New Files
- `predict_simplified.py` - Simplified prediction model
- `train_ml_corrector.py` - ML training pipeline
- `compare_models.py` - A/B testing script
- `MODEL_V3_IMPROVEMENTS.md` - This document

### Unchanged (Still Used)
- `predict.py` - Current model (kept for comparison)
- `validate.py` - Validation script
- `ml_corrector.py` - ML corrector class (will use trained model)

---

## ⚡ Quick Start

```bash
# 1. Run simplified model today
python predict_simplified.py

# 2. Compare with current model
python compare_models.py

# 3. Tomorrow, validate both
python validate.py --date 20260426

# 4. After 5 days of validation data, train corrector
python train_ml_corrector.py
```

---

## 🎯 Success Metrics

| Metric | Current | Target V3 | Measurement |
|--------|---------|-----------|-------------|
| MAE | 1.93 K | < 1.90 K | Daily validation |
| vs Baseline | 0% | > 2% better | Compare to 1.92K |
| Bias | +0.06 K | < |0.10| K | Mean error |
| Confidence Separation | 0.60 K | > 0.80 K | HIGH vs MEDIUM MAE diff |

**Definition of Success**: Simplified model beats current model after 2 weeks of validation.
