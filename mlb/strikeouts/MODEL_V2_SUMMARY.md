# MLB Strikeout Model V2 - Improvements Summary

## 🎯 **What We Built**

### **New Features:**

1. **Expected IP Calculator** (`pitcher_context.py`)
   - Calculates pitcher's average IP from last 10 starts
   - Caps projections realistically (no more 10 K projections for 5 IP pitchers)
   - Accounts for pitcher stamina and usage patterns

2. **Day/Night Splits** (`pitcher_context.py`)
   - Uses pitcher's actual K/9 in day vs night games
   - Falls back to season average if insufficient data
   - More accurate than blanket multipliers

3. **Short Rest Detection** (`pitcher_context.py`)
   - Identifies pitchers on <4 days rest
   - Applies 10% penalty to projection
   - Reduces expected IP

4. **Workload Monitoring** (`pitcher_context.py`)
   - Tracks recent pitch counts
   - Flags high workload (>110 pitches in last 2 starts)
   - Helps predict early exits

5. **Fangraphs Lineup Scraper** (`fangraphs_lineups.py`)
   - Gets projected starting lineups
   - Falls back when actual lineup not available
   - More accurate than team averages

### **Simplified Prediction Formula:**

```python
# 1. Base K/9 (weighted: 60% season, 40% recent)
base_k9 = (season_k9 * 0.6) + (recent_k9 * 0.4)

# 2. Adjust for day/night if data available
adjusted_k9 = day_k9 if is_day_game else night_k9

# 3. Calculate expected IP from recent starts
expected_ip = avg_ip_last_10_starts
expected_ip = min(expected_ip, 6.5)  # Cap at reasonable max

# 4. Base projection
base_projection = (adjusted_k9 / 9) * expected_ip

# 5. Opponent adjustment (using actual lineup K%)
opponent_mult = lineup_k_rate / league_avg_k_rate

# 6. Other adjustments
home_mult = 1.05 if is_home else 0.95
rest_mult = 0.90 if short_rest else 1.0

# 7. Final projection
final = base_projection * opponent_mult * home_mult * rest_mult

# 8. Blowout cap
if blowout_risk > 0.5:
    final = min(final, 5.5)
```

---

## 🔧 **Key Improvements Over V1:**

### **What We Removed:**
- ❌ Multiple overlapping recent form metrics
- ❌ Complex weighted averages
- ❌ Excessive opponent stats beyond K%
- ❌ Artificial bias/deflation

### **What We Added:**
- ✅ **Expected IP cap** - Prevents unrealistic projections
- ✅ **Day/night splits** - Uses actual pitcher performance
- ✅ **Short rest penalty** - Accounts for fatigue
- ✅ **Projected lineups** - More accurate opponent K%
- ✅ **Workload tracking** - Predicts early exits

---

## 📊 **Expected Impact:**

### **Problem in V1:**
- Over-projecting by 2.8 K on average
- Only 28% of predictions within 2 K
- Ladder probabilities way off (32% hit vs 84% projected)

### **How V2 Fixes This:**

1. **Expected IP Cap**
   - V1: Projected 10 K for pitchers who only go 5 IP
   - V2: Caps at realistic IP × K/9

2. **No Artificial Deflation**
   - V1: Would have needed 15-20% blanket reduction
   - V2: Natural cap from IP limits

3. **Better Context**
   - V1: Used generic multipliers
   - V2: Uses pitcher's actual day/night performance

4. **Short Rest**
   - V1: Ignored rest days
   - V2: 10% penalty for short rest

---

## 🧪 **Testing Plan:**

1. **Run V2 on yesterday's games** (April 14, 2024)
2. **Compare predictions:**
   - V1 projections
   - V2 projections
   - Actual results
3. **Calculate improvements:**
   - MAE (Mean Absolute Error)
   - Bias (over/under projection)
   - Hit rates at each line

---

## 📁 **New Files Created:**

1. `/mlb/shared/features/pitcher_context.py`
   - Expected IP calculator
   - Day/night splits
   - Short rest detection
   - Workload monitoring

2. `/mlb/shared/scrapers/fangraphs_lineups.py`
   - Projected lineup scraper
   - Team abbreviation mapping

3. `/mlb/strikeouts/predict_v2.py`
   - Simplified prediction script
   - Uses all new features
   - Cleaner, more maintainable code

---

## 🚀 **Next Steps:**

1. ✅ Fix any remaining bugs in predict_v2.py
2. ⏳ Run on yesterday's games
3. ⏳ Compare V1 vs V2 vs Actual
4. ⏳ Validate improvements
5. ⏳ If successful, replace V1 with V2

---

## 💡 **Future Enhancements (Phase 2):**

- Weather data collection & validation
- Actual lineup updates (2 hours before game)
- Umpire strike zone data
- Batter vs pitcher history (if predictive)
- Pitch arsenal analysis
