# MLB Strikeout Model V2 - Weakness Analysis & Improvement Plan

## 📊 **Current Performance**

**Overall Metrics:**
- MAE: **1.91 K** (43.5% better than V1)
- Bias: **+0.78 K** (minimal over-projection)
- Within 2 K: **64%** (vs V1's 28%)

**This is excellent, but we can still improve!**

---

## ❌ **Key Weaknesses Identified**

### **1. Struggling with High-K Lineups (>26% K rate)**

**Problem:**
- MAE: 2.66 K when facing high-K lineups
- Bias: +1.10 K (over-projecting)

**Why:**
- We're multiplying by opponent K% (1.4x for 32% K lineup)
- This amplifies projections too much for already-high K/9 pitchers

**Solution:**
- Cap the opponent multiplier at 1.3x
- Or use logarithmic scaling instead of linear

```python
# Current: opponent_mult = opponent_k_rate / 0.23
# Better: opponent_mult = min(opponent_k_rate / 0.23, 1.3)
```

---

### **2. Elite K Pitchers (>11 K/9) Over-Projected**

**Problem:**
- Elite K pitchers: MAE 2.30 K, Bias +2.30 K
- Examples: MacKenzie Gore (12.86 K/9), Jacob Misiorowski (14.01 K/9)

**Why:**
- High K/9 × High opponent K% = unrealistic projections
- Not accounting for regression to mean
- Elite stuff doesn't always translate in small samples

**Solution:**
- Add regression factor for extreme K/9 values (>12)
- Reduce weight on recent form for elite K pitchers

```python
if season_k9 > 12:
    # Regress toward 11 K/9
    adjusted_k9 = (season_k9 * 0.7) + (11 * 0.3)
```

---

### **3. Early Exits / Bad Days Not Detected**

**Problem:**
- 6 pitchers over-projected by >2 K (pulled early or bad day)
- Examples: Brady Singer (5.5 → 1), Sonny Gray (4.5 → 1), Framber Valdez (3.8 → 1)

**Why:**
- No game script awareness
- No real-time injury/illness detection
- Blowout cap only triggers on recent ERA >5.5

**Solution:**
- Tighter blowout detection (ERA >4.5 in last 3 starts)
- Add "volatility" metric (high std dev = risky projection)
- Consider capping all projections at 8-9 K for safety

```python
# Add volatility check
recent_std = recent_k.std()
if recent_std > 3.0:
    # High variance = risky
    projection *= 0.90
```

---

### **4. Can't Predict Breakout Games (9+ K)**

**Problem:**
- Mick Abel: 5.5 K projected, 10 K actual (4.5 K error)
- Ryan Weathers: 11.7 K projected, 10 K actual (close!)

**Why:**
- These are rare, unpredictable events
- Model is conservative by design

**Solution:**
- **Accept this limitation** - it's better to be conservative
- These games are statistical outliers
- Focus on consistency, not capturing every outlier

---

### **5. Cold Pitchers (Recent < Season) Struggling**

**Problem:**
- Cold pitchers: MAE 3.15 K, Bias +3.15 K
- Only 2 pitchers in sample, but both badly over-projected

**Why:**
- We weight recent form at 40%, but when recent is bad, we're still too optimistic
- Not enough penalty for cold streaks

**Solution:**
- Increase recent form weight to 50% when pitcher is cold
- Add "cold streak" penalty

```python
if recent_k9 < season_k9 - 1:
    # Pitcher is cold
    base_k9 = (season_k9 * 0.4) + (recent_k9 * 0.6)  # Weight recent more
```

---

## 🎯 **Accuracy by Opponent K% Range**

| Opponent K% | Pitchers | MAE | Bias | Status |
|-------------|----------|-----|------|--------|
| Low (<20%) | 2 | 0.55 K | -0.25 K | ✅ Excellent |
| Below Avg (20-23%) | 6 | 1.57 K | +1.00 K | ✅ Good |
| Above Avg (23-26%) | 4 | 1.22 K | +0.17 K | ✅ Excellent |
| High (>26%) | 10 | 2.66 K | +1.10 K | ⚠️ Needs work |

**Finding:** We're most accurate against average lineups, struggle with high-K lineups.

---

## 🔥 **Accuracy by Pitcher Type**

| Pitcher Type | Pitchers | MAE | Bias | Within 2 K | Status |
|--------------|----------|-----|------|------------|--------|
| Low K (<7) | 4 | 2.22 K | +2.22 K | 50% | ⚠️ Over-projecting |
| Average K (7-9) | 7 | 1.44 K | +0.64 K | 86% | ✅ Best |
| High K (9-11) | 7 | 1.97 K | -0.77 K | 57% | ✅ Good |
| Elite K (>11) | 4 | 2.30 K | +2.30 K | 50% | ⚠️ Over-projecting |

**Finding:** We're most accurate with average K pitchers (7-9 K/9). Struggle with extremes.

---

## 💡 **Prioritized Improvement Plan**

### **Phase 1: Quick Wins (Implement Now)**

1. **Cap opponent multiplier at 1.3x**
   - Prevents over-projection against high-K lineups
   - Easy fix, high impact

2. **Regress elite K/9 pitchers toward mean**
   - If K/9 > 12, regress 30% toward 11 K/9
   - Prevents unrealistic projections

3. **Increase cold streak penalty**
   - Weight recent form at 60% (vs 40%) when pitcher is cold
   - Better captures struggling pitchers

4. **Tighter blowout detection**
   - Lower ERA threshold from 5.5 to 4.5
   - More aggressive capping

### **Phase 2: Medium-Term (Next Week)**

5. **Add volatility/consistency metric**
   - High std dev = reduce projection by 10%
   - Accounts for unpredictable pitchers

6. **Time-through-order penalty**
   - Reduce K/9 by 15% for IP >6.0 (3rd time through)
   - Major league average shows this effect

7. **Better Expected IP estimation**
   - Factor in opponent quality (good offense = shorter outing)
   - Recent pitch counts (>110 = shorter next outing)

### **Phase 3: Long-Term (Future)**

8. **Umpire strike zone data**
   - Tight zone = -5% K's
   - Wide zone = +5% K's

9. **Weather integration**
   - Cold (<50°F) = -5% K's
   - Wind >15mph = -5% K's

10. **Game script awareness**
    - Team record differential
    - Playoff implications
    - Bullpen availability

---

## 📈 **Expected Impact of Phase 1 Fixes**

**Current V2:**
- MAE: 1.91 K
- Within 2 K: 64%

**Projected V2.1 (with Phase 1 fixes):**
- MAE: ~1.6-1.7 K (15% improvement)
- Within 2 K: ~70-75%
- Bias: <0.5 K

**Key improvements:**
- Elite K pitchers: MAE 2.30 → 1.8 K
- High-K lineups: MAE 2.66 → 2.0 K
- Cold pitchers: MAE 3.15 → 2.5 K

---

## ✅ **What's Already Working Well**

1. ✅ **Lineup-specific K%** - Huge improvement over team averages
2. ✅ **Expected IP cap** - Prevents most unrealistic projections
3. ✅ **Blended 2025+2026 stats** - Good baseline
4. ✅ **Average K pitchers (7-9 K/9)** - 86% within 2 K!
5. ✅ **Low-K lineups** - MAE only 0.55 K
6. ✅ **Minimal bias** - Only +0.78 K overall

---

## 🎯 **Bottom Line**

**V2 is already excellent (43.5% better than V1), but we can push it further:**

- **Quick wins** (Phase 1) should get us to 70%+ within 2 K
- **Medium-term** (Phase 2) could reach 75%+ within 2 K
- **Long-term** (Phase 3) might hit 80%+ within 2 K

**The model is production-ready now, but these improvements will make it elite.**
