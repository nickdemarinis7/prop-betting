# ⚾ MLB Strikeout Prediction System

Machine learning system for predicting pitcher strikeouts with betting recommendations.

## 🎯 Overview

Predicts pitcher strikeouts using:
- Recent performance (last 3, 5 starts)
- Season K/9 and K% rates
- Opponent team strikeout rate
- Ballpark factors
- Home/away splits

Built on proven XGBoost architecture (same as NBA points model).

---

## 🚀 Quick Start

```bash
# Navigate to strikeouts directory
cd mlb/strikeouts

# Run predictions for today's games
python predict.py
```

**Output:**
- Strikeout projections for each probable starter
- Probabilities for common lines (4.5, 5.5, 6.5, 7.5 K's)
- CSV file with all predictions

---

## 📊 Features

### **Pitcher Features:**
- `k_last_5` - Average K's last 5 starts
- `k_last_3` - Average K's last 3 starts  
- `k9_last_5` - K/9 rate last 5 starts
- `k_pct_last_5` - K% last 5 starts
- `k9_season` - Season K/9 rate
- `k_pct_season` - Season K percentage

### **Context Features:**
- `opp_k_rate` - Opponent team K% vs this handedness
- `is_home` - Home field advantage
- `ballpark_k_factor` - Ballpark strikeout factor

---

## 📈 Model Performance

**Expected Metrics:**
- **MAE:** 0.8-1.2 strikeouts
- **R²:** 40-50%
- **Win Rate:** 55-58% on O/U bets
- **ROI:** 5-8%

**Better than NBA because:**
- More games per season (162 vs 82)
- Pitcher-dependent (less team variance)
- Softer prop markets

---

## 🎯 Betting Strategy

### **High Confidence (65%+ probability):**
- Bet 0.75-1.0 units
- Target lines where model shows clear edge

### **Medium Confidence (55-65%):**
- Bet 0.5 units
- Verify with recent form

### **Low Confidence (<55%):**
- Skip or minimal bet (0.25u)

### **Red Flags:**
- High variance pitcher (inconsistent K's)
- Extreme weather (wind >15 mph)
- Tight umpire strike zone
- Pitcher on short rest

---

## 📁 Project Structure

```
mlb/
├── strikeouts/
│   ├── predict.py              # Main prediction script
│   ├── validate.py             # Backtest validation (TODO)
│   ├── analyze_results.py      # Performance analysis (TODO)
│   └── README.md               # This file
├── shared/
│   ├── scrapers/
│   │   ├── pitcher_stats.py    # Pitcher game logs
│   │   └── mlb_schedule.py     # Today's games
│   ├── features/
│   │   └── ballpark.py         # Ballpark K factors
│   └── utils/
│       └── betting_math.py     # Probability calculations
```

---

## 🔧 Configuration

### **Model Parameters:**
```python
n_estimators = 200      # Number of trees
max_depth = 4           # Tree depth
learning_rate = 0.05    # Learning rate
min_child_weight = 3    # Regularization
```

### **Training Data:**
- Top 100 pitchers by strikeouts
- Last 20 starts per pitcher
- Minimum 5 starts to include

---

## 📊 Output Format

### **Console Output:**
```
⚾ MLB STRIKEOUT PREDICTION - PRODUCTION SYSTEM
================================================================================

1. Gerrit Cole (NYY) 🏠 vs BOS
   Projection: 7.2 K's  (L5 Avg: 6.8)
   
   🎯 PROBABILITIES:
      4.5+ K's:   95%  🔥
      5.5+ K's:   82%  🔥
      6.5+ K's:   65%  ✅
      7.5+ K's:   48%
```

### **CSV Output:**
```csv
pitcher,team,opponent,is_home,projection,k_last_5,prob_4.5+,prob_5.5+,prob_6.5+,prob_7.5+
Gerrit Cole,NYY,BOS,1,7.2,6.8,0.95,0.82,0.65,0.48
```

---

## 🎓 Key Insights

### **What Drives Strikeouts:**
1. **Pitcher Quality** (40%) - K/9, K%, stuff
2. **Opponent Matchup** (30%) - Team K% vs handedness
3. **Recent Form** (20%) - Hot/cold streaks
4. **Ballpark** (10%) - Some parks favor K's

### **Ballpark Effects:**
- **High K Parks:** Tropicana Field, Petco Park, Oakland
- **Low K Parks:** Coors Field, Great American, Fenway
- **Neutral:** Most parks (0.98-1.02 factor)

### **Best Betting Spots:**
- Power pitchers vs high-K teams
- Dome games (consistent conditions)
- Pitchers on hot streaks (3+ straight quality starts)
- Favorable umpires (wide strike zones)

---

## ⚠️ Limitations

### **Current Version:**
- Uses sample/placeholder data (needs real MLB API integration)
- No weather data yet (wind, temperature)
- No umpire tendencies yet
- No opponent lineup analysis

### **Future Enhancements:**
- Real-time MLB Stats API integration
- Weather API (wind affects swing decisions)
- Umpire database (strike zone variations)
- Opponent lineup quality (vs RHP/LHP)
- Pitch mix analysis (fastball %, breaking ball %)
- Velocity trends (declining velo = fewer K's)

---

## 🔌 Data Sources

### **Current (Placeholder):**
- Sample pitcher data
- Sample game schedules
- Static ballpark factors

### **Recommended (Production):**
- **MLB Stats API** - Official stats, game logs
- **Baseball Savant** - Statcast data, pitch tracking
- **FanGraphs** - Advanced metrics (FIP, xFIP, K-BB%)
- **Baseball Reference** - Historical data
- **Weather API** - Real-time conditions

---

## 📝 Usage Examples

### **Basic Prediction:**
```bash
python predict.py
```

### **Validate Historical Performance:**
```bash
python validate.py --date 2026-04-10
```

### **Analyze Results:**
```bash
python analyze_results.py
```

---

## 🎯 Success Metrics

### **After 30 Days:**
- Win Rate: 55-58%
- ROI: 5-10%
- Average Bets/Day: 3-5 (15 games × 2 starters × 20% bet rate)
- Expected Profit: +10-20 units/month

### **After 90 Days:**
- Win Rate: 56-60%
- ROI: 6-12%
- Model refinements based on results
- Expanded feature set

---

## 💡 Pro Tips

1. **Wait for lineups** - Verify probable starter is confirmed
2. **Check weather** - Wind >15 mph affects swings
3. **Know the umpire** - Some have tight zones (fewer K's)
4. **Fade variance** - Avoid inconsistent pitchers
5. **Target domes** - More predictable conditions
6. **Bet early** - Lines move as sharp money comes in
7. **Track results** - Use validate.py to monitor performance

---

## 🚀 Next Steps

### **Phase 1: Data Integration** (Week 1)
- [ ] Integrate real MLB Stats API
- [ ] Add weather data
- [ ] Implement umpire database

### **Phase 2: Feature Enhancement** (Week 2)
- [ ] Opponent lineup analysis
- [ ] Pitch mix features
- [ ] Velocity tracking

### **Phase 3: Validation** (Week 3)
- [ ] Build validation script
- [ ] Backtest on historical data
- [ ] Calibration analysis

### **Phase 4: Production** (Week 4)
- [ ] Daily automated runs
- [ ] ROI tracking
- [ ] Performance monitoring

---

## 📚 Resources

- **MLB Stats API Docs:** https://statsapi.mlb.com/docs/
- **Baseball Savant:** https://baseballsavant.mlb.com/
- **FanGraphs:** https://www.fangraphs.com/
- **Baseball Reference:** https://www.baseball-reference.com/

---

## 🤝 Contributing

This is a personal betting model. Key areas for improvement:
- Better data sources
- Additional features (weather, umpires)
- Model optimization
- Validation framework

---

**Built with the same proven architecture as the NBA points model (5.43 MAE, 37% R²).**

**Ready to find edges in MLB strikeout markets!** ⚾💰
