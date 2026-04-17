# 🏀 Sports Betting Prediction System

A professional, scalable ML-powered system for sports prop betting with ladder betting strategies.

## 🎯 Current Features

### NBA
- ✅ **Assists Predictions** - ML model with 13.20% ROI
- 🚧 **Points Predictions** - In development

### Capabilities
- Machine learning projections (XGBoost, Random Forest, Gradient Boosting)
- Ladder betting probabilities (multiple thresholds)
- Tier-based confidence classification
- Unit sizing recommendations
- Red flag warnings (momentum, variance, blowouts)
- Injury filtering with accent-normalized name matching
- Historical validation and performance tracking
- Clean CSV exports for analysis

---

## 📁 Project Structure

```
windsurf-project/
├── run.py                        # Main entry point
├── config.py                     # Configuration
├── requirements.txt              # Dependencies
│
├── nba/                          # NBA-specific code
│   ├── assists/                  # Assists prop betting
│   │   ├── predict.py           # Prediction script
│   │   ├── validate.py          # Validation script
│   │   └── models/              # Trained models
│   │
│   ├── points/                   # Points prop betting
│   │   ├── predict.py           # Prediction script
│   │   ├── validate.py          # Validation script
│   │   └── models/              # Trained models
│   │
│   ├── shared/                   # Shared NBA utilities
│   │   ├── scrapers/            # Data scrapers
│   │   ├── features/            # Feature engineering
│   │   ├── models/              # ML models
│   │   └── utils/               # Utilities
│   │
│   └── data/                     # Data storage
│       ├── predictions/
│       └── validations/
│
├── core/                         # Cross-sport utilities
│   ├── ladder_betting.py        # Generic ladder logic
│   ├── unit_sizing.py           # Unit sizing
│   └── validation.py            # Validation framework
│
└── src/                          # Legacy code (deprecated)
```

---

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Usage

#### NBA Assists Predictions
```bash
# Generate today's predictions
python run.py nba assists predict

# Validate yesterday's picks
python run.py nba assists validate --date 20260408
```

#### NBA Points Predictions (Coming Soon)
```bash
# Generate today's predictions
python run.py nba points predict

# Validate yesterday's picks
python run.py nba points validate --date 20260408
```

---

## 📊 Output

### Console Output
- **Tier 1 (Safest Bets)** - High confidence, low risk
- **Tier 2 (Good Value)** - Solid opportunities with minor concerns
- **Tier 3 (Higher Risk)** - Negative momentum or high variance

For each pick:
- Projection vs recent averages
- Ladder probabilities (5+, 7+, 10+ assists)
- Minimum odds needed for +EV
- Historical hit rates
- Recommended unit sizes
- Red flags and warnings

### CSV Export
Located in `nba/data/predictions/{prop}/`

Columns include:
- Player, Team, Opponent, Home/Away
- Projection, L10 average, Ratio
- Ladder probabilities (5+%, 7+%, 10+%)
- Minimum odds for +EV
- Recommended units
- Tier classification
- Red flags
- Context (matchup, pace, injuries)

---

## 🎯 Ladder Betting Strategy

### What is Ladder Betting?
Place multiple bets on the same player at different thresholds:
- 5+ assists @ -200 (1.25u)
- 7+ assists @ +150 (0.50u)
- 10+ assists @ +1200 (0.25u)

**Total risk:** 2.0 units  
**Profit if all hit:** ~4.5 units  
**Profit if only 5+ hits:** Still profitable!

### Unit Sizing
- **1.5u** - Very high confidence (80%+ probability)
- **1.25u** - High confidence (70-79%)
- **1.0u** - Good confidence (60-69%)
- **0.75u** - Moderate confidence (50-59%)
- **0.5u** - Lower confidence (40-49%)
- **0.25u** - Small bet (25-39%)
- **0.1u** - Lottery ticket (<25%)

Adjusted for:
- **Tier** (1 = bonus, 3 = penalty)
- **Red flags** (reduce units)

---

## 📈 Performance

### NBA Assists (April 8, 2026)
- **ROI:** 13.20%
- **Profit:** +0.9 units
- **Record:** 5-7 (41.7% win rate)
- **Result:** Profitable despite sub-50% win rate (ladder betting works!)

---

## 🛠️ Technical Details

### Data Sources
- NBA.com Stats API (player stats, game logs)
- ESPN (injury reports)
- Custom scrapers for pace, defense, usage

### ML Models
- XGBoost
- Random Forest
- Gradient Boosting Ensemble
- Feature engineering: 50+ features per prediction

### Key Features
- Recent form (L3, L5, L10 weighted)
- Opponent defense rating
- Pace matchup
- Usage boost (injured teammates)
- Momentum indicators
- Consistency metrics
- Blowout risk assessment

---

## 🔧 Configuration

Edit `config.py` to customize:
- Model parameters
- Threshold values
- Unit sizing rules
- Data sources

---

## 📝 Validation

Track performance with validation scripts:

```bash
# Validate specific date
python run.py nba assists validate --date 20260408

# Results saved to:
# nba/data/validations/assists/validation_results_YYYYMMDD.csv
```

Validation includes:
- Hit rates vs expected
- Mean absolute error
- Individual player results
- Ladder betting outcomes

---

## 🚧 Roadmap

### Short Term
- [ ] Complete NBA Points model
- [ ] Add rebounds predictions
- [ ] Add 3-pointers predictions
- [ ] Unified dashboard

### Long Term
- [ ] NFL props (passing yards, TDs, receptions)
- [ ] MLB props (hits, strikeouts, home runs)
- [ ] Live betting integration
- [ ] Automated bet placement

---

## ⚠️ Disclaimer

This system is for educational and research purposes. Sports betting involves risk. Past performance does not guarantee future results. Bet responsibly.

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🤝 Contributing

This is a personal project, but suggestions and improvements are welcome!

---

**Built with ❤️ for profitable sports betting**
