# Setting Up Sportsbook Odds Integration

## 🎯 Overview

We use **The Odds API** to fetch real-time sportsbook odds for MLB pitcher strikeouts. This allows us to:
- Compare our model's probabilities to sportsbook odds
- Calculate expected value (EV) for each bet
- Find the best value bets automatically
- Get odds from multiple sportsbooks (DraftKings, FanDuel, BetMGM, etc.)

---

## 📝 Step 1: Get a Free API Key

1. Go to: **https://the-odds-api.com/**
2. Click "Get API Key" or "Sign Up"
3. Create a free account
4. Copy your API key

**Free Tier Limits:**
- 500 requests per month
- ~16 requests per day
- Perfect for daily strikeout betting

---

## 🔧 Step 2: Set Up API Key

### Option A: Environment Variable (Recommended)

**Mac/Linux:**
```bash
# Add to ~/.zshrc or ~/.bashrc
export ODDS_API_KEY='your_api_key_here'

# Reload shell
source ~/.zshrc
```

**Windows:**
```cmd
setx ODDS_API_KEY "your_api_key_here"
```

### Option B: Pass Directly to Script

```bash
python ladder_with_odds.py --api-key your_api_key_here
```

---

## 🚀 Step 3: Run the Ladder Strategy with Odds

```bash
cd mlb/strikeouts

# Generate predictions first
python predict.py

# Run ladder strategy with odds
python ladder_with_odds.py
```

---

## 📊 What You'll Get

### Without Odds API:
```
🎯 SWEET SPOT PITCHERS (5-8 K Projection)

1. Ryan Weathers (NYY vs LAA)
   Projection: 8.0 K's
   5.5+: 91% | 6.5+: 79% | 7.5+: 61%
```

### With Odds API:
```
🔥 TOP VALUE BETS (Sorted by Edge)

1. Ryan Weathers OVER 6.5 K's
   NYY vs LAA | Projection: 8.0 K's
   📊 Our Probability: 79%
   📖 DraftKings: +110 (Implied: 48%)
   💰 EDGE: 31% | EV: +64% | High Confidence

2. Ryan Weathers OVER 7.5 K's
   NYY vs LAA | Projection: 8.0 K's
   📊 Our Probability: 61%
   📖 FanDuel: +180 (Implied: 36%)
   💰 EDGE: 25% | EV: +71% | High Confidence
```

---

## 💰 Understanding the Output

### Edge
**Edge = Our Probability - Implied Probability**

Example:
- Our model: 79% chance of OVER 6.5 K
- DraftKings +110 implies: 48% chance
- **Edge: 31%** (huge value!)

### Expected Value (EV)
**How much you expect to win per $1 bet (long-term)**

Example:
- +64% EV means you expect to win $0.64 per $1 bet
- Over 100 bets of $100, you'd profit $6,400

### Confidence Levels
- **High (>15% edge)**: Strong bet, 1-2 units
- **Medium (5-15% edge)**: Good bet, 0.5-1 unit
- **Low (<5% edge)**: Skip or tiny bet

---

## 🎯 Recommended Ladder Strategy

When you see a pitcher with multiple value lines:

**Example: Ryan Weathers**
```
5.5+ K: 91% prob, -150 odds → 2 units (safe floor)
6.5+ K: 79% prob, +110 odds → 1.5 units (good value)
7.5+ K: 61% prob, +180 odds → 1 unit (high upside)
```

**Total Risk: 4.5 units**
**Potential Outcomes:**
- 5 K: Lose all (-4.5u)
- 6 K: Win 2u, lose 2.5u = -0.5u
- 7 K: Win 3.5u, lose 1u = +2.5u
- 8+ K: Win all = +6.3u

---

## 📈 API Usage Tips

### Monitor Your Quota
The script shows remaining requests:
```
📊 Odds API requests remaining: 487
```

### Optimize Usage
- Run once per day (morning for best lines)
- Don't refresh constantly
- 500 requests = ~30 days of daily use

### What Uses Requests
- Getting all MLB games: 1 request
- Getting strikeout odds: 1 request
- **Total per run: ~2 requests**

---

## 🔍 Troubleshooting

### "No API key found"
```bash
# Check if env variable is set
echo $ODDS_API_KEY

# If empty, set it
export ODDS_API_KEY='your_key'
```

### "No strikeout odds available"
- Odds may not be posted yet (check morning of game day)
- Not all books offer strikeout props for all games
- Try again closer to game time

### "Rate limit exceeded"
- You've used your 500 monthly requests
- Wait until next month or upgrade to paid plan
- Paid plan: $50/month for 10,000 requests

---

## 💡 Pro Tips

1. **Line Shopping**
   - Script shows best odds across all books
   - Always verify odds before betting
   - Lines move quickly, act fast on value

2. **Timing**
   - Best odds: 2-4 hours before first pitch
   - Avoid betting too early (lines not sharp)
   - Avoid betting too late (value gone)

3. **Bankroll Management**
   - Never bet more than 5% on one bet
   - Ladder total should be <10% of bankroll
   - Track results, adjust unit size

4. **Value Threshold**
   - Only bet when edge >5%
   - Prefer edge >10% for confidence
   - Edge >20% is rare, bet heavy

---

## 📚 Additional Resources

- **The Odds API Docs**: https://the-odds-api.com/liveapi/guides/v4/
- **Betting Markets**: https://the-odds-api.com/sports-odds-data/betting-markets.html
- **Historical Data**: https://the-odds-api.com/historical-odds-data/

---

## 🎓 Example Workflow

```bash
# Morning routine (9am)
cd mlb/strikeouts

# 1. Generate predictions
python predict.py

# 2. Find value bets with odds
python ladder_with_odds.py

# 3. Review recommendations
# 4. Place bets on sportsbook
# 5. Track results in spreadsheet

# Evening (after games)
python validate.py  # Check how model performed
```

---

**Ready to find value bets! 🚀💰**
