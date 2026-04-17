"""
Ladder Betting Strategy with Sportsbook Odds
Find the best value bets by comparing model projections to actual odds
"""

import pandas as pd
import sys
import os

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mlb.shared.scrapers.odds_api import OddsAPIScraper, calculate_implied_probability, calculate_expected_value


def analyze_value_bets(predictions_file, api_key=None):
    """
    Find best value bets by comparing model to sportsbook odds
    
    Args:
        predictions_file: Path to predictions CSV
        api_key: The Odds API key (optional)
    """
    print("=" * 80)
    print("💰 LADDER BETTING STRATEGY - WITH SPORTSBOOK ODDS")
    print("=" * 80)
    
    # Load predictions
    predictions = pd.read_csv(predictions_file)
    print(f"\n📊 Loaded {len(predictions)} pitcher projections")
    
    # Get sportsbook odds
    print("\n📡 Fetching sportsbook odds...")
    odds_scraper = OddsAPIScraper(api_key=api_key)
    odds_df = odds_scraper.get_all_strikeout_odds()
    
    if odds_df.empty:
        print("\n⚠️  No sportsbook odds available")
        print("   To get real odds:")
        print("   1. Get free API key at: https://the-odds-api.com/")
        print("   2. Set environment variable: export ODDS_API_KEY='your_key'")
        print("   3. Re-run this script")
        print("\n   Showing projections only (no odds comparison)...\n")
        return show_projections_only(predictions)
    
    print(f"✅ Found odds for {len(odds_df)} pitcher/line combinations")
    
    # Find value bets
    value_bets = []
    
    for _, pred in predictions.iterrows():
        pitcher = pred['pitcher']
        
        # Get odds for this pitcher
        pitcher_odds = odds_df[
            odds_df['pitcher'].str.contains(pitcher.split()[-1], case=False, na=False)  # Match by last name
        ]
        
        if pitcher_odds.empty:
            continue
        
        # Check each line
        for line in [4.5, 5.5, 6.5, 7.5, 8.5]:
            prob_col = f'prob_{line}+'
            if prob_col not in pred:
                continue
            
            our_prob = pred[prob_col]
            
            # Get best OVER odds for this line
            line_odds = pitcher_odds[
                (pitcher_odds['line'] == line) &
                (pitcher_odds['over_under'] == 'Over')
            ]
            
            if line_odds.empty:
                continue
            
            # Find best odds across bookmakers
            best_odds_row = line_odds.loc[line_odds['odds'].idxmax()]
            best_odds = best_odds_row['odds']
            bookmaker = best_odds_row['bookmaker']
            
            # Calculate edge
            implied_prob = calculate_implied_probability(best_odds)
            ev = calculate_expected_value(our_prob, best_odds)
            edge = our_prob - implied_prob
            
            # Filter criteria:
            # 1. Must have at least 5% edge
            # 2. Odds must be -200 or better (no heavy favorites)
            if edge > 0.05 and best_odds > -200:  # At least 5% edge and reasonable odds
                value_bets.append({
                    'pitcher': pitcher,
                    'team': pred.get('team', ''),
                    'opponent': pred.get('opponent', ''),
                    'projection': pred['projection'],
                    'line': line,
                    'our_prob': our_prob,
                    'book_odds': best_odds,
                    'bookmaker': bookmaker,
                    'implied_prob': implied_prob,
                    'edge': edge,
                    'ev': ev,
                    'confidence': 'High' if edge > 0.15 else 'Medium'
                })
    
    if not value_bets:
        print("\n⚠️  No value bets found (no significant edges)")
        return
    
    # Convert to DataFrame and sort by edge
    value_df = pd.DataFrame(value_bets).sort_values('edge', ascending=False)
    
    # Display results
    print("\n" + "=" * 80)
    print("🔥 TOP VALUE BETS (Sorted by Edge)")
    print("=" * 80)
    print("\nFilters: Edge >5% | Odds better than -200")
    print("These are bets where our model gives significantly higher probability")
    print("than the sportsbook's implied probability.\n")
    
    for i, (_, bet) in enumerate(value_df.head(15).iterrows(), 1):
        print(f"{i:2d}. {bet['pitcher']:25s} OVER {bet['line']} K's")
        print(f"    {bet['team']} vs {bet['opponent']} | Projection: {bet['projection']:.1f} K's")
        print(f"    📊 Our Probability: {bet['our_prob']:.1%}")
        print(f"    📖 {bet['bookmaker']}: {bet['book_odds']:+d} (Implied: {bet['implied_prob']:.1%})")
        print(f"    💰 EDGE: {bet['edge']:.1%} | EV: {bet['ev']:+.1%} | {bet['confidence']} Confidence")
        print()
    
    # Ladder recommendations
    print("=" * 80)
    print("🎯 RECOMMENDED LADDER STRATEGIES")
    print("=" * 80)
    
    # Group by pitcher
    pitcher_groups = value_df.groupby('pitcher')
    
    ladder_count = 0
    for pitcher, group in pitcher_groups:
        if len(group) >= 2 and ladder_count < 5:  # At least 2 lines with value
            ladder_count += 1
            
            pitcher_pred = predictions[predictions['pitcher'] == pitcher].iloc[0]
            
            print(f"\n{ladder_count}. {pitcher}")
            print(f"   Projection: {pitcher_pred['projection']:.1f} K's")
            print(f"   Ladder Strategy:")
            
            for _, bet in group.iterrows():
                units = 1.0 if bet['edge'] > 0.15 else 0.5
                print(f"   • {units:.1f}u on OVER {bet['line']} @ {bet['book_odds']:+d} ({bet['bookmaker']})")
                print(f"     Edge: {bet['edge']:.1%} | Our Prob: {bet['our_prob']:.1%}")
            print()
    
    if ladder_count == 0:
        print("\n⚠️  No multi-line ladder opportunities found")
        print("   Consider single bets from the value list above")
    
    # Summary stats
    print("=" * 80)
    print("📈 SUMMARY STATISTICS")
    print("=" * 80)
    print(f"\nTotal Value Bets Found: {len(value_df)}")
    print(f"Average Edge: {value_df['edge'].mean():.1%}")
    print(f"Average EV: {value_df['ev'].mean():+.1%}")
    print(f"High Confidence Bets (>15% edge): {len(value_df[value_df['edge'] > 0.15])}")
    print(f"Medium Confidence Bets (5-15% edge): {len(value_df[(value_df['edge'] >= 0.05) & (value_df['edge'] <= 0.15)])}")
    
    print("\n" + "=" * 80)
    print("💡 BETTING TIPS")
    print("=" * 80)
    print("""
1. EDGE INTERPRETATION:
   - >20% edge: Rare, bet heavy (2-3 units)
   - 15-20% edge: Strong value (1.5-2 units)
   - 10-15% edge: Good value (1 unit)
   - 5-10% edge: Small value (0.5 units)

2. LADDER STRATEGY:
   - Bet more on lower lines (higher probability)
   - Bet less on higher lines (lower probability but better odds)
   - Example: 2u on 5.5, 1u on 6.5, 0.5u on 7.5

3. BANKROLL MANAGEMENT:
   - Never bet more than 5% of bankroll on one bet
   - Ladder total should not exceed 10% of bankroll
   - Track results and adjust unit size based on performance

4. LINE SHOPPING:
   - Odds shown are best available across books
   - Always check multiple sportsbooks before placing bet
   - Even small odds differences matter over time
""")


def show_projections_only(predictions):
    """Show projections without odds comparison"""
    from ladder_strategy import find_sweet_spot_pitchers
    
    print("\n" + "=" * 80)
    print("🎯 SWEET SPOT PITCHERS (5-8 K Projection)")
    print("=" * 80)
    
    sweet_spot = find_sweet_spot_pitchers('predictions_strikeouts_' + datetime.now().strftime('%Y%m%d') + '.csv', 5.0, 8.0)
    
    if not sweet_spot.empty:
        for i, (_, row) in enumerate(sweet_spot.head(10).iterrows(), 1):
            print(f"\n{i:2d}. {row['pitcher']:25s} ({row.get('team', '')} vs {row.get('opponent', '')})")
            print(f"    Projection: {row['projection']:.1f} K's")
            print(f"    5.5+: {row.get('prob_5.5+', 0):.0%} | 6.5+: {row.get('prob_6.5+', 0):.0%} | 7.5+: {row.get('prob_7.5+', 0):.0%}")


if __name__ == "__main__":
    from datetime import datetime
    
    # Get API key from environment or command line
    api_key = os.getenv('ODDS_API_KEY')
    
    if len(sys.argv) > 1:
        predictions_file = sys.argv[1]
    else:
        today = datetime.now().strftime('%Y%m%d')
        predictions_file = f"predictions_strikeouts_{today}.csv"
    
    if not os.path.exists(predictions_file):
        print(f"❌ Predictions file not found: {predictions_file}")
        print(f"   Run predict.py first to generate predictions")
        sys.exit(1)
    
    analyze_value_bets(predictions_file, api_key=api_key)
