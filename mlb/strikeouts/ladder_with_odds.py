"""
Ladder Betting Strategy with Sportsbook Odds
Find the best value bets by comparing model projections to actual odds
"""

import pandas as pd
import sys
import os
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(os.path.join(os.path.dirname(__file__), '../../.env'))

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mlb.shared.scrapers.odds_api import OddsAPIScraper, calculate_implied_probability, calculate_expected_value


MAX_KELLY_UNITS = 3.0        # Cap any single bet
MAX_LADDER_UNITS = 5.0       # Cap total exposure per pitcher
KELLY_FRACTION = 0.25        # Quarter-Kelly (conservative)
MIN_EDGE = 0.05              # 5% minimum edge
SUSPICIOUS_EDGE = 0.25       # Flag edges above this
MIN_PROB = 0.15              # Don't bet if our prob < 15%
MAX_PROB = 0.95              # Don't bet if our prob > 95% (no value in heavy locks)


def _to_decimal_odds(american_odds):
    """Convert American odds to decimal odds."""
    if american_odds < 0:
        return 1 + (100 / abs(american_odds))
    else:
        return 1 + (american_odds / 100)


def _kelly_units(our_prob, american_odds, fraction=KELLY_FRACTION):
    """
    Calculate Kelly Criterion bet size.

    Returns units to bet (capped at MAX_KELLY_UNITS).
    """
    decimal = _to_decimal_odds(american_odds)
    b = decimal - 1  # net payout per $1
    q = 1 - our_prob

    if b <= 0:
        return 0.0

    kelly = (our_prob * b - q) / b
    if kelly <= 0:
        return 0.0

    # Use fractional Kelly, then scale so 1 full-Kelly ≈ 2 units
    units = kelly * fraction * 10
    return round(min(units, MAX_KELLY_UNITS), 2)


def analyze_value_bets(predictions_file, api_key=None):
    """
    Find best value bets by comparing model to sportsbook odds.

    Improvements over v1:
    - Evaluates both OVER and UNDER sides
    - Ranks by Expected Value (not raw edge)
    - Uses Kelly Criterion for unit sizing
    - Flags suspiciously large edges for manual review
    - Caps total ladder exposure per pitcher
    """
    print("=" * 80)
    print("💰 LADDER BETTING STRATEGY V2 - WITH SPORTSBOOK ODDS")
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

    # ------------------------------------------------------------------
    # Scan every pitcher × line × side for value
    # ------------------------------------------------------------------
    value_bets = []

    for _, pred in predictions.iterrows():
        pitcher = pred['pitcher']
        projection = pred['projection']
        confidence = pred.get('confidence', 'HIGH')

        # Skip LOW confidence pitchers entirely
        if confidence == 'LOW':
            continue

        # Match odds by last name
        pitcher_odds = odds_df[
            odds_df['pitcher'].str.contains(
                pitcher.split()[-1], case=False, na=False
            )
        ]
        if pitcher_odds.empty:
            continue

        for line in [3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]:
            prob_col = f'prob_{line}+'
            if prob_col not in pred:
                continue

            over_prob = pred[prob_col]
            under_prob = 1 - over_prob

            # --- Check OVER side ---
            over_odds = pitcher_odds[
                (pitcher_odds['line'] == line) &
                (pitcher_odds['over_under'] == 'Over')
            ]
            if not over_odds.empty:
                best = over_odds.loc[over_odds['odds'].idxmax()]
                _maybe_add_bet(
                    value_bets, pitcher, pred, projection, line, 'OVER',
                    over_prob, best['odds'], best['bookmaker']
                )

            # --- Check UNDER side ---
            under_odds = pitcher_odds[
                (pitcher_odds['line'] == line) &
                (pitcher_odds['over_under'] == 'Under')
            ]
            if not under_odds.empty:
                best = under_odds.loc[under_odds['odds'].idxmax()]
                _maybe_add_bet(
                    value_bets, pitcher, pred, projection, line, 'UNDER',
                    under_prob, best['odds'], best['bookmaker']
                )

    if not value_bets:
        print("\n⚠️  No value bets found (no significant edges)")
        return

    # ------------------------------------------------------------------
    # Rank by EV (not raw edge)
    # ------------------------------------------------------------------
    value_df = pd.DataFrame(value_bets).sort_values('ev', ascending=False)

    # ------------------------------------------------------------------
    # Display: Top Value Bets
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("🔥 TOP VALUE BETS (Ranked by Expected Value)")
    print("=" * 80)
    print(f"\nFilters: Edge >{MIN_EDGE:.0%} | Prob {MIN_PROB:.0%}-{MAX_PROB:.0%} | Odds better than -200\n")

    for i, (_, bet) in enumerate(value_df.head(15).iterrows(), 1):
        side_icon = "📈" if bet['side'] == 'OVER' else "📉"
        flag = " ⚠️  VERIFY" if bet['suspicious'] else ""
        conf = bet.get('confidence', '')
        conf_icon = '🟢' if conf == 'HIGH' else '🟡'
        rf = bet.get('red_flags', 'None')
        rf_str = f" | ⚠️ {rf}" if rf != 'None' and pd.notna(rf) else ''
        print(f"{i:2d}. {conf_icon} {bet['pitcher']:25s} {bet['side']} {bet['line']} K's{flag}")
        print(f"    {side_icon} {bet['team']} vs {bet['opponent']} | Proj: {bet['projection']:.1f} K's{rf_str}")
        print(f"    📊 Our Prob: {bet['our_prob']:.1%}")
        print(f"    📖 {bet['bookmaker']}: {bet['book_odds']:+d} (Implied: {bet['implied_prob']:.1%})")
        print(f"    💰 EV: {bet['ev']:+.1%} | Edge: {bet['edge']:.1%} | Kelly: {bet['kelly_units']:.2f}u")
        print()

    # ------------------------------------------------------------------
    # Ladder recommendations (user's preferred style)
    # ------------------------------------------------------------------
    print("=" * 80)
    print("🎯 RECOMMENDED LADDER STRATEGIES")
    print("=" * 80)
    print("\nStrategy: Either 1 pitcher with 2+ rungs, or 2-3 pitchers with matching rungs")
    print("Max exposure: 1 unit per pitcher\n")

    pitcher_groups = {pitcher: group for pitcher, group in value_df.groupby('pitcher')}
    
    # Find pitchers with 2+ good value bets (single-pitcher ladder)
    single_pitcher_ladders = []
    for pitcher, group in pitcher_groups.items():
        if len(group) >= 2:
            # Keep original Kelly units, then cap total to 1.0u
            total_ev = group['ev'].sum()
            bets = group.sort_values('ev', ascending=False).head(3).copy()
            
            # Calculate total Kelly units
            total_kelly = bets['kelly_units'].sum()
            
            # Scale down if exceeds 1.0u
            if total_kelly > 1.0:
                scale_factor = 1.0 / total_kelly
                bets['kelly_units'] = (bets['kelly_units'] * scale_factor).round(2)
            
            single_pitcher_ladders.append({
                'pitcher': pitcher,
                'bets': bets,
                'total_ev': total_ev,
                'type': 'single'
            })
    
    # Find groups of 2-3 pitchers with matching rung counts
    multi_pitcher_groups = []
    pitcher_rung_counts = {p: len(g) for p, g in pitcher_groups.items()}
    
    # Group by rung count
    for rung_count in [2, 3, 4, 5]:
        pitchers_with_rungs = [p for p, rc in pitcher_rung_counts.items() if rc >= rung_count]
        if len(pitchers_with_rungs) >= 2:
            # Sort by total EV
            sorted_pitchers = sorted(pitchers_with_rungs, 
                                    key=lambda p: pitcher_groups[p]['ev'].sum(), 
                                    reverse=True)
            
            # Generate multiple combinations
            # 2-pitcher groups
            for i in range(len(sorted_pitchers)):
                for j in range(i + 1, len(sorted_pitchers)):
                    group = [sorted_pitchers[i], sorted_pitchers[j]]
                    group_bets = []
                    total_ev = 0
                    all_bets = []
                    for p in group:
                        g = pitcher_groups[p].sort_values('ev', ascending=False).head(rung_count).copy()
                        all_bets.extend(g.to_dict('records'))
                        group_bets.append({'pitcher': p, 'bets': g})
                        total_ev += g['ev'].sum()
                    
                    # Calculate total Kelly units across all bets
                    total_kelly = sum(b['kelly_units'] for b in all_bets)
                    
                    # Scale down if exceeds 1.0u
                    if total_kelly > 1.0:
                        scale_factor = 1.0 / total_kelly
                        for pb in group_bets:
                            pb['bets']['kelly_units'] = (pb['bets']['kelly_units'] * scale_factor).round(2)
                    
                    multi_pitcher_groups.append({
                        'pitchers': group,
                        'bets': group_bets,
                        'rung_count': rung_count,
                        'total_ev': total_ev,
                        'type': 'multi'
                    })
            
            # 3-pitcher groups (if enough pitchers)
            if len(sorted_pitchers) >= 3:
                for i in range(len(sorted_pitchers)):
                    for j in range(i + 1, len(sorted_pitchers)):
                        for k in range(j + 1, len(sorted_pitchers)):
                            group = [sorted_pitchers[i], sorted_pitchers[j], sorted_pitchers[k]]
                            group_bets = []
                            total_ev = 0
                            all_bets = []
                            for p in group:
                                g = pitcher_groups[p].sort_values('ev', ascending=False).head(rung_count).copy()
                                all_bets.extend(g.to_dict('records'))
                                group_bets.append({'pitcher': p, 'bets': g})
                                total_ev += g['ev'].sum()
                            
                            # Calculate total Kelly units across all bets
                            total_kelly = sum(b['kelly_units'] for b in all_bets)
                            
                            # Scale down if exceeds 1.0u
                            if total_kelly > 1.0:
                                scale_factor = 1.0 / total_kelly
                                for pb in group_bets:
                                    pb['bets']['kelly_units'] = (pb['bets']['kelly_units'] * scale_factor).round(2)
                            
                            multi_pitcher_groups.append({
                                'pitchers': group,
                                'bets': group_bets,
                                'rung_count': rung_count,
                                'total_ev': total_ev,
                                'type': 'multi'
                            })
    
    # Display top 10 single-pitcher ladders (if exist)
    if single_pitcher_ladders:
        print("🎯 OPTION 1: Single Pitcher Ladders (Top 10)")
        print("=" * 80)
        
        sorted_singles = sorted(single_pitcher_ladders, key=lambda x: x['total_ev'], reverse=True)[:10]
        for i, ladder in enumerate(sorted_singles, 1):
            pitcher_pred = predictions[predictions['pitcher'] == ladder['pitcher']].iloc[0]
            print(f"\n{i}. {ladder['pitcher']}")
            print(f"   Projection: {pitcher_pred['projection']:.1f} K's | Total EV: {ladder['total_ev']:+.1%}")
            print(f"   Total Exposure: 1.0u ({len(ladder['bets'])} rungs)\n")
            
            for _, bet in ladder['bets'].iterrows():
                side_icon = "📈" if bet['side'] == 'OVER' else "📉"
                print(f"   • {bet['kelly_units']:.2f}u {side_icon} {bet['side']} {bet['line']} @ {bet['book_odds']:+d} ({bet['bookmaker']})")
                print(f"     EV: {bet['ev']:+.1%} | Edge: {bet['edge']:.1%} | Our Prob: {bet['our_prob']:.1%}")
            print()
    
    # Display top 3 multi-pitcher groups (if exist)
    if multi_pitcher_groups:
        print("\n🎯 OPTION 2: Multi-Pitcher Ladders (Top 3)")
        print("=" * 80)
        
        sorted_multis = sorted(multi_pitcher_groups, key=lambda x: x['total_ev'], reverse=True)[:3]
        for i, ladder in enumerate(sorted_multis, 1):
            print(f"\n{i}. {ladder['rung_count']} rungs each | Total EV: {ladder['total_ev']:+.1%}")
            print(f"   Total Exposure: 1.0u (split across {len(ladder['pitchers'])} pitchers)\n")
            
            for j, p_bets in enumerate(ladder['bets'], 1):
                pitcher = p_bets['pitcher']
                pitcher_pred = predictions[predictions['pitcher'] == pitcher].iloc[0]
                pitcher_total = p_bets['bets']['kelly_units'].sum()
                print(f"   {j}. {pitcher} (Proj: {pitcher_pred['projection']:.1f} K's) - Exposure: {pitcher_total:.2f}u")
                for _, bet in p_bets['bets'].iterrows():
                    side_icon = "📈" if bet['side'] == 'OVER' else "📉"
                    print(f"      • {bet['kelly_units']:.2f}u {side_icon} {bet['side']} {bet['line']} @ {bet['book_odds']:+d} ({bet['bookmaker']})")
                    print(f"        EV: {bet['ev']:+.1%} | Edge: {bet['edge']:.1%}")
                print()
    
    if not single_pitcher_ladders and not multi_pitcher_groups:
        print("\n⚠️  No suitable ladder opportunities found")
        print("   Need pitchers with 3+ value bets to form ladders")
        print("   Consider single bets from the value list above")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("=" * 80)
    print("📈 SUMMARY STATISTICS")
    print("=" * 80)

    over_bets = value_df[value_df['side'] == 'OVER']
    under_bets = value_df[value_df['side'] == 'UNDER']
    suspicious = value_df[value_df['suspicious']]

    print(f"\nTotal Value Bets:    {len(value_df)} ({len(over_bets)} overs, {len(under_bets)} unders)")
    print(f"Average EV:          {value_df['ev'].mean():+.1%}")
    print(f"Average Edge:        {value_df['edge'].mean():.1%}")
    print(f"Total Kelly Units:   {value_df['kelly_units'].sum():.1f}u")
    if len(suspicious) > 0:
        print(f"⚠️  Suspicious Edges: {len(suspicious)} bets (edge >{SUSPICIOUS_EDGE:.0%} — verify before placing)")

    print("\n" + "=" * 80)
    print("💡 BETTING GUIDE")
    print("=" * 80)
    print("""
1. UNIT SIZING (Kelly Criterion):
   Units are calculated from edge & odds — bigger edge at longer
   odds = more units.  We use quarter-Kelly for safety.
   Max single bet: %.1fu | Max per pitcher: %.1fu

2. OVER vs UNDER:
   📈 OVER = pitcher exceeds the line
   📉 UNDER = pitcher stays below the line
   Low-K pitchers often have UNDER value the books miss.

3. ⚠️  SUSPICIOUS EDGES (>%d%%):
   Edges above %d%% are unusually large. They may indicate:
   - Model overconfidence on this pitcher profile
   - Stale odds (line already moved)
   - Genuine mispricing (rare but real)
   Always double-check these before betting.

4. BANKROLL MANAGEMENT:
   - Never exceed listed units (already Kelly-sized)
   - Total daily exposure should stay under 10%% of bankroll
   - Track results to calibrate model confidence over time
""" % (MAX_KELLY_UNITS, MAX_LADDER_UNITS, int(SUSPICIOUS_EDGE * 100), int(SUSPICIOUS_EDGE * 100)))


def _maybe_add_bet(value_bets, pitcher, pred, projection, line, side,
                    our_prob, book_odds, bookmaker):
    """Evaluate one side of a line and append to value_bets if it qualifies."""
    if our_prob < MIN_PROB or our_prob > MAX_PROB:
        return
    if book_odds <= -200:
        return

    implied_prob = calculate_implied_probability(book_odds)
    edge = our_prob - implied_prob
    ev = calculate_expected_value(our_prob, book_odds)

    if edge < MIN_EDGE:
        return

    kelly = _kelly_units(our_prob, book_odds)
    if kelly <= 0:
        return

    suspicious = edge > SUSPICIOUS_EDGE

    confidence = pred.get('confidence', 'HIGH')
    red_flags = pred.get('red_flags', 'None')

    # Reduce Kelly sizing for MEDIUM confidence
    if confidence == 'MEDIUM':
        kelly = round(kelly * 0.6, 2)

    value_bets.append({
        'pitcher': pitcher,
        'team': pred.get('team', ''),
        'opponent': pred.get('opponent', ''),
        'projection': projection,
        'line': line,
        'side': side,
        'our_prob': our_prob,
        'book_odds': book_odds,
        'bookmaker': bookmaker,
        'implied_prob': implied_prob,
        'edge': edge,
        'ev': ev,
        'kelly_units': kelly,
        'suspicious': suspicious,
        'confidence': confidence,
        'red_flags': red_flags,
    })


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
