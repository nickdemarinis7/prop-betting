"""
Probability Calculator for Assist Thresholds
Calculates hit rates for 3+, 5+, 7+, 10+ assists based on historical data and model projection
"""

import numpy as np
from scipy import stats
import pandas as pd


class ProbabilityCalculator:
    """Calculate probability of hitting various assist thresholds"""
    
    def __init__(self):
        self.thresholds = [3, 5, 7, 10, 12, 15]
    
    def calculate_probabilities(self, projection, std_dev, recent_games_df=None):
        """
        Calculate probability of hitting each threshold
        
        Args:
            projection: Model's assist projection
            std_dev: Standard deviation from recent games
            recent_games_df: DataFrame of recent games (optional, for empirical method)
        
        Returns:
            Dictionary of {threshold: probability}
        """
        probabilities = {}
        
        # Method 1: Normal distribution approximation
        # Assists roughly follow a normal distribution
        for threshold in self.thresholds:
            if std_dev > 0:
                # Calculate z-score
                z_score = (threshold - 0.5 - projection) / std_dev  # -0.5 for continuity correction
                # Probability of exceeding threshold
                prob = 1 - stats.norm.cdf(z_score)
            else:
                # If no variance, use simple comparison
                prob = 1.0 if projection >= threshold else 0.0
            
            probabilities[f'{threshold}+'] = prob
        
        # Method 2: Empirical hit rate from recent games (if available)
        if recent_games_df is not None and 'AST' in recent_games_df.columns:
            empirical_probs = self._calculate_empirical_probabilities(recent_games_df)
            
            # Blend model-based and empirical (60% model, 40% empirical)
            for threshold_key in probabilities.keys():
                threshold = int(threshold_key.replace('+', ''))
                if threshold_key in empirical_probs:
                    model_prob = probabilities[threshold_key]
                    empirical_prob = empirical_probs[threshold_key]
                    blended_prob = 0.6 * model_prob + 0.4 * empirical_prob
                    probabilities[threshold_key] = blended_prob
        
        return probabilities
    
    def _calculate_empirical_probabilities(self, recent_games_df):
        """Calculate actual hit rates from recent games"""
        probabilities = {}
        
        assists = recent_games_df['AST'].values
        n_games = len(assists)
        
        if n_games == 0:
            return probabilities
        
        for threshold in self.thresholds:
            hit_count = np.sum(assists >= threshold)
            hit_rate = hit_count / n_games
            probabilities[f'{threshold}+'] = hit_rate
        
        return probabilities
    
    def calculate_expected_value(self, probability, american_odds):
        """
        Calculate expected value of a bet
        
        Args:
            probability: Your estimated probability (0-1)
            american_odds: Sportsbook odds (e.g., +430, -150)
        
        Returns:
            Expected value as percentage (positive = +EV)
        """
        if american_odds > 0:
            # Positive odds (underdog)
            decimal_odds = (american_odds / 100) + 1
        else:
            # Negative odds (favorite)
            decimal_odds = (100 / abs(american_odds)) + 1
        
        # Implied probability from odds
        implied_prob = 1 / decimal_odds
        
        # Expected value
        ev = (probability * (decimal_odds - 1)) - (1 - probability)
        ev_percentage = ev * 100
        
        # Edge (your probability vs implied probability)
        edge = (probability - implied_prob) * 100
        
        return {
            'ev_percentage': ev_percentage,
            'edge': edge,
            'implied_prob': implied_prob,
            'your_prob': probability,
            'decimal_odds': decimal_odds
        }
    
    def find_best_ladder_bets(self, probabilities, odds_dict):
        """
        Find the best ladder bet opportunities
        
        Args:
            probabilities: Dict of {threshold: probability}
            odds_dict: Dict of {threshold: american_odds}
        
        Returns:
            List of bets sorted by EV
        """
        bets = []
        
        for threshold_key, prob in probabilities.items():
            if threshold_key in odds_dict:
                odds = odds_dict[threshold_key]
                ev_data = self.calculate_expected_value(prob, odds)
                
                bets.append({
                    'threshold': threshold_key,
                    'probability': prob,
                    'odds': odds,
                    'ev': ev_data['ev_percentage'],
                    'edge': ev_data['edge'],
                    'implied_prob': ev_data['implied_prob']
                })
        
        # Sort by EV (highest first)
        bets.sort(key=lambda x: x['ev'], reverse=True)
        
        return bets
    
    def recommend_ladder_strategy(self, probabilities, odds_dict, bankroll_unit=100):
        """
        Recommend ladder betting strategy with Kelly Criterion sizing
        
        Args:
            probabilities: Dict of {threshold: probability}
            odds_dict: Dict of {threshold: american_odds}
            bankroll_unit: Base betting unit
        
        Returns:
            Recommended bets with sizing
        """
        bets = self.find_best_ladder_bets(probabilities, odds_dict)
        
        recommendations = []
        
        for bet in bets:
            # Only recommend positive EV bets
            if bet['ev'] > 0:
                # Kelly Criterion: f = (bp - q) / b
                # where b = decimal odds - 1, p = probability, q = 1-p
                prob = bet['probability']
                
                if bet['odds'] > 0:
                    b = bet['odds'] / 100
                else:
                    b = 100 / abs(bet['odds'])
                
                kelly_fraction = (b * prob - (1 - prob)) / b
                
                # Use fractional Kelly (25% of full Kelly for safety)
                conservative_kelly = kelly_fraction * 0.25
                
                # Calculate bet size
                bet_size = max(bankroll_unit * conservative_kelly, 0)
                
                # Only recommend if bet size is meaningful
                if bet_size >= bankroll_unit * 0.1:  # At least 10% of unit
                    recommendations.append({
                        **bet,
                        'kelly_fraction': kelly_fraction,
                        'recommended_units': conservative_kelly,
                        'bet_size': bet_size
                    })
        
        return recommendations


def format_ladder_recommendations(player_name, projection, recommendations):
    """Format ladder bet recommendations for display"""
    
    output = []
    output.append("=" * 80)
    output.append(f"🎯 LADDER BET ANALYSIS: {player_name}")
    output.append("=" * 80)
    output.append(f"Model Projection: {projection:.1f} AST")
    output.append("")
    
    if not recommendations:
        output.append("❌ No positive EV opportunities found")
        return "\n".join(output)
    
    output.append("✅ RECOMMENDED LADDER BETS (sorted by EV):")
    output.append("")
    
    for i, bet in enumerate(recommendations, 1):
        output.append(f"{i}. {bet['threshold']} AST @ {bet['odds']:+d}")
        output.append(f"   Your Probability: {bet['probability']:.1%}")
        output.append(f"   Implied Probability: {bet['implied_prob']:.1%}")
        output.append(f"   Edge: {bet['edge']:+.1f}%")
        output.append(f"   Expected Value: {bet['ev']:+.1f}%")
        output.append(f"   💰 Recommended Bet: {bet['recommended_units']:.2f} units (${bet['bet_size']:.0f})")
        
        if bet['ev'] > 10:
            output.append(f"   🔥 STRONG VALUE!")
        elif bet['ev'] > 5:
            output.append(f"   ✅ Good value")
        
        output.append("")
    
    total_units = sum(bet['recommended_units'] for bet in recommendations)
    total_amount = sum(bet['bet_size'] for bet in recommendations)
    
    output.append(f"Total Ladder Investment: {total_units:.2f} units (${total_amount:.0f})")
    output.append("=" * 80)
    
    return "\n".join(output)
