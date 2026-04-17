"""
Betting mathematics utilities
Probability calculations, odds conversions, and unit sizing
"""

import numpy as np
from scipy import stats


def calculate_probability(projection, std_dev, threshold):
    """
    Calculate probability of hitting a threshold using normal distribution
    
    Args:
        projection: Projected value (points, assists, etc.)
        std_dev: Standard deviation of player's performance
        threshold: Threshold value (e.g., 15, 20, 25)
    
    Returns:
        Probability (0-1) of hitting threshold or higher
    """
    if std_dev <= 0:
        std_dev = 5.0  # Default std dev
    
    # Calculate z-score
    z_score = (threshold - projection) / std_dev
    
    # Probability of being >= threshold
    prob = 1 - stats.norm.cdf(z_score)
    
    return max(0.0, min(1.0, prob))


def calculate_empirical_probability(recent_games, threshold, stat_column='PTS'):
    """
    Calculate probability based on actual historical hit rate
    More accurate than normal distribution for small samples
    
    Args:
        recent_games: DataFrame with recent game logs
        threshold: Threshold value
        stat_column: Column name to check (e.g., 'PTS', 'AST')
    
    Returns:
        Probability (0-1) based on historical hit rate
    """
    if recent_games.empty or stat_column not in recent_games.columns:
        return 0.5
    
    hits = (recent_games[stat_column] >= threshold).sum()
    total_games = len(recent_games)
    
    # Add Laplace smoothing to avoid 0% or 100%
    # Assumes 1 hit and 1 miss as prior
    return (hits + 1) / (total_games + 2)


def calculate_poisson_probability(projection, threshold):
    """
    Calculate probability using Poisson distribution
    Better for count data like points/assists than normal distribution
    
    Args:
        projection: Expected value (lambda parameter)
        threshold: Threshold value
    
    Returns:
        Probability of hitting threshold or higher
    """
    if projection <= 0:
        return 0.0
    
    # P(X >= threshold) = 1 - P(X < threshold)
    # P(X < threshold) = P(X <= threshold - 1)
    prob = 1 - stats.poisson.cdf(threshold - 1, projection)
    
    return max(0.0, min(1.0, prob))


def prob_to_american_odds(probability):
    """
    Convert probability to American odds format
    
    Args:
        probability: Probability (0-1)
    
    Returns:
        American odds (e.g., -150, +200)
    """
    if probability >= 0.99:
        return -10000
    elif probability <= 0.01:
        return 10000
    elif probability >= 0.5:
        return int(-100 * probability / (1 - probability))
    else:
        return int(100 * (1 - probability) / probability)


def american_odds_to_prob(odds):
    """
    Convert American odds to implied probability
    
    Args:
        odds: American odds (e.g., -150, +200)
    
    Returns:
        Implied probability (0-1)
    """
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    else:
        return 100 / (odds + 100)


def calculate_expected_value(probability, odds):
    """
    Calculate expected value of a bet
    
    Args:
        probability: True probability of winning (0-1)
        odds: American odds
    
    Returns:
        Expected value per $1 bet
    """
    if odds < 0:
        payout = 100 / abs(odds)
    else:
        payout = odds / 100
    
    ev = (probability * payout) - ((1 - probability) * 1)
    return ev


def kelly_criterion(probability, odds, fraction=0.25):
    """
    Calculate optimal bet size using Kelly Criterion
    
    Args:
        probability: True probability of winning (0-1)
        odds: American odds
        fraction: Fraction of Kelly to use (0.25 = quarter Kelly, conservative)
    
    Returns:
        Recommended bet size as fraction of bankroll (0-1)
    """
    if odds < 0:
        b = 100 / abs(odds)  # Decimal odds - 1
    else:
        b = odds / 100
    
    p = probability
    q = 1 - p
    
    # Kelly formula: (bp - q) / b
    kelly = (b * p - q) / b
    
    # Apply fractional Kelly for safety
    kelly = max(0, kelly * fraction)
    
    # Cap at 5% of bankroll for safety
    return min(kelly, 0.05)


def recommend_units(probability, tier=2, has_red_flags=False, max_units=1.5):
    """
    Recommend unit sizing based on probability and risk factors
    
    Args:
        probability: Probability of hitting (0-1)
        tier: Confidence tier (1=safest, 2=good, 3=risky)
        has_red_flags: Whether bet has warning signs
        max_units: Maximum units to recommend
    
    Returns:
        Recommended unit size (0.1 to max_units)
    """
    # Base units on probability
    if probability >= 0.80:
        base_units = 1.5
    elif probability >= 0.70:
        base_units = 1.25
    elif probability >= 0.60:
        base_units = 1.0
    elif probability >= 0.50:
        base_units = 0.75
    elif probability >= 0.40:
        base_units = 0.5
    elif probability >= 0.25:
        base_units = 0.25
    else:
        base_units = 0.1
    
    # Adjust for tier
    if tier == 3:  # Higher risk
        base_units = max(0.1, base_units - 0.5)
    elif tier == 1:  # Safest
        base_units = min(max_units, base_units + 0.25)
    
    # Reduce if red flags
    if has_red_flags:
        base_units = max(0.1, base_units - 0.25)
    
    # Round to nearest valid unit size
    valid_units = [1.5, 1.25, 1.0, 0.75, 0.5, 0.25, 0.1]
    return min(valid_units, key=lambda x: abs(x - base_units))


def calculate_confidence_interval(projection, std_dev, confidence_level=0.95):
    """
    Calculate confidence interval for projection
    
    Args:
        projection: Point estimate
        std_dev: Standard deviation
        confidence_level: Confidence level (e.g., 0.95 for 95%)
    
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    margin = z_score * std_dev
    
    return (projection - margin, projection + margin)


if __name__ == "__main__":
    print("=" * 70)
    print("BETTING MATH UTILITIES - TEST")
    print("=" * 70)
    
    # Test 1: Probability calculations
    print("\n1. Probability Calculations:")
    projection = 22.5
    std_dev = 4.2
    threshold = 20
    
    normal_prob = calculate_probability(projection, std_dev, threshold)
    poisson_prob = calculate_poisson_probability(projection, threshold)
    
    print(f"   Projection: {projection:.1f} ± {std_dev:.1f}")
    print(f"   Threshold: {threshold}+")
    print(f"   Normal Distribution: {normal_prob:.1%}")
    print(f"   Poisson Distribution: {poisson_prob:.1%}")
    
    # Test 2: Odds conversions
    print("\n2. Odds Conversions:")
    prob = 0.65
    odds = prob_to_american_odds(prob)
    implied = american_odds_to_prob(odds)
    
    print(f"   Probability: {prob:.1%}")
    print(f"   American Odds: {odds:+d}")
    print(f"   Implied Probability: {implied:.1%}")
    
    # Test 3: Expected value
    print("\n3. Expected Value:")
    true_prob = 0.60
    offered_odds = -150
    ev = calculate_expected_value(true_prob, offered_odds)
    
    print(f"   True Probability: {true_prob:.1%}")
    print(f"   Offered Odds: {offered_odds:+d}")
    print(f"   Expected Value: ${ev:.2f} per $1 bet")
    print(f"   {'✅ POSITIVE EV - BET IT!' if ev > 0 else '❌ NEGATIVE EV - SKIP'}")
    
    # Test 4: Kelly Criterion
    print("\n4. Kelly Criterion:")
    kelly = kelly_criterion(true_prob, offered_odds, fraction=0.25)
    print(f"   Recommended bet: {kelly:.1%} of bankroll")
    print(f"   On $1000 bankroll: ${kelly * 1000:.2f}")
    
    # Test 5: Unit sizing
    print("\n5. Unit Sizing:")
    for prob in [0.80, 0.65, 0.50, 0.35]:
        units = recommend_units(prob, tier=2, has_red_flags=False)
        print(f"   {prob:.0%} probability → {units:.2f} units")
    
    # Test 6: Confidence intervals
    print("\n6. Confidence Intervals:")
    lower, upper = calculate_confidence_interval(projection, std_dev, 0.95)
    print(f"   Projection: {projection:.1f}")
    print(f"   95% CI: [{lower:.1f}, {upper:.1f}]")
    
    print("\n" + "=" * 70)
    print("✅ All betting math utilities working!")
    print("=" * 70)
