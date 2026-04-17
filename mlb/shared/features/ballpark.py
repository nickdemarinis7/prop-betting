"""
Ballpark Factors
Strikeout rates vary by ballpark due to dimensions, altitude, etc.
"""


class BallparkFactors:
    """Ballpark adjustment factors for strikeouts"""
    
    def __init__(self):
        # Strikeout factors by ballpark (1.0 = league average)
        # Higher = more strikeouts, Lower = fewer strikeouts
        self.k_factors = {
            # High strikeout parks
            'Coors Field': 0.92,  # Altitude = better bat speed
            'Great American Ball Park': 0.95,  # Hitter-friendly
            'Fenway Park': 0.96,  # Small park, aggressive swings
            
            # Average parks
            'Yankee Stadium': 1.00,
            'Dodger Stadium': 1.02,
            'Wrigley Field': 0.98,
            'Oracle Park': 1.01,
            'Petco Park': 1.03,
            'T-Mobile Park': 1.02,
            'Minute Maid Park': 0.99,
            'Globe Life Field': 0.97,
            'Truist Park': 0.98,
            'Busch Stadium': 1.01,
            'Citizens Bank Park': 0.96,
            'Comerica Park': 1.02,
            'Guaranteed Rate Field': 0.99,
            'Kauffman Stadium': 1.01,
            'Angel Stadium': 1.00,
            'Oakland Coliseum': 1.03,
            'Progressive Field': 1.01,
            'Target Field': 1.02,
            'Tropicana Field': 1.04,  # Dome, consistent conditions
            'Rogers Centre': 1.03,  # Dome
            'Chase Field': 1.01,  # Retractable roof
            'Marlins Park': 1.02,
            'Nationals Park': 0.99,
            'PNC Park': 1.00,
            'American Family Field': 1.01,
            'Camden Yards': 0.98,
            'Citi Field': 1.02,
            'loanDepot park': 1.02,
            
            # Default
            'Average': 1.00,
            'Unknown': 1.00
        }
        
        # Run environment factors (for future use)
        self.run_factors = {
            'Coors Field': 1.25,  # Highest scoring
            'Great American Ball Park': 1.15,
            'Fenway Park': 1.10,
            'Yankee Stadium': 1.08,
            'Citizens Bank Park': 1.12,
            'Globe Life Field': 1.10,
            'Dodger Stadium': 0.95,  # Pitcher-friendly
            'Oracle Park': 0.92,
            'Petco Park': 0.90,
            'T-Mobile Park': 0.95,
            'Tropicana Field': 0.98,
            'Oakland Coliseum': 0.93,
            'Average': 1.00
        }
    
    def get_k_factor(self, ballpark):
        """
        Get strikeout factor for a ballpark
        
        Args:
            ballpark: Ballpark name
            
        Returns:
            Float factor (1.0 = average)
        """
        return self.k_factors.get(ballpark, 1.00)
    
    def get_run_factor(self, ballpark):
        """
        Get run environment factor for a ballpark
        
        Args:
            ballpark: Ballpark name
            
        Returns:
            Float factor (1.0 = average)
        """
        return self.run_factors.get(ballpark, 1.00)
    
    def get_all_factors(self, ballpark):
        """
        Get all factors for a ballpark
        
        Args:
            ballpark: Ballpark name
            
        Returns:
            Dict with all factors
        """
        return {
            'ballpark': ballpark,
            'k_factor': self.get_k_factor(ballpark),
            'run_factor': self.get_run_factor(ballpark)
        }
    
    def get_high_k_parks(self, threshold=1.02):
        """
        Get ballparks that favor strikeouts
        
        Args:
            threshold: Minimum K factor
            
        Returns:
            List of high-K ballparks
        """
        return [park for park, factor in self.k_factors.items() 
                if factor >= threshold]
    
    def get_low_k_parks(self, threshold=0.98):
        """
        Get ballparks that suppress strikeouts
        
        Args:
            threshold: Maximum K factor
            
        Returns:
            List of low-K ballparks
        """
        return [park for park, factor in self.k_factors.items() 
                if factor <= threshold]


# Example usage
if __name__ == "__main__":
    ballpark = BallparkFactors()
    
    # Test some parks
    test_parks = ['Coors Field', 'Tropicana Field', 'Yankee Stadium', 'Unknown']
    
    print("Ballpark K Factors:")
    for park in test_parks:
        k_factor = ballpark.get_k_factor(park)
        print(f"  {park:25s} {k_factor:.2f}")
    
    print("\nHigh-K Parks (1.02+):")
    for park in ballpark.get_high_k_parks():
        print(f"  • {park}")
    
    print("\nLow-K Parks (0.98 or less):")
    for park in ballpark.get_low_k_parks():
        print(f"  • {park}")
