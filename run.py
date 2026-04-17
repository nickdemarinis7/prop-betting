#!/usr/bin/env python3
"""
Main entry point for sports betting prediction system
Supports multiple sports and prop types
"""

import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description='Sports Betting Prediction System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run NBA assists predictions
  python run.py nba assists predict
  
  # Validate NBA assists picks from yesterday
  python run.py nba assists validate --date 20260408
  
  # Run NBA points predictions
  python run.py nba points predict
  
  # Validate NBA points picks
  python run.py nba points validate --date 20260408
        """
    )
    
    parser.add_argument('sport', choices=['nba', 'nfl', 'mlb'], 
                       help='Sport to analyze')
    parser.add_argument('prop', choices=['assists', 'points', 'rebounds', 'threes'],
                       help='Prop type to predict')
    parser.add_argument('action', choices=['predict', 'validate'],
                       help='Action to perform')
    parser.add_argument('--date', type=str, default=None,
                       help='Date for validation (YYYYMMDD format)')
    
    args = parser.parse_args()
    
    # Route to appropriate module
    if args.sport == 'nba':
        if args.prop == 'assists':
            if args.action == 'predict':
                from nba.assists import predict
                predict.main()
            elif args.action == 'validate':
                from nba.assists import validate
                validate.main(args.date)
        
        elif args.prop == 'points':
            if args.action == 'predict':
                from nba.points import predict
                predict.main()
            elif args.action == 'validate':
                from nba.points import validate
                validate.main(args.date)
        
        else:
            print(f"❌ {args.prop} not yet implemented for NBA")
            sys.exit(1)
    
    else:
        print(f"❌ {args.sport} not yet implemented")
        sys.exit(1)

if __name__ == "__main__":
    main()
