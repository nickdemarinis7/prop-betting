#!/usr/bin/env python3
"""
ROI Tracking System
Track betting performance and returns over time
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json


class ROITracker:
    """Track betting ROI and performance metrics"""
    
    def __init__(self, tracking_file='roi_tracking.json'):
        self.tracking_file = tracking_file
        self.data = self.load_tracking_data()
    
    def load_tracking_data(self):
        """Load historical tracking data"""
        if os.path.exists(self.tracking_file):
            with open(self.tracking_file, 'r') as f:
                return json.load(f)
        return {
            'bets': [],
            'daily_summary': {},
            'overall_stats': {
                'total_bets': 0,
                'total_units_wagered': 0,
                'total_units_won': 0,
                'total_units_lost': 0,
                'net_units': 0,
                'roi': 0,
                'win_rate': 0
            }
        }
    
    def save_tracking_data(self):
        """Save tracking data to file"""
        with open(self.tracking_file, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def add_bet(self, date, player, threshold, units, odds, result=None, actual_pts=None):
        """Add a bet to tracking"""
        bet = {
            'date': date,
            'player': player,
            'threshold': threshold,
            'units': units,
            'odds': odds,
            'result': result,  # 'win', 'loss', or None (pending)
            'actual_pts': actual_pts,
            'profit': None
        }
        
        # Calculate profit if result is known
        if result == 'win':
            if odds < 0:
                bet['profit'] = units * (100 / abs(odds))
            else:
                bet['profit'] = units * (odds / 100)
        elif result == 'loss':
            bet['profit'] = -units
        
        self.data['bets'].append(bet)
        self.save_tracking_data()
    
    def update_bet_result(self, bet_index, result, actual_pts):
        """Update bet result after game completes"""
        if bet_index >= len(self.data['bets']):
            return False
        
        bet = self.data['bets'][bet_index]
        bet['result'] = result
        bet['actual_pts'] = actual_pts
        
        # Calculate profit
        if result == 'win':
            if bet['odds'] < 0:
                bet['profit'] = bet['units'] * (100 / abs(bet['odds']))
            else:
                bet['profit'] = bet['units'] * (bet['odds'] / 100)
        elif result == 'loss':
            bet['profit'] = -bet['units']
        
        self.save_tracking_data()
        return True
    
    def calculate_overall_stats(self):
        """Calculate overall performance statistics"""
        completed_bets = [b for b in self.data['bets'] if b['result'] is not None]
        
        if not completed_bets:
            return self.data['overall_stats']
        
        total_bets = len(completed_bets)
        wins = len([b for b in completed_bets if b['result'] == 'win'])
        
        total_wagered = sum(b['units'] for b in completed_bets)
        total_profit = sum(b['profit'] for b in completed_bets if b['profit'] is not None)
        
        stats = {
            'total_bets': total_bets,
            'wins': wins,
            'losses': total_bets - wins,
            'win_rate': wins / total_bets if total_bets > 0 else 0,
            'total_units_wagered': total_wagered,
            'net_units': total_profit,
            'roi': (total_profit / total_wagered * 100) if total_wagered > 0 else 0,
            'avg_bet_size': total_wagered / total_bets if total_bets > 0 else 0,
            'avg_profit_per_bet': total_profit / total_bets if total_bets > 0 else 0
        }
        
        self.data['overall_stats'] = stats
        self.save_tracking_data()
        
        return stats
    
    def get_daily_summary(self, date):
        """Get summary for a specific date"""
        date_bets = [b for b in self.data['bets'] if b['date'] == date]
        
        if not date_bets:
            return None
        
        completed = [b for b in date_bets if b['result'] is not None]
        
        if not completed:
            return {
                'date': date,
                'total_bets': len(date_bets),
                'pending': len(date_bets),
                'completed': 0
            }
        
        wins = len([b for b in completed if b['result'] == 'win'])
        total_profit = sum(b['profit'] for b in completed if b['profit'] is not None)
        total_wagered = sum(b['units'] for b in completed)
        
        return {
            'date': date,
            'total_bets': len(date_bets),
            'completed': len(completed),
            'pending': len(date_bets) - len(completed),
            'wins': wins,
            'losses': len(completed) - wins,
            'win_rate': wins / len(completed) if completed else 0,
            'net_units': total_profit,
            'roi': (total_profit / total_wagered * 100) if total_wagered > 0 else 0
        }
    
    def print_summary(self):
        """Print performance summary"""
        stats = self.calculate_overall_stats()
        
        print("=" * 80)
        print("💰 ROI TRACKING SUMMARY")
        print("=" * 80)
        
        print(f"\n📊 Overall Performance:")
        print(f"   Total Bets: {stats['total_bets']}")
        print(f"   Wins: {stats.get('wins', 0)} | Losses: {stats.get('losses', 0)}")
        print(f"   Win Rate: {stats['win_rate']:.1%}")
        
        print(f"\n💵 Financial Performance:")
        print(f"   Total Units Wagered: {stats['total_units_wagered']:.2f}u")
        print(f"   Net Profit/Loss: {stats['net_units']:+.2f}u")
        print(f"   ROI: {stats['roi']:+.1f}%")
        print(f"   Avg Bet Size: {stats.get('avg_bet_size', 0):.2f}u")
        print(f"   Avg Profit/Bet: {stats.get('avg_profit_per_bet', 0):+.2f}u")
        
        # Show recent performance
        print(f"\n📅 Recent Performance (Last 7 Days):")
        for i in range(7):
            date = (datetime.now() - timedelta(days=i)).strftime('%Y%m%d')
            summary = self.get_daily_summary(date)
            
            if summary and summary['completed'] > 0:
                print(f"   {date}: {summary['wins']}-{summary['losses']} " +
                      f"({summary['win_rate']:.0%}), " +
                      f"{summary['net_units']:+.2f}u, " +
                      f"ROI: {summary['roi']:+.1f}%")
        
        print("\n" + "=" * 80)


def main():
    """Example usage"""
    tracker = ROITracker()
    
    # Example: Add some test bets
    # tracker.add_bet('20260409', 'Alperen Sengun', '25+', 0.75, -150, 'win', 33.6)
    # tracker.add_bet('20260409', 'Aaron Nesmith', '20+', 0.5, +120, 'loss', 18.2)
    
    # Print summary
    tracker.print_summary()
    
    print("\n💡 Usage:")
    print("   1. After making bets: tracker.add_bet(date, player, threshold, units, odds)")
    print("   2. After games complete: tracker.update_bet_result(index, 'win'/'loss', actual_pts)")
    print("   3. View performance: tracker.print_summary()")
    print("\n   Track file: roi_tracking.json")


if __name__ == "__main__":
    main()
