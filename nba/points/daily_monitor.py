#!/usr/bin/env python3
"""
Daily Monitoring Script
Automated checks for model health and performance
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json


class DailyMonitor:
    """Monitor model performance and health"""
    
    def __init__(self):
        self.alerts = []
        self.metrics = {}
    
    def check_prediction_file(self, date_str=None):
        """Check if today's predictions exist"""
        if date_str is None:
            date_str = datetime.now().strftime('%Y%m%d')
        
        filename = f'predictions_production_{date_str}.csv'
        
        if not os.path.exists(filename):
            self.alerts.append(f"❌ Missing predictions for {date_str}")
            return None
        
        df = pd.read_csv(filename)
        self.metrics['prediction_count'] = len(df)
        self.metrics['top_plays'] = len(df[df['Type'] == 'TOP PLAY'])
        
        return df
    
    def check_model_quality(self, df):
        """Check prediction quality metrics"""
        if df is None or df.empty:
            return
        
        top_plays = df[df['Type'] == 'TOP PLAY']
        
        # Average quality score
        avg_quality = top_plays['Quality'].mean()
        self.metrics['avg_quality'] = avg_quality
        
        if avg_quality < 70:
            self.alerts.append(f"⚠️  Low average quality: {avg_quality:.1f}/100")
        
        # Average ladder value
        avg_ladder = top_plays['Ladder_Value'].mean()
        self.metrics['avg_ladder'] = avg_ladder
        
        if avg_ladder < 60:
            self.alerts.append(f"⚠️  Low ladder value: {avg_ladder:.1f}/100")
        
        # Tier distribution
        tier_dist = top_plays['Tier'].value_counts().to_dict()
        self.metrics['tier_distribution'] = tier_dist
        
        if tier_dist.get(1, 0) == 0 and tier_dist.get(2, 0) == 0:
            self.alerts.append("⚠️  No Tier 1 or Tier 2 plays")
        
        # Variance check
        avg_std = top_plays['StdDev'].mean()
        self.metrics['avg_std_dev'] = avg_std
        
        if avg_std > 7.0:
            self.alerts.append(f"⚠️  High average variance: {avg_std:.1f}")
    
    def check_usage_boost_opportunities(self, df):
        """Check for usage boost plays"""
        if df is None or df.empty:
            return
        
        top_plays = df[df['Type'] == 'TOP PLAY']
        usage_boost_plays = top_plays[top_plays['Tmts_Out'] >= 3]
        
        self.metrics['usage_boost_plays'] = len(usage_boost_plays)
        
        if len(usage_boost_plays) > 0:
            print(f"\n✅ {len(usage_boost_plays)} usage boost opportunities identified")
    
    def check_data_freshness(self):
        """Check if data files are up to date"""
        # Check prediction file age
        today = datetime.now().strftime('%Y%m%d')
        filename = f'predictions_production_{today}.csv'
        
        if os.path.exists(filename):
            file_time = datetime.fromtimestamp(os.path.getmtime(filename))
            age_hours = (datetime.now() - file_time).total_seconds() / 3600
            
            self.metrics['prediction_age_hours'] = age_hours
            
            if age_hours > 24:
                self.alerts.append(f"⚠️  Predictions are {age_hours:.1f} hours old")
    
    def check_historical_performance(self, days_back=7):
        """Check performance over last N days"""
        hit_rates = []
        
        for i in range(1, days_back + 1):
            date = datetime.now() - timedelta(days=i)
            date_str = date.strftime('%Y%m%d')
            
            # This would need actual validation results
            # For now, just check if file exists
            filename = f'predictions_production_{date_str}.csv'
            if os.path.exists(filename):
                hit_rates.append(1)  # Placeholder
        
        self.metrics['days_with_predictions'] = len(hit_rates)
        
        if len(hit_rates) < days_back * 0.7:
            self.alerts.append(f"⚠️  Only {len(hit_rates)}/{days_back} days have predictions")
    
    def generate_report(self):
        """Generate monitoring report"""
        print("=" * 80)
        print("📊 DAILY MONITORING REPORT")
        print(f"   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        # Metrics
        print("\n📈 Key Metrics:")
        for key, value in self.metrics.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.2f}")
            elif isinstance(value, dict):
                print(f"   {key}: {value}")
            else:
                print(f"   {key}: {value}")
        
        # Alerts
        if self.alerts:
            print("\n⚠️  Alerts:")
            for alert in self.alerts:
                print(f"   {alert}")
        else:
            print("\n✅ No alerts - all systems normal")
        
        # Recommendations
        print("\n💡 Recommendations:")
        if self.metrics.get('avg_quality', 100) < 80:
            print("   • Review model parameters")
            print("   • Check for data quality issues")
        
        if self.metrics.get('usage_boost_plays', 0) > 5:
            print("   • High injury opportunity - focus on usage boost plays")
        
        if len(self.alerts) > 3:
            print("   • Multiple alerts - investigate model health")
        
        print("\n" + "=" * 80)
    
    def save_report(self):
        """Save report to file"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'metrics': self.metrics,
            'alerts': self.alerts
        }
        
        filename = f'monitoring_report_{datetime.now().strftime("%Y%m%d")}.json'
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n💾 Report saved to {filename}")


def main():
    """Run daily monitoring"""
    monitor = DailyMonitor()
    
    # Run checks
    print("🔍 Running daily checks...\n")
    
    df = monitor.check_prediction_file()
    monitor.check_model_quality(df)
    monitor.check_usage_boost_opportunities(df)
    monitor.check_data_freshness()
    monitor.check_historical_performance(days_back=7)
    
    # Generate report
    monitor.generate_report()
    monitor.save_report()


if __name__ == "__main__":
    main()
