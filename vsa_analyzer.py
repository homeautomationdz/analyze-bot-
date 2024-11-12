import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib
matplotlib.use('Agg')

class VSAAnalyzer:
    def __init__(self):
        self.setup_vsa_parameters()
        
    def setup_vsa_parameters(self):
        self.vsa_thresholds = {
            'high_volume_std': 1.5,  # More sensitive volume threshold
            'low_volume_std': -1.5,
            'range_multiplier': 1.2,
            'wick_ratio': 0.6,
            'body_ratio': 0.4
        }

    def analyze_candles(self, df):
        # Calculate basic metrics
        df['Range'] = df['high'] - df['low']
        df['BodyRange'] = abs(df['close'] - df['open'])
        df['UpperWick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['LowerWick'] = df[['open', 'close']].min(axis=1) - df['low']
        
        # Volume thresholds
        high_volume_threshold = df['volume'].mean() + self.vsa_thresholds['high_volume_std'] * df['volume'].std()
        low_volume_threshold = df['volume'].mean() + self.vsa_thresholds['low_volume_std'] * df['volume'].std()
        
        # Identify candle types
        df['VSA_Pattern'] = df.apply(
            lambda x: self._identify_candle_type(
                x, 
                high_volume_threshold, 
                low_volume_threshold,
                df['Range'].mean()
            ), 
            axis=1
        )
        
        return df
    def analyze_market_conditions(self, df, timeframe):
        # Single analysis pass
        vsa_df = self.analyze_candles(df)
        signals = {
            'vsa': self.analyze_vsa_patterns(vsa_df),
            'support': self.analyze_support_volume(vsa_df),
            'wick': self.analyze_wick_rejection(df, timeframe)
        }
        return signals, vsa_df

    def analyze_support_volume(self, df):
        # Calculate support levels using rolling min
        df['support'] = df['low'].rolling(window=20).min()
        
        # Calculate wick sizes
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        df['wick_ratio'] = df['lower_wick'] / (df['high'] - df['low'])
        
        # Calculate volume thresholds
        volume_mean = df['volume'].mean()
        volume_std = df['volume'].std()
        high_volume_threshold = volume_mean + (2 * volume_std)
        
        # Check for high volume at support with large lower wick
        support_signals = []
        
        # Look at the latest candle
        latest = df.iloc[-1]
        if (abs(latest['low'] - latest['support']) / latest['support'] < 0.003 and  # Price near support (0.3% threshold)
            latest['volume'] > high_volume_threshold and  # High volume
            latest['wick_ratio'] > 0.3):  # Large lower wick (30% of candle)
            
            support_signals.append({
                'type': 'HIGH_VOLUME_SUPPORT_REJECTION',
                'price': latest['close'],
                'volume': latest['volume'],
                'wick_ratio': latest['wick_ratio'],
                'support_level': latest['support']
            })
        
        return support_signals
    def analyze_wick_rejection(self, df, timeframe):
        signals = []
        
        # Calculate support levels using previous lows
        df['support'] = df['low'].rolling(window=20).min()
        
        # Calculate wick sizes
        df['upper_wick'] = df['high'] - df['close']
        df['lower_wick'] = df['close'] - df['low']
        df['body_size'] = abs(df['open'] - df['close'])
        
        # Wick rejection criteria
        df['significant_lower_wick'] = (df['lower_wick'] > df['body_size'] * 2) & (df['lower_wick'] > df['upper_wick'] * 3)
        
        latest = df.iloc[-1]
        if latest['significant_lower_wick']:
            # Check if price is near support
            if abs(latest['low'] - latest['support']) / latest['low'] < 0.01:  # 1% threshold
                signals.append({
                    'type': 'WICK_REJECTION_BUY',
                    'price': latest['close'],
                    'support_level': latest['support'],
                    'timeframe': timeframe,
                    'wick_size': latest['lower_wick'],
                    'strength': 'strong'
                })
        
        return signals

    def plot_vsa_analysis(self, df, market_name):
        plt.figure(figsize=(15, 8))
        
        # Plot price
        plt.subplot(2, 1, 1)
        plt.plot(df.index, df['close'], label='Price')
        
        # Plot patterns
        patterns = {
            'High_Volume_Bullish_Thrust': ('^', 'green'),
            'High_Volume_Bearish_Thrust': ('v', 'red'),
            'Low_Volume_Wide_Range': ('s', 'orange'),
            'High_Volume_Narrow_Range': ('d', 'purple'),
            'No_Demand': ('x', 'gray')
        }
        
        for pattern, (marker, color) in patterns.items():
            mask = df['VSA_Pattern'] == pattern
            if mask.any():
                plt.scatter(
                    df[mask].index,
                    df[mask]['close'],
                    marker=marker,
                    color=color,
                    label=pattern,
                    s=100
                )
        
        plt.title(f'VSA Analysis for {market_name}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot volume
        plt.subplot(2, 1, 2)
        plt.bar(df.index, df['volume'], color='blue', alpha=0.3, label='Volume')
        plt.title('Volume')
        
        plt.tight_layout()
        return plt.gcf()
       
    def _identify_candle_type(self, row, high_vol, low_vol, avg_range):
        if row['Range'] > avg_range:
            if row['volume'] > high_vol:
                if row['close'] > row['open']:
                    return 'High_Volume_Bullish_Thrust'
                else:
                    return 'High_Volume_Bearish_Thrust'
            elif row['volume'] < low_vol:
                return 'Low_Volume_Wide_Range'
        else:
            if row['volume'] > high_vol:
                return 'High_Volume_Narrow_Range'
            elif row['volume'] < low_vol:
                return 'No_Demand'
        return 'Neutral'
