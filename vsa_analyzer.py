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
