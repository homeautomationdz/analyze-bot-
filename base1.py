import os
import sqlite3
import logging
import threading
import urllib.request
import requests
import tkinter as tk
from tkinter import messagebox
import time
from datetime import datetime
import talib
import pandas as pd
import ccxt
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

class BaseScanner(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.setup_core_components()
        self.initialize_signal_processor()
    def setup_database_config(self):
        db_path = os.path.dirname(os.path.abspath(__file__))
        self.db_connect = os.path.join(db_path, "connection.db")
        self.db_signals = os.path.join(db_path, "Signals.db")

    def setup_core_components(self):
        # Window configuration
        if self.master:
            self.master.title("Market Scanner")
            self.master.configure(bg="#000000")
        
        # System configuration
        self.tm = int(datetime.now().timestamp())
        self.fixed = 1929999999
        self.setup_logger('logs/scanner.log')
        
        # Threading
        self.lock = threading.Lock()
        
        # Authentication
        self.auth = None
        self.api_key = None
        self.secret_key = None
        self.phrase = None
        self.tel_id = None
        self.bot_token = None
        
        # State
        self.binance = None
        self.valid = False
        self.counter = 0
        
        # Database configuration
        self.setup_database_config()
        
        # Enhanced database configuration
        self.db_config = {
            'pool_size': 5,
            'timeout': 30,
            'retry_count': 3
        }
        
        # Signal processing configuration
        self.signal_config = {
            'min_volume': 100000,
            'min_score': 3.0,
            'confirmation_count': 2,
            'timeframe_weights': {
                '1m': 0.5, '5m': 0.7, '15m': 0.8,
                '1h': 1.0, '4h': 1.2, '1d': 1.5
            }
        }
        
        # Rate limiting configuration
        self.rate_config = {
            'max_calls': 1200,
            'window': 60,
            'buffer': 0.9
        }

       # Add to BaseScanner class
    def setup_logger(self, log_file):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(file_handler)

    def create_binance_object(self):
        if not self.valid:
            return None

        exchange_name = self.choose_listex.get()
        
        if not self.api_key or not self.secret_key:
            return None

        try:
            exchange_configs = {
                'CoinEx': ccxt.coinex,
                'BingX': ccxt.bingx,
                'okx': ccxt.okx,
                'Binance': ccxt.binance,
                'Kucoin': ccxt.kucoin,
                'Bybit': ccxt.bybit
            }

            if exchange_name not in exchange_configs:
                return None

            config = {
                'apiKey': self.api_key,
                'secret': self.secret_key,
                'enableRateLimit': True,
                'options': {
                    'adjustForTimeDifference': True,
                    'recvWindow': 5000,
                }
            }

            if exchange_name in ['okx', 'Kucoin']:
                config['password'] = self.phrase

            self.binance = exchange_configs[exchange_name](config)
            print(f"Created exchange object for {exchange_name}: {self.binance}")
            print(f"Type of binance object: {type(self.binance)}")
            return self.binance

        except Exception as e:
            self.logger.error(f"Error creating exchange object: {e}")
            return None

    def sql_operations(self, operation, db_file, table_name, **kwargs):
        try:
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()

            if operation == 'create':
                if table_name == 'userinfo':
                    sql_query = """
                    CREATE TABLE IF NOT EXISTS userinfo (
                        name TEXT,
                        key TEXT,
                        secret TEXT,
                        phrase TEXT,
                        tel_id TEXT,
                        tel_token TEXT
                    )
                    """
                elif table_name == 'Signals':
                    sql_query = """
                    CREATE TABLE IF NOT EXISTS Signals (
                        market TEXT,
                        timeframe TEXT,
                        signal_type TEXT,
                        price REAL,
                        volume_trend TEXT,
                        vwap REAL,
                        rsi REAL,
                        timestamp TEXT
                    )
                    """
                cursor.execute(sql_query)

            elif operation == 'insert':
                columns = ', '.join(kwargs.keys())
                placeholders = ', '.join('?' for _ in kwargs)
                values = tuple(kwargs.values())
                sql_query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
                cursor.execute(sql_query, values)

            elif operation == 'fetch':
                sql_query = f"SELECT * FROM {table_name}"
                cursor.execute(sql_query)
                columns = [description[0] for description in cursor.description]
                result = [dict(zip(columns, row)) for row in cursor.fetchall()]
                return result

            conn.commit()
        except Exception as e:
            self.logger.error(f"Database error: {e}")
        finally:
            if conn:
                conn.close()


    def check_internet_connection(self):
        try:
            urllib.request.urlopen('https://api.binance.com/api/v3/ping', timeout=1)
            return True
        except:
            return False

    def run(self, func):
        thread = threading.Thread(target=func, daemon=True)
        thread.start()

    def setup_core_components(self):
        # Window configuration
        if self.master:
            self.master.title("Market Scanner")
            self.master.configure(bg="#000000")
            
        # System configuration
        self.tm = int(datetime.now().timestamp())
        self.fixed = 1929999999
        self.setup_logger('logs/scanner.log')
        
        # Threading
        self.lock = threading.Lock()
        
        # Authentication
        self.auth = None
        self.api_key = None
        self.secret_key = None
        self.phrase = None
        self.tel_id = None
        self.bot_token = None
        
        # State
        self.binance = None
        self.valid = False
        self.counter = 0
        
        # Database configuration
        self.setup_database_config()
        
        # Enhanced database configuration
        self.db_config = {
            'pool_size': 5,
            'timeout': 30,
            'retry_count': 3
        }
        
        # Signal processing configuration
        self.signal_config = {
            'min_volume': 100000,
            'min_score': 3.0,
            'confirmation_count': 2,
            'timeframe_weights': {
                '1m': 0.5, '5m': 0.7, '15m': 0.8,
                '1h': 1.0, '4h': 1.2, '1d': 1.5
            }
        }
        
        # Rate limiting configuration
        self.rate_config = {
            'max_calls': 1200,
            'window': 60,
            'buffer': 0.9
        }
    def initialize_signal_processor(self):
        self.signal_thresholds = {
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'volume_multiplier': 2.5,
            'atr_multiplier': 1.5,
            'trend_strength': 25
        }
        
        self.signal_weights = {
            'volume': 0.3,
            'trend': 0.3,
            'momentum': 0.2,
            'pattern': 0.2
        }
        
        self.timeframe_weights = {
            '1m': 0.5,
            '5m': 0.7,
            '15m': 0.8,
            '1h': 1.0,
            '4h': 1.2,
            '1d': 1.5
        }
        
    def calculate_technical_indicators(self, df):
        df['rsi'] = talib.RSI(df['close'])
        df['ema_20'] = talib.EMA(df['close'], timeperiod=20)
        df['ema_50'] = talib.EMA(df['close'], timeperiod=50)
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        return df

    def process_market_signal(self, market, timeframe, signal_data):
        try:
            # Validate signal data
            if not self.validate_signal_data(signal_data):
                return None
                
            # Apply timeframe weight
            signal_data['score'] *= self.signal_config['timeframe_weights'].get(timeframe, 1.0)
            
            # Check for signal confirmation
            confirmation = self.check_signal_confirmation(market, signal_data)
            if confirmation['confirmed']:
                signal_data.update(confirmation)
                
            # Store signal if valid
            if signal_data['score'] >= self.signal_config['min_score']:
                self.store_signal(market, timeframe, signal_data)
                self.update_performance_metrics(market, signal_data)
                return signal_data
                
            return None
            
        except Exception as e:
            self.logger.error(f"Signal processing error: {e}")
            return None

    def validate_signal_data(self, signal_data):
        required_fields = ['type', 'price', 'score', 'timestamp']
        return all(field in signal_data for field in required_fields)

    def check_signal_confirmation(self, market, signal_data):
        signal_type = signal_data['type']
        current_time = time.time()
        
        if market not in self.signal_processor['confirmations']:
            self.signal_processor['confirmations'][market] = {}
            
        market_confirms = self.signal_processor['confirmations'][market]
        
        if signal_type not in market_confirms:
            market_confirms[signal_type] = {
                'count': 1,
                'first_seen': current_time,
                'signals': [signal_data]
            }
        else:
            market_confirms[signal_type]['count'] += 1
            market_confirms[signal_type]['signals'].append(signal_data)
            
        confirmation_data = market_confirms[signal_type]
        is_confirmed = confirmation_data['count'] >= self.signal_config['confirmation_count']
        
        return {
            'confirmed': is_confirmed,
            'confirmation_count': confirmation_data['count'],
            'confirmation_time': current_time - confirmation_data['first_seen']
        }

    def update_performance_metrics(self, market, signal_data):
        if market not in self.signal_processor['performance_metrics']:
            self.signal_processor['performance_metrics'][market] = {
                'total_signals': 0,
                'successful_signals': 0,
                'average_score': 0
            }
            
        metrics = self.signal_processor['performance_metrics'][market]
        metrics['total_signals'] += 1
        metrics['average_score'] = (
            (metrics['average_score'] * (metrics['total_signals'] - 1) + signal_data['score'])
            / metrics['total_signals']
        )

    def get_market_performance(self, market):
        return self.signal_processor['performance_metrics'].get(market, {})

    def analyze_pattern_sequence(self, df):
        patterns = {
            'double_bottom': self.detect_double_bottom,
            'triple_bottom': self.detect_triple_bottom,
            'bull_flag': self.detect_bull_flag,
            'cup_handle': self.detect_cup_handle
        }
        
        pattern_signals = []
        for pattern_name, detector in patterns.items():
            if result := detector(df):
                pattern_signals.append({
                    'type': pattern_name,
                    'strength': result['strength'],
                    'score': result['score'],
                    'confirmation': result['confirmation']
                })
                
        return pattern_signals

    def detect_double_bottom(self, df):
        try:
            # Get local minima
            df['min'] = df['low'].rolling(window=5, center=True).min()
            bottoms = df[df['low'] == df['min']].index.tolist()
            
            if len(bottoms) >= 2:
                bottom1_price = df.loc[bottoms[-2], 'low']
                bottom2_price = df.loc[bottoms[-1], 'low']
                
                # Check if bottoms are within 2% of each other
                if abs(bottom1_price - bottom2_price) / bottom1_price < 0.02:
                    return {
                        'strength': 'strong',
                        'score': 0.8,
                        'confirmation': True
                    }
                    
            return None
            
        except Exception as e:
            self.logger.error(f"Double bottom detection error: {e}")
            return None

    def detect_bull_flag(self, df):
        try:
            # Calculate pole (strong upward move)
            df['returns'] = df['close'].pct_change()
            df['upward_move'] = df['returns'].rolling(window=5).sum()
            
            # Check for consolidation after upward move
            df['volatility'] = df['returns'].rolling(window=5).std()
            
            if (df['upward_move'].iloc[-6] > 0.1 and  # 10% upward move
                df['volatility'].iloc[-5:].mean() < df['volatility'].iloc[-10:-5].mean()):
                return {
                    'strength': 'moderate',
                    'score': 0.6,
                    'confirmation': True
                }
                
            return None
            
        except Exception as e:
            self.logger.error(f"Bull flag detection error: {e}")
            return None

    def analyze_volume_patterns(self, df):
        volume_signals = []
        
        # Volume Spread Analysis (VSA)
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['spread'] = df['high'] - df['low']
        df['close_loc'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Effort vs Result
        if df['volume'].iloc[-1] > df['volume_ma'].iloc[-1] * 2:  # High volume
            if df['spread'].iloc[-1] < df['spread'].rolling(window=20).mean().iloc[-1]:  # Low spread
                volume_signals.append({
                    'type': 'VSA_COMPRESSION',
                    'strength': 'high',
                    'score': 0.7
                })
                
        return volume_signals

    def detect_divergences(self, df):
        divergences = []
        
        # Calculate RSI
        df['rsi'] = talib.RSI(df['close'])
        
        # Look for price/RSI divergences
        price_lows = df['low'].rolling(window=5, center=True).min()
        rsi_lows = df['rsi'].rolling(window=5, center=True).min()
        
        if (price_lows.iloc[-1] < price_lows.iloc[-2] and 
            rsi_lows.iloc[-1] > rsi_lows.iloc[-2]):
            divergences.append({
                'type': 'BULLISH_DIVERGENCE',
                'strength': 'strong',
                'score': 0.9
            })
            
        return divergences

    def combine_pattern_signals(self, pattern_signals, volume_signals, divergences):
        combined_score = 0
        signal_count = 0
        
        for signal_list in [pattern_signals, volume_signals, divergences]:
            for signal in signal_list:
                combined_score += signal['score']
                signal_count += 1
                
        if signal_count > 0:
            return {
                'total_score': combined_score / signal_count,
                'pattern_count': len(pattern_signals),
                'volume_confirmations': len(volume_signals),
                'divergence_confirmations': len(divergences)
            }
            
        return None


    def check_rate_limit(self):
        current_time = time.time()
        if current_time - self.rate_limits['last_reset'] >= 60:
            self.rate_limits['api_calls'] = 0
            self.rate_limits['last_reset'] = current_time
            
        if self.rate_limits['api_calls'] >= self.rate_limits['max_calls_per_minute']:
            sleep_time = 60 - (current_time - self.rate_limits['last_reset'])
            if sleep_time > 0:
                time.sleep(sleep_time)


    def con(self, file_name):
        try:
            db1 = sqlite3.connect('connection.db')
            c1 = db1.cursor()
            c1.execute('select * from userinfo')
            data = c1.fetchone()

            if data:
                if self.tm < self.fixed:
                    print('valid')
                    return True
                else:
                    self.master.destroy()
                    return False
            else:
                self.master.destroy()
                return None
        finally:
            if 'db1' in locals():
                db1.close()

    def check_rate_limit(self):
        current_time = time.time()
        if current_time - self.rate_limits['last_reset'] >= self.rate_limits['cooldown_period']:
            self.rate_limits['api_calls'] = 0
            self.rate_limits['last_reset'] = current_time
        
        if self.rate_limits['api_calls'] >= self.rate_limits['max_calls_per_minute']:
            sleep_time = self.rate_limits['cooldown_period'] - (current_time - self.rate_limits['last_reset'])
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        self.rate_limits['api_calls'] += 1
