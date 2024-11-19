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

class BaseScanner(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master  # Set master first
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
            self.master.configure(bg="#E6E6E6")
        
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
    def setup_power_tools(self):
        self.setup_signal_builder()
        self.setup_risk_calculator()
        self.setup_position_optimizer()
        self.setup_correlation_tracker()

    def setup_signal_builder(self):
        self.signal_templates = {}
        self.custom_signals = {}
        self.load_signal_templates()

    def setup_risk_calculator(self):
        self.risk_params = {
            'max_position_size': 0.02,
            'max_portfolio_risk': 0.05,
            'correlation_threshold': 0.7
        }
    def setup_market_regime(self):
        self.market_regimes = {
            'trend_following': {
                'adx_threshold': 25,
                'volatility_threshold': 1.5
            },
            'mean_reversion': {
                'rsi_threshold': 30,
                'bollinger_threshold': 2.0
            },
            'breakout': {
                'volume_threshold': 2.0,
                'range_threshold': 1.5
            }
        }
    def setup_ml_components(self):
        self.ml_config = {
            'pattern_recognition': {
                'window_size': 20,
                'confidence_threshold': 0.8,
                'min_pattern_quality': 0.7
            },
            'trend_prediction': {
                'lookback_period': 100,
                'forecast_horizon': 5
            }
        }
    def initialize_ml_models(self):
        self.ml_models = {
            'pattern_recognition': self.setup_pattern_recognition_model(),
            'trend_prediction': self.setup_trend_prediction_model(),
            'sentiment_analysis': self.setup_sentiment_model()
        }
                
    def setup_backtesting_engine(self):
        self.backtest_config = {
            'lookback_period': 500,
            'timeframes': ['15m', '1h', '4h', '1d'],
            'metrics': {
                'win_rate': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0
            },
            'optimization': {
                'enabled': True,
                'parameters': ['stop_loss', 'take_profit', 'entry_confirmation']
            }
        }
    def initialize_advanced_backtesting(self):
        self.backtest_engine = {
            'multi_timeframe': self.setup_mtf_backtest(),
            'monte_carlo': self.setup_monte_carlo(),
            'optimization': self.setup_walk_forward()
        }

    def setup_trade_journal(self):
        self.journal_config = {
            'trade_metrics': ['entry', 'exit', 'pnl', 'risk_reward', 'setup_quality'],
            'market_metrics': ['volume_profile', 'session_data', 'market_regime'],
            'performance_tracking': ['daily_pnl', 'win_rate', 'avg_trade']
        }

    def integrate_core_components(self):
        self.signal_processor = {
            'confirmations': {},
            'performance_metrics': {},
            'market_regimes': {},
            'trade_journal': {},
            'backtest_results': {}
        }
        
        self.risk_manager = {
            'position_sizing': self.calculate_position_size,
            'risk_metrics': self.calculate_risk_level,
            'exposure_limits': self.calculate_exposure
        }
        
        self.market_analyzer = {
            'order_flow': self.analyze_order_flow,
            'session_data': self.detect_session_momentum,
            'regime_detection': self.analyze_market_regime
        }


    def setup_logger(self, log_file):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        
        # File handler with UTF-8 encoding
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        
        # Console handler with UTF-8 encoding
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
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
                'Okex': ccxt.okex,
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

            if exchange_name in ['Okex', 'Kucoin']:
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
    def analyze_momentum(self, df):
        momentum_data = {
            'rsi_momentum': self.classify_rsi_momentum(df['rsi'].iloc[-1]),
            'macd_momentum': self.classify_macd_momentum(df['macd'].iloc[-1], df['macd_signal'].iloc[-1]),
            'trend_strength': self.classify_adx_strength(df['adx'].iloc[-1])
        }
        
        return self.calculate_momentum_score(momentum_data)
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
    def analyze_market_performance(self, market, timeframe):
        df = self.fetch_market_data(market, timeframe)
        performance_metrics = {
            'technical_analysis': self.calculate_technical_indicators(df),
            'volume_analysis': self.analyze_volume_profile(df),
            'pattern_analysis': self.detect_complex_patterns(df),
            'sentiment': self.analyze_market_sentiment(market),
            'institutional_activity': self.detect_institutional_activity(market)
        }
        
        return {
            'overall_score': self.calculate_performance_score(performance_metrics),
            'metrics': performance_metrics,
            'recommendations': self.generate_trading_recommendations(performance_metrics)
        }

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
    def calculate_vwap_levels(self, df):
        df['vwap'] = (df['volume'] * ((df['high'] + df['low'] + df['close']) / 3)).cumsum() / df['volume'].cumsum()
        df['vwap_upper'] = df['vwap'] * 1.01  # 1% upper band
        df['vwap_lower'] = df['vwap'] * 0.99  # 1% lower band
        return df

    def calculate_pivot_points(self, df):
        pivot = (df['high'].iloc[-1] + df['low'].iloc[-1] + df['close'].iloc[-1]) / 3
        r1 = 2 * pivot - df['low'].iloc[-1]
        s1 = 2 * pivot - df['high'].iloc[-1]
        r2 = pivot + (df['high'].iloc[-1] - df['low'].iloc[-1])
        s2 = pivot - (df['high'].iloc[-1] - df['low'].iloc[-1])
        
        return {
            'pivot': pivot,
            'r1': r1, 'r2': r2,
            's1': s1, 's2': s2
        }

    def calculate_fibonacci_levels(self, df):
        high = df['high'].max()
        low = df['low'].min()
        diff = high - low
        
        return {
            'level_0': high,
            'level_236': high - (diff * 0.236),
            'level_382': high - (diff * 0.382),
            'level_500': high - (diff * 0.500),
            'level_618': high - (diff * 0.618),
            'level_100': low
        }

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
    def detect_complex_patterns(self, df):
        patterns = {
            'triple_bottom': self.detect_triple_bottom(df),
            'head_shoulders': self.detect_head_shoulders(df),
            'cup_handle': self.detect_cup_handle(df)
        }
        return {k: v for k, v in patterns.items() if v is not None}
    def detect_head_shoulders(self, df):
        try:
            # Find potential shoulders and head
            peaks = self.find_peaks(df['high'], distance=10)
            troughs = self.find_peaks(-df['low'], distance=10)
            
            pattern_data = {
                'left_shoulder': self.validate_shoulder(df, peaks, 0),
                'head': self.validate_head(df, peaks, troughs),
                'right_shoulder': self.validate_shoulder(df, peaks, -1),
                'neckline': self.calculate_neckline(df, troughs)
            }
            
            if self.validate_hs_pattern(pattern_data):
                return {
                    'type': 'HEAD_AND_SHOULDERS',
                    'probability': self.calculate_pattern_probability(pattern_data),
                    'target': self.calculate_hs_target(pattern_data),
                    'validation': pattern_data
                }
            return None
        except Exception as e:
            self.logger.error(f"Head and Shoulders detection error: {e}")
            return None

    def validate_pattern_completion(self, df, pattern_type):
        validation_metrics = {
            'volume_confirmation': self.check_volume_confirmation(df),
            'price_action': self.analyze_price_action_confirmation(df),
            'momentum_alignment': self.check_momentum_alignment(df),
            'time_symmetry': self.check_pattern_symmetry(df)
        }
        
        completion_score = sum(validation_metrics.values()) / len(validation_metrics)
        return {
            'completion_score': completion_score,
            'metrics': validation_metrics,
            'probability': self.calculate_completion_probability(completion_score)
        }


    def setup_deep_learning_models(self):
        self.dl_models = {
            'pattern_recognition': self.create_pattern_recognition_model(),
            'trend_prediction': self.create_trend_prediction_model(),
            'sentiment_analyzer': self.create_sentiment_model()
        }
        return self.dl_models

    def create_pattern_recognition_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(64, input_shape=(30, 5)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def predict_market_trends(self, market, timeframe):
        df = self.fetch_market_data(market, timeframe)
        features = self.prepare_prediction_features(df)
        
        predictions = {
            'short_term': self.predict_trend_timeframe(features, timeframe='short'),
            'medium_term': self.predict_trend_timeframe(features, timeframe='medium'),
            'long_term': self.predict_trend_timeframe(features, timeframe='long')
        }
        
        return {
            'predictions': predictions,
            'confidence_scores': self.calculate_prediction_confidence(predictions),
            'market_context': self.get_market_context(df)
        }

    def analyze_market_sentiment(self, market):
        news_data = self.fetch_market_news(market)
        social_data = self.fetch_social_metrics(market)
        
        sentiment_scores = {
            'news_sentiment': self.analyze_news_sentiment(news_data),
            'social_sentiment': self.analyze_social_sentiment(social_data),
            'technical_sentiment': self.analyze_technical_sentiment(market)
        }
        
        return {
            'overall_sentiment': self.combine_sentiment_scores(sentiment_scores),
            'sentiment_breakdown': sentiment_scores,
            'confidence_level': self.calculate_sentiment_confidence(sentiment_scores)
        }

    def fetch_market_news(self, market):
        try:
            # Extract base asset from market pair
            base_asset = market.split('/')[0]
            
            # Fetch market data
            news_data = {
                'market': market,
                'sentiment': self.analyze_market_sentiment(market),
                'volume_profile': self.analyze_volume_profile(
                    self.fetch_market_data(market, '1h', limit=100)
                ),
                'technical_signals': self.get_technical_sentiment(
                    self.fetch_market_data(market, '1h', limit=100)
                )
            }
            
            return news_data
            
        except Exception as e:
            self.logger.error(f"Error fetching news for {market}: {e}")
            return None

    def fetch_social_metrics(self, market):
        try:
            # Extract base asset from market pair
            base_asset = market.split('/')[0]
            
            # Initialize default metrics
            social_metrics = {
                'market': market,
                'sentiment_score': 0.0,
                'volume_score': 0.0,
                'social_volume': 0.0,
                'trend_strength': 0.0
            }
            
            # Calculate metrics from available data
            market_data = self.fetch_market_data(market, '1h', limit=24)
            if market_data is not None:
                # Volume-based social score
                volume_mean = market_data['volume'].mean()
                volume_std = market_data['volume'].std()
                social_metrics['volume_score'] = (market_data['volume'].iloc[-1] - volume_mean) / volume_std if volume_std > 0 else 0
                
                # Trend strength
                social_metrics['trend_strength'] = self.detect_trend_strength(market_data)
                
                # Overall sentiment
                sentiment_data = self.analyze_market_sentiment(market)
                social_metrics['sentiment_score'] = sentiment_data.get('sentiment_score', 0.0)
                
            return social_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating social metrics for {market}: {e}")
            return None
    def analyze_news_sentiment(self, market):
        try:
            # Initialize sentiment metrics
            sentiment_data = {
                'market': market,
                'overall_score': 0.0,
                'technical_score': 0.0,
                'volume_score': 0.0,
                'momentum_score': 0.0
            }
            
            # Get market data without recursion
            df = self.fetch_market_data(market, '1h', limit=24)
            if df is not None:
                # Technical analysis score
                technical = self.get_technical_sentiment(df)
                sentiment_data['technical_score'] = (
                    1.0 if technical['rsi_sentiment'] == 'oversold' else
                    0.5 if technical['rsi_sentiment'] == 'neutral' else 0.0
                )
                
                # Volume analysis
                volume_profile = self.analyze_volume_profile(df)
                if volume_profile:
                    sentiment_data['volume_score'] = (
                        1.0 if volume_profile['volume_trend'] == 'increasing' else 0.0
                    )
                
                # Calculate overall score
                sentiment_data['overall_score'] = (
                    sentiment_data['technical_score'] * 0.5 +
                    sentiment_data['volume_score'] * 0.3 +
                    sentiment_data['momentum_score'] * 0.2
                )
                
            return sentiment_data
            
        except Exception as e:
            self.logger.error(f"Error analyzing news sentiment for {market}: {e}")
            return None
    def analyze_social_sentiment(self, market):
        if not market or not isinstance(market, str):
            return None
            
        try:
            sentiment_data = {
                'market': market,
                'social_score': 0.0,
                'volume_impact': 0.0,
                'trend_score': 0.0,
                'overall_sentiment': 'neutral'
            }
            
            # Fetch market data with proper error handling
            df = self.fetch_market_data(market, '1h', limit=24)
            if df is not None and not df.empty:
                # Volume analysis
                volume_profile = self.analyze_volume_profile(df)
                if volume_profile:
                    sentiment_data['volume_impact'] = 1.0 if volume_profile['volume_trend'] == 'increasing' else 0.0
                
                # Trend analysis
                trend_strength = self.detect_trend_strength(df)
                sentiment_data['trend_score'] = trend_strength / 100.0 if trend_strength else 0.0
                
                # Technical sentiment
                tech_sentiment = self.get_technical_sentiment(df)
                if tech_sentiment:
                    sentiment_data['social_score'] = (
                        1.0 if tech_sentiment['rsi_sentiment'] == 'oversold' else
                        0.5 if tech_sentiment['rsi_sentiment'] == 'neutral' else 0.0
                    )
                
                # Calculate overall sentiment
                overall_score = (
                    sentiment_data['social_score'] * 0.4 +
                    sentiment_data['volume_impact'] * 0.3 +
                    sentiment_data['trend_score'] * 0.3
                )
                
                sentiment_data['overall_sentiment'] = (
                    'bullish' if overall_score > 0.7 else
                    'bearish' if overall_score < 0.3 else
                    'neutral'
                )
                
            return sentiment_data
            
        except Exception as e:
            self.logger.error(f"Error analyzing social sentiment for {market}: {e}")
            return None

    def analyze_technical_sentiment(self, market):
        if not market or not isinstance(market, str):
            return None
            
        try:
            technical_data = {
                'market': market,
                'indicators': {},
                'patterns': [],
                'strength': 0.0,
                'signal': 'neutral'
            }
            
            # Fetch market data with proper error handling
            df = self.fetch_market_data(market, '1h', limit=100)
            if df is not None and not df.empty:
                # Calculate technical indicators
                df['rsi'] = talib.RSI(df['close'])
                df['macd'], df['macd_signal'], _ = talib.MACD(df['close'])
                df['ema_20'] = talib.EMA(df['close'], timeperiod=20)
                df['ema_50'] = talib.EMA(df['close'], timeperiod=50)
                
                latest = df.iloc[-1]
                
                # Analyze indicators
                technical_data['indicators'] = {
                    'rsi': {
                        'value': latest['rsi'],
                        'signal': 'oversold' if latest['rsi'] < 30 else 'overbought' if latest['rsi'] > 70 else 'neutral'
                    },
                    'macd': {
                        'value': latest['macd'],
                        'signal': 'bullish' if latest['macd'] > latest['macd_signal'] else 'bearish'
                    },
                    'ema': {
                        'signal': 'bullish' if latest['ema_20'] > latest['ema_50'] else 'bearish'
                    }
                }
                
                # Calculate strength score
                strength_factors = {
                    'rsi': 0.3,
                    'macd': 0.4,
                    'ema': 0.3
                }
                
                technical_data['strength'] = sum([
                    strength_factors['rsi'] * (1.0 if technical_data['indicators']['rsi']['signal'] == 'oversold' else 0.0),
                    strength_factors['macd'] * (1.0 if technical_data['indicators']['macd']['signal'] == 'bullish' else 0.0),
                    strength_factors['ema'] * (1.0 if technical_data['indicators']['ema']['signal'] == 'bullish' else 0.0)
                ])
                
                # Determine overall signal
                technical_data['signal'] = (
                    'strong_buy' if technical_data['strength'] > 0.7 else
                    'buy' if technical_data['strength'] > 0.5 else
                    'sell' if technical_data['strength'] < 0.3 else
                    'neutral'
                )
                
            return technical_data
            
        except Exception as e:
            self.logger.error(f"Error analyzing technical sentiment for {market}: {e}")
            return None

    def combine_sentiment_scores(self, market):
        if not market or not isinstance(market, str):
            return None
            
        try:
            # Initialize combined sentiment data
            combined_data = {
                'market': market,
                'technical_score': 0.0,
                'social_score': 0.0,
                'news_score': 0.0,
                'overall_score': 0.0,
                'signal': 'neutral'
            }
            
            # Get individual sentiment scores
            technical = self.analyze_technical_sentiment(market)
            social = self.analyze_social_sentiment(market)
            news = self.analyze_news_sentiment(market)
            
            # Weight factors for different sentiment types
            weights = {
                'technical': 0.5,
                'social': 0.3,
                'news': 0.2
            }
            
            # Calculate weighted scores
            if technical:
                combined_data['technical_score'] = technical['strength'] * weights['technical']
            if social:
                combined_data['social_score'] = (
                    social['social_score'] * weights['social']
                    if 'social_score' in social else 0.0
                )
            if news:
                combined_data['news_score'] = (
                    news['overall_score'] * weights['news']
                    if 'overall_score' in news else 0.0
                )
                
            # Calculate overall score
            combined_data['overall_score'] = (
                combined_data['technical_score'] +
                combined_data['social_score'] +
                combined_data['news_score']
            )
            
            # Determine signal based on overall score
            combined_data['signal'] = (
                'strong_buy' if combined_data['overall_score'] > 0.7 else
                'buy' if combined_data['overall_score'] > 0.5 else
                'sell' if combined_data['overall_score'] < 0.3 else
                'neutral'
            )
            
            return combined_data
            
        except Exception as e:
            self.logger.error(f"Error combining sentiment scores for {market}: {e}")
            return None
    def calculate_sentiment_confidence(self, market):
        if not market or not isinstance(market, str):
            return None
            
        try:
            confidence_data = {
                'market': market,
                'confidence_score': 0.0,
                'signal_strength': 'weak',
                'confirmation_count': 0,
                'metrics': {}
            }
            
            # Get sentiment data
            technical = self.analyze_technical_sentiment(market)
            social = self.analyze_social_sentiment(market)
            news = self.analyze_news_sentiment(market)
            
            # Count confirmations
            signals = []
            if technical and technical.get('signal') != 'neutral':
                signals.append(technical['signal'])
                confidence_data['metrics']['technical'] = technical['strength']
                
            if social and social.get('overall_sentiment') != 'neutral':
                signals.append(social['overall_sentiment'])
                confidence_data['metrics']['social'] = social.get('social_score', 0.0)
                
            if news and news.get('overall_score', 0) > 0:
                signals.append('bullish' if news['overall_score'] > 0.5 else 'bearish')
                confidence_data['metrics']['news'] = news['overall_score']
            
            # Calculate confidence score
            confidence_data['confirmation_count'] = len(signals)
            if confidence_data['confirmation_count'] > 0:
                confidence_data['confidence_score'] = sum(confidence_data['metrics'].values()) / len(signals)
                
            # Determine signal strength
            confidence_data['signal_strength'] = (
                'very_strong' if confidence_data['confidence_score'] > 0.8 else
                'strong' if confidence_data['confidence_score'] > 0.6 else
                'moderate' if confidence_data['confidence_score'] > 0.4 else
                'weak'
            )
            
            return confidence_data
            
        except Exception as e:
            self.logger.error(f"Error calculating sentiment confidence for {market}: {e}")
            return None

    def prepare_prediction_features(self, df):
        # Technical indicators
        df['rsi'] = talib.RSI(df['close'])
        df['macd'], _, _ = talib.MACD(df['close'])
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'])
        
        # Volume features
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Price action features
        df['price_momentum'] = df['close'].pct_change(5)
        df['volatility'] = df['close'].rolling(20).std()
        
        return df

    def train_models_incremental(self, new_data):
        for model_name, model in self.dl_models.items():
            X, y = self.prepare_training_data(new_data, model_name)
            model.fit(X, y, epochs=1, batch_size=32, verbose=0)
            self.update_model_metrics(model_name, model)
    def initialize_backtesting_engine(self):
        self.backtest_config = {
            'timeframes': ['15m', '1h', '4h', '1d'],
            'lookback_periods': [100, 200, 500],
            'monte_carlo_iterations': 1000,
            'optimization_windows': {'train': 200, 'test': 50}
        }
        return self.setup_backtest_environment()

    def run_multi_timeframe_backtest(self, strategy, market):
        results = {}
        for timeframe in self.backtest_config['timeframes']:
            df = self.fetch_market_data(market, timeframe)
            signals = self.generate_backtest_signals(df, strategy)
            
            performance = {
                'returns': self.calculate_returns(signals),
                'metrics': self.calculate_performance_metrics(signals),
                'drawdown': self.calculate_drawdown_metrics(signals)
            }
            
            results[timeframe] = {
                'signals': signals,
                'performance': performance,
                'validation': self.validate_strategy_results(performance)
            }
        
        return self.combine_timeframe_results(results)

    def monte_carlo_simulation(self, signals, iterations=1000):
        base_equity = 10000  # Starting equity
        results = []
        
        for _ in range(iterations):
            shuffled_returns = self.shuffle_returns(signals['returns'])
            equity_curve = self.simulate_equity_curve(shuffled_returns, base_equity)
            
            results.append({
                'final_equity': equity_curve[-1],
                'max_drawdown': self.calculate_max_drawdown(equity_curve),
                'sharpe_ratio': self.calculate_sharpe_ratio(shuffled_returns)
            })
        
        return {
            'confidence_intervals': self.calculate_confidence_intervals(results),
            'risk_metrics': self.analyze_simulation_risks(results),
            'optimization_suggestions': self.generate_optimization_suggestions(results)
        }

    def walk_forward_optimization(self, strategy, market):
        windows = self.backtest_config['optimization_windows']
        optimization_results = []
        
        for i in range(0, len(self.data) - windows['train'] - windows['test'], windows['test']):
            train_data = self.data[i:i + windows['train']]
            test_data = self.data[i + windows['train']:i + windows['train'] + windows['test']]
            
            # Optimize parameters on training data
            optimal_params = self.optimize_strategy_parameters(strategy, train_data)
            
            # Test on out-of-sample data
            test_results = self.test_strategy(strategy, test_data, optimal_params)
            
            optimization_results.append({
                'window': {'start': i, 'end': i + windows['train'] + windows['test']},
                'optimal_params': optimal_params,
                'test_performance': test_results
            })
        
        return {
            'optimization_summary': self.summarize_optimization_results(optimization_results),
            'parameter_stability': self.analyze_parameter_stability(optimization_results),
            'recommended_parameters': self.get_recommended_parameters(optimization_results)
        }
    def analyze_enhanced_order_flow(self, market):
        order_book = self.binance.fetch_order_book(market, limit=100)
        trades = self.binance.fetch_trades(market, limit=1000)
        
        return {
            'liquidity_analysis': self.analyze_liquidity_levels(order_book),
            'trade_flow': self.analyze_trade_flow(trades),
            'order_book_imbalance': self.calculate_order_imbalance(order_book),
            'smart_money_activity': self.detect_smart_money(trades, order_book)
        }

    def detect_institutional_activity(self, market):
        trades = self.binance.fetch_trades(market, limit=1000)
        order_book = self.binance.fetch_order_book(market, limit=100)
        
        return {
            'large_orders': self.track_large_orders(trades),
            'iceberg_detection': self.detect_iceberg_orders(market),
            'accumulation_zones': self.find_accumulation_zones(market),
            'institutional_levels': self.identify_institutional_levels(market)
        }

    def enhanced_volume_profile(self, market, timeframe):
        df = self.fetch_market_data(market, timeframe)
        
        profile_data = {
            'volume_nodes': self.calculate_volume_nodes(df),
            'poc_analysis': self.analyze_poc_levels(df),
            'value_areas': self.calculate_value_areas(df),
            'volume_delta': self.calculate_volume_delta(df)
        }
        
        return {
            'profile': profile_data,
            'key_levels': self.identify_key_levels(profile_data),
            'trading_opportunities': self.find_volume_opportunities(profile_data)
        }

    def analyze_liquidity_levels(self, order_book):
        bids = pd.DataFrame(order_book['bids'], columns=['price', 'volume'])
        asks = pd.DataFrame(order_book['asks'], columns=['price', 'volume'])
        
        return {
            'bid_clusters': self.find_liquidity_clusters(bids),
            'ask_clusters': self.find_liquidity_clusters(asks),
            'liquidity_score': self.calculate_liquidity_score(bids, asks),
            'significant_levels': self.find_significant_levels(bids, asks)
        }

    def detect_smart_money(self, trades, order_book):
        large_trades = [trade for trade in trades if float(trade['amount']) > self.get_size_threshold(trades)]
        
        return {
            'large_trade_impact': self.analyze_price_impact(large_trades),
            'accumulation_patterns': self.detect_accumulation(trades),
            'distribution_patterns': self.detect_distribution(trades),
            'smart_money_levels': self.identify_smart_money_levels(trades, order_book)
        }

    def calculate_volume_delta(self, df):
        df['buy_volume'] = df['volume'] * (df['close'] > df['open']).astype(float)
        df['sell_volume'] = df['volume'] * (df['close'] <= df['open']).astype(float)
        
        return {
            'delta': df['buy_volume'] - df['sell_volume'],
            'cumulative_delta': (df['buy_volume'] - df['sell_volume']).cumsum(),
            'delta_strength': self.calculate_delta_strength(df),
            'volume_trends': self.identify_volume_trends(df)
        }

    def validate_pattern_structure(self, df, pattern_type):
        structure_requirements = {
            'HEAD_AND_SHOULDERS': {
                'peak_ratios': self.check_peak_ratios,
                'volume_profile': self.check_volume_pattern,
                'time_symmetry': self.check_time_symmetry
            },
            'TRIPLE_BOTTOM': {
                'trough_alignment': self.check_trough_alignment,
                'price_confirmation': self.check_price_confirmation,
                'momentum_divergence': self.check_momentum_divergence
            }
        }
        
        validation_results = {}
        for check_name, check_func in structure_requirements[pattern_type].items():
            validation_results[check_name] = check_func(df)
        
        return {
            'valid': all(validation_results.values()),
            'details': validation_results,
            'confidence': self.calculate_validation_confidence(validation_results)
        }

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
        if current_time - self.rate_limits['last_reset'] >= self.rate_limits['cooldown_period']:
            self.rate_limits['api_calls'] = 0
            self.rate_limits['last_reset'] = current_time
        
        if self.rate_limits['api_calls'] >= self.rate_limits['max_calls_per_minute']:
            sleep_time = self.rate_limits['cooldown_period'] - (current_time - self.rate_limits['last_reset'])
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        self.rate_limits['api_calls'] += 1


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

