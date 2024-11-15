import ccxt
import pandas as pd
import talib
from datetime import datetime
import time
from base import BaseScanner
import requests
import logging
import threading
import sqlite3
import os
import json

import numpy as np







class MarketScanner(BaseScanner):
    def __init__(self, master=None):
        super().__init__(master)
        self.setup_rate_limiting()
        self.setup_trailing_stops()

    def setup_rate_limiting(self):
        self.rate_limits = {
            'api_calls': 0,
            'last_reset': time.time(),
            'max_calls_per_minute': 1200,
            'cooldown_period': 60
        }

    def setup_trailing_stops(self):
        self.trailing_data = {
            'highest_price': {},
            'lowest_price': {},
            'stop_levels': {},
            'active_signals': set()
        }
    def store_signals_db(self, signals, market, timeframe):
        for signal in signals:
            signal_data = {
                'market': market,
                'timeframe': timeframe,
                'signal_type': signal['type'],
                'price': signal['price'],
                'timestamp': str(datetime.now())
            }
            self.sql_operations('insert', self.db_signals, 'Signals', **signal_data)

    def scan_for_signals(self):
        while self.scanning:
            try:
                self.check_rate_limit()

                if self.user_choice.get() == 0:
                    markets = self.change(self.choose_list.get())
                else:
                    markets = self.selected_markets

                for market in markets:
                    if not self.scanning:
                        break

                    timeframe = self.choose_time.get()
                    df = self.fetch_market_data(market, timeframe)

                    if df is not None and not df.empty:
                        signals = self.generate_signals(df, market)
                        validated_signals = self.validate_signals_with_trailing(signals, market)
                        filtered_signals = self.filter_signals(validated_signals, market)

                        if filtered_signals:
                            self.process_signals(market, timeframe, filtered_signals)
                            self.store_signals_db(filtered_signals, market, timeframe)
                            self.update_trailing_stops(market, df['close'].iloc[-1])
                            self.update_dashboard()  # Update GUI with new data

                    time.sleep(0.1)

            except Exception as e:
                self.logger.error(f"Scanning error: {e}")
                time.sleep(5)
    def validate_signals_with_trailing(self, signals, market):
        try:
            validated = []
            current_price = float(self.binance.fetch_ticker(market)['last'])

            for signal in signals:
                if market not in self.trailing_data['stop_levels']:
                    self.trailing_data['stop_levels'][market] = current_price * 0.99

                if signal['type'] in ['RSI_OVERSOLD', 'BB_OVERSOLD', 'EMA_GOLDEN_CROSS']:
                    if current_price > self.trailing_data['stop_levels'][market]:
                        signal['trailing_stop'] = self.trailing_data['stop_levels'][market]
                        signal['current_price'] = current_price
                        validated.append(signal)
                        
                        # Log validation
                        self.logger.info(f"Signal validated for {market}")

            return validated
        except Exception as e:
            self.logger.error(f"Signal validation error: {e}")
            return []

    
    def filter_signals(self, signals, market):
        try:
            filtered = []
            for signal in signals:
                # Add market context to each signal
                signal['market'] = market
                
                # Get sentiment analysis
                sentiment = self.analyze_market_sentiment(market)
                if sentiment['sentiment_score'] >= 0.6:
                    signal['sentiment'] = sentiment
                    filtered.append(signal)
                    
                    # Log successful signal
                    self.logger.info(f"Valid signal detected for {market}: {signal['type']}")
            
            return filtered
        except Exception as e:
            self.logger.error(f"Error filtering signals: {e}")
            return []


    def update_trailing_stops(self, market, current_price):
        if market in self.trailing_data['active_signals']:
            if market not in self.trailing_data['highest_price']:
                self.trailing_data['highest_price'][market] = current_price

            elif current_price > self.trailing_data['highest_price'][market]:
                self.trailing_data['highest_price'][market] = current_price
                self.trailing_data['stop_levels'][market] = current_price * 0.99

    def analyze_market_correlations(self, markets, timeframe='1h'):
        correlation_matrix = {}
        market_data = {}
        
        for market in markets:
            df = self.fetch_market_data(market, timeframe, limit=100)
            if df is not None:
                market_data[market] = df['close']
                
        for market1 in market_data:
            correlation_matrix[market1] = {}
            for market2 in market_data:
                if market1 != market2:
                    corr = market_data[market1].corr(market_data[market2])
                    correlation_matrix[market1][market2] = corr
                    
        return correlation_matrix

    def multi_timeframe_scan(self, market):
        timeframes = ['5m', '15m', '1h', '4h']
        mtf_signals = {}
        
        for tf in timeframes:
            df = self.fetch_market_data(market, tf)
            if df is not None:
                signals = self.generate_signals(df)
                mtf_signals[tf] = signals
                
        return self.analyze_mtf_signals(mtf_signals)

    def analyze_mtf_signals(self, mtf_signals):
        signal_strength = 0
        confirmed_signals = []
        
        for signal_type in ['RSI_OVERSOLD', 'MACD_CROSSOVER', 'BB_OVERSOLD']:
            timeframe_count = sum(
                1 for tf_signals in mtf_signals.values()
                for signal in tf_signals
                if signal['type'] == signal_type
            )
            
            if timeframe_count >= 2:
                signal_strength += timeframe_count * 0.5
                confirmed_signals.append(signal_type)
                
        return {
            'strength': signal_strength,
            'confirmed_signals': confirmed_signals,
            'timeframe_data': mtf_signals
        }

    def scan_correlated_markets(self, base_market):
        correlations = self.analyze_market_correlations([base_market] + self.selected_markets)
        correlated_markets = [
            market for market, corr in correlations[base_market].items()
            if abs(corr) > 0.7  # High correlation threshold
        ]

        correlated_signals = {}
        for market in correlated_markets:
            mtf_analysis = self.multi_timeframe_scan(market)
            if mtf_analysis['strength'] > 2:
                correlated_signals[market] = mtf_analysis

        return correlated_signals

    def analyze_market_sentiment(self, market):
        # Fetch data from multiple sources
        price_data = self.fetch_market_data(market, '1h', limit=200)
        volume_profile = self.analyze_volume_profile(price_data)
        funding_rate = self.get_funding_rate(market)

        sentiment_score = self.calculate_sentiment_score({
            'price_action': self.analyze_price_action(price_data),
            'volume_profile': volume_profile,
            'funding_rate': funding_rate,
            'technical_signals': self.get_technical_sentiment(price_data)
        })

        return {
            'market': market,
            'sentiment_score': sentiment_score,
            'volume_profile': volume_profile,
            'funding_data': funding_rate
        }

    def analyze_volume_profile(self, df):
        if df is None or df.empty:
            return None
            
        # Calculate volume-weighted price levels
        df['vwap'] = (df['volume'] * df['close']).cumsum() / df['volume'].cumsum()
        
        # Identify key volume levels with explicit observed parameter
        volume_levels = pd.cut(df['close'], bins=10)
        volume_profile = df.groupby(volume_levels, observed=True)['volume'].sum()
        
        # Find high volume nodes
        high_volume_levels = volume_profile[volume_profile > volume_profile.mean()]
        
        # Calculate additional volume metrics
        volume_sma = df['volume'].rolling(window=20).mean()
        volume_ratio = df['volume'] / volume_sma
        
        return {
            'high_volume_levels': high_volume_levels.to_dict(),
            'vwap': df['vwap'].iloc[-1],
            'volume_trend': 'increasing' if df['volume'].is_monotonic_increasing else 'decreasing',
            'volume_ratio': volume_ratio.iloc[-1],
            'volume_sma': volume_sma.iloc[-1]
        }

    def detect_volume_climax(self, df):
        climax_points = []
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        
        for i in range(1, len(df)):
            if (df['volume'].iloc[i] > df['volume_sma'].iloc[i] * 2 and
                df['close'].iloc[i] > df['open'].iloc[i]):
                climax_points.append({
                    'price': df['low'].iloc[i],
                    'volume': df['volume'].iloc[i],
                    'strength': 'high'
                })
        return climax_points

    def identify_no_demand_zones(self, df):
        no_demand_zones = []
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['price_range'] = (df['high'] - df['low']) / df['low'] * 100
        
        for i in range(1, len(df)):
            if (df['volume'].iloc[i] < df['volume_sma'].iloc[i] * 0.5 and
                df['price_range'].iloc[i] < 0.5):
                no_demand_zones.append({
                    'price': df['low'].iloc[i],
                    'volume': df['volume'].iloc[i],
                    'strength': 'moderate'
                })
        return no_demand_zones

    def find_stopping_volume(self, df):
        stopping_points = []
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        
        for i in range(1, len(df)):
            if (df['volume'].iloc[i] > df['volume_sma'].iloc[i] * 1.5 and
                abs(df['close'].iloc[i] - df['open'].iloc[i]) < abs(df['high'].iloc[i] - df['low'].iloc[i]) * 0.3):
                stopping_points.append({
                    'price': df['low'].iloc[i],
                    'volume': df['volume'].iloc[i],
                    'strength': 'high'
                })
        return stopping_points

    def detect_selling_climax(self, df):
        climax_points = []
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        
        for i in range(1, len(df)):
            if (df['volume'].iloc[i] > df['volume_sma'].iloc[i] * 2 and
                df['close'].iloc[i] < df['open'].iloc[i] and
                (df['close'].iloc[i] - df['low'].iloc[i]) > (df['high'].iloc[i] - df['low'].iloc[i]) * 0.6):
                climax_points.append({
                    'price': df['low'].iloc[i],
                    'volume': df['volume'].iloc[i],
                    'strength': 'extreme'
                })
        return climax_points

    def evaluate_vsa_strength(self, vsa_indicators):
        strength_score = 0
        
        if vsa_indicators['volume_climax']:
            strength_score += 0.3
        if vsa_indicators['stopping_volume']:
            strength_score += 0.3
        if vsa_indicators['selling_climax']:
            strength_score += 0.3
        if vsa_indicators['no_demand']:
            strength_score += 0.1
            
        return {
            'score': strength_score,
            'indicators': vsa_indicators
        }

    def define_strong_support(self, df, market):
        """Comprehensive support strength analysis"""
        support_analysis = {
            'price_action': self.detect_price_action_supports(df),
            'vsa': self.apply_vsa_support_analysis(df),
            'multi_timeframe': self.confirm_multi_timeframe_supports(market),
            'order_book': self.verify_support_with_order_book(market)
        }
        
        return self.synthesize_support_strength(support_analysis)

    def detect_price_action_supports(self, df):
        support_levels = []
        for window in [20, 50, 100]:
            rolling_low = df['low'].rolling(window=window).min()
            support_points = rolling_low[
                (rolling_low == df['low']) & 
                (df['low'].shift(1) > df['low'])
            ]
            for price in support_points:
                support_levels.append({
                    'price': price,
                    'bounce_count': self.count_support_bounces(df, price),
                    'timeframe_weight': window/20
                })
        
        return {
            'score': len(support_levels) * 0.1,
            'levels': support_levels
        }

    def synthesize_support_strength(self, support_analysis):
        try:
            weights = {
                'price_action': 0.3,
                'vsa': 0.3,
                'multi_timeframe': 0.25,
                'order_book': 0.15
            }
            
            price_action_score = float(support_analysis['price_action'].get('score', 0))
            vsa_score = float(support_analysis['vsa'].get('score', 0))
            mtf_score = float(support_analysis['multi_timeframe'])
            ob_score = float(support_analysis['order_book'].get('total_bid_volume', 0)) / 1000
            
            total_score = (
                price_action_score * weights['price_action'] +
                vsa_score * weights['vsa'] +
                mtf_score * weights['multi_timeframe'] +
                min(ob_score, 1.0) * weights['order_book']
            )
            
            return {
                'strength': 'Very Strong' if total_score > 0.8 else
                        'Strong' if total_score > 0.6 else
                        'Moderate' if total_score > 0.4 else 'Weak',
                'score': total_score,
                'details': support_analysis
            }
        except Exception as e:
            self.logger.error(f"Error in support strength calculation: {e}")
            return {'strength': 'Weak', 'score': 0, 'details': {}}

    def apply_vsa_support_analysis(self, df):
        """Volume Spread Analysis for support detection"""
        if df is None or df.empty:
            return None
            
        vsa_indicators = {
            'volume_climax': self.detect_volume_climax(df),
            'no_demand': self.identify_no_demand_zones(df),
            'stopping_volume': self.find_stopping_volume(df),
            'selling_climax': self.detect_selling_climax(df)
        }
        
        return self.evaluate_vsa_strength(vsa_indicators)

    def evaluate_vsa_strength(self, vsa_indicators):
        try:
            strength_score = 0
            
            # Score volume climax
            if vsa_indicators['volume_climax']:
                strength_score += len(vsa_indicators['volume_climax']) * 0.3
                
            # Score stopping volume
            if vsa_indicators['stopping_volume']:
                strength_score += len(vsa_indicators['stopping_volume']) * 0.3
                
            # Score selling climax
            if vsa_indicators['selling_climax']:
                strength_score += len(vsa_indicators['selling_climax']) * 0.3
                
            # Score no demand zones
            if vsa_indicators['no_demand']:
                strength_score += len(vsa_indicators['no_demand']) * 0.1
                
            return {
                'score': min(strength_score, 1.0),  # Cap at 1.0
                'indicators': vsa_indicators
            }
            
        except Exception as e:
            self.logger.error(f"VSA strength evaluation error: {e}")
            return {'score': 0, 'indicators': {}}

    def analyze_vsa_signals(self, df):
        vsa_signals = []
        
        # Calculate volume metrics
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['price_range'] = df['high'] - df['low']
        df['range_ma'] = df['price_range'].rolling(window=20).mean()
        
        # Identify support/resistance levels
        df['support'] = df['low'].rolling(window=20).min()
        df['resistance'] = df['high'].rolling(window=20).max()
        
        # Analyze last candle
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # High volume bullish signal near support
        if (latest['low'] <= latest['support'] and 
            latest['close'] > latest['open'] and
            latest['volume'] > latest['volume_ma'] * 2):
            vsa_signals.append({
                'type': 'VSA_BULLISH',
                'strength': 'high',
                'price': latest['close'],
                'volume_ratio': latest['volume'] / latest['volume_ma']
            })
        
        # Narrow range high volume at resistance (absorption)
        if (latest['high'] >= latest['resistance'] and
            latest['price_range'] < latest['range_ma'] * 0.5 and
            latest['volume'] > latest['volume_ma'] * 2):
            vsa_signals.append({
                'type': 'VSA_ABSORPTION',
                'strength': 'high',
                'price': latest['close'],
                'volume_ratio': latest['volume'] / latest['volume_ma']
            })
            
        return vsa_signals
    def confirm_htf_trend(self, market):
        htf_frames = ['4h', '1d']
        bullish_count = 0
        
        for tf in htf_frames:
            df = self.fetch_market_data(market, tf)
            if df is not None and not df.empty:
                if self.detect_trend_strength(df) > 25:
                    bullish_count += 1
                    
        return bullish_count >= len(htf_frames)/2    
    def analyze_market_performance(self, market):
        # Fetch data for multiple timeframes for HTF analysis
        timeframes = ['1h', '4h', '1d']
        df_dict = {}
        for tf in timeframes:
            df_dict[tf] = self.fetch_market_data(market, tf)
        
        df = df_dict[self.choose_time.get()]
        if df is not None and not df.empty:
            # Enhanced trend strength with HTF confirmation
            trend_strength = sum(
                self.detect_trend_strength(df_dict[tf]) * (1 + timeframes.index(tf)/len(timeframes))
                for tf in timeframes
            ) * self.trend_weight.get()

            # Volume analysis with order block detection
            volume_profile = self.analyze_volume_profile(df)
            volume_score = (
                volume_profile['volume_ratio'] * 
                (2 if volume_profile['volume_trend'] == 'increasing' else 1) * 
                self.volume_weight.get()
            )

            # RSI with volume confirmation
            rsi = talib.RSI(df['close']).iloc[-1]
            rsi_score = ((100-rsi)/100 * 
                        (1.5 if volume_profile['volume_ratio'] > 1.5 else 1) * 
                        self.rsi_weight.get())

            # Multi-timeframe support strength
            support_analysis = self.define_strong_support(df, market)
            support_strength = (
                support_analysis['score'] * 
                (2 if support_analysis['strength'] == 'Very Strong' else 1) * 
                self.support_weight.get()
            )

            # VSA signals integration
            vsa_signals = self.analyze_vsa_signals(df)
            vsa_score = len(vsa_signals) * 0.2

            # Calculate enhanced composite score
            score = (
                trend_strength * 0.2 +
                volume_score * 0.3 +
                rsi_score * 0.15 +
                support_strength * 0.25 +
                vsa_score * 0.1
            )

            return {
                'market': market,
                'score': score,
                'timeframe': self.choose_time.get(),
                'exchange': self.choose_listex.get(),
                'trend_score': trend_strength,
                'volume_score': volume_score,
                'rsi_score': rsi,
                'support_score': support_strength,
                'vsa_signals': len(vsa_signals),
                'support_strength': support_analysis['strength'],
                'volume_trend': volume_profile['volume_trend'],
                'htf_confirmation': any(
                    self.detect_trend_strength(df_dict[tf]) > 25 
                    for tf in ['4h', '1d']
                )
            }
        return None

    def find_best_buy_opportunity(self):
        best_market = None
        best_score = -float('inf')
        for market in self.selected_markets:
            score = self.analyze_market_performance(market)
            if score and score > best_score:
                best_score = score
                best_market = market

        if best_market:
            message = f"Best Buy Opportunity:\nMarket: {best_market}\nScore: {best_score}"
            self.send_telegram_update(message)    
    
    def count_support_bounces(self, df, support_price):
        return sum(
            1 for i in range(len(df)) 
            if abs(df['low'].iloc[i] - support_price) / support_price < 0.02 and
            df['close'].iloc[i] > df['open'].iloc[i]
        )

    def confirm_multi_timeframe_supports(self, market):
        timeframes = ['1h', '4h', '1d', '1w']
        support_strength = 0
        
        for tf in timeframes:
            df = self.fetch_market_data(market, tf)
            if df is not None:
                supports = self.detect_price_action_supports(df)
                support_strength += len(supports) * (timeframes.index(tf) + 1)
        
        return support_strength

    def verify_support_with_order_book(self, market):
        order_book = self.binance.fetch_order_book(market)
        support_zones = {}
        
        for bid in order_book['bids'][:10]:  # Top 10 bid levels
            price_level = bid[0]
            volume = bid[1]
            support_zones[price_level] = volume
            
        return {
            'support_zones': support_zones,
            'total_bid_volume': sum(support_zones.values())
        }

    def detect_bull_trap(self, df):
        try:
            # Check for false breakout above resistance
            df['resistance'] = df['high'].rolling(window=20).max()
            return (df['high'].iloc[-1] > df['resistance'].iloc[-2] and
                    df['close'].iloc[-1] < df['resistance'].iloc[-2] and
                    df['volume'].iloc[-1] > df['volume'].rolling(20).mean().iloc[-1])
        except Exception as e:
            self.logger.error(f"Error detecting bull trap: {e}")
            return False

    def detect_bear_trap(self, df):
        try:
            # Check for false breakout below support
            df['support'] = df['low'].rolling(window=20).min()
            return (df['low'].iloc[-1] < df['support'].iloc[-2] and
                    df['close'].iloc[-1] > df['support'].iloc[-2] and
                    df['volume'].iloc[-1] > df['volume'].rolling(20).mean().iloc[-1])
        except Exception as e:
            self.logger.error(f"Error detecting bear trap: {e}")
            return False

    def detect_trend_strength(self, df):
        df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        return df['adx'].iloc[-1]

    def get_technical_sentiment(self, df):
        if df is None or df.empty:
            return None

        # Calculate technical indicators
        df['rsi'] = talib.RSI(df['close'])
        df['macd'], df['macd_signal'], _ = talib.MACD(df['close'])
        df['slowk'], df['slowd'] = talib.STOCH(df['high'], df['low'], df['close'])

        latest = df.iloc[-1]

        return {
            'rsi_sentiment': 'oversold' if latest['rsi'] < 30 else 'overbought' if latest['rsi'] > 70 else 'neutral',
            'macd_sentiment': 'bullish' if latest['macd'] > latest['macd_signal'] else 'bearish',
            'stoch_sentiment': 'oversold' if latest['slowk'] < 20 else 'overbought' if latest['slowk'] > 80 else 'neutral'
        }

    def calculate_sentiment_score(self, sentiment_data):
        score = 0
        weights = {
            'price_action': 0.3,
            'volume_profile': 0.2,
            'funding_rate': 0.2,
            'technical_signals': 0.3
        }

        # Score price action
        if sentiment_data['price_action'].get('trend') == 'uptrend':
            score += weights['price_action']

        # Score volume profile
        if sentiment_data['volume_profile'].get('volume_trend') == 'increasing':
            score += weights['volume_profile']

        # Score funding rate
        if sentiment_data['funding_rate'].get('rate', 0) < 0:
            score += weights['funding_rate']

        # Score technical signals
        tech = sentiment_data['technical_signals']
        if tech:
            if tech['rsi_sentiment'] == 'oversold':
                score += weights['technical_signals'] * 0.4
            if tech['macd_sentiment'] == 'bullish':
                score += weights['technical_signals'] * 0.3
            if tech['stoch_sentiment'] == 'oversold':
                score += weights['technical_signals'] * 0.3

        return score

    def filter_signals_by_sentiment(self, signals, sentiment_threshold=0.6):
        filtered_signals = []

        for signal in signals:
            sentiment = self.analyze_market_sentiment(signal['market'])
            if sentiment['sentiment_score'] >= sentiment_threshold:
                signal['sentiment_data'] = sentiment
                filtered_signals.append(signal)

        return filtered_signals

    def fetch_market_data(self, market, timeframe, limit=100):
        try:
            ohlcv = self.binance.fetch_ohlcv(market, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            self.logger.error(f"Data fetch error for {market}: {e}")
            return None

    def get_funding_rate(self, market):
        """
        Retrieve funding rate with comprehensive error handling
        """
        try:
            # Placeholder implementation - replace with actual exchange API call
            return {
                'rate': 0,  # Neutral funding rate
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Funding rate retrieval error for {market}: {e}")
            return {'rate': 0, 'timestamp': datetime.now().isoformat()}

    def analyze_price_action(self, df):
        """
        Comprehensive price action analysis with robust error handling
        """
        try:
            if df is None or df.empty:
                return {'trend': 'neutral', 'volatility': 0}
            
            # Calculate price change
            price_change = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0] * 100
            
            # Determine trend
            if price_change > 2:
                trend = 'uptrend'
            elif price_change < -2:
                trend = 'downtrend'
            else:
                trend = 'neutral'
            
            # Volatility calculation
            volatility = df['close'].pct_change().std() * 100
            
            return {
                'trend': trend,
                'price_change_percent': price_change,
                'volatility': volatility,
                'current_price': df['close'].iloc[-1],
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Price action analysis error: {e}")
            return {
                'trend': 'neutral', 
                'price_change_percent': 0, 
                'volatility': 0, 
                'current_price': None,
                'timestamp': datetime.now().isoformat()
            }
    def generate_signals(self, df, market):
        signals = []
        try:
            # Comprehensive validation
            if df is None or df.empty:
                self.logger.error(f"DataFrame is None or empty for market: {market}")
                return []
            
            # Ensure minimum data points
            if len(df) < 20:
                self.logger.warning(f"Insufficient data points for {market}: {len(df)} rows")
                return []
            
            # Ensure required columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                self.logger.error(f"Missing columns in DataFrame: {missing_columns}")
                return []
            
            # Calculate indicators safely
            try:
                close = df['close'].values
                high = df['high'].values
                low = df['low'].values
                
                # Safe indicator calculations with fallback
                def safe_ta_lib_calc(func, *args, **kwargs):
                    try:
                        result = func(*args, **kwargs)
                        if isinstance(result, tuple):
                            if len(result) == 3:
                                return result
                            else:
                                return result[-1]
                        return result
                    except Exception as e:
                        self.logger.error(f"TA-Lib calculation error for {func.__name__}: {e}")
                        return None
                
                # Calculate indicators
                df['rsi'] = safe_ta_lib_calc(talib.RSI, close, timeperiod=14)
                
                # Special handling for MACD
                macd_result = safe_ta_lib_calc(talib.MACD, close)
                if macd_result is not None:
                    if isinstance(macd_result, tuple):
                        df['macd'], df['macd_signal'], df['macd_hist'] = macd_result
                    else:
                        # Fallback if unexpected result
                        df['macd'] = df['macd_signal'] = df['macd_hist'] = None
                
                df['ema_20'] = safe_ta_lib_calc(talib.EMA, close, timeperiod=20)
                df['ema_50'] = safe_ta_lib_calc(talib.EMA, close, timeperiod=50)
                
                # Fallback calculations if TA-Lib fails
                if df['rsi'] is None:
                    df['rsi'] = self._manual_rsi_calculation(close)
                
                if df['ema_20'] is None or df['ema_50'] is None:
                    df['ema_20'] = self._manual_ema_calculation(close, 20)
                    df['ema_50'] = self._manual_ema_calculation(close, 50)
            
            except Exception as indicator_error:
                self.logger.error(f"Indicator calculation error for {market}: {indicator_error}")
                return []
            
            # Simplified support strength analysis
            try:
                support_strength = self._simplified_support_strength(df)
            except Exception as support_error:
                self.logger.error(f"Support strength analysis error: {support_error}")
                support_strength = {'score': 0.5}  # Default moderate support
            
            # Signal generation strategies
            signal_strategies = [
                self._generate_rsi_signals,
                self._generate_macd_signals,
                self._generate_ema_signals,
                self._generate_vsa_signals
            ]
            
            # Collect signals from different strategies
            for strategy in signal_strategies:
                try:
                    strategy_signals = strategy(df, market, support_strength)
                    signals.extend(strategy_signals)
                except Exception as strategy_error:
                    self.logger.error(f"Signal strategy error for {strategy.__name__}: {strategy_error}")
            
            # Fallback signal generation if no signals
            if not signals:
                signals = self._generate_fallback_signals(df, market)
            
            # Log signal generation
            if signals:
                self.logger.info(f"Generated {len(signals)} signals for {market}")
            else:
                self.logger.info(f"No signals generated for {market}")
            
            return signals
        
        except Exception as e:
            self.logger.error(f"Comprehensive signal generation error for {market}: {e}")
            return []
    def scan_for_signals(self):
        while self.scanning:
            try:
                markets = self.selected_markets if self.user_choice.get() == 1 else self.change(self.choose_list.get())
                
                for market in markets:
                    if not self.scanning:
                        break
                        
                    timeframe = self.choose_time.get()
                    df = self.fetch_market_data(market, timeframe)
                    
                    if df is not None and not df.empty:
                        signals = self.generate_signals(df, market)
                        if signals:
                            self.process_signals(market, timeframe, signals)
                            self.store_signals_db(signals, market, timeframe)
                            self.update_dashboard()
                            
                    time.sleep(0.5)
                    
            except Exception as e:
                self.logger.error(f"Scanning error: {e}")
                time.sleep(5)

    def _manual_rsi_calculation(self, close_prices, period=14):
        """Manual RSI calculation as a fallback"""
        try:
            import numpy as np
            
            changes = np.diff(close_prices)
            gains = changes.clip(min=0)
            losses = -changes.clip(max=0)
            
            avg_gain = np.mean(gains[:period])
            avg_loss = np.mean(losses[:period])
            
            rs = avg_gain / avg_loss if avg_loss != 0 else 0
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
        except Exception as e:
            self.logger.error(f"Manual RSI calculation error: {e}")
            return None

    def _manual_ema_calculation(self, close_prices, period):
        """Manual EMA calculation as a fallback"""
        try:
            smoothing = 2 / (period + 1)
            ema = [close_prices[0]]
            
            for price in close_prices[1:]:
                ema.append((price * smoothing) + (ema[-1] * (1 - smoothing)))
            
            return ema[-1]
        except Exception as e:
            self.logger.error(f"Manual EMA calculation error: {e}")
            return None

    def _simplified_support_strength(self, df):
        """Simplified support strength calculation"""
        try:
            # Basic support calculation
            support = df['low'].rolling(window=10).min().iloc[-1]
            resistance = df['high'].rolling(window=10).max().iloc[-1]
            
            # Calculate support strength based on price proximity to support
            current_price = df['close'].iloc[-1]
            distance_from_support = abs(current_price - support) / current_price
            
            # Score calculation
            support_score = max(0, 1 - distance_from_support)
            
            return {
                'score': support_score,
                'support_level': support,
                'resistance_level': resistance
            }
        except Exception as e:
            self.logger.error(f"Simplified support strength error: {e}")
            return {'score': 0.5}

    def _generate_rsi_signals(self, df, market, support_strength):
        signals = []
        try:
            if df['rsi'] is None:
                return signals
            
            # RSI oversold signal
            if df['rsi'].iloc[-1] < 30:
                signals.append({
                    'type': 'RSI_OVERSOLD',
                    'market': market,
                    'price': df['close'].iloc[-1],
                    'rsi': df['rsi'].iloc[-1],
                    'support_strength': support_strength['score'],
                    'timestamp': datetime.now().isoformat()
                })
            
            # RSI overbought signal
            if df['rsi'].iloc[-1] > 70:
                signals.append({
                    'type': 'RSI_OVERBOUGHT',
                    'market': market,
                    'price': df['close'].iloc[-1],
                    'rsi': df['rsi'].iloc[-1],
                    'support_strength': support_strength['score'],
                    'timestamp': datetime.now().isoformat()
                })
        except Exception as e:
            self.logger.error(f"RSI signal generation error: {e}")
        
        return signals
    def _generate_ema_signals(self, df, market, support_strength):
        """Generate EMA-based signals"""
        signals = []
        try:
            # EMA crossover signal
            if (df['ema_20'].iloc[-1] > df['ema_50'].iloc[-1] and 
                df['ema_20'].iloc[-2] <= df['ema_50'].iloc[-2]):
                signals.append({
                    'type': 'EMA_GOLDEN_CROSS',
                    'market': market,
                    'price': df['close'].iloc[-1],
                    'ema_20': df['ema_20'].iloc[-1],
                    'ema_50': df['ema_50'].iloc[-1],
                    'support_strength': support_strength['score'],
                    'timestamp': datetime.now().isoformat()
                })
        except Exception as e:
            self.logger.error(f"EMA signal generation error: {e}")
        
        return signals

    def _generate_basic_signals(self, df, market, support_strength):
        """Basic signal generation with minimal requirements"""
        signals = []
        try:
            current_price = df['close'].iloc[-1]
            
            # Price change signals
            price_change = (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2] * 100
            
            if abs(price_change) > 2:  # 2% price change
                signal_type = 'PRICE_SURGE_BULLISH' if price_change > 0 else 'PRICE_SURGE_BEARISH'
                signals.append({
                    'type': signal_type,
                    'market': market,
                    'price': current_price,
                    'price_change_percent': price_change,
                    'support_strength': support_strength['score'],
                    'timestamp': datetime.now().isoformat()
                })
            
            # Volume spike signal
            volume_change = (df['volume'].iloc[-1] - df['volume'].iloc[-2]) / df['volume'].iloc[-2] * 100
            if volume_change > 50:  # 50% volume increase
                signals.append({
                    'type': 'VOLUME_SPIKE',
                    'market': market,
                    'price': current_price,
                    'volume_change_percent': volume_change,
                    'support_strength': support_strength['score'],
                    'timestamp': datetime.now().isoformat()
                })
        
        except Exception as e:
            self.logger.error(f"Basic signal generation error: {e}")
        
        return signals

    def _generate_fallback_signals(self, df, market):
        """Absolute fallback signal generation"""
        signals = []
        try:
            current_price = df['close'].iloc[-1]
            
            # Extremely basic signals
            signals.append({
                'type': 'MARKET_OBSERVATION',
                'market': market,
                'price': current_price,
                'timestamp': datetime.now().isoformat()
            })
        
        except Exception as e:
            self.logger.error(f"Fallback signal generation error: {e}")
        
        return signals



    def _generate_macd_signals(self, df, market, support_strength):
        signals = []
        try:
            # Check if MACD columns exist and are not None
            if (not hasattr(df, 'macd') or not hasattr(df, 'macd_signal') or 
                df['macd'] is None or df['macd_signal'] is None):
                return signals
            
            # MACD bullish crossover
            if (len(df) >= 2 and 
                df['macd'].iloc[-1] is not None and 
                df['macd_signal'].iloc[-1] is not None and
                df['macd'].iloc[-1] > df['macd_signal'].iloc[-1] and 
                df['macd'].iloc[-2] <= df['macd_signal'].iloc[-2]):
                signals.append({
                    'type': 'MACD_BULLISH_CROSS',
                    'market': market,
                    'price': df['close'].iloc[-1],
                    'macd': df['macd'].iloc[-1],
                    'macd_signal': df['macd_signal'].iloc[-1],
                    'support_strength': support_strength['score'],
                    'timestamp': datetime.now().isoformat()
                })
        except Exception as e:
            self.logger.error(f"MACD signal generation error: {e}")
        
        return signals

    def _generate_vsa_signals(self, df, market, support_strength):
        signals = []
        try:
            # Volume analysis
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            
            # High volume with price action
            if (df['volume'].iloc[-1] > df['volume_ma'].iloc[-1] * 2 and
                df['close'].iloc[-1] > df['open'].iloc[-1]):
                signals.append({
                    'type': 'VSA_BULLISH',
                    'market': market,
                    'price': df['close'].iloc[-1],
                    'volume_ratio': df['volume'].iloc[-1] / df['volume_ma'].iloc[-1],
                    'support_strength': support_strength['score'],
                    'timestamp': datetime.now().isoformat()
                })
        except Exception as e:
            self.logger.error(f"VSA signal generation error: {e}")
        
        return signals



    def process_signals(self, market, timeframe, signals):
        for signal in signals:
            # Get comprehensive market analysis
            df = self.fetch_market_data(market, timeframe)
            
            # Calculate technical indicators
            df['rsi'] = talib.RSI(df['close'])
            volume_profile = self.analyze_volume_profile(df)
            support_analysis = self.define_strong_support(df, market)
            sentiment = self.analyze_market_sentiment(market)
            
            # Calculate signal strength (0-5 green dots)
            strength_score = (
                (sentiment['sentiment_score'] > 0.6) + 
                (volume_profile['volume_ratio'] > 1.5) +
                (support_analysis['strength'] in ['Strong', 'Very Strong']) +
                (df['rsi'].iloc[-1] < 40) +
                (volume_profile['volume_trend'] == 'increasing')
            )
            strength_indicators = "🟢" * strength_score
            
            signal_data = {
                'market': market,
                'timeframe': timeframe,
                'signal_type': signal['type'],
                'price': signal['price'],
                'volume_trend': volume_profile['volume_trend'],
                'volume_ratio': volume_profile['volume_ratio'],
                'vwap': volume_profile['vwap'],
                'rsi': df['rsi'].iloc[-1],
                'support_strength': support_analysis['strength'],
                'sentiment_score': sentiment['sentiment_score'],
                'strength_score': strength_score,
                'timestamp': str(datetime.now())
            }
            
            self.sql_operations('insert', self.db_signals, 'Signals', **signal_data)
            
            message = (
                f"{strength_indicators}\n"
                f"🎯 Strong Buy Signal Detected!\n\n"
                f"🪙 Coin: {market}\n"
                f"⏰ Timeframe: {timeframe}\n"
                f"💰 Current Price: {signal['price']:.8f}\n\n"
                f"📊 Volume Analysis:\n"
                f"• Trend: {volume_profile['volume_trend']}\n"
                f"• Ratio: {volume_profile['volume_ratio']:.2f}x\n"
                f"• VWAP: {volume_profile['vwap']:.2f}\n\n"
                f"💪 Support Analysis:\n"
                f"• Strength: {support_analysis['strength']}\n"
                f"• Level: {support_analysis.get('support_level', 0):.8f}\n\n"
                f"📈 Technical Indicators:\n"
                f"• RSI: {signal_data['rsi']:.2f}\n"
                f"• Signal Type: {signal['type']}\n"
                f"• Sentiment Score: {sentiment['sentiment_score']:.2f}\n\n"
                f"🎯 Entry Zone: {signal['price']:.8f} - {signal['price']*1.01:.8f}"
            )
            
            if hasattr(self, 'master') and self.master:
                self.master.after(0, self.update_dashboard)
            
            self.logger.info(f"Processing signal for {market}: {signal['type']}")
            self.send_telegram_update(message)



    def monitor_price_action(self):
        while self.scanning:
            try:
                for market in self.selected_markets:
                    df = self.fetch_market_data(market, self.choose_time.get())
                    if df is not None and not df.empty:
                        # Analyze price action patterns
                        patterns = self.detect_chart_patterns(df)
                        
                        # Check for support/resistance levels
                        support_analysis = self.define_strong_support(df, market)
                        
                        # Volume analysis
                        volume_profile = self.analyze_volume_profile(df)
                        
                        # Generate signals if patterns found
                        if patterns or support_analysis['strength'] == 'Strong':
                            signal_data = {
                                'market': market,
                                'patterns': patterns,
                                'support': support_analysis,
                                'volume': volume_profile
                            }
                            self.process_signals(market, self.choose_time.get(), [signal_data])
                            
                time.sleep(self.rate_config['window'])
                
            except Exception as e:
                self.logger.error(f"Price action monitoring error: {e}")
                time.sleep(5)

    def send_telegram_update(self, message):
        if self.tel_id and self.bot_token:
            try:
                url = f'https://api.telegram.org/bot{self.bot_token}/sendMessage'
                params = {
                    'chat_id': self.tel_id,
                    'text': message,
                    'parse_mode': 'HTML'
                }
                response = requests.post(url, params=params)
                if not response.ok:
                    self.logger.error(f"Telegram API error: {response.text}")
            except Exception as e:
                self.logger.error(f"Telegram error: {e}")
                time.sleep(5)

    def detect_chart_patterns(self, df):
        patterns = []
        
        # Candlestick patterns
        if talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])[-1] > 0:
            patterns.append('BULLISH_ENGULFING')
        
        if talib.CDLMORNINGSTAR(df['open'], df['high'], df['low'], df['close'])[-1] > 0:
            patterns.append('MORNING_STAR')
            
        # Support/Resistance levels
        df['support'] = df['low'].rolling(window=20).min()
        df['resistance'] = df['high'].rolling(window=20).max()
        
        return patterns

    def calculate_atr(self, symbol, tf):
        df = self.data(symbol, tf, limit=14)
        if df.empty:
            return None
        atr = df['ATR'].iloc[-1]
        return atr

    def trend_filter(self, df, ma1, ma2, length):
        df['Trend'] = 'none'
        for i in range(length, len(df)):
            if all(df[ma1].iloc[i-length:i] > df[ma2].iloc[i-length:i]):
                df.loc[df.index[i], 'Trend'] = 'up'
            elif all(df[ma1].iloc[i-length:i] < df[ma2].iloc[i-length:i]):
                df.loc[df.index[i], 'Trend'] = 'down'

