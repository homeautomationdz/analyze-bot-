import ccxt
import pandas as pd
import talib
from datetime import datetime
import time
from base import BaseScanner
import requests
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
    def scan_for_signals(self):
        while self.scanning:
            try:
                self.check_rate_limit()
                self.logger.info("Starting market scan cycle")

                if self.user_choice.get() == 0:
                    markets = self.change(self.choose_list.get())
                    self.logger.info(f"Scanning {len(markets)} markets from auto-selection")
                else:
                    markets = self.selected_markets
                    self.logger.info(f"Scanning {len(markets)} manually selected markets")

                for market in markets:
                    if not self.scanning:
                        break

                    timeframe = self.choose_time.get()
                    df = self.fetch_market_data(market, timeframe)

                    if df is not None and not df.empty:
                        self.logger.info(f"Analyzing {market} on {timeframe} timeframe")
                        
                        df = self.calculate_technical_indicators(df)
                        key_levels = self.analyze_key_levels(df)
                        
                        signals = self.generate_signals(df)
                        vsa_signals = self.analyze_vsa_signals(df)
                        volume_patterns = self.analyze_volume_patterns(df)
                        pattern_signals = self.analyze_pattern_sequence(df)
                        divergence_signals = self.detect_divergences(df)
                        
                        all_base_signals = []
                        all_base_signals.extend(signals)
                        all_base_signals.extend(vsa_signals)
                        all_base_signals.extend(volume_patterns)
                        all_base_signals.extend(pattern_signals)
                        all_base_signals.extend(divergence_signals)

                        signal_count = len(all_base_signals)
                        dots = self.get_signal_dots(signal_count)
                        self.logger.info(f"Found {signal_count} initial signals for {market} {dots}")

                        strong_signals = []
                        for signal in all_base_signals:
                            signal['key_levels'] = key_levels
                            strength_validation = self.validate_strong_buy(signal, df, key_levels, market)
                            
                            if strength_validation['is_strong_buy']:
                                signal['strength'] = 'STRONG'
                                signal['confirmation_score'] = strength_validation['strength_score']
                                signal['confirmation_factors'] = strength_validation['confirmation_factors']
                                strong_signals.append(signal)

                        if strong_signals:
                            self.logger.info(f"Found {len(strong_signals)} strong signals for {market}")
                            validated_signals = self.validate_signals_with_trailing(strong_signals, market)
                            filtered_signals = self.filter_signals(validated_signals, market)

                            htf_wick_signal = self.detect_htf_support_wick(market)
                            drop_recovery_signal = self.detect_support_drop_recovery(market)
                            range_breakout_signal = self.detect_range_breakout_wick(market)

                            all_signals = []
                            if filtered_signals:
                                all_signals.extend(filtered_signals)
                            if htf_wick_signal:
                                all_signals.append(htf_wick_signal)
                            if drop_recovery_signal:
                                all_signals.append(drop_recovery_signal)
                            if range_breakout_signal:
                                all_signals.append(range_breakout_signal)

                            if all_signals:
                                self.logger.info(f"Processing {len(all_signals)} final signals for {market}")
                                market_regime = self.analyze_market_regime(df)
                                for signal in all_signals:
                                    signal['market_regime'] = market_regime
                                    signal['volume_profile'] = self.analyze_volume_profile(df)

                                self.process_signals(market, timeframe, all_signals)
                                self.store_signals_db(all_signals, market, timeframe)
                                self.update_trailing_stops(market, df['close'].iloc[-1])

                    time.sleep(0.1)

            except Exception as e:
                self.logger.error(f"Scanning error: {e}")
                time.sleep(5)

    def validate_signals_with_trailing(self, signals, market):
        validated = []
        try:
            # Convert market symbol format
            symbol = market.replace('/', '')
            current_price = float(self.binance.fetch_ticker(symbol)['last'])
            
            for signal in signals:
                if market not in self.trailing_data['stop_levels']:
                    self.trailing_data['stop_levels'][market] = current_price * 0.99
                
                # Volume validation
                ticker = self.binance.fetch_ticker(symbol)
                volume = float(ticker['quoteVolume'])
                
                if volume >= self.signal_config['min_volume']:
                    signal['volume'] = volume
                    signal['trailing_stop'] = self.trailing_data['stop_levels'][market]
                    validated.append(signal)
                    
            return validated
            
        except Exception as e:
            self.logger.error(f"Signal validation error: {e}")
            return []


    def filter_signals(self, signals, market):
        filtered = []
        try:
            for signal in signals:
                # Get order book data
                order_book = self.binance.fetch_order_book(market)
                bid_depth = sum(bid[1] for bid in order_book['bids'][:10])
                ask_depth = sum(ask[1] for ask in order_book['asks'][:10])
                
                # Calculate imbalance
                imbalance = abs(bid_depth - ask_depth) / (bid_depth + ask_depth)
                
                # Filter criteria
                if (signal.get('strength', '') in ['strong', 'very strong'] and
                    signal.get('score', 0) >= 0.7 and
                    imbalance >= 0.2):  # 20% minimum imbalance
                    
                    signal['order_book_imbalance'] = imbalance
                    filtered.append(signal)
                    
            return filtered
            
        except Exception as e:
            self.logger.error(f"Signal filtering error: {e}")
            return []


    def check_volume_threshold(self, market):
        try:
            ticker = self.binance.fetch_ticker(market)
            volume = float(ticker['quoteVolume'])
            return volume >= self.signal_config['min_volume']
        except Exception as e:
            self.logger.error(f"Volume check error for {market}: {e}")
            return False

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
    def analyze_price_action(self, df):
        if df is None or df.empty:
            return {'trend': 'neutral'}
            
        # Calculate price action metrics
        df['sma20'] = df['close'].rolling(window=20).mean()
        df['sma50'] = df['close'].rolling(window=50).mean()
        
        latest_close = df['close'].iloc[-1]
        latest_sma20 = df['sma20'].iloc[-1]
        latest_sma50 = df['sma50'].iloc[-1]
        
        # Determine trend
        if latest_close > latest_sma20 and latest_sma20 > latest_sma50:
            trend = 'uptrend'
        elif latest_close < latest_sma20 and latest_sma20 < latest_sma50:
            trend = 'downtrend'
        else:
            trend = 'neutral'
            
        return {
            'trend': trend,
            'close': latest_close,
            'sma20': latest_sma20,
            'sma50': latest_sma50
        }


    def get_funding_rate(self, market):
        try:
            # Use standard futures API endpoint
            symbol = market.replace('/', '')
            funding_info = self.binance.futures_funding_rate(symbol=symbol)
            
            if funding_info:
                return {
                    'rate': float(funding_info[0]['fundingRate']),
                    'timestamp': funding_info[0]['fundingTime']
                }
            
            return {
                'rate': 0,
                'timestamp': None
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching funding rate for {market}: {e}")
            return {
                'rate': 0,
                'timestamp': None
            }


    def analyze_volume_profile(self, df):
        if df is None or df.empty:
            return None
                
        # Create a copy to avoid SettingWithCopyWarning
        df_copy = df.copy()
        
        # Calculate volume-weighted price levels using loc
        df_copy.loc[:, 'vwap'] = (df_copy['volume'] * df_copy['close']).cumsum() / df_copy['volume'].cumsum()
        
        # Identify key volume levels with explicit observed parameter
        volume_levels = pd.cut(df_copy['close'], bins=10)
        volume_profile = df_copy.groupby(volume_levels, observed=True)['volume'].sum()
        
        # Find high volume nodes
        high_volume_levels = volume_profile[volume_profile > volume_profile.mean()]
        
        # Calculate additional volume metrics
        volume_sma = df_copy['volume'].rolling(window=20).mean()
        volume_ratio = df_copy['volume'] / volume_sma
        
        return {
            'high_volume_levels': high_volume_levels.to_dict(),
            'vwap': df_copy['vwap'].iloc[-1],
            'volume_trend': 'increasing' if df_copy['volume'].is_monotonic_increasing else 'decreasing',
            'volume_ratio': volume_ratio.iloc[-1],
            'volume_sma': volume_sma.iloc[-1]
        }
    def analyze_key_levels(self, df):
        # Enhanced volume analysis at support/resistance
        volume_profile = {
            'support_volume': df['volume'].rolling(20).mean().iloc[-1],
            'resistance_volume': df['volume'].rolling(20).max().iloc[-1],
            'volume_surge': (df['volume'].iloc[-1] > df['volume'].rolling(20).mean().iloc[-1] * 2)
        }
        
        # Candle close analysis near support
        latest_candle = df.iloc[-1]
        range_size = latest_candle['high'] - latest_candle['low']
        
        candle_quality = {
            'close_to_support': (abs(latest_candle['close'] - latest_candle['low']) /
                            (range_size if range_size > 0 else 1)) < 0.2,
            'strong_rejection': (latest_candle['high'] - latest_candle['close'] <
                            latest_candle['close'] - latest_candle['low']),
            'volume_confirmation': (latest_candle['volume'] >
                                df['volume'].rolling(20).mean().iloc[-1] * 1.5)
        }
        
        return {
            'volume_profile': volume_profile,
            'candle_quality': candle_quality,
            'support_strength': all(candle_quality.values())
        }

    def validate_strong_buy(self, signal, df, key_levels, market):
        # Volume confirmation with enhanced criteria
        volume_strength = (
            key_levels['volume_profile']['volume_surge'] and
            key_levels['candle_quality']['volume_confirmation']
        )
        
        # Price action confirmation using existing quality metrics
        price_strength = (
            key_levels['candle_quality']['close_to_support'] and
            key_levels['candle_quality']['strong_rejection']
        )
        
        # Support level strength using multiple timeframe confirmation
        support_test = (
            key_levels['support_strength'] and
            self.multiple_timeframe_support(market)
        )
        
        # Calculate weighted strength score
        strength_score = (
            volume_strength * 0.4 +
            price_strength * 0.4 +
            support_test * 0.2
        )
        
        return {
            'is_strong_buy': strength_score >= 0.6,
            'strength_score': strength_score,
            'confirmation_factors': {
                'volume': volume_strength,
                'price_action': price_strength,
                'support': support_test
            }
        }

    def validate_strong_buy(self, signal, df, key_levels, market):
        # Volume confirmation - made more lenient
        volume_strength = (
            df['volume'].iloc[-1] > df['volume'].rolling(20).mean().iloc[-1] * 1.5  # Reduced from 2x to 1.5x
        )
        
        # Price action confirmation - more flexible conditions
        price_strength = (
            key_levels['candle_quality']['close_to_support'] or 
            key_levels['candle_quality']['strong_rejection']
        )
        
        # Support level strength - simplified
        support_test = self.is_at_support(df)
        
        # Calculate strength score
        strength_score = sum([
            volume_strength * 0.4,
            price_strength * 0.4,
            support_test * 0.2
        ])
        
        # More lenient strong buy condition
        is_strong = strength_score >= 0.6  # Reduced from implicit 1.0
        
        return {
            'is_strong_buy': is_strong,
            'strength_score': strength_score,
            'confirmation_factors': {
                'volume': volume_strength,
                'price_action': price_strength,
                'support': support_test
            }
        }

    def is_at_support(self, df):
        latest_candle = df.iloc[-1]
        support_level = df['low'].rolling(20).min().iloc[-1]
        return abs(latest_candle['low'] - support_level) / support_level < 0.002

    def multiple_timeframe_support(self, market):
        timeframes = ['15m', '1h', '4h']
        support_count = 0
        for tf in timeframes:
            df = self.fetch_market_data(market, tf)
            if df is not None and self.is_at_support(df):
                support_count += 1
        return support_count >= 2

    def validate_strong_buy(self, signal, df, key_levels, market):  # Added market parameter
        # Volume confirmation
        volume_strength = (
            df['volume'].iloc[-1] > df['volume'].rolling(20).mean().iloc[-1] * 2 and
            key_levels['volume_profile']['volume_surge']
        )
        
        # Price action confirmation
        price_strength = (
            key_levels['candle_quality']['close_to_support'] and
            key_levels['candle_quality']['strong_rejection']
        )
        
        # Support level strength
        support_test = (
            self.is_at_support(df) and
            self.multiple_timeframe_support(market)  # Now market is defined
        )
        
        return {
            'is_strong_buy': volume_strength and price_strength and support_test,
            'strength_score': sum([volume_strength, price_strength, support_test]) / 3,
            'confirmation_factors': {
                'volume': volume_strength,
                'price_action': price_strength,
                'support': support_test
            }
        }
    def get_signal_dots(self, signal_count):
        if signal_count <= 4:
            return "🟢" * signal_count
        else:
            return "🟣" * (signal_count - 3)  # Purple dots for excess signals

    def analyze_market_regime(self, df):
        regime_data = {
            'trend_strength': self.detect_trend_strength(df),
            'volatility': df['close'].pct_change().std() * np.sqrt(252),
            'volume_profile': self.analyze_volume_profile(df),
            'market_phase': self.detect_market_phase(df)
        }
        return regime_data

    def detect_market_phase(self, df):
        adx = talib.ADX(df['high'], df['low'], df['close'])
        volatility = df['close'].pct_change().std() * np.sqrt(252)
        
        if adx.iloc[-1] > 25:
            return 'Trending' if volatility > 0.2 else 'Low Volatility Trend'
        else:
            return 'Ranging' if volatility > 0.2 else 'Accumulation'



    def detect_htf_support_wick(self, market):
        # Get higher timeframe supports
        htf_data = {
            '4h': self.fetch_market_data(market, '4h', limit=100),
            '1d': self.fetch_market_data(market, '1d', limit=100),
            '1w': self.fetch_market_data(market, '1w', limit=100)
        }
        
        # Get lower timeframe data
        ltf_data = {
            '15m': self.fetch_market_data(market, '15m', limit=100),
            '30m': self.fetch_market_data(market, '30m', limit=100)
        }
        
        # Find support levels on higher timeframes
        supports = {}
        for tf, df in htf_data.items():
            if df is not None:
                supports[tf] = self.define_strong_support(df, market)
        
        # Check for wicks on lower timeframes near HTF supports
        for tf, df in ltf_data.items():
            if df is not None:
                last_candle = df.iloc[-1]
                wick_size = (last_candle['open'] - last_candle['low']) / last_candle['open']
                
                # Check if price is near any HTF support with significant wick
                for htf, support_levels in supports.items():
                    if support_levels['strength'] == 'Strong' or support_levels['strength'] == 'Very Strong':
                        for support_price in support_levels['details']['price_action_supports']:
                            if (abs(last_candle['low'] - support_price['price']) / support_price['price'] < 0.01 and 
                                wick_size > 0.003):  # 0.3% wick size threshold
                                return {
                                    'type': 'HTF_SUPPORT_WICK',
                                    'support_tf': htf,
                                    'wick_tf': tf,
                                    'support_price': support_price['price'],
                                    'wick_size': wick_size,
                                    'strength': support_levels['strength']
                                }
        return None
    def detect_support_drop_recovery(self, market):
        # Get daily and 4h data
        daily_data = self.fetch_market_data(market, '1d', limit=100)
        four_hour_data = self.fetch_market_data(market, '4h', limit=100)
        
        if daily_data is None:
            return None
                
        # Check for 7.5% drop
        recent_high = daily_data['high'].rolling(window=7).max().iloc[-1]
        current_price = daily_data['close'].iloc[-1]
        price_drop = (recent_high - current_price) / recent_high * 100
        
        if price_drop >= 7.5:
            # Find support levels
            supports = self.define_strong_support(daily_data, market)
            
            # Check for range formation
            last_3_days = daily_data.tail(3)
            if len(last_3_days) < 3:
                return None
                
            # Volume analysis
            volume_profile = self.analyze_volume_profile(last_3_days)
            
            if supports['strength'] in ['Strong', 'Very Strong'] and volume_profile['volume_ratio'] > 2:
                return {
                    'type': 'SUPPORT_DROP_RECOVERY',
                    'dropped_support': recent_high,
                    'next_support': supports['details']['price_action_supports'][0]['price'],
                    'days_below': 3,
                    'wick_size': (last_3_days['open'].iloc[-1] - last_3_days['low'].iloc[-1]) / last_3_days['open'].iloc[-1],
                    'strength': supports['strength']
                }
        
        return None
    def detect_range_breakout_wick(self, market):
        # Get data for both timeframes
        tf_30m = self.fetch_market_data(market, '30m', limit=200)
        tf_4h = self.fetch_market_data(market, '4h', limit=100)
        
        for df in [tf_30m, tf_4h]:
            if df is None or df.empty:
                continue
                
            # Identify ranging area
            df['high_range'] = df['high'].rolling(window=20).max()
            df['low_range'] = df['low'].rolling(window=20).min()
            df['range_size'] = (df['high_range'] - df['low_range']) / df['low_range'] * 100
            
            # Check for historical selling pressure
            df['down_pressure'] = df['close'] < df['open']
            selling_pressure = df['down_pressure'].rolling(window=20).sum() > 12  # More than 60% bearish candles
            
            # Latest candle analysis
            last_candle = df.iloc[-1]
            prev_candle = df.iloc[-2]
            
            # Check for breakout with upper wick
            if (last_candle['close'] > df['high_range'].iloc[-2] and  # Closed above range
                last_candle['high'] > last_candle['close'] and  # Has upper wick
                selling_pressure.iloc[-1]):  # Historical selling pressure exists
                
                # Calculate wick size
                wick_size = (last_candle['high'] - last_candle['close']) / (last_candle['high'] - last_candle['low'])
                
                if wick_size > 0.3:  # Significant wick (>30% of candle)
                    timeframe = '30m' if df.equals(tf_30m) else '4h'
                    return {
                        'type': 'RANGE_BREAKOUT_WICK',
                        'timeframe': timeframe,
                        'range_high': df['high_range'].iloc[-2],
                        'breakout_price': last_candle['close'],
                        'wick_size': wick_size,
                        'range_size': df['range_size'].iloc[-1],
                        'selling_pressure': df['down_pressure'].rolling(window=20).sum().iloc[-1],
                        'strength': 'Strong' if wick_size > 0.5 else 'Moderate'
                    }
        
        return None





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
    def apply_vsa_support_analysis(self, df):
        if df is None or df.empty:
            return 0
            
        vsa_score = 0
        
        # Calculate volume metrics
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['price_range'] = df['high'] - df['low']
        df['range_ma'] = df['price_range'].rolling(window=20).mean()
        
        # Analyze last few candles
        last_candles = df.tail(3)
        
        for _, candle in last_candles.iterrows():
            # High volume with narrow range near support (absorption)
            if (candle['volume'] > candle['volume_ma'] * 2 and 
                candle['price_range'] < candle['range_ma'] * 0.5):
                vsa_score += 1
                
            # Strong bullish close with above average volume
            if (candle['close'] > candle['open'] and
                candle['volume'] > candle['volume_ma'] * 1.5):
                vsa_score += 0.5
                
        return vsa_score

    # Add new support analysis methods here
    def define_strong_support(self, df, market):
        support_analysis = {
            'price_action_supports': self.detect_price_action_supports(df),
            'vsa_support': self.apply_vsa_support_analysis(df),
            'multi_timeframe_supports': self.confirm_multi_timeframe_supports(market),
            'order_book_supports': self.verify_support_with_order_book(market)
        }
        return self.synthesize_support_strength(support_analysis)

    def detect_price_action_supports(self, df):
        support_levels = []
        for window in [20, 50, 100]:  # Multiple timeframe windows
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
        return support_levels

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

    def synthesize_support_strength(self, support_analysis):
        total_strength = (
            len(support_analysis['price_action_supports']) * 0.3 +
            support_analysis['vsa_support'] * 0.25 +
            support_analysis['multi_timeframe_supports'] * 0.25 +
            (support_analysis['order_book_supports']['total_bid_volume'] > 100) * 0.2
        )
        
        return {
            'strength': 'Very Strong' if total_strength > 0.8 else
                       'Strong' if total_strength > 0.6 else
                       'Moderate' if total_strength > 0.4 else 'Weak',
            'score': total_strength,
            'details': support_analysis
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
    def classify_trading_style(self, signal_data, timeframe):
        day_trading_timeframes = ['1m', '5m', '15m', '30m', '1h']
        swing_trading_timeframes = ['4h', '1d', '1w']
        
        if timeframe in day_trading_timeframes:
            signal_data['trading_style'] = 'Day Trade 📈'
            signal_data['position_holding'] = '1-8 hours'
        else:
            signal_data['trading_style'] = 'Swing Trade 🌊'
            signal_data['position_holding'] = '1-7 days'
        
        return signal_data
    def validate_timeframe_strategy(self, signal, timeframe):
        if timeframe in ['1m', '5m', '15m']:
            return signal['type'] in ['VSA_COMPRESSION', 'RANGE_BREAKOUT_WICK']
        elif timeframe in ['1h', '4h']:
            return signal['type'] in ['HTF_SUPPORT_WICK', 'DOUBLE_BOTTOM', 'TRIPLE_BOTTOM']
        else:
            return signal['type'] in ['SUPPORT_DROP_RECOVERY', 'CUP_AND_HANDLE']

    def calculate_strategy_risk(self, signal, timeframe):
        if signal['trading_style'] == 'Day Trade 📈':
            return {
                'stop_loss': 0.5,  # 0.5% for day trades
                'take_profit': 1.5  # 1.5% for day trades
            }
        else:
            return {
                'stop_loss': 2.0,  # 2% for swing trades
                'take_profit': 6.0  # 6% for swing trades
            }
    def generate_signals(self, df):
        signals = []
        try:
            # Calculate technical indicators
            df = self.calculate_technical_indicators(df)
            
            # RSI signals
            if df['rsi'].iloc[-1] < 30:
                signals.append({
                    'type': 'RSI_OVERSOLD',
                    'price': df['close'].iloc[-1],
                    'strength': 'strong',
                    'score': 0.8
                })
            
            # EMA crossover
            if df['ema_20'].iloc[-1] > df['ema_50'].iloc[-1] and \
            df['ema_20'].iloc[-2] <= df['ema_50'].iloc[-2]:
                signals.append({
                    'type': 'EMA_CROSS',
                    'price': df['close'].iloc[-1],
                    'strength': 'moderate',
                    'score': 0.7
                })
            
            # VWAP analysis
            df = self.calculate_vwap_levels(df)
            if df['close'].iloc[-1] < df['vwap_lower'].iloc[-1]:
                signals.append({
                    'type': 'VWAP_SUPPORT',
                    'price': df['close'].iloc[-1],
                    'strength': 'strong',
                    'score': 0.75
                })
            
            # Pattern detection
            if pattern_signals := self.analyze_pattern_sequence(df):
                signals.extend(pattern_signals)
            
            # Volume analysis
            if volume_signals := self.analyze_volume_patterns(df):
                signals.extend(volume_signals)
            
            # Divergence detection
            if divergence_signals := self.detect_divergences(df):
                signals.extend(divergence_signals)
                
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return []


    def process_signals(self, market, timeframe, signals):
        for signal in signals:
            try:
                # Classify trading style and calculate additional metrics
                signal = self.classify_trading_style(signal, timeframe)
                risk_level = self.calculate_risk_level(signal)
                
                # Get technical levels
                df = self.fetch_market_data(market, timeframe)
                vwap_data = self.calculate_vwap_levels(df)
                pivot_points = self.calculate_pivot_points(df)
                fib_levels = self.calculate_fibonacci_levels(df)
                
                # Check for high volume wick
                wick_signal = self.detect_high_volume_wick(market, df)
                if wick_signal:
                    signals.append(wick_signal)
                
                # Log signal detection
                self.logger.info(f"New signal detected for {market}: {signal['type']}")
                
                # Store in database
                self.store_signals_db([signal], market, timeframe)
                
                # Construct enhanced message based on signal type
                if signal['type'] == 'HIGH_VOLUME_WICK':
                    message = (
                        f"🕐 {signal['trading_style']}\n"
                        f"High Volume Wick Signal\n"
                        f"Market: {market}\n"
                        f"Time: {signal['time']}\n"
                        f"Price: {signal['price']:.8f}\n"
                        f"Upper Wick Ratio: {signal['upper_wick_ratio']:.2f}\n"
                        f"Lower Wick Ratio: {signal['lower_wick_ratio']:.2f}\n"
                        f"Volume: {signal['volume_ratio']:.2f}x average\n"
                        f"Strength: {signal['strength']}\n"
                        f"Risk Level: {risk_level}\n"
                        f"VWAP: {vwap_data['vwap'].iloc[-1]:.8f}\n"
                        f"Next Support: {pivot_points['s1']:.8f}"
                    )
                
                elif signal['type'] == 'HTF_SUPPORT_WICK':
                    message = (
                        f"🎯 {signal['trading_style']}\n"
                        f"HTF Support Wick Signal\n"
                        f"Market: {market}\n"
                        f"Support TF: {signal['support_tf']}\n"
                        f"Wick TF: {signal['wick_tf']}\n"
                        f"Support: {signal['support_price']:.8f}\n"
                        f"Wick Size: {signal['wick_size']:.2%}\n"
                        f"Strength: {signal['strength']}\n"
                        f"Risk Level: {risk_level}\n"
                        f"VWAP: {vwap_data['vwap'].iloc[-1]:.8f}\n"
                        f"Next Pivot: {pivot_points['r1']:.8f}"
                    )
                
                elif signal['type'] == 'SUPPORT_DROP_RECOVERY':
                    message = (
                        f"📊 {signal['trading_style']}\n"
                        f"Support Drop Recovery Signal\n"
                        f"Market: {market}\n"
                        f"Dropped Support: {signal['dropped_support']:.8f}\n"
                        f"Next Support: {signal['next_support']:.8f}\n"
                        f"Days Below: {signal['days_below']}\n"
                        f"Wick Size: {signal['wick_size']:.2%}\n"
                        f"Strength: {signal['strength']}\n"
                        f"Risk Level: {risk_level}\n"
                        f"Fib Level: {fib_levels['level_618']:.8f}"
                    )
                
                elif signal['type'] == 'RANGE_BREAKOUT_WICK':
                    message = (
                        f"⚡ {signal['trading_style']}\n"
                        f"Range Breakout Wick Signal\n"
                        f"Market: {market}\n"
                        f"Timeframe: {signal['timeframe']}\n"
                        f"Breakout Price: {signal['breakout_price']:.8f}\n"
                        f"Range High: {signal['range_high']:.8f}\n"
                        f"Wick Size: {signal['wick_size']:.2%}\n"
                        f"Range Size: {signal['range_size']:.2f}%\n"
                        f"Selling Pressure: {signal['selling_pressure']} of 20\n"
                        f"Signal Strength: {signal['strength']}\n"
                        f"Risk Level: {risk_level}\n"
                        f"VWAP: {vwap_data['vwap'].iloc[-1]:.8f}\n"
                        f"Next Resistance: {pivot_points['r1']:.8f}"
                    )
                
                elif signal['type'] == 'VSA_COMPRESSION':
                    message = (
                        f"📊 {signal['trading_style']}\n"
                        f"Volume Spread Analysis Signal\n"
                        f"Market: {market}\n"
                        f"Price: {signal['price']:.8f}\n"
                        f"Volume Ratio: {signal.get('volume_ratio', 0):.2f}x\n"
                        f"Spread: {signal.get('spread', 0):.2%}\n"
                        f"Strength: {signal['strength']}\n"
                        f"Risk Level: {risk_level}\n"
                        f"VWAP: {vwap_data['vwap'].iloc[-1]:.8f}"
                    )
                
                elif signal['type'] in ['double_bottom', 'DOUBLE_BOTTOM', 'TRIPLE_BOTTOM']:
                    message = (
                        f"📈 {signal['trading_style']}\n"
                        f"{signal['type'].replace('_', ' ').title()} Pattern\n"
                        f"Market: {market}\n"
                        f"Price: {signal.get('price', df['close'].iloc[-1]):.8f}\n"
                        f"Pattern Score: {signal.get('score', 0):.2f}\n"
                        f"Strength: {signal.get('strength', 'strong')}\n"
                        f"Risk Level: {risk_level}\n"
                        f"VWAP: {vwap_data['vwap'].iloc[-1]:.8f}\n"
                        f"Fib Target: {fib_levels['level_618']:.8f}"
                    )
                
                elif signal['type'] == 'CUP_AND_HANDLE':
                    message = (
                        f"☕ {signal['trading_style']}\n"
                        f"Cup and Handle Pattern\n"
                        f"Market: {market}\n"
                        f"Price: {signal['price']:.8f}\n"
                        f"Target: {signal.get('target', 0):.8f}\n"
                        f"Strength: {signal['strength']}\n"
                        f"Risk Level: {risk_level}\n"
                        f"VWAP: {vwap_data['vwap'].iloc[-1]:.8f}"
                    )
                
                elif signal['type'] == 'BULLISH_DIVERGENCE':
                    message = (
                        f"↗️ {signal['trading_style']}\n"
                        f"Bullish Divergence\n"
                        f"Market: {market}\n"
                        f"Price: {signal['price']:.8f}\n"
                        f"RSI: {signal.get('rsi', 0):.2f}\n"
                        f"Strength: {signal['strength']}\n"
                        f"Risk Level: {risk_level}\n"
                        f"VWAP: {vwap_data['vwap'].iloc[-1]:.8f}"
                    )
                
                else:
                    message = (
                        f"🔔 {signal['trading_style']}\n"
                        f"Signal Alert!\n"
                        f"Market: {market}\n"
                        f"Signal: {signal['type']}\n"
                        f"Price: {signal.get('price', df['close'].iloc[-1]):.8f}\n"
                        f"Timeframe: {timeframe}\n"
                        f"Risk Level: {risk_level}\n"
                        f"VWAP: {vwap_data['vwap'].iloc[-1]:.8f}\n"
                        f"Pivot Points:\n"
                        f"R1: {pivot_points['r1']:.8f}\n"
                        f"S1: {pivot_points['s1']:.8f}\n"
                        f"Volume Profile: {signal.get('volume_context', 'N/A')}"
                    )
                
                # Send telegram notification
                if self.tel_id and self.bot_token:
                    self.logger.info(f"Sending signal notification for {market}")
                    self.send_telegram_update(message)
                    self.logger.info(f"Signal notification sent for {market}")
                    
            except Exception as e:
                self.logger.error(f"Signal processing error for {market}: {e}")

    def calculate_risk_level(self, signal):
        risk_score = 0
        
        # Volume-based risk
        if signal.get('volume_ratio', 0) > 2.0:
            risk_score += 1
        
        # Trend strength risk
        if signal.get('trend_strength', 0) > 25:
            risk_score += 1
        
        # Pattern reliability
        if signal.get('confirmation_count', 0) >= 2:
            risk_score += 1
        
        # Market volatility
        if signal.get('atr_ratio', 0) > 1.5:
            risk_score += 1
        
        risk_levels = {
            0: "Low Risk 🟢",
            1: "Moderate Risk 🟡",
            2: "Medium Risk 🟠",
            3: "High Risk 🔴",
            4: "Very High Risk ⛔"
        }
        
        return risk_levels.get(risk_score, "Unknown Risk ⚠️")

    def calculate_position_size(self, signal, risk_percentage=1.0):
        account_balance = float(self.binance.fetch_balance()['total']['USDT'])
        risk_amount = account_balance * (risk_percentage / 100)
        
        stop_loss = self.calculate_stop_loss(signal)
        entry_price = float(signal['price'])
        
        if stop_loss:
            risk_per_unit = abs(entry_price - stop_loss)
            position_size = risk_amount / risk_per_unit
            return round(position_size, 8)
        return None
    def analyze_order_flow(self, market):
        order_book = self.binance.fetch_order_book(market)
        trades = self.binance.fetch_trades(market, limit=100)
        
        buy_volume = sum(trade['amount'] for trade in trades if trade['side'] == 'buy')
        sell_volume = sum(trade['amount'] for trade in trades if trade['side'] == 'sell')
        
        bid_depth = sum(bid[1] for bid in order_book['bids'][:10])
        ask_depth = sum(ask[1] for ask in order_book['asks'][:10])
        
        return {
            'buy_pressure': buy_volume / (buy_volume + sell_volume),
            'bid_ask_ratio': bid_depth / ask_depth,
            'depth_imbalance': (bid_depth - ask_depth) / (bid_depth + ask_depth)
        }

    def detect_session_momentum(self, market, timeframe):
        df = self.fetch_market_data(market, timeframe)
        session_data = {
            'asian': df.between_time('00:00', '08:00'),
            'london': df.between_time('08:00', '16:00'),
            'new_york': df.between_time('13:00', '21:00')
        }
        
        momentum_scores = {}
        for session, data in session_data.items():
            momentum_scores[session] = {
                'volume': data['volume'].mean(),
                'volatility': data['close'].pct_change().std(),
                'trend': data['close'].iloc[-1] - data['open'].iloc[0]
            }
        
        return momentum_scores

    def integrate_analysis_components(self, market, timeframe):
        # Collect all analysis data
        order_flow = self.analyze_order_flow(market)
        session_data = self.detect_session_momentum(market, timeframe)
        market_regime = self.analyze_market_regime(self.fetch_market_data(market, timeframe))
        
        # Combine with technical signals
        signals = self.generate_signals(self.fetch_market_data(market, timeframe))
        
        for signal in signals:
            signal.update({
                'order_flow_data': order_flow,
                'session_momentum': session_data,
                'market_regime': market_regime,
                'position_size': self.calculate_position_size(signal),
                'risk_metrics': self.calculate_risk_level(signal)
            })
        
        return signals

    def process_integrated_signals(self, market, timeframe):
        signals = self.integrate_analysis_components(market, timeframe)
        
        for signal in signals:
            # Update trade journal
            self.update_trade_journal(signal)
            
            # Update performance metrics
            self.update_performance_metrics(market, signal)
            
            # Run through backtesting engine
            backtest_results = self.run_backtest(signal)
            
            # Update dashboard
            self.update_performance_dashboard(signal, backtest_results)
    def calculate_win_rate(self):
        trades = self.sql_operations('fetch', self.db_signals, 'Signals')
        if not trades:
            return 0.0
            
        winning_trades = sum(1 for trade in trades if trade.get('pnl', 0) > 0)
        return (winning_trades / len(trades)) * 100

    def calculate_profit_factor(self):
        trades = self.sql_operations('fetch', self.db_signals, 'Signals')
        if not trades:
            return 0.0
            
        gross_profit = sum(trade.get('pnl', 0) for trade in trades if trade.get('pnl', 0) > 0)
        gross_loss = abs(sum(trade.get('pnl', 0) for trade in trades if trade.get('pnl', 0) < 0))
        
        return gross_profit / gross_loss if gross_loss else 0.0

    def calculate_sharpe_ratio(self):
        trades = self.sql_operations('fetch', self.db_signals, 'Signals')
        if not trades:
            return 0.0
            
        returns = [trade.get('pnl', 0) for trade in trades]
        if not returns:
            return 0.0
            
        return (np.mean(returns) / np.std(returns)) * np.sqrt(252)

    def calculate_drawdown(self):
        trades = self.sql_operations('fetch', self.db_signals, 'Signals')
        if not trades:
            return 0.0
            
        cumulative_returns = np.cumsum([trade.get('pnl', 0) for trade in trades])
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        
        return abs(min(drawdown)) * 100

    def calculate_var(self, confidence_level=0.95):
        trades = self.sql_operations('fetch', self.db_signals, 'Signals')
        if not trades:
            return 0.0
            
        returns = [trade.get('pnl', 0) for trade in trades]
        return np.percentile(returns, (1 - confidence_level) * 100)

    def calculate_exposure(self):
        active_positions = self.binance.fetch_positions()
        total_exposure = sum(abs(float(pos['notional'])) for pos in active_positions if pos['notional'])
        return total_exposure


    def store_signals_db(self, signals, market, timeframe):
        for signal in signals:
            self.sql_operations('insert', self.db_signals, 'Signals',
                market=market,
                timeframe=timeframe,
                signal_type=signal['type'],
                price=signal.get('price', 0),
                volume_trend=signal.get('volume_context', ''),
                vwap=signal.get('vwap', 0.0),
                rsi=signal.get('rsi', 0.0),
                timestamp=str(datetime.now())
            )

    def send_telegram_update(self, message):
        if self.tel_id and self.bot_token:
            try:
                # Format message with proper encoding
                formatted_message = message.encode('utf-8').decode('utf-8')
                
                # Construct API URL
                url = f'https://api.telegram.org/bot{self.bot_token}/sendMessage'
                
                # Setup request parameters
                params = {
                    'chat_id': self.tel_id,
                    'text': formatted_message,
                    'parse_mode': 'HTML',
                    'disable_web_page_preview': True
                }
                
                # Add request headers
                headers = {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                }
                
                # Send request with timeout
                response = requests.post(
                    url, 
                    json=params, 
                    headers=headers, 
                    timeout=10
                )
                
                # Log response details
                self.logger.info(f"Telegram message status: {response.status_code}")
                
                if response.status_code == 200:
                    self.logger.info(f"Signal notification sent for message: {message[:100]}...")
                else:
                    self.logger.error(f"Telegram API error: {response.text}")
                    
            except requests.exceptions.Timeout:
                self.logger.error("Telegram request timed out")
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Telegram request error: {e}")
            except Exception as e:
                self.logger.error(f"Telegram sending error: {e}")

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
    def filter_by_volume(self, market):
        try:
            ticker = self.binance.fetch_ticker(market.replace('/', ''))
            volume_usd = float(ticker['quoteVolume'])
            return volume_usd > 1000000  # $1M minimum volume
        except Exception as e:
            self.logger.error(f"Volume filter error: {e}")
            return False

    def validate_trend_strength(self, df):
        try:
            adx = talib.ADX(df['high'], df['low'], df['close'])
            rsi = talib.RSI(df['close'])
            trend_strength = adx.iloc[-1]
            rsi_value = rsi.iloc[-1]
            return trend_strength > 25 and (rsi_value < 30 or rsi_value > 70)
        except Exception as e:
            self.logger.error(f"Trend validation error: {e}")
            return False

    def check_signal_cooldown(self, market):
        current_time = time.time()
        if not hasattr(self, 'last_signal_time'):
            self.last_signal_time = {}
        if market in self.last_signal_time:
            if current_time - self.last_signal_time[market] < 3600:
                return False
        self.last_signal_time[market] = current_time
        return True

    def confirm_multiple_timeframes(self, market):
        timeframes = ['15m', '1h', '4h']
        confirmations = 0
        for tf in timeframes:
            df = self.fetch_market_data(market, tf)
            if df is not None and self.validate_trend_strength(df):
                confirmations += 1
        return confirmations >= 2

    def detect_high_volume_wick(self, market, df):
        try:
            current_candle = df.iloc[-1]
            
            # Calculate wick sizes
            upper_wick = current_candle['high'] - max(current_candle['open'], current_candle['close'])
            lower_wick = min(current_candle['open'], current_candle['close']) - current_candle['low']
            
            # Calculate wick ratios
            candle_body = abs(current_candle['close'] - current_candle['open'])
            upper_wick_ratio = upper_wick / candle_body if candle_body != 0 else 0
            lower_wick_ratio = lower_wick / candle_body if candle_body != 0 else 0
            
            # Volume analysis
            volume_ratio = current_candle['volume'] / df['volume'].rolling(20).mean().iloc[-1]
            
            # Signal conditions
            if volume_ratio > 2.0 and (upper_wick_ratio > 1.5 or lower_wick_ratio > 1.5):
                return {
                    'type': 'HIGH_VOLUME_WICK',
                    'price': current_candle['close'],
                    'strength': 'strong' if volume_ratio > 3.0 else 'moderate',
                    'score': 0.9 if volume_ratio > 3.0 else 0.7,
                    'upper_wick_ratio': upper_wick_ratio,
                    'lower_wick_ratio': lower_wick_ratio,
                    'volume_ratio': volume_ratio,
                    'time': pd.to_datetime(current_candle['timestamp']).strftime('%H:%M UTC')
                }
        except Exception as e:
            self.logger.error(f"High volume wick detection error: {e}")
        return None

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