import ccxt
import pandas as pd
import talib
from datetime import datetime
import time
from base import BaseScanner
import requests

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
                        # Original signal generation
                        signals = self.generate_signals(df)
                        validated_signals = self.validate_signals_with_trailing(signals, market)
                        filtered_signals = self.filter_signals(validated_signals, market)

                        # New strategy signals
                        htf_wick_signal = self.detect_htf_support_wick(market)
                        drop_recovery_signal = self.detect_support_drop_recovery(market)

                        # Combine all signals
                        all_signals = []
                        if filtered_signals:
                            all_signals.extend(filtered_signals)
                        if htf_wick_signal:
                            all_signals.append(htf_wick_signal)
                        if drop_recovery_signal:
                            all_signals.append(drop_recovery_signal)

                        # Process all signals together
                        if all_signals:
                            self.process_signals(market, timeframe, all_signals)
                            self.store_signals_db(all_signals, market, timeframe)
                            self.update_trailing_stops(market, df['close'].iloc[-1])

                    time.sleep(0.1)

            except Exception as e:
                self.logger.error(f"Scanning error: {e}")
                time.sleep(5)

    def validate_signals_with_trailing(self, signals, market):
        validated = []
        current_price = float(self.binance.fetch_ticker(market)['last'])

        for signal in signals:
            if market not in self.trailing_data['stop_levels']:
                self.trailing_data['stop_levels'][market] = current_price * 0.99  # 1% initial stop

            if signal['type'] in ['RSI_OVERSOLD', 'BB_OVERSOLD', 'EMA_GOLDEN_CROSS']:
                if current_price > self.trailing_data['stop_levels'][market]:
                    signal['trailing_stop'] = self.trailing_data['stop_levels'][market]
                    validated.append(signal)

        return validated
    def filter_signals(self, signals, market):
        filtered = []
        for signal in signals:
            # Apply volume filter
            if self.check_volume_threshold(market):
                # Apply sentiment filter
                sentiment = self.analyze_market_sentiment(market)
                if sentiment['sentiment_score'] >= 0.6:
                    # Add additional signal metadata
                    signal['sentiment'] = sentiment
                    signal['volume_profile'] = self.analyze_volume_profile(
                        self.fetch_market_data(market, '1h', limit=100)
                    )
                    filtered.append(signal)
        return filtered

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
        # Get daily data
        daily_data = self.fetch_market_data(market, '1d', limit=100)
        if daily_data is None:
            return None
            
        # Find last two higher supports
        supports = self.define_strong_support(daily_data, market)
        support_levels = sorted([s['price'] for s in supports['details']['price_action_supports']], reverse=True)
        
        if len(support_levels) < 2:
            return None
        
        # Check for 3-day drop
        last_3_days = daily_data.tail(3)
        if all(last_3_days['close'] < support_levels[1]):
            # Check if price is approaching next support
            current_price = last_3_days['close'].iloc[-1]
            if len(support_levels) > 2:
                next_support = support_levels[2]
                
                # Check for wick formation near next support
                if (abs(current_price - next_support) / next_support < 0.02):  # Within 2% of support
                    wick_size = (last_3_days['open'].iloc[-1] - last_3_days['low'].iloc[-1]) / last_3_days['open'].iloc[-1]
                    
                    if wick_size > 0.003:  # 0.3% wick size threshold
                        return {
                            'type': 'SUPPORT_DROP_RECOVERY',
                            'dropped_support': support_levels[1],
                            'next_support': next_support,
                            'days_below': 3,
                            'wick_size': wick_size,
                            'strength': supports['strength']
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

    def generate_signals(self, df):
        signals = []
        try:
            # Technical indicators
            df['rsi'] = talib.RSI(df['close'])
            df['ema_20'] = talib.EMA(df['close'], timeperiod=20)
            df['ema_50'] = talib.EMA(df['close'], timeperiod=50)
            
            # VSA analysis
            vsa_signals = self.analyze_vsa_signals(df)
            signals.extend(vsa_signals)
            
            # Trap detection
            if self.detect_bull_trap(df):
                signals.append({
                    'type': 'BULL_TRAP',
                    'price': df['close'].iloc[-1],
                    'strength': 'high'
                })
                
            if self.detect_bear_trap(df):
                signals.append({
                    'type': 'BEAR_TRAP',
                    'price': df['close'].iloc[-1],
                    'strength': 'high'
                })
            
            # Traditional signals
            if df['rsi'].iloc[-1] < 30:
                signals.append({
                    'type': 'RSI_OVERSOLD',
                    'price': df['close'].iloc[-1]
                })
                
            if df['ema_20'].iloc[-1] > df['ema_50'].iloc[-1] and \
            df['ema_20'].iloc[-2] <= df['ema_50'].iloc[-2]:
                signals.append({
                    'type': 'EMA_CROSS',
                    'price': df['close'].iloc[-1]
                })
            
            return signals
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return []

    def process_signals(self, market, timeframe, signals):
        for signal in signals:
            try:
                # Log signal detection
                self.logger.info(f"New signal detected for {market}: {signal['type']}")
                
                # Store in database first
                self.store_signals_db([signal], market, timeframe)
                self.logger.info(f"Signal stored in database for {market}")
                
                # Construct message based on signal type
                if signal['type'] == 'HTF_SUPPORT_WICK':
                    message = (
                        f"🎯 HTF Support Wick Signal\n"
                        f"Market: {market}\n"
                        f"Support Timeframe: {signal['support_tf']}\n"
                        f"Wick Timeframe: {signal['wick_tf']}\n"
                        f"Support Price: {signal['support_price']:.8f}\n"
                        f"Wick Size: {signal['wick_size']:.2%}\n"
                        f"Support Strength: {signal['strength']}"
                    )
                elif signal['type'] == 'SUPPORT_DROP_RECOVERY':
                    message = (
                        f"📊 Support Drop Recovery Signal\n"
                        f"Market: {market}\n"
                        f"Dropped Support: {signal['dropped_support']:.8f}\n"
                        f"Next Support: {signal['next_support']:.8f}\n"
                        f"Days Below: {signal['days_below']}\n"
                        f"Wick Size: {signal['wick_size']:.2%}\n"
                        f"Support Strength: {signal['strength']}"
                    )
                else:
                    message = (
                        f"🔔 Signal Alert!\n"
                        f"Market: {market}\n"
                        f"Signal: {signal['type']}\n"
                        f"Price: {signal.get('price', 'N/A')}\n"
                        f"Timeframe: {timeframe}"
                    )
                
                # Send telegram notification with logging
                if self.tel_id and self.bot_token:
                    self.logger.info(f"Sending signal notification for {market}")
                    self.send_telegram_update(message)
                    self.logger.info(f"Signal notification sent for {market}")
                
            except Exception as e:
                self.logger.error(f"Signal processing error for {market}: {e}")


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
                url = f'https://api.telegram.org/bot{self.bot_token}/sendMessage'
                params = {
                    'chat_id': self.tel_id,
                    'text': message,
                    'parse_mode': 'HTML'
                }
                self.logger.info(f"Sending telegram message: {message}")
                response = requests.post(url, params=params)
                if response.ok:
                    self.logger.info("Telegram message sent successfully")
                else:
                    self.logger.error(f"Telegram API error: {response.text}")
            except Exception as e:
                self.logger.error(f"Telegram error: {e}")

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