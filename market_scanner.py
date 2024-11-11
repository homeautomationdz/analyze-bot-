import ccxt
import pandas as pd
import talib
from datetime import datetime
import time
from base import BaseScanner

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
                        signals = self.generate_signals(df)
                        validated_signals = self.validate_signals_with_trailing(signals, market)
                        filtered_signals = self.filter_signals(validated_signals, market)

                        if filtered_signals:
                            self.process_signals(market, timeframe, filtered_signals)
                            self.store_signals_db(filtered_signals, market, timeframe)
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
            if tech['rsi_sentiment'] == 'oversold': score += weights['technical_signals'] * 0.4
            if tech['macd_sentiment'] == 'bullish': score += weights['technical_signals'] * 0.3
            if tech['stoch_sentiment'] == 'oversold': score += weights['technical_signals'] * 0.3

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
            # Calculate technical indicators
            df['rsi'] = talib.RSI(df['close'])
            df['ema_20'] = talib.EMA(df['close'], timeperiod=20)
            df['ema_50'] = talib.EMA(df['close'], timeperiod=50)

            # Check for bullish conditions
            if df['rsi'].iloc[-1] < 30:  # Oversold
                signals.append({
                    'type': 'RSI_OVERSOLD',
                    'price': df['close'].iloc[-1]
                })

            if df['ema_20'].iloc[-1] > df['ema_50'].iloc[-1] and \
                    df['ema_20'].iloc[-2] <= df['ema_50'].iloc[-2]:  # Golden Cross
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
            self.sql_operations('insert', self.db_signals, 'Signals',
                                market=market,
                                timeframe=timeframe,
                                signal_type=signal['type'],
                                price=signal['price'],
                                timestamp=str(datetime.now()))

            message = f"Signal: {signal['type']}\nMarket: {market}\nTimeframe: {timeframe}\nPrice: {signal['price']}"
            self.send_telegram_update(message)
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
