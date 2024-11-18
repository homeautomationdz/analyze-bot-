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
                        # Signal generation and validation
                        signals = self.generate_signals(df)
                        validated_signals = self.validate_signals_with_trailing(signals, market)
                        filtered_signals = self.filter_signals(validated_signals, market)

                        # Strategy signals
                        htf_wick_signal = self.detect_htf_support_wick(market)
                        drop_recovery_signal = self.detect_support_drop_recovery(market)

                        # Combine signals
                        all_signals = []
                        if filtered_signals:
                            all_signals.extend(filtered_signals)
                        if htf_wick_signal:
                            all_signals.append(htf_wick_signal)
                        if drop_recovery_signal:
                            all_signals.append(drop_recovery_signal)

                        # Process signals
                        if all_signals:
                            self.process_signals(market, timeframe, all_signals)
                            self.store_signals_db(all_signals, market, timeframe)
                            self.update_trailing_stops(market, df['close'].iloc[-1])

                            # Update UI components
                            self.master.after(100, self.update_signals_display)
                            self.master.after(100, self.update_performance_metrics)
                            self.master.after(100, self.update_market_overview)

                    time.sleep(0.1)

            except Exception as e:
                self.logger.error(f"Scanning error: {e}")
                time.sleep(5)





    def find_liquidity_clusters(self, orders):
        clusters = []
        volume_threshold = orders['volume'].mean() * 2
        
        for price, group in orders.groupby(pd.cut(orders['price'], bins=50)):
            total_volume = group['volume'].sum()
            if total_volume > volume_threshold:
                clusters.append({
                    'price': price.mid,
                    'volume': total_volume,
                    'orders_count': len(group)
                })
        
        return clusters

    def calculate_order_imbalance(self, order_book):
        bid_volume = sum(bid[1] for bid in order_book['bids'][:10])
        ask_volume = sum(ask[1] for ask in order_book['asks'][:10])
        
        imbalance_ratio = bid_volume / (bid_volume + ask_volume)
        return {
            'ratio': imbalance_ratio,
            'bid_volume': bid_volume,
            'ask_volume': ask_volume,
            'bias': 'bullish' if imbalance_ratio > 0.6 else 'bearish' if imbalance_ratio < 0.4 else 'neutral'
        }


    def track_large_orders(self, trades):
        df = pd.DataFrame(trades)
        volume_threshold = df['amount'].mean() * 3
        
        large_trades = df[df['amount'] > volume_threshold].copy()
        large_trades['impact'] = self.calculate_price_impact(large_trades)
        
        return large_trades.to_dict('records')

    def detect_iceberg_orders(self, market):
        timeframes = ['1m', '5m', '15m']
        iceberg_patterns = []
        
        for tf in timeframes:
            df = self.fetch_market_data(market, tf)
            if df is not None:
                patterns = self.analyze_iceberg_patterns(df)
                iceberg_patterns.extend(patterns)
        
        return iceberg_patterns

    def find_accumulation_zones(self, market):
        df = self.fetch_market_data(market, '1h', limit=200)
        zones = []
        
        if df is not None:
            df['volume_ma'] = df['volume'].rolling(20).mean()
            df['price_range'] = df['high'] - df['low']
            
            for i in range(20, len(df)):
                if self.is_accumulation_zone(df.iloc[i-20:i]):
                    zones.append({
                        'price': df['close'].iloc[i],
                        'volume_increase': df['volume'].iloc[i] / df['volume_ma'].iloc[i],
                        'strength': self.calculate_zone_strength(df.iloc[i-20:i])
                    })
        
        return zones

    def identify_institutional_levels(self, market):
        df = self.fetch_market_data(market, '4h', limit=500)
        levels = []
        
        if df is not None:
            # Find high volume nodes
            volume_profile = self.calculate_volume_profile(df)
            high_volume_nodes = self.identify_high_volume_nodes(volume_profile)
            
            # Find institutional support/resistance
            for node in high_volume_nodes:
                if self.validate_institutional_level(df, node['price']):
                    levels.append({
                        'price': node['price'],
                        'type': self.determine_level_type(df, node['price']),
                        'strength': node['volume_score']
                    })
        
        return levels


    def calculate_volume_nodes(self, df):
        price_bins = pd.qcut(df['close'], q=50, duplicates='drop')
        volume_profile = df.groupby(price_bins)['volume'].sum()
        
        nodes = []
        for price_range, volume in volume_profile.items():
            nodes.append({
                'price_level': price_range.mid,
                'volume': volume,
                'strength': volume / volume_profile.mean()
            })
        
        return sorted(nodes, key=lambda x: x['volume'], reverse=True)

    def analyze_poc_levels(self, df):
        volume_nodes = self.calculate_volume_nodes(df)
        poc_level = max(volume_nodes, key=lambda x: x['volume'])
        
        return {
            'poc_price': poc_level['price_level'],
            'poc_volume': poc_level['volume'],
            'poc_strength': poc_level['strength'],
            'value_area': self.calculate_value_area_range(volume_nodes)
        }

    def calculate_value_areas(self, df):
        volume_nodes = self.calculate_volume_nodes(df)
        total_volume = sum(node['volume'] for node in volume_nodes)
        
        value_areas = {
            'high_volume': [],
            'medium_volume': [],
            'low_volume': []
        }
        
        cumulative_volume = 0
        for node in sorted(volume_nodes, key=lambda x: x['price_level']):
            cumulative_volume += node['volume']
            percentage = cumulative_volume / total_volume
            
            if percentage <= 0.7:
                value_areas['high_volume'].append(node)
            elif percentage <= 0.9:
                value_areas['medium_volume'].append(node)
            else:
                value_areas['low_volume'].append(node)
        
        return value_areas

    def find_volume_opportunities(self, profile_data):
        opportunities = []
        poc = profile_data['profile']['poc_analysis']
        nodes = profile_data['profile']['volume_nodes']
        
        for node in nodes:
            if self.is_trading_opportunity(node, poc, profile_data['key_levels']):
                opportunities.append({
                    'price_level': node['price_level'],
                    'type': self.determine_opportunity_type(node, poc),
                    'strength': self.calculate_opportunity_strength(node, profile_data),
                    'volume_confirmation': node['volume'] > poc['poc_volume'] * 0.5
                })
        
        return opportunities
    def calculate_value_area_range(self, volume_nodes):
        total_volume = sum(node['volume'] for node in volume_nodes)
        target_volume = total_volume * 0.68  # 68% value area
        
        cumulative_volume = 0
        value_area_nodes = []
        
        for node in sorted(volume_nodes, key=lambda x: x['volume'], reverse=True):
            cumulative_volume += node['volume']
            value_area_nodes.append(node)
            if cumulative_volume >= target_volume:
                break
        
        return {
            'upper': max(node['price_level'] for node in value_area_nodes),
            'lower': min(node['price_level'] for node in value_area_nodes),
            'nodes': value_area_nodes
        }

    def is_trading_opportunity(self, node, poc, key_levels):
        price_level = node['price_level']
        
        # Check if price is near key level
        near_key_level = any(
            abs(price_level - level['price']) / level['price'] < 0.01 
            for level in key_levels
        )
        
        # Check volume significance
        volume_significant = node['volume'] > poc['poc_volume'] * 0.3
        
        # Check if price is in value area
        in_value_area = (
            poc['value_area']['lower'] <= price_level <= poc['value_area']['upper']
        )
        
        return near_key_level and volume_significant and in_value_area

    def determine_opportunity_type(self, node, poc):
        price = node['price_level']
        poc_price = poc['poc_price']
        
        if price > poc_price:
            return 'resistance' if node['volume'] > poc['poc_volume'] else 'weak_resistance'
        else:
            return 'support' if node['volume'] > poc['poc_volume'] else 'weak_support'

    def calculate_opportunity_strength(self, node, profile_data):
        base_strength = node['strength']
        
        # Adjust strength based on value area position
        value_areas = profile_data['profile']['value_areas']
        if node in value_areas['high_volume']:
            base_strength *= 1.5
        elif node in value_areas['medium_volume']:
            base_strength *= 1.2
        
        # Adjust strength based on nearby key levels
        key_levels = profile_data['key_levels']
        for level in key_levels:
            if abs(node['price_level'] - level['price']) / level['price'] < 0.01:
                base_strength *= 1.3
        
        return min(base_strength, 10.0)  # Cap strength at 10

    def find_peaks(self, data, distance=10):
        peaks = []
        for i in range(distance, len(data) - distance):
            if all(data[i] > data[j] for j in range(i-distance, i+distance+1) if i != j):
                peaks.append(i)
        return peaks

    def validate_shoulder(self, df, peaks, position):
        if not peaks or abs(position) >= len(peaks):
            return None
            
        peak_idx = peaks[position]
        return {
            'price': df['high'].iloc[peak_idx],
            'volume': df['volume'].iloc[peak_idx],
            'index': peak_idx
        }

    def validate_head(self, df, peaks, troughs):
        if len(peaks) < 3 or len(troughs) < 2:
            return None
            
        head_candidates = [p for p in peaks if peaks[0] < p < peaks[-1]]
        if not head_candidates:
            return None
            
        head_idx = max(head_candidates, key=lambda x: df['high'].iloc[x])
        return {
            'price': df['high'].iloc[head_idx],
            'volume': df['volume'].iloc[head_idx],
            'index': head_idx
        }

    def calculate_neckline(self, df, troughs):
        if len(troughs) < 2:
            return None
            
        left_trough = troughs[0]
        right_trough = troughs[-1]
        
        slope = (df['low'].iloc[right_trough] - df['low'].iloc[left_trough]) / (right_trough - left_trough)
        intercept = df['low'].iloc[left_trough] - slope * left_trough
        
        return {
            'slope': slope,
            'intercept': intercept,
            'left_point': (left_trough, df['low'].iloc[left_trough]),
            'right_point': (right_trough, df['low'].iloc[right_trough])
        }

    def calculate_hs_target(self, pattern_data):
        neckline = pattern_data['neckline']
        head_price = pattern_data['head']['price']
        neckline_price = self.get_neckline_price(neckline, pattern_data['right_shoulder']['index'])
        
        # Calculate pattern height
        height = head_price - neckline_price
        
        # Project target below neckline
        target = neckline_price - height
        
        return {
            'price': target,
            'height': height,
            'risk_reward': self.calculate_risk_reward(target, neckline_price, head_price)
        }


    def get_neckline_price(self, neckline, index):
        return neckline['slope'] * index + neckline['intercept']

    def calculate_pattern_completion(self, df, pattern_data):
        completion_data = {
            'status': 'incomplete',
            'progress': 0.0,
            'confirmation_points': []
        }
        
        # Get latest price
        current_price = df['close'].iloc[-1]
        neckline_price = self.get_neckline_price(pattern_data['neckline'], len(df) - 1)
        
        # Check pattern formation progress
        if pattern_data['right_shoulder']:
            completion_data['progress'] = self.calculate_formation_progress(pattern_data)
            
            # Check for neckline break
            if current_price < neckline_price:
                completion_data['status'] = 'confirmed'
                completion_data['confirmation_points'].append({
                    'type': 'neckline_break',
                    'price': current_price,
                    'index': len(df) - 1
                })
        
        return completion_data
    def calculate_formation_progress(self, pattern_data):
        total_points = 4  # Left shoulder, head, right shoulder, neckline break
        completed_points = 0
        
        if pattern_data['left_shoulder']:
            completed_points += 1
        if pattern_data['head']:
            completed_points += 1
        if pattern_data['right_shoulder']:
            completed_points += 1
        if pattern_data.get('neckline_break'):
            completed_points += 1
        
        return completed_points / total_points
    def validate_pattern_breakout(self, df, pattern_data):
        if not pattern_data['neckline']:
            return False
            
        neckline_price = self.get_neckline_price(pattern_data['neckline'], len(df) - 1)
        current_price = df['close'].iloc[-1]
        
        # Volume confirmation
        recent_volume = df['volume'].iloc[-5:].mean()
        pattern_volume = df['volume'].iloc[-20:].mean()
        
        return {
            'breakout_confirmed': current_price < neckline_price,
            'volume_confirmed': recent_volume > pattern_volume * 1.5,
            'momentum_confirmed': self.confirm_momentum(df)
        }
    def confirm_momentum(self, df):
        # Calculate momentum indicators
        df['rsi'] = talib.RSI(df['close'])
        df['macd'], df['macd_signal'], _ = talib.MACD(df['close'])
        df['adx'] = talib.ADX(df['high'], df['low'], df['close'])
        
        # Get latest values
        current_rsi = df['rsi'].iloc[-1]
        current_macd = df['macd'].iloc[-1]
        current_macd_signal = df['macd_signal'].iloc[-1]
        current_adx = df['adx'].iloc[-1]
        
        return {
            'momentum_strong': current_adx > 25,
            'rsi_confirmed': current_rsi < 30,
            'macd_confirmed': current_macd < current_macd_signal,
            'overall_confirmation': all([
                current_adx > 25,
                current_rsi < 30,
                current_macd < current_macd_signal
            ])
        }

    def filter_patterns(self, df, patterns):
        filtered_patterns = []
        
        for pattern in patterns:
            # Calculate pattern quality score
            quality_score = self.calculate_pattern_quality(df, pattern)
            
            # Check volume profile
            volume_valid = self.validate_volume_profile(df, pattern)
            
            # Check market context
            context_valid = self.validate_market_context(df, pattern)
            
            if quality_score > 0.7 and volume_valid and context_valid:
                pattern['quality_score'] = quality_score
                filtered_patterns.append(pattern)
        
        return filtered_patterns

    def calculate_pattern_quality(self, df, pattern):
        scores = []
        
        # Price action quality
        price_score = self.assess_price_action(df, pattern)
        scores.append(price_score * 0.4)  # 40% weight
        
        # Volume confirmation
        volume_score = self.assess_volume_confirmation(df, pattern)
        scores.append(volume_score * 0.3)  # 30% weight
        
        # Technical indicator alignment
        indicator_score = self.assess_indicator_alignment(df, pattern)
        scores.append(indicator_score * 0.3)  # 30% weight
        
        return sum(scores)
    def validate_volume_profile(self, df, pattern):
        volume_data = self.analyze_volume_profile(df)
        
        # Check volume at pattern points
        pattern_volumes = []
        for point in ['left_shoulder', 'head', 'right_shoulder']:
            if point in pattern:
                idx = pattern[point]['index']
                pattern_volumes.append(df['volume'].iloc[idx])
        
        avg_volume = df['volume'].mean()
        return all(vol > avg_volume for vol in pattern_volumes)

    def validate_market_context(self, df, pattern):
        # Check market trend
        trend = self.detect_trend_strength(df)
        
        # Check support/resistance levels
        levels = self.identify_key_levels(df)
        
        # Check volatility
        volatility = df['close'].pct_change().std()
        
        return {
            'trend_aligned': trend > 25,
            'near_key_level': self.is_near_key_level(pattern, levels),
            'volatility_suitable': 0.01 < volatility < 0.05
        }
    def assess_price_action(self, df, pattern):
        score = 0
        
        # Check candlestick patterns
        if self.validate_candlestick_patterns(df):
            score += 0.4
        
        # Check price momentum
        if self.validate_price_momentum(df):
            score += 0.3
        
        # Check support/resistance levels
        if self.validate_key_levels(df, pattern):
            score += 0.3
        
        return score
    def assess_indicator_alignment(self, df, pattern):
        score = 0
        
        # RSI alignment
        rsi = talib.RSI(df['close'])
        if rsi.iloc[-1] < 30:
            score += 0.3
        
        # MACD confirmation
        macd, signal, _ = talib.MACD(df['close'])
        if macd.iloc[-1] > signal.iloc[-1]:
            score += 0.3
        
        # ADX trend strength
        adx = talib.ADX(df['high'], df['low'], df['close'])
        if adx.iloc[-1] > 25:
            score += 0.4
        
        return score
    def assess_volume_confirmation(self, df, pattern):
        # Calculate relative volume metrics
        avg_volume = df['volume'].rolling(20).mean()
        current_volume = df['volume'].iloc[-1]
        
        # Volume trend analysis
        volume_trend = self.analyze_volume_trend(df)
        
        # Score based on volume characteristics
        score = 0
        if current_volume > avg_volume * 1.5:
            score += 0.5
        if volume_trend['increasing']:
            score += 0.3
        if volume_trend['consistent']:
            score += 0.2
            
        return score

    def get_pattern_volumes(self, df, pattern):
        ls_vol = df['volume'].iloc[pattern['left_shoulder']['index']]
        h_vol = df['volume'].iloc[pattern['head']['index']]
        rs_vol = df['volume'].iloc[pattern['right_shoulder']['index']]
        
        return {
            'left_shoulder': ls_vol,
            'head': h_vol,
            'right_shoulder': rs_vol,
            'shoulders_avg': (ls_vol + rs_vol) / 2
        }

    def check_volume_consistency(self, df, pattern):
        # Get volume for pattern formation period
        start_idx = pattern['left_shoulder']['index']
        end_idx = pattern['right_shoulder']['index']
        formation_volume = df['volume'].iloc[start_idx:end_idx]
        
        # Calculate volume consistency metrics
        volume_std = formation_volume.std()
        volume_mean = formation_volume.mean()
        coefficient_variation = volume_std / volume_mean
        
        return coefficient_variation < 0.5

    def check_level_interaction(self, df, pattern):
        # Get key price levels
        levels = self.identify_key_levels(df)
        neckline_price = self.get_neckline_price(pattern['neckline'], len(df)-1)
        
        # Check if neckline coincides with any key level
        for level in levels:
            if abs(level['price'] - neckline_price) / neckline_price < 0.01:
                return True
                
        return False

    def identify_key_levels(self, df):
        levels = []
        
        # Support levels
        supports = self.find_support_levels(df)
        for support in supports:
            levels.append({
                'price': support,
                'type': 'support',
                'strength': self.calculate_level_strength(df, support)
            })
        
        # Resistance levels
        resistances = self.find_resistance_levels(df)
        for resistance in resistances:
            levels.append({
                'price': resistance,
                'type': 'resistance',
                'strength': self.calculate_level_strength(df, resistance)
            })
        
        return levels
    def find_support_levels(self, df):
        supports = []
        window = 20
        
        for i in range(window, len(df) - window):
            if self.is_support(df, i):
                price = df['low'].iloc[i]
                if not self.is_duplicate_level(supports, price):
                    supports.append(price)
        
        return supports

    def find_resistance_levels(self, df):
        resistances = []
        window = 20
        
        for i in range(window, len(df) - window):
            if self.is_resistance(df, i):
                price = df['high'].iloc[i]
                if not self.is_duplicate_level(resistances, price):
                    resistances.append(price)
        
        return resistances

    def is_support(self, df, index):
        current_low = df['low'].iloc[index]
        for i in range(index - 5, index + 6):
            if i != index and df['low'].iloc[i] < current_low:
                return False
        return True

    def is_resistance(self, df, index):
        current_high = df['high'].iloc[index]
        for i in range(index - 5, index + 6):
            if i != index and df['high'].iloc[i] > current_high:
                return False
        return True

    def is_duplicate_level(self, levels, price):
        if not levels:
            return False
        return any(abs(level - price) / price < 0.001 for level in levels)

    def calculate_level_strength(self, df, level_price):
        touches = 0
        bounces = 0
        
        for i in range(len(df)):
            if abs(df['low'].iloc[i] - level_price) / level_price < 0.001:
                touches += 1
                if i < len(df) - 1 and df['close'].iloc[i + 1] > level_price:
                    bounces += 1
        
        return {
            'touches': touches,
            'bounces': bounces,
            'strength_score': bounces / touches if touches > 0 else 0
        }
    def validate_pattern_structure(self, df, pattern):
        validation = {
            'price_structure': self.validate_price_structure(pattern),
            'time_structure': self.validate_time_structure(pattern),
            'volume_structure': self.validate_volume_structure(df, pattern),
            'momentum_structure': self.validate_momentum_structure(df, pattern)
        }
        
        return all(validation.values())

    def validate_price_structure(self, pattern):
        ls_price = pattern['left_shoulder']['price']
        h_price = pattern['head']['price']
        rs_price = pattern['right_shoulder']['price']
        
        # Head must be higher than shoulders
        if not (h_price > ls_price and h_price > rs_price):
            return False
        
        # Shoulders should be within 10% of each other
        shoulder_diff = abs(ls_price - rs_price) / ls_price
        return shoulder_diff <= 0.10

    def validate_time_structure(self, pattern):
        ls_idx = pattern['left_shoulder']['index']
        h_idx = pattern['head']['index']
        rs_idx = pattern['right_shoulder']['index']
        
        # Check time symmetry
        left_span = h_idx - ls_idx
        right_span = rs_idx - h_idx
        
        time_symmetry = abs(left_span - right_span) / left_span
        return time_symmetry <= 0.20

    def validate_volume_structure(self, df, pattern):
        volumes = self.get_pattern_volumes(df, pattern)
        
        # Head volume should be highest
        if volumes['head'] <= volumes['shoulders_avg']:
            return False
        
        # Volume should decline in right shoulder
        return volumes['right_shoulder'] < volumes['left_shoulder']

    def validate_momentum_structure(self, df, pattern):
        # Calculate momentum indicators at pattern points
        rsi = talib.RSI(df['close'])
        macd, signal, _ = talib.MACD(df['close'])
        
        head_idx = pattern['head']['index']
        rs_idx = pattern['right_shoulder']['index']
        
        # Check for momentum divergence
        price_trend = pattern['head']['price'] > pattern['right_shoulder']['price']
        rsi_trend = rsi.iloc[head_idx] < rsi.iloc[rs_idx]
        macd_trend = macd.iloc[head_idx] < macd.iloc[rs_idx]
        
        return price_trend and (rsi_trend or macd_trend)
    def project_pattern_targets(self, df, pattern):
        neckline_price = self.get_neckline_price(pattern['neckline'], len(df)-1)
        pattern_height = pattern['head']['price'] - neckline_price
        
        return {
            'primary_target': neckline_price - pattern_height,
            'secondary_target': neckline_price - (pattern_height * 1.618),
            'stop_loss': pattern['head']['price'],
            'risk_reward': self.calculate_target_risk_reward(pattern_height, neckline_price)
        }

    def calculate_target_risk_reward(self, height, neckline):
        reward = height
        risk = height * 0.3  # Stop loss at 30% of pattern height
        return reward / risk

    def validate_pattern_completion(self, df, pattern):
        current_price = df['close'].iloc[-1]
        neckline_price = self.get_neckline_price(pattern['neckline'], len(df)-1)
        
        completion_data = {
            'status': 'incomplete',
            'confirmation_level': 0,
            'volume_confirmation': False,
            'momentum_confirmation': False
        }
        
        if current_price < neckline_price:
            completion_data.update(
                self.analyze_breakout_confirmation(df, pattern, neckline_price)
            )
        
        return completion_data

    def analyze_breakout_confirmation(self, df, pattern, neckline_price):
        recent_volume = df['volume'].iloc[-5:].mean()
        pattern_volume = df['volume'].iloc[-20:].mean()
        
        momentum = self.confirm_momentum(df)
        
        return {
            'status': 'complete',
            'confirmation_level': self.calculate_confirmation_strength(df, pattern),
            'volume_confirmation': recent_volume > pattern_volume * 1.5,
            'momentum_confirmation': momentum['overall_confirmation']
        }
    def calculate_confirmation_strength(self, df, pattern):
        confirmation_factors = {
            'price_action': self.analyze_price_confirmation(df),
            'volume_profile': self.analyze_volume_confirmation(df),
            'technical_signals': self.analyze_technical_confirmation(df),
            'market_context': self.analyze_market_context(df)
        }
        
        return self.weighted_confirmation_score(confirmation_factors)

    def analyze_price_confirmation(self, df):
        recent_candles = df.tail(5)
        
        return {
            'close_below_neckline': all(candle['close'] < candle['open'] for _, candle in recent_candles.iterrows()),
            'strong_bearish_candles': any(candle['open'] - candle['close'] > candle['high'] - candle['low'] * 0.7 
                                        for _, candle in recent_candles.iterrows()),
            'no_upper_wicks': all(candle['high'] - candle['open'] < candle['open'] - candle['close'] * 0.3 
                                for _, candle in recent_candles.iterrows())
        }

    def analyze_technical_confirmation(self, df):
        rsi = talib.RSI(df['close'])
        macd, signal, _ = talib.MACD(df['close'])
        adx = talib.ADX(df['high'], df['low'], df['close'])
        
        return {
            'rsi_trending': rsi.iloc[-1] < 40,
            'macd_bearish': macd.iloc[-1] < signal.iloc[-1],
            'strong_trend': adx.iloc[-1] > 25
        }

    def weighted_confirmation_score(self, factors):
        weights = {
            'price_action': 0.4,
            'volume_profile': 0.3,
            'technical_signals': 0.2,
            'market_context': 0.1
        }
        
        score = 0
        for factor, weight in weights.items():
            if isinstance(factors[factor], dict):
                factor_score = sum(1 for v in factors[factor].values() if v) / len(factors[factor])
            else:
                factor_score = 1 if factors[factor] else 0
            score += factor_score * weight
        
        return score
    def analyze_market_context(self, df):
        return {
            'trend_analysis': self.analyze_market_trend(df),
            'volatility_state': self.analyze_market_volatility(df),
            'liquidity_analysis': self.analyze_market_liquidity(df),
            'correlation_impact': self.analyze_market_correlations(df)
        }

    def analyze_market_trend(self, df):
        ema20 = talib.EMA(df['close'], timeperiod=20)
        ema50 = talib.EMA(df['close'], timeperiod=50)
        ema200 = talib.EMA(df['close'], timeperiod=200)
        
        trend_data = {
            'short_term': ema20.iloc[-1] > ema50.iloc[-1],
            'long_term': ema50.iloc[-1] > ema200.iloc[-1],
            'trend_strength': self.calculate_trend_strength(df),
            'trend_duration': self.calculate_trend_duration(df)
        }
        
        return self.classify_trend_state(trend_data)

    def analyze_market_volatility(self, df):
        atr = talib.ATR(df['high'], df['low'], df['close'])
        volatility = df['close'].pct_change().std()
        
        return {
            'current_atr': atr.iloc[-1],
            'atr_percentile': self.calculate_percentile(atr),
            'volatility_regime': self.classify_volatility(volatility),
            'volatility_trend': self.analyze_volatility_trend(atr)
        }

    def analyze_market_liquidity(self, df):
        volume_ma = df['volume'].rolling(window=20).mean()
        spread = df['high'] - df['low']
        
        return {
            'volume_trend': df['volume'].iloc[-1] > volume_ma.iloc[-1],
            'spread_analysis': spread.mean() / df['close'].mean(),
            'liquidity_score': self.calculate_liquidity_score(df),
            'market_depth': self.analyze_market_depth()
        }

    def analyze_market_correlations(self, df):
        btc_correlation = self.calculate_btc_correlation(df)
        sector_correlation = self.calculate_sector_correlation(df)
        
        return {
            'btc_correlation': btc_correlation,
            'sector_correlation': sector_correlation,
            'correlation_impact': self.assess_correlation_impact(btc_correlation, sector_correlation)
        }

    def calculate_btc_correlation(self, df):
        btc_data = self.fetch_market_data('BTC/USDT', df.index[0])
        if btc_data is not None:
            return df['close'].corr(btc_data['close'])
        return 0

    def calculate_sector_correlation(self, df):
        sector_pairs = self.get_sector_pairs(df.name)
        correlations = []
        
        for pair in sector_pairs:
            pair_data = self.fetch_market_data(pair, df.index[0])
            if pair_data is not None:
                correlation = df['close'].corr(pair_data['close'])
                correlations.append(correlation)
        
        return sum(correlations) / len(correlations) if correlations else 0

    def get_sector_pairs(self, symbol):
        sectors = {
            'DEFI': ['UNI', 'AAVE', 'CAKE', 'COMP', 'MKR'],
            'L1': ['ETH', 'SOL', 'ADA', 'AVAX', 'DOT'],
            'GAMING': ['AXS', 'SAND', 'MANA', 'ENJ', 'GALA'],
            'ORACLE': ['LINK', 'BAND', 'TRB', 'API3', 'NEST']
        }
        
        for sector, tokens in sectors.items():
            if any(token in symbol for token in tokens):
                return [f"{token}/USDT" for token in tokens]
        return []

    def assess_correlation_impact(self, btc_corr, sector_corr):
        impact_score = (btc_corr * 0.6) + (sector_corr * 0.4)
        
        if impact_score > 0.8:
            return 'high_correlation'
        elif impact_score > 0.5:
            return 'moderate_correlation'
        else:
            return 'low_correlation'
    def calculate_trend_strength(self, df):
        adx = talib.ADX(df['high'], df['low'], df['close'])
        rsi = talib.RSI(df['close'])
        
        trend_metrics = {
            'adx_strength': adx.iloc[-1] > 25,
            'rsi_trend': rsi.iloc[-1] > 50,
            'price_action': df['close'].iloc[-1] > df['close'].rolling(20).mean().iloc[-1]
        }
        
        return sum(1 for metric in trend_metrics.values() if metric) / len(trend_metrics)

    def calculate_trend_duration(self, df):
        ema20 = talib.EMA(df['close'], timeperiod=20)
        trend_direction = 'bullish' if ema20.iloc[-1] > ema20.iloc[-2] else 'bearish'
        
        duration = 0
        for i in range(len(ema20)-2, 0, -1):
            if trend_direction == 'bullish' and ema20.iloc[i] <= ema20.iloc[i-1]:
                break
            elif trend_direction == 'bearish' and ema20.iloc[i] >= ema20.iloc[i-1]:
                break
            duration += 1
        
        return {
            'direction': trend_direction,
            'duration': duration,
            'strength': 'strong' if duration > 20 else 'moderate' if duration > 10 else 'weak'
        }

    def classify_volatility(self, volatility):
        if volatility > 0.05:
            return 'high'
        elif volatility > 0.02:
            return 'medium'
        else:
            return 'low'

    def analyze_volatility_trend(self, atr):
        current_atr = atr.iloc[-1]
        atr_sma = atr.rolling(window=20).mean().iloc[-1]
        
        return {
            'current_state': 'increasing' if current_atr > atr_sma else 'decreasing',
            'magnitude': current_atr / atr_sma,
            'percentile': self.calculate_percentile(atr)
        }
    def calculate_liquidity_score(self, df):
        volume_profile = self.analyze_volume_profile(df)
        spread_analysis = self.analyze_spread_distribution(df)
        depth_analysis = self.analyze_market_depth()
        
        return {
            'volume_score': self.score_volume_metrics(volume_profile),
            'spread_score': self.score_spread_metrics(spread_analysis),
            'depth_score': self.score_depth_metrics(depth_analysis)
        }

    def analyze_spread_distribution(self, df):
        spreads = df['high'] - df['low']
        avg_spread = spreads.mean()
        spread_volatility = spreads.std()
        
        return {
            'average_spread': avg_spread,
            'spread_volatility': spread_volatility,
            'spread_trend': self.calculate_spread_trend(spreads)
        }

    def analyze_market_depth(self):
        order_book = self.binance.fetch_order_book(self.symbol)
        
        return {
            'bid_depth': self.calculate_depth_metrics(order_book['bids']),
            'ask_depth': self.calculate_depth_metrics(order_book['asks']),
            'depth_ratio': self.calculate_depth_ratio(order_book)
        }

    def calculate_depth_metrics(self, orders):
        total_volume = sum(order[1] for order in orders[:10])
        price_levels = len(set(order[0] for order in orders[:10]))
        
        return {
            'total_volume': total_volume,
            'price_levels': price_levels,
            'average_size': total_volume / price_levels if price_levels > 0 else 0
        }

    def calculate_depth_ratio(self, order_book):
        bid_volume = sum(bid[1] for bid in order_book['bids'][:10])
        ask_volume = sum(ask[1] for ask in order_book['asks'][:10])
        
        return bid_volume / ask_volume if ask_volume > 0 else 0
    def score_volume_metrics(self, volume_profile):
        score = 0
        
        # Volume consistency score (40%)
        if volume_profile['volume_trend'] == 'increasing':
            score += 40
        elif volume_profile['volume_trend'] == 'stable':
            score += 25
        
        # Volume relative to average score (30%)
        volume_ratio = volume_profile['volume_ratio']
        if volume_ratio > 2.0:
            score += 30
        elif volume_ratio > 1.5:
            score += 20
        elif volume_ratio > 1.0:
            score += 10
        
        # Volume distribution score (30%)
        if volume_profile['volume_sma'] > volume_profile['volume_sma'].mean():
            score += 30
        
        return score / 100

    def score_spread_metrics(self, spread_analysis):
        score = 0
        
        # Average spread score (40%)
        avg_spread = spread_analysis['average_spread']
        if avg_spread < 0.001:  # Less than 0.1%
            score += 40
        elif avg_spread < 0.002:
            score += 25
        
        # Spread volatility score (30%)
        spread_vol = spread_analysis['spread_volatility']
        if spread_vol < 0.0005:
            score += 30
        elif spread_vol < 0.001:
            score += 15
        
        # Spread trend score (30%)
        if spread_analysis['spread_trend'] == 'decreasing':
            score += 30
        elif spread_analysis['spread_trend'] == 'stable':
            score += 15
        
        return score / 100

    def score_depth_metrics(self, depth_analysis):
        score = 0
        
        # Total depth score (40%)
        bid_depth = depth_analysis['bid_depth']['total_volume']
        ask_depth = depth_analysis['ask_depth']['total_volume']
        total_depth = bid_depth + ask_depth
        
        if total_depth > 1000000:  # Adjust threshold based on your market
            score += 40
        elif total_depth > 500000:
            score += 25
        
        # Depth ratio score (30%)
        depth_ratio = depth_analysis['depth_ratio']
        if 0.8 < depth_ratio < 1.2:
            score += 30
        elif 0.5 < depth_ratio < 1.5:
            score += 15
        
        # Price levels score (30%)
        avg_levels = (depth_analysis['bid_depth']['price_levels'] + 
                    depth_analysis['ask_depth']['price_levels']) / 2
        if avg_levels > 8:
            score += 30
        elif avg_levels > 5:
            score += 15
        
        return score / 100
    def classify_trend_state(self, trend_data):
        trend_score = 0
        
        # Short-term trend weight (40%)
        if trend_data['short_term']:
            trend_score += 40
        
        # Long-term trend weight (30%)
        if trend_data['long_term']:
            trend_score += 30
        
        # Trend strength weight (20%)
        if trend_data['trend_strength'] > 0.7:
            trend_score += 20
        elif trend_data['trend_strength'] > 0.5:
            trend_score += 10
        
        # Trend duration weight (10%)
        if trend_data['trend_duration']['strength'] == 'strong':
            trend_score += 10
        elif trend_data['trend_duration']['strength'] == 'moderate':
            trend_score += 5
        
        return {
            'trend_score': trend_score,
            'trend_state': self.get_trend_state(trend_score),
            'trend_metrics': trend_data
        }

    def get_trend_state(self, trend_score):
        if trend_score >= 80:
            return 'strong_uptrend'
        elif trend_score >= 60:
            return 'moderate_uptrend'
        elif trend_score >= 40:
            return 'neutral'
        elif trend_score >= 20:
            return 'moderate_downtrend'
        else:
            return 'strong_downtrend'

    def analyze_trend_momentum(self, df):
        rsi = talib.RSI(df['close'])
        macd, signal, _ = talib.MACD(df['close'])
        adx = talib.ADX(df['high'], df['low'], df['close'])
        
        momentum_data = {
            'rsi_momentum': self.classify_rsi_momentum(rsi.iloc[-1]),
            'macd_momentum': self.classify_macd_momentum(macd.iloc[-1], signal.iloc[-1]),
            'trend_strength': self.classify_adx_strength(adx.iloc[-1])
        }
        
        return self.calculate_momentum_score(momentum_data)
    def classify_rsi_momentum(self, rsi_value):
        if rsi_value > 70:
            return {'state': 'overbought', 'strength': (rsi_value - 70) / 30}
        elif rsi_value < 30:
            return {'state': 'oversold', 'strength': (30 - rsi_value) / 30}
        else:
            return {'state': 'neutral', 'strength': abs(50 - rsi_value) / 20}

    def classify_macd_momentum(self, macd_value, signal_value):
        momentum = macd_value - signal_value
        return {
            'state': 'bullish' if momentum > 0 else 'bearish',
            'strength': abs(momentum),
            'divergence': self.check_macd_divergence(macd_value, signal_value)
        }

    def classify_adx_strength(self, adx_value):
        if adx_value > 40:
            return {'strength': 'very_strong', 'value': adx_value}
        elif adx_value > 25:
            return {'strength': 'strong', 'value': adx_value}
        elif adx_value > 15:
            return {'strength': 'moderate', 'value': adx_value}
        else:
            return {'strength': 'weak', 'value': adx_value}

    def calculate_momentum_score(self, momentum_data):
        score = 0
        
        # RSI Component (30%)
        rsi_state = momentum_data['rsi_momentum']
        if rsi_state['state'] == 'oversold':
            score += 30 * rsi_state['strength']
        
        # MACD Component (40%)
        macd_state = momentum_data['macd_momentum']
        if macd_state['state'] == 'bullish':
            score += 40 * (macd_state['strength'] / 0.01)  # Normalize strength
        
        # ADX Component (30%)
        adx_state = momentum_data['trend_strength']
        if adx_state['strength'] in ['strong', 'very_strong']:
            score += 30 * (adx_state['value'] / 50)  # Normalize ADX value
        
        return {
            'total_score': min(score, 100),
            'momentum_state': self.classify_momentum_state(score),
            'components': momentum_data
        }
    def classify_momentum_state(self, momentum_score):
        if momentum_score >= 80:
            return {
                'state': 'strong_bullish',
                'confidence': 'high',
                'suggested_action': 'aggressive_buy'
            }
        elif momentum_score >= 60:
            return {
                'state': 'moderately_bullish',
                'confidence': 'medium',
                'suggested_action': 'cautious_buy'
            }
        elif momentum_score >= 40:
            return {
                'state': 'neutral',
                'confidence': 'low',
                'suggested_action': 'wait_and_observe'
            }
        elif momentum_score >= 20:
            return {
                'state': 'moderately_bearish',
                'confidence': 'medium',
                'suggested_action': 'reduce_exposure'
            }
        else:
            return {
                'state': 'strong_bearish',
                'confidence': 'high',
                'suggested_action': 'exit_positions'
            }

    def check_macd_divergence(self, macd_values, signal_values):
        divergence_data = {
            'type': None,
            'strength': 0,
            'confirmation_count': 0
        }
        
        # Regular Bullish Divergence
        if self.detect_regular_bullish_divergence(macd_values, signal_values):
            divergence_data.update({
                'type': 'regular_bullish',
                'strength': self.calculate_divergence_strength(macd_values, signal_values),
                'confirmation_count': self.count_divergence_confirmations(macd_values, signal_values)
            })
        
        # Hidden Bullish Divergence
        elif self.detect_hidden_bullish_divergence(macd_values, signal_values):
            divergence_data.update({
                'type': 'hidden_bullish',
                'strength': self.calculate_divergence_strength(macd_values, signal_values) * 0.8,
                'confirmation_count': self.count_divergence_confirmations(macd_values, signal_values)
            })
        
        return divergence_data
    def detect_regular_bullish_divergence(self, macd_values, signal_values):
        price_lows = self.find_price_lows()
        macd_lows = self.find_indicator_lows(macd_values)
        
        if len(price_lows) >= 2 and len(macd_lows) >= 2:
            price_trend = price_lows[-1] < price_lows[-2]
            macd_trend = macd_lows[-1] > macd_lows[-2]
            return price_trend and macd_trend
        return False

    def detect_hidden_bullish_divergence(self, macd_values, signal_values):
        price_lows = self.find_price_lows()
        macd_lows = self.find_indicator_lows(macd_values)
        
        if len(price_lows) >= 2 and len(macd_lows) >= 2:
            price_trend = price_lows[-1] > price_lows[-2]
            macd_trend = macd_lows[-1] < macd_lows[-2]
            return price_trend and macd_trend
        return False

    def calculate_divergence_strength(self, macd_values, signal_values):
        recent_macd = macd_values[-5:]
        recent_signal = signal_values[-5:]
        
        histogram = recent_macd - recent_signal
        strength = abs(histogram.mean()) / recent_macd.std()
        
        return min(strength, 1.0)

    def count_divergence_confirmations(self, macd_values, signal_values):
        confirmations = 0
        lookback = 5
        
        for i in range(-lookback, 0):
            if macd_values[i] > signal_values[i]:
                confirmations += 1
        
        return confirmations
    def find_price_lows(self):
        df = self.current_data
        window_size = 10
        price_lows = []
        
        for i in range(window_size, len(df) - window_size):
            if self.is_local_minimum(df['low'], i, window_size):
                price_lows.append({
                    'price': df['low'].iloc[i],
                    'index': i,
                    'strength': self.calculate_swing_strength(df, i)
                })
        
        return sorted(price_lows, key=lambda x: x['index'])

    def find_indicator_lows(self, indicator_values):
        window_size = 5
        indicator_lows = []
        
        for i in range(window_size, len(indicator_values) - window_size):
            if self.is_local_minimum(indicator_values, i, window_size):
                indicator_lows.append({
                    'value': indicator_values[i],
                    'index': i,
                    'strength': self.calculate_indicator_strength(indicator_values, i)
                })
        
        return sorted(indicator_lows, key=lambda x: x['index'])

    def is_local_minimum(self, series, index, window):
        left_window = series[index - window:index]
        right_window = series[index + 1:index + window + 1]
        current_value = series[index]
        
        return all(current_value <= value for value in left_window) and \
            all(current_value <= value for value in right_window)

    def calculate_swing_strength(self, df, index):
        window_size = 5
        current_low = df['low'].iloc[index]
        surrounding_highs = max(df['high'].iloc[index - window_size:index + window_size])
        
        return (surrounding_highs - current_low) / current_low

    def calculate_indicator_strength(self, indicator_values, index):
        window_size = 5
        current_value = indicator_values[index]
        surrounding_values = indicator_values[index - window_size:index + window_size]
        
        return abs(current_value - np.mean(surrounding_values)) / np.std(surrounding_values)
    def analyze_swing_patterns(self, df):
        swings = self.identify_swing_points(df)
        patterns = {
            'swing_highs': self.validate_swing_highs(swings['highs']),
            'swing_lows': self.validate_swing_lows(swings['lows']),
            'patterns': self.detect_swing_patterns(swings)
        }
        return self.calculate_swing_metrics(patterns)

    def identify_swing_points(self, df):
        swing_points = {
            'highs': [],
            'lows': []
        }
        
        for i in range(2, len(df) - 2):
            if self.is_swing_high(df, i):
                swing_points['highs'].append({
                    'price': df['high'].iloc[i],
                    'index': i,
                    'volume': df['volume'].iloc[i]
                })
            elif self.is_swing_low(df, i):
                swing_points['lows'].append({
                    'price': df['low'].iloc[i],
                    'index': i,
                    'volume': df['volume'].iloc[i]
                })
        
        return swing_points

    def is_swing_high(self, df, index):
        return (df['high'].iloc[index] > df['high'].iloc[index - 1] and 
                df['high'].iloc[index] > df['high'].iloc[index - 2] and
                df['high'].iloc[index] > df['high'].iloc[index + 1] and 
                df['high'].iloc[index] > df['high'].iloc[index + 2])

    def is_swing_low(self, df, index):
        return (df['low'].iloc[index] < df['low'].iloc[index - 1] and 
                df['low'].iloc[index] < df['low'].iloc[index - 2] and
                df['low'].iloc[index] < df['low'].iloc[index + 1] and 
                df['low'].iloc[index] < df['low'].iloc[index + 2])

    def validate_swing_points(self, swing_points, threshold=0.01):
        validated_points = []
        
        for i in range(len(swing_points)):
            current_point = swing_points[i]
            if i > 0:
                price_change = abs(current_point['price'] - swing_points[i-1]['price']) / swing_points[i-1]['price']
                if price_change > threshold:
                    validated_points.append(current_point)
        
        return validated_points
    def detect_swing_patterns(self, swings):
        patterns = {
            'double_bottom': self.find_double_bottoms(swings['lows']),
            'double_top': self.find_double_tops(swings['highs']),
            'higher_lows': self.check_higher_lows(swings['lows']),
            'lower_highs': self.check_lower_highs(swings['highs'])
        }
        
        return self.validate_patterns(patterns)

    def find_double_bottoms(self, lows):
        double_bottoms = []
        
        for i in range(len(lows) - 1):
            for j in range(i + 1, len(lows)):
                if self.validate_double_bottom(lows[i], lows[j]):
                    double_bottoms.append({
                        'first_low': lows[i],
                        'second_low': lows[j],
                        'strength': self.calculate_pattern_strength(lows[i], lows[j])
                    })
        
        return double_bottoms

    def find_double_tops(self, highs):
        double_tops = []
        
        for i in range(len(highs) - 1):
            for j in range(i + 1, len(highs)):
                if self.validate_double_top(highs[i], highs[j]):
                    double_tops.append({
                        'first_high': highs[i],
                        'second_high': highs[j],
                        'strength': self.calculate_pattern_strength(highs[i], highs[j])
                    })
        
        return double_tops

    def calculate_pattern_strength(self, point1, point2):
        price_similarity = 1 - abs(point1['price'] - point2['price']) / point1['price']
        volume_confirmation = point2['volume'] > point1['volume']
        time_spacing = 0.8 if 5 <= point2['index'] - point1['index'] <= 20 else 0.5
        
        return (price_similarity * 0.5 + 
                volume_confirmation * 0.3 + 
                time_spacing * 0.2)
    def validate_patterns(self, patterns):
        validated_patterns = {}
        
        for pattern_type, pattern_list in patterns.items():
            validated_patterns[pattern_type] = [
                pattern for pattern in pattern_list
                if pattern['strength'] > 0.7
            ]
        
        return {
            'valid_patterns': validated_patterns,
            'pattern_count': sum(len(patterns) for patterns in validated_patterns.values()),
            'strongest_pattern': self.find_strongest_pattern(validated_patterns)
        }

    def check_higher_lows(self, lows):
        if len(lows) < 2:
            return []
        
        higher_lows = []
        current_low = lows[0]
        
        for i in range(1, len(lows)):
            if lows[i]['price'] > current_low['price']:
                higher_lows.append({
                    'start_point': current_low,
                    'end_point': lows[i],
                    'strength': self.calculate_trend_strength(current_low, lows[i])
                })
                current_low = lows[i]
        
        return higher_lows

    def check_lower_highs(self, highs):
        if len(highs) < 2:
            return []
        
        lower_highs = []
        current_high = highs[0]
        
        for i in range(1, len(highs)):
            if highs[i]['price'] < current_high['price']:
                lower_highs.append({
                    'start_point': current_high,
                    'end_point': highs[i],
                    'strength': self.calculate_trend_strength(current_high, highs[i])
                })
                current_high = highs[i]
        
        return lower_highs

    def find_strongest_pattern(self, validated_patterns):
        strongest = {
            'type': None,
            'strength': 0,
            'pattern': None
        }
        
        for pattern_type, patterns in validated_patterns.items():
            for pattern in patterns:
                if pattern['strength'] > strongest['strength']:
                    strongest = {
                        'type': pattern_type,
                        'strength': pattern['strength'],
                        'pattern': pattern
                    }
        
        return strongest

    def calculate_trend_strength(self, start_point, end_point):
        price_change = abs(end_point['price'] - start_point['price']) / start_point['price']
        time_duration = end_point['index'] - start_point['index']
        volume_trend = end_point['volume'] > start_point['volume']
        
        strength_score = (
            price_change * 0.4 +
            (1 / (1 + np.exp(-time_duration/10))) * 0.3 +
            volume_trend * 0.3
        )
        
        return min(strength_score, 1.0)

    def confirm_pattern_breakout(self, pattern, current_data):
        confirmation = {
            'confirmed': False,
            'breakout_price': None,
            'confirmation_factors': []
        }
        
        if pattern['type'] == 'double_bottom':
            confirmation = self.confirm_double_bottom_breakout(pattern, current_data)
        elif pattern['type'] == 'double_top':
            confirmation = self.confirm_double_top_breakout(pattern, current_data)
        
        confirmation['strength'] = self.calculate_breakout_strength(confirmation)
        return confirmation

    def calculate_breakout_strength(self, confirmation_data):
        if not confirmation_data['confirmed']:
            return 0
        
        strength_factors = {
            'volume_surge': 0.4,
            'price_momentum': 0.3,
            'technical_confirmation': 0.3
        }
        
        total_strength = sum(
            strength_factors[factor] 
            for factor in confirmation_data['confirmation_factors']
        )
        
        return total_strength

    def validate_breakout_volume(self, current_volume, pattern_volume):
        return {
            'valid': current_volume > pattern_volume * 1.5,
            'strength': min(current_volume / pattern_volume, 2.0) / 2
        }
    def confirm_double_bottom_breakout(self, pattern, current_data):
        neckline_price = max(pattern['first_low']['price'], pattern['second_low']['price'])
        current_price = current_data['close'].iloc[-1]
        current_volume = current_data['volume'].iloc[-1]
        
        confirmation = {
            'confirmed': current_price > neckline_price,
            'breakout_price': neckline_price,
            'confirmation_factors': []
        }
        
        if confirmation['confirmed']:
            # Volume confirmation
            if self.validate_breakout_volume(current_volume, pattern['second_low']['volume'])['valid']:
                confirmation['confirmation_factors'].append('volume_surge')
            
            # Momentum confirmation
            if self.confirm_momentum(current_data)['overall_confirmation']:
                confirmation['confirmation_factors'].append('price_momentum')
            
            # Technical indicator confirmation
            if self.confirm_technical_breakout(current_data):
                confirmation['confirmation_factors'].append('technical_confirmation')
        
        return confirmation

    def confirm_double_top_breakout(self, pattern, current_data):
        neckline_price = min(pattern['first_high']['price'], pattern['second_high']['price'])
        current_price = current_data['close'].iloc[-1]
        current_volume = current_data['volume'].iloc[-1]
        
        confirmation = {
            'confirmed': current_price < neckline_price,
            'breakout_price': neckline_price,
            'confirmation_factors': []
        }
        
        if confirmation['confirmed']:
            # Volume confirmation
            if self.validate_breakout_volume(current_volume, pattern['second_high']['volume'])['valid']:
                confirmation['confirmation_factors'].append('volume_surge')
            
            # Momentum confirmation
            if not self.confirm_momentum(current_data)['overall_confirmation']:
                confirmation['confirmation_factors'].append('price_momentum')
            
            # Technical indicator confirmation
            if self.confirm_technical_breakout(current_data):
                confirmation['confirmation_factors'].append('technical_confirmation')
        
        return confirmation
    def confirm_technical_breakout(self, current_data):
        rsi_confirm = self.confirm_rsi_breakout(current_data)
        macd_confirm = self.confirm_macd_breakout(current_data)
        volume_confirm = self.confirm_volume_breakout(current_data)
        
        return {
            'confirmed': all([rsi_confirm, macd_confirm, volume_confirm]),
            'indicators': {
                'rsi': rsi_confirm,
                'macd': macd_confirm,
                'volume': volume_confirm
            }
        }

    def confirm_rsi_breakout(self, current_data):
        rsi = talib.RSI(current_data['close'])
        current_rsi = rsi.iloc[-1]
        prev_rsi = rsi.iloc[-2]
        
        return {
            'confirmed': current_rsi > prev_rsi,
            'value': current_rsi,
            'trend': 'bullish' if current_rsi > prev_rsi else 'bearish'
        }

    def confirm_macd_breakout(self, current_data):
        macd, signal, hist = talib.MACD(current_data['close'])
        
        return {
            'confirmed': hist.iloc[-1] > 0 and hist.iloc[-1] > hist.iloc[-2],
            'histogram': hist.iloc[-1],
            'trend': 'bullish' if hist.iloc[-1] > 0 else 'bearish'
        }

    def confirm_volume_breakout(self, current_data):
        volume_sma = current_data['volume'].rolling(window=20).mean()
        current_volume = current_data['volume'].iloc[-1]
        
        return {
            'confirmed': current_volume > volume_sma.iloc[-1] * 1.5,
            'volume_ratio': current_volume / volume_sma.iloc[-1],
            'trend': 'increasing' if current_volume > volume_sma.iloc[-1] else 'decreasing'
        }
    def calculate_pattern_targets(self, pattern, breakout_price):
        targets = {
            'entry': breakout_price,
            'targets': self.project_price_targets(pattern, breakout_price),
            'stops': self.calculate_stop_levels(pattern, breakout_price),
            'risk_reward': self.calculate_risk_reward_ratios(pattern, breakout_price)
        }
        
        return self.validate_target_levels(targets)

    def project_price_targets(self, pattern, breakout_price):
        pattern_height = self.calculate_pattern_height(pattern)
        
        return {
            'target_1': breakout_price + pattern_height * 0.618,  # First target using Fibonacci
            'target_2': breakout_price + pattern_height,          # Full pattern height
            'target_3': breakout_price + pattern_height * 1.618   # Extended target
        }

    def calculate_stop_levels(self, pattern, breakout_price):
        pattern_height = self.calculate_pattern_height(pattern)
        atr = self.calculate_atr(pattern)
        
        return {
            'tight_stop': breakout_price - atr,
            'pattern_stop': pattern['lowest_point'] - (pattern_height * 0.1),
            'swing_stop': self.find_nearest_swing_low(pattern)
        }

    def calculate_risk_reward_ratios(self, pattern, breakout_price):
        targets = self.project_price_targets(pattern, breakout_price)
        stops = self.calculate_stop_levels(pattern, breakout_price)
        
        return {
            'target_1_rr': (targets['target_1'] - breakout_price) / (breakout_price - stops['tight_stop']),
            'target_2_rr': (targets['target_2'] - breakout_price) / (breakout_price - stops['pattern_stop']),
            'target_3_rr': (targets['target_3'] - breakout_price) / (breakout_price - stops['swing_stop'])
        }
    def validate_target_levels(self, targets):
        validated_targets = targets.copy()
        key_levels = self.identify_key_levels(self.current_data)
        
        # Adjust targets based on key levels
        for target_name, target_price in validated_targets['targets'].items():
            nearest_level = self.find_nearest_key_level(target_price, key_levels)
            if nearest_level:
                validated_targets['targets'][target_name] = self.adjust_target_to_level(
                    target_price, nearest_level
                )
        
        return {
            'original_targets': targets,
            'adjusted_targets': validated_targets,
            'key_levels': key_levels
        }

    def calculate_atr(self, pattern):
        df = self.current_data
        atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        return atr.iloc[-1]

    def calculate_pattern_height(self, pattern):
        if 'first_high' in pattern:  # Double Top
            return pattern['first_high']['price'] - pattern['neckline']
        elif 'first_low' in pattern:  # Double Bottom
            return pattern['neckline'] - pattern['first_low']['price']
        return 0

    def find_nearest_key_level(self, price, key_levels):
        if not key_levels:
            return None
            
        nearest_level = min(key_levels, 
                        key=lambda x: abs(x['price'] - price))
        
        # Only return if within 2% of the target
        if abs(nearest_level['price'] - price) / price <= 0.02:
            return nearest_level
        return None
    def adjust_target_to_level(self, target_price, key_level):
        adjusted_target = {
            'price': key_level['price'],
            'original_price': target_price,
            'adjustment_reason': key_level['type'],
            'level_strength': key_level['strength']
        }
        
        return adjusted_target

    def calculate_position_size(self, entry_price, stop_loss, risk_percentage=1.0):
        account_balance = self.get_account_balance()
        risk_amount = account_balance * (risk_percentage / 100)
        
        price_difference = abs(entry_price - stop_loss)
        position_size = risk_amount / price_difference
        
        return self.validate_position_size(position_size, entry_price)

    def validate_position_size(self, position_size, entry_price):
        market_limits = self.get_market_limits()
        
        # Adjust for minimum trade size
        if position_size < market_limits['min_size']:
            position_size = market_limits['min_size']
        
        # Adjust for maximum trade size
        if position_size > market_limits['max_size']:
            position_size = market_limits['max_size']
        
        return {
            'size': position_size,
            'value': position_size * entry_price,
            'limits_applied': position_size != position_size
        }

    def get_account_balance(self):
        try:
            balance = self.binance.fetch_balance()
            return float(balance['total']['USDT'])
        except Exception as e:
            self.logger.error(f"Error fetching account balance: {e}")
            return 0.0

    def get_market_limits(self):
        try:
            market_info = self.binance.fetch_market(self.symbol)
            return {
                'min_size': float(market_info['limits']['amount']['min']),
                'max_size': float(market_info['limits']['amount']['max']),
                'price_precision': market_info['precision']['price'],
                'size_precision': market_info['precision']['amount']
            }
        except Exception as e:
            self.logger.error(f"Error fetching market limits: {e}")
            return {
                'min_size': 0,
                'max_size': float('inf'),
                'price_precision': 8,
                'size_precision': 8
            }

    def calculate_order_quantities(self, entry_price, position_size):
        market_limits = self.get_market_limits()
        
        base_quantity = position_size
        quote_quantity = position_size * entry_price
        
        return {
            'base_quantity': self.round_to_precision(base_quantity, market_limits['size_precision']),
            'quote_quantity': self.round_to_precision(quote_quantity, market_limits['price_precision']),
            'entry_price': self.round_to_precision(entry_price, market_limits['price_precision'])
        }

    def round_to_precision(self, value, precision):
        return round(value, precision)
    def execute_pattern_trade(self, pattern, entry_price, position_size):
        order_data = {
            'symbol': self.symbol,
            'type': 'LIMIT',
            'side': self.determine_trade_side(pattern),
            'amount': position_size,
            'price': entry_price,
            'params': self.get_order_params(pattern)
        }
        
        return self.place_order(order_data)

    def determine_trade_side(self, pattern):
        pattern_types = {
            'double_bottom': 'buy',
            'double_top': 'sell',
            'higher_lows': 'buy',
            'lower_highs': 'sell'
        }
        return pattern_types.get(pattern['type'], 'buy')

    def get_order_params(self, pattern):
        return {
            'timeInForce': 'GTC',
            'postOnly': True,
            'reduceOnly': False
        }

    def place_order(self, order_data):
        try:
            order = self.binance.create_order(**order_data)
            self.logger.info(f"Order placed successfully: {order['id']}")
            return self.track_order(order)
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            return None

    def track_order(self, order):
        order_tracker = {
            'order_id': order['id'],
            'symbol': order['symbol'],
            'status': order['status'],
            'entry_price': float(order['price']),
            'size': float(order['amount']),
            'side': order['side'],
            'fills': [],
            'trailing_stop': None
        }
        
        self.active_orders[order['id']] = order_tracker
        return order_tracker

    def monitor_active_orders(self):
        for order_id, tracker in self.active_orders.items():
            try:
                order_status = self.binance.fetch_order(order_id, tracker['symbol'])
                self.update_order_tracker(tracker, order_status)
                
                if order_status['status'] == 'filled':
                    self.handle_filled_order(tracker)
                elif order_status['status'] == 'canceled':
                    self.handle_canceled_order(tracker)
                    
            except Exception as e:
                self.logger.error(f"Error monitoring order {order_id}: {e}")

    def update_order_tracker(self, tracker, order_status):
        tracker.update({
            'status': order_status['status'],
            'fills': order_status.get('fills', []),
            'last_update': self.current_timestamp
        })
    def handle_filled_order(self, tracker):
        self.logger.info(f"Order filled: {tracker['order_id']}")
        
        # Set up trailing stop
        stop_price = self.calculate_trailing_stop(tracker)
        tracker['trailing_stop'] = self.place_trailing_stop(
            tracker['symbol'],
            tracker['side'],
            tracker['size'],
            stop_price
        )
        
        # Update position tracking
        self.active_positions[tracker['symbol']] = {
            'entry_price': tracker['entry_price'],
            'size': tracker['size'],
            'side': tracker['side'],
            'trailing_stop': tracker['trailing_stop'],
            'unrealized_pnl': 0
        }
        
        # Remove from active orders
        del self.active_orders[tracker['order_id']]

    def handle_canceled_order(self, tracker):
        self.logger.info(f"Order canceled: {tracker['order_id']}")
        del self.active_orders[tracker['order_id']]

    def place_trailing_stop(self, symbol, side, size, stop_price):
        try:
            stop_side = 'sell' if side == 'buy' else 'buy'
            stop_order = self.binance.create_order(
                symbol=symbol,
                type='TRAILING_STOP_MARKET',
                side=stop_side,
                amount=size,
                params={
                    'stopPrice': stop_price,
                    'callbackRate': 1.0  # 1% callback rate
                }
            )
            return stop_order
        except Exception as e:
            self.logger.error(f"Error placing trailing stop: {e}")
            return None
    def monitor_positions(self):
        for symbol, position in self.active_positions.items():
            current_price = self.get_current_price(symbol)
            
            # Update unrealized PnL
            position['unrealized_pnl'] = self.calculate_pnl(
                position['side'],
                position['entry_price'],
                current_price,
                position['size']
            )
            
            # Update trailing stop
            if position['trailing_stop']:
                self.update_trailing_stop(position, current_price)
            
            # Check risk limits
            self.check_position_risk(symbol, position)

    def calculate_pnl(self, side, entry_price, current_price, size):
        if side == 'buy':
            return (current_price - entry_price) * size
        else:
            return (entry_price - current_price) * size

    def update_trailing_stop(self, position, current_price):
        if position['side'] == 'buy':
            new_stop = current_price * 0.99  # 1% trailing stop
            if new_stop > position['trailing_stop']['stopPrice']:
                self.modify_trailing_stop(position['trailing_stop'], new_stop)
        else:
            new_stop = current_price * 1.01
            if new_stop < position['trailing_stop']['stopPrice']:
                self.modify_trailing_stop(position['trailing_stop'], new_stop)

    def check_position_risk(self, symbol, position):
        risk_metrics = self.calculate_risk_metrics(position)
        
        if risk_metrics['drawdown'] > self.max_drawdown:
            self.close_position(symbol, position)
        elif risk_metrics['risk_ratio'] > self.max_risk_ratio:
            self.reduce_position(symbol, position)
    def calculate_risk_metrics(self, position):
        current_price = self.get_current_price(position['symbol'])
        
        return {
            'drawdown': self.calculate_drawdown(position, current_price),
            'risk_ratio': self.calculate_risk_ratio(position),
            'volatility': self.calculate_position_volatility(position),
            'exposure': self.calculate_exposure(position)
        }

    def calculate_drawdown(self, position, current_price):
        peak_price = max(position['entry_price'], current_price)
        return (peak_price - current_price) / peak_price * 100

    def calculate_risk_ratio(self, position):
        account_value = self.get_account_balance()
        position_value = position['size'] * position['entry_price']
        return position_value / account_value * 100

    def calculate_position_volatility(self, position):
        df = self.fetch_market_data(position['symbol'], '1h', limit=24)
        returns = df['close'].pct_change().dropna()
        return returns.std() * np.sqrt(24) * 100

    def calculate_exposure(self, position):
        total_exposure = sum(pos['size'] * pos['entry_price'] 
                            for pos in self.active_positions.values())
        return (position['size'] * position['entry_price']) / total_exposure * 100
    def reduce_position(self, symbol, position):
        reduction_amount = position['size'] * 0.5  # Reduce by 50%
        
        try:
            order = self.binance.create_order(
                symbol=symbol,
                type='MARKET',
                side='sell' if position['side'] == 'buy' else 'buy',
                amount=reduction_amount
            )
            
            # Update position size
            position['size'] -= reduction_amount
            self.logger.info(f"Position reduced for {symbol}: {reduction_amount}")
            
            return order
        except Exception as e:
            self.logger.error(f"Error reducing position: {e}")
            return None

    def close_position(self, symbol, position):
        try:
            order = self.binance.create_order(
                symbol=symbol,
                type='MARKET',
                side='sell' if position['side'] == 'buy' else 'buy',
                amount=position['size']
            )
            
            # Remove position from tracking
            del self.active_positions[symbol]
            self.logger.info(f"Position closed for {symbol}")
            
            # Record trade history
            self.record_trade_history(symbol, position, order)
            
            return order
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
            return None

    def record_trade_history(self, symbol, position, close_order):
        trade_record = {
            'symbol': symbol,
            'entry_price': position['entry_price'],
            'exit_price': float(close_order['price']),
            'size': position['size'],
            'side': position['side'],
            'pnl': self.calculate_pnl(
                position['side'],
                position['entry_price'],
                float(close_order['price']),
                position['size']
            ),
            'duration': self.calculate_trade_duration(position),
            'timestamp': self.current_timestamp
        }
        
        self.trade_history.append(trade_record)
    def analyze_trade_performance(self):
        performance_metrics = {
            'total_trades': len(self.trade_history),
            'winning_trades': self.count_winning_trades(),
            'profit_factor': self.calculate_profit_factor(),
            'average_return': self.calculate_average_return(),
            'sharpe_ratio': self.calculate_sharpe_ratio(),
            'max_drawdown': self.calculate_max_drawdown(),
            'win_rate': self.calculate_win_rate()
        }
        
        return self.generate_performance_report(performance_metrics)

    def count_winning_trades(self):
        return sum(1 for trade in self.trade_history if trade['pnl'] > 0)

    def calculate_profit_factor(self):
        winning_pnl = sum(trade['pnl'] for trade in self.trade_history if trade['pnl'] > 0)
        losing_pnl = abs(sum(trade['pnl'] for trade in self.trade_history if trade['pnl'] < 0))
        return winning_pnl / losing_pnl if losing_pnl > 0 else float('inf')

    def calculate_average_return(self):
        if not self.trade_history:
            return 0
        total_return = sum(trade['pnl'] for trade in self.trade_history)
        return total_return / len(self.trade_history)

    def calculate_sharpe_ratio(self):
        returns = [trade['pnl'] for trade in self.trade_history]
        if not returns:
            return 0
        return np.mean(returns) / (np.std(returns) if np.std(returns) > 0 else 1)
    def calculate_max_drawdown(self):
        equity_curve = self.generate_equity_curve()
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (running_max - equity_curve) / running_max
        return np.max(drawdowns) * 100

    def calculate_win_rate(self):
        if not self.trade_history:
            return 0
        winning_trades = self.count_winning_trades()
        return (winning_trades / len(self.trade_history)) * 100

    def generate_equity_curve(self):
        equity = [1000]  # Starting capital
        for trade in self.trade_history:
            equity.append(equity[-1] + trade['pnl'])
        return np.array(equity)

    def generate_performance_report(self, metrics):
        report = {
            'summary': {
                'total_trades': metrics['total_trades'],
                'win_rate': f"{metrics['win_rate']:.2f}%",
                'profit_factor': f"{metrics['profit_factor']:.2f}",
                'sharpe_ratio': f"{metrics['sharpe_ratio']:.2f}",
                'max_drawdown': f"{metrics['max_drawdown']:.2f}%"
            },
            'detailed_metrics': self.calculate_detailed_metrics(),
            'trade_distribution': self.analyze_trade_distribution(),
            'recommendations': self.generate_trading_recommendations()
        }
        
        return report
    def calculate_detailed_metrics(self):
        return {
            'trade_metrics': {
                'average_trade_duration': self.calculate_average_trade_duration(),
                'largest_win': self.get_largest_win(),
                'largest_loss': self.get_largest_loss(),
                'average_win': self.calculate_average_win(),
                'average_loss': self.calculate_average_loss(),
                'risk_reward_ratio': self.calculate_risk_reward_ratio()
            },
            'volatility_metrics': {
                'daily_volatility': self.calculate_daily_volatility(),
                'monthly_volatility': self.calculate_monthly_volatility(),
                'beta': self.calculate_market_beta()
            },
            'risk_metrics': {
                'value_at_risk': self.calculate_value_at_risk(),
                'expected_shortfall': self.calculate_expected_shortfall(),
                'risk_adjusted_return': self.calculate_risk_adjusted_return()
            }
        }

    def analyze_trade_distribution(self):
        return {
            'time_analysis': self.analyze_time_distribution(),
            'size_analysis': self.analyze_position_sizes(),
            'pattern_analysis': self.analyze_pattern_performance(),
            'market_conditions': self.analyze_market_conditions()
        }

    def generate_trading_recommendations(self):
        return {
            'position_sizing': self.recommend_position_sizing(),
            'risk_management': self.recommend_risk_parameters(),
            'pattern_preferences': self.recommend_pattern_preferences(),
            'timing_optimization': self.recommend_timing_optimization()
        }
    def calculate_average_trade_duration(self):
        durations = [trade['duration'] for trade in self.trade_history]
        return np.mean(durations) if durations else 0

    def get_largest_win(self):
        wins = [trade['pnl'] for trade in self.trade_history if trade['pnl'] > 0]
        return max(wins) if wins else 0

    def get_largest_loss(self):
        losses = [trade['pnl'] for trade in self.trade_history if trade['pnl'] < 0]
        return min(losses) if losses else 0

    def calculate_average_win(self):
        wins = [trade['pnl'] for trade in self.trade_history if trade['pnl'] > 0]
        return np.mean(wins) if wins else 0

    def calculate_average_loss(self):
        losses = [trade['pnl'] for trade in self.trade_history if trade['pnl'] < 0]
        return np.mean(losses) if losses else 0

    def calculate_risk_reward_ratio(self):
        avg_win = self.calculate_average_win()
        avg_loss = abs(self.calculate_average_loss())
        return avg_win / avg_loss if avg_loss != 0 else float('inf')
    def calculate_daily_volatility(self):
        daily_returns = self.get_daily_returns()
        return np.std(daily_returns) * np.sqrt(252) if len(daily_returns) > 0 else 0

    def calculate_monthly_volatility(self):
        monthly_returns = self.get_monthly_returns()
        return np.std(monthly_returns) * np.sqrt(12) if len(monthly_returns) > 0 else 0

    def calculate_market_beta(self):
        market_returns = self.get_market_returns()
        strategy_returns = self.get_strategy_returns()
        
        if len(market_returns) == len(strategy_returns) and len(market_returns) > 0:
            covariance = np.cov(strategy_returns, market_returns)[0][1]
            market_variance = np.var(market_returns)
            return covariance / market_variance if market_variance != 0 else 0
        return 0

    def get_daily_returns(self):
        equity_curve = self.generate_equity_curve()
        return np.diff(equity_curve) / equity_curve[:-1]

    def get_monthly_returns(self):
        daily_returns = self.get_daily_returns()
        return np.array([np.sum(daily_returns[i:i+21]) for i in range(0, len(daily_returns), 21)])

    def calculate_value_at_risk(self, confidence_level=0.95):
        returns = self.get_daily_returns()
        if len(returns) > 0:
            return np.percentile(returns, (1 - confidence_level) * 100)
        return 0

    def calculate_expected_shortfall(self, confidence_level=0.95):
        returns = self.get_daily_returns()
        var = self.calculate_value_at_risk(confidence_level)
        if len(returns) > 0:
            return np.mean(returns[returns <= var])
        return 0

    def calculate_risk_adjusted_return(self):
        returns = self.get_daily_returns()
        if len(returns) > 0:
            avg_return = np.mean(returns)
            volatility = np.std(returns)
            return (avg_return / volatility) * np.sqrt(252) if volatility != 0 else 0
        return 0

    def analyze_drawdown_periods(self):
        equity_curve = self.generate_equity_curve()
        peak = equity_curve[0]
        drawdowns = []
        current_drawdown = None
        
        for i, value in enumerate(equity_curve):
            if value > peak:
                peak = value
                if current_drawdown:
                    current_drawdown['end'] = i - 1
                    drawdowns.append(current_drawdown)
                    current_drawdown = None
            else:
                drawdown = (peak - value) / peak
                if current_drawdown is None:
                    current_drawdown = {'start': i, 'peak': peak, 'depth': drawdown}
                elif drawdown > current_drawdown['depth']:
                    current_drawdown['depth'] = drawdown
        
        return drawdowns
    def analyze_time_distribution(self):
        trades = pd.DataFrame(self.trade_history)
        return {
            'hourly_distribution': trades.groupby(trades['timestamp'].dt.hour)['pnl'].mean().to_dict(),
            'daily_distribution': trades.groupby(trades['timestamp'].dt.dayofweek)['pnl'].mean().to_dict(),
            'monthly_distribution': trades.groupby(trades['timestamp'].dt.month)['pnl'].mean().to_dict(),
            'best_trading_hours': self.identify_best_trading_hours(trades)
        }

    def analyze_position_sizes(self):
        trades = pd.DataFrame(self.trade_history)
        return {
            'size_distribution': trades.groupby(pd.qcut(trades['size'], 5))['pnl'].mean().to_dict(),
            'optimal_size_range': self.calculate_optimal_size_range(trades),
            'size_performance_correlation': trades['size'].corr(trades['pnl']),
            'risk_adjusted_sizes': self.calculate_risk_adjusted_sizes(trades)
        }

    def analyze_pattern_performance(self):
        trades = pd.DataFrame(self.trade_history)
        return {
            'pattern_win_rates': trades.groupby('pattern')['pnl'].apply(lambda x: (x > 0).mean()).to_dict(),
            'pattern_profit_factors': self.calculate_pattern_profit_factors(trades),
            'pattern_risk_metrics': self.calculate_pattern_risk_metrics(trades),
            'best_performing_patterns': self.identify_best_patterns(trades)
        }

    def analyze_market_conditions(self):
        trades = pd.DataFrame(self.trade_history)
        return {
            'volatility_performance': self.analyze_volatility_impact(trades),
            'trend_performance': self.analyze_trend_impact(trades),
            'volume_performance': self.analyze_volume_impact(trades),
            'optimal_conditions': self.identify_optimal_conditions(trades)
        }

    def recommend_position_sizing(self):
        analysis = self.analyze_position_sizes()
        return {
            'optimal_size': analysis['optimal_size_range']['median'],
            'size_ranges': {
                'conservative': analysis['optimal_size_range']['min'],
                'moderate': analysis['optimal_size_range']['median'],
                'aggressive': analysis['optimal_size_range']['max']
            },
            'scaling_factors': self.calculate_position_scaling_factors(),
            'risk_based_adjustments': self.calculate_risk_based_size_adjustments()
        }

    def recommend_risk_parameters(self):
        return {
            'stop_loss_levels': {
                'tight': self.calculate_optimal_stop_loss(0.5),
                'moderate': self.calculate_optimal_stop_loss(1.0),
                'wide': self.calculate_optimal_stop_loss(2.0)
            },
            'position_limits': {
                'max_position_size': self.calculate_max_position_size(),
                'max_portfolio_exposure': self.calculate_max_portfolio_exposure()
            },
            'risk_limits': {
                'daily_loss_limit': self.calculate_daily_risk_limit(),
                'max_drawdown_limit': self.calculate_max_drawdown_limit()
            }
        }

    def recommend_pattern_preferences(self):
        pattern_analysis = self.analyze_pattern_performance()
        return {
            'preferred_patterns': [p for p, metrics in pattern_analysis['pattern_win_rates'].items() 
                                if metrics > 0.6],
            'pattern_rankings': self.rank_patterns_by_performance(),
            'pattern_combinations': self.identify_synergistic_patterns(),
            'market_specific_patterns': self.analyze_market_specific_patterns()
        }

    def recommend_timing_optimization(self):
        time_analysis = self.analyze_time_distribution()
        return {
            'optimal_trading_hours': time_analysis['best_trading_hours'],
            'session_recommendations': {
                'asian': self.analyze_session_performance('asian'),
                'european': self.analyze_session_performance('european'),
                'american': self.analyze_session_performance('american')
            },
            'timing_adjustments': self.calculate_timing_adjustments(),
            'volatility_based_timing': self.analyze_volatility_based_timing()
        }



    def calculate_risk_reward(self, target, neckline, head):
        # Calculate potential reward
        reward = neckline - target
        
        # Calculate risk (stop loss above head)
        risk = head - neckline
        
        return reward / risk if risk != 0 else 0



    def validate_hs_pattern(self, pattern_data):
        if not all([pattern_data['left_shoulder'], pattern_data['head'], 
                    pattern_data['right_shoulder'], pattern_data['neckline']]):
            return False
        
        # Validate price relationships
        ls_price = pattern_data['left_shoulder']['price']
        h_price = pattern_data['head']['price']
        rs_price = pattern_data['right_shoulder']['price']
        
        # Head must be higher than shoulders
        if not (h_price > ls_price and h_price > rs_price):
            return False
        
        # Shoulders should be roughly equal height (within 10%)
        shoulder_diff = abs(ls_price - rs_price) / ls_price
        if shoulder_diff > 0.10:
            return False
        
        # Validate time symmetry
        ls_idx = pattern_data['left_shoulder']['index']
        h_idx = pattern_data['head']['index']
        rs_idx = pattern_data['right_shoulder']['index']
        
        left_distance = h_idx - ls_idx
        right_distance = rs_idx - h_idx
        
        # Time symmetry should be within 20%
        time_symmetry = abs(left_distance - right_distance) / left_distance
        if time_symmetry > 0.20:
            return False
        
        return True

    def calculate_pattern_probability(self, pattern_data):
        score = 0
        max_score = 100
        
        # Price symmetry score (30 points)
        ls_price = pattern_data['left_shoulder']['price']
        rs_price = pattern_data['right_shoulder']['price']
        price_symmetry = 1 - (abs(ls_price - rs_price) / ls_price)
        score += 30 * price_symmetry
        
        # Volume confirmation score (30 points)
        volume_score = self.calculate_volume_confirmation(pattern_data)
        score += 30 * volume_score
        
        # Neckline quality score (20 points)
        neckline_score = self.calculate_neckline_quality(pattern_data['neckline'])
        score += 20 * neckline_score
        
        # Time symmetry score (20 points)
        time_symmetry = self.calculate_time_symmetry(pattern_data)
        score += 20 * time_symmetry
        
        return score / max_score



    def calculate_volume_confirmation(self, pattern_data):
        # Volume should be highest at left shoulder and head
        volumes = [
            pattern_data['left_shoulder']['volume'],
            pattern_data['head']['volume'],
            pattern_data['right_shoulder']['volume']
        ]
        
        if volumes[0] > volumes[2] and volumes[1] > volumes[2]:
            return 1.0
        elif volumes[1] > volumes[2]:
            return 0.7
        else:
            return 0.3

    def calculate_neckline_quality(self, neckline):
        # Assess neckline slope and consistency
        slope = abs(neckline['slope'])
        
        if slope < 0.1:  # Nearly horizontal neckline (ideal)
            return 1.0
        elif slope < 0.2:  # Slightly sloped
            return 0.8
        elif slope < 0.3:  # Moderately sloped
            return 0.5
        else:  # Steeply sloped
            return 0.3

    def calculate_time_symmetry(self, pattern_data):
        ls_idx = pattern_data['left_shoulder']['index']
        h_idx = pattern_data['head']['index']
        rs_idx = pattern_data['right_shoulder']['index']
        
        left_distance = h_idx - ls_idx
        right_distance = rs_idx - h_idx
        
        symmetry = 1 - (abs(left_distance - right_distance) / left_distance)
        return max(0, min(1, symmetry))

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
    def analyze_market_regime(self, df):
        df = self.calculate_technical_indicators(df)
        
        regime_data = {
            'trend_strength': self.detect_trend_strength(df),
            'volatility': df['close'].pct_change().std() * np.sqrt(252),
            'volume_profile': self.analyze_volume_profile(df),
            'market_phase': self.detect_market_phase(df),
            'momentum': self.analyze_momentum(df),
            'support_resistance': self.identify_key_levels(df),
            'order_flow': self.analyze_order_flow(df)
        }
        
        return regime_data
    def calculate_dmi_adx(self, df):
        df['plus_di'] = talib.PLUS_DI(df['high'], df['low'], df['close'])
        df['minus_di'] = talib.MINUS_DI(df['high'], df['low'], df['close'])
        df['adx'] = talib.ADX(df['high'], df['low'], df['close'])
        return df
    def analyze_adx_strategy(self, market, timeframes=['15m', '1h', '4h']):
        adx_signals = {}
        for tf in timeframes:
            df = self.fetch_market_data(market, tf)
            df = self.calculate_dmi_adx(df)
            
            adx_signals[tf] = {
                'trend_strength': self.confirm_trend_strength(df),
                'entry_signals': self.generate_adx_entry_signals(df),
                'exit_signals': self.generate_adx_exit_signals(df)
            }
        
        return self.combine_adx_timeframes(adx_signals)

    def confirm_trend_strength(self, df):
        return {
            'strong_trend': df['adx'].iloc[-1] > 25,
            'trend_direction': 'bullish' if df['plus_di'].iloc[-1] > df['minus_di'].iloc[-1] else 'bearish',
            'trend_momentum': df['adx'].diff().iloc[-1]
        }

    def generate_adx_entry_signals(self, df):
        signals = []
        
        # DI+ crosses above DI-
        if (df['plus_di'].iloc[-2] <= df['minus_di'].iloc[-2] and 
            df['plus_di'].iloc[-1] > df['minus_di'].iloc[-1] and 
            df['adx'].iloc[-1] > 25):
            signals.append({
                'type': 'ADX_BULLISH_CROSS',
                'strength': df['adx'].iloc[-1],
                'probability': self.calculate_signal_probability(df)
            })
        
        return signals

    def generate_adx_exit_signals(self, df):
        signals = []
        
        # ADX weakening
        if (df['adx'].iloc[-1] < df['adx'].iloc[-2] and 
            df['adx'].iloc[-2] < df['adx'].iloc[-3]):
            signals.append({
                'type': 'ADX_TREND_WEAKENING',
                'strength': df['adx'].iloc[-1],
                'probability': self.calculate_signal_probability(df)
            })
        
        return signals

    def combine_adx_timeframes(self, adx_signals):
        combined_score = 0
        weights = {'15m': 0.3, '1h': 0.5, '4h': 0.7}
        
        for tf, signals in adx_signals.items():
            if signals['trend_strength']['strong_trend']:
                combined_score += weights[tf]
        
        return {
            'overall_strength': combined_score,
            'signals': adx_signals,
            'recommendation': 'strong_buy' if combined_score > 1.2 else 'neutral'
        }

    def analyze_market_structure(self, df):
        df['swing_high'] = df['high'].rolling(window=5, center=True).max()
        df['swing_low'] = df['low'].rolling(window=5, center=True).min()
        df['higher_highs'] = df['swing_high'] > df['swing_high'].shift(1)
        df['higher_lows'] = df['swing_low'] > df['swing_low'].shift(1)
        return df

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
        if market.startswith('USDT/'):
            return None
            
        try:
            ohlcv = self.binance.fetch_ohlcv(market, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            self.logger.error(f"Data fetch error for {market}: {e}")
            return None
    def analyze_market(self, market, timeframe):
        try:
            df = self.fetch_market_data(market, timeframe)
            if df is not None and not df.empty:
                # Calculate technical indicators
                df['rsi'] = talib.RSI(df['close'])
                df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'])
                df['ema_20'] = talib.EMA(df['close'], timeperiod=20)
                df['ema_50'] = talib.EMA(df['close'], timeperiod=50)
                
                volume_profile = self.analyze_volume_profile(df)
                signals = self.generate_signals(df)
                
                if volume_profile and signals:
                    for signal in signals:
                        signal['volume_context'] = volume_profile['volume_trend']
                        signal['vwap'] = volume_profile['vwap']
                
                if signals:
                    for signal in signals:
                        message = (
                            f"🔔 Signal Alert!\n"
                            f"Market: {market}\n"
                            f"Signal: {signal['type']}\n"
                            f"Price: {signal['price']}\n"
                            f"VWAP: {signal.get('vwap', 'N/A')}\n"
                            f"Volume Trend: {signal.get('volume_context', 'N/A')}\n"
                            f"RSI: {df['rsi'].iloc[-1]:.2f}\n"
                            f"Timeframe: {timeframe}"
                        )
                        self.send_telegram_update(message)
                        
                        self.sql_operations('insert', self.db_signals, 'Signals',
                                        market=market,
                                        timeframe=timeframe,
                                        signal_type=signal['type'],
                                        price=signal['price'],
                                        volume_trend=signal.get('volume_context', ''),
                                        vwap=signal.get('vwap', 0.0),
                                        rsi=df['rsi'].iloc[-1],
                                        timestamp=str(datetime.now()))
                        
        except Exception as e:
            self.logger.error(f"Error analyzing market {market}: {e}")

    def _analyze_volume_data(self, df):
        """Analyze volume patterns efficiently"""
        return {
            'volume_sma': df['volume'].rolling(20).mean().iloc[-1],
            'volume_trend': 'increasing' if df['volume'].is_monotonic_increasing else 'decreasing',
            'volume_ratio': df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
        }

    def _calculate_base_sentiment(self, df):
        """Calculate base sentiment metrics"""
        return {
            'price_trend': 'bullish' if df['close'].iloc[-1] > df['close'].iloc[-2] else 'bearish',
            'volume_impact': 'high' if df['volume'].iloc[-1] > df['volume'].rolling(20).mean().iloc[-1] else 'low',
            'momentum': df['close'].pct_change().iloc[-1]
        }

    def _generate_market_analysis(self, market_data):
        """Generate final market analysis"""
        if not market_data:
            return None
            
        return {
            'market': market_data['market'],
            'analysis': {
                'technical_score': self._calculate_technical_score(market_data['technical']),
                'volume_score': self._calculate_volume_score(market_data['volume']),
                'sentiment_score': self._calculate_sentiment_score(market_data['sentiment'])
            },
            'signals': self._generate_signals(market_data)
        }
    def classify_trading_style(self, signal_data, timeframe):
        # Day trading timeframes
        day_trading_timeframes = ['1m', '5m', '15m', '30m', '1h']
        
        is_day_trade = timeframe in day_trading_timeframes
        
        # Add trading style to signal data
        signal_data['trading_style'] = 'Day Trade 📈' if is_day_trade else 'Swing Trade 🌊'
        return signal_data
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
                # Classify trading style and calculate additional metrics
                signal = self.classify_trading_style(signal, timeframe)
                risk_level = self.calculate_risk_level(signal)
                
                # Get technical levels
                df = self.fetch_market_data(market, timeframe)
                vwap_data = self.calculate_vwap_levels(df)
                pivot_points = self.calculate_pivot_points(df)
                fib_levels = self.calculate_fibonacci_levels(df)
                
                # Log signal detection
                self.logger.info(f"New signal detected for {market}: {signal['type']}")
                
                # Store in database
                self.store_signals_db([signal], market, timeframe)
                
                # Construct enhanced message based on signal type
                if signal['type'] == 'HTF_SUPPORT_WICK':
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
                
                else:
                    message = (
                        f"🔔 {signal['trading_style']}\n"
                        f"Signal Alert!\n"
                        f"Market: {market}\n"
                        f"Signal: {signal['type']}\n"
                        f"Price: {signal.get('price', 'N/A')}\n"
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
    def dynamic_position_sizing(self, market, signal):
        volatility = self.calculate_market_volatility(market)
        correlation_factor = self.get_portfolio_correlation(market)
        account_risk = self.calculate_account_risk()
        
        base_position = self.calculate_position_size(signal)
        adjusted_position = base_position * (1 - correlation_factor) * (1 - volatility)
        
        return min(adjusted_position, account_risk['max_position'])

    def portfolio_correlation_manager(self):
        portfolio = self.get_active_positions()
        correlation_matrix = self.market_correlation_analysis()
        
        risk_adjustments = {}
        for market in portfolio:
            corr_score = correlation_matrix[market].mean()
            risk_adjustments[market] = {
                'current_exposure': portfolio[market]['exposure'],
                'suggested_adjustment': 1 - corr_score,
                'max_allowed': self.calculate_max_exposure(market)
            }
        return risk_adjustments




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
    def analyze_market_profile(self, market, timeframe):
        df = self.fetch_market_data(market, timeframe)
        volume_profile = self.calculate_volume_profile(df)
        poc_levels = self.find_poc_levels(volume_profile)
        value_areas = self.calculate_value_areas(volume_profile)
        
        return {
            'volume_profile': volume_profile,
            'poc_levels': poc_levels,
            'value_areas': value_areas,
            'distribution_type': self.classify_distribution(volume_profile)
        }

    def institutional_order_detection(self, market):
        order_flow = self.analyze_order_flow(market)
        large_orders = self.detect_large_orders(market)
        iceberg_orders = self.detect_iceberg_orders(market)
        
        return {
            'large_orders': large_orders,
            'iceberg_detection': iceberg_orders,
            'order_flow_imbalance': order_flow['depth_imbalance'],
            'institutional_levels': self.find_institutional_levels(market)
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