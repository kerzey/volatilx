# import yfinance as yf
# import pandas_ta as ta
# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta
# import matplotlib.pyplot as plt
# from scipy.signal import find_peaks
# import warnings
# warnings.filterwarnings('ignore')

# class ComprehensiveIndicatorFetcher:
#     def __init__(self):
#         self.data_cache = {}
    
#     def get_stock_data(self, symbol: str, period: str = "6mo"):
#         """Fetch stock data using yfinance"""
#         if symbol not in self.data_cache:
#             ticker = yf.Ticker(symbol)
#             self.data_cache[symbol] = ticker.history(period=period)
#         return self.data_cache[symbol]
    
#     def get_all_indicators(self, symbol: str, period: str = "6mo"):
#         """Get all your requested indicators using pandas-ta"""
#         df = self.get_stock_data(symbol, period)
        
#         if df.empty:
#             return None
        
#         # Add all indicators to the dataframe
#         indicators = {}
        
#         # 1. MACD (12, 26, 9) - Ready-made
#         macd_data = ta.macd(df['Close'], fast=12, slow=26, signal=9)
#         indicators['MACD'] = {
#             'macd': macd_data['MACD_12_26_9'].iloc[-1],
#             'signal': macd_data['MACDs_12_26_9'].iloc[-1],
#             'histogram': macd_data['MACDh_12_26_9'].iloc[-1],
#             'crossover': 'bullish' if macd_data['MACD_12_26_9'].iloc[-1] > macd_data['MACDs_12_26_9'].iloc[-1] else 'bearish'
#         }
        
#         # 2. ADX and DI (14-20 periods) - Ready-made
#         for period in [14, 20]:
#             adx_data = ta.adx(df['High'], df['Low'], df['Close'], length=period)
#             indicators[f'ADX_{period}'] = {
#                 'adx': adx_data[f'ADX_{period}'].iloc[-1],
#                 'di_plus': adx_data[f'DMP_{period}'].iloc[-1],
#                 'di_minus': adx_data[f'DMN_{period}'].iloc[-1],
#                 'trend_strength': 'strong' if adx_data[f'ADX_{period}'].iloc[-1] > 25 else 'weak'
#             }
        
#         # 3. OBV (On-Balance Volume) - Ready-made
#         obv_data = ta.obv(df['Close'], df['Volume'])
#         indicators['OBV'] = {
#             'current': obv_data.iloc[-1],
#             'previous': obv_data.iloc[-2],
#             'trend': 'bullish' if obv_data.iloc[-1] > obv_data.iloc[-2] else 'bearish'
#         }
        
#         # 4. Wavetrend - Ready-made in pandas-ta
#         # Note: pandas-ta has a wavetrend implementation
#         try:
#             wt_data = ta.wt(df['High'], df['Low'], df['Close'])
#             if wt_data is not None and not wt_data.empty:
#                 wt1 = wt_data.iloc[:, 0].iloc[-1]  # WT1
#                 wt2 = wt_data.iloc[:, 1].iloc[-1]  # WT2
#                 indicators['Wavetrend'] = {
#                     'wt1': wt1,
#                     'wt2': wt2,
#                     'cross': 'bullish' if wt1 > wt2 else 'bearish',
#                     'overbought': wt1 > 60,
#                     'oversold': wt1 < -60
#                 }
#         except:
#             indicators['Wavetrend'] = {'error': 'Not available'}
        
#         # 5. Fibonacci Retracements - Semi-automated
#         indicators['Fibonacci'] = self.calculate_fibonacci_levels(df)
        
#         # 6. Elliott Wave - Pattern detection (basic)
#         indicators['Elliott_Wave'] = self.detect_elliott_wave_pattern(df)
        
#         return {
#             'symbol': symbol,
#             'timestamp': datetime.now().isoformat(),
#             'price': df['Close'].iloc[-1],
#             'indicators': indicators
#         }
    
#     def calculate_fibonacci_levels(self, df):
#         """Calculate Fibonacci retracement levels"""
#         # Get recent high and low (last 50 periods)
#         recent_data = df.tail(50)
#         high = recent_data['High'].max()
#         low = recent_data['Low'].min()
        
#         diff = high - low
        
#         return {
#             'high': high,
#             'low': low,
#             'levels': {
#                 '0%': high,
#                 '23.6%': high - (diff * 0.236),
#                 '38.2%': high - (diff * 0.382),
#                 '50%': high - (diff * 0.5),
#                 '61.8%': high - (diff * 0.618),
#                 '78.6%': high - (diff * 0.786),
#                 '100%': low
#             }
#         }
    
#     def detect_elliott_wave_pattern(self, df):
#         """Basic Elliott Wave pattern detection"""
#         # This is a simplified version - true Elliott Wave is very complex
#         recent_data = df.tail(20)
#         highs = recent_data['High'].values
#         lows = recent_data['Low'].values
        
#         # Simple trend detection
#         if len(highs) >= 5:
#             trend = "bullish" if highs[-1] > highs[0] else "bearish"
#             return {
#                 'trend': trend,
#                 'note': 'Simplified pattern detection - not full Elliott Wave analysis'
#             }
        
#         return {'error': 'Insufficient data for pattern detection'}

# class ComprehensiveMultiTimeframeAnalyzer:
#     def __init__(self):
#         self.fib_analyzer = AdvancedFibonacciAnalyzer()
#         self.wave_analyzer = ElliottWaveAnalyzer()
        
#         self.valid_intervals = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
#         self.interval_limits = {
#             '1m': {'max_period': '7d', 'max_days': 7},
#             '2m': {'max_period': '60d', 'max_days': 60},
#             '5m': {'max_period': '60d', 'max_days': 60},
#             '15m': {'max_period': '60d', 'max_days': 60},
#             '30m': {'max_period': '60d', 'max_days': 60},
#             '60m': {'max_period': '730d', 'max_days': 730},
#             '90m': {'max_period': '60d', 'max_days': 60},
#             '1h': {'max_period': '730d', 'max_days': 730},
#             '1d': {'max_period': 'max', 'max_days': 'unlimited'},
#             '5d': {'max_period': 'max', 'max_days': 'unlimited'},
#             '1wk': {'max_period': 'max', 'max_days': 'unlimited'},
#             '1mo': {'max_period': 'max', 'max_days': 'unlimited'},
#             '3mo': {'max_period': 'max', 'max_days': 'unlimited'}
#         }
    
#     def get_stock_data(self, symbol, interval='1d', period='6mo'):
#         """Fetch stock data with specified interval"""
#         if interval not in self.valid_intervals:
#             raise ValueError(f"Invalid interval. Must be one of: {self.valid_intervals}")
        
#         # Check period limits for the interval
#         max_period = self.interval_limits[interval]['max_period']
#         if max_period != 'max' and self._period_exceeds_limit(period, max_period):
#             print(f"Warning: Period '{period}' may exceed limit for interval '{interval}'. Using '{max_period}' instead.")
#             period = max_period
        
#         try:
#             ticker = yf.Ticker(symbol)
#             df = ticker.history(period=period, interval=interval)
            
#             if df.empty:
#                 print(f"No data returned for {symbol} with interval {interval} and period {period}")
#                 return None
            
#             print(f"✅ Fetched {len(df)} {interval} candles for {symbol} over {period}")
#             return df
            
#         except Exception as e:
#             print(f"Error fetching data for {symbol}: {e}")
#             return None
    
#     def _period_exceeds_limit(self, period, max_period):
#         """Check if requested period exceeds the limit for the interval"""
#         period_days = self._period_to_days(period)
#         max_days = self._period_to_days(max_period)
        
#         if period_days and max_days:
#             return period_days > max_days
#         return False
    
#     def _period_to_days(self, period):
#         """Convert period string to approximate days"""
#         if period == 'max':
#             return float('inf')
        
#         period_map = {
#             '1d': 1, '2d': 2, '5d': 5, '7d': 7, '10d': 10,
#             '1mo': 30, '2mo': 60, '3mo': 90, '6mo': 180,
#             '1y': 365, '2y': 730, '5y': 1825, '10y': 3650,
#             '60d': 60, '730d': 730
#         }
        
#         return period_map.get(period, None)
    
#     def calculate_comprehensive_indicators(self, df, interval):
#         """Calculate all indicators including basic, Fibonacci, and Elliott Wave"""
#         indicators = {}
        
#         # Adjust indicator periods based on timeframe
#         if interval in ['1m', '2m', '5m']:
#             # Faster settings for short timeframes
#             macd_fast, macd_slow, macd_signal = 6, 13, 4
#             adx_period = 7
#             rsi_period = 7
#         elif interval in ['15m', '30m']:
#             # Medium settings
#             macd_fast, macd_slow, macd_signal = 8, 17, 6
#             adx_period = 10
#             rsi_period = 10
#         elif interval in ['1h', '90m']:
#             # Standard settings
#             macd_fast, macd_slow, macd_signal = 12, 26, 9
#             adx_period = 14
#             rsi_period = 14
#         else:
#             # Daily+ settings
#             macd_fast, macd_slow, macd_signal = 12, 26, 9
#             adx_period = 14
#             rsi_period = 14
        
#         try:
#             # Basic Technical Indicators
#             indicators['basic'] = self._calculate_basic_indicators(df, macd_fast, macd_slow, macd_signal, adx_period, rsi_period)
            
#             # Fibonacci Analysis
#             fib_analysis = self.fib_analyzer.calculate_fibonacci_retracements(df)
#             indicators['fibonacci'] = fib_analysis if fib_analysis else {'error': 'Insufficient data for Fibonacci analysis'}
            
#             # Elliott Wave Analysis
#             wave_analysis = self.wave_analyzer.detect_wave_structure(df)
#             indicators['elliott_wave'] = wave_analysis
            
#             # Combined Trading Signals
#             indicators['trading_signals'] = self._generate_comprehensive_signals(
#                 indicators['basic'], 
#                 indicators['fibonacci'], 
#                 indicators['elliott_wave'],
#                 interval
#             )
            
#         except Exception as e:
#             print(f"Error calculating indicators for {interval}: {e}")
#             indicators['error'] = str(e)
        
#         return indicators
    
#     def _calculate_basic_indicators(self, df, macd_fast, macd_slow, macd_signal, adx_period, rsi_period):
#         """Calculate basic technical indicators with improved error handling"""
#         basic_indicators = {}
        
#         try:
#             # MACD
#             macd_data = ta.macd(df['Close'], fast=macd_fast, slow=macd_slow, signal=macd_signal)
#             if macd_data is not None and not macd_data.empty:
#                 macd_col = f'MACD_{macd_fast}_{macd_slow}_{macd_signal}'
#                 signal_col = f'MACDs_{macd_fast}_{macd_slow}_{macd_signal}'
#                 hist_col = f'MACDh_{macd_fast}_{macd_slow}_{macd_signal}'
                
#                 basic_indicators['MACD'] = {
#                     'macd': macd_data[macd_col].iloc[-1] if macd_col in macd_data.columns else None,
#                     'signal': macd_data[signal_col].iloc[-1] if signal_col in macd_data.columns else None,
#                     'histogram': macd_data[hist_col].iloc[-1] if hist_col in macd_data.columns else None,
#                     'settings': f"({macd_fast},{macd_slow},{macd_signal})"
#                 }
                
#                 if basic_indicators['MACD']['macd'] and basic_indicators['MACD']['signal']:
#                     basic_indicators['MACD']['crossover'] = 'bullish' if basic_indicators['MACD']['macd'] > basic_indicators['MACD']['signal'] else 'bearish'
            
#             # ADX
#             adx_data = ta.adx(df['High'], df['Low'], df['Close'], length=adx_period)
#             if adx_data is not None and not adx_data.empty:
#                 basic_indicators['ADX'] = {
#                     'adx': adx_data[f'ADX_{adx_period}'].iloc[-1],
#                     'di_plus': adx_data[f'DMP_{adx_period}'].iloc[-1],
#                     'di_minus': adx_data[f'DMN_{adx_period}'].iloc[-1],
#                     'period': adx_period,
#                     'trend_strength': 'strong' if adx_data[f'ADX_{adx_period}'].iloc[-1] > 25 else 'weak'
#                 }
            
#             # RSI
#             rsi_data = ta.rsi(df['Close'], length=rsi_period)
#             if rsi_data is not None:
#                 basic_indicators['RSI'] = {
#                     'value': rsi_data.iloc[-1],
#                     'period': rsi_period,
#                     'overbought': rsi_data.iloc[-1] > 70,
#                     'oversold': rsi_data.iloc[-1] < 30
#                 }
            
#             # OBV
#             obv_data = ta.obv(df['Close'], df['Volume'])
#             if obv_data is not None and len(obv_data) > 1:
#                 basic_indicators['OBV'] = {
#                     'current': obv_data.iloc[-1],
#                     'previous': obv_data.iloc[-2],
#                     'trend': 'bullish' if obv_data.iloc[-1] > obv_data.iloc[-2] else 'bearish'
#                 }
            
#             # Moving Averages
#             sma_20 = ta.sma(df['Close'], length=min(20, len(df)//2))
#             sma_50 = ta.sma(df['Close'], length=min(50, len(df)//2))
            
#             if sma_20 is not None and sma_50 is not None and len(sma_20) > 0 and len(sma_50) > 0:
#                 basic_indicators['Moving_Averages'] = {
#                     'sma_20': sma_20.iloc[-1],
#                     'sma_50': sma_50.iloc[-1],
#                     'trend': 'bullish' if sma_20.iloc[-1] > sma_50.iloc[-1] else 'bearish'
#                 }
            
#             # Bollinger Bands - Fixed with better error handling
#             try:
#                 bb_data = ta.bbands(df['Close'], length=20, std=2)
#                 if bb_data is not None and not bb_data.empty:
#                     # Check for different possible column naming conventions
#                     upper_col = None
#                     middle_col = None
#                     lower_col = None
                    
#                     # Common column name patterns for Bollinger Bands
#                     possible_upper = ['BBU_20_2.0', 'BBU_20_2', 'upper', 'Upper', 'BB_upper']
#                     possible_middle = ['BBM_20_2.0', 'BBM_20_2', 'middle', 'Middle', 'BB_middle']
#                     possible_lower = ['BBL_20_2.0', 'BBL_20_2', 'lower', 'Lower', 'BB_lower']
                    
#                     # Find the correct column names
#                     for col in bb_data.columns:
#                         if any(pattern in col for pattern in possible_upper):
#                             upper_col = col
#                         elif any(pattern in col for pattern in possible_middle):
#                             middle_col = col
#                         elif any(pattern in col for pattern in possible_lower):
#                             lower_col = col
                    
#                     # If standard naming doesn't work, try to identify by position
#                     if not all([upper_col, middle_col, lower_col]) and len(bb_data.columns) >= 3:
#                         cols = list(bb_data.columns)
#                         # Typically: lower, middle, upper or upper, middle, lower
#                         if len(cols) == 3:
#                             # Sort by last value to identify upper/lower
#                             last_values = [(col, bb_data[col].iloc[-1]) for col in cols]
#                             last_values.sort(key=lambda x: x[1])
#                             lower_col = last_values[0][0]
#                             middle_col = last_values[1][0]
#                             upper_col = last_values[2][0]
                    
#                     if all([upper_col, middle_col, lower_col]):
#                         current_price = df['Close'].iloc[-1]
#                         upper_val = bb_data[upper_col].iloc[-1]
#                         middle_val = bb_data[middle_col].iloc[-1]
#                         lower_val = bb_data[lower_col].iloc[-1]
                        
#                         basic_indicators['Bollinger_Bands'] = {
#                             'upper': upper_val,
#                             'middle': middle_val,
#                             'lower': lower_val,
#                             'position': self._get_bb_position_safe(current_price, upper_val, middle_val, lower_val),
#                             'columns_used': {
#                                 'upper': upper_col,
#                                 'middle': middle_col,
#                                 'lower': lower_col
#                             }
#                         }
#                     else:
#                         print(f"Warning: Could not identify Bollinger Bands columns. Available columns: {list(bb_data.columns)}")
#                         basic_indicators['Bollinger_Bands'] = {
#                             'error': 'Could not identify column names',
#                             'available_columns': list(bb_data.columns)
#                         }
#                 else:
#                     print("Warning: Bollinger Bands calculation returned empty data")
                    
#             except Exception as bb_error:
#                 print(f"Bollinger Bands calculation error: {bb_error}")
#                 basic_indicators['Bollinger_Bands'] = {'error': str(bb_error)}
            
#         except Exception as e:
#             print(f"Error in basic indicators calculation: {e}")
#             basic_indicators['error'] = str(e)
        
#         return basic_indicators
    
#     def _get_bb_position_safe(self, current_price, upper, middle, lower):
#         """Safe version of Bollinger Bands position calculation"""
#         try:
#             # Calculate percentage position within bands
#             if upper != lower and upper is not None and lower is not None:
#                 bb_percentage = (current_price - lower) / (upper - lower) * 100
#             else:
#                 bb_percentage = 50  # Default to middle if bands are flat or invalid
            
#             # Determine position category
#             if current_price > upper:
#                 category = 'above_upper'
#             elif current_price > middle:
#                 category = 'upper_half'
#             elif current_price > lower:
#                 category = 'lower_half'
#             else:
#                 category = 'below_lower'
            
#             position_data = {
#                 'category': category,
#                 'percentage': round(bb_percentage, 2),
#                 'distance_from_middle': round(abs(current_price - middle), 4) if middle is not None else None
#             }
            
#             return position_data
            
#         except Exception as e:
#             print(f"Error calculating BB position: {e}")
#             return {
#                 'category': 'error', 
#                 'percentage': None, 
#                 'distance_from_middle': None,
#                 'error': str(e)
#             }
        
#     def _get_bb_position(self, current_price, bb_row):
#         """Enhanced version with better error handling for Bollinger Bands position"""
#         try:
#             # Handle both dictionary and Series input
#             if isinstance(bb_row, dict):
#                 upper = bb_row.get('upper')
#                 middle = bb_row.get('middle') 
#                 lower = bb_row.get('lower')
#             else:
#                 # Try different column name patterns
#                 upper = None
#                 middle = None
#                 lower = None
                
#                 # Check for standard column names
#                 for col in bb_row.index:
#                     if 'BBU' in col or 'upper' in col.lower():
#                         upper = bb_row[col]
#                     elif 'BBM' in col or 'middle' in col.lower():
#                         middle = bb_row[col]
#                     elif 'BBL' in col or 'lower' in col.lower():
#                         lower = bb_row[col]
            
#             # Validate values
#             if any(val is None or pd.isna(val) for val in [upper, middle, lower]):
#                 return {
#                     'category': 'insufficient_data',
#                     'percentage': None,
#                     'distance_from_middle': None
#                 }
            
#             # Calculate percentage position within bands
#             if upper != lower:  # Avoid division by zero
#                 bb_percentage = (current_price - lower) / (upper - lower) * 100
#             else:
#                 bb_percentage = 50  # Default to middle if bands are flat
            
#             # Determine category
#             category = self._get_position_category(current_price, upper, middle, lower)
            
#             position_data = {
#                 'category': category,
#                 'percentage': round(bb_percentage, 2),
#                 'distance_from_middle': round(abs(current_price - middle), 4)
#             }
            
#             return position_data
            
#         except Exception as e:
#             print(f"Error calculating Bollinger Bands position: {e}")
#             return {
#                 'category': 'error', 
#                 'percentage': None, 
#                 'distance_from_middle': None,
#                 'error_details': str(e)
#             }
    
#     def _generate_comprehensive_signals(self, basic_indicators, fib_analysis, wave_analysis, interval):
#         """Generate comprehensive trading signals combining all analyses"""
#         signals = {
#             'timeframe': interval,
#             'overall_bias': 'neutral',
#             'strength': 0,
#             'confidence': 'low',
#             'entry_signals': [],
#             'exit_signals': [],
#             'key_levels': {},
#             'risk_reward': {},
#             'signal_breakdown': {
#                 'basic_score': 0,
#                 'fibonacci_score': 0,
#                 'elliott_wave_score': 0,
#                 'total_score': 0
#             }
#         }
        
#         # Scoring system
#         bullish_score = 0
#         bearish_score = 0
        
#         # Basic Indicators Scoring
#         basic_bullish, basic_bearish, basic_signals = self._score_basic_indicators(basic_indicators)
#         bullish_score += basic_bullish
#         bearish_score += basic_bearish
#         signals['entry_signals'].extend(basic_signals['bullish'])
#         signals['exit_signals'].extend(basic_signals['bearish'])
#         signals['signal_breakdown']['basic_score'] = basic_bullish - basic_bearish
        
#         # Fibonacci Analysis Scoring
#         if fib_analysis and 'error' not in fib_analysis:
#             fib_bullish, fib_bearish, fib_signals = self._score_fibonacci_analysis(fib_analysis)
#             bullish_score += fib_bullish
#             bearish_score += fib_bearish
#             signals['entry_signals'].extend(fib_signals['bullish'])
#             signals['exit_signals'].extend(fib_signals['bearish'])
#             signals['signal_breakdown']['fibonacci_score'] = fib_bullish - fib_bearish
            
#             # Set key Fibonacci levels
#             signals['key_levels'].update({
#                 'fib_support': fib_analysis['nearest_support']['price'] if fib_analysis['nearest_support'] else None,
#                 'fib_resistance': fib_analysis['nearest_resistance']['price'] if fib_analysis['nearest_resistance'] else None,
#                 'fib_382': fib_analysis['key_levels']['38.2%'],
#                 'fib_618': fib_analysis['key_levels']['61.8%']
#             })
        
#         # Elliott Wave Analysis Scoring
#         if wave_analysis and wave_analysis.get('confidence', 0) > 0:
#             wave_bullish, wave_bearish, wave_signals = self._score_elliott_wave_analysis(wave_analysis)
#             bullish_score += wave_bullish
#             bearish_score += wave_bearish
#             signals['entry_signals'].extend(wave_signals['bullish'])
#             signals['exit_signals'].extend(wave_signals['bearish'])
#             signals['signal_breakdown']['elliott_wave_score'] = wave_bullish - wave_bearish
#         # Calculate overall bias and strength
#         total_score = bullish_score + bearish_score
#         signals['signal_breakdown']['total_score'] = bullish_score - bearish_score
        
#         if total_score > 0:
#             bullish_pct = (bullish_score / total_score) * 100
            
#             if bullish_pct >= 75:
#                 signals['overall_bias'] = 'strong_bullish'
#                 signals['strength'] = bullish_pct
#                 signals['confidence'] = 'high'
#             elif bullish_pct >= 60:
#                 signals['overall_bias'] = 'bullish'
#                 signals['strength'] = bullish_pct
#                 signals['confidence'] = 'medium'
#             elif bullish_pct >= 40:
#                 signals['overall_bias'] = 'neutral'
#                 signals['strength'] = 50
#                 signals['confidence'] = 'low'
#             elif bullish_pct >= 25:
#                 signals['overall_bias'] = 'bearish'
#                 signals['strength'] = 100 - bullish_pct
#                 signals['confidence'] = 'medium'
#             else:
#                 signals['overall_bias'] = 'strong_bearish'
#                 signals['strength'] = 100 - bullish_pct
#                 signals['confidence'] = 'high'
        
#         # Calculate risk/reward if we have key levels
#         if signals['key_levels'].get('fib_support') and signals['key_levels'].get('fib_resistance'):
#             current_price = fib_analysis['current_price']
#             support = signals['key_levels']['fib_support']
#             resistance = signals['key_levels']['fib_resistance']
            
#             risk = abs(current_price - support)
#             reward = abs(resistance - current_price)
            
#             if risk > 0:
#                 signals['risk_reward'] = {
#                     'ratio': reward / risk,
#                     'risk_amount': risk,
#                     'reward_potential': reward,
#                     'stop_loss': support,
#                     'take_profit': resistance
#                 }
        
#         return signals
    
#     def _score_basic_indicators(self, basic_indicators):
#         """Score basic technical indicators"""
#         bullish_score = 0
#         bearish_score = 0
#         signals = {'bullish': [], 'bearish': []}
        
#         # MACD scoring
#         if 'MACD' in basic_indicators and basic_indicators['MACD'].get('crossover'):
#             if basic_indicators['MACD']['crossover'] == 'bullish':
#                 bullish_score += 3
#                 signals['bullish'].append(f"MACD bullish crossover {basic_indicators['MACD']['settings']}")
#             else:
#                 bearish_score += 3
#                 signals['bearish'].append(f"MACD bearish crossover {basic_indicators['MACD']['settings']}")
        
#         # ADX scoring
#         if 'ADX' in basic_indicators:
#             adx_data = basic_indicators['ADX']
#             if adx_data['trend_strength'] == 'strong':
#                 if adx_data['di_plus'] > adx_data['di_minus']:
#                     bullish_score += 4
#                     signals['bullish'].append(f"Strong bullish trend (ADX: {adx_data['adx']:.1f})")
#                 else:
#                     bearish_score += 4
#                     signals['bearish'].append(f"Strong bearish trend (ADX: {adx_data['adx']:.1f})")
        
#         # RSI scoring
#         if 'RSI' in basic_indicators:
#             rsi_data = basic_indicators['RSI']
#             if rsi_data['oversold']:
#                 bullish_score += 2
#                 signals['bullish'].append(f"RSI oversold ({rsi_data['value']:.1f})")
#             elif rsi_data['overbought']:
#                 bearish_score += 2
#                 signals['bearish'].append(f"RSI overbought ({rsi_data['value']:.1f})")
        
#         # OBV scoring
#         if 'OBV' in basic_indicators:
#             obv_data = basic_indicators['OBV']
#             if obv_data['trend'] == 'bullish':
#                 bullish_score += 2
#                 signals['bullish'].append("Volume supporting uptrend (OBV)")
#             else:
#                 bearish_score += 2
#                 signals['bearish'].append("Volume supporting downtrend (OBV)")
        
#         # Moving Average scoring
#         if 'Moving_Averages' in basic_indicators:
#             ma_data = basic_indicators['Moving_Averages']
#             if ma_data['trend'] == 'bullish':
#                 bullish_score += 2
#                 signals['bullish'].append("Price above moving averages")
#             else:
#                 bearish_score += 2
#                 signals['bearish'].append("Price below moving averages")
        
#         # Bollinger Bands scoring
#         if 'Bollinger_Bands' in basic_indicators:
#             bb_data = basic_indicators['Bollinger_Bands']
#             if bb_data['position'] == 'below_lower':
#                 bullish_score += 1
#                 signals['bullish'].append("Price below lower Bollinger Band (oversold)")
#             elif bb_data['position'] == 'above_upper':
#                 bearish_score += 1
#                 signals['bearish'].append("Price above upper Bollinger Band (overbought)")
        
#         return bullish_score, bearish_score, signals
    
#     def _score_fibonacci_analysis(self, fib_analysis):
#         """Score Fibonacci retracement analysis"""
#         bullish_score = 0
#         bearish_score = 0
#         signals = {'bullish': [], 'bearish': []}
        
#         current_price = fib_analysis['current_price']
        
#         # Check proximity to key Fibonacci levels
#         for level_name, level_price in fib_analysis['key_levels'].items():
#             distance_pct = abs((current_price - level_price) / current_price) * 100
            
#             if distance_pct < 1.5:  # Within 1.5% of Fibonacci level
#                 if level_name in ['38.2%', '50%', '61.8%']:  # Key retracement levels
#                     if fib_analysis['trend_direction'] == 'uptrend':
#                         if current_price >= level_price:
#                             bullish_score += 3
#                             signals['bullish'].append(f"Price holding above {level_name} Fibonacci support")
#                         else:
#                             bearish_score += 2
#                             signals['bearish'].append(f"Price below {level_name} Fibonacci support")
#                     else:  # downtrend
#                         if current_price <= level_price:
#                             bearish_score += 3
#                             signals['bearish'].append(f"Price holding below {level_name} Fibonacci resistance")
#                         else:
#                             bullish_score += 2
#                             signals['bullish'].append(f"Price above {level_name} Fibonacci resistance")
        
#         # Trend direction scoring
#         if fib_analysis['trend_direction'] == 'uptrend':
#             bullish_score += 1
#             signals['bullish'].append("Fibonacci analysis shows uptrend structure")
#         else:
#             bearish_score += 1
#             signals['bearish'].append("Fibonacci analysis shows downtrend structure")
        
#         return bullish_score, bearish_score, signals
    
#     def _score_elliott_wave_analysis(self, wave_analysis):
#         """Score Elliott Wave analysis"""
#         bullish_score = 0
#         bearish_score = 0
#         signals = {'bullish': [], 'bearish': []}
        
#         confidence = wave_analysis.get('confidence', 0)
#         pattern = wave_analysis.get('pattern', '')
#         trend = wave_analysis.get('trend', '')
        
#         # Weight scoring by confidence
#         confidence_multiplier = confidence / 100
        
#         if pattern == 'impulse' and confidence > 50:
#             if trend == 'bullish':
#                 score = int(4 * confidence_multiplier)
#                 bullish_score += score
#                 signals['bullish'].append(f"Bullish impulse wave pattern (confidence: {confidence:.0f}%)")
#             elif trend == 'bearish':
#                 score = int(4 * confidence_multiplier)
#                 bearish_score += score
#                 signals['bearish'].append(f"Bearish impulse wave pattern (confidence: {confidence:.0f}%)")
        
#         elif pattern == 'corrective' and confidence > 40:
#             # Corrective patterns suggest trend reversal
#             if trend == 'bullish':
#                 score = int(2 * confidence_multiplier)
#                 bearish_score += score  # Corrective in bullish trend suggests bearish reversal
#                 signals['bearish'].append(f"Corrective wave pattern suggests trend exhaustion (confidence: {confidence:.0f}%)")
#             elif trend == 'bearish':
#                 score = int(2 * confidence_multiplier)
#                 bullish_score += score  # Corrective in bearish trend suggests bullish reversal
#                 signals['bullish'].append(f"Corrective wave pattern suggests trend exhaustion (confidence: {confidence:.0f}%)")
        
#         return bullish_score, bearish_score, signals
    
#     def analyze_comprehensive_multi_timeframe(self, symbol, timeframes=None, base_period='6mo'):
#         """Comprehensive analysis across multiple timeframes"""
#         if timeframes is None:
#             timeframes = ['5m', '15m', '1h', '1d']
        
#         results = {}
        
#         print(f"\n🚀 Starting comprehensive multi-timeframe analysis for {symbol}...")
#         print(f"Timeframes: {', '.join(timeframes)}")
#         print("=" * 80)
        
#         for interval in timeframes:
#             print(f"\n📊 Analyzing {symbol} on {interval} timeframe...")
            
#             # Adjust period based on interval
#             if interval in ['1m', '2m']:
#                 period = '1d'
#             elif interval in ['5m', '15m', '30m', '90m']:
#                 period = '5d'
#             elif interval in ['60m', '1h']:
#                 period = '1mo'
#             else:
#                 period = base_period
            
#             # Get data
#             df = self.get_stock_data(symbol, interval=interval, period=period)
            
#             if df is not None:
#                 # Calculate comprehensive indicators
#                 indicators = self.calculate_comprehensive_indicators(df, interval)
                
#                 results[interval] = {
#                     'data': df,
#                     'indicators': indicators,
#                     'current_price': df['Close'].iloc[-1],
#                     'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#                 }
        
#         return results
    
#     def print_comprehensive_analysis(self, symbol, results):
#         """Print detailed comprehensive analysis"""
#         if not results:
#             print("No analysis data available")
#             return
        
#         print(f"\n{'='*100}")
#         print(f"COMPREHENSIVE MULTI-TIMEFRAME ANALYSIS: {symbol}")
#         print(f"{'='*100}")
#         print(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
#         # Summary table
#         print(f"\n📊 TIMEFRAME SUMMARY:")
#         print(f"{'TF':<6} {'Price':<10} {'Bias':<15} {'Strength':<8} {'Conf':<6} {'MACD':<8} {'ADX':<6} {'RSI':<6} {'Fib':<8} {'Wave':<12}")
#         print("-" * 100)
        
#         for timeframe, data in results.items():
#             if 'indicators' in data and 'trading_signals' in data['indicators']:
#                 current_price = data['current_price']
#                 signals = data['indicators']['trading_signals']
#                 basic = data['indicators'].get('basic', {})
#                 fib = data['indicators'].get('fibonacci', {})
#                 wave = data['indicators'].get('elliott_wave', {})
                
#                 # Format data for table
#                 bias = signals['overall_bias'].replace('_', ' ').title()[:14]
#                 strength = f"{signals['strength']:.0f}%"
#                 confidence = signals['confidence'][:4].title()
                
#                 macd_status = "Bull" if basic.get('MACD', {}).get('crossover') == 'bullish' else "Bear" if basic.get('MACD', {}).get('crossover') == 'bearish' else "N/A"
#                 adx_val = f"{basic.get('ADX', {}).get('adx', 0):.0f}" if 'ADX' in basic else "N/A"
#                 rsi_val = f"{basic.get('RSI', {}).get('value', 0):.0f}" if 'RSI' in basic else "N/A"
                
#                 fib_trend = fib.get('trend_direction', 'N/A')[:7] if 'error' not in fib else "N/A"
#                 wave_pattern = f"{wave.get('pattern', 'N/A')[:4]}-{wave.get('confidence', 0):.0f}%" if wave.get('confidence', 0) > 0 else "N/A"
                
#                 print(f"{timeframe:<6} ${current_price:<9.2f} {bias:<15} {strength:<8} {confidence:<6} {macd_status:<8} {adx_val:<6} {rsi_val:<6} {fib_trend:<8} {wave_pattern:<12}")
        
#         # Detailed analysis for each timeframe
#         for timeframe, data in results.items():
#             if 'indicators' in data:
#                 self._print_detailed_timeframe_analysis(symbol, timeframe, data)
        
#         # Multi-timeframe consensus
#         self._print_multi_timeframe_consensus(symbol, results)
    
#     def _print_detailed_timeframe_analysis(self, symbol, timeframe, data):
#         """Print detailed analysis for a specific timeframe with improved error handling"""
#         print(f"\n🔍 DETAILED ANALYSIS - {timeframe.upper()} TIMEFRAME:")
#         print("-" * 60)
        
#         indicators = data['indicators']
#         df = data['data']
#         current_price = data['current_price']
        
#         print(f"Data Points: {len(df)} candles | Current Price: ${current_price:.2f}")
#         print(f"Price Range: ${df['Low'].min():.2f} - ${df['High'].max():.2f}")
        
#         # Basic Indicators
#         if 'basic' in indicators:
#             basic = indicators['basic']
#             print(f"\n📈 Basic Indicators:")
            
#             if 'MACD' in basic:
#                 macd = basic['MACD']
#                 print(f"  MACD {macd.get('settings', '')}: {macd.get('macd', 0):.4f} | Signal: {macd.get('signal', 0):.4f} | Cross: {macd.get('crossover', 'N/A')}")
            
#             if 'ADX' in basic:
#                 adx = basic['ADX']
#                 print(f"  ADX({adx.get('period', 14)}): {adx.get('adx', 0):.2f} | +DI: {adx.get('di_plus', 0):.2f} | -DI: {adx.get('di_minus', 0):.2f} | Strength: {adx.get('trend_strength', 'N/A')}")
            
#             if 'RSI' in basic:
#                 rsi = basic['RSI']
#                 status = "Overbought" if rsi.get('overbought') else "Oversold" if rsi.get('oversold') else "Normal"
#                 print(f"  RSI({rsi.get('period', 14)}): {rsi.get('value', 0):.2f} ({status})")
            
#             if 'OBV' in basic:
#                 obv = basic['OBV']
#                 print(f"  OBV: {obv.get('current', 0):,.0f} | Trend: {obv.get('trend', 'N/A')}")
            
#             if 'Moving_Averages' in basic:
#                 ma = basic['Moving_Averages']
#                 print(f"  SMA20: ${ma.get('sma_20', 0):.2f} | SMA50: ${ma.get('sma_50', 0):.2f} | Trend: {ma.get('trend', 'N/A')}")
            
#             if 'Bollinger_Bands' in basic:
#                 bb = basic['Bollinger_Bands']
                
#                 # Handle error cases
#                 if 'error' in bb:
#                     print(f"  Bollinger Bands: Error - {bb['error']}")
#                     if 'available_columns' in bb:
#                         print(f"    Available columns: {bb['available_columns']}")
#                 else:
#                     # Normal case - display BB data
#                     print(f"  Bollinger Bands: Upper ${bb.get('upper', 0):.2f} | Middle ${bb.get('middle', 0):.2f} | Lower ${bb.get('lower', 0):.2f}")
                    
#                     # Handle position display safely
#                     position = bb.get('position', 'N/A')
#                     if isinstance(position, dict):
#                         # Extract category from position dictionary
#                         position_str = position.get('category', 'N/A')
#                         percentage = position.get('percentage', 'N/A')
#                         distance = position.get('distance_from_middle', 'N/A')
                        
#                         # Format position string
#                         if position_str != 'N/A':
#                             position_display = position_str.replace('_', ' ').title()
#                         else:
#                             position_display = 'N/A'
                        
#                         print(f"  Position: {position_display}")
#                         if percentage != 'N/A':
#                             print(f"    BB Percentage: {percentage}%")
#                         if distance != 'N/A':
#                             print(f"    Distance from Middle: ${distance}")
                            
#                     elif isinstance(position, str):
#                         # Handle string position (legacy format)
#                         position_display = position.replace('_', ' ').title()
#                         print(f"  Position: {position_display}")
#                     else:
#                         print(f"  Position: N/A")
        
#         # Fibonacci Analysis
#         if 'fibonacci' in indicators and 'error' not in indicators['fibonacci']:
#             fib = indicators['fibonacci']
#             print(f"\n🌀 Fibonacci Analysis:")
#             print(f"  Trend Direction: {fib.get('trend_direction', 'N/A').title()}")
#             print(f"  Price Range: ${fib.get('low_price', 0):.2f} - ${fib.get('high_price', 0):.2f}")
#             print(f"  Key Levels:")
            
#             key_levels = fib.get('key_levels', {})
#             for level in ['38.2%', '50%', '61.8%']:
#                 if level in key_levels:
#                     price = key_levels[level]
#                     distance = abs(current_price - price)
#                     pct_away = (distance / current_price) * 100
#                     print(f"    {level}: ${price:.2f} ({pct_away:.1f}% away)")
            
#             if fib.get('nearest_support'):
#                 support = fib['nearest_support']
#                 print(f"  Nearest Support: {support['level']} at ${support['price']:.2f}")
            
#             if fib.get('nearest_resistance'):
#                 resistance = fib['nearest_resistance']
#                 print(f"  Nearest Resistance: {resistance['level']} at ${resistance['price']:.2f}")
        
#         # Elliott Wave Analysis
#         if 'elliott_wave' in indicators:
#             wave = indicators['elliott_wave']
#             print(f"\n🌊 Elliott Wave Analysis:")
#             print(f"  Pattern: {wave.get('pattern', 'N/A').title()}")
#             print(f"  Trend: {wave.get('trend', 'N/A').title()}")
#             print(f"  Confidence: {wave.get('confidence', 0):.0f}%")
            
#             if wave.get('next_expectation'):
#                 print(f"  Next Expectation: {wave['next_expectation']}")
            
#             if wave.get('wave_points'):
#                 print(f"  Wave Points Detected: {len(wave['wave_points'])}")
        
#         # Trading Signals
#         if 'trading_signals' in indicators:
#             signals = indicators['trading_signals']
#             print(f"\n🎯 Trading Signals:")
#             print(f"  Overall Bias: {signals['overall_bias'].replace('_', ' ').title()}")
#             print(f"  Strength: {signals['strength']:.0f}%")
#             print(f"  Confidence: {signals['confidence'].title()}")
            
#             # Signal breakdown
#             breakdown = signals.get('signal_breakdown', {})
#             print(f"  Score Breakdown:")
#             print(f"    Basic Indicators: {breakdown.get('basic_score', 0):+d}")
#             print(f"    Fibonacci: {breakdown.get('fibonacci_score', 0):+d}")
#             print(f"    Elliott Wave: {breakdown.get('elliott_wave_score', 0):+d}")
#             print(f"    Total Score: {breakdown.get('total_score', 0):+d}")
            
#             # Entry signals
#             if signals.get('entry_signals'):
#                 print(f"  📈 Bullish Signals:")
#                 for signal in signals['entry_signals'][:3]:  # Show top 3
#                     print(f"    • {signal}")
            
#             # Exit signals
#             if signals.get('exit_signals'):
#                 print(f"  📉 Bearish Signals:")
#                 for signal in signals['exit_signals'][:3]:  # Show top 3
#                     print(f"    • {signal}")
            
#             # Risk/Reward
#             if signals.get('risk_reward'):
#                 rr = signals['risk_reward']
#                 print(f"  💰 Risk/Reward Analysis:")
#                 print(f"    Ratio: {rr.get('ratio', 0):.2f}:1")
#                 print(f"    Stop Loss: ${rr.get('stop_loss', 0):.2f}")
#                 print(f"    Take Profit: ${rr.get('take_profit', 0):.2f}")
#                 print(f"    Risk Amount: ${rr.get('risk_amount', 0):.2f}")
#                 print(f"    Reward Potential: ${rr.get('reward_potential', 0):.2f}")
    
#     def _print_multi_timeframe_consensus(self, symbol, results):
#         """Print consensus analysis across all timeframes"""
#         print(f"\n🎯 MULTI-TIMEFRAME CONSENSUS:")
#         print("-" * 60)
        
#         # Collect all signals
#         all_biases = []
#         all_strengths = []
#         all_confidences = []
#         timeframe_recommendations = {}
        
#         for timeframe, data in results.items():
#             if 'indicators' in data and 'trading_signals' in data['indicators']:
#                 signals = data['indicators']['trading_signals']
#                 bias = signals['overall_bias']
#                 strength = signals['strength']
#                 confidence = signals['confidence']
                
#                 all_biases.append(bias)
#                 all_strengths.append(strength)
#                 all_confidences.append(confidence)
                
#                 # Convert to recommendation
#                 if 'bullish' in bias:
#                     timeframe_recommendations[timeframe] = 'BUY'
#                 elif 'bearish' in bias:
#                     timeframe_recommendations[timeframe] = 'SELL'
#                 else:
#                     timeframe_recommendations[timeframe] = 'HOLD'
        
#         if not all_biases:
#             print("No consensus data available")
#             return
        
#         # Count recommendations
#         buy_count = sum(1 for rec in timeframe_recommendations.values() if rec == 'BUY')
#         sell_count = sum(1 for rec in timeframe_recommendations.values() if rec == 'SELL')
#         hold_count = sum(1 for rec in timeframe_recommendations.values() if rec == 'HOLD')
#         total_timeframes = len(timeframe_recommendations)
        
#         # Calculate percentages
#         buy_pct = (buy_count / total_timeframes) * 100 if total_timeframes > 0 else 0
#         sell_pct = (sell_count / total_timeframes) * 100 if total_timeframes > 0 else 0
#         hold_pct = (hold_count / total_timeframes) * 100 if total_timeframes > 0 else 0
        
#         # Determine overall consensus
#         if buy_pct >= 60:
#             overall_recommendation = 'BUY'
#             consensus_confidence = 'High' if buy_pct >= 75 else 'Medium'
#         elif sell_pct >= 60:
#             overall_recommendation = 'SELL'
#             consensus_confidence = 'High' if sell_pct >= 75 else 'Medium'
#         else:
#             overall_recommendation = 'HOLD'
#             consensus_confidence = 'Low'
        
#         # Print consensus results
#         print(f"Overall Recommendation: {overall_recommendation}")
#         print(f"Consensus Confidence: {consensus_confidence}")
#         print(f"Agreement: {max(buy_pct, sell_pct, hold_pct):.1f}%")
#         print()
#         print(f"Timeframe Breakdown:")
#         print(f"  BUY signals: {buy_count}/{total_timeframes} ({buy_pct:.1f}%)")
#         print(f"  SELL signals: {sell_count}/{total_timeframes} ({sell_pct:.1f}%)")
#         print(f"  HOLD signals: {hold_count}/{total_timeframes} ({hold_pct:.1f}%)")
#         print()
#         print(f"Individual Timeframe Recommendations:")
#         for tf, rec in timeframe_recommendations.items():
#             print(f"  {tf}: {rec}")
        
#         # Average strength across timeframes
#         avg_strength = np.mean(all_strengths) if all_strengths else 0
#         print(f"\nAverage Signal Strength: {avg_strength:.1f}%")
        
#         # High confidence timeframes
#         high_conf_timeframes = [tf for tf, data in results.items() 
#                                if data.get('indicators', {}).get('trading_signals', {}).get('confidence') == 'high']
#         if high_conf_timeframes:
#             print(f"High Confidence Timeframes: {', '.join(high_conf_timeframes)}")
   
#     def get_timeframe_consensus(self, results):
#         """Get consensus across all timeframes"""
#         if not results:
#             return None
        
#         recommendations = []
#         strengths = []
        
#         for timeframe, data in results.items():
#             if 'timeframe_analysis' in data:
#                 analysis = data['timeframe_analysis']
#                 recommendations.append(analysis['recommendation'])
#                 strengths.append(analysis['strength'])
        
#         # Count recommendations
#         buy_count = recommendations.count('buy')
#         sell_count = recommendations.count('sell')
#         hold_count = recommendations.count('hold')
        
#         total = len(recommendations)
        
#         consensus = {
#             'total_timeframes': total,
#             'buy_signals': buy_count,
#             'sell_signals': sell_count,
#             'hold_signals': hold_count,
#             'buy_percentage': (buy_count / total) * 100 if total > 0 else 0,
#             'sell_percentage': (sell_count / total) * 100 if total > 0 else 0,
#             'hold_percentage': (hold_count / total) * 100 if total > 0 else 0
#         }
        
#         # Determine overall consensus
#         if consensus['buy_percentage'] >= 60:
#             consensus['overall_recommendation'] = 'BUY'
#             consensus['confidence'] = 'High' if consensus['buy_percentage'] >= 75 else 'Medium'
#         elif consensus['sell_percentage'] >= 60:
#             consensus['overall_recommendation'] = 'SELL'
#             consensus['confidence'] = 'High' if consensus['sell_percentage'] >= 75 else 'Medium'
#         else:
#             consensus['overall_recommendation'] = 'HOLD'
#             consensus['confidence'] = 'Low'
        
#         return consensus
    
# class MultiTimeframeAnalyzer:
#     def __init__(self):
#         self.valid_intervals = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
#         self.interval_limits = {
#             # yfinance has different period limits for different intervals
#             '1m': {'max_period': '7d', 'max_days': 7},
#             '2m': {'max_period': '60d', 'max_days': 60},
#             '5m': {'max_period': '60d', 'max_days': 60},
#             '15m': {'max_period': '60d', 'max_days': 60},
#             '30m': {'max_period': '60d', 'max_days': 60},
#             '60m': {'max_period': '730d', 'max_days': 730},
#             '90m': {'max_period': '60d', 'max_days': 60},
#             '1h': {'max_period': '730d', 'max_days': 730},
#             '1d': {'max_period': 'max', 'max_days': 'unlimited'},
#             '5d': {'max_period': 'max', 'max_days': 'unlimited'},
#             '1wk': {'max_period': 'max', 'max_days': 'unlimited'},
#             '1mo': {'max_period': 'max', 'max_days': 'unlimited'},
#             '3mo': {'max_period': 'max', 'max_days': 'unlimited'}
#         }
    
#     def get_stock_data(self, symbol, interval='1d', period='6mo'):
#         """
#         Fetch stock data with specified interval
        
#         Parameters:
#         - symbol: Stock symbol (e.g., 'AAPL')
#         - interval: Time interval ('1m', '5m', '15m', '30m', '1h', '1d', etc.)
#         - period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
#         """
        
#         # Validate interval
#         if interval not in self.valid_intervals:
#             raise ValueError(f"Invalid interval. Must be one of: {self.valid_intervals}")
        
#         # Check period limits for the interval
#         max_period = self.interval_limits[interval]['max_period']
#         if max_period != 'max' and self._period_exceeds_limit(period, max_period):
#             print(f"Warning: Period '{period}' may exceed limit for interval '{interval}'. Using '{max_period}' instead.")
#             period = max_period
        
#         try:
#             ticker = yf.Ticker(symbol)
#             df = ticker.history(period=period, interval=interval)
            
#             if df.empty:
#                 print(f"No data returned for {symbol} with interval {interval} and period {period}")
#                 return None
            
#             print(f"✅ Fetched {len(df)} {interval} candles for {symbol} over {period}")
#             return df
            
#         except Exception as e:
#             print(f"Error fetching data for {symbol}: {e}")
#             return None
    
#     def _period_exceeds_limit(self, period, max_period):
#         """Check if requested period exceeds the limit for the interval"""
#         # Simple check - you could make this more sophisticated
#         period_days = self._period_to_days(period)
#         max_days = self._period_to_days(max_period)
        
#         if period_days and max_days:
#             return period_days > max_days
#         return False
    
#     def _period_to_days(self, period):
#         """Convert period string to approximate days"""
#         if period == 'max':
#             return float('inf')
        
#         period_map = {
#             '1d': 1, '2d': 2, '5d': 5, '7d': 7, '10d': 10,
#             '1mo': 30, '2mo': 60, '3mo': 90, '6mo': 180,
#             '1y': 365, '2y': 730, '5y': 1825, '10y': 3650,
#             '60d': 60, '730d': 730
#         }
        
#         return period_map.get(period, None)
    
#     def analyze_multiple_timeframes(self, symbol, timeframes=None, base_period='6mo'):
#         """
#         Analyze symbol across multiple timeframes
        
#         Parameters:
#         - symbol: Stock symbol
#         - timeframes: List of intervals to analyze
#         - base_period: Base period for daily+ intervals
#         """
        
#         if timeframes is None:
#             timeframes = ['5m', '15m', '1h', '1d']
        
#         results = {}
        
#         for interval in timeframes:
#             print(f"\n📊 Analyzing {symbol} on {interval} timeframe...")
            
#             # Adjust period based on interval
#             if interval in ['1m', '2m']:
#                 period = '1d'  # 1-2 minute data limited to 1 day
#             elif interval in ['5m', '15m', '30m', '90m']:
#                 period = '5d'  # Intraday data limited to shorter periods
#             elif interval in ['60m', '1h']:
#                 period = '1mo'  # Hourly data can go back further
#             else:
#                 period = base_period  # Daily+ can use full period
            
#             # Get data
#             df = self.get_stock_data(symbol, interval=interval, period=period)
            
#             if df is not None:
#                 # Calculate indicators
#                 indicators = self.calculate_indicators(df, interval)
                
#                 results[interval] = {
#                     'data': df,
#                     'indicators': indicators,
#                     'timeframe_analysis': self.analyze_timeframe_signals(indicators, interval)
#                 }
        
#         return results
    
#     def calculate_indicators(self, df, interval):
#         """Calculate indicators adjusted for timeframe"""
#         indicators = {}
        
#         # Adjust indicator periods based on timeframe
#         if interval in ['1m', '2m', '5m']:
#             # Faster settings for short timeframes
#             macd_fast, macd_slow, macd_signal = 6, 13, 4
#             adx_period = 7
#             lookback = 20
#         elif interval in ['15m', '30m']:
#             # Medium settings
#             macd_fast, macd_slow, macd_signal = 8, 17, 6
#             adx_period = 10
#             lookback = 30
#         elif interval in ['1h', '90m']:
#             # Standard settings
#             macd_fast, macd_slow, macd_signal = 12, 26, 9
#             adx_period = 14
#             lookback = 50
#         else:
#             # Daily+ settings
#             macd_fast, macd_slow, macd_signal = 12, 26, 9
#             adx_period = 14
#             lookback = 50
        
#         try:
#             # MACD with adjusted periods
#             macd_data = ta.macd(df['Close'], fast=macd_fast, slow=macd_slow, signal=macd_signal)
#             if macd_data is not None and not macd_data.empty:
#                 macd_col = f'MACD_{macd_fast}_{macd_slow}_{macd_signal}'
#                 signal_col = f'MACDs_{macd_fast}_{macd_slow}_{macd_signal}'
#                 hist_col = f'MACDh_{macd_fast}_{macd_slow}_{macd_signal}'
                
#                 indicators['MACD'] = {
#                     'macd': macd_data[macd_col].iloc[-1] if macd_col in macd_data.columns else None,
#                     'signal': macd_data[signal_col].iloc[-1] if signal_col in macd_data.columns else None,
#                     'histogram': macd_data[hist_col].iloc[-1] if hist_col in macd_data.columns else None,
#                     'settings': f"({macd_fast},{macd_slow},{macd_signal})"
#                 }
                
#                 if indicators['MACD']['macd'] and indicators['MACD']['signal']:
#                     indicators['MACD']['crossover'] = 'bullish' if indicators['MACD']['macd'] > indicators['MACD']['signal'] else 'bearish'
        
#             # ADX with adjusted period
#             adx_data = ta.adx(df['High'], df['Low'], df['Close'], length=adx_period)
#             if adx_data is not None and not adx_data.empty:
#                 indicators['ADX'] = {
#                     'adx': adx_data[f'ADX_{adx_period}'].iloc[-1],
#                     'di_plus': adx_data[f'DMP_{adx_period}'].iloc[-1],
#                     'di_minus': adx_data[f'DMN_{adx_period}'].iloc[-1],
#                     'period': adx_period
#                 }
        
#             # OBV
#             obv_data = ta.obv(df['Close'], df['Volume'])
#             if obv_data is not None and len(obv_data) > 1:
#                 indicators['OBV'] = {
#                     'current': obv_data.iloc[-1],
#                     'previous': obv_data.iloc[-2],
#                     'trend': 'bullish' if obv_data.iloc[-1] > obv_data.iloc[-2] else 'bearish'
#                 }
        
#             # RSI with adjusted period
#             rsi_period = max(7, adx_period)  # Minimum 7 for RSI
#             rsi_data = ta.rsi(df['Close'], length=rsi_period)
#             if rsi_data is not None:
#                 indicators['RSI'] = {
#                     'value': rsi_data.iloc[-1],
#                     'period': rsi_period,
#                     'overbought': rsi_data.iloc[-1] > 70,
#                     'oversold': rsi_data.iloc[-1] < 30
#                 }
        
#             # Moving averages
#             sma_20 = ta.sma(df['Close'], length=min(20, len(df)//2))
#             sma_50 = ta.sma(df['Close'], length=min(50, len(df)//2))
            
#             if sma_20 is not None and sma_50 is not None:
#                 indicators['Moving_Averages'] = {
#                     'sma_20': sma_20.iloc[-1],
#                     'sma_50': sma_50.iloc[-1],
#                     'trend': 'bullish' if sma_20.iloc[-1] > sma_50.iloc[-1] else 'bearish'
#                 }
        
#         except Exception as e:
#             print(f"Error calculating indicators: {e}")
        
#         return indicators
    
#     def analyze_timeframe_signals(self, indicators, interval):
#         """Analyze signals specific to timeframe"""
#         analysis = {
#             'timeframe': interval,
#             'signals': [],
#             'strength': 'neutral',
#             'recommendation': 'hold'
#         }
        
#         bullish_signals = 0
#         bearish_signals = 0
        
#         # MACD analysis
#         if 'MACD' in indicators and indicators['MACD'].get('crossover'):
#             if indicators['MACD']['crossover'] == 'bullish':
#                 bullish_signals += 1
#                 analysis['signals'].append(f"MACD bullish crossover {indicators['MACD']['settings']}")
#             else:
#                 bearish_signals += 1
#                 analysis['signals'].append(f"MACD bearish crossover {indicators['MACD']['settings']}")
        
#         # ADX analysis
#         if 'ADX' in indicators:
#             adx_val = indicators['ADX']['adx']
#             if adx_val > 25:
#                 if indicators['ADX']['di_plus'] > indicators['ADX']['di_minus']:
#                     bullish_signals += 2
#                     analysis['signals'].append(f"Strong bullish trend (ADX: {adx_val:.1f})")
#                 else:
#                     bearish_signals += 2
#                     analysis['signals'].append(f"Strong bearish trend (ADX: {adx_val:.1f})")
        
#         # RSI analysis
#         if 'RSI' in indicators:
#             rsi_val = indicators['RSI']['value']
#             if rsi_val > 70:
#                 bearish_signals += 1
#                 analysis['signals'].append(f"RSI overbought ({rsi_val:.1f})")
#             elif rsi_val < 30:
#                 bullish_signals += 1
#                 analysis['signals'].append(f"RSI oversold ({rsi_val:.1f})")
#          # Determine overall strength and recommendation
#         total_signals = bullish_signals + bearish_signals        
#         if total_signals > 0:
#             bullish_pct = (bullish_signals / total_signals) * 100
            
#             if bullish_pct >= 75:
#                 analysis['strength'] = 'strong_bullish'
#                 analysis['recommendation'] = 'buy'
#             elif bullish_pct >= 60:
#                 analysis['strength'] = 'bullish'
#                 analysis['recommendation'] = 'buy'
#             elif bullish_pct >= 40:
#                 analysis['strength'] = 'neutral'
#                 analysis['recommendation'] = 'hold'
#             elif bullish_pct >= 25:
#                 analysis['strength'] = 'bearish'
#                 analysis['recommendation'] = 'sell'
#             else:
#                 analysis['strength'] = 'strong_bearish'
#                 analysis['recommendation'] = 'sell'
        
#         return analysis
    
#     def print_multi_timeframe_analysis(self, symbol, results):
#         """Print comprehensive multi-timeframe analysis"""
#         print(f"\n{'='*80}")
#         print(f"MULTI-TIMEFRAME ANALYSIS: {symbol}")
#         print(f"{'='*80}")
#         print(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
#         # Summary table
#         print(f"\n📊 TIMEFRAME SUMMARY:")
#         print(f"{'Timeframe':<10} {'Price':<10} {'MACD':<12} {'ADX':<8} {'RSI':<8} {'Strength':<15} {'Rec':<6}")
#         print("-" * 80)
        
#         for timeframe, data in results.items():
#             if 'indicators' in data and 'timeframe_analysis' in data:
#                 indicators = data['indicators']
#                 analysis = data['timeframe_analysis']
                
#                 # Get current price
#                 current_price = data['data']['Close'].iloc[-1]
                
#                 # Format MACD
#                 macd_str = "N/A"
#                 if 'MACD' in indicators and indicators['MACD'].get('crossover'):
#                     cross = indicators['MACD']['crossover']
#                     macd_str = f"{cross[:4]}"  # bull/bear
                
#                 # Format ADX
#                 adx_str = "N/A"
#                 if 'ADX' in indicators:
#                     adx_val = indicators['ADX']['adx']
#                     adx_str = f"{adx_val:.1f}"
                
#                 # Format RSI
#                 rsi_str = "N/A"
#                 if 'RSI' in indicators:
#                     rsi_val = indicators['RSI']['value']
#                     rsi_str = f"{rsi_val:.1f}"
                
#                 # Format strength
#                 strength = analysis['strength'].replace('_', ' ').title()
#                 recommendation = analysis['recommendation'].upper()
                
#                 print(f"{timeframe:<10} ${current_price:<9.2f} {macd_str:<12} {adx_str:<8} {rsi_str:<8} {strength:<15} {recommendation:<6}")
        
#         # Detailed analysis for each timeframe
#         for timeframe, data in results.items():
#             if 'indicators' in data and 'timeframe_analysis' in data:
#                 print(f"\n🔍 DETAILED ANALYSIS - {timeframe.upper()} TIMEFRAME:")
                
#                 indicators = data['indicators']
#                 analysis = data['timeframe_analysis']
#                 df = data['data']
                
#                 print(f"Data Points: {len(df)} candles")
#                 print(f"Price Range: ${df['Low'].min():.2f} - ${df['High'].max():.2f}")
#                 print(f"Current Price: ${df['Close'].iloc[-1]:.2f}")
                
#                 # Detailed indicators
#                 if 'MACD' in indicators:
#                     macd = indicators['MACD']
#                     if macd.get('macd') and macd.get('signal'):
#                         print(f"MACD {macd['settings']}: {macd['macd']:.4f} | Signal: {macd['signal']:.4f} | Cross: {macd.get('crossover', 'N/A')}")
                
#                 if 'ADX' in indicators:
#                     adx = indicators['ADX']
#                     print(f"ADX({adx['period']}): {adx['adx']:.2f} | +DI: {adx['di_plus']:.2f} | -DI: {adx['di_minus']:.2f}")
                
#                 if 'RSI' in indicators:
#                     rsi = indicators['RSI']
#                     status = "Overbought" if rsi['overbought'] else "Oversold" if rsi['oversold'] else "Normal"
#                     print(f"RSI({rsi['period']}): {rsi['value']:.2f} ({status})")
                
#                 if 'OBV' in indicators:
#                     obv = indicators['OBV']
#                     print(f"OBV: {obv['current']:,.0f} | Trend: {obv['trend']}")
                
#                 if 'Moving_Averages' in indicators:
#                     ma = indicators['Moving_Averages']
#                     print(f"SMA20: ${ma['sma_20']:.2f} | SMA50: ${ma['sma_50']:.2f} | Trend: {ma['trend']}")
                
#                 # Signals
#                 if analysis['signals']:
#                     print("Signals:")
#                     for signal in analysis['signals']:
#                         print(f"  • {signal}")
                
#                 print(f"Overall: {analysis['strength'].replace('_', ' ').title()} → {analysis['recommendation'].upper()}")
    
#     def get_timeframe_consensus(self, results):
#         """Get consensus across all timeframes"""
#         if not results:
#             return None
        
#         recommendations = []
#         strengths = []
        
#         for timeframe, data in results.items():
#             if 'timeframe_analysis' in data:
#                 analysis = data['timeframe_analysis']
#                 recommendations.append(analysis['recommendation'])
#                 strengths.append(analysis['strength'])
        
#         # Count recommendations
#         buy_count = recommendations.count('buy')
#         sell_count = recommendations.count('sell')
#         hold_count = recommendations.count('hold')
        
#         total = len(recommendations)
        
#         consensus = {
#             'total_timeframes': total,
#             'buy_signals': buy_count,
#             'sell_signals': sell_count,
#             'hold_signals': hold_count,
#             'buy_percentage': (buy_count / total) * 100 if total > 0 else 0,
#             'sell_percentage': (sell_count / total) * 100 if total > 0 else 0,
#             'hold_percentage': (hold_count / total) * 100 if total > 0 else 0
#         }
        
#         # Determine overall consensus
#         if consensus['buy_percentage'] >= 60:
#             consensus['overall_recommendation'] = 'BUY'
#             consensus['confidence'] = 'High' if consensus['buy_percentage'] >= 75 else 'Medium'
#         elif consensus['sell_percentage'] >= 60:
#             consensus['overall_recommendation'] = 'SELL'
#             consensus['confidence'] = 'High' if consensus['sell_percentage'] >= 75 else 'Medium'
#         else:
#             consensus['overall_recommendation'] = 'HOLD'
#             consensus['confidence'] = 'Low'
        
#         return consensus
    
# class AdvancedFibonacciAnalyzer:
#     def __init__(self):
#         self.fib_levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.414, 1.618]
#         self.fib_names = ['0%', '23.6%', '38.2%', '50%', '61.8%', '78.6%', '100%', '127.2%', '141.4%', '161.8%']
    
#     def find_swing_points(self, df, lookback=10):
#         """Find swing highs and lows using peak detection"""
#         highs = df['High'].values
#         lows = df['Low'].values
        
#         # Find peaks (swing highs)
#         high_peaks, _ = find_peaks(highs, distance=lookback)
#         # Find troughs (swing lows) - invert the data
#         low_peaks, _ = find_peaks(-lows, distance=lookback)
        
#         swing_highs = [(df.index[i], highs[i]) for i in high_peaks]
#         swing_lows = [(df.index[i], lows[i]) for i in low_peaks]
        
#         return swing_highs, swing_lows
    
#     def calculate_fibonacci_retracements(self, df, period_days=60, auto_detect=True):
#         """Calculate Fibonacci retracements with multiple methods"""
#         recent_data = df.tail(period_days)
        
#         if auto_detect:
#             # Method 1: Auto-detect swing points
#             swing_highs, swing_lows = self.find_swing_points(recent_data, lookback=5)
            
#             if swing_highs and swing_lows:
#                 # Get the most recent significant high and low
#                 recent_high = max(swing_highs, key=lambda x: x[1])
#                 recent_low = min(swing_lows, key=lambda x: x[1])
                
#                 high_price = recent_high[1]
#                 low_price = recent_low[1]
#                 high_date = recent_high[0]
#                 low_date = recent_low[0]
#             else:
#                 # Fallback to simple high/low
#                 high_price = recent_data['High'].max()
#                 low_price = recent_data['Low'].min()
#                 high_date = recent_data['High'].idxmax()
#                 low_date = recent_data['Low'].idxmin()
#         else:
#             # Method 2: Simple high/low approach
#             high_price = recent_data['High'].max()
#             low_price = recent_data['Low'].min()
#             high_date = recent_data['High'].idxmax()
#             low_date = recent_data['Low'].idxmin()
        
#         # Determine if we're in uptrend or downtrend
#         trend_direction = "uptrend" if high_date > low_date else "downtrend"
        
#         # Calculate Fibonacci levels
#         price_range = high_price - low_price
#         current_price = df['Close'].iloc[-1]
        
#         fib_levels = {}
#         for i, level in enumerate(self.fib_levels):
#             if trend_direction == "uptrend":
#                 # Retracement from high
#                 fib_price = high_price - (price_range * level)
#             else:
#                 # Extension from low
#                 fib_price = low_price + (price_range * level)
            
#             fib_levels[self.fib_names[i]] = {
#                 'price': fib_price,
#                 'distance_from_current': abs(current_price - fib_price),
#                 'percentage_from_current': ((fib_price - current_price) / current_price) * 100
#             }
        
#         # Find nearest support and resistance levels
#         nearest_support = None
#         nearest_resistance = None
        
#         for name, data in fib_levels.items():
#             price = data['price']
#             if price < current_price:
#                 if nearest_support is None or price > nearest_support['price']:
#                     nearest_support = {'level': name, 'price': price}
#             elif price > current_price:
#                 if nearest_resistance is None or price < nearest_resistance['price']:
#                     nearest_resistance = {'level': name, 'price': price}
        
#         return {
#             'trend_direction': trend_direction,
#             'high_price': high_price,
#             'low_price': low_price,
#             'high_date': high_date,
#             'low_date': low_date,
#             'current_price': current_price,
#             'price_range': price_range,
#             'fib_levels': fib_levels,
#             'nearest_support': nearest_support,
#             'nearest_resistance': nearest_resistance,
#             'key_levels': {
#                 '38.2%': fib_levels['38.2%']['price'],
#                 '50%': fib_levels['50%']['price'],
#                 '61.8%': fib_levels['61.8%']['price']
#             }
#         }

# class ElliottWaveAnalyzer:
#     def __init__(self):
#         self.wave_patterns = {
#             'impulse': [1, 2, 3, 4, 5],
#             'corrective': ['A', 'B', 'C']
#         }
    
#     def detect_wave_structure(self, df, lookback=50):
#         """Detect potential Elliott Wave structure"""
#         recent_data = df.tail(lookback)
        
#         # Find significant swing points
#         highs = recent_data['High'].values
#         lows = recent_data['Low'].values
        
#         # Find peaks and troughs
#         high_peaks, _ = find_peaks(highs, distance=5, prominence=np.std(highs)*0.5)
#         low_peaks, _ = find_peaks(-lows, distance=5, prominence=np.std(lows)*0.5)
        
#         # Combine and sort all turning points
#         all_points = []
#         for i in high_peaks:
#             all_points.append((recent_data.index[i], highs[i], 'high'))
#         for i in low_peaks:
#             all_points.append((recent_data.index[i], lows[i], 'low'))
        
#         # Sort by date
#         all_points.sort(key=lambda x: x[0])
        
#         if len(all_points) < 5:
#             return {
#                 'wave_count': len(all_points),
#                 'pattern': 'insufficient_data',
#                 'confidence': 0,
#                 'description': 'Need at least 5 turning points for wave analysis'
#             }
        
#         # Analyze the pattern
#         wave_analysis = self.analyze_wave_pattern(all_points, recent_data)
        
#         return wave_analysis
    
#     def analyze_wave_pattern(self, points, df):
#         """Analyze the wave pattern from turning points"""
#         if len(points) < 5:
#             return {'pattern': 'insufficient_data', 'confidence': 0}
        
#         # Take the last 5 significant points
#         last_5_points = points[-5:]
        
#         # Determine overall trend
#         first_price = last_5_points[0][1]
#         last_price = last_5_points[-1][1]
#         overall_trend = "bullish" if last_price > first_price else "bearish"
        
#         # Check for impulse wave characteristics
#         impulse_score = self.check_impulse_pattern(last_5_points)
#         corrective_score = self.check_corrective_pattern(last_5_points)
        
#         # Determine most likely pattern
#         if impulse_score > corrective_score:
#             pattern_type = "impulse"
#             confidence = impulse_score
#             next_expectation = self.predict_next_impulse_move(last_5_points, overall_trend)
#         else:
#             pattern_type = "corrective"
#             confidence = corrective_score
#             next_expectation = self.predict_next_corrective_move(last_5_points, overall_trend)
        
#         return {
#             'pattern': pattern_type,
#             'trend': overall_trend,
#             'confidence': confidence,
#             'wave_points': last_5_points,
#             'next_expectation': next_expectation,
#             'impulse_score': impulse_score,
#             'corrective_score': corrective_score
#         }
    
#     def check_impulse_pattern(self, points):
#         """Check if points match impulse wave pattern (1-2-3-4-5)"""
#         if len(points) != 5:
#             return 0
        
#         score = 0
        
#         # Wave 3 should be the longest (often)
#         wave_lengths = []
#         for i in range(1, len(points)):
#             wave_lengths.append(abs(points[i][1] - points[i-1][1]))
        
#         if len(wave_lengths) >= 3:
#             # Wave 3 (index 2) should be strong
#             if wave_lengths[2] == max(wave_lengths):
#                 score += 30
#             elif wave_lengths[2] > np.mean(wave_lengths):
#                 score += 15
        
#         # Wave 2 shouldn't retrace more than 100% of wave 1
#         if len(points) >= 3:
#             wave1_size = abs(points[1][1] - points[0][1])
#             wave2_retrace = abs(points[2][1] - points[1][1])
#             if wave2_retrace < wave1_size:
#                 score += 20
        
#         # Wave 4 shouldn't overlap with wave 1 territory
#         if len(points) >= 5:
#             wave1_end = points[1][1]
#             wave4_end = points[4][1]
#             if (points[0][1] < points[1][1] and wave4_end > points[0][1]) or \
#                (points[0][1] > points[1][1] and wave4_end < points[0][1]):
#                 score += 25
        
#         return min(score, 100)
    
#     def check_corrective_pattern(self, points):
#         """Check if points match corrective wave pattern (A-B-C)"""
#         if len(points) < 3:
#             return 0
        
#         score = 0
        
#         # Take last 3 points for A-B-C analysis
#         abc_points = points[-3:]
        
#         # A and C waves should be roughly similar in size
#         a_wave = abs(abc_points[1][1] - abc_points[0][1])
#         c_wave = abs(abc_points[2][1] - abc_points[1][1])
        
#         if a_wave > 0 and c_wave > 0:
#             ratio = min(a_wave, c_wave) / max(a_wave, c_wave)
#             if ratio > 0.8:  # Very similar
#                 score += 40
#             elif ratio > 0.6:  # Reasonably similar
#                 score += 25
        
#         # B wave should be a counter-trend move
#         if len(abc_points) >= 3:
#             a_direction = 1 if abc_points[1][1] > abc_points[0][1] else -1
#             b_direction = 1 if abc_points[2][1] > abc_points[1][1] else -1
            
#             if a_direction != b_direction:
#                 score += 30
        
#         return min(score, 100)
    
#     def predict_next_impulse_move(self, points, trend):
#         """Predict next move in impulse pattern"""
#         if len(points) < 5:
#             return "Need more data"
        
#         # Assume we're at the end of wave 5
#         wave_5_end = points[-1][1]
#         wave_3_high = max(points[2][1], points[4][1]) if trend == "bullish" else min(points[2][1], points[4][1])
        
#         if trend == "bullish":
#             return f"Expect correction to {wave_3_high * 0.618:.2f} - {wave_3_high * 0.382:.2f} range"
#         else:
#             return f"Expect bounce to {wave_3_high * 1.382:.2f} - {wave_3_high * 1.618:.2f} range"
    
#     def predict_next_corrective_move(self, points, trend):
#         """Predict next move in corrective pattern"""
#         return "Corrective pattern - expect continuation of main trend after completion"

# class ComprehensiveTechnicalAnalyzer:
#     def __init__(self):
#         self.fib_analyzer = AdvancedFibonacciAnalyzer()
#         self.wave_analyzer = ElliottWaveAnalyzer()
    
#     def analyze_symbol(self, symbol, period="6mo"):
#         """Complete technical analysis including Fibonacci and Elliott Wave"""
#         # Get data
#         ticker = yf.Ticker(symbol)
#         df = ticker.history(period=period)
        
#         if df.empty:
#             return None
        
#         # Basic indicators (your existing code)
#         indicators = self.get_basic_indicators(df)
        
#         # Advanced Fibonacci analysis
#         fib_analysis = self.fib_analyzer.calculate_fibonacci_retracements(df, period_days=60)
        
#         # Elliott Wave analysis
#         wave_analysis = self.wave_analyzer.detect_wave_structure(df, lookback=50)
        
#         # Combine everything
#         return {
#             'symbol': symbol,
#             'timestamp': datetime.now().isoformat(),
#             'price': df['Close'].iloc[-1],
#             'basic_indicators': indicators,
#             'fibonacci_analysis': fib_analysis,
#             'elliott_wave_analysis': wave_analysis,
#             'trading_signals': self.generate_trading_signals(indicators, fib_analysis, wave_analysis)
#         }
    
#     def get_basic_indicators(self, df):
#         """Your existing indicator calculations"""
#         indicators = {}
        
#         # MACD
#         macd_data = ta.macd(df['Close'], fast=12, slow=26, signal=9)
#         indicators['MACD'] = {
#             'macd': macd_data['MACD_12_26_9'].iloc[-1],
#             'signal': macd_data['MACDs_12_26_9'].iloc[-1],
#             'histogram': macd_data['MACDh_12_26_9'].iloc[-1],
#             'crossover': 'bullish' if macd_data['MACD_12_26_9'].iloc[-1] > macd_data['MACDs_12_26_9'].iloc[-1] else 'bearish'
#         }
        
#         # ADX and DI for multiple periods
#         for period in [14, 20]:
#             adx_data = ta.adx(df['High'], df['Low'], df['Close'], length=period)
#             indicators[f'ADX_{period}'] = {
#                 'adx': adx_data[f'ADX_{period}'].iloc[-1],
#                 'di_plus': adx_data[f'DMP_{period}'].iloc[-1],
#                 'di_minus': adx_data[f'DMN_{period}'].iloc[-1],
#                 'trend_strength': 'strong' if adx_data[f'ADX_{period}'].iloc[-1] > 25 else 'weak'
#             }
        
#         # OBV
#         obv_data = ta.obv(df['Close'], df['Volume'])
#         indicators['OBV'] = {
#             'current': obv_data.iloc[-1],
#             'previous': obv_data.iloc[-2],
#             'trend': 'bullish' if obv_data.iloc[-1] > obv_data.iloc[-2] else 'bearish'
#         }
        
#         # Wavetrend
#         try:
#             wt_data = ta.wt(df['High'], df['Low'], df['Close'])
#             if wt_data is not None and not wt_data.empty:
#                 wt1 = wt_data.iloc[:, 0].iloc[-1]
#                 wt2 = wt_data.iloc[:, 1].iloc[-1]
#                 indicators['Wavetrend'] = {
#                     'wt1': wt1,
#                     'wt2': wt2,
#                     'cross': 'bullish' if wt1 > wt2 else 'bearish',
#                     'overbought': wt1 > 60,
#                     'oversold': wt1 < -60
#                 }
#         except:
#             indicators['Wavetrend'] = {'error': 'Not available'}
        
#         return indicators
    
#     def generate_trading_signals(self, indicators, fib_analysis, wave_analysis):
#         """Generate comprehensive trading signals"""
#         signals = {
#             'overall_bias': 'neutral',
#             'strength': 0,
#             'entry_signals': [],
#             'exit_signals': [],
#             'key_levels': {},
#             'risk_reward': {}
#         }
        
#         # Scoring system
#         bullish_score = 0
#         bearish_score = 0
        
#         # MACD signals
#         if indicators['MACD']['crossover'] == 'bullish':
#             bullish_score += 2
#             signals['entry_signals'].append('MACD bullish crossover')
#         else:
#             bearish_score += 2
#             signals['exit_signals'].append('MACD bearish crossover')
        
#         # ADX trend strength
#         if indicators['ADX_14']['trend_strength'] == 'strong':
#             if indicators['ADX_14']['di_plus'] > indicators['ADX_14']['di_minus']:
#                 bullish_score += 3
#                 signals['entry_signals'].append('Strong bullish trend (ADX)')
#             else:
#                 bearish_score += 3
#                 signals['exit_signals'].append('Strong bearish trend (ADX)')
        
#         # OBV confirmation
#         if indicators['OBV']['trend'] == 'bullish':
#             bullish_score += 1
#             signals['entry_signals'].append('Volume supporting uptrend')
#         else:
#             bearish_score += 1
#             signals['exit_signals'].append('Volume supporting downtrend')
        
#         # Fibonacci level analysis
#         current_price = fib_analysis['current_price']
        
#         # Check if price is near key Fibonacci levels
#         for level_name, level_price in fib_analysis['key_levels'].items():
#             distance_pct = abs((current_price - level_price) / current_price) * 100
            
#             if distance_pct < 2:  # Within 2% of Fibonacci level
#                 if current_price > level_price:
#                     signals['entry_signals'].append(f'Price above {level_name} Fibonacci support')
#                     bullish_score += 1
#                 else:
#                     signals['exit_signals'].append(f'Price below {level_name} Fibonacci resistance')
#                     bearish_score += 1
        
#         # Elliott Wave signals
#         if wave_analysis['pattern'] == 'impulse':
#             if wave_analysis['trend'] == 'bullish':
#                 bullish_score += 2
#                 signals['entry_signals'].append('Bullish impulse wave pattern detected')
#             else:
#                 bearish_score += 2
#                 signals['exit_signals'].append('Bearish impulse wave pattern detected')
        
#         # Determine overall bias
#         total_score = bullish_score + bearish_score
#         if total_score > 0:
#             bullish_pct = (bullish_score / total_score) * 100
            
#             if bullish_pct >= 70:
#                 signals['overall_bias'] = 'bullish'
#                 signals['strength'] = bullish_pct
#             elif bullish_pct <= 30:
#                 signals['overall_bias'] = 'bearish'
#                 signals['strength'] = 100 - bullish_pct
#             else:
#                 signals['overall_bias'] = 'neutral'
#                 signals['strength'] = 50
        
#         # Set key levels
#         signals['key_levels'] = {
#             'support': fib_analysis['nearest_support']['price'] if fib_analysis['nearest_support'] else None,
#             'resistance': fib_analysis['nearest_resistance']['price'] if fib_analysis['nearest_resistance'] else None,
#             'fibonacci_382': fib_analysis['key_levels']['38.2%'],
#             'fibonacci_618': fib_analysis['key_levels']['61.8%']
#         }
        
#         # Calculate risk/reward
#         if signals['key_levels']['support'] and signals['key_levels']['resistance']:
#             risk = abs(current_price - signals['key_levels']['support'])
#             reward = abs(signals['key_levels']['resistance'] - current_price)
#             if risk > 0:
#                 signals['risk_reward']['ratio'] = reward / risk
#                 signals['risk_reward']['risk_amount'] = risk
#                 signals['risk_reward']['reward_potential'] = reward
        
#         return signals
    
#     def print_comprehensive_analysis(self, analysis):
#         """Print detailed analysis results"""
#         if not analysis:
#             print("No analysis data available")
#             return
        
#         symbol = analysis['symbol']
#         price = analysis['price']
        
#         print(f"\n{'='*60}")
#         print(f"COMPREHENSIVE ANALYSIS: {symbol}")
#         print(f"{'='*60}")
#         print(f"Current Price: ${price:.2f}")
#         print(f"Analysis Time: {analysis['timestamp']}")
        
#         # Basic Indicators
#         print(f"\n📊 BASIC INDICATORS:")
#         indicators = analysis['basic_indicators']
        
#         macd = indicators['MACD']
#         print(f"MACD: {macd['macd']:.4f} | Signal: {macd['signal']:.4f} | Cross: {macd['crossover']}")
        
#         adx_14 = indicators['ADX_14']
#         print(f"ADX(14): {adx_14['adx']:.2f} | +DI: {adx_14['di_plus']:.2f} | -DI: {adx_14['di_minus']:.2f}")
        
#         obv = indicators['OBV']
#         print(f"OBV: {obv['current']:,.0f} | Trend: {obv['trend']}")
        
#         # Fibonacci Analysis
#         print(f"\n📈 FIBONACCI ANALYSIS:")
#         fib = analysis['fibonacci_analysis']
#         print(f"Trend Direction: {fib['trend_direction']}")
#         print(f"Price Range: ${fib['low_price']:.2f} - ${fib['high_price']:.2f}")
#         print(f"Key Levels:")
#         for level, price_data in fib['key_levels'].items():
#             distance = price_data['percentage_from_current']
#             print(f"  {level}: ${price_data:.2f} ({distance:+.1f}%)")
        
#         if fib['nearest_support']:
#             print(f"Nearest Support: {fib['nearest_support']['level']} at ${fib['nearest_support']['price']:.2f}")
#         if fib['nearest_resistance']:
#             print(f"Nearest Resistance: {fib['nearest_resistance']['level']} at ${fib['nearest_resistance']['price']:.2f}")
        
#         # Elliott Wave Analysis
#         print(f"\n🌊 ELLIOTT WAVE ANALYSIS:")
#         wave = analysis['elliott_wave_analysis']
#         print(f"Pattern: {wave['pattern']} | Trend: {wave.get('trend', 'N/A')}")
#         print(f"Confidence: {wave['confidence']:.0f}%")
#         if 'next_expectation' in wave:
#             print(f"Next Expected Move: {wave['next_expectation']}")
        
#         # Trading Signals
#         print(f"\n🎯 TRADING SIGNALS:")
#         signals = analysis['trading_signals']
#         print(f"Overall Bias: {signals['overall_bias'].upper()} (Strength: {signals['strength']:.0f}%)")
        
#         if signals['entry_signals']:
#             print("Entry Signals:")
#             for signal in signals['entry_signals']:
#                 print(f"  ✅ {signal}")
        
#         if signals['exit_signals']:
#             print("Exit Signals:")
#             for signal in signals['exit_signals']:
#                 print(f"  ❌ {signal}")
        
#         # Key Levels
#         print(f"\n🎯 KEY LEVELS:")
#         levels = signals['key_levels']
#         if levels['resistance']:
#             print(f"Resistance: ${levels['resistance']:.2f}")
#         print(f"Current: ${price:.2f}")
#         if levels['support']:
#             print(f"Support: ${levels['support']:.2f}")
        
#         # Risk/Reward
#         if 'ratio' in signals['risk_reward']:
#             rr = signals['risk_reward']
#             print(f"\n💰 RISK/REWARD:")
#             print(f"Risk: ${rr['risk_amount']:.2f}")
#             print(f"Reward: ${rr['reward_potential']:.2f}")
#             print(f"R/R Ratio: 1:{rr['ratio']:.2f}")

# def quick_analysis(symbol, timeframes=['5m', '15m', '1h', '1d']):
#     """Quick multi-timeframe analysis"""
#     analyzer = MultiTimeframeAnalyzer()
    
#     print(f"🚀 Starting multi-timeframe analysis for {symbol}...")
#     results = analyzer.analyze_multiple_timeframes(symbol, timeframes)
    
#     if results:
#         analyzer.print_multi_timeframe_analysis(symbol, results)
        
#         # Get consensus
#         consensus = analyzer.get_timeframe_consensus(results)
#         if consensus:
#             print(f"\n🎯 CONSENSUS ACROSS ALL TIMEFRAMES:")
#             print(f"Overall Recommendation: {consensus['overall_recommendation']} (Confidence: {consensus['confidence']})")
#             print(f"Buy Signals: {consensus['buy_signals']}/{consensus['total_timeframes']} ({consensus['buy_percentage']:.1f}%)")
#             print(f"Sell Signals: {consensus['sell_signals']}/{consensus['total_timeframes']} ({consensus['sell_percentage']:.1f}%)")
#             print(f"Hold Signals: {consensus['hold_signals']}/{consensus['total_timeframes']} ({consensus['hold_percentage']:.1f}%)")
    
#     return results

# def compare_intervals_demo():
#     """Demonstrate different intervals and their data limits"""
#     analyzer = MultiTimeframeAnalyzer()
    
#     print("📋 YFINANCE INTERVAL LIMITS:")
#     print(f"{'Interval':<10} {'Description':<15} {'Max Period':<12} {'Typical Use'}")
#     print("-" * 70)
    
#     interval_descriptions = {
#         '1m': ('1 minute', '7d', 'Scalping'),
#         '5m': ('5 minutes', '60d', 'Day trading'),
#         '15m': ('15 minutes', '60d', 'Swing entry'),
#         '1h': ('1 hour', '730d', 'Swing trading'),
#         '1d': ('1 day', 'max', 'Position trading'),
#         '1wk': ('1 week', 'max', 'Long-term'),
#     }
    
#     for interval, (desc, max_per, use) in interval_descriptions.items():
#         print(f"{interval:<10} {desc:<15} {max_per:<12} {use}")

# def test_different_intervals(symbol='TQQQ'):
#     """Test fetching data with different intervals"""
#     analyzer = MultiTimeframeAnalyzer()
    
#     test_intervals = [
#         ('1m', '1d'),    # 1-minute for 1 day
#         ('5m', '5d'),    # 5-minute for 5 days  
#         ('15m', '1mo'),  # 15-minute for 1 month
#         ('1h', '3mo'),   # 1-hour for 3 months
#         ('1d', '1y'),    # Daily for 1 year
#     ]
    
#     print(f"🧪 TESTING DIFFERENT INTERVALS FOR {symbol}:")
#     print("-" * 60)
    
#     for interval, period in test_intervals:
#         print(f"\nTesting {interval} interval with {period} period...")
#         df = analyzer.get_stock_data(symbol, interval=interval, period=period)
        
#         if df is not None:
#             print(f"✅ Success: {len(df)} candles from {df.index[0]} to {df.index[-1]}")
#             print(f"   Price range: ${df['Low'].min():.2f} - ${df['High'].max():.2f}")
#         else:
#             print("❌ Failed to fetch data")

# # Enhanced Usage Functions
# def comprehensive_quick_analysis(symbol, timeframes=['5m', '15m', '1h', '1d']):
#     """Quick comprehensive multi-timeframe analysis"""
#     analyzer = ComprehensiveMultiTimeframeAnalyzer()
    
#     print(f"🚀 Starting comprehensive analysis for {symbol}...")
#     results = analyzer.analyze_comprehensive_multi_timeframe(symbol, timeframes)
    
#     if results:
#         analyzer.print_comprehensive_analysis(symbol, results)
    
#     return results

# def compare_symbols_multi_timeframe(symbols, timeframes=['15m', '1h', '1d']):
#     """Compare multiple symbols across timeframes"""
#     analyzer = ComprehensiveMultiTimeframeAnalyzer()
    
#     print(f"📊 MULTI-SYMBOL COMPARISON")
#     print(f"Symbols: {', '.join(symbols)}")
#     print(f"Timeframes: {', '.join(timeframes)}")
#     print("=" * 80)
    
#     all_results = {}
    
#     for symbol in symbols:
#         print(f"\nAnalyzing {symbol}...")
#         results = analyzer.analyze_comprehensive_multi_timeframe(symbol, timeframes)
#         all_results[symbol] = results
    
#     # Print comparison summary
#     print(f"\n📋 COMPARISON SUMMARY:")
#     print(f"{'Symbol':<8} {'TF':<6} {'Price':<10} {'Bias':<15} {'Strength':<8} {'Confidence':<10}")
#     print("-" * 70)
    
#     for symbol, results in all_results.items():
#         for timeframe, data in results.items():
#             if 'indicators' in data and 'trading_signals' in data['indicators']:
#                 signals = data['indicators']['trading_signals']
#                 price = data['current_price']
#                 bias = signals['overall_bias'].replace('_', ' ').title()
#                 strength = f"{signals['strength']:.0f}%"
#                 confidence = signals['confidence'].title()
                
#                 print(f"{symbol:<8} {timeframe:<6} ${price:<9.2f} {bias:<15} {strength:<8} {confidence:<10}")
    
#     return all_results

# def demo_comprehensive_system():
#     """Demonstrate the comprehensive system capabilities"""
#     print("🎯 COMPREHENSIVE MULTI-TIMEFRAME TECHNICAL ANALYSIS SYSTEM")
#     print("=" * 80)
#     print("Features:")
#     print("✅ Multi-timeframe analysis (1m to 1mo)")
#     print("✅ Basic technical indicators (MACD, RSI, ADX, OBV, MA, BB)")
#     print("✅ Advanced Fibonacci retracement analysis")
#     print("✅ Elliott Wave pattern detection")
#     print("✅ Comprehensive signal scoring system")
#     print("✅ Risk/reward calculations")
#     print("✅ Multi-timeframe consensus building")
#     print("=" * 80)
    
#     # Demo with popular stocks
#     symbols = ['ASML']
#     timeframes = ['2m','5m', '15m','30m','1h','90m','1d', '5d', '1wk']
    
#     for symbol in symbols:
#         print(f"\n{'='*50}")
#         print(f"DEMO ANALYSIS: {symbol}")
#         print(f"{'='*50}")
        
#         comprehensive_quick_analysis(symbol, timeframes)
        
#         print("\n" + "="*50)

# # Main execution
# def main():
#     """Main function to run comprehensive analysis"""
#     demo_comprehensive_system()
#     # test_different_intervals()


# if __name__ == "__main__":
#     main()
# Code Generated by Sidekick is for learning and experimentation purposes only.
import alpaca_trade_api as tradeapi
import pandas_ta as ta
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import warnings
import os
warnings.filterwarnings('ignore')

class ComprehensiveIndicatorFetcher:
    def __init__(self, api_key=None, secret_key=None, base_url='https://paper-api.alpaca.markets'):
        """Initialize with Alpaca API credentials"""
        api_key = "PKYJLOK4LZBY56NZKXZLNSG665"
        secret_key = "4VVHMnrYEqVv4Jd1oMZMow15DrRVn5p8VD7eEK6TjYZ1"
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url
        self.api = None
        self.data_cache = {}
        
        if api_key and secret_key:
            self.api = tradeapi.REST(api_key, secret_key, base_url, api_version='v2')
    
    def set_credentials(self, api_key, secret_key, base_url='https://paper-api.alpaca.markets'):
        """Set Alpaca API credentials"""
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url
        self.api = tradeapi.REST(api_key, secret_key, base_url, api_version='v2')
    
    def _convert_period_to_dates(self, period):
        """Convert yfinance period format to start/end dates for Alpaca"""
        end_date = datetime.now()
        
        period_map = {
            '1d': 1, '2d': 2, '5d': 5, '7d': 7, '10d': 10,
            '1mo': 30, '2mo': 60, '3mo': 90, '6mo': 180,
            '1y': 365, '2y': 730, '5y': 1825, '10y': 3650,
            'ytd': (datetime.now() - datetime(datetime.now().year, 1, 1)).days,
            'max': 3650  # Default to 10 years for max
        }
        
        days = period_map.get(period, 180)  # Default to 6 months
        start_date = end_date - timedelta(days=days)
        
        return start_date, end_date
    
    def get_stock_data(self, symbol: str, period: str = "6mo"):
        """Fetch stock data using Alpaca API (maintains yfinance interface)"""
        if not self.api:
            raise ValueError("Alpaca API credentials not set. Use set_credentials() method.")
        
        if symbol not in self.data_cache:
            try:
                start_date, end_date = self._convert_period_to_dates(period)
                
                # Get bars from Alpaca
                bars = self.api.get_bars(
                    symbol,
                    tradeapi.TimeFrame.Day,
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    adjustment='raw'
                ).df
                
                if bars.empty:
                    print(f"No data returned for {symbol}")
                    return pd.DataFrame()
                
                # Convert to yfinance-like format
                df = pd.DataFrame()
                df['Open'] = bars['open']
                df['High'] = bars['high']
                df['Low'] = bars['low']
                df['Close'] = bars['close']
                df['Volume'] = bars['volume']
                df.index = bars.index
                
                self.data_cache[symbol] = df
                
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
                return pd.DataFrame()
        
        return self.data_cache[symbol]
    
    def get_all_indicators(self, symbol: str, period: str = "6mo"):
        """Get all your requested indicators using pandas-ta"""
        df = self.get_stock_data(symbol, period)
        
        if df.empty:
            return None
        
        # Add all indicators to the dataframe
        indicators = {}
        
        # 1. MACD (12, 26, 9) - Ready-made
        macd_data = ta.macd(df['Close'], fast=12, slow=26, signal=9)
        indicators['MACD'] = {
            'macd': macd_data['MACD_12_26_9'].iloc[-1],
            'signal': macd_data['MACDs_12_26_9'].iloc[-1],
            'histogram': macd_data['MACDh_12_26_9'].iloc[-1],
            'crossover': 'bullish' if macd_data['MACD_12_26_9'].iloc[-1] > macd_data['MACDs_12_26_9'].iloc[-1] else 'bearish'
        }
        
        # 2. ADX and DI (14-20 periods) - Ready-made
        for period in [14, 20]:
            adx_data = ta.adx(df['High'], df['Low'], df['Close'], length=period)
            indicators[f'ADX_{period}'] = {
                'adx': adx_data[f'ADX_{period}'].iloc[-1],
                'di_plus': adx_data[f'DMP_{period}'].iloc[-1],
                'di_minus': adx_data[f'DMN_{period}'].iloc[-1],
                'trend_strength': 'strong' if adx_data[f'ADX_{period}'].iloc[-1] > 25 else 'weak'
            }
        
        # 3. OBV (On-Balance Volume) - Ready-made
        obv_data = ta.obv(df['Close'], df['Volume'])
        indicators['OBV'] = {
            'current': obv_data.iloc[-1],
            'previous': obv_data.iloc[-2],
            'trend': 'bullish' if obv_data.iloc[-1] > obv_data.iloc[-2] else 'bearish'
        }
        
        # 4. Wavetrend - Ready-made in pandas-ta
        try:
            wt_data = ta.wt(df['High'], df['Low'], df['Close'])
            if wt_data is not None and not wt_data.empty:
                wt1 = wt_data.iloc[:, 0].iloc[-1]  # WT1
                wt2 = wt_data.iloc[:, 1].iloc[-1]  # WT2
                indicators['Wavetrend'] = {
                    'wt1': wt1,
                    'wt2': wt2,
                    'cross': 'bullish' if wt1 > wt2 else 'bearish',
                    'overbought': wt1 > 60,
                    'oversold': wt1 < -60
                }
        except:
            indicators['Wavetrend'] = {'error': 'Not available'}
        
        # 5. Fibonacci Retracements - Semi-automated
        indicators['Fibonacci'] = self.calculate_fibonacci_levels(df)
        
        # 6. Elliott Wave - Pattern detection (basic)
        indicators['Elliott_Wave'] = self.detect_elliott_wave_pattern(df)
        
        return {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'price': df['Close'].iloc[-1],
            'indicators': indicators
        }
    
    def calculate_fibonacci_levels(self, df):
        """Calculate Fibonacci retracement levels"""
        # Get recent high and low (last 50 periods)
        recent_data = df.tail(50)
        high = recent_data['High'].max()
        low = recent_data['Low'].min()
        
        diff = high - low
        
        return {
            'high': high,
            'low': low,
            'levels': {
                '0%': high,
                '23.6%': high - (diff * 0.236),
                '38.2%': high - (diff * 0.382),
                '50%': high - (diff * 0.5),
                '61.8%': high - (diff * 0.618),
                '78.6%': high - (diff * 0.786),
                '100%': low
            }
        }
    
    def detect_elliott_wave_pattern(self, df):
        """Basic Elliott Wave pattern detection"""
        # This is a simplified version - true Elliott Wave is very complex
        recent_data = df.tail(20)
        highs = recent_data['High'].values
        lows = recent_data['Low'].values
        
        # Simple trend detection
        if len(highs) >= 5:
            trend = "bullish" if highs[-1] > highs[0] else "bearish"
            return {
                'trend': trend,
                'note': 'Simplified pattern detection - not full Elliott Wave analysis'
            }
        
        return {'error': 'Insufficient data for pattern detection'}

class ComprehensiveMultiTimeframeAnalyzer:
    def __init__(self, api_key=None, secret_key=None, base_url='https://paper-api.alpaca.markets'):
        """Initialize with Alpaca API credentials"""
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url
        self.api = None
        
        if api_key and secret_key:
            self.api = tradeapi.REST(api_key, secret_key, base_url, api_version='v2')
        
        self.fib_analyzer = AdvancedFibonacciAnalyzer()
        self.wave_analyzer = ElliottWaveAnalyzer()
        
        # Alpaca timeframe mapping
        self.alpaca_timeframes = {
            '1m': tradeapi.TimeFrame.Minute,
            '2m': tradeapi.TimeFrame(2, tradeapi.TimeFrameUnit.Minute),
            '5m': tradeapi.TimeFrame(5, tradeapi.TimeFrameUnit.Minute),
            '15m': tradeapi.TimeFrame(15, tradeapi.TimeFrameUnit.Minute),
            '30m': tradeapi.TimeFrame(30, tradeapi.TimeFrameUnit.Minute),
            '60m': tradeapi.TimeFrame.Hour,
            '1h': tradeapi.TimeFrame.Hour,
            '1d': tradeapi.TimeFrame.Day,
            '1wk': tradeapi.TimeFrame.Week,
            '1mo': tradeapi.TimeFrame.Month,
            '3mo': tradeapi.TimeFrame(3, tradeapi.TimeFrameUnit.Month)
        }
        
        self.valid_intervals = list(self.alpaca_timeframes.keys())
        
        # Alpaca has different limits than yfinance
        self.interval_limits = {
            '1m': {'max_days': 7},
            '2m': {'max_days': 30},
            '5m': {'max_days': 60},
            '15m': {'max_days': 60},
            '30m': {'max_days': 90},
            '60m': {'max_days': 365},
            '1h': {'max_days': 365},
            '1d': {'max_days': 'unlimited'},
            '1wk': {'max_days': 'unlimited'},
            '1mo': {'max_days': 'unlimited'},
            '3mo': {'max_days': 'unlimited'}
        }
    
    def set_credentials(self, api_key, secret_key, base_url='https://paper-api.alpaca.markets'):
        """Set Alpaca API credentials"""
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url
        self.api = tradeapi.REST(api_key, secret_key, base_url, api_version='v2')
    
    def _convert_period_to_dates(self, period):
        """Convert period to start/end dates"""
        end_date = datetime.now()
        
        period_map = {
            '1d': 1, '2d': 2, '5d': 5, '7d': 7, '10d': 10,
            '1mo': 30, '2mo': 60, '3mo': 90, '6mo': 180,
            '1y': 365, '2y': 730, '5y': 1825, '10y': 3650,
            'ytd': (datetime.now() - datetime(datetime.now().year, 1, 1)).days,
            'max': 3650
        }
        
        days = period_map.get(period, 180)
        start_date = end_date - timedelta(days=days)
        
        return start_date, end_date
    
    def get_stock_data(self, symbol, interval='1d', period='5y'):
        """Fetch stock data with specified interval using Alpaca API"""
        if not self.api:
            raise ValueError("Alpaca API credentials not set. Use set_credentials() method.")
        
        if interval not in self.valid_intervals:
            raise ValueError(f"Invalid interval. Must be one of: {self.valid_intervals}")
        
        try:
            start_date, end_date = self._convert_period_to_dates(period)
            
            # Check limits for the interval
            max_days = self.interval_limits[interval]['max_days']
            if max_days != 'unlimited':
                requested_days = (end_date - start_date).days
                if requested_days > max_days:
                    print(f"Warning: Requested {requested_days} days exceeds limit of {max_days} for {interval}. Adjusting...")
                    start_date = end_date - timedelta(days=max_days)
            
            # Get the appropriate timeframe
            timeframe = self.alpaca_timeframes[interval]
            
            # Fetch data from Alpaca
            bars = self.api.get_bars(
                symbol,
                timeframe,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                adjustment='raw'
            ).df
            
            if bars.empty:
                print(f"No data returned for {symbol} with interval {interval} and period {period}")
                return None
            
            # Convert to yfinance-like format
            df = pd.DataFrame()
            df['Open'] = bars['open']
            df['High'] = bars['high']
            df['Low'] = bars['low']
            df['Close'] = bars['close']
            df['Volume'] = bars['volume']
            df.index = bars.index
            
            print(f"✅ Fetched {len(df)} {interval} candles for {symbol} over {period}")
            return df
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    def _period_exceeds_limit(self, period, max_period):
        """Check if requested period exceeds the limit for the interval"""
        period_days = self._period_to_days(period)
        max_days = self._period_to_days(max_period)
        
        if period_days and max_days:
            return period_days > max_days
        return False
    
    def _period_to_days(self, period):
        """Convert period string to approximate days"""
        if period == 'max' or period == 'unlimited':
            return float('inf')
        
        period_map = {
            '1d': 1, '2d': 2, '5d': 5, '7d': 7, '10d': 10,
            '1mo': 30, '2mo': 60, '3mo': 90, '6mo': 180,
            '1y': 365, '2y': 730, '5y': 1825, '10y': 3650,
            '60d': 60, '730d': 730
        }
        
        return period_map.get(period, None)
    
    def calculate_comprehensive_indicators(self, df, interval):
        """Calculate all indicators including basic, Fibonacci, and Elliott Wave"""
        indicators = {}
        
        # Adjust indicator periods based on timeframe
        if interval in ['1m', '2m', '5m']:
            # Faster settings for short timeframes
            macd_fast, macd_slow, macd_signal = 6, 13, 4
            adx_period = 7
            rsi_period = 7
        elif interval in ['15m', '30m']:
            # Medium settings
            macd_fast, macd_slow, macd_signal = 8, 17, 6
            adx_period = 10
            rsi_period = 10
        elif interval in ['1h']:
            # Standard settings
            macd_fast, macd_slow, macd_signal = 12, 26, 9
            adx_period = 14
            rsi_period = 14
        else:
            # Daily+ settings
            macd_fast, macd_slow, macd_signal = 12, 26, 9
            adx_period = 14
            rsi_period = 14
        
        try:
            # Basic Technical Indicators
            indicators['basic'] = self._calculate_basic_indicators(df, macd_fast, macd_slow, macd_signal, adx_period, rsi_period)
            
            # Fibonacci Analysis
            fib_analysis = self.fib_analyzer.calculate_fibonacci_retracements(df)
            indicators['fibonacci'] = fib_analysis if fib_analysis else {'error': 'Insufficient data for Fibonacci analysis'}
            
            # Elliott Wave Analysis
            wave_analysis = self.wave_analyzer.detect_wave_structure(df)
            indicators['elliott_wave'] = wave_analysis
            
            # Combined Trading Signals
            indicators['trading_signals'] = self._generate_comprehensive_signals(
                indicators['basic'], 
                indicators['fibonacci'], 
                indicators['elliott_wave'],
                interval
            )
            
        except Exception as e:
            print(f"Error calculating indicators for {interval}: {e}")
            indicators['error'] = str(e)
        
        return indicators
    
    def _calculate_basic_indicators(self, df, macd_fast, macd_slow, macd_signal, adx_period, rsi_period):
        """Calculate basic technical indicators with improved error handling"""
        basic_indicators = {}
        
        try:
            # MACD
            macd_data = ta.macd(df['Close'], fast=macd_fast, slow=macd_slow, signal=macd_signal)
            if macd_data is not None and not macd_data.empty:
                macd_col = f'MACD_{macd_fast}_{macd_slow}_{macd_signal}'
                signal_col = f'MACDs_{macd_fast}_{macd_slow}_{macd_signal}'
                hist_col = f'MACDh_{macd_fast}_{macd_slow}_{macd_signal}'
                
                basic_indicators['MACD'] = {
                    'macd': macd_data[macd_col].iloc[-1] if macd_col in macd_data.columns else None,
                    'signal': macd_data[signal_col].iloc[-1] if signal_col in macd_data.columns else None,
                    'histogram': macd_data[hist_col].iloc[-1] if hist_col in macd_data.columns else None,
                    'settings': f"({macd_fast},{macd_slow},{macd_signal})"
                }
                
                if basic_indicators['MACD']['macd'] and basic_indicators['MACD']['signal']:
                    basic_indicators['MACD']['crossover'] = 'bullish' if basic_indicators['MACD']['macd'] > basic_indicators['MACD']['signal'] else 'bearish'
        
            # ADX
            adx_data = ta.adx(df['High'], df['Low'], df['Close'], length=adx_period)
            if adx_data is not None and not adx_data.empty:
                basic_indicators['ADX'] = {
                    'adx': adx_data[f'ADX_{adx_period}'].iloc[-1],
                    'di_plus': adx_data[f'DMP_{adx_period}'].iloc[-1],
                    'di_minus': adx_data[f'DMN_{adx_period}'].iloc[-1],
                    'period': adx_period,
                    'trend_strength': 'strong' if adx_data[f'ADX_{adx_period}'].iloc[-1] > 25 else 'weak'
                }
            
            # RSI
            rsi_data = ta.rsi(df['Close'], length=rsi_period)
            if rsi_data is not None:
                basic_indicators['RSI'] = {
                    'value': rsi_data.iloc[-1],
                    'period': rsi_period,
                    'overbought': rsi_data.iloc[-1] > 70,
                    'oversold': rsi_data.iloc[-1] < 30
                }
            
            # OBV
            obv_data = ta.obv(df['Close'], df['Volume'])
            if obv_data is not None and len(obv_data) > 1:
                basic_indicators['OBV'] = {
                    'current': obv_data.iloc[-1],
                    'previous': obv_data.iloc[-2],
                    'trend': 'bullish' if obv_data.iloc[-1] > obv_data.iloc[-2] else 'bearish'
                }
            
            # Moving Averages
            sma_20 = ta.sma(df['Close'], length=min(20, len(df)//2))
            sma_50 = ta.sma(df['Close'], length=min(50, len(df)//2))
            
            if sma_20 is not None and sma_50 is not None and len(sma_20) > 0 and len(sma_50) > 0:
                basic_indicators['Moving_Averages'] = {
                    'sma_20': sma_20.iloc[-1],
                    'sma_50': sma_50.iloc[-1],
                    'trend': 'bullish' if sma_20.iloc[-1] > sma_50.iloc[-1] else 'bearish'
                }
            
            # Bollinger Bands - Fixed with better error handling
            try:
                bb_data = ta.bbands(df['Close'], length=20, std=2)
                if bb_data is not None and not bb_data.empty:
                    # Check for different possible column naming conventions
                    upper_col = None
                    middle_col = None
                    lower_col = None
                    
                    # Common column name patterns for Bollinger Bands
                    possible_upper = ['BBU_20_2.0', 'BBU_20_2', 'upper', 'Upper', 'BB_upper']
                    possible_middle = ['BBM_20_2.0', 'BBM_20_2', 'middle', 'Middle', 'BB_middle']
                    possible_lower = ['BBL_20_2.0', 'BBL_20_2', 'lower', 'Lower', 'BB_lower']
                    
                    # Find the correct column names
                    for col in bb_data.columns:
                        if any(pattern in col for pattern in possible_upper):
                            upper_col = col
                        elif any(pattern in col for pattern in possible_middle):
                            middle_col = col
                        elif any(pattern in col for pattern in possible_lower):
                            lower_col = col
                    
                    # If standard naming doesn't work, try to identify by position
                    if not all([upper_col, middle_col, lower_col]) and len(bb_data.columns) >= 3:
                        cols = list(bb_data.columns)
                        # Typically: lower, middle, upper or upper, middle, lower
                        if len(cols) == 3:
                            # Sort by last value to identify upper/lower
                            last_values = [(col, bb_data[col].iloc[-1]) for col in cols]
                            last_values.sort(key=lambda x: x[1])
                            lower_col = last_values[0][0]
                            middle_col = last_values[1][0]
                            upper_col = last_values[2][0]
                    
                    if all([upper_col, middle_col, lower_col]):
                        current_price = df['Close'].iloc[-1]
                        upper_val = bb_data[upper_col].iloc[-1]
                        middle_val = bb_data[middle_col].iloc[-1]
                        lower_val = bb_data[lower_col].iloc[-1]
                        
                        basic_indicators['Bollinger_Bands'] = {
                            'upper': upper_val,
                            'middle': middle_val,
                            'lower': lower_val,
                            'position': self._get_bb_position_safe(current_price, upper_val, middle_val, lower_val),
                            'columns_used': {
                                'upper': upper_col,
                                'middle': middle_col,
                                'lower': lower_col
                            }
                        }
                    else:
                        print(f"Warning: Could not identify Bollinger Bands columns. Available columns: {list(bb_data.columns)}")
                        basic_indicators['Bollinger_Bands'] = {
                            'error': 'Could not identify column names',
                            'available_columns': list(bb_data.columns)
                        }
                else:
                    print("Warning: Bollinger Bands calculation returned empty data")
                    
            except Exception as bb_error:
                print(f"Bollinger Bands calculation error: {bb_error}")
                basic_indicators['Bollinger_Bands'] = {'error': str(bb_error)}
            
        except Exception as e:
            print(f"Error in basic indicators calculation: {e}")
            basic_indicators['error'] = str(e)
        
        return basic_indicators
    
    def _get_bb_position_safe(self, current_price, upper, middle, lower):
        """Safe version of Bollinger Bands position calculation"""
        try:
            # Calculate percentage position within bands
            if upper != lower and upper is not None and lower is not None:
                bb_percentage = (current_price - lower) / (upper - lower) * 100
            else:
                bb_percentage = 50  # Default to middle if bands are flat or invalid
            
            # Determine position category
            if current_price > upper:
                category = 'above_upper'
            elif current_price > middle:
                category = 'upper_half'
            elif current_price > lower:
                category = 'lower_half'
            else:
                category = 'below_lower'
            
            position_data = {
                'category': category,
                'percentage': round(bb_percentage, 2),
                'distance_from_middle': round(abs(current_price - middle), 4) if middle is not None else None
            }
            
            return position_data
            
        except Exception as e:
            print(f"Error calculating BB position: {e}")
            return {
                'category': 'error', 
                'percentage': None, 
                'distance_from_middle': None,
                'error': str(e)
            }
        
    def _get_bb_position(self, current_price, bb_row):
        """Enhanced version with better error handling for Bollinger Bands position"""
        try:
            # Handle both dictionary and Series input
            if isinstance(bb_row, dict):
                upper = bb_row.get('upper')
                middle = bb_row.get('middle') 
                lower = bb_row.get('lower')
            else:
                # Try different column name patterns
                upper = None
                middle = None
                lower = None
                
                # Check for standard column names
                for col in bb_row.index:
                    if 'BBU' in col or 'upper' in col.lower():
                        upper = bb_row[col]
                    elif 'BBM' in col or 'middle' in col.lower():
                        middle = bb_row[col]
                    elif 'BBL' in col or 'lower' in col.lower():
                        lower = bb_row[col]
            
            # Validate values
            if any(val is None or pd.isna(val) for val in [upper, middle, lower]):
                return {
                    'category': 'insufficient_data',
                    'percentage': None,
                    'distance_from_middle': None
                }
            
            # Calculate percentage position within bands
            if upper != lower:  # Avoid division by zero
                bb_percentage = (current_price - lower) / (upper - lower) * 100
            else:
                bb_percentage = 50  # Default to middle if bands are flat
            
            # Determine category
            category = self._get_position_category(current_price, upper, middle, lower)
            
            position_data = {
                'category': category,
                'percentage': round(bb_percentage, 2),
                'distance_from_middle': round(abs(current_price - middle), 4)
            }
            
            return position_data
            
        except Exception as e:
            print(f"Error calculating Bollinger Bands position: {e}")
            return {
                'category': 'error', 
                'percentage': None, 
                'distance_from_middle': None,
                'error_details': str(e)
            }
    
    def _generate_comprehensive_signals(self, basic_indicators, fib_analysis, wave_analysis, interval):
        """Generate comprehensive trading signals combining all analyses"""
        signals = {
            'timeframe': interval,
            'overall_bias': 'neutral',
            'strength': 0,
            'confidence': 'low',
            'entry_signals': [],
            'exit_signals': [],
            'key_levels': {},
            'risk_reward': {},
            'signal_breakdown': {
                'basic_score': 0,
                'fibonacci_score': 0,
                'elliott_wave_score': 0,
                'total_score': 0
            }
        }
        
        # Scoring system
        bullish_score = 0
        bearish_score = 0
        
        # Basic Indicators Scoring
        basic_bullish, basic_bearish, basic_signals = self._score_basic_indicators(basic_indicators)
        bullish_score += basic_bullish
        bearish_score += basic_bearish
        signals['entry_signals'].extend(basic_signals['bullish'])
        signals['exit_signals'].extend(basic_signals['bearish'])
        signals['signal_breakdown']['basic_score'] = basic_bullish - basic_bearish
        
        # Fibonacci Analysis Scoring
        if fib_analysis and 'error' not in fib_analysis:
            fib_bullish, fib_bearish, fib_signals = self._score_fibonacci_analysis(fib_analysis)
            bullish_score += fib_bullish
            bearish_score += fib_bearish
            signals['entry_signals'].extend(fib_signals['bullish'])
            signals['exit_signals'].extend(fib_signals['bearish'])
            signals['signal_breakdown']['fibonacci_score'] = fib_bullish - fib_bearish
            
            # Set key Fibonacci levels
            signals['key_levels'].update({
                'fib_support': fib_analysis['nearest_support']['price'] if fib_analysis['nearest_support'] else None,
                'fib_resistance': fib_analysis['nearest_resistance']['price'] if fib_analysis['nearest_resistance'] else None,
                'fib_382': fib_analysis['key_levels']['38.2%'],
                'fib_618': fib_analysis['key_levels']['61.8%']
            })
        
        # Elliott Wave Analysis Scoring
        if wave_analysis and wave_analysis.get('confidence', 0) > 0:
            wave_bullish, wave_bearish, wave_signals = self._score_elliott_wave_analysis(wave_analysis)
            bullish_score += wave_bullish
            bearish_score += wave_bearish
            signals['entry_signals'].extend(wave_signals['bullish'])
            signals['exit_signals'].extend(wave_signals['bearish'])
            signals['signal_breakdown']['elliott_wave_score'] = wave_bullish - wave_bearish
        
        # Calculate overall bias and strength
        total_score = bullish_score + bearish_score
        signals['signal_breakdown']['total_score'] = bullish_score - bearish_score
        
        if total_score > 0:
            bullish_pct = (bullish_score / total_score) * 100
            
            if bullish_pct >= 75:
                signals['overall_bias'] = 'strong_bullish'
                signals['strength'] = bullish_pct
                signals['confidence'] = 'high'
            elif bullish_pct >= 60:
                signals['overall_bias'] = 'bullish'
                signals['strength'] = bullish_pct
                signals['confidence'] = 'medium'
            elif bullish_pct >= 40:
                signals['overall_bias'] = 'neutral'
                signals['strength'] = 50
                signals['confidence'] = 'low'
            elif bullish_pct >= 25:
                signals['overall_bias'] = 'bearish'
                signals['strength'] = 100 - bullish_pct
                signals['confidence'] = 'medium'
            else:
                signals['overall_bias'] = 'strong_bearish'
                signals['strength'] = 100 - bullish_pct
                signals['confidence'] = 'high'
        
        # Calculate risk/reward if we have key levels
        if signals['key_levels'].get('fib_support') and signals['key_levels'].get('fib_resistance'):
            current_price = fib_analysis['current_price']
            support = signals['key_levels']['fib_support']
            resistance = signals['key_levels']['fib_resistance']
            
            risk = abs(current_price - support)
            reward = abs(resistance - current_price)
            
            if risk > 0:
                signals['risk_reward'] = {
                    'ratio': reward / risk,
                    'risk_amount': risk,
                    'reward_potential': reward,
                    'stop_loss': support,
                    'take_profit': resistance
                }
        # print("Signallllls", signals)
        return signals
    
    def _score_basic_indicators(self, basic_indicators):
        """Score basic technical indicators"""
        bullish_score = 0
        bearish_score = 0
        signals = {'bullish': [], 'bearish': []}
        
        # MACD scoring
        if 'MACD' in basic_indicators and basic_indicators['MACD'].get('crossover'):
            if basic_indicators['MACD']['crossover'] == 'bullish':
                bullish_score += 3
                signals['bullish'].append(f"MACD bullish crossover {basic_indicators['MACD']['settings']}")
            else:
                bearish_score += 3
                signals['bearish'].append(f"MACD bearish crossover {basic_indicators['MACD']['settings']}")
        
        # ADX scoring
        if 'ADX' in basic_indicators:
            adx_data = basic_indicators['ADX']
            if adx_data['trend_strength'] == 'strong':
                if adx_data['di_plus'] > adx_data['di_minus']:
                    bullish_score += 4
                    signals['bullish'].append(f"Strong bullish trend (ADX: {adx_data['adx']:.1f})")
                else:
                    bearish_score += 4
                    signals['bearish'].append(f"Strong bearish trend (ADX: {adx_data['adx']:.1f})")
        
        # RSI scoring
        if 'RSI' in basic_indicators:
            rsi_data = basic_indicators['RSI']
            if rsi_data['oversold']:
                bullish_score += 2
                signals['bullish'].append(f"RSI oversold ({rsi_data['value']:.1f})")
            elif rsi_data['overbought']:
                bearish_score += 2
                signals['bearish'].append(f"RSI overbought ({rsi_data['value']:.1f})")
        
        # OBV scoring
        if 'OBV' in basic_indicators:
            obv_data = basic_indicators['OBV']
            if obv_data['trend'] == 'bullish':
                bullish_score += 2
                signals['bullish'].append("Volume supporting uptrend (OBV)")
            else:
                bearish_score += 2
                signals['bearish'].append("Volume supporting downtrend (OBV)")
        
        # Moving Average scoring
        if 'Moving_Averages' in basic_indicators:
            ma_data = basic_indicators['Moving_Averages']
            if ma_data['trend'] == 'bullish':
                bullish_score += 2
                signals['bullish'].append("Price above moving averages")
            else:
                bearish_score += 2
                signals['bearish'].append("Price below moving averages")
        
        # Bollinger Bands scoring
        if 'Bollinger_Bands' in basic_indicators:
            bb_data = basic_indicators['Bollinger_Bands']
            if isinstance(bb_data.get('position'), dict):
                position_category = bb_data['position'].get('category', 'unknown')
                if position_category == 'below_lower':
                    bullish_score += 1
                    signals['bullish'].append("Price below lower Bollinger Band (oversold)")
                elif position_category == 'above_upper':
                    bearish_score += 1
                    signals['bearish'].append("Price above upper Bollinger Band (overbought)")
        
        return bullish_score, bearish_score, signals
    
    def _score_fibonacci_analysis(self, fib_analysis):
        """Score Fibonacci retracement analysis"""
        bullish_score = 0
        bearish_score = 0
        signals = {'bullish': [], 'bearish': []}
        
        current_price = fib_analysis['current_price']
        
        # Check proximity to key Fibonacci levels
        for level_name, level_price in fib_analysis['key_levels'].items():
            distance_pct = abs((current_price - level_price) / current_price) * 100
            
            if distance_pct < 1.5:  # Within 1.5% of Fibonacci level
                if level_name in ['38.2%', '50%', '61.8%']:  # Key retracement levels
                    if fib_analysis['trend_direction'] == 'uptrend':
                        if current_price >= level_price:
                            bullish_score += 3
                            signals['bullish'].append(f"Price holding above {level_name} Fibonacci support")
                        else:
                            bearish_score += 2
                            signals['bearish'].append(f"Price below {level_name} Fibonacci support")
                    else:  # downtrend
                        if current_price <= level_price:
                            bearish_score += 3
                            signals['bearish'].append(f"Price holding below {level_name} Fibonacci resistance")
                        else:
                            bullish_score += 2
                            signals['bullish'].append(f"Price above {level_name} Fibonacci resistance")
        
        # Trend direction scoring
        if fib_analysis['trend_direction'] == 'uptrend':
            bullish_score += 1
            signals['bullish'].append("Fibonacci analysis shows uptrend structure")
        else:
            bearish_score += 1
            signals['bearish'].append("Fibonacci analysis shows downtrend structure")
        
        return bullish_score, bearish_score, signals
    
    def _score_elliott_wave_analysis(self, wave_analysis):
        """Score Elliott Wave analysis"""
        bullish_score = 0
        bearish_score = 0
        signals = {'bullish': [], 'bearish': []}
        
        confidence = wave_analysis.get('confidence', 0)
        pattern = wave_analysis.get('pattern', '')
        trend = wave_analysis.get('trend', '')
        
        # Weight scoring by confidence
        confidence_multiplier = confidence / 100
        
        if pattern == 'impulse' and confidence > 50:
            if trend == 'bullish':
                score = int(4 * confidence_multiplier)
                bullish_score += score
                signals['bullish'].append(f"Bullish impulse wave pattern (confidence: {confidence:.0f}%)")
            elif trend == 'bearish':
                score = int(4 * confidence_multiplier)
                bearish_score += score
                signals['bearish'].append(f"Bearish impulse wave pattern (confidence: {confidence:.0f}%)")
        
        elif pattern == 'corrective' and confidence > 40:
            # Corrective patterns suggest trend reversal
            if trend == 'bullish':
                score = int(2 * confidence_multiplier)
                bearish_score += score  # Corrective in bullish trend suggests bearish reversal
                signals['bearish'].append(f"Corrective wave pattern suggests trend exhaustion (confidence: {confidence:.0f}%)")
            elif trend == 'bearish':
                score = int(2 * confidence_multiplier)
                bullish_score += score  # Corrective in bearish trend suggests bullish reversal
                signals['bullish'].append(f"Corrective wave pattern suggests trend exhaustion (confidence: {confidence:.0f}%)")
        
        return bullish_score, bearish_score, signals
    
    def analyze_comprehensive_multi_timeframe(self, symbol, timeframes=None, base_period='5y'):
        """Comprehensive analysis across multiple timeframes"""
        if timeframes is None:
            timeframes = ['5m', '15m', '1h', '1d']
        
        results = {}
        
        print(f"\n🚀 Starting comprehensive multi-timeframe analysis for {symbol}...")
        print(f"Timeframes: {', '.join(timeframes)}")
        print("=" * 80)
        
        for interval in timeframes:
            print(f"\n📊 Analyzing {symbol} on {interval} timeframe...")
            
            # Adjust period based on interval
            if interval in ['1m', '2m']:
                period = '5d'
            elif interval in ['5m', '15m', '30m']:
                period = '10d'
            elif interval in ['60m', '1h']:
                period = '3 mo'
            elif interval in ['1d', '1d']:
                period='1y'
            elif interval in ['1wk', '1wk']:
                period = '2y'
            else:
                period = base_period

            # Get data
            df = self.get_stock_data(symbol, interval=interval, period=period)
            
            if df is not None:
                # Calculate comprehensive indicators
                indicators = self.calculate_comprehensive_indicators(df, interval)
                
                results[interval] = {
                    'data': df,
                    'indicators': indicators,
                    'current_price': df['Close'].iloc[-1],
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
        
        return results
    
    def print_comprehensive_analysis(self, symbol, results):
        """Print detailed comprehensive analysis"""
        if not results:
            print("No analysis data available")
            return
        
        print(f"\n{'='*100}")
        print(f"COMPREHENSIVE MULTI-TIMEFRAME ANALYSIS: {symbol}")
        print(f"{'='*100}")
        print(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Summary table
        print(f"\n📊 TIMEFRAME SUMMARY:")
        print(f"{'TF':<6} {'Price':<10} {'Bias':<15} {'Strength':<8} {'Conf':<6} {'MACD':<8} {'ADX':<6} {'RSI':<6} {'Fib':<8} {'Wave':<12}")
        print("-" * 100)
        
        for timeframe, data in results.items():
            if 'indicators' in data and 'trading_signals' in data['indicators']:
                current_price = data['current_price']
                signals = data['indicators']['trading_signals']
                basic = data['indicators'].get('basic', {})
                fib = data['indicators'].get('fibonacci', {})
                wave = data['indicators'].get('elliott_wave', {})
                
                # Format data for table
                bias = signals['overall_bias'].replace('_', ' ').title()[:14]
                strength = f"{signals['strength']:.0f}%"
                confidence = signals['confidence'][:4].title()
                
                macd_status = "Bull" if basic.get('MACD', {}).get('crossover') == 'bullish' else "Bear" if basic.get('MACD', {}).get('crossover') == 'bearish' else "N/A"
                adx_val = f"{basic.get('ADX', {}).get('adx', 0):.0f}" if 'ADX' in basic else "N/A"
                rsi_val = f"{basic.get('RSI', {}).get('value', 0):.0f}" if 'RSI' in basic else "N/A"
                
                fib_trend = fib.get('trend_direction', 'N/A')[:7] if 'error' not in fib else "N/A"
                wave_pattern = f"{wave.get('pattern', 'N/A')[:4]}-{wave.get('confidence', 0):.0f}%" if wave.get('confidence', 0) > 0 else "N/A"
                
                print(f"{timeframe:<6} ${current_price:<9.2f} {bias:<15} {strength:<8} {confidence:<6} {macd_status:<8} {adx_val:<6} {rsi_val:<6} {fib_trend:<8} {wave_pattern:<12}")
        
        # Detailed analysis for each timeframe
        for timeframe, data in results.items():
            if 'indicators' in data:
                self._print_detailed_timeframe_analysis(symbol, timeframe, data)
        
        # Multi-timeframe consensus
        self._print_multi_timeframe_consensus(symbol, results)
    
    def _print_detailed_timeframe_analysis(self, symbol, timeframe, data):
        """Print detailed analysis for a specific timeframe with improved error handling"""
        print(f"\n🔍 DETAILED ANALYSIS - {timeframe.upper()} TIMEFRAME:")
        print("-" * 60)
        
        indicators = data['indicators']
        df = data['data']
        current_price = data['current_price']
        
        print(f"Data Points: {len(df)} candles | Current Price: ${current_price:.2f}")
        print(f"Price Range: ${df['Low'].min():.2f} - ${df['High'].max():.2f}")
        
        # Basic Indicators
        if 'basic' in indicators:
            basic = indicators['basic']
            print(f"\n📈 Basic Indicators:")
            
            if 'MACD' in basic:
                macd = basic['MACD']
                print(f"  MACD {macd.get('settings', '')}: {macd.get('macd', 0):.4f} | Signal: {macd.get('signal', 0):.4f} | Cross: {macd.get('crossover', 'N/A')}")
            
            if 'ADX' in basic:
                adx = basic['ADX']
                print(f"  ADX({adx.get('period', 14)}): {adx.get('adx', 0):.2f} | +DI: {adx.get('di_plus', 0):.2f} | -DI: {adx.get('di_minus', 0):.2f} | Strength: {adx.get('trend_strength', 'N/A')}")
            
            if 'RSI' in basic:
                rsi = basic['RSI']
                status = "Overbought" if rsi.get('overbought') else "Oversold" if rsi.get('oversold') else "Normal"
                print(f"  RSI({rsi.get('period', 14)}): {rsi.get('value', 0):.2f} ({status})")
            
            if 'OBV' in basic:
                obv = basic['OBV']
                print(f"  OBV: {obv.get('current', 0):,.0f} | Trend: {obv.get('trend', 'N/A')}")
            
            if 'Moving_Averages' in basic:
                ma = basic['Moving_Averages']
                print(f"  SMA20: ${ma.get('sma_20', 0):.2f} | SMA50: ${ma.get('sma_50', 0):.2f} | Trend: {ma.get('trend', 'N/A')}")
            
            if 'Bollinger_Bands' in basic:
                bb = basic['Bollinger_Bands']
                
                # Handle error cases
                if 'error' in bb:
                    print(f"  Bollinger Bands: Error - {bb['error']}")
                    if 'available_columns' in bb:
                        print(f"    Available columns: {bb['available_columns']}")
                else:
                    # Normal case - display BB data
                    print(f"  Bollinger Bands: Upper ${bb.get('upper', 0):.2f} | Middle ${bb.get('middle', 0):.2f} | Lower ${bb.get('lower', 0):.2f}")
                    
                    # Handle position display safely
                    position = bb.get('position', 'N/A')
                    if isinstance(position, dict):
                        # Extract category from position dictionary
                        position_str = position.get('category', 'N/A')
                        percentage = position.get('percentage', 'N/A')
                        distance = position.get('distance_from_middle', 'N/A')
                        
                        # Format position string
                        if position_str != 'N/A':
                            position_display = position_str.replace('_', ' ').title()
                        else:
                            position_display = 'N/A'
                        
                        print(f"  Position: {position_display}")
                        if percentage != 'N/A':
                            print(f"    BB Percentage: {percentage}%")
                        if distance != 'N/A':
                            print(f"    Distance from Middle: ${distance}")
                            
                    elif isinstance(position, str):
                        # Handle string position (legacy format)
                        position_display = position.replace('_', ' ').title()
                        print(f"  Position: {position_display}")
                    else:
                        print(f"  Position: N/A")
        
        # Fibonacci Analysis
        if 'fibonacci' in indicators and 'error' not in indicators['fibonacci']:
            fib = indicators['fibonacci']
            print(f"\n🌀 Fibonacci Analysis:")
            print(f"  Trend Direction: {fib.get('trend_direction', 'N/A').title()}")
            print(f"  Price Range: ${fib.get('low_price', 0):.2f} - ${fib.get('high_price', 0):.2f}")
            print(f"  Key Levels:")
            
            key_levels = fib.get('key_levels', {})
            for level in ['38.2%', '50%', '61.8%']:
                if level in key_levels:
                    price = key_levels[level]
                    distance = abs(current_price - price)
                    pct_away = (distance / current_price) * 100
                    print(f"    {level}: ${price:.2f} ({pct_away:.1f}% away)")
            
            if fib.get('nearest_support'):
                support = fib['nearest_support']
                print(f"  Nearest Support: {support['level']} at ${support['price']:.2f}")
            
            if fib.get('nearest_resistance'):
                resistance = fib['nearest_resistance']
                print(f"  Nearest Resistance: {resistance['level']} at ${resistance['price']:.2f}")
        
        # Elliott Wave Analysis
        if 'elliott_wave' in indicators:
            wave = indicators['elliott_wave']
            print(f"\n🌊 Elliott Wave Analysis:")
            print(f"  Pattern: {wave.get('pattern', 'N/A').title()}")
            print(f"  Trend: {wave.get('trend', 'N/A').title()}")
            print(f"  Confidence: {wave.get('confidence', 0):.0f}%")
            
            if wave.get('next_expectation'):
                print(f"  Next Expectation: {wave['next_expectation']}")
            
            if wave.get('wave_points'):
                print(f"  Wave Points Detected: {len(wave['wave_points'])}")
        
        # Trading Signals
        if 'trading_signals' in indicators:
            signals = indicators['trading_signals']
            print(f"\n🎯 Trading Signals:")
            print(f"  Overall Bias: {signals['overall_bias'].replace('_', ' ').title()}")
            print(f"  Strength: {signals['strength']:.0f}%")
            print(f"  Confidence: {signals['confidence'].title()}")
            
            # Signal breakdown
            breakdown = signals.get('signal_breakdown', {})
            print(f"  Score Breakdown:")
            print(f"    Basic Indicators: {breakdown.get('basic_score', 0):+d}")
            print(f"    Fibonacci: {breakdown.get('fibonacci_score', 0):+d}")
            print(f"    Elliott Wave: {breakdown.get('elliott_wave_score', 0):+d}")
            print(f"    Total Score: {breakdown.get('total_score', 0):+d}")
            
            # Entry signals
            if signals.get('entry_signals'):
                print(f"  📈 Bullish Signals:")
                for signal in signals['entry_signals'][:3]:  # Show top 3
                    print(f"    • {signal}")
            
            # Exit signals
            if signals.get('exit_signals'):
                print(f"  📉 Bearish Signals:")
                for signal in signals['exit_signals'][:3]:  # Show top 3
                    print(f"    • {signal}")
            
            # Risk/Reward
            if signals.get('risk_reward'):
                rr = signals['risk_reward']
                print(f"  💰 Risk/Reward Analysis:")
                print(f"    Ratio: {rr.get('ratio', 0):.2f}:1")
                print(f"    Stop Loss: ${rr.get('stop_loss', 0):.2f}")
                print(f"    Take Profit: ${rr.get('take_profit', 0):.2f}")
                print(f"    Risk Amount: ${rr.get('risk_amount', 0):.2f}")
                print(f"    Reward Potential: ${rr.get('reward_potential', 0):.2f}")
    
    def _print_multi_timeframe_consensus(self, symbol, results):
        """Print consensus analysis across all timeframes"""
        print(f"\n🎯 MULTI-TIMEFRAME CONSENSUS:")
        print("-" * 60)
        
        # Collect all signals
        all_biases = []
        all_strengths = []
        all_confidences = []
        timeframe_recommendations = {}
        
        for timeframe, data in results.items():
            if 'indicators' in data and 'trading_signals' in data['indicators']:
                signals = data['indicators']['trading_signals']
                bias = signals['overall_bias']
                strength = signals['strength']
                confidence = signals['confidence']
                
                all_biases.append(bias)
                all_strengths.append(strength)
                all_confidences.append(confidence)
                
                # Convert to recommendation
                if 'bullish' in bias:
                    timeframe_recommendations[timeframe] = 'BUY'
                elif 'bearish' in bias:
                    timeframe_recommendations[timeframe] = 'SELL'
                else:
                    timeframe_recommendations[timeframe] = 'HOLD'
        
        if not all_biases:
            print("No consensus data available")
            return
        
        # Count recommendations
        buy_count = sum(1 for rec in timeframe_recommendations.values() if rec == 'BUY')
        sell_count = sum(1 for rec in timeframe_recommendations.values() if rec == 'SELL')
        hold_count = sum(1 for rec in timeframe_recommendations.values() if rec == 'HOLD')
        total_timeframes = len(timeframe_recommendations)
        
        # Calculate percentages
        buy_pct = (buy_count / total_timeframes) * 100 if total_timeframes > 0 else 0
        sell_pct = (sell_count / total_timeframes) * 100 if total_timeframes > 0 else 0
        hold_pct = (hold_count / total_timeframes) * 100 if total_timeframes > 0 else 0
        
        # Determine overall consensus
        if buy_pct >= 60:
            overall_recommendation = 'BUY'
            consensus_confidence = 'High' if buy_pct >= 75 else 'Medium'
        elif sell_pct >= 60:
            overall_recommendation = 'SELL'
            consensus_confidence = 'High' if sell_pct >= 75 else 'Medium'
        else:
            overall_recommendation = 'HOLD'
            consensus_confidence = 'Low'
        
        # Print consensus results
        print(f"Overall Recommendation: {overall_recommendation}")
        print(f"Consensus Confidence: {consensus_confidence}")
        print(f"Agreement: {max(buy_pct, sell_pct, hold_pct):.1f}%")
        print()
        print(f"Timeframe Breakdown:")
        print(f"  BUY signals: {buy_count}/{total_timeframes} ({buy_pct:.1f}%)")
        print(f"  SELL signals: {sell_count}/{total_timeframes} ({sell_pct:.1f}%)")
        print(f"  HOLD signals: {hold_count}/{total_timeframes} ({hold_pct:.1f}%)")
        print()
        print(f"Individual Timeframe Recommendations:")
        for tf, rec in timeframe_recommendations.items():
            print(f"  {tf}: {rec}")
        
        # Average strength across timeframes
        avg_strength = np.mean(all_strengths) if all_strengths else 0
        print(f"\nAverage Signal Strength: {avg_strength:.1f}%")
        
        # High confidence timeframes
        high_conf_timeframes = [tf for tf, data in results.items() 
                               if data.get('indicators', {}).get('trading_signals', {}).get('confidence') == 'high']
        if high_conf_timeframes:
            print(f"High Confidence Timeframes: {', '.join(high_conf_timeframes)}")
   
    def get_timeframe_consensus(self, results):
        """Get consensus across all timeframes"""
        if not results:
            return None
        
        recommendations = []
        strengths = []
        
        for timeframe, data in results.items():
            if 'timeframe_analysis' in data:
                analysis = data['timeframe_analysis']
                recommendations.append(analysis['recommendation'])
                strengths.append(analysis['strength'])
        
        # Count recommendations
        buy_count = recommendations.count('buy')
        sell_count = recommendations.count('sell')
        hold_count = recommendations.count('hold')
        
        total = len(recommendations)
        
        consensus = {
            'total_timeframes': total,
            'buy_signals': buy_count,
            'sell_signals': sell_count,
            'hold_signals': hold_count,
            'buy_percentage': (buy_count / total) * 100 if total > 0 else 0,
            'sell_percentage': (sell_count / total) * 100 if total > 0 else 0,
            'hold_percentage': (hold_count / total) * 100 if total > 0 else 0
        }
        
        # Determine overall consensus
        if consensus['buy_percentage'] >= 60:
            consensus['overall_recommendation'] = 'BUY'
            consensus['confidence'] = 'High' if consensus['buy_percentage'] >= 75 else 'Medium'
        elif consensus['sell_percentage'] >= 60:
            consensus['overall_recommendation'] = 'SELL'
            consensus['confidence'] = 'High' if consensus['sell_percentage'] >= 75 else 'Medium'
        else:
            consensus['overall_recommendation'] = 'HOLD'
            consensus['confidence'] = 'Low'
        
        return consensus

class MultiTimeframeAnalyzer:
    def __init__(self, api_key=None, secret_key=None, base_url='https://paper-api.alpaca.markets'):
        """Initialize with Alpaca API credentials"""
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url
        self.api = None
        
        if api_key and secret_key:
            self.api = tradeapi.REST(api_key, secret_key, base_url, api_version='v2')
        
        # Alpaca timeframe mapping
        self.alpaca_timeframes = {
            '1m': tradeapi.TimeFrame.Minute,
            '2m': tradeapi.TimeFrame(2, tradeapi.TimeFrameUnit.Minute),
            '5m': tradeapi.TimeFrame(5, tradeapi.TimeFrameUnit.Minute),
            '15m': tradeapi.TimeFrame(15, tradeapi.TimeFrameUnit.Minute),
            '30m': tradeapi.TimeFrame(30, tradeapi.TimeFrameUnit.Minute),
            '60m': tradeapi.TimeFrame.Hour,
            '1h': tradeapi.TimeFrame.Hour,
            '1d': tradeapi.TimeFrame.Day,
            '1wk': tradeapi.TimeFrame.Week,
            '1mo': tradeapi.TimeFrame.Month,
            '3mo': tradeapi.TimeFrame(3, tradeapi.TimeFrameUnit.Month)
        }
        
        self.valid_intervals = list(self.alpaca_timeframes.keys())
        
        self.interval_limits = {
            '1m': {'max_period': '7d', 'max_days': 7},
            '2m': {'max_period': '60d', 'max_days': 60},
            '5m': {'max_period': '60d', 'max_days': 60},
            '15m': {'max_period': '60d', 'max_days': 60},
            '30m': {'max_period': '60d', 'max_days': 60},
            '60m': {'max_period': '730d', 'max_days': 730},
            '1h': {'max_period': '730d', 'max_days': 730},
            '1d': {'max_period': 'max', 'max_days': 'unlimited'},
            '5d': {'max_period': 'max', 'max_days': 'unlimited'},
            '1wk': {'max_period': 'max', 'max_days': 'unlimited'},
            '1mo': {'max_period': 'max', 'max_days': 'unlimited'},
            '3mo': {'max_period': 'max', 'max_days': 'unlimited'}
        }
    
    def set_credentials(self, api_key, secret_key, base_url='https://paper-api.alpaca.markets'):
        """Set Alpaca API credentials"""
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url
        self.api = tradeapi.REST(api_key, secret_key, base_url, api_version='v2')
    
    def _convert_period_to_dates(self, period):
        """Convert period to start/end dates"""
        end_date = datetime.now()
        
        period_map = {
            '1d': 1, '2d': 2, '5d': 5, '7d': 7, '10d': 10,
            '1mo': 30, '2mo': 60, '3mo': 90, '6mo': 180,
            '1y': 365, '2y': 730, '5y': 1825, '10y': 3650,
            '60d': 60, '730d': 730
        }
        
        days = period_map.get(period, 180)
        start_date = end_date - timedelta(days=days)
        
        return start_date, end_date
    
    def get_stock_data(self, symbol, interval='1d', period='18mo'):
        """Fetch stock data with specified interval using Alpaca API"""
        if not self.api:
            raise ValueError("Alpaca API credentials not set. Use set_credentials() method.")
        
        if interval not in self.valid_intervals:
            raise ValueError(f"Invalid interval. Must be one of: {self.valid_intervals}")
        
        # Check period limits for the interval
        max_period = self.interval_limits[interval]['max_period']
        if max_period != 'max' and self._period_exceeds_limit(period, max_period):
            print(f"Warning: Period '{period}' may exceed limit for interval '{interval}'. Using '{max_period}' instead.")
            period = max_period
        
        try:
            start_date, end_date = self._convert_period_to_dates(period)
            timeframe = self.alpaca_timeframes[interval]
            
            bars = self.api.get_bars(
                symbol,
                timeframe,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                adjustment='raw'
            ).df
            
            if bars.empty:
                print(f"No data returned for {symbol} with interval {interval} and period {period}")
                return None
            
            # Convert to yfinance-like format
            df = pd.DataFrame()
            df['Open'] = bars['open']
            df['High'] = bars['high']
            df['Low'] = bars['low']
            df['Close'] = bars['close']
            df['Volume'] = bars['volume']
            df.index = bars.index
            
            print(f"✅ Fetched {len(df)} {interval} candles for {symbol} over {period}")
            return df
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    def _period_exceeds_limit(self, period, max_period):
        """Check if requested period exceeds the limit for the interval"""
        period_days = self._period_to_days(period)
        max_days = self._period_to_days(max_period)
        
        if period_days and max_days:
            return period_days > max_days
        return False
    
    def _period_to_days(self, period):
        """Convert period string to approximate days"""
        if period == 'max':
            return float('inf')
        
        period_map = {
            '1d': 1, '2d': 2, '5d': 5, '7d': 7, '10d': 10,
            '1mo': 30, '2mo': 60, '3mo': 90, '6mo': 180,
            '1y': 365, '2y': 730, '5y': 1825, '10y': 3650,
            '60d': 60, '730d': 730
        }
        
        return period_map.get(period, None)
    
    def analyze_multiple_timeframes(self, symbol, timeframes=None, base_period='6mo'):
        """Analyze symbol across multiple timeframes"""
        if timeframes is None:
            timeframes = ['5m', '15m', '1h', '1d']
        
        results = {}
        
        for interval in timeframes:
            print(f"\n📊 Analyzing {symbol} on {interval} timeframe...")
            
            # Adjust period based on interval
            if interval in ['1m', '2m']:
                period = '1d'
            elif interval in ['5m', '15m', '30m']:
                period = '5d'
            elif interval in ['60m', '1h']:
                period = '1mo'
            else:
                period = base_period
            
            # Get data
            df = self.get_stock_data(symbol, interval=interval, period=period)
            
            if df is not None:
                # Calculate indicators
                indicators = self.calculate_indicators(df, interval)
                
                results[interval] = {
                    'data': df,
                    'indicators': indicators,
                    'timeframe_analysis': self.analyze_timeframe_signals(indicators, interval)
                }
        
        return results
    
    def calculate_indicators(self, df, interval):
        """Calculate indicators adjusted for timeframe"""
        indicators = {}
        
        # Adjust indicator periods based on timeframe
        if interval in ['1m', '2m', '5m']:
            macd_fast, macd_slow, macd_signal = 6, 13, 4
            adx_period = 7
            lookback = 20
        elif interval in ['15m', '30m']:
            macd_fast, macd_slow, macd_signal = 8, 17, 6
            adx_period = 10
            lookback = 30
        elif interval in ['1h']:
            macd_fast, macd_slow, macd_signal = 12, 26, 9
            adx_period = 14
            lookback = 50
        else:
            macd_fast, macd_slow, macd_signal = 12, 26, 9
            adx_period = 14
            lookback = 50
        
        try:
            # MACD with adjusted periods
            macd_data = ta.macd(df['Close'], fast=macd_fast, slow=macd_slow, signal=macd_signal)
            if macd_data is not None and not macd_data.empty:
                macd_col = f'MACD_{macd_fast}_{macd_slow}_{macd_signal}'
                signal_col = f'MACDs_{macd_fast}_{macd_slow}_{macd_signal}'
                hist_col = f'MACDh_{macd_fast}_{macd_slow}_{macd_signal}'
                
                indicators['MACD'] = {
                    'macd': macd_data[macd_col].iloc[-1] if macd_col in macd_data.columns else None,
                    'signal': macd_data[signal_col].iloc[-1] if signal_col in macd_data.columns else None,
                    'histogram': macd_data[hist_col].iloc[-1] if hist_col in macd_data.columns else None,
                    'settings': f"({macd_fast},{macd_slow},{macd_signal})"
                }
                
                if indicators['MACD']['macd'] and indicators['MACD']['signal']:
                    indicators['MACD']['crossover'] = 'bullish' if indicators['MACD']['macd'] > indicators['MACD']['signal'] else 'bearish'
        
            # ADX with adjusted period
            adx_data = ta.adx(df['High'], df['Low'], df['Close'], length=adx_period)
            if adx_data is not None and not adx_data.empty:
                indicators['ADX'] = {
                    'adx': adx_data[f'ADX_{adx_period}'].iloc[-1],
                    'di_plus': adx_data[f'DMP_{adx_period}'].iloc[-1],
                    'di_minus': adx_data[f'DMN_{adx_period}'].iloc[-1],
                    'period': adx_period
                }
        
            # OBV
            obv_data = ta.obv(df['Close'], df['Volume'])
            if obv_data is not None and len(obv_data) > 1:
                indicators['OBV'] = {
                    'current': obv_data.iloc[-1],
                    'previous': obv_data.iloc[-2],
                    'trend': 'bullish' if obv_data.iloc[-1] > obv_data.iloc[-2] else 'bearish'
                }
        
            # RSI with adjusted period
            rsi_period = max(7, adx_period)
            rsi_data = ta.rsi(df['Close'], length=rsi_period)
            if rsi_data is not None:
                indicators['RSI'] = {
                    'value': rsi_data.iloc[-1],
                    'period': rsi_period,
                    'overbought': rsi_data.iloc[-1] > 70,
                    'oversold': rsi_data.iloc[-1] < 30
                }
        
            # Moving averages
            sma_20 = ta.sma(df['Close'], length=min(20, len(df)//2))
            sma_50 = ta.sma(df['Close'], length=min(50, len(df)//2))
            
            if sma_20 is not None and sma_50 is not None:
                indicators['Moving_Averages'] = {
                    'sma_20': sma_20.iloc[-1],
                    'sma_50': sma_50.iloc[-1],
                    'trend': 'bullish' if sma_20.iloc[-1] > sma_50.iloc[-1] else 'bearish'
                }
        
        except Exception as e:
            print(f"Error calculating indicators: {e}")
        
        return indicators
    
    def analyze_timeframe_signals(self, indicators, interval):
        """Analyze signals specific to timeframe"""
        analysis = {
            'timeframe': interval,
            'signals': [],
            'strength': 'neutral',
            'recommendation': 'hold'
        }
        
        bullish_signals = 0
        bearish_signals = 0
        
        # MACD analysis
        if 'MACD' in indicators and indicators['MACD'].get('crossover'):
            if indicators['MACD']['crossover'] == 'bullish':
                bullish_signals += 1
                analysis['signals'].append(f"MACD bullish crossover {indicators['MACD']['settings']}")
            else:
                bearish_signals += 1
                analysis['signals'].append(f"MACD bearish crossover {indicators['MACD']['settings']}")
        
        # ADX analysis
        if 'ADX' in indicators:
            adx_val = indicators['ADX']['adx']
            if adx_val > 25:
                if indicators['ADX']['di_plus'] > indicators['ADX']['di_minus']:
                    bullish_signals += 2
                    analysis['signals'].append(f"Strong bullish trend (ADX: {adx_val:.1f})")
                else:
                    bearish_signals += 2
                    analysis['signals'].append(f"Strong bearish trend (ADX: {adx_val:.1f})")
        
        # RSI analysis
        if 'RSI' in indicators:
            rsi_val = indicators['RSI']['value']
            if rsi_val > 70:
                bearish_signals += 1
                analysis['signals'].append(f"RSI overbought ({rsi_val:.1f})")
            elif rsi_val < 30:
                bullish_signals += 1
                analysis['signals'].append(f"RSI oversold ({rsi_val:.1f})")
        
        # Determine overall strength and recommendation
        total_signals = bullish_signals + bearish_signals        
        if total_signals > 0:
            bullish_pct = (bullish_signals / total_signals) * 100
            
            if bullish_pct >= 75:
                analysis['strength'] = 'strong_bullish'
                analysis['recommendation'] = 'buy'
            elif bullish_pct >= 60:
                analysis['strength'] = 'bullish'
                analysis['recommendation'] = 'buy'
            elif bullish_pct >= 40:
                analysis['strength'] = 'neutral'
                analysis['recommendation'] = 'hold'
            elif bullish_pct >= 25:
                analysis['strength'] = 'bearish'
                analysis['recommendation'] = 'sell'
            else:
                analysis['strength'] = 'strong_bearish'
                analysis['recommendation'] = 'sell'
        
        return analysis
    
    def print_multi_timeframe_analysis(self, symbol, results):
        """Print comprehensive multi-timeframe analysis"""
        print(f"\n{'='*80}")
        print(f"MULTI-TIMEFRAME ANALYSIS: {symbol}")
        print(f"{'='*80}")
        print(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Summary table
        print(f"\n📊 TIMEFRAME SUMMARY:")
        print(f"{'Timeframe':<10} {'Price':<10} {'MACD':<12} {'ADX':<8} {'RSI':<8} {'Strength':<15} {'Rec':<6}")
        print("-" * 80)
        
        for timeframe, data in results.items():
            if 'indicators' in data and 'timeframe_analysis' in data:
                indicators = data['indicators']
                analysis = data['timeframe_analysis']
                
                # Get current price
                current_price = data['data']['Close'].iloc[-1]
                
                # Format MACD
                macd_str = "N/A"
                if 'MACD' in indicators and indicators['MACD'].get('crossover'):
                    cross = indicators['MACD']['crossover']
                    macd_str = f"{cross[:4]}"  # bull/bear
                
                # Format ADX
                adx_str = "N/A"
                if 'ADX' in indicators:
                    adx_val = indicators['ADX']['adx']
                    adx_str = f"{adx_val:.1f}"
                
                # Format RSI
                rsi_str = "N/A"
                if 'RSI' in indicators:
                    rsi_val = indicators['RSI']['value']
                    rsi_str = f"{rsi_val:.1f}"
                
                # Format strength
                strength = analysis['strength'].replace('_', ' ').title()
                recommendation = analysis['recommendation'].upper()
                
                print(f"{timeframe:<10} ${current_price:<9.2f} {macd_str:<12} {adx_str:<8} {rsi_str:<8} {strength:<15} {recommendation:<6}")
        
        # Detailed analysis for each timeframe
        for timeframe, data in results.items():
            if 'indicators' in data and 'timeframe_analysis' in data:
                print(f"\n🔍 DETAILED ANALYSIS - {timeframe.upper()} TIMEFRAME:")
                
                indicators = data['indicators']
                analysis = data['timeframe_analysis']
                df = data['data']
                
                print(f"Data Points: {len(df)} candles")
                print(f"Price Range: ${df['Low'].min():.2f} - ${df['High'].max():.2f}")
                print(f"Current Price: ${df['Close'].iloc[-1]:.2f}")
                
                # Detailed indicators
                if 'MACD' in indicators:
                    macd = indicators['MACD']
                    if macd.get('macd') and macd.get('signal'):
                        print(f"MACD {macd['settings']}: {macd['macd']:.4f} | Signal: {macd['signal']:.4f} | Cross: {macd.get('crossover', 'N/A')}")
                
                if 'ADX' in indicators:
                    adx = indicators['ADX']
                    print(f"ADX({adx['period']}): {adx['adx']:.2f} | +DI: {adx['di_plus']:.2f} | -DI: {adx['di_minus']:.2f}")
                
                if 'RSI' in indicators:
                    rsi = indicators['RSI']
                    status = "Overbought" if rsi['overbought'] else "Oversold" if rsi['oversold'] else "Normal"
                    print(f"RSI({rsi['period']}): {rsi['value']:.2f} ({status})")
                
                if 'OBV' in indicators:
                    obv = indicators['OBV']
                    print(f"OBV: {obv['current']:,.0f} | Trend: {obv['trend']}")
                
                if 'Moving_Averages' in indicators:
                    ma = indicators['Moving_Averages']
                    print(f"SMA20: ${ma['sma_20']:.2f} | SMA50: ${ma['sma_50']:.2f} | Trend: {ma['trend']}")
                
                # Signals
                if analysis['signals']:
                    print("Signals:")
                    for signal in analysis['signals']:
                        print(f"  • {signal}")
                
                print(f"Overall: {analysis['strength'].replace('_', ' ').title()} → {analysis['recommendation'].upper()}")
    
    def get_timeframe_consensus(self, results):
        """Get consensus across all timeframes"""
        if not results:
            return None
        
        recommendations = []
        strengths = []
        
        for timeframe, data in results.items():
            if 'timeframe_analysis' in data:
                analysis = data['timeframe_analysis']
                recommendations.append(analysis['recommendation'])
                strengths.append(analysis['strength'])
        
        # Count recommendations
        buy_count = recommendations.count('buy')
        sell_count = recommendations.count('sell')
        hold_count = recommendations.count('hold')
        
        total = len(recommendations)
        
        consensus = {
            'total_timeframes': total,
            'buy_signals': buy_count,
            'sell_signals': sell_count,
            'hold_signals': hold_count,
            'buy_percentage': (buy_count / total) * 100 if total > 0 else 0,
            'sell_percentage': (sell_count / total) * 100 if total > 0 else 0,
            'hold_percentage': (hold_count / total) * 100 if total > 0 else 0
        }
        
        # Determine overall consensus
        if consensus['buy_percentage'] >= 60:
            consensus['overall_recommendation'] = 'BUY'
            consensus['confidence'] = 'High' if consensus['buy_percentage'] >= 75 else 'Medium'
        elif consensus['sell_percentage'] >= 60:
            consensus['overall_recommendation'] = 'SELL'
            consensus['confidence'] = 'High' if consensus['sell_percentage'] >= 75 else 'Medium'
        else:
            consensus['overall_recommendation'] = 'HOLD'
            consensus['confidence'] = 'Low'
        
        return consensus

# Keep all the existing analyzer classes with Alpaca integration
class AdvancedFibonacciAnalyzer:
    def __init__(self):
        self.fib_levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.414, 1.618]
        self.fib_names = ['0%', '23.6%', '38.2%', '50%', '61.8%', '78.6%', '100%', '127.2%', '141.4%', '161.8%']
    
    def find_swing_points(self, df, lookback=15):
      
        """Find swing highs and lows using peak detection"""
        highs = df['High'].values
        lows = df['Low'].values
        
        # Find peaks (swing highs)
        high_peaks, _ = find_peaks(highs, distance=lookback)
        # Find troughs (swing lows) - invert the data
        low_peaks, _ = find_peaks(-lows, distance=lookback)
        
        swing_highs = [(df.index[i], highs[i]) for i in high_peaks]
        swing_lows = [(df.index[i], lows[i]) for i in low_peaks]
        
        return swing_highs, swing_lows
    
    def calculate_fibonacci_retracements(self, df, period_days=60, auto_detect=True):
        # print("printttt df", df)
        """Calculate Fibonacci retracements with multiple methods"""
        recent_data = df.tail(period_days)
          # Adjust period based on interval
        # if recent_data in ['1m', '2m']:
        #     period = '5d'
        # elif recent_data in ['5m', '15m', '30m']:
        #     period = '10d'
        # elif recent_data in ['60m', '1h']:
        #     period = '3 mo'
        # elif intrecent_dataerval in ['1d', '1wk']:
        #     period = '2y'
        # else:
        #     period = base_period
        
        if auto_detect:
            # Method 1: Auto-detect swing points
            swing_highs, swing_lows = self.find_swing_points(recent_data, lookback=15)
            
            if swing_highs and swing_lows:
                # Get the most recent significant high and low
                recent_high = max(swing_highs, key=lambda x: x[1])
                recent_low = min(swing_lows, key=lambda x: x[1])
                
                high_price = recent_high[1]
                low_price = recent_low[1]
                high_date = recent_high[0]
                low_date = recent_low[0]
            else:
                # Fallback to simple high/low
                high_price = recent_data['High'].max()
                low_price = recent_data['Low'].min()
                high_date = recent_data['High'].idxmax()
                low_date = recent_data['Low'].idxmin()
        else:
            # Method 2: Simple high/low approach
            high_price = recent_data['High'].max()
            low_price = recent_data['Low'].min()
            high_date = recent_data['High'].idxmax()
            low_date = recent_data['Low'].idxmin()
        
        # Determine if we're in uptrend or downtrend
        trend_direction = "uptrend" if high_date > low_date else "downtrend"
        
        # Calculate Fibonacci levels
        price_range = high_price - low_price
        current_price = df['Close'].iloc[-1]
        
        fib_levels = {}
        for i, level in enumerate(self.fib_levels):
            if trend_direction == "uptrend":
                # Retracement from high
                fib_price = high_price - (price_range * level)
            else:
                # Extension from low
                fib_price = low_price + (price_range * level)
            
            fib_levels[self.fib_names[i]] = {
                'price': fib_price,
                'distance_from_current': abs(current_price - fib_price),
                'percentage_from_current': ((fib_price - current_price) / current_price) * 100
            }
        
        # Find nearest support and resistance levels
        nearest_support = None
        nearest_resistance = None
        
        for name, data in fib_levels.items():
            price = data['price']
            if price < current_price:
                if nearest_support is None or price > nearest_support['price']:
                    nearest_support = {'level': name, 'price': price}
            elif price > current_price:
                if nearest_resistance is None or price < nearest_resistance['price']:
                    nearest_resistance = {'level': name, 'price': price}
        
        return {
            'trend_direction': trend_direction,
            'high_price': high_price,
            'low_price': low_price,
            'high_date': high_date,
            'low_date': low_date,
            'current_price': current_price,
            'price_range': price_range,
            'fib_levels': fib_levels,
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance,
            'key_levels': {
                '23.6%': fib_levels['23.6%']['price'],
                '38.2%': fib_levels['38.2%']['price'],
                '50%': fib_levels['50%']['price'],
                '61.8%': fib_levels['61.8%']['price'],
                '78.6%': fib_levels['78.6%']['price'],
                '100%': fib_levels['100%']['price'],
                '127.2%': fib_levels['127.2%']['price'],
                '141.4%': fib_levels['141.4%']['price'],
                '161.8%': fib_levels['161.8%']['price']
            }
        }

class ElliottWaveAnalyzer:
    def __init__(self):
        self.wave_patterns = {
            'impulse': [1, 2, 3, 4, 5],
            'corrective': ['A', 'B', 'C']
        }
    
    def detect_wave_structure(self, df, lookback=200):
        """Detect potential Elliott Wave structure"""
        recent_data = df.tail(lookback)
        
        # Find significant swing points
        highs = recent_data['High'].values
        lows = recent_data['Low'].values
        
        # Find peaks and troughs
        high_peaks, _ = find_peaks(highs, distance=5, prominence=np.std(highs)*0.5)
        low_peaks, _ = find_peaks(-lows, distance=5, prominence=np.std(lows)*0.5)
        
        # Combine and sort all turning points
        all_points = []
        for i in high_peaks:
            all_points.append((recent_data.index[i], highs[i], 'high'))
        for i in low_peaks:
            all_points.append((recent_data.index[i], lows[i], 'low'))
        
        # Sort by date
        all_points.sort(key=lambda x: x[0])
        
        if len(all_points) < 5:
            return {
                'wave_count': len(all_points),
                'pattern': 'insufficient_data',
                'confidence': 0,
                'description': 'Need at least 5 turning points for wave analysis'
            }
        
        # Analyze the pattern
        wave_analysis = self.analyze_wave_pattern(all_points, recent_data)
        print ('Wave Anaysis', wave_analysis)
        return wave_analysis
    
    def analyze_wave_pattern(self, points, df):
        """Analyze the wave pattern from turning points"""
        if len(points) < 5:
            return {'pattern': 'insufficient_data', 'confidence': 0}
        
        # Take the last 5 significant points
        last_5_points = points[-5:]
        
        # Determine overall trend
        first_price = last_5_points[0][1]
        last_price = last_5_points[-1][1]
        overall_trend = "bullish" if last_price > first_price else "bearish"

         # Check for impulse wave characteristics
        impulse_confidence = self.check_impulse_pattern(last_5_points)
        corrective_confidence = self.check_corrective_pattern(last_5_points)
        
        # Determine the most likely pattern
        if impulse_confidence > corrective_confidence and impulse_confidence > 40:
            pattern = 'impulse'
            confidence = impulse_confidence
            description = f"Potential {overall_trend} impulse wave structure detected"
            next_expectation = self.predict_next_wave_impulse(last_5_points, overall_trend)
        elif corrective_confidence > 40:
            pattern = 'corrective'
            confidence = corrective_confidence
            description = f"Potential corrective wave structure detected"
            next_expectation = self.predict_next_wave_corrective(last_5_points, overall_trend)
        else:
            pattern = 'unclear'
            confidence = max(impulse_confidence, corrective_confidence)
            description = "Wave structure is unclear or developing"
            next_expectation = "Monitor for clearer pattern development"
        
        return {
            'pattern': pattern,
            'trend': overall_trend,
            'confidence': confidence,
            'description': description,
            'next_expectation': next_expectation,
            'wave_points': last_5_points,
            'impulse_score': impulse_confidence,
            'corrective_score': corrective_confidence
        }
    
    def check_impulse_pattern(self, points):
        """Check if points match impulse wave characteristics"""
        if len(points) < 5:
            return 0
        
        confidence = 0
        
        # Basic impulse rules:
        # 1. Wave 3 cannot be the shortest
        # 2. Wave 2 cannot retrace more than 100% of wave 1
        # 3. Wave 4 cannot overlap with wave 1 price territory
        
        try:
            # Calculate wave magnitudes
            wave1 = abs(points[1][1] - points[0][1])
            wave2 = abs(points[2][1] - points[1][1])
            wave3 = abs(points[3][1] - points[2][1])
            wave4 = abs(points[4][1] - points[3][1])
            
            waves = [wave1, wave2, wave3, wave4]
            
            # Rule 1: Wave 3 not shortest (among 1, 3, 5)
            if len(points) >= 5:
                if wave3 >= wave1:  # Simplified check
                    confidence += 30
            
            # Rule 2: Wave 2 retracement check
            wave2_retrace = wave2 / wave1 if wave1 > 0 else 0
            if 0.3 <= wave2_retrace <= 0.8:  # Typical retracement range
                confidence += 25
            
            # Rule 3: No overlap between waves 1 and 4
            wave1_end = points[1][1]
            wave4_end = points[4][1] if len(points) > 4 else points[3][1]
            
            # Check trend consistency
            trend_changes = 0
            for i in range(1, len(points)):
                if i % 2 == 1:  # Odd waves should go in main trend direction
                    if (points[i][1] > points[i-1][1]) == (points[-1][1] > points[0][1]):
                        confidence += 10
                else:  # Even waves should be corrections
                    if (points[i][1] < points[i-1][1]) == (points[-1][1] > points[0][1]):
                        confidence += 10
            
        except (ZeroDivisionError, IndexError):
            confidence = 0
        
        return min(confidence, 100)
    
    def check_corrective_pattern(self, points):
        """Check if points match corrective wave characteristics (ABC)"""
        if len(points) < 3:
            return 0
        
        confidence = 0
        
        try:
            # Take last 3 points for ABC pattern
            last_3 = points[-3:]
            
            # Calculate wave magnitudes
            wave_a = abs(last_3[1][1] - last_3[0][1])
            wave_b = abs(last_3[2][1] - last_3[1][1]) if len(last_3) > 2 else 0
            
            # Check for typical corrective ratios
            if wave_b > 0:
                b_to_a_ratio = wave_b / wave_a
                
                # Wave B typically retraces 38-78% of wave A
                if 0.3 <= b_to_a_ratio <= 0.8:
                    confidence += 40
                
                # Check alternation (different wave structures)
                confidence += 20
            
            # Check overall corrective nature (against main trend)
            if len(points) >= 5:
                main_trend_up = points[-1][1] > points[-5][1]
                correction_up = last_3[-1][1] > last_3[0][1]
                
                if main_trend_up != correction_up:  # Correction against trend
                    confidence += 30
        
        except (ZeroDivisionError, IndexError):
            confidence = 0
        
        return min(confidence, 100)
    
    def predict_next_wave_impulse(self, points, trend):
        """Predict next wave in impulse sequence"""
        wave_count = len(points)
        
        if wave_count == 5:
            return f"Impulse sequence may be complete. Expect corrective phase against {trend} trend."
        elif wave_count < 5:
            next_wave = wave_count + 1
            if next_wave % 2 == 1:  # Odd waves (1, 3, 5)
                return f"Expect wave {next_wave} to continue {trend} trend"
            else:  # Even waves (2, 4)
                return f"Expect wave {next_wave} correction against {trend} trend"
        else:
            return "Monitor for new impulse sequence or corrective phase"
    
    def predict_next_wave_corrective(self, points, trend):
        """Predict next wave in corrective sequence"""
        return f"Corrective phase may continue. Watch for completion and resumption of main trend."

# Example usage and testing functions
def run_comprehensive_analysis_example():
    """Example of how to use the comprehensive analyzer"""
    
    # Initialize the analyzer (you'll need to provide your Alpaca API credentials)
    analyzer = ComprehensiveMultiTimeframeAnalyzer()
    api_key = "PKYJLOK4LZBY56NZKXZLNSG665"
    secret_key = "4VVHMnrYEqVv4Jd1oMZMow15DrRVn5p8VD7eEK6TjYZ1"
    analyzer.set_credentials(api_key,secret_key)
    # For demonstration, let's analyze a popular stock
    symbol = "ASML"
    timeframes = ['5m','15m','30m', '1h', '1d', '1wk','1mo']
    
    print(f"🚀 Running comprehensive analysis for {symbol}")
    print("=" * 80)
    
    try:
        # Run the analysis
        results = analyzer.analyze_comprehensive_multi_timeframe(
            symbol=symbol,
            timeframes=timeframes,
            base_period='5y'
        )
        
        # Print the results
        analyzer.print_comprehensive_analysis(symbol, results)
        
        return results
        
    except Exception as e:
        print(f"Error running analysis: {e}")
        print("Make sure you have set up your Alpaca API credentials")
        return None

def setup_analyzer_with_credentials():
    """Helper function to set up analyzer with credentials"""
    print("Setting up Comprehensive Stock Analyzer...")
    print("You'll need Alpaca API credentials for live data.")
    print("Get them from: https://alpaca.markets/")
    
    # You would replace these with your actual credentials
    api_key = "YOUR_ALPACA_API_KEY"
    secret_key = "YOUR_ALPACA_SECRET_KEY"
    
    analyzer = ComprehensiveMultiTimeframeAnalyzer(api_key=api_key, secret_key=secret_key)
    
    return analyzer

def demo_fibonacci_analysis():
    """Demonstrate Fibonacci analysis capabilities"""
    print("\n🌀 Fibonacci Analysis Demo")
    print("-" * 40)
    
    # This would work with real data
    # For demo purposes, we'll show the structure
    fib_analyzer = AdvancedFibonacciAnalyzer()
    
    print("Fibonacci Analyzer initialized with levels:")
    for i, (level, name) in enumerate(zip(fib_analyzer.fib_levels, fib_analyzer.fib_names)):
        print(f"  {name}: {level}")
    
    print("\nFibonacci analysis provides:")
    print("• Automatic swing point detection")
    print("• Trend direction identification")
    print("• Key support/resistance levels")
    print("• Retracement and extension targets")

def demo_elliott_wave_analysis():
    """Demonstrate Elliott Wave analysis capabilities"""
    print("\n🌊 Elliott Wave Analysis Demo")
    print("-" * 40)
    
    wave_analyzer = ElliottWaveAnalyzer()
    
    print("Elliott Wave Analyzer features:")
    print("• Automatic wave pattern detection")
    print("• Impulse vs Corrective wave identification")
    print("• Wave count and structure analysis")
    print("• Next wave predictions")
    print("• Confidence scoring")
    
    print(f"\nSupported patterns: {list(wave_analyzer.wave_patterns.keys())}")

if __name__ == "__main__":
    run_comprehensive_analysis_example()
    # multi_trader = ComprehensiveMultiTimeframeAnalyzer(
    #     symbols=['DUOL'],
    #     timeframes=['1m','5m', '15m', '30m', '1h', '1d', '1wk','1mo']  # Removed 90m, 5d, 1wk for Alpaca compatibility
    # )
    # api_key = "PKYJLOK4LZBY56NZKXZLNSG665"
    # secret_key = "4VVHMnrYEqVv4Jd1oMZMow15DrRVn5p8VD7eEK6TjYZ1"
    # multi_trader.set_credentials(api_key=api_key, secret_key=secret_key)
    # run_comprehensive_analysis_example()

    # print("🚀 Comprehensive Stock Analysis System")
    # print("=" * 50)
    
    # print("\nThis system provides:")
    # print("✅ Multi-timeframe technical analysis")
    # print("✅ Advanced Fibonacci retracements")
    # print("✅ Elliott Wave pattern detection")
    # print("✅ Comprehensive trading signals")
    # print("✅ Risk/reward calculations")
    # print("✅ Multi-timeframe consensus")
    
    # print("\n📊 Features:")
    # print("• MACD, ADX, RSI, OBV, Bollinger Bands")
    # print("• Automatic swing point detection")
    # print("• Wave pattern recognition")
    # print("• Support/resistance identification")
    # print("• Trading signal generation")
    # print("• Timeframe correlation analysis")
    
    # Run demos
    # demo_fibonacci_analysis()
    # demo_elliott_wave_analysis()
    # 
    # print("\n" + "=" * 50)
    # print("To use with real data:")
    # print("1. Get Alpaca API credentials")
    # print("2. Initialize analyzer with credentials")
    # print("3. Run analysis on your chosen symbols")
    # print("4. Review multi-timeframe results")
    # print("5. Make informed trading decisions")