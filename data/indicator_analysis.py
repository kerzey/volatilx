import numpy as np
import pandas as pd
import pandas_ta as ta
from scipy.signal import find_peaks

def get_params_by_timeframe(timeframe):
    # You can expand and fine-tune these mappings as needed
    tf_settings = {
        '1m':   {'macd': (6, 13, 4), 'adx': 7,  'sma': [10, 20], 'ema': [10, 20], 'rsi': 7,  'cci': 10, 'atr': 7,  'stoch': (10, 3), 'roc': 7, 'lr': 10, 'donchian': 10, 'bb': 10, 'hv': 10},
        '5m':   {'macd': (8, 17, 6), 'adx': 10, 'sma': [15, 30], 'ema': [15, 30], 'rsi': 10, 'cci': 15, 'atr': 10, 'stoch': (12, 3), 'roc': 10, 'lr': 12, 'donchian': 12, 'bb': 12, 'hv': 12},
        '15m':  {'macd': (10, 20, 7), 'adx': 14,'sma': [20, 50], 'ema': [20, 50], 'rsi': 14, 'cci': 20, 'atr': 14, 'stoch': (14, 5), 'roc': 14, 'lr': 14, 'donchian': 14, 'bb': 14, 'hv': 14},
        '1h':   {'macd': (12, 26, 9), 'adx': 14,'sma': [20, 50], 'ema': [20, 50], 'rsi': 14, 'cci': 20, 'atr': 14, 'stoch': (14, 5), 'roc': 14, 'lr': 14, 'donchian': 14, 'bb': 14, 'hv': 14},
        'day':  {'macd': (12, 26, 9), 'adx': 14,'sma': [20, 200],'ema': [20, 200],'rsi': 14, 'cci': 20, 'atr': 14, 'stoch': (14, 5), 'roc': 14, 'lr': 14, 'donchian': 20, 'bb': 20, 'hv': 20},
    }
    return tf_settings.get(timeframe, tf_settings['1h'])

# def calculate_fibonacci_levels(df, lookback=100):
#     high = df['high'][-lookback:].max()
#     low = df['low'][-lookback:].min()
#     diff = high - low
#     levels = {
#         "0.0%": high,
#         "23.6%": high - 0.236 * diff,
#         "38.2%": high - 0.382 * diff,
#         "50.0%": high - 0.5 * diff,
#         "61.8%": high - 0.618 * diff,
#         "78.6%": high - 0.786 * diff,
#         "100.0%": low
#     }
#     return levels

# def naive_elliott_wave_analysis(df, pivots=5):
#     pivots_high = df['high'].nlargest(pivots)
#     pivots_low = df['low'].nsmallest(pivots)
#     return {"pivot_highs": list(pivots_high), "pivot_lows": list(pivots_low)}

class AdvancedFibonacciAnalyzer:
    def __init__(self):
        # Key Fibonacci retracement and extension ratios
        self.fib_levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.414, 1.618]
        self.fib_names = [
            '0%', '23.6%', '38.2%', '50%', '61.8%',
            '78.6%', '100%', '127.2%', '141.4%', '161.8%'
        ]
    
    def find_swing_points(self, df, lookback=10):
        """Detect swing highs and lows via peak finding."""
        highs = df['high'].values
        lows = df['low'].values

        # Find swing highs
        high_peaks, _ = find_peaks(highs, distance=lookback)
        # Find swing lows (troughs)
        low_peaks, _ = find_peaks(-lows, distance=lookback)

        swing_highs = [(df.index[i], highs[i]) for i in high_peaks]
        swing_lows = [(df.index[i], lows[i]) for i in low_peaks]

        return swing_highs, swing_lows

    def calculate_fibonacci_retracements(self, df, period_days=60, auto_detect=True):
        """Calculate Fibonacci retracements using swing analysis or high/low bounds."""
        recent_data = df.tail(period_days)

        # Swing point method
        if auto_detect:
            swing_highs, swing_lows = self.find_swing_points(recent_data, lookback=5)
            if swing_highs and swing_lows:
                # Most recent/highest swing high and lowest swing low
                recent_high = max(swing_highs, key=lambda x: x[1])
                recent_low = min(swing_lows, key=lambda x: x[1])
                high_price = recent_high[1]
                low_price = recent_low[1]
                high_date = recent_high[0]
                low_date = recent_low[0]
            else:
                # Fallback to data-wide high/low
                high_price = recent_data['high'].max()
                low_price = recent_data['low'].min()
                high_date = recent_data['high'].idxmax()
                low_date = recent_data['low'].idxmin()
        else:
            # Simple high/low approach
            high_price = recent_data['high'].max()
            low_price = recent_data['low'].min()
            high_date = recent_data['high'].idxmax()
            low_date = recent_data['low'].idxmin()

        # Determine trend direction (did high come after low?)
        trend_direction = "uptrend" if high_date > low_date else "downtrend"
        price_range = high_price - low_price
        current_price = df['close'].iloc[-1]

        # Calculate Fib levels based on trend
        fib_levels = {}
        for i, level in enumerate(self.fib_levels):
            if trend_direction == "uptrend":
                fib_price = high_price - (price_range * level)
            else:
                fib_price = low_price + (price_range * level)
            fib_levels[self.fib_names[i]] = {
                'price': fib_price,
                'distance_from_current': abs(current_price - fib_price),
                'percentage_from_current': ((fib_price - current_price) / current_price) * 100
            }

        # Find nearest support (below) and resistance (above) relative to current price
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

        # Collect key retracement levels for fast access
        key_levels = {
            '38.2%': fib_levels['38.2%']['price'],
            '50%': fib_levels['50%']['price'],
            '61.8%': fib_levels['61.8%']['price']
        }

        # Complete analysis dict
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
            'key_levels': key_levels
        }

class ElliottWaveAnalyzer:
    def __init__(self):
        self.wave_patterns = {
            'impulse': [1, 2, 3, 4, 5],
            'corrective': ['A', 'B', 'C']
        }

    def detect_wave_structure(self, df, lookback=100):
        """
        Detect potential Elliott Wave structure using peak/trough detection.
        Returns a dict with pattern, trend, confidence, latest wave points, and expectation hints.
        """
        recent_data = df.tail(lookback).copy()
        highs = recent_data['high'].values
        lows = recent_data['low'].values

        # Detect peaks (swing highs) and troughs (swing lows)
        high_peaks, _ = find_peaks(highs, distance=5, prominence=np.std(highs) * 0.5)
        low_peaks, _ = find_peaks(-lows, distance=5, prominence=np.std(lows) * 0.5)

        # Gather swing points with their index, value, and type
        turning_points = []
        for i in high_peaks:
            turning_points.append((recent_data.index[i], highs[i], 'high'))
        for i in low_peaks:
            turning_points.append((recent_data.index[i], lows[i], 'low'))

        # Sort by chronological order
        turning_points.sort(key=lambda x: x[0])

        # Not enough points for meaningful analysis
        if len(turning_points) < 5:
            return {
                'wave_count': len(turning_points),
                'pattern': 'insufficient_data',
                'confidence': 0,
                'description': 'Need at least 5 distinct turning points for Elliott Wave analysis.'
            }

        # Analyze the latest 5 turning points to infer pattern
        return self.analyze_wave_pattern(turning_points, recent_data)

    def analyze_wave_pattern(self, points, df):
        """Analyze possible wave pattern among the most recent swing points."""
        if len(points) < 5:
            return {'pattern': 'insufficient_data', 'confidence': 0}

        last_5 = points[-5:]
        first_price = last_5[0][1]
        last_price = last_5[-1][1]
        trend = "bullish" if last_price > first_price else "bearish"

        impulse_score = self.check_impulse_pattern(last_5)
        corrective_score = self.check_corrective_pattern(last_5)

        if impulse_score > corrective_score:
            pattern = "impulse"
            confidence = impulse_score
            next_exp = self.predict_next_impulse_move(last_5, trend)
        else:
            pattern = "corrective"
            confidence = corrective_score
            next_exp = self.predict_next_corrective_move(last_5, trend)

        return {
            'pattern': pattern,
            'trend': trend,
            'confidence': confidence,
            'wave_points': last_5,
            'next_expectation': next_exp,
            'impulse_score': impulse_score,
            'corrective_score': corrective_score
        }

    def check_impulse_pattern(self, points):
        """Check if last 5 points follow a typical impulse (1-2-3-4-5) structure."""
        if len(points) != 5:
            return 0
        score = 0

        # Wave lengths between successive points
        waves = [abs(points[i][1] - points[i-1][1]) for i in range(1, 5)]

        # Wave 3 is often longest
        if waves[2] == max(waves):
            score += 30
        elif waves[2] > np.mean(waves):
            score += 15

        # Wave 2 retrace < 100% of wave 1
        if waves[1] < waves[0]:
            score += 20

        # Wave 4 doesn't overlap wave 1
        wave1_end = points[1][1]
        wave4_end = points[4][1]
        if (points[0][1] < wave1_end and wave4_end > points[0][1]) or (points[0][1] > wave1_end and wave4_end < points[0][1]):
            score += 25

        return min(score, 100)

    def check_corrective_pattern(self, points):
        """Check if last 3 of 5 points could be an A-B-C (corrective) structure."""
        if len(points) < 3:
            return 0
        score = 0

        abc = points[-3:]
        a_wave = abs(abc[1][1] - abc[0][1])
        c_wave = abs(abc[2][1] - abc[1][1])

        # A and C waves similar length
        if a_wave > 0 and c_wave > 0:
            ratio = min(a_wave, c_wave) / max(a_wave, c_wave)
            if ratio > 0.8:
                score += 40
            elif ratio > 0.6:
                score += 25

        # B is counter-trend
        a_dir = 1 if abc[1][1] > abc[0][1] else -1
        b_dir = 1 if abc[2][1] > abc[1][1] else -1
        if a_dir != b_dir:
            score += 30

        return min(score, 100)

    def predict_next_impulse_move(self, points, trend):
        """Suggest next likely move after impulse pattern."""
        if len(points) < 5:
            return "Need more data"
        # Usually retracement after 5th wave
        wave_3_val = max(points[2][1], points[4][1]) if trend == "bullish" else min(points[2][1], points[4][1])
        if trend == "bullish":
            return f"Expect correction to {wave_3_val*0.618:.2f} - {wave_3_val*0.382:.2f}"
        else:
            return f"Expect bounce to {wave_3_val*1.382:.2f} - {wave_3_val*1.618:.2f}"

    def predict_next_corrective_move(self, points, trend):
        return "Corrective pattern—look for continuation in primary trend after completion."

# def calculate_all_indicators(df, timeframe='1h'):
#     params = get_params_by_timeframe(timeframe)
#     indicators = pd.DataFrame(index=df.index)

#     # MACD
#     macd = ta.macd(df['close'], fast=params['macd'][0], slow=params['macd'][1], signal=params['macd'][2])
#     indicators = pd.concat([indicators, macd], axis=1)

#     # ADX
#     adx = ta.adx(df['high'], df['low'], df['close'], length=params['adx'])
#     indicators = pd.concat([indicators, adx], axis=1)

#     # RSI
#     rsi = ta.rsi(df['close'], length=params['rsi'])
#     indicators[f'RSI_{params["rsi"]}'] = rsi

#     # OBV
#     obv = ta.obv(df['close'], df['volume'])
#     indicators['OBV'] = obv

#     # SMA, EMA
#     for len_ in params['sma']:
#         indicators[f'SMA_{len_}'] = ta.sma(df['close'], length=len_)
#     for len_ in params['ema']:
#         indicators[f'EMA_{len_}'] = ta.ema(df['close'], length=len_)

#     # Bollinger Bands
#     bb = ta.bbands(df['close'], length=params['bb'], std=2)
#     indicators = pd.concat([indicators, bb], axis=1)

#     # ATR
#     atr = ta.atr(df['high'], df['low'], df['close'], length=params['atr'])
#     indicators[f'ATR_{params["atr"]}'] = atr

#     # Supertrend
#     sup = ta.supertrend(df['high'], df['low'], df['close'],
#                         length=10 if params['atr']<10 else params['atr'], multiplier=3.0)
#     indicators = pd.concat([indicators, sup], axis=1)

#     # Linear Regression Angle
#     linreg = ta.linreg(df['close'], length=params['lr'])
#     indicators[f'LinReg_{params["lr"]}'] = linreg

#     # (Optional) To compute the "angle" in degrees:
#     # angle = arctangent(slope) * (180/pi)  -- but pandas_ta doesn't provide slope by default.
#     # So, you can calculate the difference over window as a proxy slope:
#     slope = df['close'].diff(periods=params['lr']) / params['lr']
#     angle = np.degrees(np.arctan(slope))
#     indicators[f'LinReg_Angle_{params["lr"]}'] = angle

#     # Stochastic Oscillator
#     stoch = ta.stoch(df['high'], df['low'], df['close'],
#                      k=params['stoch'][0], d=params['stoch'][1])
#     indicators = pd.concat([indicators, stoch], axis=1)

#     # CCI
#     cci = ta.cci(df['high'], df['low'], df['close'], length=params['cci'])
#     indicators[f'CCI_{params["cci"]}'] = cci

#     # ROC
#     roc = ta.roc(df['close'], length=params['roc'])
#     indicators[f'ROC_{params["roc"]}'] = roc

#     # Donchian Channel
#     donchian = ta.donchian(df['high'], df['low'], lower_length=params['donchian'], upper_length=params['donchian'])
#     indicators = pd.concat([indicators, donchian], axis=1)

#     # Historical Volatility
#     returns = np.log(df['close'] / df['close'].shift(1))
#     hist_vol = returns.rolling(window=params['hv']).std() * np.sqrt(252)
#     indicators[f'Hist_Volatility_{params["hv"]}'] = hist_vol

#     # Fibonacci & Elliott Wave
#     indicators['Fibonacci'] = [calculate_fibonacci_levels(df[:i+1], lookback=min(params['bb'],100)) for i in range(len(df))]
#     indicators['Elliott_Wave'] = [naive_elliott_wave_analysis(df[:i+1], pivots=5) for i in range(len(df))]

#     return indicators

def calculate_all_indicators(df, timeframe='1h'):
    params = get_params_by_timeframe(timeframe)
    indicators = pd.DataFrame(index=df.index)

    # MACD
    macd = ta.macd(df['close'], fast=params['macd'][0], slow=params['macd'][1], signal=params['macd'][2])
    indicators = pd.concat([indicators, macd], axis=1)

    # ADX
    adx = ta.adx(df['high'], df['low'], df['close'], length=params['adx'])
    indicators = pd.concat([indicators, adx], axis=1)

    # RSI
    rsi = ta.rsi(df['close'], length=params['rsi'])
    indicators[f'RSI_{params["rsi"]}'] = rsi

    # OBV
    obv = ta.obv(df['close'], df['volume'])
    indicators['OBV'] = obv

    # SMA, EMA
    for len_ in params['sma']:
        indicators[f'SMA_{len_}'] = ta.sma(df['close'], length=len_)
    for len_ in params['ema']:
        indicators[f'EMA_{len_}'] = ta.ema(df['close'], length=len_)

    # Bollinger Bands
    bb = ta.bbands(df['close'], length=params['bb'], std=2)
    indicators = pd.concat([indicators, bb], axis=1)

    # ATR
    atr = ta.atr(df['high'], df['low'], df['close'], length=params['atr'])
    indicators[f'ATR_{params["atr"]}'] = atr

    # Supertrend
    sup = ta.supertrend(df['high'], df['low'], df['close'],
                        length=10 if params['atr'] < 10 else params['atr'], multiplier=3.0)
    indicators = pd.concat([indicators, sup], axis=1)

    # Linear Regression Angle
    linreg = ta.linreg(df['close'], length=params['lr'])
    indicators[f'LinReg_{params["lr"]}'] = linreg
    slope = df['close'].diff(periods=params['lr']) / params['lr']
    angle = np.degrees(np.arctan(slope))
    indicators[f'LinReg_Angle_{params["lr"]}'] = angle

    # Stochastic Oscillator
    stoch = ta.stoch(df['high'], df['low'], df['close'],
                     k=params['stoch'][0], d=params['stoch'][1])
    indicators = pd.concat([indicators, stoch], axis=1)

    # CCI
    cci = ta.cci(df['high'], df['low'], df['close'], length=params['cci'])
    indicators[f'CCI_{params["cci"]}'] = cci

    # ROC
    roc = ta.roc(df['close'], length=params['roc'])
    indicators[f'ROC_{params["roc"]}'] = roc

    # Donchian Channel
    donchian = ta.donchian(df['high'], df['low'], lower_length=params['donchian'], upper_length=params['donchian'])
    indicators = pd.concat([indicators, donchian], axis=1)

    # Historical Volatility
    returns = np.log(df['close'] / df['close'].shift(1))
    hist_vol = returns.rolling(window=params['hv']).std() * np.sqrt(252)
    indicators[f'Hist_Volatility_{params["hv"]}'] = hist_vol

    # -------- Advanced Fibonacci & Elliott Wave (new logic) ---------
    fib_analyzer = AdvancedFibonacciAnalyzer()
    elliott_analyzer = ElliottWaveAnalyzer()

    indicators['Fibonacci'] = [
        fib_analyzer.calculate_fibonacci_retracements(
            df.iloc[:i+1],
            period_days=min(len(df.iloc[:i+1]), max(60, params['bb'])),  # Uses last N bars or all if smaller
            auto_detect=True
        )
        for i in range(len(df))
    ]
    indicators['Elliott_Wave'] = [
        elliott_analyzer.detect_wave_structure(
            df.iloc[:i+1],
            lookback=min(len(df.iloc[:i+1]), 50)
        )
        for i in range(len(df))
    ]

    return indicators