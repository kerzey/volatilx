# import numpy as np

# class ComprehensiveSignalGenerator:
#     """
#     Generates trading signals using technical indicators calculated via pandas_ta.
#     Designed for an Alpaca data pipeline, with multi-timeframe support.
#     """

#     def __init__(self, indicator_params=None):
#         """
#         Initialize with custom indicator parameter mapping if needed (optional).
#         """
#         self.indicator_params = indicator_params if indicator_params else {}

#     def extract_macd_crossover(self, df, fast=12, slow=26, signal=9):
#         macd_col = f"MACD_{fast}_{slow}_{signal}"
#         macds_col = f"MACDs_{fast}_{slow}_{signal}"
#         if len(df) < 2:
#             return 'neutral'  # not enough data
#         prev_macd, prev_macds = df.iloc[-2][macd_col], df.iloc[-2][macds_col]
#         curr_macd, curr_macds = df.iloc[-1][macd_col], df.iloc[-1][macds_col]
#         if prev_macd < prev_macds and curr_macd > curr_macds:
#             return 'bullish'
#         elif prev_macd > prev_macds and curr_macd < curr_macds:
#             return 'bearish'
#         else:
#             return 'neutral'

#     def analyze_row(self, row, prev_row=None, params=None):
#         """
#         Analyze a single indicator row, return a signal dict.
#         """
#         # Indicator parameter mapping (fall back to reasonable defaults)
#         macd_fast = params.get("macd_fast", 12)
#         macd_slow = params.get("macd_slow", 26)
#         macd_signal = params.get("macd_signal", 9)
#         adx_len = params.get("adx_len", 14)
#         rsi_len = params.get("rsi_len", 14)
#         atr_len = params.get("atr_len", 14)
#         cci_len = params.get("cci_len", 20)

#         signals = {'entry_signals': [], 'exit_signals': [], 'scores': {'bullish': 0, 'bearish': 0}}
        
#         # MACD logic
#         macd_col = f"MACD_{macd_fast}_{macd_slow}_{macd_signal}"
#         macds_col = f"MACDs_{macd_fast}_{macd_slow}_{macd_signal}"
#         macdh_col = f"MACDh_{macd_fast}_{macd_slow}_{macd_signal}"
#         crossover = "neutral"
#         if prev_row is not None:
#             prev_macd = prev_row.get(macd_col, np.nan)
#             prev_macds = prev_row.get(macds_col, np.nan)
#             curr_macd = row.get(macd_col, np.nan)
#             curr_macds = row.get(macds_col, np.nan)
#             if not np.isnan(prev_macd) and not np.isnan(prev_macds) and not np.isnan(curr_macd) and not np.isnan(curr_macds):
#                 if prev_macd < prev_macds and curr_macd > curr_macds:
#                     crossover = "bullish"
#                 elif prev_macd > prev_macds and curr_macd < curr_macds:
#                     crossover = "bearish"
#         row_macdh = row.get(macdh_col, 0)
#         if crossover == "bullish":
#             signals['entry_signals'].append("MACD bullish crossover")
#             signals['scores']['bullish'] += 2
#         elif crossover == "bearish":
#             signals['exit_signals'].append("MACD bearish crossover")
#             signals['scores']['bearish'] += 2

#         # ADX
#         adx_col = f"ADX_{adx_len}"
#         adx_val = row.get(adx_col, 0)
#         adx_trend = 'neutral'
#         if adx_val > 25:
#             adx_trend = 'strong'
#             # For DI+ and DI-, you may need to extract those columns e.g. "DMP_14", "DMN_14"
#             di_plus = row.get(f"DMP_{adx_len}", 0)
#             di_minus = row.get(f"DMN_{adx_len}", 0)
#             if di_plus > di_minus:
#                 signals['entry_signals'].append("Strong bullish trend (ADX)")
#                 signals['scores']['bullish'] += 2
#             else:
#                 signals['exit_signals'].append("Strong bearish trend (ADX)")
#                 signals['scores']['bearish'] += 2

#         # RSI
#         rsi_col = f"RSI_{rsi_len}"
#         rsi_val = row.get(rsi_col, np.nan)
#         if rsi_val < 30:
#             signals['entry_signals'].append("RSI oversold")
#             signals['scores']['bullish'] += 1
#         elif rsi_val > 70:
#             signals['exit_signals'].append("RSI overbought")
#             signals['scores']['bearish'] += 1

#         # OBV (Momentum)
#         obv = row.get("OBV", 0)
#         if obv > 0:
#             signals['entry_signals'].append("OBV bullish")
#             signals['scores']['bullish'] += 1
#         else:
#             signals['exit_signals'].append("OBV bearish")
#             signals['scores']['bearish'] += 1

#         # SMA/EMA trend
#         sma_fast = row.get("SMA_20", np.nan)
#         sma_slow = row.get("SMA_50", np.nan)
#         if not np.isnan(sma_fast) and not np.isnan(sma_slow):
#             ma_trend = "bullish" if sma_fast > sma_slow else "bearish"
#             if ma_trend == "bullish":
#                 signals['entry_signals'].append("SMA bullish crossover")
#                 signals['scores']['bullish'] += 1
#             else:
#                 signals['exit_signals'].append("SMA bearish crossover")
#                 signals['scores']['bearish'] += 1

#         # Supertrend
#         supertrend_col = "SUPERT_14_3.0"
#         supertrend = row.get(supertrend_col, 0)
#         if supertrend > 0:
#             signals['entry_signals'].append("Supertrend bullish")
#             signals['scores']['bullish'] += 2
#         else:
#             signals['exit_signals'].append("Supertrend bearish")
#             signals['scores']['bearish'] += 2

#         # ATR (Volatility)
#         atr_col = f"ATR_{atr_len}"
#         atr = row.get(atr_col, np.nan)
#         if not np.isnan(atr) and atr > 0.8 * atr:
#             signals['entry_signals'].append("High ATR (volatility)")
#             signals['scores']['bullish'] += 1

#         # Linear Regression Angle (if exists)
#         lr_angle_col = f"LinReg_Angle_{cci_len}"  # use CCI length here, or your param
#         lr_angle = row.get(lr_angle_col, 0)
#         if lr_angle > 20:
#             signals['entry_signals'].append("Strong uptrend (LinReg)")
#             signals['scores']['bullish'] += 1
#         elif lr_angle < -20:
#             signals['exit_signals'].append("Strong downtrend (LinReg)")
#             signals['scores']['bearish'] += 1

#         # Stochastic Oscillator
#         stoch_k = row.get("STOCHk_14_5_3", 0)
#         stoch_d = row.get("STOCHd_14_5_3", 0)
#         if stoch_k < 20 and stoch_d < 20:
#             signals['entry_signals'].append("Stochastic oversold")
#             signals['scores']['bullish'] += 1
#         elif stoch_k > 80 and stoch_d > 80:
#             signals['exit_signals'].append("Stochastic overbought")
#             signals['scores']['bearish'] += 1

#         # CCI
#         cci_col = f"CCI_{cci_len}"
#         cci = row.get(cci_col, np.nan)
#         if cci > 100:
#             signals['entry_signals'].append("CCI bullish")
#             signals['scores']['bullish'] += 1
#         elif cci < -100:
#             signals['exit_signals'].append("CCI bearish")
#             signals['scores']['bearish'] += 1

#         # ROC
#         roc_col = f"ROC_{cci_len}"
#         roc = row.get(roc_col, np.nan)
#         if roc > 5:
#             signals['entry_signals'].append("ROC positive momentum")
#             signals['scores']['bullish'] += 1
#         elif roc < -5:
#             signals['exit_signals'].append("ROC negative momentum")
#             signals['scores']['bearish'] += 1

#         # Donchian Channel
#         donchian_upper = row.get("DCHU_14_14", np.nan)
#         donchian_lower = row.get("DCHL_14_14", np.nan)
#         last_close = row.get("close", np.nan)
#         if not np.isnan(last_close) and not np.isnan(donchian_upper) and not np.isnan(donchian_lower):
#             if last_close >= donchian_upper:
#                 signals['exit_signals'].append("Price breaking upper Donchian")
#                 signals['scores']['bearish'] += 1
#             elif last_close <= donchian_lower:
#                 signals['entry_signals'].append("Price breaking lower Donchian")
#                 signals['scores']['bullish'] += 1

#         # Historical Volatility
#         hist_vol_col = f"Hist_Volatility_{cci_len}"
#         hist_vol = row.get(hist_vol_col, 0)
#         if hist_vol > 0.05:
#             signals['entry_signals'].append("Historically high volatility")
#             signals['scores']['bullish'] += 1

#         # Fibonacci Proximity
#         fib = row.get("Fibonacci", {})
#         if fib and isinstance(fib, dict):
#             if np.isclose(last_close, fib.get("61.8%", -1), atol=0.01 * last_close):
#                 signals['entry_signals'].append("Price near Fibonacci 61.8% retracement")
#                 signals['scores']['bullish'] += 1
#             elif np.isclose(last_close, fib.get("38.2%", -1), atol=0.01 * last_close):
#                 signals['exit_signals'].append("Price near Fibonacci 38.2% retracement")
#                 signals['scores']['bearish'] += 1

#         # Elliott Wave (if exists)
#         elliot = row.get("Elliott_Wave", {})
#         if elliot and isinstance(elliot, dict):
#             if last_close in elliot.get('pivot_lows', []):
#                 signals['entry_signals'].append("Elliott Wave pivot low")
#                 signals['scores']['bullish'] += 1
#             elif last_close in elliot.get('pivot_highs', []):
#                 signals['exit_signals'].append("Elliott Wave pivot high")
#                 signals['scores']['bearish'] += 1

#         # Compute summary bias and strength
#         total = signals['scores']['bullish'] + signals['scores']['bearish']
#         if total == 0:
#             signals['overall_bias'] = "neutral"
#             signals['signal_strength'] = 0
#         else:
#             pct = (signals['scores']['bullish'] / total) * 100
#             signals['overall_bias'] = "bullish" if pct > 65 else "bearish" if pct < 35 else "neutral"
#             signals['signal_strength'] = pct

#         return signals

#     def analyze(self, indicators_df, timeframe_params=None):
#         """
#         Analyze an indicators dataframe across all rows.
#         Returns a list of dicts of signals, one per row.
#         """
#         results = []
#         params = timeframe_params or {}
#         for i in range(len(indicators_df)):
#             row = indicators_df.iloc[i].to_dict()
#             prev_row = indicators_df.iloc[i-1].to_dict() if i > 0 else None
#             signals = self.analyze_row(row, prev_row, params)
#             results.append(signals)
#         return results

#     def last_signal(self, indicators_df, params=None):
#         """
#         Returns the most recent signal dict.
#         """
#         if len(indicators_df) == 0:
#             return {}
#         row = indicators_df.iloc[-1].to_dict()
#         prev_row = indicators_df.iloc[-2].to_dict() if len(indicators_df) > 1 else None
#         return self.analyze_row(row, prev_row, params or {})
    

# # Example usage:
# # signal_gen = ComprehensiveSignalGenerator()
# # signals_list = signal_gen.analyze(indicators_df)
# # last_row_signal = signal_gen.last_signal(indicators_df)

import numpy as np
class ComprehensiveSignalGenerator:
    """
    Generates trading signals using enhanced indicators, 
    including advanced Fibonacci and ElliottWave analysis.
    """

    def __init__(self, indicator_params=None):
        self.indicator_params = indicator_params if indicator_params else {}

    def analyze_row(self, row, prev_row=None, params=None):
        macd_fast = params.get("macd_fast", 12)
        macd_slow = params.get("macd_slow", 26)
        macd_signal = params.get("macd_signal", 9)
        adx_len = params.get("adx_len", 14)
        rsi_len = params.get("rsi_len", 14)
        atr_len = params.get("atr_len", 14)
        cci_len = params.get("cci_len", 20)
        signals = {'entry_signals': [], 'exit_signals': [], 'scores': {'bullish': 0, 'bearish': 0}}

        # MACD
        macd_col = f"MACD_{macd_fast}_{macd_slow}_{macd_signal}"
        macds_col = f"MACDs_{macd_fast}_{macd_slow}_{macd_signal}"
        macdh_col = f"MACDh_{macd_fast}_{macd_slow}_{macd_signal}"
        crossover = "neutral"
        if prev_row is not None:
            prev_macd = prev_row.get(macd_col, np.nan)
            prev_macds = prev_row.get(macds_col, np.nan)
            curr_macd = row.get(macd_col, np.nan)
            curr_macds = row.get(macds_col, np.nan)
            if not np.isnan(prev_macd) and not np.isnan(prev_macds) and not np.isnan(curr_macd) and not np.isnan(curr_macds):
                if prev_macd < prev_macds and curr_macd > curr_macds:
                    crossover = "bullish"
                elif prev_macd > prev_macds and curr_macd < curr_macds:
                    crossover = "bearish"
        row_macdh = row.get(macdh_col, 0)
        if crossover == "bullish":
            signals['entry_signals'].append("MACD bullish crossover")
            signals['scores']['bullish'] += 2
        elif crossover == "bearish":
            signals['exit_signals'].append("MACD bearish crossover")
            signals['scores']['bearish'] += 2

        # ADX
        adx_col = f"ADX_{adx_len}"
        adx_val = row.get(adx_col, 0)
        adx_trend = 'neutral'
        if adx_val > 25:
            adx_trend = 'strong'
            di_plus = row.get(f"DMP_{adx_len}", 0)
            di_minus = row.get(f"DMN_{adx_len}", 0)
            if di_plus > di_minus:
                signals['entry_signals'].append("Strong bullish trend (ADX)")
                signals['scores']['bullish'] += 2
            else:
                signals['exit_signals'].append("Strong bearish trend (ADX)")
                signals['scores']['bearish'] += 2

        # RSI
        rsi_col = f"RSI_{rsi_len}"
        rsi_val = row.get(rsi_col, np.nan)
        if rsi_val < 30:
            signals['entry_signals'].append("RSI oversold")
            signals['scores']['bullish'] += 1
        elif rsi_val > 70:
            signals['exit_signals'].append("RSI overbought")
            signals['scores']['bearish'] += 1

        # OBV
        obv = row.get("OBV", 0)
        if obv > 0:
            signals['entry_signals'].append("OBV bullish")
            signals['scores']['bullish'] += 1
        else:
            signals['exit_signals'].append("OBV bearish")
            signals['scores']['bearish'] += 1

        # SMA/EMA trend
        sma_fast = row.get("SMA_20", np.nan)
        sma_slow = row.get("SMA_50", np.nan)
        if not np.isnan(sma_fast) and not np.isnan(sma_slow):
            ma_trend = "bullish" if sma_fast > sma_slow else "bearish"
            if ma_trend == "bullish":
                signals['entry_signals'].append("SMA bullish crossover")
                signals['scores']['bullish'] += 1
            else:
                signals['exit_signals'].append("SMA bearish crossover")
                signals['scores']['bearish'] += 1

        # Supertrend
        supertrend_col = "SUPERT_14_3.0"
        supertrend = row.get(supertrend_col, 0)
        if supertrend > 0:
            signals['entry_signals'].append("Supertrend bullish")
            signals['scores']['bullish'] += 2
        else:
            signals['exit_signals'].append("Supertrend bearish")
            signals['scores']['bearish'] += 2

        # ATR
        atr_col = f"ATR_{atr_len}"
        atr = row.get(atr_col, np.nan)
        if not np.isnan(atr) and atr > 0.8 * atr:
            signals['entry_signals'].append("High ATR (volatility)")
            signals['scores']['bullish'] += 1

        # Linear Regression Angle
        lr_angle_col = f"LinReg_Angle_{cci_len}"  # adjust param for your config
        lr_angle = row.get(lr_angle_col, 0)
        if lr_angle > 20:
            signals['entry_signals'].append("Strong uptrend (LinReg)")
            signals['scores']['bullish'] += 1
        elif lr_angle < -20:
            signals['exit_signals'].append("Strong downtrend (LinReg)")
            signals['scores']['bearish'] += 1

        # Stochastic Oscillator
        stoch_k = row.get("STOCHk_14_5_3", 0)
        stoch_d = row.get("STOCHd_14_5_3", 0)
        if stoch_k < 20 and stoch_d < 20:
            signals['entry_signals'].append("Stochastic oversold")
            signals['scores']['bullish'] += 1
        elif stoch_k > 80 and stoch_d > 80:
            signals['exit_signals'].append("Stochastic overbought")
            signals['scores']['bearish'] += 1

        # CCI
        cci_col = f"CCI_{cci_len}"
        cci = row.get(cci_col, np.nan)
        if cci > 100:
            signals['entry_signals'].append("CCI bullish")
            signals['scores']['bullish'] += 1
        elif cci < -100:
            signals['exit_signals'].append("CCI bearish")
            signals['scores']['bearish'] += 1

        # ROC
        roc_col = f"ROC_{cci_len}"
        roc = row.get(roc_col, np.nan)
        if roc > 5:
            signals['entry_signals'].append("ROC positive momentum")
            signals['scores']['bullish'] += 1
        elif roc < -5:
            signals['exit_signals'].append("ROC negative momentum")
            signals['scores']['bearish'] += 1

        # Donchian Channel
        donchian_upper = row.get("DCHU_14_14", np.nan)
        donchian_lower = row.get("DCHL_14_14", np.nan)
        last_close = row.get("close", np.nan)
        if not np.isnan(last_close) and not np.isnan(donchian_upper) and not np.isnan(donchian_lower):
            if last_close >= donchian_upper:
                signals['exit_signals'].append("Price breaking upper Donchian")
                signals['scores']['bearish'] += 1
            elif last_close <= donchian_lower:
                signals['entry_signals'].append("Price breaking lower Donchian")
                signals['scores']['bullish'] += 1

        # Historical Volatility
        hist_vol_col = f"Hist_Volatility_{cci_len}"
        hist_vol = row.get(hist_vol_col, 0)
        if hist_vol > 0.05:
            signals['entry_signals'].append("Historically high volatility")
            signals['scores']['bullish'] += 1

        # --------- ADVANCED FIBONACCI ANALYSIS ----------
        fib = row.get("Fibonacci", {})
        if isinstance(fib, dict):
            fib_trend = fib.get("trend_direction", "")
            fib_support = fib.get("nearest_support", {})
            fib_resist = fib.get("nearest_resistance", {})
            current_price = last_close
            # Entry: Price close to uptrend nearest support
            if fib_trend == "uptrend" and fib_support and abs(current_price - fib_support.get('price', -999)) < 0.01 * current_price:
                signals['entry_signals'].append("Price near uptrend Fibonacci support")
                signals['scores']['bullish'] += 2
            # Exit: Price close to downtrend nearest resistance
            if fib_trend == "downtrend" and fib_resist and abs(current_price - fib_resist.get('price', -999)) < 0.01 * current_price:
                signals['exit_signals'].append("Price near downtrend Fibonacci resistance")
                signals['scores']['bearish'] += 2
            # Proximity to key levels (works for both trends)
            key_levels = fib.get('key_levels', {})
            if key_levels.get("61.8%") and np.isclose(current_price, key_levels["61.8%"], atol=0.01 * current_price):
                signals['entry_signals'].append("Price near Fibonacci 61.8% retracement")
                signals['scores']['bullish'] += 1
            if key_levels.get("38.2%") and np.isclose(current_price, key_levels["38.2%"], atol=0.01 * current_price):
                signals['exit_signals'].append("Price near Fibonacci 38.2% retracement")
                signals['scores']['bearish'] += 1

        # --------- ADVANCED ELLIOTT WAVE ANALYSIS ----------
        ew = row.get("Elliott_Wave", {})
        if isinstance(ew, dict):
            ew_pattern = ew.get("pattern", "")
            ew_trend = ew.get("trend", "")
            ew_conf = ew.get("confidence", 0)
            ew_next = ew.get("next_expectation", "")
            # Entry: Bullish impulse with high confidence
            if ew_pattern == "impulse" and ew_trend == "bullish" and ew_conf >= 60:
                signals['entry_signals'].append(f"Elliott impulse, bullish ({ew_conf}%)")
                signals['scores']['bullish'] += 2
            # Exit: Bearish impulse with high confidence
            if ew_pattern == "impulse" and ew_trend == "bearish" and ew_conf >= 60:
                signals['exit_signals'].append(f"Elliott impulse, bearish ({ew_conf}%)")
                signals['scores']['bearish'] += 2
            # Entry when corrective is "completing" and main trend is up
            if ew_pattern == "corrective" and ew_trend == "bullish" and ew_conf >= 40:
                signals['entry_signals'].append(f"Elliott corrective, bullish ({ew_conf}%)")
                signals['scores']['bullish'] += 1
            # Exit when corrective is "completing" in downtrend
            if ew_pattern == "corrective" and ew_trend == "bearish" and ew_conf >= 40:
                signals['exit_signals'].append(f"Elliott corrective, bearish ({ew_conf}%)")
                signals['scores']['bearish'] += 1
            # Optionally use next_expectation string for commentary/future actions

        # Compute summary bias and strength
        total = signals['scores']['bullish'] + signals['scores']['bearish']
        if total == 0:
            signals['overall_bias'] = "neutral"
            signals['signal_strength'] = 0
        else:
            pct = (signals['scores']['bullish'] / total) * 100
            signals['overall_bias'] = "bullish" if pct > 65 else "bearish" if pct < 35 else "neutral"
            signals['signal_strength'] = pct

        return signals

    def analyze(self, indicators_df, timeframe_params=None):
        results = []
        params = timeframe_params or {}
        for i in range(len(indicators_df)):
            row = indicators_df.iloc[i].to_dict()
            prev_row = indicators_df.iloc[i-1].to_dict() if i > 0 else None
            signals = self.analyze_row(row, prev_row, params)
            results.append(signals)
        return results

    def last_signal(self, indicators_df, params=None):
        if len(indicators_df) == 0:
            return {}
        row = indicators_df.iloc[-1].to_dict()
        prev_row = indicators_df.iloc[-2].to_dict() if len(indicators_df) > 1 else None
        return self.analyze_row(row, prev_row, params or {})

# Example usage:
# signal_gen = ComprehensiveSignalGenerator()
# signals_list = signal_gen.analyze(indicators_df)
# last_row_signal = signal_gen.last_signal(indicators_df)