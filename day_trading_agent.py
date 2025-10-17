import time
import threading
from indicator_fetcher import ComprehensiveMultiTimeframeAnalyzer
#######################################################################################################################################
#Basic version without Fibonacci analysis
#######################################################################################################################################
# class DayTraderAgent:
#     def __init__(self, symbol, timeframes=['5m', '15m', '1h', '1d']):
#         self.symbol = symbol
#         self.timeframes = timeframes
#         self.analyzer = ComprehensiveMultiTimeframeAnalyzer()

#     def analyze_market(self):
#         """Analyze the market using comprehensive analysis."""
#         results = self.analyzer.analyze_comprehensive_multi_timeframe(self.symbol, self.timeframes)
#         return results

#     def make_decision(self, analysis_results):
#         """Make buy/sell/hold decisions based on analysis results."""
#         for timeframe, data in analysis_results.items():
#             if 'indicators' in data and 'trading_signals' in data['indicators']:
#                 signals = data['indicators']['trading_signals']
#                 bias = signals['overall_bias']
#                 confidence = signals['confidence']
#                 print(f"Timeframe: {timeframe}, Bias: {bias}, Confidence: {confidence}")

#                 if bias == 'strong_bullish' and confidence == 'high':
#                     print("Decision: BUY")
#                 elif bias == 'strong_bearish' and confidence == 'high':
#                     print("Decision: SELL")
#                 else:
#                     print("Decision: HOLD")

#     def run(self):
#         """Run the agent to continuously analyze and make trading decisions."""
#         while True:
#             print(f"Analyzing market for {self.symbol}...")
#             analysis_results = self.analyze_market()
#             self.make_decision(analysis_results)
#             time.sleep(60)  # Wait for 1 minute before the next analysis

# # Usage
# if __name__ == "__main__":
#     trader_agent = DayTraderAgent(symbol='ASML')
#     trader_agent.run()

#######################################################################################################################################
#Version with Fibonacci analysis
#######################################################################################################################################
# class DayTraderAgent:
#     def __init__(self, symbol, timeframes=['1m','5m', '15m', '1h']):
#         self.symbol = symbol
#         self.timeframes = timeframes
#         self.analyzer = ComprehensiveMultiTimeframeAnalyzer()

#     def analyze_market(self):
#         """Analyze the market using comprehensive analysis."""
#         results = self.analyzer.analyze_comprehensive_multi_timeframe(self.symbol, self.timeframes)
#         return results

#     def make_decision(self, analysis_results):
#         """Make buy/sell/hold decisions based on analysis results."""
#         for timeframe, data in analysis_results.items():
#             if 'indicators' in data and 'trading_signals' in data['indicators']:
#                 signals = data['indicators']['trading_signals']
#                 fib_analysis = data['indicators'].get('fibonacci', {})
#                 bias = signals['overall_bias']
#                 confidence = signals['confidence']
#                 current_price = data['current_price']
#                 fib_support = fib_analysis.get('nearest_support', {}).get('price')
#                 fib_resistance = fib_analysis.get('nearest_resistance', {}).get('price')

#                 print(f"Timeframe: {timeframe}, Bias: {bias}, Confidence: {confidence}")
#                 print(f"Current Price: {current_price}, Fibonacci Support: {fib_support}, Fibonacci Resistance: {fib_resistance}")

#                 # Decision based on bias and confidence
#                 if bias == 'strong_bullish' and confidence == 'high':
#                     print("Decision: BUY")
#                 elif bias == 'strong_bearish' and confidence == 'high':
#                     print("Decision: SELL")
#                 # Additional decision based on Fibonacci levels
#                 elif fib_support and current_price <= fib_support:
#                     print("Decision: BUY (near Fibonacci support)")
#                 elif fib_resistance and current_price >= fib_resistance:
#                     print("Decision: SELL (near Fibonacci resistance)")
#                 else:
#                     print("Decision: HOLD")

#     def run(self):
#         """Run the agent to continuously analyze and make trading decisions."""
#         while True:
#             print(f"Analyzing market for {self.symbol}...")
#             analysis_results = self.analyze_market()
#             self.make_decision(analysis_results)
#             time.sleep(60)  # Wait for 1 minute before the next analysis

# # Usage
# if __name__ == "__main__":
#     trader_agent = DayTraderAgent(symbol='SPY')
#     trader_agent.run()

#######################################################################################################################################
#Version with Fibonacci analysis and Multiple Symbols
#######################################################################################################################################
class MultiSymbolDayTraderAgent:
    def __init__(self, symbols, timeframes=['5m', '15m', '1h']):
        self.symbols = symbols if isinstance(symbols, list) else [symbols]
        self.timeframes = timeframes
        self.analyzer = ComprehensiveMultiTimeframeAnalyzer()
        self.running = True

    def analyze_symbol(self, symbol):
        """Analyze a single symbol."""
        try:
            print(f"\n{'='*60}")
            print(f"ANALYZING {symbol}")
            print(f"{'='*60}")
            
            results = self.analyzer.analyze_comprehensive_multi_timeframe(symbol, self.timeframes)
            self.make_decision(symbol, results)
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")

    def make_decision(self, symbol, analysis_results):
        """Make buy/sell/hold decisions based on analysis results."""
        print(f"\n--- ANALYSIS FOR {symbol} ---")
        
        for timeframe, data in analysis_results.items():
            if 'indicators' in data and 'trading_signals' in data['indicators']:
                signals = data['indicators']['trading_signals']
                fib_analysis = data['indicators'].get('fibonacci', {})
                bias = signals['overall_bias']
                confidence = signals['confidence']
                current_price = data['current_price']
                fib_support = fib_analysis.get('nearest_support', {}).get('price')
                fib_resistance = fib_analysis.get('nearest_resistance', {}).get('price')

                print(f"  {timeframe}: Bias: {bias}, Confidence: {confidence}")
                print(f"  Price: {current_price}, Support: {fib_support}, Resistance: {fib_resistance}")

                # Decision logic
                if bias == 'strong_bullish' and confidence == 'high':
                    print(f"  Decision: BUY {symbol}")
                elif bias == 'strong_bearish' and confidence == 'high':
                    print(f"  Decision: SELL {symbol}")
                elif fib_support and current_price <= fib_support:
                    print(f"  Decision: BUY {symbol} (near Fibonacci support)")
                elif fib_resistance and current_price >= fib_resistance:
                    print(f"  Decision: SELL {symbol} (near Fibonacci resistance)")
                else:
                    print(f"  Decision: HOLD {symbol}")

    def run_sequential(self):
        # """Run analysis for all symbols sequentially continuously"""
        # while self.running:
        #     for symbol in self.symbols:
        #         if not self.running:
        #             break
        #         self.analyze_symbol(symbol)
            
        #     print(f"\nWaiting 60 seconds before next analysis cycle...")
        #     time.sleep(60)
        """Run analysis for all symbols one time."""
        for symbol in self.symbols:
            self.analyze_symbol(symbol)

    def run_parallel(self):
        """Run analysis for all symbols in parallel using threads."""
        def analyze_all_symbols():
            while self.running:
                threads = []
                for symbol in self.symbols:
                    if not self.running:
                        break
                    thread = threading.Thread(target=self.analyze_symbol, args=(symbol,))
                    threads.append(thread)
                    thread.start()
                
                # Wait for all threads to complete
                for thread in threads:
                    thread.join()
                
                print(f"\nWaiting 60 seconds before next analysis cycle...")
                time.sleep(60)
        
        analysis_thread = threading.Thread(target=analyze_all_symbols)
        analysis_thread.start()
        return analysis_thread

    def stop(self):
        """Stop the analysis."""
        self.running = False
if __name__ == "__main__":
    # Multiple symbols with parallel processing
    symbols = ['DUOL', 'SPY', 'TQQQ', 'AMD', 'ASML','ORCL']
    multi_trader = MultiSymbolDayTraderAgent(symbols=symbols)
    
    # Run in parallel (faster but uses more resources)
    # thread = multi_trader.run_parallel()
    
    # Or run sequentially (slower but more stable)
    multi_trader.run_sequential()
#######################################################################################################################################
# Version with Fibonacci and Elliot Wave analysis
#######################################################################################################################################
# class DayTraderAgent:
#     def __init__(self, symbol, timeframes=['5m', '15m', '1h', '1d']):
#         self.symbol = symbol
#         self.timeframes = timeframes
#         self.analyzer = ComprehensiveMultiTimeframeAnalyzer()

#     def safe_format_price(self, price):
#         """Safely format price values, handling both strings and numbers."""
#         if price is None:
#             return "N/A"
#         try:
#             # Try to convert to float if it's a string
#             if isinstance(price, str):
#                 price = float(price)
#             return f"${price:.2f}"
#         except (ValueError, TypeError):
#             return str(price)

#     def safe_format_number(self, number, decimal_places=2):
#         """Safely format numeric values."""
#         if number is None:
#             return "N/A"
#         try:
#             if isinstance(number, str):
#                 number = float(number)
#             return f"{number:.{decimal_places}f}"
#         except (ValueError, TypeError):
#             return str(number)

#     def convert_confidence_to_numeric(self, confidence):
#         """Convert string confidence levels to numeric values."""
#         if isinstance(confidence, (int, float)):
#             return float(confidence)
        
#         if isinstance(confidence, str):
#             confidence_lower = confidence.lower()
#             confidence_mapping = {
#                 'very_high': 0.95,
#                 'high': 0.8,
#                 'medium_high': 0.7,
#                 'medium': 0.6,
#                 'medium_low': 0.4,
#                 'low': 0.3,
#                 'very_low': 0.1,
#                 'strong': 0.85,
#                 'weak': 0.25,
#                 'neutral': 0.5
#             }
#             return confidence_mapping.get(confidence_lower, 0.5)
        
#         return 0.5  # Default medium confidence

#     def analyze_market(self):
#         """Analyze the market using comprehensive analysis."""
#         results = self.analyzer.analyze_comprehensive_multi_timeframe(self.symbol, self.timeframes)
#         return results

#     def analyze_elliott_wave(self, data):
#         """Analyze Elliott Wave patterns and determine current wave position."""
#         try:
#             elliott_analysis = self.analyzer.fetch_elliott_wave_data(self)
            
#             if not elliott_analysis:
#                 return None, None, None, 0
            
#             current_wave = elliott_analysis.get('current_wave')
#             wave_confidence = elliott_analysis.get('confidence', 0)
#             wave_direction = elliott_analysis.get('direction')
#             next_expected_wave = elliott_analysis.get('next_wave')
            
#             # Convert confidence to numeric
#             wave_confidence = self.convert_confidence_to_numeric(wave_confidence)
            
#             return current_wave, next_expected_wave, wave_direction, wave_confidence
#         except Exception as e:
#             print(f"Error in Elliott Wave analysis: {e}")
#             return None, None, None, 0

#     def get_elliott_wave_recommendation(self, current_wave, next_wave, direction, confidence):
#         """Generate trading recommendations based on Elliott Wave analysis."""
#         if confidence < 0.6:  # Low confidence threshold
#             return "HOLD", "Low Elliott Wave confidence"
        
#         recommendations = {
#             # Impulse waves (1, 3, 5) - trending moves
#             1: {
#                 'bullish': ("BUY", "Wave 1 beginning - early bullish entry"),
#                 'bearish': ("SELL", "Wave 1 beginning - early bearish entry")
#             },
#             2: {
#                 'bullish': ("BUY", "Wave 2 correction ending - prepare for Wave 3"),
#                 'bearish': ("SELL", "Wave 2 correction ending - prepare for Wave 3 down")
#             },
#             3: {
#                 'bullish': ("STRONG_BUY", "Wave 3 - strongest bullish impulse"),
#                 'bearish': ("STRONG_SELL", "Wave 3 - strongest bearish impulse")
#             },
#             4: {
#                 'bullish': ("HOLD", "Wave 4 correction - wait for Wave 5 setup"),
#                 'bearish': ("HOLD", "Wave 4 correction - wait for Wave 5 setup")
#             },
#             5: {
#                 'bullish': ("SELL", "Wave 5 ending - prepare for reversal"),
#                 'bearish': ("BUY", "Wave 5 ending - prepare for reversal")
#             },
#             # Corrective waves (A, B, C)
#             'A': {
#                 'bullish': ("SELL", "Wave A correction - bearish move in bull market"),
#                 'bearish': ("BUY", "Wave A correction - bullish move in bear market")
#             },
#             'B': {
#                 'bullish': ("HOLD", "Wave B - counter-trend move, wait for clarity"),
#                 'bearish': ("HOLD", "Wave B - counter-trend move, wait for clarity")
#             },
#             'C': {
#                 'bullish': ("BUY", "Wave C ending - final correction before trend resumes"),
#                 'bearish': ("SELL", "Wave C ending - final correction before trend resumes")
#             }
#         }
        
#         if current_wave in recommendations and direction:
#             wave_rec = recommendations[current_wave].get(direction, ("HOLD", "Unclear direction"))
#             return wave_rec[0], wave_rec[1]
        
#         return "HOLD", "Unknown Elliott Wave pattern"

#     def make_decision(self, analysis_results):
#         """Make buy/sell/hold decisions based on comprehensive analysis."""
#         final_recommendations = []
        
#         for timeframe, data in analysis_results.items():
#             try:
#                 if 'indicators' in data and 'trading_signals' in data['indicators']:
#                     signals = data['indicators']['trading_signals']
#                     fib_analysis = data['indicators'].get('fibonacci', {})
#                     bias = signals.get('overall_bias', 'neutral')
#                     confidence = signals.get('confidence', 0)
#                     current_price = data.get('current_price', 0)
                    
#                     # Convert confidence to numeric
#                     numeric_confidence = self.convert_confidence_to_numeric(confidence)
                    
#                     # Safely extract Fibonacci levels
#                     fib_support = None
#                     fib_resistance = None
                    
#                     if fib_analysis:
#                         nearest_support = fib_analysis.get('nearest_support', {})
#                         nearest_resistance = fib_analysis.get('nearest_resistance', {})
                        
#                         if isinstance(nearest_support, dict):
#                             fib_support = nearest_support.get('price')
#                         if isinstance(nearest_resistance, dict):
#                             fib_resistance = nearest_resistance.get('price')

#                     print(f"\n=== {timeframe.upper()} TIMEFRAME ANALYSIS ===")
#                     print(f"Current Price: {self.safe_format_price(current_price)}")
#                     print(f"Overall Bias: {bias} (Confidence: {confidence} -> {self.safe_format_number(numeric_confidence)})")
                    
#                     if fib_support:
#                         print(f"Fibonacci Support: {self.safe_format_price(fib_support)}")
#                     else:
#                         print("Fibonacci Support: N/A")
                        
#                     if fib_resistance:
#                         print(f"Fibonacci Resistance: {self.safe_format_price(fib_resistance)}")
#                     else:
#                         print("Fibonacci Resistance: N/A")

#                     # Elliott Wave Analysis
#                     current_wave, next_wave, wave_direction, wave_confidence = self.analyze_elliott_wave(data)
                    
#                     if current_wave:
#                         print(f"Elliott Wave - Current: Wave {current_wave} ({wave_direction if wave_direction else 'Unknown'})")
#                         print(f"Elliott Wave - Next Expected: Wave {next_wave if next_wave else 'Unknown'}")
#                         print(f"Elliott Wave Confidence: {self.safe_format_number(wave_confidence)}")
                        
#                         elliott_action, elliott_reason = self.get_elliott_wave_recommendation(
#                             current_wave, next_wave, wave_direction, wave_confidence
#                         )
#                         print(f"Elliott Wave Recommendation: {elliott_action} - {elliott_reason}")
#                     else:
#                         elliott_action, elliott_reason = "HOLD", "No Elliott Wave data"
#                         print("Elliott Wave: No data available")

#                     # Combined Decision Logic
#                     decision = self.combine_signals(bias, numeric_confidence, current_price, 
#                                                   fib_support, fib_resistance, 
#                                                   elliott_action, wave_confidence)
                    
#                     print(f"FINAL DECISION: {decision}")
#                     final_recommendations.append((timeframe, decision))
#                     print("-" * 50)
#                 else:
#                     print(f"\n=== {timeframe.upper()} TIMEFRAME ANALYSIS ===")
#                     print("No trading signals data available")
#                     final_recommendations.append((timeframe, "HOLD"))
#                     print("-" * 50)
                    
#             except Exception as e:
#                 print(f"Error analyzing {timeframe}: {e}")
#                 final_recommendations.append((timeframe, "HOLD"))
#                 continue

#         return final_recommendations

#     def combine_signals(self, bias, confidence, current_price, fib_support, fib_resistance, 
#                        elliott_action, wave_confidence):
#         """Combine all signals to make final trading decision."""
        
#         try:
#             # Ensure all values are numeric
#             confidence = self.convert_confidence_to_numeric(confidence)
#             wave_confidence = self.convert_confidence_to_numeric(wave_confidence)
            
#             # Convert price values to float
#             if isinstance(current_price, str):
#                 current_price = float(current_price) if current_price != 'N/A' else 0
#             if isinstance(fib_support, str):
#                 fib_support = float(fib_support) if fib_support != 'N/A' else None
#             if isinstance(fib_resistance, str):
#                 fib_resistance = float(fib_resistance) if fib_resistance != 'N/A' else None
            
#             # Weight different signals
#             bias_weight = confidence * 0.4
#             fib_weight = 0.3
#             elliott_weight = wave_confidence * 0.3
            
#             buy_score = 0
#             sell_score = 0
            
#             print(f"Signal Analysis:")
#             print(f"  Bias: {bias} (weight: {bias_weight:.2f})")
#             print(f"  Fibonacci weight: {fib_weight}")
#             print(f"  Elliott Wave weight: {elliott_weight:.2f}")
            
#             # Bias scoring
#             if bias in ['strong_bullish', 'bullish']:
#                 buy_score += bias_weight
#                 print(f"  Bias adds {bias_weight:.2f} to BUY score")
#             elif bias in ['strong_bearish', 'bearish']:
#                 sell_score += bias_weight
#                 print(f"  Bias adds {bias_weight:.2f} to SELL score")
                
#             # Fibonacci scoring
#             if fib_support and current_price and current_price <= fib_support * 1.01:  # Within 1% of support
#                 buy_score += fib_weight
#                 print(f"  Near Fibonacci support adds {fib_weight} to BUY score")
#             elif fib_resistance and current_price and current_price >= fib_resistance * 0.99:  # Within 1% of resistance
#                 sell_score += fib_weight
#                 print(f"  Near Fibonacci resistance adds {fib_weight} to SELL score")
                
#             # Elliott Wave scoring
#             if elliott_action in ['BUY', 'STRONG_BUY']:
#                 buy_score += elliott_weight
#                 print(f"  Elliott Wave adds {elliott_weight:.2f} to BUY score")
#             elif elliott_action in ['SELL', 'STRONG_SELL']:
#                 sell_score += elliott_weight
#                 print(f"  Elliott Wave adds {elliott_weight:.2f} to SELL score")
            
#             print(f"  Final scores - BUY: {buy_score:.2f}, SELL: {sell_score:.2f}")
                
#             # Final decision
#             if buy_score > sell_score and buy_score > 0.6:
#                 if elliott_action == 'STRONG_BUY' or bias == 'strong_bullish':
#                     return "STRONG BUY"
#                 return "BUY"
#             elif sell_score > buy_score and sell_score > 0.6:
#                 if elliott_action == 'STRONG_SELL' or bias == 'strong_bearish':
#                     return "STRONG SELL"
#                 return "SELL"
#             else:
#                 return "HOLD"
                
#         except Exception as e:
#             print(f"Error in signal combination: {e}")
#             return "HOLD"

#     def run(self, iterations=None):
#         """Run the agent to continuously analyze and make trading decisions."""
#         count = 0
#         while True:
#             print(f"\n{'='*60}")
#             print(f"MARKET ANALYSIS #{count + 1} for {self.symbol}")
#             print(f"{'='*60}")
            
#             try:
#                 analysis_results = self.analyze_market()
#                 recommendations = self.make_decision(analysis_results)
                
#                 # Summary of all timeframe recommendations
#                 print(f"\n=== SUMMARY OF RECOMMENDATIONS ===")
#                 for timeframe, decision in recommendations:
#                     print(f"{timeframe}: {decision}")
                    
#             except Exception as e:
#                 print(f"Error during analysis: {e}")
            
#             count += 1
#             if iterations and count >= iterations:
#                 break
                
#             print(f"\nWaiting 60 seconds for next analysis...")
#             time.sleep(60)

# # Enhanced usage with additional features
# if __name__ == "__main__":
#     # Initialize trader agent
#     trader_agent = DayTraderAgent(symbol='AAPL', timeframes=['5m', '15m', '1h', '1d'])
    
#     # Run for a specific number of iterations (or indefinitely if None)
#     trader_agent.run(iterations=5)  # Run 5 analysis cycles for testing