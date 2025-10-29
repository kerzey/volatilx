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

# #######################################################################################################################################
# #Version with Fibonacci analysis and Multiple Symbols
# #######################################################################################################################################
# class MultiSymbolDayTraderAgent:
#     def __init__(self, symbols, timeframes=['2m','5m','15m','30m','1h','90m','1d','5d','1wk']):
#         self.symbols = symbols if isinstance(symbols, list) else [symbols]
#         self.timeframes = timeframes
#         self.analyzer = ComprehensiveMultiTimeframeAnalyzer()
#         self.running = True
    
#     def analyze_symbol(self, symbol):
#         """Analyze a single symbol and return the results."""
#         try:
#             analysis_output = {
#                 "symbol": symbol,
#                 "status": "success",
#                 "analysis": [],
#                 "error": None
#             }

#             analysis_output["analysis"].append(f"{'='*60}")
#             analysis_output["analysis"].append(f"ANALYZING {symbol}")
#             analysis_output["analysis"].append(f"{'='*60}")
            
#             results = self.analyzer.analyze_comprehensive_multi_timeframe(symbol, self.timeframes)
#             decision_results = self.make_decision(symbol, results)
#             # print("Results ", results)  # Print the raw analysis results
#             analysis_output["analysis"].extend(decision_results)  # Append decision results
#         except Exception as e:
#             analysis_output["status"] = "error"
#             analysis_output["error"] = str(e)
#         print("Analysis output ", analysis_output)  # Print the final analysis output
#         return analysis_output
    
#     def make_decision(self, symbol, analysis_results):
#         """Make buy/sell/hold decisions based on analysis results and return the results."""
#         decision_output = [f"\n--- ANALYSIS FOR {symbol} ---"]
        
#         for timeframe, data in analysis_results.items():
#             # Defensive: indicators may be missing or None
#             indicators = data.get('indicators') or {}
#             signals = indicators.get('trading_signals')
#             fib_analysis = indicators.get('fibonacci') or {}

#             if signals is not None:
#                 bias = signals.get('overall_bias')
#                 confidence = signals.get('confidence')
#                 current_price = data.get('current_price')

#                 fib_support = None
#                 fib_resistance = None
#                 if isinstance(fib_analysis, dict):
#                     nearest_support = fib_analysis.get('nearest_support')
#                     nearest_resistance = fib_analysis.get('nearest_resistance')
#                     if isinstance(nearest_support, dict):
#                         fib_support = nearest_support.get('price')
#                     if isinstance(nearest_resistance, dict):
#                         fib_resistance = nearest_resistance.get('price')

#                 decision_output.append(f"  {timeframe}: Bias: {bias}, Confidence: {confidence}")
#                 decision_output.append(f"  Price: {current_price}, Support: {fib_support}, Resistance: {fib_resistance}")

#                 # Decision logic
#                 if bias == 'strong_bullish' and confidence == 'high':
#                     decision_output.append(f"  Decision: BUY {symbol}")
#                 elif bias == 'strong_bearish' and confidence == 'high':
#                     decision_output.append(f"  Decision: SELL {symbol}")
#                 elif fib_support is not None and current_price <= fib_support:
#                     decision_output.append(f"  Decision: BUY {symbol} (near Fibonacci support)")
#                 elif fib_resistance is not None and current_price >= fib_resistance:
#                     decision_output.append(f"  Decision: SELL {symbol} (near Fibonacci resistance)")
#                 else:
#                     decision_output.append(f"  Decision: HOLD {symbol}")
#             else:
#                 decision_output.append(f"  {timeframe}: indicators or trading_signals missing")
#         # print("Decision output ", decision_output)
#         return decision_output

#     def run_sequential(self):
#         """Run analysis for all symbols one time and return results."""
#         results = {}  # Dictionary to store results for each symbol
#         for symbol in self.symbols:
#             results[symbol] = self.analyze_symbol(symbol)  # Collect results for each symbol
#         return results

#     def run_parallel(self):
#         """Run analysis for all symbols in parallel using threads."""
#         def analyze_all_symbols():
#             while self.running:
#                 threads = []
#                 for symbol in self.symbols:
#                     if not self.running:
#                         break
#                     thread = threading.Thread(target=self.analyze_symbol, args=(symbol,))
#                     threads.append(thread)
#                     thread.start()
                
#                 # Wait for all threads to complete
#                 for thread in threads:
#                     thread.join()
                
#                 print(f"\nWaiting 60 seconds before next analysis cycle...")
#                 time.sleep(60)
        
#         analysis_thread = threading.Thread(target=analyze_all_symbols)
#         analysis_thread.start()
#         return analysis_thread

#     def stop(self):
#         """Stop the analysis."""
#         self.running = False
# if __name__ == "__main__":
#     # Multiple symbols with parallel processing
#     symbols = ['ASML']
#     multi_trader = MultiSymbolDayTraderAgent(symbols=symbols)
    
#     # Run in parallel (faster but uses more resources)
#     # thread = multi_trader.run_parallel()
    
#     # Or run sequentially (slower but more stable)
#     result = multi_trader.run_sequential()
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

#######################################################################################################################################
# Alpaca Version with Fibonacci analysis and Multiple Symbols
# #######################################################################################################################################
import time
import threading
import json
from datetime import datetime
from indicator_fetcher import ComprehensiveMultiTimeframeAnalyzer

class MultiSymbolDayTraderAgent:
    def __init__(self, symbols, timeframes=['2m','5m', '15m', '30m', '1h', '1d'], api_key=None, secret_key=None):
        """
        Initialize the Multi-Symbol Day Trader Agent
        
        Args:
            symbols: List of symbols to analyze or single symbol string
            timeframes: List of timeframes to analyze (avoid 90m - use 1h instead)
            api_key: Alpaca API key (optional, can be set later)
            secret_key: Alpaca secret key (optional, can be set later)
        """
        self.symbols = symbols if isinstance(symbols, list) else [symbols]
        
        # Filter out invalid timeframes for Alpaca
        valid_timeframes = []
        for tf in timeframes:
            if tf == '90m':
                print(f"⚠️ Warning: '90m' timeframe not supported by Alpaca. Using '1h' instead.")
                if '1h' not in valid_timeframes:
                    valid_timeframes.append('1h')
            else:
                valid_timeframes.append(tf)
        
        self.timeframes = valid_timeframes
        self.analyzer = ComprehensiveMultiTimeframeAnalyzer(api_key=api_key, secret_key=secret_key)
        self.running = True
        self.analysis_history = {}
        
        # Trading thresholds
        self.buy_threshold = 70  # Minimum strength for buy signal
        self.sell_threshold = 70  # Minimum strength for sell signal
        self.high_confidence_only = True  # Only trade on high confidence signals
        
        print(f"🚀 Multi-Symbol Day Trader Agent initialized")
        print(f"📊 Symbols: {', '.join(self.symbols)}")
        print(f"⏰ Timeframes: {', '.join(self.timeframes)}")
    
    def set_credentials(self, api_key, secret_key, base_url='https://paper-api.alpaca.markets'):
        """Set Alpaca API credentials"""
        self.analyzer.set_credentials(api_key, secret_key, base_url)
        print("✅ API credentials updated")
    
    def analyze_symbol(self, symbol):
        """Analyze a single symbol and return comprehensive results"""
        try:
            analysis_output = {
                "symbol": symbol,
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "status": "success",
                "analysis": [],
                "decisions": {},
                "consensus": None,
                "error": None,
                "raw_data": None
            }

            analysis_output["analysis"].append(f"{'='*80}")
            analysis_output["analysis"].append(f"🔍 COMPREHENSIVE ANALYSIS: {symbol}")
            analysis_output["analysis"].append(f"{'='*80}")
            analysis_output["analysis"].append(f"⏰ Analysis Time: {analysis_output['timestamp']}")
            
            # Run comprehensive multi-timeframe analysis
            results = self.analyzer.analyze_comprehensive_multi_timeframe(
                symbol=symbol, 
                timeframes=self.timeframes,
                base_period='5y'
            )
            
            if not results:
                analysis_output["status"] = "error"
                analysis_output["error"] = "No analysis results returned"
                return analysis_output
            
            # Store raw data for further processing
            analysis_output["raw_data"] = results
            
            # Generate trading decisions for each timeframe
            decision_results = self.make_trading_decisions(symbol, results)
            analysis_output["decisions"] = decision_results["timeframe_decisions"]
            analysis_output["consensus"] = decision_results["consensus"]
            
            # Format analysis output
            analysis_output["analysis"].extend(self.format_analysis_output(symbol, results, decision_results))
            
            # Store in history
            if symbol not in self.analysis_history:
                self.analysis_history[symbol] = []
            self.analysis_history[symbol].append({
                'timestamp': analysis_output['timestamp'],
                'consensus': analysis_output['consensus'],
                'decisions': analysis_output['decisions']
            })
            
        except Exception as e:
            analysis_output["status"] = "error"
            analysis_output["error"] = str(e)
            analysis_output["analysis"].append(f"❌ Error analyzing {symbol}: {str(e)}")
        
        return analysis_output
    
    def make_trading_decisions(self, symbol, analysis_results):
        """Make comprehensive trading decisions based on analysis results"""
        decision_output = {
            "timeframe_decisions": {},
            "consensus": {
                "overall_recommendation": "HOLD",
                "confidence": "LOW",
                "strength": 0,
                "buy_signals": 0,
                "sell_signals": 0,
                "hold_signals": 0,
                "reasoning": []
            }
        }
        
        buy_count = 0
        sell_count = 0
        hold_count = 0
        total_strength = 0
        high_confidence_signals = 0
        
        for timeframe, data in analysis_results.items():
            timeframe_decision = {
                "recommendation": "HOLD",
                "confidence": "LOW",
                "strength": 0,
                "reasoning": [],
                "entry_price": None,
                "stop_loss": None,
                "take_profit": None,
                "risk_reward_ratio": None
            }
            
            # Extract indicators and signals
            indicators = data.get('indicators', {})
            trading_signals = indicators.get('trading_signals', {})
            fibonacci = indicators.get('fibonacci', {})
            current_price = data.get('current_price', 0)
            
            if trading_signals:
                bias = trading_signals.get('overall_bias', 'neutral')
                confidence = trading_signals.get('confidence', 'low')
                strength = trading_signals.get('strength', 0)
                
                timeframe_decision["confidence"] = confidence
                timeframe_decision["strength"] = strength
                timeframe_decision["entry_price"] = current_price
                
                # Extract key levels for stop loss and take profit
                key_levels = trading_signals.get('key_levels', {})
                risk_reward = trading_signals.get('risk_reward', {})
                
                if risk_reward:
                    timeframe_decision["stop_loss"] = risk_reward.get('stop_loss')
                    timeframe_decision["take_profit"] = risk_reward.get('take_profit')
                    timeframe_decision["risk_reward_ratio"] = risk_reward.get('ratio')
                
                # Decision logic based on bias and strength
                if bias in ['strong_bullish', 'bullish'] and strength >= self.buy_threshold:
                    if not self.high_confidence_only or confidence == 'high':
                        timeframe_decision["recommendation"] = "BUY"
                        buy_count += 1
                        timeframe_decision["reasoning"].append(f"Strong {bias} signal with {strength:.0f}% strength")
                        
                        if confidence == 'high':
                            high_confidence_signals += 1
                            timeframe_decision["reasoning"].append("High confidence signal")
                
                elif bias in ['strong_bearish', 'bearish'] and strength >= self.sell_threshold:
                    if not self.high_confidence_only or confidence == 'high':
                        timeframe_decision["recommendation"] = "SELL"
                        sell_count += 1
                        timeframe_decision["reasoning"].append(f"Strong {bias} signal with {strength:.0f}% strength")
                        
                        if confidence == 'high':
                            high_confidence_signals += 1
                            timeframe_decision["reasoning"].append("High confidence signal")
                
                else:
                    hold_count += 1
                    timeframe_decision["reasoning"].append(f"Neutral or weak signal: {bias} ({strength:.0f}%)")
                
                # Additional Fibonacci-based decisions
                if 'error' not in fibonacci:
                    nearest_support = fibonacci.get('nearest_support')
                    nearest_resistance = fibonacci.get('nearest_resistance')
                    
                    if nearest_support and isinstance(nearest_support, dict):
                        support_price = nearest_support.get('price', 0)
                        if current_price <= support_price * 1.02:  # Within 2% of support
                            if timeframe_decision["recommendation"] == "HOLD":
                                timeframe_decision["recommendation"] = "BUY"
                                buy_count += 1
                                hold_count -= 1
                            timeframe_decision["reasoning"].append(f"Near Fibonacci support at ${support_price:.2f}")
                    
                    if nearest_resistance and isinstance(nearest_resistance, dict):
                        resistance_price = nearest_resistance.get('price', 0)
                        if current_price >= resistance_price * 0.98:  # Within 2% of resistance
                            if timeframe_decision["recommendation"] == "HOLD":
                                timeframe_decision["recommendation"] = "SELL"
                                sell_count += 1
                                hold_count -= 1
                            timeframe_decision["reasoning"].append(f"Near Fibonacci resistance at ${resistance_price:.2f}")
                
                total_strength += strength
            
            else:
                timeframe_decision["reasoning"].append("No trading signals available")
                hold_count += 1
            
            decision_output["timeframe_decisions"][timeframe] = timeframe_decision
        
        # Calculate consensus
        total_timeframes = len(analysis_results)
        if total_timeframes > 0:
            avg_strength = total_strength / total_timeframes
            
            buy_pct = (buy_count / total_timeframes) * 100
            sell_pct = (sell_count / total_timeframes) * 100
            hold_pct = (hold_count / total_timeframes) * 100
            
            decision_output["consensus"]["buy_signals"] = buy_count
            decision_output["consensus"]["sell_signals"] = sell_count
            decision_output["consensus"]["hold_signals"] = hold_count
            decision_output["consensus"]["strength"] = avg_strength
            
            # Determine overall recommendation
            if buy_pct >= 60 and avg_strength >= self.buy_threshold:
                decision_output["consensus"]["overall_recommendation"] = "BUY"
                decision_output["consensus"]["confidence"] = "HIGH" if buy_pct >= 75 else "MEDIUM"
                decision_output["consensus"]["reasoning"].append(f"{buy_pct:.0f}% of timeframes suggest BUY")
            elif sell_pct >= 60 and avg_strength >= self.sell_threshold:
                decision_output["consensus"]["overall_recommendation"] = "SELL"
                decision_output["consensus"]["confidence"] = "HIGH" if sell_pct >= 75 else "MEDIUM"
                decision_output["consensus"]["reasoning"].append(f"{sell_pct:.0f}% of timeframes suggest SELL")
            else:
                decision_output["consensus"]["overall_recommendation"] = "HOLD"
                decision_output["consensus"]["confidence"] = "LOW"
                decision_output["consensus"]["reasoning"].append("Mixed or weak signals across timeframes")
            
            # Add high confidence signal info
            if high_confidence_signals > 0:
                decision_output["consensus"]["reasoning"].append(f"{high_confidence_signals} high-confidence signals detected")
        
        return decision_output
    
    def format_analysis_output(self, symbol, results, decisions):
        """Format the analysis output for display"""
        output = []
        
        # Summary table
        output.append(f"\n📊 TIMEFRAME SUMMARY:")
        output.append(f"{'TF':<6} {'Price':<10} {'Bias':<15} {'Strength':<8} {'Conf':<6} {'Decision':<8} {'R/R':<6}")
        output.append("-" * 70)
        
        for timeframe, data in results.items():
            current_price = data.get('current_price', 0)
            indicators = data.get('indicators', {})
            signals = indicators.get('trading_signals', {})
            
            bias = signals.get('overall_bias', 'N/A')[:14]
            strength = f"{signals.get('strength', 0):.0f}%"
            confidence = signals.get('confidence', 'N/A')[:4]
            
            tf_decision = decisions['timeframe_decisions'].get(timeframe, {})
            recommendation = tf_decision.get('recommendation', 'HOLD')
            rr_ratio = tf_decision.get('risk_reward_ratio', 0)
            rr_str = f"{rr_ratio:.1f}" if rr_ratio else "N/A"
            
            output.append(f"{timeframe:<6} ${current_price:<9.2f} {bias:<15} {strength:<8} {confidence:<6} {recommendation:<8} {rr_str:<6}")
        
        # Consensus decision
        consensus = decisions['consensus']
        output.append(f"\n🎯 CONSENSUS DECISION:")
        output.append(f"Overall Recommendation: {consensus['overall_recommendation']}")
        output.append(f"Confidence: {consensus['confidence']}")
        output.append(f"Average Strength: {consensus['strength']:.1f}%")
        output.append(f"Signal Distribution: BUY({consensus['buy_signals']}) SELL({consensus['sell_signals']}) HOLD({consensus['hold_signals']})")
        
        if consensus['reasoning']:
            output.append(f"Reasoning:")
            for reason in consensus['reasoning']:
                output.append(f"  • {reason}")
        
        # Individual timeframe details
        output.append(f"\n📋 TIMEFRAME DECISIONS:")
        for timeframe, tf_decision in decisions['timeframe_decisions'].items():
            output.append(f"\n{timeframe.upper()} Timeframe:")
            output.append(f"  Recommendation: {tf_decision['recommendation']}")
            output.append(f"  Confidence: {tf_decision['confidence']}")
            output.append(f"  Strength: {tf_decision['strength']:.0f}%")
            
            if tf_decision.get('entry_price'):
                output.append(f"  Entry Price: ${tf_decision['entry_price']:.2f}")
            if tf_decision.get('stop_loss'):
                output.append(f"  Stop Loss: ${tf_decision['stop_loss']:.2f}")
            if tf_decision.get('take_profit'):
                output.append(f"  Take Profit: ${tf_decision['take_profit']:.2f}")
            if tf_decision.get('risk_reward_ratio'):
                output.append(f"  Risk/Reward: {tf_decision['risk_reward_ratio']:.2f}:1")
            
            if tf_decision.get('reasoning'):
                output.append(f"  Reasoning:")
                for reason in tf_decision['reasoning']:
                    output.append(f"    • {reason}")
        
        return output
    
    def run_sequential(self):
        """Run analysis for all symbols sequentially and return results"""
        results = {}
        
        print(f"\n🚀 Starting sequential analysis of {len(self.symbols)} symbols...")
        print(f"⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        for i, symbol in enumerate(self.symbols, 1):
            print(f"\n📊 Analyzing symbol {i}/{len(self.symbols)}: {symbol}")
            
            start_time = time.time()
            symbol_result = self.analyze_symbol(symbol)
            end_time = time.time()
            
            analysis_time = end_time - start_time
            symbol_result['analysis_time_seconds'] = analysis_time
            
            results[symbol] = symbol_result
            
            # Print quick summary
            if symbol_result['status'] == 'success' and symbol_result['consensus']:
                consensus = symbol_result['consensus']
                print(f"✅ {symbol}: {consensus['overall_recommendation']} "
                      f"({consensus['confidence']} confidence, {consensus['strength']:.0f}% strength) "
                      f"[{analysis_time:.1f}s]")
            else:
                print(f"❌ {symbol}: Analysis failed - {symbol_result.get('error', 'Unknown error')} [{analysis_time:.1f}s]")
            
            # Print detailed analysis if requested
            if symbol_result['status'] == 'success':
                for line in symbol_result['analysis']:
                    print(line)
        
        # Print overall summary
        self.print_portfolio_summary(results)
        
        return results
    
    def run_parallel(self, analysis_interval=300):
        """Run analysis for all symbols in parallel using threads"""
        def analyze_all_symbols():
            cycle_count = 0
            while self.running:
                cycle_count += 1
                print(f"\n🔄 Starting analysis cycle #{cycle_count}")
                print(f"⏰ Cycle start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                threads = []
                results = {}
                
                # Create and start threads for each symbol
                for symbol in self.symbols:
                    if not self.running:
                        break
                    
                    def analyze_wrapper(sym):
                        results[sym] = self.analyze_symbol(sym)
                    
                    thread = threading.Thread(target=analyze_wrapper, args=(symbol,))
                    threads.append(thread)
                    thread.start()
                
                # Wait for all threads to complete
                for thread in threads:
                    thread.join()
                
                # Print cycle summary
                if results:
                    self.print_portfolio_summary(results)
                
                if self.running:
                    print(f"\n⏳ Waiting {analysis_interval} seconds before next cycle...")
                    time.sleep(analysis_interval)
        
        analysis_thread = threading.Thread(target=analyze_all_symbols)
        analysis_thread.daemon = True
        analysis_thread.start()
        return analysis_thread
    
    def print_portfolio_summary(self, results):
        """Print a summary of all analyzed symbols"""
        print(f"\n{'='*80}")
        print(f"📈 PORTFOLIO ANALYSIS SUMMARY")
        print(f"{'='*80}")
        
        buy_symbols = []
        sell_symbols = []
        hold_symbols = []
        error_symbols = []
        
        total_analysis_time = 0
        
        for symbol, result in results.items():
            if result['status'] == 'success':
                consensus = result.get('consensus', {})
                recommendation = consensus.get('overall_recommendation', 'HOLD')
                confidence = consensus.get('confidence', 'LOW')
                strength = consensus.get('strength', 0)
                
                symbol_info = f"{symbol} ({confidence} conf, {strength:.0f}%)"
                
                if recommendation == 'BUY':
                    buy_symbols.append(symbol_info)
                elif recommendation == 'SELL':
                    sell_symbols.append(symbol_info)
                else:
                    hold_symbols.append(symbol_info)
                
                total_analysis_time += result.get('analysis_time_seconds', 0)
            else:
                error_symbols.append(f"{symbol} - {result.get('error', 'Unknown error')}")
        
        # Print recommendations
        print(f"\n🟢 BUY RECOMMENDATIONS ({len(buy_symbols)}):")
        for symbol in buy_symbols:
            print(f"  • {symbol}")
        
        print(f"\n🔴 SELL RECOMMENDATIONS ({len(sell_symbols)}):")
        for symbol in sell_symbols:
            print(f"  • {symbol}")
        
        print(f"\n🟡 HOLD RECOMMENDATIONS ({len(hold_symbols)}):")
        for symbol in hold_symbols:
            print(f"  • {symbol}")
        
        if error_symbols:
            print(f"\n❌ ANALYSIS ERRORS ({len(error_symbols)}):")
            for symbol in error_symbols:
                print(f"  • {symbol}")
        
        # Statistics
        total_symbols = len(results)
        success_rate = ((total_symbols - len(error_symbols)) / total_symbols * 100) if total_symbols > 0 else 0
        avg_analysis_time = total_analysis_time / total_symbols if total_symbols > 0 else 0
        
        print(f"\n📊 STATISTICS:")
        print(f"Total Symbols: {total_symbols}")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Average Analysis Time: {avg_analysis_time:.1f} seconds")
        print(f"Total Analysis Time: {total_analysis_time:.1f} seconds")
        print(f"Analysis Completion: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def get_symbol_history(self, symbol, limit=10):
        """Get analysis history for a specific symbol"""
        if symbol in self.analysis_history:
            return self.analysis_history[symbol][-limit:]
        return []
   # Bunu daha sonra kullanacağım. Bu sonucu capture etmek istedigimde.
    # def export_results(self, results, filename=None):
    #     """Export analysis results to JSON file"""
    #     if filename is None:
    #         timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    #         filename = f"trading_analysis_{timestamp}.json"
        
    #     # Prepare data for export (remove non-serializable objects)
    #     export_data = {}
    #     for symbol, result in results.items():
    #         export_data[symbol] = {
    #             'symbol': result['symbol'],
    #             'timestamp': result['timestamp'],
    #             'status': result['status'],
    #             'consensus': result.get('consensus'),
    #             'decisions': result.get('decisions'),
    #             'error': result.get('error'),
    #             'analysis_time_seconds': result.get('analysis_time_seconds')
    #         }
        
    #     try:
    #         with open(filename, 'w') as f:
    #             json.dump(export_data, f, indent=2, default=str)
    #         print(f"✅ Results exported to {filename}")
    #         return export_data, filename
    #     except Exception as e:
    #         print(f"❌ Error exporting results: {e}")
    #         return None
    def export_results(self, results, filename=None):
        """Export analysis results to JSON format (returns data without writing to file)"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"trading_analysis_{timestamp}.json"
        
        # Prepare data for export (remove non-serializable objects)
        export_data = {}
        for symbol, result in results.items():
            export_data[symbol] = {
                'symbol': result['symbol'],
                'timestamp': result['timestamp'],
                'status': result['status'],
                'consensus': result.get('consensus'),
                'decisions': result.get('decisions'),
                'error': result.get('error'),
                'analysis_time_seconds': result.get('analysis_time_seconds')
            }
        
        try:
            # Convert to JSON string to ensure it's serializable
            json_string = json.dumps(export_data, indent=2, default=str)
            print(f"✅ Results prepared for export (filename would be: {filename})")
            
            # Return both the data and the JSON string
            # print('prinnntt jsonnn', json_string)

            return export_data, json_string, filename
         
        except Exception as e:
            print(f"❌ Error preparing results for export: {e}")
            return None, None, None
        
    def set_trading_parameters(self, buy_threshold=70, sell_threshold=70, high_confidence_only=True):
        """Set trading decision parameters"""
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.high_confidence_only = high_confidence_only
        
        print(f"🎯 Trading parameters updated:")
        print(f"  Buy threshold: {buy_threshold}%")
        print(f"  Sell threshold: {sell_threshold}%")
        print(f"  High confidence only: {high_confidence_only}")
    
    def add_symbols(self, new_symbols):
        """Add new symbols to the analysis list"""
        if isinstance(new_symbols, str):
            new_symbols = [new_symbols]
        
        for symbol in new_symbols:
            if symbol not in self.symbols:
                self.symbols.append(symbol)
                print(f"✅ Added {symbol} to analysis list")
            else:
                print(f"⚠️ {symbol} already in analysis list")
    
    def remove_symbols(self, symbols_to_remove):
        """Remove symbols from the analysis list"""
        if isinstance(symbols_to_remove, str):
            symbols_to_remove = [symbols_to_remove]
        
        for symbol in symbols_to_remove:
            if symbol in self.symbols:
                self.symbols.remove(symbol)
                print(f"✅ Removed {symbol} from analysis list")
            else:
                print(f"⚠️ {symbol} not found in analysis list")
    
    def stop(self):
        """Stop the analysis"""
        self.running = False
        print("🛑 Analysis stopped")
    
    def get_current_symbols(self):
        """Get the current list of symbols being analyzed"""
        return self.symbols.copy()
    
    def print_status(self):
        """Print current agent status"""
        print(f"\n📊 DAY TRADING AGENT STATUS:")
        print(f"Running: {self.running}")
        print(f"Symbols ({len(self.symbols)}): {', '.join(self.symbols)}")
        print(f"Timeframes: {', '.join(self.timeframes)}")
        print(f"Buy Threshold: {self.buy_threshold}%")
        print(f"Sell Threshold: {self.sell_threshold}%")
        print(f"High Confidence Only: {self.high_confidence_only}")
        
        # Show recent analysis counts
        total_analyses = sum(len(history) for history in self.analysis_history.values())
        print(f"Total Analyses Performed: {total_analyses}")

# Example usage and testing functions
def demo_single_symbol():
    """Demo with single symbol analysis"""
    print("🚀 Single Symbol Demo")
    
    # Initialize with single symbol - avoid 90m timeframe
    agent = MultiSymbolDayTraderAgent(
        symbols='AAPL',
        timeframes=['15m', '1h', '1d']  # Valid timeframes only
    )
    
    # You would set your Alpaca credentials here
    # agent.set_credentials('your_api_key', 'your_secret_key')
    
    # Run analysis
    results = agent.run_sequential()
    
    # Export results
    agent.export_results(results)
    
    return results

def demo_multiple_symbols():
    """Demo with multiple symbols"""
    print("🚀 Multiple Symbols Demo")
    
    # Initialize with multiple symbols - use valid timeframes
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
    agent = MultiSymbolDayTraderAgent(
        symbols=symbols,
        timeframes=['5m', '15m', '1h', '1d']  # Valid timeframes only
    )
    
    # Set custom trading parameters
    agent.set_trading_parameters(
        buy_threshold=75,
        sell_threshold=75,
        high_confidence_only=True
    )
    
    # You would set your Alpaca credentials here
    # agent.set_credentials('your_api_key', 'your_secret_key')
    
    # Run analysis
    results = agent.run_sequential()
    
    return results

def demo_parallel_analysis():
    """Demo with parallel analysis (continuous monitoring)"""
    print("🚀 Parallel Analysis Demo")
    
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    agent = MultiSymbolDayTraderAgent(
        symbols=symbols,
        timeframes=['5m', '15m', '1h', '1d']  # Valid timeframes only
    )
    
    # You would set your Alpaca credentials here
    # agent.set_credentials('your_api_key', 'your_secret_key')
    
    print("Starting parallel analysis... Press Ctrl+C to stop")
    
    try:
        # Start parallel analysis (runs every 5 minutes)
        thread = agent.run_parallel(analysis_interval=300)
        
        # Keep the main thread alive
        while agent.running:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping analysis...")
        agent.stop()
        thread.join(timeout=5)

if __name__ == "__main__":
    print("🚀 Multi-Symbol Day Trading Agent")
    print("=" * 50)
    
    # Example with single symbol - use valid timeframes only
    symbols = ['ASML']#,'ASML','TSLA', "LULU","ORCL"]#,'SpY','QQQ','ETH','SMCI','BTC','NVDA']
    
    # Initialize the agent with valid timeframes (no 90m)
    multi_trader = MultiSymbolDayTraderAgent(
        symbols=symbols,
        timeframes=['1m','5m', '15m', '30m', '1h', '1d', '1wk','1mo']  # Removed 90m, 5d, 1wk for Alpaca compatibility
    )
    api_key = "PKYJLOK4LZBY56NZKXZLNSG665"
    secret_key = "4VVHMnrYEqVv4Jd1oMZMow15DrRVn5p8VD7eEK6TjYZ1"
    multi_trader.set_credentials(api_key=api_key, secret_key=secret_key)
    
    # Set trading parameters
    multi_trader.set_trading_parameters(
        buy_threshold=70,
        sell_threshold=70,
        high_confidence_only=True
    )
    
    # Print current status
    # multi_trader.print_status()
    
    # You would uncomment and set your actual Alpaca credentials here:
    # multi_trader.set_credentials('your_api_key', 'your_secret_key')
    
    print(f"\n🔍 Running analysis...")
    
    # Run sequential analysis (recommended for testing)
    results = multi_trader.run_sequential()
    # print("Resutls ", results)
    # Export results to file
    if results:
        a=multi_trader.export_results(results)
    print('aaaaa', a[1])
    
    # Uncomment to run parallel analysis instead:
    # thread = multi_trader.run_parallel()
    # 
    # try:
    #     while multi_trader.running:
    #         time.sleep(10)
    # except KeyboardInterrupt:
    #     print("\n🛑 Stopping...")
    #     multi_trader.stop()
