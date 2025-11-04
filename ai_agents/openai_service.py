from datetime import datetime
from openai import OpenAI
import os
from typing import Dict, Any
import logging
from dotenv import load_dotenv
import json
from textwrap import dedent
import re
# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class OpenAIAnalysisService:
    def __init__(self):
        # Use YOUR API key from environment variables
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables. Please add your API key to .env file")
        
        # Validate API key format
        if not self.api_key.startswith(('sk-', 'sk-proj-')):
            raise ValueError("Invalid OpenAI API key format. Key should start with 'sk-' or 'sk-proj-'")
        
        # Initialize OpenAI client with new API format
        try:
            self.client = OpenAI(api_key=self.api_key)
            logger.info(f"OpenAI service initialized with API key: {self.api_key[:20]}...{self.api_key[-10:]}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
    
    def test_connection(self) -> Dict[str, Any]:
        """Test if OpenAI API key is working with detailed error info"""
        try:
            logger.info("Testing OpenAI connection...")
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=1,
                temperature=0
            )
            
            logger.info("OpenAI connection test successful")
            return {
                "success": True, 
                "message": "OpenAI API is working (v1.x)",
                "tokens_used": response.usage.total_tokens if response.usage else 1,
                "model": response.model,
                "api_key_preview": f"{self.api_key[:10]}...{self.api_key[-6:]}"
            }
        
        except Exception as e:
            error_str = str(e)
            logger.error(f"OpenAI connection test failed: {error_str}")
            
            # More detailed error analysis
            if "401" in error_str or "Unauthorized" in error_str:
                return {
                    "success": False, 
                    "error": "OpenAI API key is invalid or has been revoked. Please check your API key at https://platform.openai.com/api-keys",
                    "error_type": "auth_error",
                    "details": error_str
                }
            elif "403" in error_str or "Forbidden" in error_str:
                return {
                    "success": False,
                    "error": "OpenAI API access forbidden. Your account may need billing setup or may be suspended.",
                    "error_type": "forbidden",
                    "details": error_str
                }
            elif "429" in error_str or "rate_limit" in error_str.lower():
                return {
                    "success": False, 
                    "error": "OpenAI rate limit exceeded. Please wait and try again.",
                    "error_type": "rate_limit",
                    "details": error_str
                }
            elif "quota" in error_str.lower() or "billing" in error_str.lower() or "insufficient" in error_str.lower():
                return {
                    "success": False,
                    "error": "OpenAI quota exceeded or billing issue. Please add credits at https://platform.openai.com/account/billing",
                    "error_type": "quota_exceeded",
                    "details": error_str
                }
            else:
                return {
                    "success": False, 
                    "error": f"OpenAI connection failed: {error_str}",
                    "error_type": "unknown",
                    "details": error_str
                }
    
    def analyze_trading_data(self, structured_data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Analyze structured trading data and provide investment Strategy"""
        try:
            # Extract the actual trading result data
            trading_result = structured_data.get('trading_result', {})
            symbol_data = trading_result.get(symbol, {}) if trading_result else {}
            
            print(f"=== AI ANALYSIS DEBUG ===")
            print(f"Symbol data keys: {list(symbol_data.keys()) if symbol_data else 'No symbol data'}")
            
            if not symbol_data:
                print("No symbol data found, using fallback analysis")
                return self._generate_fallback_analysis(structured_data, symbol)
            
            # Extract consensus data
            consensus = symbol_data.get('consensus', {})
            decisions = symbol_data.get('decisions', {})
            
            print(f"Consensus: {consensus}")
            print(f"Decisions timeframes: {list(decisions.keys()) if decisions else 'No decisions'}")
            
            # Extract key metrics
            overall_recommendation = consensus.get('overall_recommendation', 'N/A')
            confidence_level = consensus.get('confidence', 'N/A')
            strength = consensus.get('strength', 0)
            buy_signals = consensus.get('buy_signals', 0)
            sell_signals = consensus.get('sell_signals', 0)
            reasoning = consensus.get('reasoning', [])
            
            # Extract entry price from any timeframe (they should be the same)
            entry_price = 'N/A'
            if decisions:
                first_decision = list(decisions.values())[0]
                entry_price = first_decision.get('entry_price', 'N/A')
                if hasattr(entry_price, 'item'):  # Handle numpy types
                    entry_price = float(entry_price.item())
            
            # Create a detailed analysis prompt with the actual data
            prompt = f"""
            
            You are an expert trading advisor. Based on the following technical analysis and stock data, give advice in the exact format below. Be concise and clear in each section:

            Result of the analysis for {symbol}:
            {symbol_data}

            Respond only in this format:
            1. Short-Term Trading (1-3 trades per day): 
            - Give one to three specific trade ideas or set-ups for very short-term (intraday) trades.

            2. Mid-Term Trading (1-3 trades per week): 
            - Give one to three trade setups or ideas for trades lasting a few days to a week.

            3. Long-Term/Hold Strategy: 
            - Give one to two recommendations for long-term investors, focusing on holding periods of weeks to months or longer.

            4. Sentiment Summary: 
            - In one sentence, what is the current sentiment for {symbol}?

            5. Fundamentals: 
            - In one sentence, summarize the key fundamental factor(s) affecting {symbol}.

            6. Major News: 
            - In one sentence, mention the most important recent news event about {symbol}, or say "No major news at this time" if nothing relevant.
            """
            prompt = dedent(prompt).strip()
            response = self.client.chat.completions.create(
                model="GPT-5 mini", #"gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
                temperature=0.7
            )

            
            analysis_text = response.choices[0].message.content
            sections = self._parse_trading_analysis_response(analysis_text)
            
            return {
                "success": True,
                "analysis": sections,
                "tokens_used": response.usage.total_tokens if response.usage else 0,
                "raw_response": analysis_text,
                "symbol": symbol,
                "timestamp": structured_data.get('timestamp', 'N/A'),
                "model": response.model,
                "data_source": "MultiSymbolDayTraderAgent_Detailed",
                "trading_summary": {
                    "recommendation": overall_recommendation,
                    "confidence": confidence_level,
                    "strength": strength,
                    "entry_price": entry_price,
                    "buy_signals": buy_signals,
                    "sell_signals": sell_signals,
                    "timeframes_analyzed": len(decisions)
                }
            }
            
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            # return self._generate_mock_analysis(structured_data, symbol, str(e))

    def _format_detailed_timeframe_analysis(self, decisions: Dict[str, Any]) -> str:
        """Format detailed timeframe-specific trading decisions"""
        if not decisions:
            return "No timeframe decisions available"
        
        analysis_parts = []
        for timeframe, decision in decisions.items():
            recommendation = decision.get('recommendation', 'N/A')
            confidence = decision.get('confidence', 'N/A')
            strength = decision.get('strength', 0)
            reasoning = decision.get('reasoning', [])
            
            # Handle numpy types for prices
            entry_price = self._safe_extract_price(decision.get('entry_price'))
            stop_loss = self._safe_extract_price(decision.get('stop_loss'))
            take_profit = self._safe_extract_price(decision.get('take_profit'))
            risk_reward = self._safe_extract_number(decision.get('risk_reward_ratio'))
            
            timeframe_analysis = f"""
            {timeframe.upper()} TIMEFRAME SIGNAL:
            • Signal: {recommendation} ({confidence} confidence)
            • Strength: {strength:.1f}%
            • Entry: {entry_price}
            • Stop Loss: {stop_loss}
            • Take Profit: {take_profit}
            • Risk/Reward: {risk_reward}
            • Technical Reasoning: {', '.join(reasoning) if reasoning else 'Standard technical analysis'}
            """.strip()
            
            analysis_parts.append(timeframe_analysis)
        
        return "\n\n".join(analysis_parts)

    def _format_risk_analysis(self, decisions: Dict[str, Any]) -> str:
        """Format risk analysis across timeframes"""
        if not decisions:
            return "No risk analysis data available"
        
        risk_parts = []
        total_risk_reward = 0
        valid_ratios = 0
        
        for timeframe, decision in decisions.items():
            risk_reward = self._safe_extract_number(decision.get('risk_reward_ratio'))
            if risk_reward != "Not available":
                total_risk_reward += float(risk_reward)
                valid_ratios += 1
                
                risk_level = "Low Risk" if float(risk_reward) > 0.5 else "Medium Risk" if float(risk_reward) > 0.3 else "High Risk"
                risk_parts.append(f"• {timeframe.upper()}: {risk_reward} R/R ({risk_level})")
        
        if valid_ratios > 0:
            avg_risk_reward = total_risk_reward / valid_ratios
            risk_parts.append(f"• Average Risk/Reward: {avg_risk_reward:.3f}")
            
            if avg_risk_reward > 0.4:
                risk_parts.append("• FAVORABLE risk profile - Good trade setup")
            elif avg_risk_reward > 0.25:
                risk_parts.append("• MODERATE risk profile - Acceptable with proper sizing")
            else:
                risk_parts.append("• HIGHER risk profile - Use smaller position size")
        
        return "\n".join(risk_parts) if risk_parts else "Risk analysis not available"

    def _safe_extract_price(self, price_value) -> str:
        """Safely extract price values handling numpy types"""
        if price_value is None:
            return "Not set"
        if hasattr(price_value, 'item'):  # numpy type
            return f"${float(price_value.item()):.2f}"
        if isinstance(price_value, (int, float)):
            return f"${float(price_value):.2f}"
        return str(price_value)

    def _safe_extract_number(self, number_value) -> str:
        """Safely extract numeric values handling numpy types"""
        if number_value is None:
            return "Not available"
        if hasattr(number_value, 'item'):  # numpy type
            return f"{float(number_value.item()):.3f}"
        if isinstance(number_value, (int, float)):
            return f"{float(number_value):.3f}"
        return str(number_value)

    def _parse_trading_analysis_response(self, response_text: str) -> Dict[str, str]:
        """Parse AI response into structured sections for trading analysis"""
        sections = {
            "short_term": re.compile(r"short[- ]?term trading.*:", re.IGNORECASE),
            "mid_term": re.compile(r"mid[- ]?term trading.*:", re.IGNORECASE),
            "long_term": re.compile(r"long[- ]?term.*(hold|strategy).*:", re.IGNORECASE),
            "sentiment": re.compile(r"sentiment summary.*:", re.IGNORECASE),
            "fundamentals": re.compile(r"fundamentals.*:", re.IGNORECASE),
            "major_news": re.compile(r"major news.*:", re.IGNORECASE),
        }

        result = {key: None for key in sections}
        current_section = None

        for line in response_text.splitlines():
            line_strip = line.strip()
            if not line_strip:
                continue

            for key, pattern in sections.items():
                if pattern.match(line_strip):
                    current_section = key
                    content = pattern.sub("", line_strip).strip(" -:\t")
                    if content:
                        result[key] = content
                    break
            else:
                if current_section:
                    result[current_section] = (
                        (result[current_section] + " " if result[current_section] else "")
                        + line_strip
                    )

        return result
        # """Parse the AI response into structured sections for trading analysis"""
        # sections = {
        #     "signal_interpretation": "",
        #     "entry_strategy": "",
        #     "risk_management": "",
        #     "timeframe_alignment": "",
        #     "action_plan": "",
        #     "summary": ""
        # }
        
        # lines = response_text.split('\n')
        # current_section = "summary"  # default
        
        # for line in lines:
        #     line = line.strip()
        #     if not line:
        #         continue
                
        #     # Detect section headers
        #     lower_line = line.lower()
        #     if any(keyword in lower_line for keyword in ["signal interpretation", "signal analysis", "interpreting"]):
        #         current_section = "signal_interpretation"
        #     elif any(keyword in lower_line for keyword in ["entry strategy", "entry plan", "entry point"]):
        #         current_section = "entry_strategy"
        #     elif any(keyword in lower_line for keyword in ["risk management", "risk", "position sizing"]):
        #         current_section = "risk_management"
        #     elif any(keyword in lower_line for keyword in ["timeframe alignment", "timeframe", "alignment"]):
        #         current_section = "timeframe_alignment"
        #     elif any(keyword in lower_line for keyword in ["action plan", "next steps", "immediate", "execute"]):
        #         current_section = "action_plan"
        #     elif any(keyword in lower_line for keyword in ["summary", "conclusion", "recommendation"]):
        #         current_section = "summary"
        #     else:
        #         # Add content to current section
        #         if sections[current_section]:
        #             sections[current_section] += "\n" + line
        #         else:
        #             sections[current_section] = line
        
        # # If parsing failed, put everything in summary
        # if not any(sections.values()):
        #     sections["summary"] = response_text
        
        # return sections

    def _generate_fallback_analysis(self, structured_data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Generate analysis when symbol data is not properly structured"""
        return {
            "success": True,
            "analysis": {
                "signal_interpretation": f"Unable to extract specific trading signals for {symbol} from the provided data structure.",
                "entry_strategy": "Please verify that the trading analysis completed successfully and contains signal data.",
                "risk_management": "Cannot provide specific risk management without signal data.",
                "timeframe_alignment": "Timeframe analysis not available in current data format.",
                "action_plan": "1. Check trading agent configuration\n2. Verify data extraction process\n3. Ensure all timeframes are analyzed",
                "summary": f"Trading analysis for {symbol} completed but signal extraction failed. Please check the data structure and try again."
            },
            "tokens_used": 0,
            "raw_response": "Fallback analysis due to data structure issues",
            "symbol": symbol,
            "timestamp": structured_data.get('timestamp', 'N/A'),
            "is_fallback": True,
            "model": "fallback-analysis",
            "data_source": "fallback"
        }

    def analyze_trading_data(self, raw_data: Any, symbol: str) -> Dict[str, Any]:
        """Analyze raw trading data and provide AI investment Strategy"""
        language = "en"  # Default to English; can be parameterized later
        language_map = {
            "en": "English",
            "tr": "Turkish",
            "es": "Spanish",
            "ja": "Japanese",
            "zh": "Chinese",
            "fr": "French",
            "de": "German",
            "ko": "Korean",
            # Add more languages as needed
        }
        language_name = language_map.get(language, "English")
         # Convert raw data to string for the AI prompt
        data_str = str(raw_data)
        if language_name == "English":
            system_message = (
                f"""You are an expert day trading advisor giveing pricise investment strategies based on trading data. You can focus on actions for multiple timeframes. For example, when short-term (5m, 15m) is bearish medium-term (1h, 4h, 1d) is bullish
                 and long-term (1w, 1mo) bearish you can come with stragies like: "Focus on short-term bearish andd there will turning points in the midterm term support areas for trading. Check again when the price close the long term resistance levels
                You can provide strategise like this at last sentence of your answer.
                """
                # f"Always reply ONLY in {language_name}. "
                # f"Do not use any English (even for labels, formatting, or explanations) unless {language_name} is English."
            )
            user_message = f"""
                Analyze the following trading data for {symbol} and provide an investment strategy. Focus on actionable insights and format your answer as described below.

                RAW TRADING DATA:
                {data_str}

                Based on the trading analysis data for {symbol}, here are specific actionable investment Strategy for day trading:

                1. Short-term (Couple times trading in a day):
                - For the 1-minute, 5-minute, and 15-minute timeframes, the recommendation is to BUY {symbol} with high confidence levels.
                - Entry price: $137.44
                - Stop loss: $137.2068 for 1m, $136.9796 for 5m, $134.5254 for 15m
                - Take profit: $137.5864 for 1m, $138.24 for 5m and 15m
                - Risk/reward ratio: 0.63 for 1m, 1.74 for 5m, 0.27 for 15m

                2. Medium to Long-term (Couple tradings in a week):
                - For the 1-hour, daily, and weekly timeframes, the recommendation is also to BUY {symbol} with high confidence levels.
                - Entry price: $137.44
                - Stop loss: $136.495 for 1h, $127.425 for 1d
                - Take profit: $141.8038 for 1h, $143.0989 for 1d
                - Risk/reward ratio: 4.62 for 1h, 0.57 for 1d

                3. Long-term (Couple tradings in months):
                - For the weekly timeframes, the recommendation is also to BUY {symbol} with high confidence levels.
                - Entry price: $137.44
                - Stop loss: $137.0022 for 1wk
                - Take profit: $158.7256 for 1wk
                - Risk/reward ratio: 48.62 for 1wk

                Based on the data provided, it is recommended to focus on short to medium-term buying opportunities for {symbol} with high confidence levels. Ensure to set appropriate entry prices, st
                """
        elif language_name == "Turkish":
            system_message = (
                f"Bir uzman gün içi ticaret danışmanısınız ve çok dilli konuşuyorsunuz."
                f"Cevaplarınızı SADECE {language_name} dilinde verin. "
                f"{language_name} İngilizce değilse, İngilizce kullanmayın (etiketler, formatlama veya açıklamalar dahil)."
            )
            user_message = f"""
                Aşağıdaki {symbol} için ticaret verilerini analiz edin ve bir yatırım stratejisi sağlayın. Eyleme geçirilebilir içgörülere odaklanın ve cevabınızı aşağıda açıklanan şekilde formatlayın.

                HAM TİCARET VERİLERİ:
                {data_str}

                {symbol} için sağlanan ticaret analiz verilerine dayanarak, işte gün içi ticaret için belirli eyleme geçirilebilir yatırım stratejileri:

                1. Kısa vadeli (Günde birkaç kez işlem):
                - 1 dakikalık, 5 dakikalık ve 15 dakikalık zaman dilimleri için öneri YUKARI ALIŞ'tır yüksek güven seviyeleri ile.
                - Giriş fiyatı: $137.44
                - Stop loss: 1m için $137.2068, 5m için $136.9796, 15m için $134.5254
                - Kar al: 1m için $137.5864, 5m ve 15m için $138.24
                - Risk/ödül oranı: 1m için 0.63, 5m için 1.74, 15m için 0.27

                2. Orta ila uzun vadeli (Haftada birkaç işlem):
                - 1 saatlik, günlük ve haftalık zaman dilimleri için öneri de YUKARI ALIŞ'tır yüksek güven seviyeleri ile.
                - Giriş fiyatı: $137.44
                - Stop loss: 1h için $136.495, 1d için $127.425
                - Kar al: 1h için $141.8038, 1d için $143.0989
                - Risk/ödül oranı: 1h için 4.62, 1d için 0.57

                3. Uzun vadeli (Ayda birkaç işlem):
                - Haftalık zaman dilimleri için öneri de YUKARI ALIŞ'tır yüksek güven seviyeleri ile.
                """

        try:
           
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=1500,
                temperature=0.7
            )
            
            analysis_text = response.choices[0].message.content
            # Simple approach - put everything in summary for now
            sections = {
                "signal_analysis": "",
                "entry_strategy": "",
                "risk_management": "",
                "position_sizing": "",
                "timeframe_strategy": "",
                "action_items": "",
                "summary": analysis_text  # Put the full response here
            }
            
            return {
                "success": True,
                "analysis": sections,
                "tokens_used": response.usage.total_tokens if response.usage else 0,
                "raw_response": analysis_text,
                "symbol": symbol,
                "timestamp": str(datetime.now()),
                "model": response.model,
                "data_source": "raw_trading_data"
            }
            
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            return {
                "success": False,
                "error": f"AI analysis failed: {str(e)}",
                "symbol": symbol,
                "timestamp": str(datetime.now())
            }
        #     # Parse the response into sections
        #     sections = self._parse_simple_analysis(analysis_text)
            
        #     return {
        #         "success": True,
        #         "analysis": sections,
        #         "tokens_used": response.usage.total_tokens if response.usage else 0,
        #         "raw_response": analysis_text,
        #         "symbol": symbol,
        #         "timestamp": str(datetime.now()),
        #         "model": response.model,
        #         "data_source": "raw_trading_data"
        #     }
            
        # except Exception as e:
        #     logger.error(f"AI analysis failed: {e}")
        #     return {
        #         "success": False,
        #         "error": f"AI analysis failed: {str(e)}",
        #         "symbol": symbol,
        #         "timestamp": str(datetime.now())
        #     }


    def _parse_simple_analysis(self, response_text: str) -> Dict[str, str]:
        """Parse AI response into sections"""
        sections = {
            "signal_analysis": "",
            "entry_strategy": "",
            "risk_management": "",
            "position_sizing": "",
            "timeframe_strategy": "",
            "action_items": "",
            "summary": ""
        }
        
        lines = response_text.split('\n')
        current_section = "summary"
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Detect section headers
            lower_line = line.lower()
            if "signal analysis" in lower_line:
                current_section = "signal_analysis"
            elif "entry strategy" in lower_line:
                current_section = "entry_strategy"
            elif "risk management" in lower_line:
                current_section = "risk_management"
            elif "position sizing" in lower_line:
                current_section = "position_sizing"
            elif "timeframe strategy" in lower_line:
                current_section = "timeframe_strategy"
            elif "action items" in lower_line:
                current_section = "action_items"
            elif "summary" in lower_line:
                current_section = "summary"
            else:
                # Add content to current section
                if sections[current_section]:
                    sections[current_section] += "\n" + line
                else:
                    sections[current_section] = line
        
        # If parsing failed, put everything in summary
        if not any(sections.values()):
            sections["summary"] = response_text
        
        return sections

    
# Create a global instance
openai_service = OpenAIAnalysisService()

# from openai import OpenAI
# import os
# from typing import Dict, Any
# import logging
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# logger = logging.getLogger(__name__)

# class OpenAIAnalysisService:
#     def __init__(self):
#         # Use YOUR API key from environment variables
#         self.api_key = os.getenv("OPENAI_API_KEY")
#         if not self.api_key:
#             raise ValueError("OPENAI_API_KEY not found in environment variables. Please add your API key to .env file")
        
#         # Initialize OpenAI client with new API format
#         self.client = OpenAI(api_key=self.api_key)
#         logger.info("OpenAI service initialized with centralized API key (v1.x)")
    
#     def test_connection(self) -> Dict[str, Any]:
#         """Test if OpenAI API key is working"""
#         try:
#             response = self.client.chat.completions.create(
#                 model="gpt-3.5-turbo",
#                 messages=[{"role": "user", "content": "Test"}],
#                 max_tokens=1,
#                 temperature=0
#             )
            
#             return {
#                 "success": True, 
#                 "message": "OpenAI API is working (v1.x)",
#                 "tokens_used": response.usage.total_tokens if response.usage else 1,
#                 "model": response.model
#             }
        
#         except Exception as e:
#             error_str = str(e).lower()
            
#             # Handle specific error types
#             if "rate_limit" in error_str or "429" in error_str:
#                 return {
#                     "success": False, 
#                     "error": "OpenAI rate limit exceeded. Please wait and try again.",
#                     "error_type": "rate_limit"
#                 }
#             elif "quota" in error_str or "billing" in error_str or "insufficient" in error_str:
#                 return {
#                     "success": False,
#                     "error": "OpenAI quota exceeded or billing issue. Please add credits to your OpenAI account at https://platform.openai.com/account/billing",
#                     "error_type": "quota_exceeded"
#                 }
#             elif "authentication" in error_str or "api_key" in error_str or "401" in error_str:
#                 return {
#                     "success": False,
#                     "error": "OpenAI API key authentication failed. Please check your API key.",
#                     "error_type": "auth_error"
#                 }
#             else:
#                 logger.error(f"OpenAI connection test failed: {e}")
#                 return {
#                     "success": False, 
#                     "error": f"OpenAI connection failed: {str(e)}",
#                     "error_type": "unknown"
#                 }
    
# def analyze_trading_data(self, trading_data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
#     """Analyze trading data and provide investment Strategy"""
#     try:
#         # Extract trading result if present
#         trading_result = trading_data.get('trading_result', {})
        
#         # Create a comprehensive prompt
#         prompt = f"""
#         As an expert financial advisor, analyze this day trading analysis result for {symbol} and provide detailed investment Strategy.
        
#         SYMBOL: {symbol}
#         ANALYSIS TYPE: {trading_data.get('analysis_type', 'Day Trading Analysis')}
#         TIMESTAMP: {trading_data.get('timestamp', 'N/A')}
        
#         TRADING PARAMETERS:
#         {self._format_trading_parameters(trading_data.get('trading_parameters', {}))}
        
#         TRADING ANALYSIS RESULT:
#         {self._format_trading_result(trading_result)}
        
#         EXTRACTED METRICS:
#         - Current Price: {trading_data.get('current_price', 'N/A')}
#         - Price Change: {trading_data.get('price_change', 'N/A')}
#         - Volume: {trading_data.get('volume', 'N/A')}
#         - Market Sentiment: {trading_data.get('market_sentiment', 'N/A')}
        
#         SIGNALS & RECOMMENDATIONS:
#         {self._format_signals_and_recommendations(trading_data)}
        
#         Please provide a structured analysis with:
        
#         1. **Trading Analysis Summary**: Interpret the day trading analysis results
#         2. **Short-term Strategy (1-7 days)**: Based on the trading signals and technical analysis
#         3. **Risk Assessment**: Evaluate the trading recommendations and associated risks
#         4. **Action Items**: Specific recommendations based on the trading analysis
#         5. **Summary**: Overall assessment and next steps
        
#         Base your analysis on the actual trading data and signals provided above.
#         """
        
#         response = self.client.chat.completions.create(
#             model="gpt-3.5-turbo",
#             messages=[{"role": "user", "content": prompt}],
#             max_tokens=1500,
#             temperature=0.7
#         )
        
#         analysis_text = response.choices[0].message.content
#         sections = self._parse_analysis_response(analysis_text)
        
#         return {
#             "success": True,
#             "analysis": sections,
#             "tokens_used": response.usage.total_tokens if response.usage else 0,
#             "raw_response": analysis_text,
#             "symbol": symbol,
#             "timestamp": trading_data.get('timestamp', 'N/A'),
#             "model": response.model,
#             "data_source": "day_trading_analysis"
#         }
        
#     except Exception as e:
#         logger.error(f"AI analysis failed: {e}")
#         return self._generate_mock_analysis(trading_data, symbol, str(e))

# def _format_trading_parameters(self, params: Dict[str, Any]) -> str:
#     """Format trading parameters for the prompt"""
#     if not params:
#         return "- No trading parameters provided"
    
#     formatted = []
#     for key, value in params.items():
#         formatted.append(f"- {key.replace('_', ' ').title()}: {value}")
    
#     return "\n".join(formatted)

# def _format_trading_result(self, result: Dict[str, Any]) -> str:
#     """Format trading result for the prompt"""
#     if not result:
#         return "- No trading result data available"
    
#     # Convert dict to readable format
#     import json
#     try:
#         return json.dumps(result, indent=2)[:1000] + "..." if len(str(result)) > 1000 else json.dumps(result, indent=2)
#     except:
#         return str(result)[:1000] + "..." if len(str(result)) > 1000 else str(result)

# def _format_signals_and_recommendations(self, trading_data: Dict[str, Any]) -> str:
#     """Format signals and recommendations"""
#     signals = trading_data.get('signals', {})
#     recommendations = trading_data.get('recommendations', {})
    
#     formatted = []
    
#     if signals:
#         formatted.append("SIGNALS:")
#         for key, value in signals.items():
#             formatted.append(f"- {key}: {value}")
    
#     if recommendations:
#         formatted.append("\nRECOMMENDATIONS:")
#         for key, value in recommendations.items():
#             formatted.append(f"- {key}: {value}")
    
#     return "\n".join(formatted) if formatted else "- No signals or recommendations available"

# def _generate_mock_analysis(self, trading_data: Dict[str, Any], symbol: str, error_msg: str) -> Dict[str, Any]:
#     """Generate mock analysis when OpenAI API is not available"""
    
#     # Extract some basic info for mock analysis
#     current_price = trading_data.get('current_price', 'N/A')
#     price_change = trading_data.get('price_change', 'N/A')
    
#     mock_analysis = {
#         "market_position": f"Based on technical analysis of {symbol}, the current market position shows mixed signals. The stock is trading at ${current_price} with recent price movement of {price_change}. Market sentiment appears cautious with moderate volatility expected.",
        
#         "short_term_strategy": f"Short-term outlook for {symbol} (1-30 days):\n• Monitor key support and resistance levels around current price\n• Watch for volume confirmation on any breakouts\n• Consider position sizing based on current volatility\n• Set stop-loss orders at 5-8% below entry point\n• Look for reversal patterns if price approaches key levels",
        
#         "long_term_strategy": f"Long-term strategy for {symbol} (3-12 months):\n• Evaluate fundamental company metrics and earnings growth\n• Consider sector trends and competitive positioning\n• Diversify portfolio to reduce single-stock concentration risk\n• Review quarterly earnings reports and forward guidance\n• Monitor industry developments and regulatory changes",
        
#         "risk_assessment": f"Risk factors for {symbol}:\n• Market volatility may impact short-term performance significantly\n• Sector-specific risks should be carefully evaluated\n• Economic conditions and interest rate changes may affect valuation\n• Individual stock risk requires proper position sizing (max 5-10% of portfolio)\n• Liquidity risk during market stress periods",
        
#         "action_items": f"Recommended actions for {symbol}:\n• Conduct thorough fundamental analysis before major positions\n• Set clear entry and exit criteria with specific price targets\n• Implement proper risk management with stop-losses\n• Monitor key technical levels: support, resistance, moving averages\n• Stay updated on company news, earnings, and sector developments\n• Review position size relative to overall portfolio risk",
        
#         "summary": f"Mock Analysis Summary for {symbol}: This demonstration analysis shows the type of insights available with AI-powered analysis. The stock shows mixed technical signals requiring careful risk management. For real-time AI insights with current market data, please ensure your OpenAI API account has sufficient credits. Always combine technical analysis with fundamental research for optimal investment decisions."
#     }
    
#     return {
#         "success": True,
#         "analysis": mock_analysis,
#         "tokens_used": 0,
#         "raw_response": "Mock analysis generated due to API limitations",
#         "symbol": symbol,
#         "timestamp": trading_data.get('timestamp', 'N/A'),
#         "is_mock": True,
#         "api_error": error_msg,
#         "model": "mock-gpt-3.5-turbo"
#     }

# def _parse_analysis_response(self, response_text: str) -> Dict[str, str]:
#         """Parse the AI response into structured sections"""
#         sections = {
#             "market_position": "",
#             "short_term_strategy": "",
#             "long_term_strategy": "",
#             "risk_assessment": "",
#             "action_items": "",
#             "summary": ""
#         }
        
#         # Simple parsing - you can make this more sophisticated
#         lines = response_text.split('\n')
#         current_section = "summary"  # default
        
#         for line in lines:
#             line = line.strip()
#             if not line:
#                 continue
                
#             # Detect section headers (case insensitive)
#             lower_line = line.lower()
#             if any(keyword in lower_line for keyword in ["market position", "position assessment", "market sentiment"]):
#                 current_section = "market_position"
#             elif any(keyword in lower_line for keyword in ["short-term", "short term", "immediate", "near term"]):
#                 current_section = "short_term_strategy"
#             elif any(keyword in lower_line for keyword in ["long-term", "long term", "strategic", "future"]):
#                 current_section = "long_term_strategy"
#             elif any(keyword in lower_line for keyword in ["risk", "risks", "risk assessment", "risk factors"]):
#                 current_section = "risk_assessment"
#             elif any(keyword in lower_line for keyword in ["action", "recommendation", "action items", "next steps"]):
#                 current_section = "action_items"
#             elif any(keyword in lower_line for keyword in ["summary", "conclusion", "overall"]):
#                 current_section = "summary"
#             else:
#                 # Add content to current section
#                 if sections[current_section]:
#                     sections[current_section] += "\n" + line
#                 else:
#                     sections[current_section] = line
        
#         # If parsing failed, put everything in summary
#         if not any(sections.values()):
#             sections["summary"] = response_text
        
#         return sections

# # Create a global instance
# openai_service = OpenAIAnalysisService()