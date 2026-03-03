# app/aws_integration/strands/agents/market_agent.py
from typing import Dict, Any
import logging
from ..base_agent import BaseAgent
from ...tools.market_tools import get_market_prices, get_price_analysis, get_selling_advice

logger = logging.getLogger(__name__)


class MarketAgent(BaseAgent):
    """Specialized agent for market prices and selling advice"""
    
    def __init__(self):
        super().__init__(
            name="market_specialist",
            description="Agricultural market expert providing real-time prices, analysis, and selling advice from government data",
            instruction="""You are a comprehensive agricultural market expert helping Indian farmers maximize their profits.

Your responsibilities:
1. Provide current market prices from government data
2. Analyze price trends and patterns
3. Recommend optimal selling strategies
4. Identify best markets for selling
5. Calculate potential revenue

Always respond in Hindi with practical, actionable advice.
Focus on helping farmers get the best prices for their produce.""",
            tools=[get_market_prices, get_price_analysis, get_selling_advice]
        )
    
    async def process(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process market-related queries"""
        try:
            logger.info(f"Market agent processing: {message[:100]}")
            
            # Extract commodity from message
            commodity = self._extract_commodity(message)
            
            if not commodity:
                return {
                    "status": "error",
                    "response": "कृपया फसल का नाम बताएं (जैसे: प्याज, टमाटर, आलू)",
                    "tools_called": []
                }
            
            # Extract location if available
            state = context.get('user_location')
            
            # Determine query type
            needs_analysis = any(word in message.lower() for word in [
                'विश्लेषण', 'ट्रेंड', 'analysis', 'trend', 'पूर्वानुमान'
            ])
            
            needs_selling_advice = any(word in message.lower() for word in [
                'बेचना', 'बेचूं', 'sell', 'कब', 'when', 'सलाह'
            ])
            
            tools_called = []
            market_data = {}
            
            # Get market prices
            prices_result = await self.invoke_tool(
                'get_market_prices',
                commodity=commodity,
                state=state,
                days=7
            )
            market_data['prices'] = prices_result
            tools_called.append('get_market_prices')
            
            # Get additional data based on query type
            if needs_analysis:
                analysis_result = await self.invoke_tool(
                    'get_price_analysis',
                    commodity=commodity,
                    analysis_days=30
                )
                market_data['analysis'] = analysis_result
                tools_called.append('get_price_analysis')
            
            if needs_selling_advice:
                advice_result = await self.invoke_tool(
                    'get_selling_advice',
                    commodity=commodity,
                    quality_grade='medium',
                    urgency='normal'
                )
                market_data['selling_advice'] = advice_result
                tools_called.append('get_selling_advice')
            
            # Generate comprehensive response
            response_prompt = f"""Based on this market data, provide comprehensive advice in Hindi.

User Query: "{message}"
Commodity: {commodity}
Location: {state or 'India'}
Market Data: {market_data}

Provide:
1. Current price summary
2. Price trends (up/down/stable)
3. Best markets to sell
4. Selling recommendations
5. Expected revenue (if quantity mentioned)

Keep response practical and easy to understand (max 250 words)."""
            
            response = await self.invoke_llm(response_prompt, temperature=0.7)
            
            return {
                "status": "success",
                "response": response,
                "tools_called": tools_called,
                "data": market_data
            }
            
        except Exception as e:
            logger.error(f"Error in market agent: {e}")
            return {
                "status": "error",
                "response": "मुझे खेद है, मार्केट की जानकारी प्राप्त करने में समस्या हुई।",
                "tools_called": []
            }
    
    def _extract_commodity(self, message: str) -> str:
        """Extract commodity name from message"""
        # Common Hindi-English commodity mappings
        commodity_map = {
            'प्याज': 'Onion',
            'टमाटर': 'Tomato',
            'आलू': 'Potato',
            'गेहूं': 'Wheat',
            'चावल': 'Rice',
            'धान': 'Paddy',
            'मक्का': 'Maize',
            'सोयाबीन': 'Soybean',
            'onion': 'Onion',
            'tomato': 'Tomato',
            'potato': 'Potato',
            'wheat': 'Wheat',
            'rice': 'Rice'
        }
        
        message_lower = message.lower()
        for hindi_name, english_name in commodity_map.items():
            if hindi_name in message_lower:
                return english_name
        
        return ""


def create_market_agent() -> MarketAgent:
    """Factory function to create market agent"""
    return MarketAgent()
