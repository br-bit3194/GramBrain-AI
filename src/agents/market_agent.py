"""Market Intelligence Agent - Provides market insights and price predictions."""

from typing import Dict, Any, List
from ..core.agent_base import Agent, Query, UserContext


class MarketAgent(Agent):
    """Provides market insights and price predictions."""
    
    def __init__(self):
        """Initialize market agent."""
        super().__init__("market_agent")
    
    async def analyze(self, query: Query, context: UserContext) -> Any:
        """Analyze market conditions and predict prices."""
        try:
            # Retrieve market knowledge
            rag_context = await self.retrieve_rag_context(
                f"market trends and pricing for {context.crop_type}",
                top_k=3
            )
            
            # Analyze market
            analysis = await self._analyze_market(context)
            
            # Generate recommendation
            recommendation = await self._generate_recommendation(
                analysis, context, rag_context
            )
            
            return self._create_output(
                query=query,
                analysis=analysis,
                recommendation=recommendation,
                confidence=analysis.get("confidence", 0.70),
                data_sources=["Agmarknet", "Commodity Exchange", "Regional Markets"],
                rag_context=rag_context,
                reasoning_chain=analysis.get("reasoning", []),
            )
        except Exception as e:
            return self._create_output(
                query=query,
                analysis={"error": str(e)},
                recommendation="Unable to analyze market conditions",
                confidence=0.0,
            )
    
    async def _analyze_market(self, context: UserContext) -> Dict[str, Any]:
        """Analyze market conditions and predict prices."""
        # Current market prices (INR/quintal) - mock data
        current_prices = {
            "wheat": 2200,
            "rice": 2800,
            "cotton": 5500,
            "pulses": 4500,
        }
        
        crop = context.crop_type or "wheat"
        current_price = current_prices.get(crop, 2500)
        
        reasoning = [
            f"Analyzing market for {crop}",
            f"Current price: ₹{current_price}/quintal",
        ]
        
        # Predict price trend (simplified)
        # Assume seasonal variation
        price_trend = "stable"
        predicted_price = current_price
        
        # Seasonal adjustments
        seasonal_factors = {
            "wheat": 1.05,  # Prices typically rise post-harvest
            "rice": 0.95,   # Prices typically fall post-harvest
            "cotton": 1.10, # Prices volatile
            "pulses": 1.08, # Prices typically rise
        }
        
        factor = seasonal_factors.get(crop, 1.0)
        predicted_price = current_price * factor
        
        if factor > 1.05:
            price_trend = "upward"
            reasoning.append("Price trend: Upward - favorable selling window")
        elif factor < 0.95:
            price_trend = "downward"
            reasoning.append("Price trend: Downward - consider storage")
        else:
            reasoning.append("Price trend: Stable")
        
        # Calculate price range
        lower_price = predicted_price * 0.90
        upper_price = predicted_price * 1.10
        
        reasoning.append(f"Predicted price: ₹{predicted_price:.0f}/quintal")
        reasoning.append(f"Range: ₹{lower_price:.0f} - ₹{upper_price:.0f}")
        
        return {
            "crop_type": crop,
            "current_price_per_quintal": current_price,
            "predicted_price_per_quintal": predicted_price,
            "lower_price_per_quintal": lower_price,
            "upper_price_per_quintal": upper_price,
            "price_trend": price_trend,
            "confidence": 0.70,
            "reasoning": reasoning,
        }
    
    async def _generate_recommendation(
        self,
        analysis: Dict[str, Any],
        context: UserContext,
        rag_context: List[str],
    ) -> str:
        """Generate market recommendation."""
        if not self.llm_client:
            trend = analysis.get("price_trend", "stable")
            if trend == "upward":
                return "Favorable market conditions. Consider selling soon."
            elif trend == "downward":
                return "Market prices declining. Consider storage or delayed selling."
            else:
                return "Market conditions stable. Sell based on your needs."
        
        prompt = f"""Provide market advice and selling recommendations.

Crop: {context.crop_type}
Current Price: ₹{analysis.get('current_price_per_quintal')}/quintal
Predicted Price: ₹{analysis.get('predicted_price_per_quintal')}/quintal
Price Trend: {analysis.get('price_trend')}
Price Range: ₹{analysis.get('lower_price_per_quintal'):.0f} - ₹{analysis.get('upper_price_per_quintal'):.0f}

Knowledge Context:
{rag_context[0] if rag_context else 'No specific knowledge available'}

Provide market recommendations including:
1. Optimal selling timing
2. Market channels (mandi, direct, cooperative)
3. Price negotiation tips"""
        
        try:
            return await self.call_llm(prompt, temperature=0.5, max_tokens=350)
        except Exception:
            trend = analysis.get("price_trend", "stable")
            if trend == "upward":
                return "Favorable market conditions. Consider selling soon."
            else:
                return "Market conditions stable. Sell based on your needs."
