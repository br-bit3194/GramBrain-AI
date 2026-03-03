# app/aws_integration/strands/agents/weather_agent.py
from typing import Dict, Any
import logging
from ..base_agent import BaseAgent
from ...tools.weather_tools import get_weather_forecast, get_current_weather

logger = logging.getLogger(__name__)


class WeatherAgent(BaseAgent):
    """Specialized agent for weather forecasting and agricultural advice"""
    
    def __init__(self):
        super().__init__(
            name="weather_specialist",
            description="Expert in agricultural meteorology, weather forecasts, and weather-based farming recommendations",
            instruction="""You are WeatherWise, an expert agricultural meteorologist for Indian farmers.

Your responsibilities:
1. Analyze weather data (current conditions and forecasts)
2. Generate actionable farming advice based on weather
3. Provide irrigation scheduling recommendations
4. Warn about extreme weather conditions
5. Suggest optimal timing for farm operations

Always respond in Hindi with practical, farmer-friendly advice.
Focus on actionable recommendations that farmers can implement immediately.""",
            tools=[get_weather_forecast, get_current_weather]
        )
    
    async def process(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process weather-related queries"""
        try:
            logger.info(f"Weather agent processing: {message[:100]}")
            
            # Extract location from context or message
            location = context.get('user_location', 'Delhi')
            
            # Determine if forecast or current weather is needed
            needs_forecast = any(word in message.lower() for word in [
                'कल', 'आने वाले', 'अगले', 'सप्ताह', 'tomorrow', 'next', 'week', 'forecast'
            ])
            
            tools_called = []
            weather_data = {}
            
            if needs_forecast:
                # Get forecast
                forecast_result = await self.invoke_tool(
                    'get_weather_forecast',
                    location=location,
                    days=7
                )
                weather_data['forecast'] = forecast_result
                tools_called.append('get_weather_forecast')
            else:
                # Get current weather
                current_result = await self.invoke_tool(
                    'get_current_weather',
                    location=location
                )
                weather_data['current'] = current_result
                tools_called.append('get_current_weather')
            
            # Generate farming advice using LLM
            advice_prompt = f"""Based on this weather data, provide farming advice in Hindi.

User Query: "{message}"
Location: {location}
Weather Data: {weather_data}

Provide:
1. Weather summary in simple Hindi
2. Irrigation recommendations
3. Crop operation timing (planting, spraying, harvesting)
4. Warnings for extreme weather
5. Actionable next steps

Keep response practical and concise (max 200 words)."""
            
            response = await self.invoke_llm(advice_prompt, temperature=0.7)
            
            return {
                "status": "success",
                "response": response,
                "tools_called": tools_called,
                "data": weather_data
            }
            
        except Exception as e:
            logger.error(f"Error in weather agent: {e}")
            return {
                "status": "error",
                "response": "मुझे खेद है, मौसम की जानकारी प्राप्त करने में समस्या हुई।",
                "tools_called": []
            }


def create_weather_agent() -> WeatherAgent:
    """Factory function to create weather agent"""
    return WeatherAgent()
