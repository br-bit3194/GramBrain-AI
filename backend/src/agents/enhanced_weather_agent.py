"""Enhanced Weather Intelligence Agent with FastMCP integration."""

from typing import Dict, Any, List, Optional
from datetime import datetime
from ..core.agent_base import Agent, AgentOutput, Query, UserContext
from ..external.weather_client import WeatherAPIClient


class EnhancedWeatherAgent(Agent):
    """
    Enhanced weather agent using FastMCP external API integration.
    
    This agent demonstrates how to integrate FastMCP tools with the agent framework.
    """
    
    def __init__(self, weather_client: Optional[WeatherAPIClient] = None):
        """
        Initialize enhanced weather agent.
        
        Args:
            weather_client: WeatherAPIClient instance (creates new if not provided)
        """
        super().__init__("enhanced_weather_agent")
        self.weather_client = weather_client or WeatherAPIClient()
    
    async def analyze(self, query: Query, context: UserContext) -> AgentOutput:
        """
        Analyze weather conditions using real external APIs.
        
        Args:
            query: User query
            context: User context with farm location
            
        Returns:
            Weather analysis and recommendations
        """
        try:
            # Fetch real weather data using FastMCP tools
            weather_data = await self._fetch_weather_data(context)
            
            # Retrieve RAG context about weather impacts
            rag_context = await self.retrieve_rag_context(
                f"weather impact on {context.crop_type} farming",
                top_k=3
            )
            
            # Analyze weather for irrigation impact
            analysis = await self._analyze_irrigation_need(weather_data, context)
            
            # Generate recommendation using LLM
            recommendation = await self._generate_recommendation(
                analysis, weather_data, context, rag_context
            )
            
            return self._create_output(
                query=query,
                analysis=analysis,
                recommendation=recommendation,
                confidence=analysis.get("confidence", 0.7),
                data_sources=weather_data.get("data_sources", ["Weather API"]),
                rag_context=rag_context,
                reasoning_chain=analysis.get("reasoning", []),
            )
        
        except Exception as e:
            # Return degraded output on error
            return self._create_output(
                query=query,
                analysis={"error": str(e)},
                recommendation="Unable to analyze weather due to data unavailability",
                confidence=0.0,
                data_sources=[],
                rag_context=[],
            )
    
    async def _fetch_weather_data(self, context: UserContext) -> Dict[str, Any]:
        """
        Fetch weather data using FastMCP weather client.
        
        Args:
            context: User context with location
            
        Returns:
            Weather data dictionary with current and forecast data
        """
        if not context.farm_location:
            raise ValueError("Farm location not provided")
        
        lat = context.farm_location.get("lat")
        lon = context.farm_location.get("lon")
        
        if lat is None or lon is None:
            raise ValueError("Invalid farm location coordinates")
        
        # Fetch current weather
        current_weather = await self.weather_client.get_current_weather(
            lat=lat,
            lon=lon,
            use_fallback=True
        )
        
        # Fetch forecast
        forecast = await self.weather_client.get_forecast(
            lat=lat,
            lon=lon,
            days=7,
            use_fallback=True
        )
        
        # Combine data
        return {
            "location": context.farm_location,
            "temperature_celsius": current_weather.get("temperature_celsius"),
            "rainfall_mm": current_weather.get("rainfall_mm"),
            "humidity_percent": current_weather.get("humidity_percent"),
            "wind_speed_kmph": current_weather.get("wind_speed_kmph"),
            "description": current_weather.get("description"),
            "forecast_days": len(forecast.get("forecast", [])),
            "rainfall_forecast_mm": [
                day.get("rainfall_mm", 0.0) 
                for day in forecast.get("forecast", [])
            ],
            "temperature_forecast": [
                day.get("temperature_celsius", 0.0)
                for day in forecast.get("forecast", [])
            ],
            "confidence": 0.85 if current_weather.get("source") != "fallback" else 0.5,
            "data_sources": [current_weather.get("source", "unknown")],
        }
    
    async def _analyze_irrigation_need(
        self,
        weather_data: Dict[str, Any],
        context: UserContext,
    ) -> Dict[str, Any]:
        """
        Analyze if irrigation is needed based on weather forecast.
        
        Args:
            weather_data: Weather data from API
            context: User context
            
        Returns:
            Analysis results
        """
        rainfall_forecast = weather_data.get("rainfall_forecast_mm", [])
        total_rainfall = sum(rainfall_forecast[:3])  # Next 3 days
        
        # Crop water requirements (simplified)
        crop_water_needs = {
            "wheat": 450,
            "rice": 1200,
            "cotton": 700,
            "pulses": 400,
            "vegetables": 500,
        }
        
        daily_need = crop_water_needs.get(context.crop_type, 500) / 120  # mm/day
        
        reasoning = [
            f"Analyzed weather forecast for {context.crop_type}",
            f"Expected rainfall in next 3 days: {total_rainfall:.1f}mm",
            f"Daily water requirement: {daily_need:.1f}mm",
        ]
        
        skip_irrigation = total_rainfall >= daily_need * 3
        water_savings = max(0, total_rainfall - daily_need * 3)
        
        if skip_irrigation:
            reasoning.append(f"Sufficient rainfall expected - skip irrigation")
        else:
            deficit = (daily_need * 3) - total_rainfall
            reasoning.append(f"Rainfall insufficient - irrigation recommended ({deficit:.1f}mm deficit)")
        
        return {
            "skip_irrigation": skip_irrigation,
            "rainfall_expected_mm": total_rainfall,
            "water_savings_liters": water_savings * (context.farm_size_hectares or 1) * 10000,
            "confidence": weather_data.get("confidence", 0.7),
            "reasoning": reasoning,
            "current_temperature": weather_data.get("temperature_celsius"),
            "current_humidity": weather_data.get("humidity_percent"),
        }
    
    async def _generate_recommendation(
        self,
        analysis: Dict[str, Any],
        weather_data: Dict[str, Any],
        context: UserContext,
        rag_context: List[str],
    ) -> str:
        """
        Generate weather recommendation using LLM.
        
        Args:
            analysis: Weather analysis
            weather_data: Weather data
            context: User context
            rag_context: Retrieved knowledge
            
        Returns:
            Recommendation text
        """
        if not self.llm_client:
            # Fallback recommendation
            if analysis.get("skip_irrigation"):
                return f"Skip irrigation for the next 3 days due to expected rainfall of {analysis.get('rainfall_expected_mm'):.1f}mm"
            else:
                return "Irrigate within the next 2 days before rainfall"
        
        prompt = f"""Based on weather analysis, provide a concise irrigation recommendation.

Weather Data:
- Current Temperature: {weather_data.get('temperature_celsius')}°C
- Current Humidity: {weather_data.get('humidity_percent')}%
- Current Conditions: {weather_data.get('description')}
- Expected Rainfall (3 days): {analysis.get('rainfall_expected_mm'):.1f}mm
- Crop: {context.crop_type}
- Growth Stage: {context.growth_stage}
- Farm Size: {context.farm_size_hectares} hectares

Analysis:
- Skip Irrigation: {analysis.get('skip_irrigation')}
- Potential Water Savings: {analysis.get('water_savings_liters'):.0f} liters

Knowledge Context:
{rag_context[0] if rag_context else 'No specific knowledge available'}

Provide a clear, actionable recommendation in 2-3 sentences. Include specific timing and water quantity if irrigation is needed."""
        
        try:
            recommendation = await self.call_llm(prompt, temperature=0.5, max_tokens=200)
            return recommendation.strip()
        except Exception:
            # Fallback
            if analysis.get("skip_irrigation"):
                return f"Skip irrigation for the next 3 days due to expected rainfall of {analysis.get('rainfall_expected_mm'):.1f}mm"
            else:
                return "Irrigate within the next 2 days"
    
    def get_registered_tools(self) -> List[Dict[str, Any]]:
        """
        Get list of registered FastMCP tools.
        
        Returns:
            List of tool schemas
        """
        tools = self.weather_client.get_tools()
        return [tool.get_schema() for tool in tools]
