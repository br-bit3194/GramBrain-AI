"""Weather Intelligence Agent - Analyzes weather data and forecasts."""

from typing import Dict, Any, List
from datetime import datetime
from ..core.agent_base import Agent, AgentOutput, Query, UserContext


class WeatherAgent(Agent):
    """Analyzes weather data and provides weather-aware recommendations."""
    
    def __init__(self):
        """Initialize weather agent."""
        super().__init__("weather_agent")
    
    async def analyze(self, query: Query, context: UserContext) -> AgentOutput:
        """
        Analyze weather conditions and generate recommendations.
        
        Args:
            query: User query
            context: User context with farm location
            
        Returns:
            Weather analysis and recommendations
        """
        try:
            # Fetch weather data
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
                data_sources=["IMD Weather API", "GFS Forecast"],
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
        Fetch weather data for farm location.
        
        Args:
            context: User context with location
            
        Returns:
            Weather data dictionary
        """
        # TODO: Integrate with IMD API and GFS
        # For now, return mock data
        return {
            "location": context.farm_location,
            "temperature_celsius": 28.5,
            "rainfall_mm": 15.0,
            "humidity_percent": 65,
            "wind_speed_kmph": 12,
            "forecast_days": 7,
            "rainfall_forecast_mm": [0, 5, 12, 8, 0, 0, 0],
            "temperature_forecast": [28, 29, 27, 26, 28, 29, 30],
            "confidence": 0.85,
        }
    
    async def _analyze_irrigation_need(
        self,
        weather_data: Dict[str, Any],
        context: UserContext,
    ) -> Dict[str, Any]:
        """
        Analyze if irrigation is needed based on weather forecast.
        
        Args:
            weather_data: Weather data
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
        }
        
        daily_need = crop_water_needs.get(context.crop_type, 500) / 120  # mm/day
        
        reasoning = [
            f"Analyzed weather forecast for {context.crop_type}",
            f"Expected rainfall in next 3 days: {total_rainfall}mm",
            f"Daily water requirement: {daily_need:.1f}mm",
        ]
        
        skip_irrigation = total_rainfall >= daily_need * 3
        water_savings = max(0, total_rainfall - daily_need * 3)
        
        if skip_irrigation:
            reasoning.append(f"Sufficient rainfall expected - skip irrigation")
        else:
            reasoning.append(f"Rainfall insufficient - irrigation recommended")
        
        return {
            "skip_irrigation": skip_irrigation,
            "rainfall_expected_mm": total_rainfall,
            "water_savings_liters": water_savings * (context.farm_size_hectares or 1) * 10000,
            "confidence": weather_data.get("confidence", 0.7),
            "reasoning": reasoning,
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
                return "Skip irrigation for the next 3 days due to expected rainfall"
            else:
                return "Irrigate within the next 2 days before rainfall"
        
        prompt = f"""Based on weather analysis, provide a concise irrigation recommendation.

Weather Data:
- Current Temperature: {weather_data.get('temperature_celsius')}°C
- Expected Rainfall (3 days): {analysis.get('rainfall_expected_mm')}mm
- Humidity: {weather_data.get('humidity_percent')}%
- Crop: {context.crop_type}
- Growth Stage: {context.growth_stage}

Analysis:
- Skip Irrigation: {analysis.get('skip_irrigation')}
- Potential Water Savings: {analysis.get('water_savings_liters'):.0f} liters

Knowledge Context:
{rag_context[0] if rag_context else 'No specific knowledge available'}

Provide a clear, actionable recommendation in 1-2 sentences."""
        
        try:
            recommendation = await self.call_llm(prompt, temperature=0.5, max_tokens=200)
            return recommendation.strip()
        except Exception:
            # Fallback
            if analysis.get("skip_irrigation"):
                return "Skip irrigation for the next 3 days due to expected rainfall"
            else:
                return "Irrigate within the next 2 days"
