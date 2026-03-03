# app/aws_integration/tools/weather_tools.py
"""Weather tools for AWS integration - reusing existing weather service"""
from typing import Dict, Any
import os
import sys

# Import from existing services
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from google_adk_integration.services.weather_service import WeatherService

# Initialize weather service
weather_service = WeatherService(api_key=os.environ.get("WEATHER_API_KEY", ""))


async def get_weather_forecast(location: str, days: int = 5) -> Dict[str, Any]:
    """
    Fetch weather forecast for agricultural planning
    
    Args:
        location: Location name (city, district)
        days: Number of days for forecast (1-10)
    
    Returns:
        Dict with forecast data and farming advice
    """
    return await weather_service.get_forecast(location, days)


async def get_current_weather(location: str) -> Dict[str, Any]:
    """
    Fetch current weather conditions
    
    Args:
        location: Location name (city, district)
    
    Returns:
        Dict with current weather and farming advice
    """
    return await weather_service.get_current_weather(location)
