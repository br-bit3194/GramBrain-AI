"""Weather API client using FastMCP tools."""

import os
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import httpx

from .base import FastMCPTool, ToolError


class WeatherAPIClient:
    """Client for weather APIs (IMD, OpenWeather) using FastMCP tools."""
    
    def __init__(
        self,
        imd_api_key: Optional[str] = None,
        openweather_api_key: Optional[str] = None,
    ):
        """
        Initialize weather API client.
        
        Args:
            imd_api_key: IMD API key (from env if not provided)
            openweather_api_key: OpenWeather API key (from env if not provided)
        """
        self.imd_api_key = imd_api_key or os.getenv('IMD_API_KEY')
        self.openweather_api_key = openweather_api_key or os.getenv('OPENWEATHER_API_KEY')
        
        # Initialize tools
        self.current_weather_tool = CurrentWeatherTool(
            api_key=self.openweather_api_key
        )
        self.forecast_tool = WeatherForecastTool(
            api_key=self.openweather_api_key
        )
        self.imd_tool = IMDWeatherTool(
            api_key=self.imd_api_key
        )
    
    async def get_current_weather(
        self,
        lat: float,
        lon: float,
        use_fallback: bool = True
    ) -> Dict[str, Any]:
        """
        Get current weather for location.
        
        Args:
            lat: Latitude
            lon: Longitude
            use_fallback: Use fallback if primary fails
            
        Returns:
            Current weather data
        """
        # Try OpenWeather first
        result = await self.current_weather_tool.execute(lat=lat, lon=lon)
        
        if result.success:
            return result.data
        
        # Fallback to IMD if available
        if use_fallback and self.imd_api_key:
            result = await self.imd_tool.execute(lat=lat, lon=lon, data_type="current")
            if result.success:
                return result.data
        
        # Return cached/mock data as last resort
        return self._get_fallback_weather(lat, lon)
    
    async def get_forecast(
        self,
        lat: float,
        lon: float,
        days: int = 7,
        use_fallback: bool = True
    ) -> Dict[str, Any]:
        """
        Get weather forecast for location.
        
        Args:
            lat: Latitude
            lon: Longitude
            days: Number of days to forecast
            use_fallback: Use fallback if primary fails
            
        Returns:
            Weather forecast data
        """
        result = await self.forecast_tool.execute(lat=lat, lon=lon, days=days)
        
        if result.success:
            return result.data
        
        # Fallback to IMD
        if use_fallback and self.imd_api_key:
            result = await self.imd_tool.execute(lat=lat, lon=lon, data_type="forecast")
            if result.success:
                return result.data
        
        return self._get_fallback_forecast(lat, lon, days)
    
    def _get_fallback_weather(self, lat: float, lon: float) -> Dict[str, Any]:
        """Get fallback weather data when APIs fail."""
        return {
            "location": {"lat": lat, "lon": lon},
            "temperature_celsius": 28.0,
            "humidity_percent": 65,
            "rainfall_mm": 0.0,
            "wind_speed_kmph": 10.0,
            "description": "Data unavailable - using fallback",
            "timestamp": datetime.now().isoformat(),
            "source": "fallback",
        }
    
    def _get_fallback_forecast(self, lat: float, lon: float, days: int) -> Dict[str, Any]:
        """Get fallback forecast data when APIs fail."""
        forecast = []
        for i in range(days):
            date = datetime.now() + timedelta(days=i)
            forecast.append({
                "date": date.date().isoformat(),
                "temperature_celsius": 28.0,
                "rainfall_mm": 0.0,
                "humidity_percent": 65,
            })
        
        return {
            "location": {"lat": lat, "lon": lon},
            "forecast": forecast,
            "source": "fallback",
        }
    
    def get_tools(self) -> List[FastMCPTool]:
        """Get all weather tools for registration."""
        return [
            self.current_weather_tool,
            self.forecast_tool,
            self.imd_tool,
        ]


class CurrentWeatherTool(FastMCPTool):
    """Tool for fetching current weather from OpenWeather API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize current weather tool."""
        super().__init__(
            tool_name="get_current_weather",
            max_retries=3,
            rate_limit_rps=10.0,
        )
        self.api_key = api_key
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"
    
    async def _execute(self, lat: float, lon: float) -> Dict[str, Any]:
        """
        Fetch current weather.
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            Current weather data
            
        Raises:
            ToolError: If API call fails
        """
        if not self.api_key:
            raise ToolError("OpenWeather API key not configured")
        
        params = {
            "lat": lat,
            "lon": lon,
            "appid": self.api_key,
            "units": "metric",
        }
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.get(self.base_url, params=params)
                response.raise_for_status()
                data = response.json()
                
                # Transform to standard format
                return {
                    "location": {"lat": lat, "lon": lon},
                    "temperature_celsius": data["main"]["temp"],
                    "humidity_percent": data["main"]["humidity"],
                    "rainfall_mm": data.get("rain", {}).get("1h", 0.0),
                    "wind_speed_kmph": data["wind"]["speed"] * 3.6,  # m/s to km/h
                    "description": data["weather"][0]["description"],
                    "timestamp": datetime.now().isoformat(),
                    "source": "openweather",
                }
            
            except httpx.HTTPStatusError as e:
                raise ToolError(f"OpenWeather API error: {e.response.status_code}")
            except httpx.RequestError as e:
                raise ToolError(f"OpenWeather request failed: {str(e)}")
            except (KeyError, ValueError) as e:
                raise ToolError(f"Failed to parse OpenWeather response: {str(e)}")
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema."""
        return {
            "name": "get_current_weather",
            "description": "Get current weather conditions for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "lat": {
                        "type": "number",
                        "description": "Latitude of the location",
                    },
                    "lon": {
                        "type": "number",
                        "description": "Longitude of the location",
                    },
                },
                "required": ["lat", "lon"],
            },
        }


class WeatherForecastTool(FastMCPTool):
    """Tool for fetching weather forecast from OpenWeather API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize weather forecast tool."""
        super().__init__(
            tool_name="get_weather_forecast",
            max_retries=3,
            rate_limit_rps=10.0,
        )
        self.api_key = api_key
        self.base_url = "https://api.openweathermap.org/data/2.5/forecast"
    
    async def _execute(self, lat: float, lon: float, days: int = 7) -> Dict[str, Any]:
        """
        Fetch weather forecast.
        
        Args:
            lat: Latitude
            lon: Longitude
            days: Number of days to forecast
            
        Returns:
            Weather forecast data
            
        Raises:
            ToolError: If API call fails
        """
        if not self.api_key:
            raise ToolError("OpenWeather API key not configured")
        
        params = {
            "lat": lat,
            "lon": lon,
            "appid": self.api_key,
            "units": "metric",
            "cnt": min(days * 8, 40),  # 3-hour intervals, max 5 days
        }
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.get(self.base_url, params=params)
                response.raise_for_status()
                data = response.json()
                
                # Aggregate by day
                daily_forecast = {}
                for item in data["list"]:
                    date = datetime.fromtimestamp(item["dt"]).date().isoformat()
                    if date not in daily_forecast:
                        daily_forecast[date] = {
                            "date": date,
                            "temperature_celsius": [],
                            "rainfall_mm": [],
                            "humidity_percent": [],
                        }
                    
                    daily_forecast[date]["temperature_celsius"].append(item["main"]["temp"])
                    daily_forecast[date]["rainfall_mm"].append(
                        item.get("rain", {}).get("3h", 0.0)
                    )
                    daily_forecast[date]["humidity_percent"].append(item["main"]["humidity"])
                
                # Average values
                forecast = []
                for date_str, values in sorted(daily_forecast.items())[:days]:
                    forecast.append({
                        "date": date_str,
                        "temperature_celsius": sum(values["temperature_celsius"]) / len(values["temperature_celsius"]),
                        "rainfall_mm": sum(values["rainfall_mm"]),
                        "humidity_percent": sum(values["humidity_percent"]) / len(values["humidity_percent"]),
                    })
                
                return {
                    "location": {"lat": lat, "lon": lon},
                    "forecast": forecast,
                    "source": "openweather",
                }
            
            except httpx.HTTPStatusError as e:
                raise ToolError(f"OpenWeather API error: {e.response.status_code}")
            except httpx.RequestError as e:
                raise ToolError(f"OpenWeather request failed: {str(e)}")
            except (KeyError, ValueError) as e:
                raise ToolError(f"Failed to parse OpenWeather response: {str(e)}")
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema."""
        return {
            "name": "get_weather_forecast",
            "description": "Get weather forecast for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "lat": {
                        "type": "number",
                        "description": "Latitude of the location",
                    },
                    "lon": {
                        "type": "number",
                        "description": "Longitude of the location",
                    },
                    "days": {
                        "type": "integer",
                        "description": "Number of days to forecast (1-7)",
                        "default": 7,
                    },
                },
                "required": ["lat", "lon"],
            },
        }


class IMDWeatherTool(FastMCPTool):
    """Tool for fetching weather from India Meteorological Department API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize IMD weather tool."""
        super().__init__(
            tool_name="get_imd_weather",
            max_retries=3,
            rate_limit_rps=5.0,  # IMD has stricter limits
        )
        self.api_key = api_key
        # Note: IMD API endpoint would be configured here
        # Using placeholder as actual IMD API details may vary
        self.base_url = "https://api.imd.gov.in/weather"
    
    async def _execute(
        self,
        lat: float,
        lon: float,
        data_type: str = "current"
    ) -> Dict[str, Any]:
        """
        Fetch weather from IMD.
        
        Args:
            lat: Latitude
            lon: Longitude
            data_type: "current" or "forecast"
            
        Returns:
            Weather data
            
        Raises:
            ToolError: If API call fails
        """
        if not self.api_key:
            raise ToolError("IMD API key not configured")
        
        # Note: This is a placeholder implementation
        # Actual IMD API integration would require specific endpoint details
        raise ToolError("IMD API integration not yet implemented")
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema."""
        return {
            "name": "get_imd_weather",
            "description": "Get weather data from India Meteorological Department",
            "parameters": {
                "type": "object",
                "properties": {
                    "lat": {
                        "type": "number",
                        "description": "Latitude of the location",
                    },
                    "lon": {
                        "type": "number",
                        "description": "Longitude of the location",
                    },
                    "data_type": {
                        "type": "string",
                        "enum": ["current", "forecast"],
                        "description": "Type of weather data",
                        "default": "current",
                    },
                },
                "required": ["lat", "lon"],
            },
        }
