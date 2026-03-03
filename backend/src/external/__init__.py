"""External API integration using FastMCP tools."""

from .base import FastMCPTool, ToolResult, ToolError
from .weather_client import WeatherAPIClient
from .satellite_client import SatelliteAPIClient
from .government_client import GovernmentAPIClient

__all__ = [
    'FastMCPTool',
    'ToolResult',
    'ToolError',
    'WeatherAPIClient',
    'SatelliteAPIClient',
    'GovernmentAPIClient',
]
