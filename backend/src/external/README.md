# External API Integration (FastMCP)

This module provides FastMCP-based integration with external APIs for weather, satellite imagery, and government data.

## Overview

The external API integration follows the FastMCP (Fast Model Context Protocol) pattern with:
- **Retry Logic**: Exponential backoff with jitter
- **Circuit Breaker**: Automatic failure detection and recovery
- **Rate Limiting**: Token bucket algorithm
- **Fallback Mechanisms**: Graceful degradation when APIs fail
- **Structured Logging**: All API calls logged with context

## Components

### Base Classes

#### `FastMCPTool`
Base class for all external API tools with built-in:
- Retry with exponential backoff
- Circuit breaker pattern
- Rate limiting
- Logging and metrics

#### `CircuitBreaker`
Implements circuit breaker pattern with three states:
- **CLOSED**: Normal operation
- **OPEN**: Service unavailable, fail fast
- **HALF_OPEN**: Testing if service recovered

#### `RateLimiter`
Token bucket rate limiter to prevent API quota exhaustion.

### API Clients

#### `WeatherAPIClient`
Integrates with weather APIs:
- **OpenWeather API**: Current weather and forecasts
- **IMD API**: India Meteorological Department data
- **Fallback**: Returns cached/mock data when APIs fail

**Tools**:
- `CurrentWeatherTool`: Get current weather conditions
- `WeatherForecastTool`: Get 7-day weather forecast
- `IMDWeatherTool`: Get IMD weather data

#### `SatelliteAPIClient`
Integrates with satellite imagery APIs:
- **Sentinel-2**: European Space Agency satellite data
- **NDVI**: Vegetation health index
- **Fallback**: Returns mock data when API fails

**Tools**:
- `NDVITool`: Get vegetation index data
- `SatelliteImageryTool`: Get satellite imagery

#### `GovernmentAPIClient`
Integrates with Indian government APIs:
- **Agmarknet**: Agricultural market prices
- **Soil Health Cards**: Soil nutrient data
- **Fallback**: Returns mock data when API fails

**Tools**:
- `MarketPricesTool`: Get commodity market prices
- `SoilHealthTool`: Get soil health card data

## Usage

### Basic Usage

```python
from backend.src.external import WeatherAPIClient

# Initialize client
weather_client = WeatherAPIClient(
    openweather_api_key="your-api-key"
)

# Get current weather
weather = await weather_client.get_current_weather(
    lat=28.6139,
    lon=77.2090
)

# Get forecast
forecast = await weather_client.get_forecast(
    lat=28.6139,
    lon=77.2090,
    days=7
)
```

### With Agents

```python
from backend.src.external import WeatherAPIClient
from backend.src.agents.weather_agent import WeatherAgent

# Initialize client
weather_client = WeatherAPIClient()

# Use in agent
class EnhancedWeatherAgent(WeatherAgent):
    def __init__(self):
        super().__init__()
        self.weather_client = weather_client
    
    async def _fetch_weather_data(self, context):
        # Use real API instead of mock data
        return await self.weather_client.get_current_weather(
            lat=context.farm_location["lat"],
            lon=context.farm_location["lon"]
        )
```

### Tool Registration

```python
# Get all tools for registration with agents
weather_tools = weather_client.get_tools()
satellite_tools = satellite_client.get_tools()
government_tools = government_client.get_tools()

# Register with agent
for tool in weather_tools:
    agent.register_tool(tool)
```

## Configuration

Set environment variables:

```bash
# Weather APIs
OPENWEATHER_API_KEY=your-openweather-key
IMD_API_KEY=your-imd-key

# Satellite APIs
SENTINEL_API_KEY=your-sentinel-key

# Government APIs
AGMARKNET_API_KEY=your-agmarknet-key
```

## Error Handling

All tools return `ToolResult` objects:

```python
result = await tool.execute(lat=28.6139, lon=77.2090)

if result.success:
    data = result.data
    print(f"Execution time: {result.execution_time_ms}ms")
else:
    error = result.error
    print(f"Failed after {result.retry_count} retries: {error}")
```

## Circuit Breaker

Circuit breaker automatically opens after 5 consecutive failures:

```python
# Circuit breaker states
# CLOSED: Normal operation
# OPEN: Fail fast, don't call API
# HALF_OPEN: Testing recovery

# Configure circuit breaker
tool = CurrentWeatherTool()
tool.circuit_breaker.failure_threshold = 5
tool.circuit_breaker.timeout_seconds = 60
```

## Rate Limiting

Rate limiting prevents quota exhaustion:

```python
# Configure rate limit
tool = CurrentWeatherTool()
tool.rate_limiter.requests_per_second = 10.0

# Rate limiter automatically queues requests
```

## Retry Configuration

Customize retry behavior:

```python
tool = CurrentWeatherTool()
tool.max_retries = 3
tool.retry_base_delay_ms = 100
tool.retry_max_delay_ms = 5000
```

## Fallback Mechanisms

All clients provide fallback data when APIs fail:

```python
# Automatic fallback
weather = await weather_client.get_current_weather(
    lat=28.6139,
    lon=77.2090,
    use_fallback=True  # Default
)

# Fallback data includes source indicator
if weather["source"] == "fallback":
    print("Using cached/mock data")
```

## Logging

All API calls are logged with structured context:

```json
{
  "tool_name": "get_current_weather",
  "execution_time_ms": 245.3,
  "retry_count": 0,
  "params": {"lat": 28.6139, "lon": 77.2090},
  "success": true
}
```

## Testing

Mock external APIs in tests:

```python
import pytest
from unittest.mock import AsyncMock

@pytest.mark.asyncio
async def test_weather_agent():
    # Mock weather client
    weather_client = AsyncMock()
    weather_client.get_current_weather.return_value = {
        "temperature_celsius": 28.0,
        "rainfall_mm": 0.0,
    }
    
    # Test agent with mock
    agent = WeatherAgent()
    agent.weather_client = weather_client
    
    result = await agent.analyze(query, context)
    assert result.confidence > 0.5
```

## API Schemas

All tools provide JSON schemas for validation:

```python
schema = tool.get_schema()
# {
#   "name": "get_current_weather",
#   "description": "Get current weather conditions",
#   "parameters": {
#     "type": "object",
#     "properties": {...},
#     "required": ["lat", "lon"]
#   }
# }
```

## Requirements Validation

This implementation validates:
- **Requirement 6.2**: Tool registration with schemas ✓
- **Requirement 6.3**: Retry and fallback mechanisms ✓
- **Requirement 6.4**: Logging for all API calls ✓
- **Requirement 6.5**: Backoff and queueing for rate limits ✓

## Future Enhancements

- OAuth2 support for Sentinel Hub
- Caching layer for API responses
- Metrics collection for monitoring
- Webhook support for real-time updates
- Additional government API integrations
