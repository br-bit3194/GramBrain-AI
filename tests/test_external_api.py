"""Tests for external API integration."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime

from backend.src.external.base import (
    FastMCPTool,
    ToolResult,
    ToolError,
    CircuitBreaker,
    RateLimiter,
    CircuitState,
)
from backend.src.external.weather_client import (
    WeatherAPIClient,
    CurrentWeatherTool,
    WeatherForecastTool,
)
from backend.src.external.satellite_client import (
    SatelliteAPIClient,
    NDVITool,
)
from backend.src.external.government_client import (
    GovernmentAPIClient,
    MarketPricesTool,
)


class TestCircuitBreaker:
    """Test circuit breaker functionality."""
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_failures(self):
        """Test that circuit breaker opens after threshold failures."""
        cb = CircuitBreaker(failure_threshold=3, timeout_seconds=60)
        
        async def failing_func():
            raise Exception("API error")
        
        # Should be closed initially
        assert cb.state == CircuitState.CLOSED
        
        # Fail 3 times
        for _ in range(3):
            with pytest.raises(Exception):
                await cb.call(failing_func)
        
        # Should be open now
        assert cb.state == CircuitState.OPEN
        
        # Next call should fail fast
        with pytest.raises(ToolError, match="Circuit breaker is OPEN"):
            await cb.call(failing_func)
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_success_resets(self):
        """Test that successful calls reset failure count."""
        cb = CircuitBreaker(failure_threshold=3)
        
        async def sometimes_failing_func(should_fail):
            if should_fail:
                raise Exception("API error")
            return "success"
        
        # Fail twice
        for _ in range(2):
            with pytest.raises(Exception):
                await cb.call(sometimes_failing_func, True)
        
        assert cb.failure_count == 2
        
        # Succeed once
        result = await cb.call(sometimes_failing_func, False)
        assert result == "success"
        assert cb.failure_count == 1  # Decremented


class TestRateLimiter:
    """Test rate limiter functionality."""
    
    @pytest.mark.asyncio
    async def test_rate_limiter_allows_requests(self):
        """Test that rate limiter allows requests within limit."""
        limiter = RateLimiter(requests_per_second=10.0)
        
        # Should allow immediate request
        await limiter.acquire()
        assert limiter.tokens < 10.0
    
    @pytest.mark.asyncio
    async def test_rate_limiter_queues_excess_requests(self):
        """Test that rate limiter queues excess requests."""
        limiter = RateLimiter(requests_per_second=2.0)
        
        # Consume all tokens
        await limiter.acquire()
        await limiter.acquire()
        
        # Next request should wait
        import time
        start = time.time()
        await limiter.acquire()
        elapsed = time.time() - start
        
        # Should have waited at least some time
        assert elapsed > 0


class TestWeatherAPIClient:
    """Test weather API client."""
    
    @pytest.mark.asyncio
    async def test_get_current_weather_success(self):
        """Test successful weather fetch."""
        client = WeatherAPIClient(openweather_api_key="test-key")
        
        # Mock the tool execution
        mock_result = ToolResult(
            success=True,
            data={
                "location": {"lat": 28.6139, "lon": 77.2090},
                "temperature_celsius": 28.5,
                "humidity_percent": 65,
                "rainfall_mm": 0.0,
                "wind_speed_kmph": 12.0,
                "description": "clear sky",
                "timestamp": datetime.now().isoformat(),
                "source": "openweather",
            },
            tool_name="get_current_weather",
        )
        
        client.current_weather_tool.execute = AsyncMock(return_value=mock_result)
        
        result = await client.get_current_weather(lat=28.6139, lon=77.2090)
        
        assert result["temperature_celsius"] == 28.5
        assert result["source"] == "openweather"
    
    @pytest.mark.asyncio
    async def test_get_current_weather_fallback(self):
        """Test weather fetch with fallback."""
        client = WeatherAPIClient(openweather_api_key="test-key")
        
        # Mock failed execution
        mock_result = ToolResult(
            success=False,
            error="API error",
            tool_name="get_current_weather",
        )
        
        client.current_weather_tool.execute = AsyncMock(return_value=mock_result)
        client.imd_tool.execute = AsyncMock(return_value=mock_result)
        
        result = await client.get_current_weather(lat=28.6139, lon=77.2090)
        
        # Should return fallback data
        assert result["source"] == "fallback"
        assert "temperature_celsius" in result
    
    @pytest.mark.asyncio
    async def test_get_forecast_success(self):
        """Test successful forecast fetch."""
        client = WeatherAPIClient(openweather_api_key="test-key")
        
        mock_result = ToolResult(
            success=True,
            data={
                "location": {"lat": 28.6139, "lon": 77.2090},
                "forecast": [
                    {"date": "2026-01-15", "temperature_celsius": 28.0, "rainfall_mm": 0.0},
                    {"date": "2026-01-16", "temperature_celsius": 29.0, "rainfall_mm": 5.0},
                ],
                "source": "openweather",
            },
            tool_name="get_weather_forecast",
        )
        
        client.forecast_tool.execute = AsyncMock(return_value=mock_result)
        
        result = await client.get_forecast(lat=28.6139, lon=77.2090, days=7)
        
        assert len(result["forecast"]) == 2
        assert result["source"] == "openweather"


class TestSatelliteAPIClient:
    """Test satellite API client."""
    
    @pytest.mark.asyncio
    async def test_get_ndvi_fallback(self):
        """Test NDVI fetch with fallback."""
        client = SatelliteAPIClient(sentinel_api_key="test-key")
        
        # Mock failed execution (Sentinel requires OAuth2)
        mock_result = ToolResult(
            success=False,
            error="OAuth2 not configured",
            tool_name="get_ndvi_data",
        )
        
        client.ndvi_tool.execute = AsyncMock(return_value=mock_result)
        
        result = await client.get_ndvi_data(lat=28.6139, lon=77.2090)
        
        # Should return fallback data
        assert result["source"] == "fallback"
        assert "ndvi_value" in result


class TestGovernmentAPIClient:
    """Test government API client."""
    
    @pytest.mark.asyncio
    async def test_get_market_prices_fallback(self):
        """Test market prices fetch with fallback."""
        client = GovernmentAPIClient(agmarknet_api_key="test-key")
        
        # Mock failed execution
        mock_result = ToolResult(
            success=False,
            error="API error",
            tool_name="get_market_prices",
        )
        
        client.market_prices_tool.execute = AsyncMock(return_value=mock_result)
        
        result = await client.get_market_prices(commodity="wheat")
        
        # Should return fallback data
        assert result["source"] == "fallback"
        assert result["commodity"] == "wheat"


class TestToolSchemas:
    """Test tool schema generation."""
    
    def test_current_weather_tool_schema(self):
        """Test current weather tool schema."""
        tool = CurrentWeatherTool(api_key="test-key")
        schema = tool.get_schema()
        
        assert schema["name"] == "get_current_weather"
        assert "description" in schema
        assert "parameters" in schema
        assert "lat" in schema["parameters"]["properties"]
        assert "lon" in schema["parameters"]["properties"]
        assert schema["parameters"]["required"] == ["lat", "lon"]
    
    def test_market_prices_tool_schema(self):
        """Test market prices tool schema."""
        tool = MarketPricesTool(api_key="test-key")
        schema = tool.get_schema()
        
        assert schema["name"] == "get_market_prices"
        assert "commodity" in schema["parameters"]["properties"]
        assert "commodity" in schema["parameters"]["required"]


class TestToolRetry:
    """Test tool retry mechanism."""
    
    @pytest.mark.asyncio
    async def test_tool_retries_on_failure(self):
        """Test that tools retry on transient failures."""
        
        class TestTool(FastMCPTool):
            def __init__(self):
                super().__init__("test_tool", max_retries=3)
                self.attempt_count = 0
            
            async def _execute(self):
                self.attempt_count += 1
                if self.attempt_count < 3:
                    raise Exception("Transient error")
                return {"success": True}
            
            def get_schema(self):
                return {"name": "test_tool"}
        
        tool = TestTool()
        result = await tool.execute()
        
        assert result.success
        assert result.retry_count >= 1  # At least one retry occurred
        assert tool.attempt_count == 3
    
    @pytest.mark.asyncio
    async def test_tool_fails_after_max_retries(self):
        """Test that tools fail after max retries."""
        
        class FailingTool(FastMCPTool):
            def __init__(self):
                super().__init__("failing_tool", max_retries=2)
            
            async def _execute(self):
                raise Exception("Permanent error")
            
            def get_schema(self):
                return {"name": "failing_tool"}
        
        tool = FailingTool()
        result = await tool.execute()
        
        assert not result.success
        assert result.retry_count == 2
        assert "Permanent error" in result.error


class TestToolLogging:
    """Test tool logging."""
    
    @pytest.mark.asyncio
    async def test_tool_logs_success(self, caplog):
        """Test that successful tool execution is logged."""
        
        class LoggingTool(FastMCPTool):
            def __init__(self):
                super().__init__("logging_tool")
            
            async def _execute(self, param1):
                return {"result": param1}
            
            def get_schema(self):
                return {"name": "logging_tool"}
        
        tool = LoggingTool()
        
        with caplog.at_level("INFO"):
            result = await tool.execute(param1="test")
        
        assert result.success
        # Check that logging occurred (logger name includes tool name)
        assert any("logging_tool" in record.name for record in caplog.records)
    
    @pytest.mark.asyncio
    async def test_tool_logs_failure(self, caplog):
        """Test that failed tool execution is logged."""
        
        class FailingLogTool(FastMCPTool):
            def __init__(self):
                super().__init__("failing_log_tool", max_retries=0)
            
            async def _execute(self):
                raise Exception("Test error")
            
            def get_schema(self):
                return {"name": "failing_log_tool"}
        
        tool = FailingLogTool()
        
        with caplog.at_level("ERROR"):
            result = await tool.execute()
        
        assert not result.success
        # Check that error logging occurred
        assert any("ERROR" in record.levelname for record in caplog.records)
