"""
Example integration of FastMCP external APIs with GramBrain agents.

This module demonstrates how to:
1. Initialize external API clients
2. Register tools with agents
3. Use tools in agent analysis
4. Handle fallbacks and errors
"""

import asyncio
from typing import Dict, Any, List

from ..config import config
from ..external.weather_client import WeatherAPIClient
from ..external.satellite_client import SatelliteAPIClient
from ..external.government_client import GovernmentAPIClient
from ..agents.enhanced_weather_agent import EnhancedWeatherAgent
from ..core.agent_base import Query, UserContext


async def initialize_external_clients() -> Dict[str, Any]:
    """
    Initialize all external API clients with configuration.
    
    Returns:
        Dictionary of initialized clients
    """
    # Initialize weather client
    weather_client = WeatherAPIClient(
        openweather_api_key=config.external_api.openweather_api_key,
        imd_api_key=config.external_api.imd_api_key,
    )
    
    # Initialize satellite client
    satellite_client = SatelliteAPIClient(
        sentinel_api_key=config.external_api.sentinel_api_key,
    )
    
    # Initialize government API client
    government_client = GovernmentAPIClient(
        agmarknet_api_key=config.external_api.agmarknet_api_key,
    )
    
    return {
        "weather": weather_client,
        "satellite": satellite_client,
        "government": government_client,
    }


def register_tools_with_agents(
    agents: List[Any],
    clients: Dict[str, Any]
) -> None:
    """
    Register FastMCP tools with agents.
    
    Args:
        agents: List of agent instances
        clients: Dictionary of external API clients
    """
    # Get all tools from clients
    all_tools = []
    
    if "weather" in clients:
        all_tools.extend(clients["weather"].get_tools())
    
    if "satellite" in clients:
        all_tools.extend(clients["satellite"].get_tools())
    
    if "government" in clients:
        all_tools.extend(clients["government"].get_tools())
    
    # Register tools with each agent
    for agent in agents:
        if hasattr(agent, 'register_tools'):
            for tool in all_tools:
                agent.register_tools(tool)
        
        # Store tool schemas for reference
        if hasattr(agent, 'available_tools'):
            agent.available_tools = [tool.get_schema() for tool in all_tools]


async def example_weather_analysis():
    """
    Example: Use weather agent with real external APIs.
    """
    print("=== Weather Analysis Example ===\n")
    
    # Initialize clients
    clients = await initialize_external_clients()
    
    # Create enhanced weather agent
    agent = EnhancedWeatherAgent(weather_client=clients["weather"])
    
    # Create query and context
    query = Query(
        text="Should I irrigate my wheat field today?",
        user_id="farmer_123",
        farm_id="farm_456",
        intent="irrigation",
    )
    
    context = UserContext(
        user_id="farmer_123",
        farm_id="farm_456",
        farm_location={"lat": 28.6139, "lon": 77.2090},  # Delhi
        farm_size_hectares=2.5,
        crop_type="wheat",
        growth_stage="flowering",
        soil_type="loamy",
    )
    
    # Analyze
    result = await agent.analyze(query, context)
    
    print(f"Agent: {result.agent_name}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Data Sources: {', '.join(result.data_sources)}")
    print(f"\nRecommendation:\n{result.recommendation}")
    print(f"\nReasoning:")
    for step in result.reasoning_chain:
        print(f"  - {step}")
    
    return result


async def example_market_prices():
    """
    Example: Fetch market prices using government API.
    """
    print("\n=== Market Prices Example ===\n")
    
    # Initialize clients
    clients = await initialize_external_clients()
    government_client = clients["government"]
    
    # Fetch market prices
    prices = await government_client.get_market_prices(
        commodity="wheat",
        state="Punjab",
        district="Ludhiana"
    )
    
    print(f"Commodity: {prices['commodity']}")
    print(f"Source: {prices['source']}")
    
    if prices.get('prices'):
        print("\nMarket Prices:")
        for price_data in prices['prices'][:3]:  # Show first 3
            print(f"  Market: {price_data.get('market')}")
            print(f"  Modal Price: ₹{price_data.get('modal_price')}/quintal")
            print(f"  Date: {price_data.get('date')}")
            print()
    else:
        print(f"Message: {prices.get('message')}")
    
    return prices


async def example_satellite_data():
    """
    Example: Fetch NDVI data using satellite API.
    """
    print("\n=== Satellite Data Example ===\n")
    
    # Initialize clients
    clients = await initialize_external_clients()
    satellite_client = clients["satellite"]
    
    # Fetch NDVI data
    ndvi_data = await satellite_client.get_ndvi_data(
        lat=28.6139,
        lon=77.2090,
        buffer_meters=500
    )
    
    print(f"Location: {ndvi_data['location']}")
    print(f"NDVI Value: {ndvi_data.get('ndvi_value', 'N/A')}")
    print(f"Category: {ndvi_data.get('ndvi_category', 'N/A')}")
    print(f"Date: {ndvi_data.get('date')}")
    print(f"Source: {ndvi_data['source']}")
    
    if ndvi_data.get('message'):
        print(f"Message: {ndvi_data['message']}")
    
    return ndvi_data


async def example_multi_source_analysis():
    """
    Example: Combine data from multiple external sources.
    """
    print("\n=== Multi-Source Analysis Example ===\n")
    
    # Initialize all clients
    clients = await initialize_external_clients()
    
    location = {"lat": 28.6139, "lon": 77.2090}
    
    # Fetch data from multiple sources concurrently
    weather_task = clients["weather"].get_current_weather(
        lat=location["lat"],
        lon=location["lon"]
    )
    
    ndvi_task = clients["satellite"].get_ndvi_data(
        lat=location["lat"],
        lon=location["lon"]
    )
    
    prices_task = clients["government"].get_market_prices(
        commodity="wheat"
    )
    
    # Wait for all tasks
    weather, ndvi, prices = await asyncio.gather(
        weather_task,
        ndvi_task,
        prices_task,
        return_exceptions=True
    )
    
    # Combine insights
    print("Combined Farm Intelligence:")
    print(f"\n1. Weather Conditions:")
    if isinstance(weather, dict):
        print(f"   Temperature: {weather.get('temperature_celsius')}°C")
        print(f"   Humidity: {weather.get('humidity_percent')}%")
        print(f"   Rainfall: {weather.get('rainfall_mm')}mm")
    
    print(f"\n2. Vegetation Health:")
    if isinstance(ndvi, dict):
        print(f"   NDVI: {ndvi.get('ndvi_value', 'N/A')}")
        print(f"   Status: {ndvi.get('ndvi_category', 'N/A')}")
    
    print(f"\n3. Market Conditions:")
    if isinstance(prices, dict) and prices.get('prices'):
        avg_price = sum(p.get('modal_price', 0) for p in prices['prices']) / len(prices['prices'])
        print(f"   Average Price: ₹{avg_price:.2f}/quintal")
    
    return {
        "weather": weather if isinstance(weather, dict) else None,
        "ndvi": ndvi if isinstance(ndvi, dict) else None,
        "prices": prices if isinstance(prices, dict) else None,
    }


async def example_error_handling():
    """
    Example: Demonstrate error handling and fallbacks.
    """
    print("\n=== Error Handling Example ===\n")
    
    # Initialize client without API keys (will use fallbacks)
    weather_client = WeatherAPIClient(
        openweather_api_key=None,  # No API key
        imd_api_key=None,
    )
    
    # Try to fetch weather (will fallback)
    weather = await weather_client.get_current_weather(
        lat=28.6139,
        lon=77.2090,
        use_fallback=True
    )
    
    print(f"Weather Source: {weather['source']}")
    print(f"Temperature: {weather['temperature_celsius']}°C")
    print(f"Message: {weather.get('message', 'Data retrieved successfully')}")
    
    if weather['source'] == 'fallback':
        print("\n✓ Fallback mechanism working correctly")
        print("  System gracefully degraded when API unavailable")
    
    return weather


async def main():
    """Run all examples."""
    print("FastMCP External API Integration Examples")
    print("=" * 50)
    
    try:
        # Run examples
        await example_weather_analysis()
        await example_market_prices()
        await example_satellite_data()
        await example_multi_source_analysis()
        await example_error_handling()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
