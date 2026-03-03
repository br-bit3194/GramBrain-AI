"""Government API client using FastMCP tools."""

import os
from typing import Dict, Any, Optional, List
from datetime import datetime
import httpx

from .base import FastMCPTool, ToolError


class GovernmentAPIClient:
    """Client for government APIs (Agmarknet) using FastMCP tools."""
    
    def __init__(
        self,
        agmarknet_api_key: Optional[str] = None,
    ):
        """
        Initialize government API client.
        
        Args:
            agmarknet_api_key: Agmarknet API key (from env if not provided)
        """
        self.agmarknet_api_key = agmarknet_api_key or os.getenv('AGMARKNET_API_KEY')
        
        # Initialize tools
        self.market_prices_tool = MarketPricesTool(api_key=self.agmarknet_api_key)
        self.soil_health_tool = SoilHealthTool(api_key=self.agmarknet_api_key)
    
    async def get_market_prices(
        self,
        commodity: str,
        state: Optional[str] = None,
        district: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get market prices for commodity.
        
        Args:
            commodity: Commodity name (e.g., "wheat", "rice")
            state: State name (optional)
            district: District name (optional)
            
        Returns:
            Market price data
        """
        result = await self.market_prices_tool.execute(
            commodity=commodity,
            state=state,
            district=district
        )
        
        if result.success:
            return result.data
        
        return self._get_fallback_prices(commodity)
    
    async def get_soil_health_data(
        self,
        state: str,
        district: str,
        village: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get soil health card data.
        
        Args:
            state: State name
            district: District name
            village: Village name (optional)
            
        Returns:
            Soil health data
        """
        result = await self.soil_health_tool.execute(
            state=state,
            district=district,
            village=village
        )
        
        if result.success:
            return result.data
        
        return self._get_fallback_soil_health(state, district)
    
    def _get_fallback_prices(self, commodity: str) -> Dict[str, Any]:
        """Get fallback price data when API fails."""
        return {
            "commodity": commodity,
            "prices": [],
            "source": "fallback",
            "message": "Market price data unavailable",
        }
    
    def _get_fallback_soil_health(self, state: str, district: str) -> Dict[str, Any]:
        """Get fallback soil health data when API fails."""
        return {
            "state": state,
            "district": district,
            "soil_type": "unknown",
            "ph": 7.0,
            "organic_carbon": "medium",
            "nitrogen": "medium",
            "phosphorus": "medium",
            "potassium": "medium",
            "source": "fallback",
            "message": "Soil health data unavailable",
        }
    
    def get_tools(self) -> List[FastMCPTool]:
        """Get all government API tools for registration."""
        return [
            self.market_prices_tool,
            self.soil_health_tool,
        ]


class MarketPricesTool(FastMCPTool):
    """Tool for fetching market prices from Agmarknet."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize market prices tool."""
        super().__init__(
            tool_name="get_market_prices",
            max_retries=3,
            rate_limit_rps=5.0,
        )
        self.api_key = api_key
        # Agmarknet API endpoint
        self.base_url = "https://api.data.gov.in/resource"
    
    async def _execute(
        self,
        commodity: str,
        state: Optional[str] = None,
        district: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Fetch market prices.
        
        Args:
            commodity: Commodity name
            state: State name
            district: District name
            
        Returns:
            Market price data
            
        Raises:
            ToolError: If API call fails
        """
        if not self.api_key:
            raise ToolError("Agmarknet API key not configured")
        
        # Build query parameters
        params = {
            "api-key": self.api_key,
            "format": "json",
            "filters[commodity]": commodity,
        }
        
        if state:
            params["filters[state]"] = state
        if district:
            params["filters[district]"] = district
        
        # Note: Using data.gov.in API structure
        # Actual endpoint ID would need to be configured
        resource_id = "9ef84268-d588-465a-a308-a864a43d0070"  # Example resource ID
        url = f"{self.base_url}/{resource_id}"
        
        async with httpx.AsyncClient(timeout=15.0) as client:
            try:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                # Transform to standard format
                records = data.get("records", [])
                prices = []
                
                for record in records:
                    prices.append({
                        "market": record.get("market"),
                        "state": record.get("state"),
                        "district": record.get("district"),
                        "commodity": record.get("commodity"),
                        "variety": record.get("variety"),
                        "min_price": float(record.get("min_price", 0)),
                        "max_price": float(record.get("max_price", 0)),
                        "modal_price": float(record.get("modal_price", 0)),
                        "date": record.get("arrival_date"),
                    })
                
                return {
                    "commodity": commodity,
                    "prices": prices,
                    "source": "agmarknet",
                    "timestamp": datetime.now().isoformat(),
                }
            
            except httpx.HTTPStatusError as e:
                raise ToolError(f"Agmarknet API error: {e.response.status_code}")
            except httpx.RequestError as e:
                raise ToolError(f"Agmarknet request failed: {str(e)}")
            except (KeyError, ValueError) as e:
                raise ToolError(f"Failed to parse Agmarknet response: {str(e)}")
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema."""
        return {
            "name": "get_market_prices",
            "description": "Get agricultural commodity market prices from Agmarknet",
            "parameters": {
                "type": "object",
                "properties": {
                    "commodity": {
                        "type": "string",
                        "description": "Commodity name (e.g., wheat, rice, cotton)",
                    },
                    "state": {
                        "type": "string",
                        "description": "State name (optional)",
                    },
                    "district": {
                        "type": "string",
                        "description": "District name (optional)",
                    },
                },
                "required": ["commodity"],
            },
        }


class SoilHealthTool(FastMCPTool):
    """Tool for fetching soil health card data."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize soil health tool."""
        super().__init__(
            tool_name="get_soil_health_data",
            max_retries=3,
            rate_limit_rps=5.0,
        )
        self.api_key = api_key
        self.base_url = "https://api.data.gov.in/resource"
    
    async def _execute(
        self,
        state: str,
        district: str,
        village: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Fetch soil health data.
        
        Args:
            state: State name
            district: District name
            village: Village name
            
        Returns:
            Soil health data
            
        Raises:
            ToolError: If API call fails
        """
        if not self.api_key:
            raise ToolError("Soil health API key not configured")
        
        params = {
            "api-key": self.api_key,
            "format": "json",
            "filters[state]": state,
            "filters[district]": district,
        }
        
        if village:
            params["filters[village]"] = village
        
        # Note: Using data.gov.in API structure
        # Actual resource ID for soil health cards would need to be configured
        resource_id = "soil-health-card-resource-id"  # Placeholder
        url = f"{self.base_url}/{resource_id}"
        
        async with httpx.AsyncClient(timeout=15.0) as client:
            try:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                # Transform to standard format
                records = data.get("records", [])
                
                if not records:
                    raise ToolError("No soil health data found for location")
                
                # Use first record (or aggregate if multiple)
                record = records[0]
                
                return {
                    "state": state,
                    "district": district,
                    "village": village,
                    "soil_type": record.get("soil_type"),
                    "ph": float(record.get("ph", 7.0)),
                    "organic_carbon": record.get("organic_carbon"),
                    "nitrogen": record.get("nitrogen"),
                    "phosphorus": record.get("phosphorus"),
                    "potassium": record.get("potassium"),
                    "recommendations": record.get("recommendations", []),
                    "source": "soil_health_card",
                    "timestamp": datetime.now().isoformat(),
                }
            
            except httpx.HTTPStatusError as e:
                raise ToolError(f"Soil health API error: {e.response.status_code}")
            except httpx.RequestError as e:
                raise ToolError(f"Soil health request failed: {str(e)}")
            except (KeyError, ValueError) as e:
                raise ToolError(f"Failed to parse soil health response: {str(e)}")
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema."""
        return {
            "name": "get_soil_health_data",
            "description": "Get soil health card data for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "state": {
                        "type": "string",
                        "description": "State name",
                    },
                    "district": {
                        "type": "string",
                        "description": "District name",
                    },
                    "village": {
                        "type": "string",
                        "description": "Village name (optional)",
                    },
                },
                "required": ["state", "district"],
            },
        }
