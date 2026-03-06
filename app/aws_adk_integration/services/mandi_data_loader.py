import requests
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any
import os
from sqlalchemy.orm import Session
import json

logger = logging.getLogger(__name__)


class MandiDataLoader:
    """Simplified Mandi data loader with caching and fallback"""

    # Cache for market data
    _cache = {
        'data': None,
        'timestamp': None,
        'cache_duration': 3600  # 1 hour in seconds
    }

    def __init__(self):
        self.api_key = os.getenv("MANDI_API_KEY")
        self.resource_id = '9ef84268-d588-465a-a308-a864a43d0070'
        self.base_url = f'https://api.data.gov.in/resource/{self.resource_id}'
        
        if not self.api_key:
            logger.warning("❌ MANDI_API_KEY not configured")
        else:
            logger.info("✅ Mandi Data Loader initialized")

    @classmethod
    def _is_cache_valid(cls) -> bool:
        """Check if cached data is still valid"""
        if cls._cache['data'] is None or cls._cache['timestamp'] is None:
            return False
        
        elapsed = (datetime.now() - cls._cache['timestamp']).total_seconds()
        return elapsed < cls._cache['cache_duration']

    @classmethod
    def _get_cached_data(cls) -> Dict[str, Any]:
        """Get cached market data"""
        if cls._is_cache_valid():
            logger.info("📦 Using cached market data")
            return cls._cache['data']
        return None

    @classmethod
    def _set_cache(cls, data: Dict[str, Any]):
        """Set cache with new data"""
        cls._cache['data'] = data
        cls._cache['timestamp'] = datetime.now()
        logger.info("💾 Market data cached")

    @staticmethod
    def _get_fallback_summary() -> Dict[str, Any]:
        """Return fallback market summary when API is unavailable"""
        return {
            "status": "success",
            "summary": {
                "total_records": 8611,
                "unique_commodities": 149,
                "unique_markets": 386,
                "unique_states": 19,
                "top_commodities": [
                    "Bitter gourd", "Mushrooms", "Season Leaves", "Leafy Vegetable",
                    "Lemon", "Paddy(Basmati)", "Custard Apple(Sharifa)", "Strawberry",
                    "Potato", "Gur(Jaggery)"
                ],
                "top_markets": [
                    "Thirukalukundram(Uzhavar Sandhai) APMC",
                    "Mohanur(Uzhavar Sandhai) APMC",
                    "Rajkot(Veg.Sub Yard) APMC",
                    "Kahithapattarai(Uzhavar Sandhai) APMC",
                    "Samalkha APMC",
                    "Babrala APMC",
                    "Mannargudi I(Uzhavar Sandhai) APMC",
                    "Ambasamudram(Uzhavar Sandhai) APMC",
                    "Palladam(Uzhavar Sandhai) APMC",
                    "Kattappana APMC"
                ],
                "top_states": [
                    "Uttarakhand", "Rajasthan", "Tamil Nadu", "Uttar Pradesh",
                    "Madhya Pradesh", "Nagaland", "Assam", "Andhra Pradesh",
                    "Himachal Pradesh", "Chandigarh"
                ],
                "data_source": "fallback"
            }
        }

    async def load_market_data(self, days_back: int = 7) -> Dict[str, Any]:
        """Load market data from government API with timeout handling"""
        try:
            if not self.api_key:
                logger.warning("⚠️ MANDI_API_KEY not configured, using fallback")
                return {
                    "status": "success",
                    "records_count": 0,
                    "data": [],
                    "message": "API key not configured, using fallback"
                }

            all_records = []
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)

            current_date = start_date
            while current_date <= end_date:
                date_str = current_date.strftime("%d-%m-%Y")
                
                try:
                    records = await self._fetch_date_data(date_str)
                    all_records.extend(records)
                    if records:
                        logger.info(f"✅ Fetched {len(records)} records for {date_str}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to fetch data for {date_str}: {str(e)[:100]}")
                
                current_date += timedelta(days=1)

            if all_records:
                logger.info(f"✅ Total records fetched: {len(all_records)}")
            else:
                logger.warning("⚠️ No records fetched from API")
            
            return {
                "status": "success",
                "records_count": len(all_records),
                "data": all_records,
                "date_range": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                }
            }

        except Exception as e:
            logger.error(f"❌ Error loading market data: {e}")
            return {
                "status": "error",
                "message": str(e),
                "data": []
            }

    async def _fetch_date_data(self, date_str: str) -> List[Dict]:
        """Fetch data for a specific date with shorter timeout"""
        records = []
        offset = 0
        limit = 1000

        while True:
            params = {
                'api-key': self.api_key,
                'format': 'json',
                'limit': limit,
                'offset': offset,
                'filters[arrival_date]': date_str
            }

            try:
                # Use shorter timeout (10 seconds instead of 30)
                response = requests.get(self.base_url, params=params, timeout=10)
                response.raise_for_status()

                data = response.json()
                batch_records = data.get('records', [])

                if not batch_records:
                    break

                records.extend(batch_records)
                offset += limit

                # Check if we got fewer records than limit (last page)
                if len(batch_records) < limit:
                    break

            except requests.exceptions.Timeout:
                logger.warning(f"⏱️ API timeout for {date_str}, returning partial data")
                break
            except requests.exceptions.RequestException as e:
                logger.warning(f"⚠️ API request failed for {date_str}: {str(e)[:100]}")
                break
            except Exception as e:
                logger.warning(f"⚠️ Error parsing response for {date_str}: {str(e)[:100]}")
                break

        return records

    async def get_commodity_prices(self, commodity: str, days_back: int = 7) -> Dict[str, Any]:
        """Get prices for a specific commodity"""
        try:
            data = await self.load_market_data(days_back)
            
            if data["status"] != "success":
                return data

            # Filter for commodity
            commodity_data = [
                record for record in data["data"]
                if record.get("commodity", "").lower() == commodity.lower()
            ]

            if not commodity_data:
                return {
                    "status": "no_data",
                    "message": f"No data found for commodity: {commodity}",
                    "commodity": commodity
                }

            # Calculate statistics
            prices = [float(r.get("modal_price", 0)) for r in commodity_data if r.get("modal_price")]
            
            if prices:
                avg_price = sum(prices) / len(prices)
                min_price = min(prices)
                max_price = max(prices)
            else:
                avg_price = min_price = max_price = 0

            return {
                "status": "success",
                "commodity": commodity,
                "records_count": len(commodity_data),
                "statistics": {
                    "average_price": round(avg_price, 2),
                    "min_price": round(min_price, 2),
                    "max_price": round(max_price, 2),
                    "price_range": round(max_price - min_price, 2)
                },
                "data": commodity_data[:10]  # Return top 10 records
            }

        except Exception as e:
            logger.error(f"❌ Error getting commodity prices: {e}")
            return {
                "status": "error",
                "message": str(e)
            }

    async def get_market_summary(self) -> Dict[str, Any]:
        """Get summary of market data with caching and fallback"""
        try:
            # Check cache first
            cached_data = self._get_cached_data()
            if cached_data:
                return cached_data

            # Try to fetch fresh data
            data = await self.load_market_data(days_back=1)
            
            if data["status"] != "success":
                logger.warning("⚠️ API failed, using fallback data")
                return self._get_fallback_summary()

            records = data["data"]
            
            if not records:
                logger.warning("⚠️ No records from API, using fallback data")
                return self._get_fallback_summary()

            # Get unique commodities
            commodities = set(r.get("commodity", "") for r in records)
            markets = set(r.get("market", "") for r in records)
            states = set(r.get("state", "") for r in records)

            result = {
                "status": "success",
                "summary": {
                    "total_records": len(records),
                    "unique_commodities": len(commodities),
                    "unique_markets": len(markets),
                    "unique_states": len(states),
                    "top_commodities": list(commodities)[:10],
                    "top_markets": list(markets)[:10],
                    "top_states": list(states)[:10],
                    "data_source": "live"
                }
            }
            
            # Cache the result
            self._set_cache(result)
            return result

        except Exception as e:
            logger.error(f"❌ Error getting market summary: {e}")
            logger.info("📦 Falling back to cached or default data")
            
            # Try to return cached data if available
            cached_data = self._get_cached_data()
            if cached_data:
                return cached_data
            
            # Return fallback data
            return self._get_fallback_summary()

    def get_service_status(self) -> Dict[str, Any]:
        """Get service status"""
        return {
            "service_available": self.api_key is not None,
            "api_configured": self.api_key is not None,
            "api_endpoint": self.base_url,
            "resource_id": self.resource_id,
            "capabilities": [
                "Load market data by date range",
                "Get commodity prices",
                "Get market summary",
                "Filter by commodity, market, state"
            ] if self.api_key else []
        }
