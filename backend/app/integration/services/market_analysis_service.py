import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from .mandi_data_loader import MandiDataLoader

logger = logging.getLogger(__name__)


class MarketAnalysisService:
    """Market analysis service using real Mandi data"""

    def __init__(self):
        self.data_loader = MandiDataLoader()
        logger.info("✅ Market Analysis Service initialized")

    async def get_commodity_prices(
            self,
            commodity: str,
            days_back: int = 7
    ) -> Dict[str, Any]:
        """Get current prices and trends for a commodity"""
        try:
            result = await self.data_loader.get_commodity_prices(commodity, days_back)
            
            if result["status"] != "success":
                return result

            # Enhance with analysis
            data = result.get("data", [])
            stats = result.get("statistics", {})

            # Calculate trend
            if len(data) > 1:
                first_price = float(data[0].get("modal_price", 0))
                last_price = float(data[-1].get("modal_price", 0))
                trend = "up" if last_price > first_price else "down" if last_price < first_price else "stable"
                change_percent = ((last_price - first_price) / first_price * 100) if first_price > 0 else 0
            else:
                trend = "stable"
                change_percent = 0

            # Get top markets
            markets = {}
            for record in data:
                market = record.get("market", "Unknown")
                price = float(record.get("modal_price", 0))
                if market not in markets:
                    markets[market] = []
                markets[market].append(price)

            top_markets = sorted(
                [(m, sum(p) / len(p)) for m, p in markets.items()],
                key=lambda x: x[1],
                reverse=True
            )[:5]

            return {
                "status": "success",
                "commodity": commodity,
                "current_price": stats.get("average_price", 0),
                "price_range": {
                    "min": stats.get("min_price", 0),
                    "max": stats.get("max_price", 0),
                    "range": stats.get("price_range", 0)
                },
                "trend": {
                    "direction": trend,
                    "change_percent": round(change_percent, 2)
                },
                "top_markets": [
                    {
                        "market": m,
                        "average_price": round(p, 2)
                    }
                    for m, p in top_markets
                ],
                "records_count": result.get("records_count", 0),
                "recommendation": self._get_selling_recommendation(trend, change_percent)
            }

        except Exception as e:
            logger.error(f"❌ Error analyzing commodity prices: {e}")
            return {
                "status": "error",
                "message": str(e)
            }

    async def get_market_prices(
            self,
            market: str,
            days_back: int = 7
    ) -> Dict[str, Any]:
        """Get all commodity prices in a specific market"""
        try:
            data = await self.data_loader.load_market_data(days_back)
            
            if data["status"] != "success":
                return data

            # Filter for market
            market_data = [
                record for record in data["data"]
                if record.get("market", "").lower() == market.lower()
            ]

            if not market_data:
                return {
                    "status": "no_data",
                    "message": f"No data found for market: {market}",
                    "market": market
                }

            # Group by commodity
            commodities = {}
            for record in market_data:
                commodity = record.get("commodity", "Unknown")
                price = float(record.get("modal_price", 0))
                
                if commodity not in commodities:
                    commodities[commodity] = []
                commodities[commodity].append(price)

            # Calculate averages
            commodity_prices = [
                {
                    "commodity": c,
                    "average_price": round(sum(p) / len(p), 2),
                    "records": len(p)
                }
                for c, p in commodities.items()
            ]

            # Sort by price
            commodity_prices.sort(key=lambda x: x["average_price"], reverse=True)

            return {
                "status": "success",
                "market": market,
                "total_commodities": len(commodities),
                "total_records": len(market_data),
                "commodities": commodity_prices[:20]  # Top 20
            }

        except Exception as e:
            logger.error(f"❌ Error getting market prices: {e}")
            return {
                "status": "error",
                "message": str(e)
            }

    async def compare_prices(
            self,
            commodity: str,
            markets: List[str],
            days_back: int = 7
    ) -> Dict[str, Any]:
        """Compare prices of a commodity across multiple markets"""
        try:
            data = await self.data_loader.load_market_data(days_back)
            
            if data["status"] != "success":
                return data

            # Filter for commodity and markets
            filtered_data = [
                record for record in data["data"]
                if record.get("commodity", "").lower() == commodity.lower()
                and record.get("market", "").lower() in [m.lower() for m in markets]
            ]

            if not filtered_data:
                return {
                    "status": "no_data",
                    "message": f"No data found for {commodity} in specified markets"
                }

            # Group by market
            market_prices = {}
            for record in filtered_data:
                market = record.get("market", "Unknown")
                price = float(record.get("modal_price", 0))
                
                if market not in market_prices:
                    market_prices[market] = []
                market_prices[market].append(price)

            # Calculate statistics
            comparison = []
            for market, prices in market_prices.items():
                avg_price = sum(prices) / len(prices)
                comparison.append({
                    "market": market,
                    "average_price": round(avg_price, 2),
                    "min_price": round(min(prices), 2),
                    "max_price": round(max(prices), 2),
                    "records": len(prices)
                })

            # Sort by price
            comparison.sort(key=lambda x: x["average_price"])

            best_market = comparison[0] if comparison else None
            worst_market = comparison[-1] if comparison else None

            return {
                "status": "success",
                "commodity": commodity,
                "comparison": comparison,
                "best_market": best_market,
                "worst_market": worst_market,
                "price_difference": round(worst_market["average_price"] - best_market["average_price"], 2) if best_market and worst_market else 0
            }

        except Exception as e:
            logger.error(f"❌ Error comparing prices: {e}")
            return {
                "status": "error",
                "message": str(e)
            }

    async def get_selling_advice(
            self,
            commodity: str,
            quantity: float,
            location: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get selling advice for a commodity"""
        try:
            # Get commodity prices
            prices_result = await self.get_commodity_prices(commodity, days_back=7)
            
            if prices_result["status"] != "success":
                return prices_result

            current_price = prices_result.get("current_price", 0)
            trend = prices_result.get("trend", {})
            top_markets = prices_result.get("top_markets", [])

            # Calculate potential revenue
            revenue = current_price * quantity

            # Generate advice
            advice = []
            if trend.get("direction") == "up":
                advice.append("✅ Prices are trending UP - Good time to sell!")
            elif trend.get("direction") == "down":
                advice.append("⚠️ Prices are trending DOWN - Consider selling soon before further decline")
            else:
                advice.append("➡️ Prices are STABLE - Reasonable time to sell")

            if top_markets:
                best_market = top_markets[0]
                advice.append(f"💰 Best market: {best_market['market']} (₹{best_market['average_price']}/unit)")

            return {
                "status": "success",
                "commodity": commodity,
                "quantity": quantity,
                "current_price": round(current_price, 2),
                "estimated_revenue": round(revenue, 2),
                "trend": trend,
                "top_markets": top_markets[:3],
                "advice": advice,
                "recommendation": self._get_selling_recommendation(
                    trend.get("direction", "stable"),
                    trend.get("change_percent", 0)
                )
            }

        except Exception as e:
            logger.error(f"❌ Error getting selling advice: {e}")
            return {
                "status": "error",
                "message": str(e)
            }

    def _get_selling_recommendation(self, trend: str, change_percent: float) -> str:
        """Generate selling recommendation based on trend"""
        if trend == "up" and change_percent > 5:
            return "🟢 SELL NOW - Strong upward trend"
        elif trend == "up":
            return "🟢 SELL - Prices improving"
        elif trend == "down" and change_percent < -5:
            return "🔴 SELL URGENTLY - Strong downward trend"
        elif trend == "down":
            return "🟡 SELL SOON - Prices declining"
        else:
            return "🟡 SELL - Prices stable"

    def get_service_status(self) -> Dict[str, Any]:
        """Get service status"""
        loader_status = self.data_loader.get_service_status()
        
        return {
            "service_available": loader_status["api_configured"],
            "data_source": "Government Mandi API",
            "capabilities": [
                "Get commodity prices and trends",
                "Get market prices",
                "Compare prices across markets",
                "Get selling advice",
                "Price trend analysis"
            ],
            "data_loader": loader_status
        }
