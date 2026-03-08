# app/integration/tools/market_tools.py
"""Market tools using DynamoDB for data storage"""
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime, timedelta

import requests

from ..config.aws_config import aws_config
from ..database.dynamodb_client import dynamodb_client

logger = logging.getLogger(__name__)


def _fetch_mandi_prices(
    commodity: Optional[str] = None,
    state: Optional[str] = None,
    district: Optional[str] = None,
    days: int = 7
) -> Dict[str, Any]:
    """Fallback: Fetch market prices from Data.gov.in mandi API."""
    if not aws_config.mandi_api_key or not aws_config.mandi_api_url:
        return {"records": [], "error": "MANDI API not configured"}

    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        params = {
            "api-key": aws_config.mandi_api_key,
            "format": "json",
            "limit": 100,
            # Data.gov.in uses filter syntax: filters[field]=value
            "filters[arrival_date]": f"{start_date.strftime('%Y-%m-%d')}..{end_date.strftime('%Y-%m-%d')}"
        }

        if commodity:
            params["filters[commodity]"] = commodity
        if state:
            params["filters[state]"] = state
        if district:
            params["filters[district]"] = district

        resp = requests.get(aws_config.mandi_api_url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        # Handle Data.gov.in metadata error responses
        if isinstance(data, dict):
            status = str(data.get("status", "")).lower()
            if status in ("error", "fail"):
                msg = data.get("message") or data.get("error") or "Unknown error"
                return {"records": [], "error": f"API status={status}: {msg}"}

        records = data.get("records") or data.get("data") or data.get("result") or []
        if not isinstance(records, list):
            return {"records": [], "error": "Unexpected API response format"}

        results: List[Dict[str, Any]] = []
        for r in records:
            try:
                modal_price = float(r.get("modal_price") or r.get("modal") or 0)
            except (TypeError, ValueError):
                modal_price = 0

            trend_raw = str(r.get("trend") or "").strip().lower()
            trend_value = 0
            if "up" in trend_raw or "increase" in trend_raw or "high" in trend_raw:
                trend_value = 1
            elif "down" in trend_raw or "decrease" in trend_raw or "low" in trend_raw:
                trend_value = -1

            results.append({
                "market": r.get("market") or r.get("market_name"),
                "district": r.get("district") or r.get("district_name"),
                "state": r.get("state") or r.get("state_name"),
                "commodity": r.get("commodity") or r.get("commodity_name") or r.get("crop") or "",
                "variety": r.get("variety") or "",
                "grade": r.get("grade") or "",
                "modal_price": modal_price,
                "min_price": r.get("min_price") or r.get("min") or 0,
                "max_price": r.get("max_price") or r.get("max") or 0,
                "arrival_date": r.get("arrival_date") or r.get("date") or "",
                "trend": trend_raw or "stable",
                "price_change": trend_value,
            })

        return {"records": results, "error": None}
    except Exception as e:
        logger.error(f"Error fetching mandi API prices: {e}")
        return {"records": [], "error": str(e)}


async def get_market_summary(
    days_back: int = 1,
    force_refresh: bool = False
) -> Dict[str, Any]:
    """Get a market summary using cached DynamoDB summary or live mandi API."""
    # Try cached summary in DynamoDB first
    if not force_refresh:
        cached = dynamodb_client.get_market_summary()
        if cached:
            # Reject stale/invalid cache if it contains no usable commodity info
            summary = cached.get("summary") if isinstance(cached, dict) else None
            valid_cache = bool(summary and isinstance(summary.get("unique_commodities"), int) and summary.get("unique_commodities", 0) > 0)
            if valid_cache:
                return {
                    "status": "success",
                    "source": "dynamodb",
                    "summary": cached
                }
            # otherwise fall through to refresh

    # Fetch live data from mandi API (limited to recent records)
    mandi_result = _fetch_mandi_prices(commodity=None, days=days_back)
    records = mandi_result.get("records", [])
    error = mandi_result.get("error")

    if not records:
        msg = "No market records fetched from mandi API"
        if error:
            msg += f" (error: {error})"
        return {
            "status": "no_data",
            "message": msg
        }

    # Build summary from records
    commodity_stats: Dict[str, Dict[str, Any]] = {}
    market_counts: Dict[str, int] = {}
    state_counts: Dict[str, int] = {}

    for r in records:
        commodity = (r.get("commodity") or "").strip()
        market = (r.get("market") or "").strip()
        state = (r.get("state") or "").strip()
        price = float(r.get("modal_price") or 0)

        if commodity:
            stats = commodity_stats.setdefault(commodity, {
                "total_price": 0.0,
                "count": 0,
                "markets": {},
                "trend": r.get("trend", "stable"),
                "latest_arrival": r.get("arrival_date", "")
            })

            stats["total_price"] += price
            stats["count"] += 1
            if market:
                stats["markets"][market] = stats["markets"].get(market, 0) + 1
            # track latest record for the commodity
            if r.get("arrival_date") and r.get("arrival_date") >= stats.get("latest_arrival", ""):
                stats["latest_arrival"] = r.get("arrival_date")
                stats["trend"] = r.get("trend", stats.get("trend"))

        if market:
            market_counts[market] = market_counts.get(market, 0) + 1

        if state:
            state_counts[state] = state_counts.get(state, 0) + 1

    # Generate sorted commodity list (by average price descending)
    top_commodities = []
    for commodity, stats in commodity_stats.items():
        avg_price = stats["total_price"] / stats["count"] if stats["count"] else 0
        top_market = ""
        if stats["markets"]:
            top_market = max(stats["markets"].items(), key=lambda kv: kv[1])[0]

        trend_val = 0
        trend_raw = (stats.get("trend") or "").lower()
        if "up" in trend_raw or "increase" in trend_raw or "high" in trend_raw:
            trend_val = 1
        elif "down" in trend_raw or "decrease" in trend_raw or "low" in trend_raw:
            trend_val = -1

        top_commodities.append({
            "commodity": commodity,
            "avg_price": round(avg_price, 2),
            "price_change": trend_val,
            "top_market": top_market,
            "trend": stats.get("trend", "stable")
        })

    top_commodities = sorted(top_commodities, key=lambda x: x["avg_price"], reverse=True)[:10]
    top_markets = [m for m, _ in sorted(market_counts.items(), key=lambda kv: kv[1], reverse=True)][:10]
    top_states = [s for s, _ in sorted(state_counts.items(), key=lambda kv: kv[1], reverse=True)][:10]

    summary = {
        "total_records": len(records),
        "unique_commodities": len(commodity_stats),
        "unique_markets": len(market_counts),
        "unique_states": len(state_counts),
        "top_commodities": top_commodities,
        "top_markets": top_markets,
        "top_states": top_states,
        "data_source": "mandi_api",
        "fetched_at": datetime.now().isoformat()
    }

    # Cache the summary in DynamoDB for later
    dynamodb_client.put_market_summary(summary)

    return {
        "status": "success",
        "source": "mandi_api",
        "summary": summary
    }


async def get_market_prices(
    commodity: str,
    state: Optional[str] = None,
    district: Optional[str] = None,
    days: int = 7
) -> Dict[str, Any]:
    """
    Get current market prices from DynamoDB
    
    Args:
        commodity: Commodity name (e.g., "Onion", "Tomato")
        state: State filter
        district: District filter
        days: Number of days to fetch
    
    Returns:
        Dict with current prices and trends
    """
    try:
        # Try DynamoDB first (best performance when available)
        prices = []
        source = "dynamodb"
        fallback_error = None
        try:
            # Avoid passing invalid location objects to DynamoDB queries
            state_str = state if isinstance(state, str) else None
            district_str = district if isinstance(district, str) else None

            prices = dynamodb_client.get_market_prices_by_commodity(
                commodity,
                days_back=days,
                state=state_str,
                district=district_str
            )
        except Exception as e:
            logger.warning(f"Error querying DynamoDB market prices, falling back to mandi API: {e}")
            prices = []

        if not prices:
            # Fallback to live mandi API if configured
            # Ensure state/district are strings before passing them to the API
            state_str = state if isinstance(state, str) else None
            district_str = district if isinstance(district, str) else None

            mandi_result = _fetch_mandi_prices(commodity, state=state_str, district=district_str, days=days)
            prices = mandi_result.get("records", [])
            fallback_error = mandi_result.get("error")

            if prices:
                source = "mandi_api"

        if not prices:
            msg = f"{commodity} के लिए हाल की कीमत डेटा उपलब्ध नहीं है"
            if fallback_error:
                msg += f" (fallback error: {fallback_error})"
            return {
                "status": "no_data",
                "message": msg,
                "commodity": commodity
            }
        
        # Process and organize results
        market_data = []
        total_price = 0
        
        for price in prices:
            market_data.append({
                "market": price.get('market'),
                "district": price.get('district'),
                "state": price.get('state'),
                "modal_price": price.get('modal_price'),
                "arrival_date": price.get('arrival_date'),
                "trend": price.get('trend', 'stable')
            })
            total_price += price.get('modal_price', 0)
        
        avg_price = total_price / len(prices) if prices else 0
        
        return {
            "status": "success",
            "commodity": commodity,
            "source": source,
            "price_summary": {
                "average_price": round(avg_price, 2),
                "total_markets": len(prices)
            },
            "best_markets": market_data[:5],
            "data_date": datetime.now().strftime("%d-%m-%Y")
        }
        
    except Exception as e:
        logger.error(f"Error fetching market prices: {e}")
        return {
            "status": "error",
            "message": str(e),
            "commodity": commodity
        }


async def get_price_analysis(
    commodity: str,
    analysis_days: int = 30
) -> Dict[str, Any]:
    """
    Get detailed price analysis from DynamoDB analytics
    
    Args:
        commodity: Commodity name
        analysis_days: Days for analysis
    
    Returns:
        Dict with comprehensive analysis
    """
    try:
        analytics = dynamodb_client.get_analytics_by_commodity(commodity, analysis_days)
        
        if not analytics:
            return {
                "status": "insufficient_data",
                "message": f"{commodity} के लिए विश्लेषण डेटा उपलब्ध नहीं है"
            }
        
        latest = analytics[0] if analytics else {}
        
        return {
            "status": "success",
            "commodity": commodity,
            "price_statistics": {
                "average_price": latest.get('avg_price'),
                "highest_price": latest.get('highest_price'),
                "lowest_price": latest.get('lowest_price')
            },
            "trend_analysis": {
                "weekly_trend": latest.get('weekly_trend'),
                "monthly_trend": latest.get('monthly_trend')
            },
            "predictions": {
                "7_day_price": latest.get('predicted_price_7d'),
                "14_day_price": latest.get('predicted_price_14d')
            }
        }
        
    except Exception as e:
        logger.error(f"Error in price analysis: {e}")
        return {
            "status": "error",
            "message": str(e)
        }


async def get_selling_advice(
    commodity: str,
    quantity: Optional[float] = None,
    quality_grade: str = "medium",
    urgency: str = "normal",
    state: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get selling advice based on market conditions
    
    Args:
        commodity: Commodity name
        quantity: Quantity in quintals
        quality_grade: Quality (high/medium/low)
        urgency: Urgency level (urgent/normal/flexible)
        state: Optional state filter for more relevant prices
    
    Returns:
        Dict with selling strategy
    """
    try:
        # Get recent prices (with fallback to mandi API)
        market_response = await get_market_prices(
            commodity,
            state=state,
            days=5
        )

        if market_response.get("status") != "success":
            return market_response

        price_summary = market_response.get("price_summary", {})
        best_markets = market_response.get("best_markets", [])
        avg_price = price_summary.get("average_price") or 0
        best_price = 0
        if best_markets:
            best_price = max(m.get("modal_price", 0) for m in best_markets)

        # Quality adjustment
        quality_multiplier = {"high": 1.15, "medium": 1.0, "low": 0.85}.get(quality_grade, 1.0)
        adjusted_price = avg_price * quality_multiplier

        return {
            "status": "success",
            "commodity": commodity,
            "financial_summary": {
                "average_expected_price": round(adjusted_price, 2),
                "best_possible_price": round(best_price * quality_multiplier, 2),
                "estimated_revenue": round(adjusted_price * quantity, 2) if quantity else None
            },
            "timing_advice": {
                "action": "sell_soon" if urgency == "urgent" else "flexible_timing",
                "reason": "Based on current market conditions"
            },
            "source": market_response.get("source")
        }
        
    except Exception as e:
        logger.error(f"Error generating selling advice: {e}")
        return {
            "status": "error",
            "message": str(e)
        }
