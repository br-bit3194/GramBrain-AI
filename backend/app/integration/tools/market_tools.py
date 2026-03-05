# app/aws_integration/tools/market_tools.py
"""Market tools using DynamoDB for data storage"""
from typing import Dict, Any, Optional
import logging
from datetime import datetime, timedelta
from ..database.dynamodb_client import dynamodb_client

logger = logging.getLogger(__name__)


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
        # Get prices from DynamoDB
        prices = dynamodb_client.get_market_prices_by_commodity(commodity, days)
        
        if not prices:
            return {
                "status": "no_data",
                "message": f"{commodity} के लिए हाल की कीमत डेटा उपलब्ध नहीं है",
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
    urgency: str = "normal"
) -> Dict[str, Any]:
    """
    Get selling advice based on market conditions
    
    Args:
        commodity: Commodity name
        quantity: Quantity in quintals
        quality_grade: Quality (high/medium/low)
        urgency: Urgency level (urgent/normal/flexible)
    
    Returns:
        Dict with selling strategy
    """
    try:
        # Get recent prices
        prices = dynamodb_client.get_market_prices_by_commodity(commodity, 5)
        
        if not prices:
            return {
                "status": "no_data",
                "message": f"{commodity} के लिए मार्केट जानकारी उपलब्ध नहीं है"
            }
        
        # Calculate average and best price
        avg_price = sum(p.get('modal_price', 0) for p in prices) / len(prices)
        best_price = max(p.get('modal_price', 0) for p in prices)
        
        # Quality adjustment
        quality_multiplier = {"high": 1.15, "medium": 1.0, "low": 0.85}[quality_grade]
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
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating selling advice: {e}")
        return {
            "status": "error",
            "message": str(e)
        }
