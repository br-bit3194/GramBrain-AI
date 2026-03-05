# app/aws_integration/config/models.py
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from datetime import datetime


class ChatResponse(BaseModel):
    """Chat response model"""
    response: str
    session_id: str
    agent_used: Optional[str] = None
    tools_called: Optional[List[str]] = None
    confidence: Optional[float] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class MarketPrice(BaseModel):
    """Market price model for DynamoDB"""
    pk: str  # Partition key: COMMODITY#{commodity}
    sk: str  # Sort key: DATE#{date}#MARKET#{market_id}
    state: str
    district: str
    market: str
    commodity: str
    variety: Optional[str] = None
    grade: Optional[str] = None
    arrival_date: str
    min_price: float
    max_price: float
    modal_price: float
    price_change: Optional[float] = 0
    percentage_change: Optional[float] = 0
    trend: Optional[str] = "stable"
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    is_active: bool = True


class SessionState(BaseModel):
    """Session state model for DynamoDB"""
    pk: str  # Partition key: SESSION#{session_id}
    sk: str  # Sort key: METADATA
    user_id: str
    session_id: str
    state: Dict[str, Any]
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    ttl: Optional[int] = None  # TTL for auto-deletion


class MarketAnalytics(BaseModel):
    """Market analytics model for DynamoDB"""
    pk: str  # Partition key: ANALYTICS#{commodity}
    sk: str  # Sort key: DATE#{date}
    commodity: str
    analysis_date: str
    avg_price: float
    highest_price: float
    lowest_price: float
    price_volatility: float
    total_markets: int
    top_market: str
    top_market_price: float
    weekly_trend: str
    monthly_trend: str
    predicted_price_7d: float
    predicted_price_14d: float
    prediction_confidence: float
    price_history: str  # JSON string
    recommendations: str  # JSON string
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
