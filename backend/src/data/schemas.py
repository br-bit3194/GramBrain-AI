"""Pydantic schemas for data validation.

These schemas provide request/response validation for API endpoints
and ensure data integrity before database operations.
"""

from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum


class UserRole(str, Enum):
    """User role types."""
    FARMER = "farmer"
    VILLAGE_LEADER = "village_leader"
    POLICYMAKER = "policymaker"
    CONSUMER = "consumer"
    AGRONOMIST = "agronomist"


class IrrigationType(str, Enum):
    """Irrigation types."""
    DRIP = "drip"
    FLOOD = "flood"
    SPRINKLER = "sprinkler"
    RAINFED = "rainfed"


class ProductCategory(str, Enum):
    """Product categories."""
    VEGETABLES = "vegetables"
    GRAINS = "grains"
    PULSES = "pulses"
    DAIRY = "dairy"
    HONEY = "honey"
    SPICES = "spices"


class UserSchema(BaseModel):
    """Pydantic schema for User validation."""
    model_config = ConfigDict(use_enum_values=True)
    
    user_id: str = Field(..., min_length=1, max_length=100, description="Unique user identifier")
    phone_number: str = Field(..., min_length=10, max_length=15, description="User phone number")
    name: str = Field(..., min_length=1, max_length=200, description="User full name")
    language_preference: str = Field(default="en", pattern="^[a-z]{2}$", description="ISO 639-1 language code")
    role: UserRole = Field(default=UserRole.FARMER, description="User role")
    created_at: datetime = Field(default_factory=datetime.now, description="Account creation timestamp")
    last_active: datetime = Field(default_factory=datetime.now, description="Last activity timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @field_validator('phone_number')
    @classmethod
    def validate_phone_number(cls, v: str) -> str:
        """Validate phone number format."""
        if not v.isdigit():
            raise ValueError('Phone number must contain only digits')
        if len(v) < 10:
            raise ValueError('Phone number must be at least 10 digits')
        return v
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate name is not empty or whitespace."""
        if not v.strip():
            raise ValueError('Name cannot be empty or whitespace')
        return v.strip()


class UserCreateSchema(BaseModel):
    """Schema for creating a new user."""
    model_config = ConfigDict(use_enum_values=True)
    
    phone_number: str = Field(..., min_length=10, max_length=15)
    name: str = Field(..., min_length=1, max_length=200)
    language_preference: str = Field(default="en", pattern="^[a-z]{2}$")
    role: UserRole = Field(default=UserRole.FARMER)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('phone_number')
    @classmethod
    def validate_phone_number(cls, v: str) -> str:
        """Validate phone number format."""
        if not v.isdigit():
            raise ValueError('Phone number must contain only digits')
        return v


class UserUpdateSchema(BaseModel):
    """Schema for updating user attributes."""
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    language_preference: Optional[str] = Field(None, pattern="^[a-z]{2}$")
    role: Optional[UserRole] = None
    metadata: Optional[Dict[str, Any]] = None


class LocationSchema(BaseModel):
    """Schema for geographic location."""
    lat: float = Field(..., ge=-90, le=90, description="Latitude")
    lon: float = Field(..., ge=-180, le=180, description="Longitude")


class FarmSchema(BaseModel):
    """Pydantic schema for Farm validation."""
    model_config = ConfigDict(use_enum_values=True)
    
    farm_id: str = Field(..., min_length=1, max_length=100, description="Unique farm identifier")
    owner_id: str = Field(..., min_length=1, max_length=100, description="Owner user ID")
    location: LocationSchema = Field(..., description="Farm location coordinates")
    area_hectares: float = Field(..., gt=0, le=100000, description="Farm area in hectares")
    soil_type: str = Field(..., min_length=1, max_length=50, description="Soil type")
    irrigation_type: IrrigationType = Field(..., description="Irrigation method")
    crops: List[str] = Field(default_factory=list, max_length=50, description="List of crops")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('crops')
    @classmethod
    def validate_crops(cls, v: List[str]) -> List[str]:
        """Validate crop names."""
        return [crop.strip() for crop in v if crop.strip()]


class FarmCreateSchema(BaseModel):
    """Schema for creating a new farm."""
    model_config = ConfigDict(use_enum_values=True)
    
    owner_id: str = Field(..., min_length=1, max_length=100)
    location: LocationSchema
    area_hectares: float = Field(..., gt=0, le=100000)
    soil_type: str = Field(..., min_length=1, max_length=50)
    irrigation_type: IrrigationType
    crops: List[str] = Field(default_factory=list, max_length=50)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FarmUpdateSchema(BaseModel):
    """Schema for updating farm attributes."""
    location: Optional[LocationSchema] = None
    area_hectares: Optional[float] = Field(None, gt=0, le=100000)
    soil_type: Optional[str] = Field(None, min_length=1, max_length=50)
    irrigation_type: Optional[IrrigationType] = None
    crops: Optional[List[str]] = Field(None, max_length=50)
    metadata: Optional[Dict[str, Any]] = None


class RecommendationSchema(BaseModel):
    """Pydantic schema for Recommendation validation."""
    recommendation_id: str = Field(..., min_length=1, max_length=100)
    query_id: str = Field(..., min_length=1, max_length=100)
    user_id: str = Field(..., min_length=1, max_length=100)
    farm_id: Optional[str] = Field(None, max_length=100)
    timestamp: datetime
    recommendation_text: str = Field(..., min_length=1, description="Recommendation content")
    reasoning_chain: List[str] = Field(default_factory=list, description="Chain of reasoning")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence score")
    agent_contributions: List[str] = Field(default_factory=list, description="Contributing agents")
    language: str = Field(default="en", pattern="^[a-z]{2}$")
    voice_audio_url: Optional[str] = Field(None, max_length=500)
    created_at: datetime = Field(default_factory=datetime.now)
    
    @field_validator('recommendation_text')
    @classmethod
    def validate_recommendation_text(cls, v: str) -> str:
        """Validate recommendation text is not empty."""
        if not v.strip():
            raise ValueError('Recommendation text cannot be empty')
        return v.strip()


class RecommendationCreateSchema(BaseModel):
    """Schema for creating a new recommendation."""
    query_id: str = Field(..., min_length=1, max_length=100)
    user_id: str = Field(..., min_length=1, max_length=100)
    farm_id: Optional[str] = Field(None, max_length=100)
    recommendation_text: str = Field(..., min_length=1)
    reasoning_chain: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    agent_contributions: List[str] = Field(default_factory=list)
    language: str = Field(default="en", pattern="^[a-z]{2}$")
    voice_audio_url: Optional[str] = Field(None, max_length=500)


class ProductSchema(BaseModel):
    """Pydantic schema for Product validation."""
    model_config = ConfigDict(use_enum_values=True)
    
    product_id: str = Field(..., min_length=1, max_length=100)
    farmer_id: str = Field(..., min_length=1, max_length=100)
    farm_id: str = Field(..., min_length=1, max_length=100)
    product_type: ProductCategory
    name: str = Field(..., min_length=1, max_length=200)
    quantity_kg: float = Field(..., gt=0, description="Quantity in kilograms")
    price_per_kg: float = Field(..., gt=0, description="Price per kilogram")
    harvest_date: datetime
    images: List[str] = Field(default_factory=list, max_length=20)
    pure_product_score: float = Field(default=0.0, ge=0.0, le=100.0)
    status: str = Field(default="available", pattern="^(available|reserved|sold)$")
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('harvest_date')
    @classmethod
    def validate_harvest_date(cls, v: datetime) -> datetime:
        """Validate harvest date is not in the future."""
        if v > datetime.now():
            raise ValueError('Harvest date cannot be in the future')
        return v


class ProductCreateSchema(BaseModel):
    """Schema for creating a new product."""
    model_config = ConfigDict(use_enum_values=True)
    
    farmer_id: str = Field(..., min_length=1, max_length=100)
    farm_id: str = Field(..., min_length=1, max_length=100)
    product_type: ProductCategory
    name: str = Field(..., min_length=1, max_length=200)
    quantity_kg: float = Field(..., gt=0)
    price_per_kg: float = Field(..., gt=0)
    harvest_date: datetime
    images: List[str] = Field(default_factory=list, max_length=20)
    pure_product_score: float = Field(default=0.0, ge=0.0, le=100.0)
    status: str = Field(default="available", pattern="^(available|reserved|sold)$")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProductUpdateSchema(BaseModel):
    """Schema for updating product attributes."""
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    quantity_kg: Optional[float] = Field(None, gt=0)
    price_per_kg: Optional[float] = Field(None, gt=0)
    images: Optional[List[str]] = Field(None, max_length=20)
    pure_product_score: Optional[float] = Field(None, ge=0.0, le=100.0)
    status: Optional[str] = Field(None, pattern="^(available|reserved|sold)$")
    metadata: Optional[Dict[str, Any]] = None


class KnowledgeChunkSchema(BaseModel):
    """Pydantic schema for Knowledge chunk validation."""
    chunk_id: str = Field(..., min_length=1, max_length=100)
    content: str = Field(..., min_length=1, description="Knowledge content")
    source: str = Field(..., min_length=1, max_length=200)
    topic: str = Field(..., min_length=1, max_length=100)
    crop_type: Optional[str] = Field(None, max_length=50)
    region: Optional[str] = Field(None, max_length=100)
    embedding_indexed: bool = Field(default=False)
    created_at: datetime = Field(default_factory=datetime.now)
    
    @field_validator('content')
    @classmethod
    def validate_content(cls, v: str) -> str:
        """Validate content is not empty."""
        if not v.strip():
            raise ValueError('Content cannot be empty')
        return v.strip()


class KnowledgeChunkCreateSchema(BaseModel):
    """Schema for creating a new knowledge chunk."""
    content: str = Field(..., min_length=1)
    source: str = Field(..., min_length=1, max_length=200)
    topic: str = Field(..., min_length=1, max_length=100)
    crop_type: Optional[str] = Field(None, max_length=50)
    region: Optional[str] = Field(None, max_length=100)


class PaginationParams(BaseModel):
    """Schema for pagination parameters."""
    limit: Optional[int] = Field(None, ge=1, le=1000, description="Maximum items to return")
    exclusive_start_key: Optional[Dict[str, Any]] = Field(None, description="Pagination token")
    
    @field_validator('limit')
    @classmethod
    def validate_limit(cls, v: Optional[int]) -> Optional[int]:
        """Validate limit is within reasonable bounds."""
        if v is not None and v < 1:
            raise ValueError('Limit must be at least 1')
        if v is not None and v > 1000:
            raise ValueError('Limit cannot exceed 1000')
        return v


class PaginatedResponse(BaseModel):
    """Schema for paginated response."""
    items: List[Any] = Field(..., description="List of items")
    last_evaluated_key: Optional[Dict[str, Any]] = Field(None, description="Pagination token for next page")
    count: int = Field(..., ge=0, description="Number of items in this page")
    
    @field_validator('count')
    @classmethod
    def validate_count_matches_items(cls, v: int, info) -> int:
        """Validate count matches items length."""
        items = info.data.get('items', [])
        if v != len(items):
            raise ValueError(f'Count {v} does not match items length {len(items)}')
        return v
