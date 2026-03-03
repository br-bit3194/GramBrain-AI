"""Data models for GramBrain system."""

from dataclasses import dataclass, field
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


class GrowthStage(str, Enum):
    """Crop growth stages."""
    GERMINATION = "germination"
    VEGETATIVE = "vegetative"
    TILLERING = "tillering"
    STEM_ELONGATION = "stem_elongation"
    FLOWERING = "flowering"
    GRAIN_FILLING = "grain_filling"
    MATURITY = "maturity"


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


@dataclass
class User:
    """User model."""
    user_id: str
    phone_number: str
    name: str
    password_hash: Optional[str] = None  # Hashed password, not returned in API
    language_preference: str = "en"
    role: UserRole = UserRole.FARMER
    created_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self, include_password: bool = False) -> Dict[str, Any]:
        """Convert to dictionary.
        
        Args:
            include_password: Whether to include password_hash (for internal use only)
        """
        data = {
            "user_id": self.user_id,
            "phone_number": self.phone_number,
            "name": self.name,
            "language_preference": self.language_preference,
            "role": self.role.value,
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat(),
            "metadata": self.metadata,
        }
        if include_password and self.password_hash:
            data["password_hash"] = self.password_hash
        return data


@dataclass
class Farm:
    """Farm model."""
    farm_id: str
    owner_id: str
    location: Dict[str, float]  # {lat, lon}
    area_hectares: float
    soil_type: str
    irrigation_type: IrrigationType
    crops: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "farm_id": self.farm_id,
            "owner_id": self.owner_id,
            "location": self.location,
            "area_hectares": self.area_hectares,
            "soil_type": self.soil_type,
            "irrigation_type": self.irrigation_type.value,
            "crops": self.crops,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class CropCycle:
    """Crop cycle model."""
    cycle_id: str
    farm_id: str
    crop_type: str
    variety: str
    planting_date: datetime
    expected_harvest_date: datetime
    actual_harvest_date: Optional[datetime] = None
    growth_stage: GrowthStage = GrowthStage.GERMINATION
    area_hectares: float = 1.0
    yield_predicted: Optional[float] = None
    yield_actual: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cycle_id": self.cycle_id,
            "farm_id": self.farm_id,
            "crop_type": self.crop_type,
            "variety": self.variety,
            "planting_date": self.planting_date.isoformat(),
            "expected_harvest_date": self.expected_harvest_date.isoformat(),
            "actual_harvest_date": self.actual_harvest_date.isoformat() if self.actual_harvest_date else None,
            "growth_stage": self.growth_stage.value,
            "area_hectares": self.area_hectares,
            "yield_predicted": self.yield_predicted,
            "yield_actual": self.yield_actual,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class SoilHealthData:
    """Soil health data model."""
    farm_id: str
    test_date: datetime
    nitrogen_kg_per_ha: float
    phosphorus_kg_per_ha: float
    potassium_kg_per_ha: float
    ph_level: float
    organic_carbon_percent: float
    electrical_conductivity: float
    micronutrients: Dict[str, float] = field(default_factory=dict)
    health_score: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "farm_id": self.farm_id,
            "test_date": self.test_date.isoformat(),
            "nitrogen_kg_per_ha": self.nitrogen_kg_per_ha,
            "phosphorus_kg_per_ha": self.phosphorus_kg_per_ha,
            "potassium_kg_per_ha": self.potassium_kg_per_ha,
            "ph_level": self.ph_level,
            "organic_carbon_percent": self.organic_carbon_percent,
            "electrical_conductivity": self.electrical_conductivity,
            "micronutrients": self.micronutrients,
            "health_score": self.health_score,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class WeatherData:
    """Weather data model."""
    location: Dict[str, float]  # {lat, lon}
    timestamp: datetime
    temperature_celsius: float
    rainfall_mm: float
    humidity_percent: float
    wind_speed_kmph: float
    solar_radiation: float
    forecast_source: str
    is_forecast: bool
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "location": self.location,
            "timestamp": self.timestamp.isoformat(),
            "temperature_celsius": self.temperature_celsius,
            "rainfall_mm": self.rainfall_mm,
            "humidity_percent": self.humidity_percent,
            "wind_speed_kmph": self.wind_speed_kmph,
            "solar_radiation": self.solar_radiation,
            "forecast_source": self.forecast_source,
            "is_forecast": self.is_forecast,
            "confidence": self.confidence,
        }


@dataclass
class InputRecord:
    """Input record model."""
    input_type: str  # fertilizer, pesticide, water, seed
    name: str
    quantity: float
    unit: str
    application_date: datetime
    cost: float
    is_organic: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "input_type": self.input_type,
            "name": self.name,
            "quantity": self.quantity,
            "unit": self.unit,
            "application_date": self.application_date.isoformat(),
            "cost": self.cost,
            "is_organic": self.is_organic,
        }


@dataclass
class Product:
    """Product model for marketplace."""
    product_id: str
    farmer_id: str
    farm_id: str
    product_type: ProductCategory
    name: str
    quantity_kg: float
    price_per_kg: float
    harvest_date: datetime
    images: List[str] = field(default_factory=list)
    pure_product_score: float = 0.0
    status: str = "available"  # available, reserved, sold
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "product_id": self.product_id,
            "farmer_id": self.farmer_id,
            "farm_id": self.farm_id,
            "product_type": self.product_type.value,
            "name": self.name,
            "quantity_kg": self.quantity_kg,
            "price_per_kg": self.price_per_kg,
            "harvest_date": self.harvest_date.isoformat(),
            "images": self.images,
            "pure_product_score": self.pure_product_score,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class PureProductScoreBreakdown:
    """Pure Product Score breakdown."""
    overall_score: float
    traceability_score: float
    sustainability_score: float
    quality_score: float
    justification: str
    data_completeness: float
    verification_status: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall_score": self.overall_score,
            "traceability_score": self.traceability_score,
            "sustainability_score": self.sustainability_score,
            "quality_score": self.quality_score,
            "justification": self.justification,
            "data_completeness": self.data_completeness,
            "verification_status": self.verification_status,
        }


@dataclass
class Recommendation:
    """Recommendation model."""
    recommendation_id: str
    query_id: str
    user_id: str
    farm_id: Optional[str]
    timestamp: datetime
    recommendation_text: str
    reasoning_chain: List[str] = field(default_factory=list)
    confidence: float = 0.0
    agent_contributions: List[str] = field(default_factory=list)
    language: str = "en"
    voice_audio_url: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "recommendation_id": self.recommendation_id,
            "query_id": self.query_id,
            "user_id": self.user_id,
            "farm_id": self.farm_id,
            "timestamp": self.timestamp.isoformat(),
            "recommendation_text": self.recommendation_text,
            "reasoning_chain": self.reasoning_chain,
            "confidence": self.confidence,
            "agent_contributions": self.agent_contributions,
            "language": self.language,
            "voice_audio_url": self.voice_audio_url,
            "created_at": self.created_at.isoformat(),
        }
