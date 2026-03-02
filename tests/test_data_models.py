"""Tests for data models."""

import pytest
from datetime import datetime

from backend.src.data.models import (
    User,
    Farm,
    CropCycle,
    SoilHealthData,
    WeatherData,
    InputRecord,
    Product,
    Recommendation,
    UserRole,
    GrowthStage,
    IrrigationType,
    ProductCategory,
)


class TestUserModel:
    """Tests for User model."""
    
    def test_user_creation(self):
        """Test user creation."""
        user = User(
            user_id="user_001",
            phone_number="9876543210",
            name="Ramesh Kumar",
            language_preference="hi",
            role=UserRole.FARMER,
        )
        
        assert user.user_id == "user_001"
        assert user.phone_number == "9876543210"
        assert user.name == "Ramesh Kumar"
        assert user.role == UserRole.FARMER
    
    def test_user_to_dict(self):
        """Test user serialization."""
        user = User(
            user_id="user_001",
            phone_number="9876543210",
            name="Ramesh Kumar",
        )
        
        user_dict = user.to_dict()
        assert user_dict["user_id"] == "user_001"
        assert user_dict["phone_number"] == "9876543210"
        assert "created_at" in user_dict


class TestFarmModel:
    """Tests for Farm model."""
    
    def test_farm_creation(self):
        """Test farm creation."""
        farm = Farm(
            farm_id="farm_001",
            owner_id="user_001",
            location={"lat": 28.5, "lon": 77.0},
            area_hectares=2.5,
            soil_type="loamy",
            irrigation_type=IrrigationType.DRIP,
        )
        
        assert farm.farm_id == "farm_001"
        assert farm.area_hectares == 2.5
        assert farm.irrigation_type == IrrigationType.DRIP
    
    def test_farm_to_dict(self):
        """Test farm serialization."""
        farm = Farm(
            farm_id="farm_001",
            owner_id="user_001",
            location={"lat": 28.5, "lon": 77.0},
            area_hectares=2.5,
            soil_type="loamy",
            irrigation_type=IrrigationType.DRIP,
        )
        
        farm_dict = farm.to_dict()
        assert farm_dict["farm_id"] == "farm_001"
        assert farm_dict["area_hectares"] == 2.5
        assert farm_dict["irrigation_type"] == "drip"


class TestCropCycleModel:
    """Tests for CropCycle model."""
    
    def test_crop_cycle_creation(self):
        """Test crop cycle creation."""
        now = datetime.now()
        cycle = CropCycle(
            cycle_id="cycle_001",
            farm_id="farm_001",
            crop_type="wheat",
            variety="HD2967",
            planting_date=now,
            expected_harvest_date=now,
            growth_stage=GrowthStage.TILLERING,
        )
        
        assert cycle.cycle_id == "cycle_001"
        assert cycle.crop_type == "wheat"
        assert cycle.growth_stage == GrowthStage.TILLERING
    
    def test_crop_cycle_to_dict(self):
        """Test crop cycle serialization."""
        now = datetime.now()
        cycle = CropCycle(
            cycle_id="cycle_001",
            farm_id="farm_001",
            crop_type="wheat",
            variety="HD2967",
            planting_date=now,
            expected_harvest_date=now,
        )
        
        cycle_dict = cycle.to_dict()
        assert cycle_dict["cycle_id"] == "cycle_001"
        assert cycle_dict["crop_type"] == "wheat"


class TestSoilHealthDataModel:
    """Tests for SoilHealthData model."""
    
    def test_soil_health_creation(self):
        """Test soil health data creation."""
        soil = SoilHealthData(
            farm_id="farm_001",
            test_date=datetime.now(),
            nitrogen_kg_per_ha=200,
            phosphorus_kg_per_ha=30,
            potassium_kg_per_ha=300,
            ph_level=6.8,
            organic_carbon_percent=0.5,
            electrical_conductivity=0.35,
        )
        
        assert soil.farm_id == "farm_001"
        assert soil.nitrogen_kg_per_ha == 200
        assert soil.ph_level == 6.8
    
    def test_soil_health_to_dict(self):
        """Test soil health serialization."""
        soil = SoilHealthData(
            farm_id="farm_001",
            test_date=datetime.now(),
            nitrogen_kg_per_ha=200,
            phosphorus_kg_per_ha=30,
            potassium_kg_per_ha=300,
            ph_level=6.8,
            organic_carbon_percent=0.5,
            electrical_conductivity=0.35,
        )
        
        soil_dict = soil.to_dict()
        assert soil_dict["farm_id"] == "farm_001"
        assert soil_dict["nitrogen_kg_per_ha"] == 200


class TestWeatherDataModel:
    """Tests for WeatherData model."""
    
    def test_weather_data_creation(self):
        """Test weather data creation."""
        weather = WeatherData(
            location={"lat": 28.5, "lon": 77.0},
            timestamp=datetime.now(),
            temperature_celsius=28.5,
            rainfall_mm=15.0,
            humidity_percent=65,
            wind_speed_kmph=12,
            solar_radiation=500,
            forecast_source="IMD",
            is_forecast=True,
            confidence=0.85,
        )
        
        assert weather.temperature_celsius == 28.5
        assert weather.rainfall_mm == 15.0
        assert weather.is_forecast is True
    
    def test_weather_data_to_dict(self):
        """Test weather data serialization."""
        weather = WeatherData(
            location={"lat": 28.5, "lon": 77.0},
            timestamp=datetime.now(),
            temperature_celsius=28.5,
            rainfall_mm=15.0,
            humidity_percent=65,
            wind_speed_kmph=12,
            solar_radiation=500,
            forecast_source="IMD",
            is_forecast=True,
        )
        
        weather_dict = weather.to_dict()
        assert weather_dict["temperature_celsius"] == 28.5
        assert weather_dict["is_forecast"] is True


class TestInputRecordModel:
    """Tests for InputRecord model."""
    
    def test_input_record_creation(self):
        """Test input record creation."""
        record = InputRecord(
            input_type="fertilizer",
            name="Urea",
            quantity=50,
            unit="kg",
            application_date=datetime.now(),
            cost=1500,
            is_organic=False,
        )
        
        assert record.input_type == "fertilizer"
        assert record.name == "Urea"
        assert record.quantity == 50
    
    def test_input_record_to_dict(self):
        """Test input record serialization."""
        record = InputRecord(
            input_type="fertilizer",
            name="Urea",
            quantity=50,
            unit="kg",
            application_date=datetime.now(),
            cost=1500,
        )
        
        record_dict = record.to_dict()
        assert record_dict["input_type"] == "fertilizer"
        assert record_dict["name"] == "Urea"


class TestProductModel:
    """Tests for Product model."""
    
    def test_product_creation(self):
        """Test product creation."""
        product = Product(
            product_id="prod_001",
            farmer_id="farmer_001",
            farm_id="farm_001",
            product_type=ProductCategory.VEGETABLES,
            name="Tomatoes",
            quantity_kg=100,
            price_per_kg=25,
            harvest_date=datetime.now(),
        )
        
        assert product.product_id == "prod_001"
        assert product.product_type == ProductCategory.VEGETABLES
        assert product.quantity_kg == 100
    
    def test_product_to_dict(self):
        """Test product serialization."""
        product = Product(
            product_id="prod_001",
            farmer_id="farmer_001",
            farm_id="farm_001",
            product_type=ProductCategory.VEGETABLES,
            name="Tomatoes",
            quantity_kg=100,
            price_per_kg=25,
            harvest_date=datetime.now(),
        )
        
        product_dict = product.to_dict()
        assert product_dict["product_id"] == "prod_001"
        assert product_dict["product_type"] == "vegetables"


class TestRecommendationModel:
    """Tests for Recommendation model."""
    
    def test_recommendation_creation(self):
        """Test recommendation creation."""
        rec = Recommendation(
            recommendation_id="rec_001",
            query_id="query_001",
            user_id="user_001",
            farm_id="farm_001",
            timestamp=datetime.now(),
            recommendation_text="Irrigate tomorrow",
            confidence=0.85,
        )
        
        assert rec.recommendation_id == "rec_001"
        assert rec.recommendation_text == "Irrigate tomorrow"
        assert rec.confidence == 0.85
    
    def test_recommendation_to_dict(self):
        """Test recommendation serialization."""
        rec = Recommendation(
            recommendation_id="rec_001",
            query_id="query_001",
            user_id="user_001",
            farm_id="farm_001",
            timestamp=datetime.now(),
            recommendation_text="Irrigate tomorrow",
            reasoning_chain=["Step 1", "Step 2"],
            confidence=0.85,
        )
        
        rec_dict = rec.to_dict()
        assert rec_dict["recommendation_id"] == "rec_001"
        assert len(rec_dict["reasoning_chain"]) == 2
