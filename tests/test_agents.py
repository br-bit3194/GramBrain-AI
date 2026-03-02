"""Tests for AI agents."""

import pytest
import asyncio
from datetime import datetime

from backend.src.core import Query, UserContext
from backend.src.agents import (
    WeatherAgent,
    SoilAgent,
    CropAdvisoryAgent,
    PestAgent,
    IrrigationAgent,
    YieldAgent,
    MarketAgent,
    SustainabilityAgent,
    MarketplaceAgent,
    FarmerInteractionAgent,
    VillageAgent,
)


@pytest.fixture
def query():
    """Create a test query."""
    return Query(
        text="Should I irrigate my wheat field?",
        user_id="farmer_001",
        farm_id="farm_001",
        intent="irrigation",
        language="en",
    )


@pytest.fixture
def context():
    """Create a test user context."""
    return UserContext(
        user_id="farmer_001",
        farm_id="farm_001",
        farm_location={"lat": 28.5, "lon": 77.0},
        farm_size_hectares=2.0,
        crop_type="wheat",
        growth_stage="tillering",
        soil_type="loamy",
        language_preference="en",
    )


class TestWeatherAgent:
    """Tests for Weather Agent."""
    
    @pytest.mark.asyncio
    async def test_weather_agent_initialization(self):
        """Test weather agent initialization."""
        agent = WeatherAgent()
        assert agent.agent_name == "weather_agent"
    
    @pytest.mark.asyncio
    async def test_weather_agent_analyze(self, query, context):
        """Test weather agent analysis."""
        agent = WeatherAgent()
        output = await agent.analyze(query, context)
        
        assert output.agent_name == "weather_agent"
        assert output.recommendation is not None
        assert 0 <= output.confidence <= 1
        assert len(output.data_sources) > 0
    
    @pytest.mark.asyncio
    async def test_weather_agent_irrigation_analysis(self, context):
        """Test irrigation need analysis."""
        agent = WeatherAgent()
        analysis = await agent._analyze_irrigation_need(
            {
                "rainfall_forecast_mm": [0, 5, 12, 8, 0, 0, 0],
                "confidence": 0.85,
            },
            context,
        )
        
        assert "skip_irrigation" in analysis
        assert "rainfall_expected_mm" in analysis
        assert "water_savings_liters" in analysis


class TestSoilAgent:
    """Tests for Soil Agent."""
    
    @pytest.mark.asyncio
    async def test_soil_agent_initialization(self):
        """Test soil agent initialization."""
        agent = SoilAgent()
        assert agent.agent_name == "soil_agent"
    
    @pytest.mark.asyncio
    async def test_soil_agent_analyze(self, query, context):
        """Test soil agent analysis."""
        agent = SoilAgent()
        output = await agent.analyze(query, context)
        
        assert output.agent_name == "soil_agent"
        assert output.recommendation is not None
        assert 0 <= output.confidence <= 1
    
    @pytest.mark.asyncio
    async def test_soil_health_analysis(self, context):
        """Test soil health analysis."""
        agent = SoilAgent()
        soil_data = {
            "nitrogen_kg_per_ha": 180,
            "phosphorus_kg_per_ha": 25,
            "potassium_kg_per_ha": 280,
            "ph_level": 6.8,
            "organic_carbon_percent": 0.45,
        }
        
        analysis = await agent._analyze_soil_health(soil_data, context)
        
        assert "health_score" in analysis
        assert "deficiencies" in analysis
        assert "npk_levels" in analysis


class TestCropAdvisoryAgent:
    """Tests for Crop Advisory Agent."""
    
    @pytest.mark.asyncio
    async def test_crop_advisory_agent_initialization(self):
        """Test crop advisory agent initialization."""
        agent = CropAdvisoryAgent()
        assert agent.agent_name == "crop_advisory_agent"
    
    @pytest.mark.asyncio
    async def test_crop_advisory_analyze(self, query, context):
        """Test crop advisory analysis."""
        agent = CropAdvisoryAgent()
        output = await agent.analyze(query, context)
        
        assert output.agent_name == "crop_advisory_agent"
        assert output.recommendation is not None


class TestPestAgent:
    """Tests for Pest Agent."""
    
    @pytest.mark.asyncio
    async def test_pest_agent_initialization(self):
        """Test pest agent initialization."""
        agent = PestAgent()
        assert agent.agent_name == "pest_agent"
    
    @pytest.mark.asyncio
    async def test_pest_risk_analysis(self, context):
        """Test pest risk analysis."""
        agent = PestAgent()
        analysis = await agent._analyze_pest_risk(context)
        
        assert "risk_level" in analysis
        assert "high_risk_pests" in analysis
        assert "medium_risk_pests" in analysis


class TestIrrigationAgent:
    """Tests for Irrigation Agent."""
    
    @pytest.mark.asyncio
    async def test_irrigation_agent_initialization(self):
        """Test irrigation agent initialization."""
        agent = IrrigationAgent()
        assert agent.agent_name == "irrigation_agent"
    
    @pytest.mark.asyncio
    async def test_water_requirement_calculation(self, context):
        """Test water requirement calculation."""
        agent = IrrigationAgent()
        analysis = await agent._calculate_water_requirements(context)
        
        assert "daily_requirement_mm" in analysis
        assert "daily_requirement_liters" in analysis
        assert "irrigation_interval_days" in analysis
        assert analysis["daily_requirement_liters"] > 0


class TestYieldAgent:
    """Tests for Yield Agent."""
    
    @pytest.mark.asyncio
    async def test_yield_agent_initialization(self):
        """Test yield agent initialization."""
        agent = YieldAgent()
        assert agent.agent_name == "yield_agent"
    
    @pytest.mark.asyncio
    async def test_yield_prediction(self, context):
        """Test yield prediction."""
        agent = YieldAgent()
        analysis = await agent._predict_yield(context)
        
        assert "predicted_yield_kg_per_ha" in analysis
        assert "lower_bound_kg_per_ha" in analysis
        assert "upper_bound_kg_per_ha" in analysis
        assert analysis["predicted_yield_kg_per_ha"] > 0


class TestMarketAgent:
    """Tests for Market Agent."""
    
    @pytest.mark.asyncio
    async def test_market_agent_initialization(self):
        """Test market agent initialization."""
        agent = MarketAgent()
        assert agent.agent_name == "market_agent"
    
    @pytest.mark.asyncio
    async def test_market_analysis(self, context):
        """Test market analysis."""
        agent = MarketAgent()
        analysis = await agent._analyze_market(context)
        
        assert "current_price_per_quintal" in analysis
        assert "predicted_price_per_quintal" in analysis
        assert "price_trend" in analysis


class TestSustainabilityAgent:
    """Tests for Sustainability Agent."""
    
    @pytest.mark.asyncio
    async def test_sustainability_agent_initialization(self):
        """Test sustainability agent initialization."""
        agent = SustainabilityAgent()
        assert agent.agent_name == "sustainability_agent"
    
    @pytest.mark.asyncio
    async def test_sustainability_metrics(self, context):
        """Test sustainability metrics calculation."""
        agent = SustainabilityAgent()
        analysis = await agent._calculate_sustainability_metrics(context)
        
        assert "soil_health_score" in analysis
        assert "water_efficiency_score" in analysis
        assert "carbon_footprint_kg_co2_per_ha" in analysis
        assert "sustainability_index" in analysis


class TestMarketplaceAgent:
    """Tests for Marketplace Agent."""
    
    @pytest.mark.asyncio
    async def test_marketplace_agent_initialization(self):
        """Test marketplace agent initialization."""
        agent = MarketplaceAgent()
        assert agent.agent_name == "marketplace_agent"
    
    @pytest.mark.asyncio
    async def test_pure_product_score_calculation(self, context):
        """Test Pure Product Score calculation."""
        agent = MarketplaceAgent()
        analysis = await agent._calculate_pure_product_score(context)
        
        assert "overall_score" in analysis
        assert "category" in analysis
        assert 0 <= analysis["overall_score"] <= 100
        assert analysis["category"] in ["Pure", "Organic", "Sustainable", "Conventional"]


class TestFarmerInteractionAgent:
    """Tests for Farmer Interaction Agent."""
    
    @pytest.mark.asyncio
    async def test_farmer_interaction_agent_initialization(self):
        """Test farmer interaction agent initialization."""
        agent = FarmerInteractionAgent()
        assert agent.agent_name == "farmer_interaction_agent"
    
    @pytest.mark.asyncio
    async def test_interaction_processing(self, query, context):
        """Test interaction processing."""
        agent = FarmerInteractionAgent()
        analysis = await agent._process_interaction(query, context)
        
        assert "language" in analysis
        assert "is_voice_input" in analysis
        assert "simplified_query" in analysis


class TestVillageAgent:
    """Tests for Village Agent."""
    
    @pytest.mark.asyncio
    async def test_village_agent_initialization(self):
        """Test village agent initialization."""
        agent = VillageAgent()
        assert agent.agent_name == "village_agent"
    
    @pytest.mark.asyncio
    async def test_village_data_aggregation(self, context):
        """Test village data aggregation."""
        agent = VillageAgent()
        analysis = await agent._aggregate_village_data(context)
        
        assert "total_farmers" in analysis
        assert "total_area_hectares" in analysis
        assert "crop_distribution" in analysis
        assert "collective_risks" in analysis
