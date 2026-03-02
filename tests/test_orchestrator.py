"""Tests for Orchestrator Agent."""

import pytest
from datetime import datetime

from src.core import OrchestratorAgent, Query, UserContext, get_registry
from src.agents import WeatherAgent, SoilAgent


@pytest.fixture
def registry():
    """Get agent registry."""
    return get_registry()


@pytest.fixture
def orchestrator(registry):
    """Create orchestrator instance."""
    # Register agents
    registry.register_agent_class("weather_agent", WeatherAgent)
    registry.register_agent_class("soil_agent", SoilAgent)
    
    return OrchestratorAgent()


@pytest.fixture
def query():
    """Create a test query."""
    return Query(
        text="What should I do with my wheat field?",
        user_id="farmer_001",
        farm_id="farm_001",
        intent="general",
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
    )


class TestOrchestratorAgent:
    """Tests for Orchestrator Agent."""
    
    def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator initialization."""
        assert orchestrator.agent_name == "orchestrator"
        assert orchestrator.registry is not None
    
    def test_determine_relevant_agents(self, orchestrator, query):
        """Test agent selection based on intent."""
        # Test irrigation intent
        query.intent = "irrigation"
        agents = orchestrator._determine_relevant_agents(query)
        assert "weather_agent" in agents
        assert "soil_agent" in agents
        
        # Test pest detection intent
        query.intent = "pest_detection"
        agents = orchestrator._determine_relevant_agents(query)
        assert len(agents) > 0
        
        # Test general intent
        query.intent = "general"
        agents = orchestrator._determine_relevant_agents(query)
        assert len(agents) > 0
    
    @pytest.mark.asyncio
    async def test_orchestrator_analyze(self, orchestrator, query, context):
        """Test orchestrator analysis."""
        output = await orchestrator.analyze(query, context)
        
        assert output.agent_name == "orchestrator"
        assert output.recommendation is not None
        assert 0 <= output.confidence <= 1
        assert len(output.reasoning_chain) > 0
    
    def test_collect_data_sources(self, orchestrator):
        """Test data source collection."""
        from src.core import AgentOutput
        
        outputs = [
            AgentOutput(
                agent_name="weather_agent",
                query_id="q1",
                timestamp=datetime.now(),
                analysis={},
                recommendation="Test",
                confidence=0.8,
                data_sources=["IMD", "GFS"],
            ),
            AgentOutput(
                agent_name="soil_agent",
                query_id="q1",
                timestamp=datetime.now(),
                analysis={},
                recommendation="Test",
                confidence=0.7,
                data_sources=["Soil Health Card"],
            ),
        ]
        
        sources = orchestrator._collect_data_sources(outputs)
        assert "IMD" in sources
        assert "GFS" in sources
        assert "Soil Health Card" in sources
    
    def test_fallback_synthesis(self, orchestrator):
        """Test fallback synthesis when LLM fails."""
        from src.core import AgentOutput
        
        outputs = [
            AgentOutput(
                agent_name="weather_agent",
                query_id="q1",
                timestamp=datetime.now(),
                analysis={},
                recommendation="Irrigate tomorrow",
                confidence=0.9,
                data_sources=[],
            ),
            AgentOutput(
                agent_name="soil_agent",
                query_id="q1",
                timestamp=datetime.now(),
                analysis={},
                recommendation="Add nitrogen",
                confidence=0.7,
                data_sources=[],
            ),
        ]
        
        synthesis = orchestrator._fallback_synthesis(outputs)
        assert synthesis["recommendation"] == "Irrigate tomorrow"
        assert synthesis["confidence"] == 0.9
