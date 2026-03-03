"""Tests for Orchestrator Agent."""

import pytest
import asyncio
from datetime import datetime

from backend.src.core import OrchestratorAgent, Query, UserContext, get_registry, AgentOutput, Agent
from backend.src.agents import WeatherAgent, SoilAgent


@pytest.fixture
def registry():
    """Get agent registry."""
    reg = get_registry()
    # Clear any existing agents
    reg.shutdown_all()
    return reg


@pytest.fixture
async def orchestrator(registry):
    """Create orchestrator instance."""
    # Register agents
    registry.register_agent_class("weather_agent", WeatherAgent)
    registry.register_agent_class("soil_agent", SoilAgent)
    
    orch = OrchestratorAgent()
    await orch.initialize()
    yield orch
    await orch.shutdown()


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
    
    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator initialization."""
        assert orchestrator.agent_name == "orchestrator"
        assert orchestrator.registry is not None
        assert orchestrator._initialized is True
        assert orchestrator.metrics is not None
    
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
    
    def test_validate_agent_output_valid(self, orchestrator):
        """Test validation of valid agent output."""
        output = AgentOutput(
            agent_name="test_agent",
            query_id="q1",
            timestamp=datetime.now(),
            analysis={"key": "value"},
            recommendation="Test recommendation",
            confidence=0.8,
            data_sources=["source1"],
            rag_context=["context1"],
            reasoning_chain=["step1"],
        )
        
        assert orchestrator._validate_agent_output(output, "test_agent") is True
    
    def test_validate_agent_output_invalid_type(self, orchestrator):
        """Test validation rejects non-AgentOutput."""
        assert orchestrator._validate_agent_output("not an output", "test") is False
        assert orchestrator._validate_agent_output({"dict": "value"}, "test") is False
    
    def test_validate_agent_output_invalid_confidence(self, orchestrator):
        """Test validation rejects invalid confidence values."""
        output = AgentOutput(
            agent_name="test_agent",
            query_id="q1",
            timestamp=datetime.now(),
            analysis={},
            recommendation="Test",
            confidence=1.5,  # Invalid: > 1
        )
        
        assert orchestrator._validate_agent_output(output, "test_agent") is False
        
        output.confidence = -0.1  # Invalid: < 0
        assert orchestrator._validate_agent_output(output, "test_agent") is False
    
    @pytest.mark.asyncio
    async def test_agent_timeout_handling(self, query, context, registry):
        """Test that agent timeouts are handled properly."""
        # Create orchestrator with shorter timeout for testing
        orch = OrchestratorAgent()
        orch.agent_timeout = 1  # 1 second timeout for testing
        await orch.initialize()
        
        # Create a slow agent that will timeout
        class SlowAgent(Agent):
            def __init__(self):
                super().__init__("slow_agent")
            
            async def analyze(self, query, context):
                await asyncio.sleep(5)  # Longer than timeout
                return AgentOutput(
                    agent_name=self.agent_name,
                    query_id=query.query_id,
                    timestamp=datetime.now(),
                    analysis={},
                    recommendation="Should not reach here",
                    confidence=0.5,
                )
        
        slow_agent = SlowAgent()
        
        # Should raise TimeoutError
        with pytest.raises(asyncio.TimeoutError):
            await orch._execute_agent_with_validation(slow_agent, query, context)
        
        # Check metrics recorded the timeout
        stats = orch.metrics.get_agent_stats("slow_agent")
        assert stats["total_executions"] == 1
        assert stats["timeout_rate"] == 1.0
        assert stats["success_rate"] == 0.0
        
        await orch.shutdown()
    
    @pytest.mark.asyncio
    async def test_agent_error_handling(self, query, context):
        """Test that agent errors are handled properly."""
        # Create orchestrator for this test
        orch = OrchestratorAgent()
        await orch.initialize()
        
        # Create an agent that raises an error
        class ErrorAgent(Agent):
            def __init__(self):
                super().__init__("error_agent")
            
            async def analyze(self, query, context):
                raise ValueError("Test error")
        
        error_agent = ErrorAgent()
        
        # Should raise the error
        with pytest.raises(ValueError):
            await orch._execute_agent_with_validation(error_agent, query, context)
        
        # Check metrics recorded the error
        stats = orch.metrics.get_agent_stats("error_agent")
        assert stats["total_executions"] == 1
        assert stats["success_rate"] == 0.0
        assert stats["timeout_rate"] == 0.0
        
        await orch.shutdown()
    
    @pytest.mark.asyncio
    async def test_metrics_tracking(self, orchestrator, query, context):
        """Test that metrics are tracked correctly."""
        # Execute analysis
        await orchestrator.analyze(query, context)
        
        # Check overall metrics
        overall = orchestrator.metrics.get_overall_stats()
        assert overall["total_executions"] > 0
        
        # Check agent-specific metrics
        metrics = orchestrator.get_metrics()
        assert "overall" in metrics
        assert "by_agent" in metrics
    
    @pytest.mark.asyncio
    async def test_dispatch_with_invalid_output(self, query, context, registry):
        """Test that invalid agent outputs are filtered out."""
        # Create a fresh orchestrator for this test
        orch = OrchestratorAgent()
        await orch.initialize()
        
        # Create an agent that returns invalid output
        class InvalidOutputAgent(Agent):
            def __init__(self):
                super().__init__("invalid_agent")
            
            async def analyze(self, query, context):
                # Return output with invalid confidence
                return AgentOutput(
                    agent_name=self.agent_name,
                    query_id=query.query_id,
                    timestamp=datetime.now(),
                    analysis={},
                    recommendation="Test",
                    confidence=2.0,  # Invalid
                )
        
        # Register the invalid agent
        registry.register_agent_class("invalid_agent", InvalidOutputAgent)
        
        # Dispatch should filter out invalid output
        outputs = await orch._dispatch_to_agents(
            ["invalid_agent"], query, context
        )
        
        # Should be empty because output was invalid
        assert len(outputs) == 0
        
        # Check metrics recorded the failure
        stats = orch.metrics.get_agent_stats("invalid_agent")
        assert stats["total_executions"] == 1
        assert stats["success_rate"] == 0.0
        
        await orch.shutdown()
    
    @pytest.mark.asyncio
    async def test_shutdown(self, registry):
        """Test orchestrator shutdown."""
        # Register common agents that will be pre-instantiated
        registry.register_agent_class("weather_agent", WeatherAgent)
        registry.register_agent_class("soil_agent", SoilAgent)
        registry.register_agent_class("crop_advisory_agent", SoilAgent)  # Use SoilAgent as placeholder
        
        orch = OrchestratorAgent()
        await orch.initialize()
        
        assert orch._initialized is True
        # At least one common agent should be instantiated
        assert len(orch._active_agents) > 0
        
        await orch.shutdown()
        
        assert orch._initialized is False
        assert len(orch._active_agents) == 0
