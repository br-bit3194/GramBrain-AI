"""GramBrain System - Main entry point for the multi-agent platform."""

import asyncio
import os
from typing import Optional
from .core import OrchestratorAgent, Query, UserContext, get_registry
from .agents import (
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
from .llm import BedrockClient
from .rag import RAGClient, InMemoryVectorDB, EmbeddingClient
from .data import DynamoDBClient, UserRepository, FarmRepository, RecommendationRepository, ProductRepository


class GramBrainSystem:
    """Main GramBrain AI system orchestrator."""
    
    def __init__(
        self,
        aws_region: str = "us-east-1",
        use_mock_llm: bool = False,
        use_mock_rag: bool = True,
    ):
        """
        Initialize GramBrain system.
        
        Args:
            aws_region: AWS region for Bedrock
            use_mock_llm: Use mock LLM for testing
            use_mock_rag: Use in-memory RAG for testing
        """
        self.registry = get_registry()
        self.aws_region = aws_region
        self.use_mock_llm = use_mock_llm
        self.use_mock_rag = use_mock_rag
        
        # Initialize components
        self.llm_client = None
        self.rag_client = None
        self.orchestrator = None
        
        # Initialize DynamoDB repositories
        env = os.getenv("DYNAMODB_ENV", "dev")
        endpoint_url = os.getenv("DYNAMODB_ENDPOINT_URL")
        self.dynamodb_client = DynamoDBClient(region_name=aws_region, endpoint_url=endpoint_url)
        self.user_repo = UserRepository(self.dynamodb_client, env=env)
        self.farm_repo = FarmRepository(self.dynamodb_client, env=env)
        self.recommendation_repo = RecommendationRepository(self.dynamodb_client, env=env)
        self.product_repo = ProductRepository(self.dynamodb_client, env=env)
    
    async def initialize(self) -> None:
        """Initialize all system components."""
        # Initialize LLM client
        if not self.use_mock_llm:
            self.llm_client = BedrockClient(region=self.aws_region)
        
        # Initialize RAG client
        if self.use_mock_rag:
            vector_db = InMemoryVectorDB()
            embedding_client = EmbeddingClient(region=self.aws_region)
            self.rag_client = RAGClient(vector_db, embedding_client)
        else:
            # Use OpenSearch for production
            from .rag import create_rag_client
            self.rag_client = await create_rag_client()
        
        # Register agents
        self._register_agents()
        
        # Initialize orchestrator
        self.orchestrator = OrchestratorAgent()
        if self.llm_client:
            self.orchestrator.set_llm_client(self.llm_client)
        if self.rag_client:
            self.orchestrator.set_rag_client(self.rag_client)
        
        # Initialize orchestrator (pre-instantiate common agents)
        await self.orchestrator.initialize()
    
    def _register_agents(self) -> None:
        """Register all specialized agents."""
        self.registry.register_agent_class("weather_agent", WeatherAgent)
        self.registry.register_agent_class("soil_agent", SoilAgent)
        self.registry.register_agent_class("crop_advisory_agent", CropAdvisoryAgent)
        self.registry.register_agent_class("pest_agent", PestAgent)
        self.registry.register_agent_class("irrigation_agent", IrrigationAgent)
        self.registry.register_agent_class("yield_agent", YieldAgent)
        self.registry.register_agent_class("market_agent", MarketAgent)
        self.registry.register_agent_class("sustainability_agent", SustainabilityAgent)
        self.registry.register_agent_class("marketplace_agent", MarketplaceAgent)
        self.registry.register_agent_class("farmer_interaction_agent", FarmerInteractionAgent)
        self.registry.register_agent_class("village_agent", VillageAgent)
    
    async def process_query(
        self,
        query_text: str,
        user_id: str,
        farm_id: Optional[str] = None,
        farm_location: Optional[dict] = None,
        farm_size_hectares: Optional[float] = None,
        crop_type: Optional[str] = None,
        growth_stage: Optional[str] = None,
        soil_type: Optional[str] = None,
        language: str = "en",
    ) -> dict:
        """
        Process a user query and return recommendation.
        
        Args:
            query_text: User query text
            user_id: User identifier
            farm_id: Farm identifier
            farm_location: Farm location {lat, lon}
            farm_size_hectares: Farm size in hectares
            crop_type: Current crop type
            growth_stage: Crop growth stage
            soil_type: Soil type
            language: User language preference
            
        Returns:
            Recommendation with reasoning and confidence
        """
        # Create query object
        query = Query(
            text=query_text,
            user_id=user_id,
            farm_id=farm_id,
            intent=self._extract_intent(query_text),
            language=language,
        )
        
        # Create user context
        context = UserContext(
            user_id=user_id,
            farm_id=farm_id,
            farm_location=farm_location,
            farm_size_hectares=farm_size_hectares,
            crop_type=crop_type,
            growth_stage=growth_stage,
            soil_type=soil_type,
            language_preference=language,
        )
        
        # Process through orchestrator
        output = asyncio.run(self.orchestrator.analyze(query, context))
        
        return output.to_dict()
    
    def _extract_intent(self, query_text: str) -> str:
        """
        Extract intent from query text.
        
        Args:
            query_text: User query
            
        Returns:
            Intent type
        """
        query_lower = query_text.lower()
        
        intent_keywords = {
            "irrigation": ["water", "irrigate", "irrigation", "moisture"],
            "pest_detection": ["pest", "disease", "insect", "damage", "spots"],
            "yield_forecast": ["yield", "harvest", "production", "output"],
            "market_advice": ["price", "market", "sell", "selling"],
            "soil_health": ["soil", "fertility", "nutrients", "npk"],
            "crop_planning": ["plant", "sow", "crop", "variety"],
        }
        
        for intent, keywords in intent_keywords.items():
            if any(kw in query_lower for kw in keywords):
                return intent
        
        return "general"
    
    async def add_knowledge(
        self,
        chunk_id: str,
        content: str,
        source: str,
        topic: str,
        crop_type: Optional[str] = None,
        region: Optional[str] = None,
    ) -> None:
        """
        Add knowledge chunk to RAG database.
        
        Args:
            chunk_id: Unique chunk identifier
            content: Knowledge content
            source: Knowledge source (research_paper, best_practice, etc.)
            topic: Topic/category
            crop_type: Relevant crop type
            region: Relevant region
        """
        if not self.rag_client:
            raise RuntimeError("RAG client not initialized")
        
        metadata = {
            "source": source,
            "topic": topic,
            "crop_type": crop_type,
            "region": region,
        }
        
        await self.rag_client.add_knowledge(chunk_id, content, metadata)
    
    def shutdown(self) -> None:
        """Shutdown system and cleanup resources."""
        # Shutdown orchestrator
        if self.orchestrator:
            asyncio.run(self.orchestrator.shutdown())
        
        # Shutdown registry
        self.registry.shutdown_all()


async def main():
    """Example usage of GramBrain system."""
    # Initialize system
    system = GramBrainSystem(use_mock_llm=True, use_mock_rag=True)
    await system.initialize()
    
    # Add some knowledge
    await system.add_knowledge(
        chunk_id="wheat_irrigation_1",
        content="Wheat requires 450-600mm of water during growing season. Optimal irrigation timing is at tillering and grain filling stages.",
        source="best_practice",
        topic="irrigation",
        crop_type="wheat",
        region="north_india",
    )
    
    # Process a query
    result = system.process_query(
        query_text="Should I irrigate my wheat field today?",
        user_id="farmer_001",
        farm_id="farm_001",
        farm_location={"lat": 28.5, "lon": 77.0},
        farm_size_hectares=2.0,
        crop_type="wheat",
        growth_stage="tillering",
        soil_type="loamy",
    )
    
    print("Recommendation:")
    print(result["recommendation"])
    print(f"\nConfidence: {result['confidence']:.2%}")
    print(f"\nReasoning:")
    for step in result["reasoning_chain"]:
        print(f"  - {step}")
    
    # Cleanup
    system.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
