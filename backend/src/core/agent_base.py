"""Base class for all AI agents in GramBrain system."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime
import uuid


@dataclass
class AgentOutput:
    """Standardized output format for all agents."""
    agent_name: str
    query_id: str
    timestamp: datetime
    analysis: Dict[str, Any]
    recommendation: str
    confidence: float  # 0-1
    data_sources: List[str] = field(default_factory=list)
    rag_context: List[str] = field(default_factory=list)
    reasoning_chain: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "agent_name": self.agent_name,
            "query_id": self.query_id,
            "timestamp": self.timestamp.isoformat(),
            "analysis": self.analysis,
            "recommendation": self.recommendation,
            "confidence": self.confidence,
            "data_sources": self.data_sources,
            "rag_context": self.rag_context,
            "reasoning_chain": self.reasoning_chain,
        }


@dataclass
class Query:
    """User query with context."""
    query_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""
    user_id: str = ""
    farm_id: Optional[str] = None
    intent: str = ""  # irrigation, pest_detection, yield_forecast, etc.
    entities: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    language: str = "en"


@dataclass
class UserContext:
    """User profile and farm context."""
    user_id: str
    farm_id: Optional[str] = None
    farm_location: Optional[Dict[str, float]] = None  # {lat, lon}
    farm_size_hectares: Optional[float] = None
    crop_type: Optional[str] = None
    growth_stage: Optional[str] = None
    soil_type: Optional[str] = None
    language_preference: str = "en"
    role: str = "farmer"  # farmer, village_leader, policymaker, consumer


class Agent(ABC):
    """Base class for all specialized AI agents."""
    
    def __init__(self, agent_name: str):
        """Initialize agent with name and dependencies."""
        self.agent_name = agent_name
        self.llm_client = None
        self.rag_client = None
        self.data_client = None
    
    @abstractmethod
    async def analyze(self, query: Query, context: UserContext) -> AgentOutput:
        """
        Analyze query and generate recommendation.
        
        Args:
            query: User query with intent and entities
            context: User profile and farm context
            
        Returns:
            AgentOutput with analysis, recommendation, and confidence
        """
        pass
    
    def set_llm_client(self, llm_client):
        """Inject LLM client dependency."""
        self.llm_client = llm_client
        return self
    
    def set_rag_client(self, rag_client):
        """Inject RAG client dependency."""
        self.rag_client = rag_client
        return self
    
    def set_data_client(self, data_client):
        """Inject data client dependency."""
        self.data_client = data_client
        return self
    
    async def retrieve_rag_context(self, query_text: str, top_k: int = 5) -> List[str]:
        """
        Retrieve relevant knowledge chunks from RAG database.
        
        Args:
            query_text: Query to search for
            top_k: Number of top results to return
            
        Returns:
            List of relevant knowledge chunks
        """
        if not self.rag_client:
            return []
        
        results = await self.rag_client.search(query_text, top_k=top_k)
        return [result.get("content", "") for result in results]
    
    async def call_llm(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000) -> str:
        """
        Call LLM with prompt.
        
        Args:
            prompt: Prompt text
            temperature: LLM temperature (0-1)
            max_tokens: Maximum tokens in response
            
        Returns:
            LLM response text
        """
        if not self.llm_client:
            raise RuntimeError(f"LLM client not configured for {self.agent_name}")
        
        response = await self.llm_client.invoke(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Handle both old string responses and new BedrockResponse objects
        if isinstance(response, str):
            return response
        else:
            # BedrockResponse object
            return response.text
    
    def _create_output(
        self,
        query: Query,
        analysis: Dict[str, Any],
        recommendation: str,
        confidence: float,
        data_sources: List[str] = None,
        rag_context: List[str] = None,
        reasoning_chain: List[str] = None,
    ) -> AgentOutput:
        """Helper to create standardized agent output."""
        return AgentOutput(
            agent_name=self.agent_name,
            query_id=query.query_id,
            timestamp=datetime.now(),
            analysis=analysis,
            recommendation=recommendation,
            confidence=confidence,
            data_sources=data_sources or [],
            rag_context=rag_context or [],
            reasoning_chain=reasoning_chain or [],
        )
