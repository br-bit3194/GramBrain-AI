"""Vector database interface for RAG knowledge retrieval."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class KnowledgeChunk:
    """Represents a chunk of knowledge in the vector database."""
    chunk_id: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any]
    similarity_score: float = 0.0


class VectorDB(ABC):
    """Abstract base class for vector database implementations."""
    
    @abstractmethod
    async def add_chunk(
        self,
        chunk_id: str,
        content: str,
        embedding: List[float],
        metadata: Dict[str, Any],
    ) -> None:
        """Add a knowledge chunk to the database."""
        pass
    
    @abstractmethod
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        min_similarity: float = 0.7,
    ) -> List[KnowledgeChunk]:
        """Search for similar chunks."""
        pass
    
    @abstractmethod
    async def delete_chunk(self, chunk_id: str) -> None:
        """Delete a chunk from the database."""
        pass
    
    @abstractmethod
    async def update_chunk(
        self,
        chunk_id: str,
        content: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update a chunk in the database."""
        pass


class InMemoryVectorDB(VectorDB):
    """Simple in-memory vector database for development/testing."""
    
    def __init__(self):
        """Initialize in-memory database."""
        self.chunks: Dict[str, KnowledgeChunk] = {}
    
    async def add_chunk(
        self,
        chunk_id: str,
        content: str,
        embedding: List[float],
        metadata: Dict[str, Any],
    ) -> None:
        """Add chunk to memory."""
        chunk = KnowledgeChunk(
            chunk_id=chunk_id,
            content=content,
            embedding=embedding,
            metadata=metadata,
        )
        self.chunks[chunk_id] = chunk
    
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        min_similarity: float = 0.7,
    ) -> List[KnowledgeChunk]:
        """Search using cosine similarity."""
        results = []
        
        for chunk in self.chunks.values():
            similarity = self._cosine_similarity(query_embedding, chunk.embedding)
            
            if similarity >= min_similarity:
                chunk.similarity_score = similarity
                results.append(chunk)
        
        # Sort by similarity and return top_k
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        return results[:top_k]
    
    async def delete_chunk(self, chunk_id: str) -> None:
        """Delete chunk from memory."""
        if chunk_id in self.chunks:
            del self.chunks[chunk_id]
    
    async def update_chunk(
        self,
        chunk_id: str,
        content: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update chunk in memory."""
        if chunk_id not in self.chunks:
            raise ValueError(f"Chunk {chunk_id} not found")
        
        chunk = self.chunks[chunk_id]
        if content is not None:
            chunk.content = content
        if embedding is not None:
            chunk.embedding = embedding
        if metadata is not None:
            chunk.metadata.update(metadata)
    
    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a ** 2 for a in vec1) ** 0.5
        magnitude2 = sum(b ** 2 for b in vec2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)


class OpenSearchVectorDB(VectorDB):
    """OpenSearch vector database implementation."""
    
    def __init__(self, endpoint: str, index_name: str = "grambrain-knowledge"):
        """
        Initialize OpenSearch client.
        
        Args:
            endpoint: OpenSearch endpoint URL
            index_name: Index name for knowledge chunks
        """
        self.endpoint = endpoint
        self.index_name = index_name
        # TODO: Initialize OpenSearch client
        # from opensearchpy import OpenSearch
        # self.client = OpenSearch([{"host": endpoint, "port": 443}])
    
    async def add_chunk(
        self,
        chunk_id: str,
        content: str,
        embedding: List[float],
        metadata: Dict[str, Any],
    ) -> None:
        """Add chunk to OpenSearch."""
        # TODO: Implement OpenSearch indexing
        pass
    
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        min_similarity: float = 0.7,
    ) -> List[KnowledgeChunk]:
        """Search OpenSearch."""
        # TODO: Implement OpenSearch search
        return []
    
    async def delete_chunk(self, chunk_id: str) -> None:
        """Delete chunk from OpenSearch."""
        # TODO: Implement OpenSearch deletion
        pass
    
    async def update_chunk(
        self,
        chunk_id: str,
        content: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update chunk in OpenSearch."""
        # TODO: Implement OpenSearch update
        pass
