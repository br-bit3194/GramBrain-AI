"""RAG retrieval pipeline for knowledge-grounded recommendations."""

from typing import List, Dict, Any, Optional
from .vector_db import VectorDB, KnowledgeChunk
from .embeddings import EmbeddingClient


class RAGClient:
    """Client for Retrieval-Augmented Generation."""
    
    def __init__(self, vector_db: VectorDB, embedding_client: EmbeddingClient):
        """
        Initialize RAG client.
        
        Args:
            vector_db: Vector database instance
            embedding_client: Embedding generation client
        """
        self.vector_db = vector_db
        self.embedding_client = embedding_client
    
    async def search(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.7,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[KnowledgeChunk]:
        """
        Search for relevant knowledge chunks.
        
        Args:
            query: Search query
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold
            filters: Optional metadata filters
            
        Returns:
            List of relevant knowledge chunks
        """
        # Generate embedding for query
        query_embedding = await self.embedding_client.embed_text(query)
        
        # Search vector database
        results = await self.vector_db.search(
            query_embedding=query_embedding,
            top_k=top_k,
            min_similarity=min_similarity,
        )
        
        # Apply metadata filters if provided
        if filters:
            results = self._apply_filters(results, filters)
        
        return results
    
    async def add_knowledge(
        self,
        chunk_id: str,
        content: str,
        metadata: Dict[str, Any],
    ) -> None:
        """
        Add knowledge chunk to RAG database.
        
        Args:
            chunk_id: Unique chunk identifier
            content: Knowledge content
            metadata: Metadata (source, topic, crop_type, region, etc.)
        """
        # Generate embedding
        embedding = await self.embedding_client.embed_text(content)
        
        # Add to vector database
        await self.vector_db.add_chunk(
            chunk_id=chunk_id,
            content=content,
            embedding=embedding,
            metadata=metadata,
        )
    
    async def update_knowledge(
        self,
        chunk_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Update knowledge chunk.
        
        Args:
            chunk_id: Chunk identifier
            content: Updated content (optional)
            metadata: Updated metadata (optional)
        """
        embedding = None
        if content:
            embedding = await self.embedding_client.embed_text(content)
        
        await self.vector_db.update_chunk(
            chunk_id=chunk_id,
            content=content,
            embedding=embedding,
            metadata=metadata,
        )
    
    async def delete_knowledge(self, chunk_id: str) -> None:
        """Delete knowledge chunk."""
        await self.vector_db.delete_chunk(chunk_id)
    
    def _apply_filters(
        self,
        chunks: List[KnowledgeChunk],
        filters: Dict[str, Any],
    ) -> List[KnowledgeChunk]:
        """Apply metadata filters to search results."""
        filtered = []
        
        for chunk in chunks:
            match = True
            for key, value in filters.items():
                if key not in chunk.metadata:
                    match = False
                    break
                
                # Handle list values (e.g., crop_type: ["wheat", "rice"])
                if isinstance(value, list):
                    if chunk.metadata[key] not in value:
                        match = False
                        break
                else:
                    if chunk.metadata[key] != value:
                        match = False
                        break
            
            if match:
                filtered.append(chunk)
        
        return filtered
    
    def format_context(self, chunks: List[KnowledgeChunk]) -> str:
        """
        Format knowledge chunks for LLM context injection.
        
        Args:
            chunks: Knowledge chunks to format
            
        Returns:
            Formatted context string
        """
        if not chunks:
            return ""
        
        context_parts = ["Retrieved Knowledge Context:"]
        
        for i, chunk in enumerate(chunks, 1):
            source = chunk.metadata.get("source", "unknown")
            similarity = chunk.similarity_score
            
            context_parts.append(
                f"\n[{i}] (Source: {source}, Relevance: {similarity:.2f})"
            )
            context_parts.append(chunk.content)
        
        return "\n".join(context_parts)
