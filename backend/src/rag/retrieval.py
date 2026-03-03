"""RAG retrieval pipeline for knowledge-grounded recommendations."""

from typing import List, Dict, Any, Optional
import time
from .vector_db import VectorDB, KnowledgeChunk
from .embeddings import EmbeddingClient


class CachedRAGClient:
    """RAG client with caching support for fallback when vector DB is unavailable."""
    
    def __init__(self, vector_db: VectorDB, embedding_client: EmbeddingClient, cache_ttl_hours: int = 24):
        """
        Initialize cached RAG client.
        
        Args:
            vector_db: Vector database instance
            embedding_client: Embedding generation client
            cache_ttl_hours: Cache TTL in hours
        """
        self.vector_db = vector_db
        self.embedding_client = embedding_client
        self.cache_ttl_seconds = cache_ttl_hours * 3600
        self._search_cache: Dict[str, tuple[List[KnowledgeChunk], float]] = {}
    
    async def search(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.7,
        filters: Optional[Dict[str, Any]] = None,
        use_cache_fallback: bool = True,
    ) -> List[KnowledgeChunk]:
        """
        Search for relevant knowledge chunks with caching fallback.
        
        Args:
            query: Search query
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold
            filters: Optional metadata filters (e.g., {"crop_type": "wheat", "region": "punjab"})
            use_cache_fallback: Use cached results if vector DB is unavailable
            
        Returns:
            List of relevant knowledge chunks
        """
        # Create cache key
        cache_key = self._make_cache_key(query, top_k, min_similarity, filters)
        
        try:
            # Generate embedding for query
            query_embedding = await self.embedding_client.embed_text(query)
            
            # Search vector database with metadata filters
            # For OpenSearch, pass filters directly; for InMemory, apply post-search
            if hasattr(self.vector_db, 'search') and 'metadata_filters' in self.vector_db.search.__code__.co_varnames:
                # OpenSearch supports native metadata filtering
                results = await self.vector_db.search(
                    query_embedding=query_embedding,
                    top_k=top_k,
                    min_similarity=min_similarity,
                    metadata_filters=filters,
                )
            else:
                # InMemory DB - apply filters after search
                results = await self.vector_db.search(
                    query_embedding=query_embedding,
                    top_k=top_k * 2 if filters else top_k,  # Get more results if filtering
                    min_similarity=min_similarity,
                )
                
                # Apply metadata filters if provided
                if filters:
                    results = self._apply_filters(results, filters)
                    results = results[:top_k]  # Trim to top_k after filtering
            
            # Cache successful results
            self._search_cache[cache_key] = (results, time.time())
            
            return results
            
        except Exception as e:
            # Vector DB unavailable - try cache fallback
            if use_cache_fallback and cache_key in self._search_cache:
                cached_results, cached_time = self._search_cache[cache_key]
                
                # Check if cache is still valid
                if time.time() - cached_time < self.cache_ttl_seconds:
                    print(f"Warning: Vector DB unavailable, using cached results: {e}")
                    return cached_results
            
            # No cache available or cache expired
            print(f"Warning: Vector DB search failed and no cache available: {e}")
            return []
    
    def _make_cache_key(
        self,
        query: str,
        top_k: int,
        min_similarity: float,
        filters: Optional[Dict[str, Any]],
    ) -> str:
        """Create cache key from search parameters."""
        filter_str = str(sorted(filters.items())) if filters else ""
        return f"{query}|{top_k}|{min_similarity}|{filter_str}"
    
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
        
        # Invalidate cache
        self._search_cache.clear()
    
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
        
        # Invalidate cache
        self._search_cache.clear()
    
    async def delete_knowledge(self, chunk_id: str) -> None:
        """Delete knowledge chunk."""
        await self.vector_db.delete_chunk(chunk_id)
        
        # Invalidate cache
        self._search_cache.clear()
    
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


# Backward compatibility alias
RAGClient = CachedRAGClient
