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
    
    def __init__(
        self,
        endpoint: str,
        index_name: str = "grambrain-knowledge",
        use_ssl: bool = True,
        verify_certs: bool = True,
        timeout: int = 30,
        embedding_dimension: int = 1536,
        aws_region: str = "us-east-1",
    ):
        """
        Initialize OpenSearch client.
        
        Args:
            endpoint: OpenSearch endpoint URL
            index_name: Index name for knowledge chunks
            use_ssl: Use SSL for connections
            verify_certs: Verify SSL certificates
            timeout: Request timeout in seconds
            embedding_dimension: Dimension of embedding vectors
            aws_region: AWS region for authentication
        """
        self.endpoint = endpoint
        self.index_name = index_name
        self.embedding_dimension = embedding_dimension
        self.client = None
        self._initialized = False
        
        # Initialize OpenSearch client
        try:
            from opensearchpy import OpenSearch, RequestsHttpConnection
            from requests_aws4auth import AWS4Auth
            import boto3
            
            # Get AWS credentials for authentication
            credentials = boto3.Session().get_credentials()
            awsauth = AWS4Auth(
                credentials.access_key,
                credentials.secret_key,
                aws_region,
                'es',
                session_token=credentials.token
            )
            
            # Parse endpoint to get host and port
            from urllib.parse import urlparse
            parsed = urlparse(endpoint if endpoint.startswith('http') else f'https://{endpoint}')
            host = parsed.hostname or endpoint
            port = parsed.port or 443
            
            self.client = OpenSearch(
                hosts=[{'host': host, 'port': port}],
                http_auth=awsauth,
                use_ssl=use_ssl,
                verify_certs=verify_certs,
                connection_class=RequestsHttpConnection,
                timeout=timeout,
            )
            self._initialized = True
        except ImportError:
            # OpenSearch library not available, will use fallback
            self._initialized = False
        except Exception as e:
            # Connection failed, will use fallback
            print(f"Warning: Failed to initialize OpenSearch client: {e}")
            self._initialized = False
    
    async def ensure_index_exists(self) -> None:
        """Create index with knn_vector mapping if it doesn't exist."""
        if not self._initialized or not self.client:
            return
        
        try:
            # Check if index exists
            if self.client.indices.exists(index=self.index_name):
                return
            
            # Create index with knn_vector mapping
            index_body = {
                "settings": {
                    "index": {
                        "knn": True,
                        "knn.algo_param.ef_search": 512,
                    },
                    "number_of_shards": 3,
                    "number_of_replicas": 1,
                },
                "mappings": {
                    "properties": {
                        "chunk_id": {
                            "type": "keyword"
                        },
                        "content": {
                            "type": "text",
                            "analyzer": "standard"
                        },
                        "embedding": {
                            "type": "knn_vector",
                            "dimension": self.embedding_dimension,
                            "method": {
                                "name": "hnsw",
                                "space_type": "cosinesimil",
                                "engine": "nmslib",
                                "parameters": {
                                    "ef_construction": 512,
                                    "m": 16
                                }
                            }
                        },
                        "metadata": {
                            "properties": {
                                "source": {"type": "keyword"},
                                "topic": {"type": "keyword"},
                                "crop_type": {"type": "keyword"},
                                "region": {"type": "keyword"},
                                "created_at": {"type": "date"}
                            }
                        }
                    }
                }
            }
            
            self.client.indices.create(index=self.index_name, body=index_body)
        except Exception as e:
            print(f"Warning: Failed to create OpenSearch index: {e}")
    
    async def add_chunk(
        self,
        chunk_id: str,
        content: str,
        embedding: List[float],
        metadata: Dict[str, Any],
    ) -> None:
        """Add chunk to OpenSearch."""
        if not self._initialized or not self.client:
            raise RuntimeError("OpenSearch client not initialized")
        
        await self.ensure_index_exists()
        
        try:
            document = {
                "chunk_id": chunk_id,
                "content": content,
                "embedding": embedding,
                "metadata": metadata,
            }
            
            self.client.index(
                index=self.index_name,
                id=chunk_id,
                body=document,
                refresh=True,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to add chunk to OpenSearch: {e}")
    
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        min_similarity: float = 0.7,
        metadata_filters: Optional[Dict[str, Any]] = None,
    ) -> List[KnowledgeChunk]:
        """
        Search OpenSearch using knn_vector similarity.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold (0-1)
            metadata_filters: Optional metadata filters
            
        Returns:
            List of matching knowledge chunks
        """
        if not self._initialized or not self.client:
            # Fallback to empty results if not initialized
            return []
        
        try:
            # Build knn query
            knn_query = {
                "size": top_k,
                "query": {
                    "knn": {
                        "embedding": {
                            "vector": query_embedding,
                            "k": top_k,
                        }
                    }
                }
            }
            
            # Add metadata filters if provided
            if metadata_filters:
                filter_clauses = []
                for key, value in metadata_filters.items():
                    if isinstance(value, list):
                        filter_clauses.append({
                            "terms": {f"metadata.{key}": value}
                        })
                    else:
                        filter_clauses.append({
                            "term": {f"metadata.{key}": value}
                        })
                
                if filter_clauses:
                    knn_query["query"] = {
                        "bool": {
                            "must": [knn_query["query"]],
                            "filter": filter_clauses
                        }
                    }
            
            # Execute search
            response = self.client.search(
                index=self.index_name,
                body=knn_query
            )
            
            # Parse results
            results = []
            for hit in response['hits']['hits']:
                score = hit['_score']
                
                # Convert OpenSearch score to similarity (0-1 range)
                # OpenSearch knn returns scores that need normalization
                similarity = self._normalize_score(score)
                
                if similarity >= min_similarity:
                    source = hit['_source']
                    chunk = KnowledgeChunk(
                        chunk_id=source['chunk_id'],
                        content=source['content'],
                        embedding=source['embedding'],
                        metadata=source.get('metadata', {}),
                        similarity_score=similarity,
                    )
                    results.append(chunk)
            
            return results
            
        except Exception as e:
            print(f"Warning: OpenSearch search failed: {e}")
            # Return empty results on failure (graceful degradation)
            return []
    
    async def delete_chunk(self, chunk_id: str) -> None:
        """Delete chunk from OpenSearch."""
        if not self._initialized or not self.client:
            raise RuntimeError("OpenSearch client not initialized")
        
        try:
            self.client.delete(
                index=self.index_name,
                id=chunk_id,
                refresh=True,
            )
        except Exception as e:
            # Ignore if document doesn't exist
            if "not_found" not in str(e).lower():
                raise RuntimeError(f"Failed to delete chunk from OpenSearch: {e}")
    
    async def update_chunk(
        self,
        chunk_id: str,
        content: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update chunk in OpenSearch."""
        if not self._initialized or not self.client:
            raise RuntimeError("OpenSearch client not initialized")
        
        try:
            # Build update document
            update_doc = {}
            if content is not None:
                update_doc['content'] = content
            if embedding is not None:
                update_doc['embedding'] = embedding
            if metadata is not None:
                update_doc['metadata'] = metadata
            
            if update_doc:
                self.client.update(
                    index=self.index_name,
                    id=chunk_id,
                    body={"doc": update_doc},
                    refresh=True,
                )
        except Exception as e:
            raise RuntimeError(f"Failed to update chunk in OpenSearch: {e}")
    
    @staticmethod
    def _normalize_score(score: float) -> float:
        """
        Normalize OpenSearch knn score to 0-1 similarity range.
        
        OpenSearch knn with cosinesimil returns scores in range [0, 2],
        where 2 is most similar. We normalize to [0, 1].
        """
        # For cosine similarity, OpenSearch returns 1 + cosine_sim
        # So score ranges from 0 (opposite) to 2 (identical)
        # Normalize to 0-1 range
        return min(max(score / 2.0, 0.0), 1.0)
    
    def is_available(self) -> bool:
        """Check if OpenSearch is available and healthy."""
        if not self._initialized or not self.client:
            return False
        
        try:
            health = self.client.cluster.health()
            return health['status'] in ['green', 'yellow']
        except Exception:
            return False



def create_vector_db(
    db_type: str,
    opensearch_endpoint: Optional[str] = None,
    opensearch_index_name: str = "grambrain-knowledge",
    opensearch_use_ssl: bool = True,
    opensearch_verify_certs: bool = True,
    opensearch_timeout: int = 30,
    embedding_dimension: int = 1536,
    aws_region: str = "us-east-1",
) -> VectorDB:
    """
    Factory function to create vector database instance.
    
    Args:
        db_type: Type of vector database ('in_memory' or 'opensearch')
        opensearch_endpoint: OpenSearch endpoint (required if db_type='opensearch')
        opensearch_index_name: Index name for OpenSearch
        opensearch_use_ssl: Use SSL for OpenSearch connections
        opensearch_verify_certs: Verify SSL certificates
        opensearch_timeout: Request timeout in seconds
        embedding_dimension: Dimension of embedding vectors
        aws_region: AWS region for authentication
        
    Returns:
        VectorDB instance
        
    Raises:
        ValueError: If invalid db_type or missing required parameters
    """
    if db_type == "in_memory":
        return InMemoryVectorDB()
    
    elif db_type == "opensearch":
        if not opensearch_endpoint:
            raise ValueError("opensearch_endpoint is required for OpenSearch vector DB")
        
        return OpenSearchVectorDB(
            endpoint=opensearch_endpoint,
            index_name=opensearch_index_name,
            use_ssl=opensearch_use_ssl,
            verify_certs=opensearch_verify_certs,
            timeout=opensearch_timeout,
            embedding_dimension=embedding_dimension,
            aws_region=aws_region,
        )
    
    else:
        raise ValueError(f"Unsupported vector database type: {db_type}")
