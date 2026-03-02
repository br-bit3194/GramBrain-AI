"""Factory for creating configured DynamoDB clients and repositories."""

from typing import Optional
import os
from dotenv import load_dotenv

from .dynamodb_client import DynamoDBClient, RetryConfig
from .repositories import (
    UserRepository,
    FarmRepository,
    RecommendationRepository,
    ProductRepository,
    KnowledgeRepository,
)

# Load environment variables
load_dotenv()


def create_dynamodb_client(
    region_name: Optional[str] = None,
    endpoint_url: Optional[str] = None,
    retry_config: Optional[RetryConfig] = None
) -> DynamoDBClient:
    """Create a configured DynamoDB client.
    
    Args:
        region_name: AWS region (defaults to AWS_REGION env var)
        endpoint_url: Custom endpoint URL (defaults to DYNAMODB_ENDPOINT_URL env var)
        retry_config: Retry configuration (defaults to env vars)
        
    Returns:
        Configured DynamoDB client
        
    Examples:
        # Production (AWS DynamoDB)
        client = create_dynamodb_client()
        
        # Local development (DynamoDB Local)
        client = create_dynamodb_client(endpoint_url='http://localhost:8000')
        
        # LocalStack
        client = create_dynamodb_client(endpoint_url='http://localhost:4566')
    """
    if region_name is None:
        region_name = os.getenv('AWS_REGION', 'us-east-1')
    
    if endpoint_url is None:
        endpoint_url = os.getenv('DYNAMODB_ENDPOINT_URL') or None
    
    if retry_config is None:
        retry_config = RetryConfig(
            max_attempts=int(os.getenv('DYNAMODB_RETRY_MAX_ATTEMPTS', '3')),
            base_delay_ms=int(os.getenv('DYNAMODB_RETRY_BASE_DELAY_MS', '100')),
            max_delay_ms=int(os.getenv('DYNAMODB_RETRY_MAX_DELAY_MS', '5000')),
            exponential_base=2.0,
            jitter=True
        )
    
    return DynamoDBClient(
        region_name=region_name,
        endpoint_url=endpoint_url,
        retry_config=retry_config
    )


def create_user_repository(
    client: Optional[DynamoDBClient] = None,
    env: Optional[str] = None
) -> UserRepository:
    """Create a UserRepository.
    
    Args:
        client: DynamoDB client (creates default if None)
        env: Environment name (defaults to DYNAMODB_ENV env var)
        
    Returns:
        UserRepository instance
    """
    if client is None:
        client = create_dynamodb_client()
    
    if env is None:
        env = os.getenv('DYNAMODB_ENV', 'dev')
    
    return UserRepository(client, env=env)


def create_farm_repository(
    client: Optional[DynamoDBClient] = None,
    env: Optional[str] = None
) -> FarmRepository:
    """Create a FarmRepository.
    
    Args:
        client: DynamoDB client (creates default if None)
        env: Environment name (defaults to DYNAMODB_ENV env var)
        
    Returns:
        FarmRepository instance
    """
    if client is None:
        client = create_dynamodb_client()
    
    if env is None:
        env = os.getenv('DYNAMODB_ENV', 'dev')
    
    return FarmRepository(client, env=env)


def create_recommendation_repository(
    client: Optional[DynamoDBClient] = None,
    env: Optional[str] = None
) -> RecommendationRepository:
    """Create a RecommendationRepository.
    
    Args:
        client: DynamoDB client (creates default if None)
        env: Environment name (defaults to DYNAMODB_ENV env var)
        
    Returns:
        RecommendationRepository instance
    """
    if client is None:
        client = create_dynamodb_client()
    
    if env is None:
        env = os.getenv('DYNAMODB_ENV', 'dev')
    
    return RecommendationRepository(client, env=env)


def create_product_repository(
    client: Optional[DynamoDBClient] = None,
    env: Optional[str] = None
) -> ProductRepository:
    """Create a ProductRepository.
    
    Args:
        client: DynamoDB client (creates default if None)
        env: Environment name (defaults to DYNAMODB_ENV env var)
        
    Returns:
        ProductRepository instance
    """
    if client is None:
        client = create_dynamodb_client()
    
    if env is None:
        env = os.getenv('DYNAMODB_ENV', 'dev')
    
    return ProductRepository(client, env=env)


def create_knowledge_repository(
    client: Optional[DynamoDBClient] = None,
    env: Optional[str] = None
) -> KnowledgeRepository:
    """Create a KnowledgeRepository.
    
    Args:
        client: DynamoDB client (creates default if None)
        env: Environment name (defaults to DYNAMODB_ENV env var)
        
    Returns:
        KnowledgeRepository instance
    """
    if client is None:
        client = create_dynamodb_client()
    
    if env is None:
        env = os.getenv('DYNAMODB_ENV', 'dev')
    
    return KnowledgeRepository(client, env=env)


# Singleton instances (lazy-loaded)
_client_instance: Optional[DynamoDBClient] = None
_user_repo_instance: Optional[UserRepository] = None
_farm_repo_instance: Optional[FarmRepository] = None
_recommendation_repo_instance: Optional[RecommendationRepository] = None
_product_repo_instance: Optional[ProductRepository] = None
_knowledge_repo_instance: Optional[KnowledgeRepository] = None


def get_dynamodb_client() -> DynamoDBClient:
    """Get singleton DynamoDB client instance."""
    global _client_instance
    if _client_instance is None:
        _client_instance = create_dynamodb_client()
    return _client_instance


def get_user_repository() -> UserRepository:
    """Get singleton UserRepository instance."""
    global _user_repo_instance
    if _user_repo_instance is None:
        _user_repo_instance = create_user_repository()
    return _user_repo_instance


def get_farm_repository() -> FarmRepository:
    """Get singleton FarmRepository instance."""
    global _farm_repo_instance
    if _farm_repo_instance is None:
        _farm_repo_instance = create_farm_repository()
    return _farm_repo_instance


def get_recommendation_repository() -> RecommendationRepository:
    """Get singleton RecommendationRepository instance."""
    global _recommendation_repo_instance
    if _recommendation_repo_instance is None:
        _recommendation_repo_instance = create_recommendation_repository()
    return _recommendation_repo_instance


def get_product_repository() -> ProductRepository:
    """Get singleton ProductRepository instance."""
    global _product_repo_instance
    if _product_repo_instance is None:
        _product_repo_instance = create_product_repository()
    return _product_repo_instance


def get_knowledge_repository() -> KnowledgeRepository:
    """Get singleton KnowledgeRepository instance."""
    global _knowledge_repo_instance
    if _knowledge_repo_instance is None:
        _knowledge_repo_instance = create_knowledge_repository()
    return _knowledge_repo_instance
