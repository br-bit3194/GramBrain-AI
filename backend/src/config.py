"""Configuration management for GramBrain AI."""

import os
from typing import Optional
from dotenv import load_dotenv
from dataclasses import dataclass

# Load environment variables
load_dotenv()


@dataclass
class AWSConfig:
    """AWS configuration."""
    region: str
    access_key_id: Optional[str]
    secret_access_key: Optional[str]
    
    @classmethod
    def from_env(cls) -> 'AWSConfig':
        """Load AWS config from environment variables."""
        return cls(
            region=os.getenv('AWS_REGION', 'us-east-1'),
            access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        )


@dataclass
class DynamoDBConfig:
    """DynamoDB configuration."""
    env: str
    retry_max_attempts: int
    retry_base_delay_ms: int
    retry_max_delay_ms: int
    
    @classmethod
    def from_env(cls) -> 'DynamoDBConfig':
        """Load DynamoDB config from environment variables."""
        return cls(
            env=os.getenv('DYNAMODB_ENV', 'dev'),
            retry_max_attempts=int(os.getenv('DYNAMODB_RETRY_MAX_ATTEMPTS', '3')),
            retry_base_delay_ms=int(os.getenv('DYNAMODB_RETRY_BASE_DELAY_MS', '100')),
            retry_max_delay_ms=int(os.getenv('DYNAMODB_RETRY_MAX_DELAY_MS', '5000')),
        )


@dataclass
class LLMConfig:
    """LLM configuration."""
    default_model: str
    temperature: float
    max_tokens: int
    
    @classmethod
    def from_env(cls) -> 'LLMConfig':
        """Load LLM config from environment variables."""
        return cls(
            default_model=os.getenv('DEFAULT_LLM_MODEL', 'anthropic.claude-3-sonnet-20240229-v1:0'),
            temperature=float(os.getenv('LLM_TEMPERATURE', '0.7')),
            max_tokens=int(os.getenv('LLM_MAX_TOKENS', '1000')),
        )


@dataclass
class S3Config:
    """S3 configuration."""
    bucket_name: str
    max_file_size_mb: int
    presigned_url_expiration: int
    
    @classmethod
    def from_env(cls) -> 'S3Config':
        """Load S3 config from environment variables."""
        return cls(
            bucket_name=os.getenv('S3_BUCKET_NAME', 'gram-brain-bucket'),
            max_file_size_mb=int(os.getenv('S3_MAX_FILE_SIZE_MB', '50')),
            presigned_url_expiration=int(os.getenv('S3_PRESIGNED_URL_EXPIRATION', '3600')),
        )


@dataclass
class RAGConfig:
    """RAG configuration."""
    vector_db_type: str
    opensearch_endpoint: Optional[str]
    opensearch_index_name: str
    opensearch_use_ssl: bool
    opensearch_verify_certs: bool
    opensearch_timeout: int
    embedding_model: str
    embedding_dimension: int
    cache_ttl_hours: int
    
    @classmethod
    def from_env(cls) -> 'RAGConfig':
        """Load RAG config from environment variables."""
        return cls(
            vector_db_type=os.getenv('VECTOR_DB_TYPE', 'in_memory'),
            opensearch_endpoint=os.getenv('OPENSEARCH_ENDPOINT'),
            opensearch_index_name=os.getenv('OPENSEARCH_INDEX_NAME', 'grambrain-knowledge'),
            opensearch_use_ssl=os.getenv('OPENSEARCH_USE_SSL', 'true').lower() == 'true',
            opensearch_verify_certs=os.getenv('OPENSEARCH_VERIFY_CERTS', 'true').lower() == 'true',
            opensearch_timeout=int(os.getenv('OPENSEARCH_TIMEOUT', '30')),
            embedding_model=os.getenv('EMBEDDING_MODEL', 'amazon.titan-embed-text-v1'),
            embedding_dimension=int(os.getenv('EMBEDDING_DIMENSION', '1536')),
            cache_ttl_hours=int(os.getenv('RAG_CACHE_TTL_HOURS', '24')),
        )


@dataclass
class SystemConfig:
    """System configuration."""
    log_level: str
    agent_timeout: int
    
    @classmethod
    def from_env(cls) -> 'SystemConfig':
        """Load system config from environment variables."""
        return cls(
            log_level=os.getenv('LOG_LEVEL', 'INFO'),
            agent_timeout=int(os.getenv('AGENT_TIMEOUT', '10')),
        )


@dataclass
class ExternalAPIConfig:
    """External API configuration."""
    openweather_api_key: Optional[str]
    imd_api_key: Optional[str]
    sentinel_api_key: Optional[str]
    agmarknet_api_key: Optional[str]
    weather_cache_ttl_hours: int
    satellite_cache_ttl_hours: int
    
    @classmethod
    def from_env(cls) -> 'ExternalAPIConfig':
        """Load external API config from environment variables."""
        return cls(
            openweather_api_key=os.getenv('OPENWEATHER_API_KEY'),
            imd_api_key=os.getenv('IMD_API_KEY'),
            sentinel_api_key=os.getenv('SENTINEL_API_KEY'),
            agmarknet_api_key=os.getenv('AGMARKNET_API_KEY'),
            weather_cache_ttl_hours=int(os.getenv('WEATHER_CACHE_TTL_HOURS', '3')),
            satellite_cache_ttl_hours=int(os.getenv('SATELLITE_CACHE_TTL_HOURS', '24')),
        )


@dataclass
class Config:
    """Application configuration."""
    aws: AWSConfig
    dynamodb: DynamoDBConfig
    llm: LLMConfig
    s3: S3Config
    rag: RAGConfig
    system: SystemConfig
    external_api: ExternalAPIConfig
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Load all configuration from environment variables."""
        return cls(
            aws=AWSConfig.from_env(),
            dynamodb=DynamoDBConfig.from_env(),
            llm=LLMConfig.from_env(),
            s3=S3Config.from_env(),
            rag=RAGConfig.from_env(),
            system=SystemConfig.from_env(),
            external_api=ExternalAPIConfig.from_env(),
        )


# Global config instance
config = Config.from_env()
