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
class RAGConfig:
    """RAG configuration."""
    vector_db_type: str
    opensearch_endpoint: Optional[str]
    embedding_model: str
    
    @classmethod
    def from_env(cls) -> 'RAGConfig':
        """Load RAG config from environment variables."""
        return cls(
            vector_db_type=os.getenv('VECTOR_DB_TYPE', 'in_memory'),
            opensearch_endpoint=os.getenv('OPENSEARCH_ENDPOINT'),
            embedding_model=os.getenv('EMBEDDING_MODEL', 'amazon.titan-embed-text-v1'),
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
class Config:
    """Application configuration."""
    aws: AWSConfig
    dynamodb: DynamoDBConfig
    llm: LLMConfig
    rag: RAGConfig
    system: SystemConfig
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Load all configuration from environment variables."""
        return cls(
            aws=AWSConfig.from_env(),
            dynamodb=DynamoDBConfig.from_env(),
            llm=LLMConfig.from_env(),
            rag=RAGConfig.from_env(),
            system=SystemConfig.from_env(),
        )


# Global config instance
config = Config.from_env()
