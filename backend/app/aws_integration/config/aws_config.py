# app/aws_integration/config/aws_config.py
from typing import Dict, Any
from pydantic_settings import BaseSettings, SettingsConfigDict


class AWSConfig(BaseSettings):
    """AWS Configuration loaded from .env file"""
    
    # AWS Credentials
    aws_region: str = "us-east-1"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    
    # Bedrock Configuration
    bedrock_model_id: str = "amazon.nova-lite-v1:0"
    bedrock_temperature: float = 0.7
    bedrock_max_tokens: int = 4096
    
    # DynamoDB Configuration
    dynamodb_market_prices_table: str = "grambrain_market_prices"
    dynamodb_sessions_table: str = "grambrain_sessions"
    dynamodb_analytics_table: str = "grambrain_analytics"
    
    # Other Services
    weather_api_key: str = ""
    elevenlabs_api_key: str = ""
    mandi_api_key: str = ""
    
    model_config = SettingsConfigDict(
        env_file="backend/.env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


# Global config instance
aws_config = AWSConfig()


def get_aws_credentials() -> Dict[str, str]:
    """Get AWS credentials for boto3"""
    return {
        "region_name": aws_config.aws_region,
        "aws_access_key_id": aws_config.aws_access_key_id,
        "aws_secret_access_key": aws_config.aws_secret_access_key
    }


def get_bedrock_config() -> Dict[str, Any]:
    """Get Bedrock configuration"""
    return {
        "model_id": aws_config.bedrock_model_id,
        "temperature": aws_config.bedrock_temperature,
        "max_tokens": aws_config.bedrock_max_tokens
    }
