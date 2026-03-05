# app/integration/config/aws_config.py
from typing import Dict, Any
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
import os
from dotenv import load_dotenv


# Find and load .env file
def find_and_load_env() -> str:
    """Find and load the .env file in various possible locations"""
    possible_paths = [
        Path("backend/.env"),  # From project root
    ]
    
    for path in possible_paths:
        if path.exists():
            load_dotenv(path)
            return str(path)
    
    # Try to load from backend/.env anyway
    load_dotenv("backend/.env")
    return "backend/.env"


# Load environment variables
env_file_path = find_and_load_env()

# Force override system environment variables with .env values
# Remove AWS_SESSION_TOKEN if it exists (from temporary credentials)
if "AWS_SESSION_TOKEN" in os.environ:
    del os.environ["AWS_SESSION_TOKEN"]

class AWSConfig(BaseSettings):
    """AWS Configuration loaded from .env file"""
    
    # AWS Credentials
    aws_region: str = "us-east-1"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    
    # Bedrock Configuration
    bedrock_model_id: str = "us.amazon.nova-lite-v1:0"
    bedrock_vision_model_id: str = "us.amazon.nova-pro-v1:0"
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
        case_sensitive=False,
        extra="ignore"
    )


# Global config instance
aws_config = AWSConfig()

# Also set them as environment variables to ensure boto3 picks them up
if os.getenv("AWS_ACCESS_KEY_ID"):
    os.environ["AWS_ACCESS_KEY_ID"] = aws_config.aws_access_key_id
if os.getenv("AWS_SECRET_ACCESS_KEY"):
    os.environ["AWS_SECRET_ACCESS_KEY"] = aws_config.aws_secret_access_key
if os.getenv("AWS_REGION"):
    os.environ["AWS_DEFAULT_REGION"] = aws_config.aws_region

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
        "vision_model_id": aws_config.bedrock_vision_model_id,
        "temperature": aws_config.bedrock_temperature,
        "max_tokens": aws_config.bedrock_max_tokens
    }
