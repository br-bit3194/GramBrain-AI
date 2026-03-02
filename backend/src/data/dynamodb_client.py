"""DynamoDB client with retry logic and error handling."""

import asyncio
import random
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import boto3
from botocore.exceptions import ClientError
import logging

logger = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay_ms: int = 100
    max_delay_ms: int = 5000
    exponential_base: float = 2.0
    jitter: bool = True


class DynamoDBClient:
    """DynamoDB client with exponential backoff retry logic."""
    
    # Retryable error codes
    RETRYABLE_ERRORS = {
        'ProvisionedThroughputExceededException',
        'ThrottlingException',
        'RequestLimitExceeded',
        'InternalServerError',
        'ServiceUnavailable',
    }
    
    def __init__(
        self,
        region_name: str = 'us-east-1',
        endpoint_url: Optional[str] = None,
        retry_config: Optional[RetryConfig] = None
    ):
        """Initialize DynamoDB client.
        
        Args:
            region_name: AWS region name
            endpoint_url: Custom endpoint URL (for LocalStack/DynamoDB Local)
            retry_config: Retry configuration
        """
        # Create boto3 client with optional endpoint URL
        client_kwargs = {'region_name': region_name}
        if endpoint_url:
            client_kwargs['endpoint_url'] = endpoint_url
        
        self.dynamodb = boto3.resource('dynamodb', **client_kwargs)
        self.client = boto3.client('dynamodb', **client_kwargs)
        self.retry_config = retry_config or RetryConfig()
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and jitter.
        
        Args:
            attempt: Current attempt number (0-indexed)
            
        Returns:
            Delay in seconds
        """
        delay_ms = min(
            self.retry_config.base_delay_ms * (self.retry_config.exponential_base ** attempt),
            self.retry_config.max_delay_ms
        )
        
        if self.retry_config.jitter:
            delay_ms = delay_ms * (0.5 + random.random() * 0.5)
        
        return delay_ms / 1000.0  # Convert to seconds
    
    def _is_retryable_error(self, error: ClientError) -> bool:
        """Check if error is retryable.
        
        Args:
            error: Boto3 ClientError
            
        Returns:
            True if error should be retried
        """
        error_code = error.response.get('Error', {}).get('Code', '')
        return error_code in self.RETRYABLE_ERRORS
    
    async def _execute_with_retry(self, operation, *args, **kwargs) -> Any:
        """Execute operation with exponential backoff retry.
        
        Args:
            operation: Function to execute
            *args: Positional arguments for operation
            **kwargs: Keyword arguments for operation
            
        Returns:
            Operation result
            
        Raises:
            ClientError: If all retry attempts fail
        """
        last_error = None
        
        for attempt in range(self.retry_config.max_attempts):
            try:
                # Execute operation in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: operation(*args, **kwargs)
                )
                return result
                
            except ClientError as e:
                last_error = e
                
                if not self._is_retryable_error(e):
                    logger.error(f"Non-retryable error: {e}")
                    raise
                
                if attempt < self.retry_config.max_attempts - 1:
                    delay = self._calculate_delay(attempt)
                    logger.warning(
                        f"Retryable error on attempt {attempt + 1}/{self.retry_config.max_attempts}: {e}. "
                        f"Retrying in {delay:.2f}s"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All retry attempts exhausted: {e}")
        
        raise last_error
    
    def get_table(self, table_name: str):
        """Get DynamoDB table resource.
        
        Args:
            table_name: Name of the table
            
        Returns:
            DynamoDB table resource
        """
        return self.dynamodb.Table(table_name)
