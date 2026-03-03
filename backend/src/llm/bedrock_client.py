"""AWS Bedrock LLM client integration."""

import json
import asyncio
import time
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum
import boto3
from botocore.exceptions import ClientError
from pydantic import BaseModel, Field, ValidationError


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreaker:
    """Circuit breaker for fault tolerance."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout_seconds: int = 60,
        half_open_max_calls: int = 3
    ):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.half_open_max_calls = half_open_max_calls
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = CircuitBreakerState.CLOSED
        self.half_open_calls = 0
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        if self.last_failure_time is None:
            return False
        return time.time() - self.last_failure_time >= self.timeout_seconds
    
    def _on_success(self):
        """Handle successful call."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.CLOSED
            self.failure_count = 0
            self.half_open_calls = 0
        elif self.state == CircuitBreakerState.CLOSED:
            self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
            self.half_open_calls = 0
        elif self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN
    
    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
                self.half_open_calls = 0
            else:
                raise RuntimeError("Circuit breaker is OPEN")
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            if self.half_open_calls >= self.half_open_max_calls:
                raise RuntimeError("Circuit breaker HALF_OPEN max calls exceeded")
            self.half_open_calls += 1
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e


class TokenUsage(BaseModel):
    """Token usage tracking."""
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    model_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    cost_estimate: Optional[float] = None


class BedrockResponse(BaseModel):
    """Validated Bedrock response."""
    text: str
    model_id: str
    token_usage: TokenUsage
    finish_reason: Optional[str] = None


class BedrockClient:
    """Client for AWS Bedrock LLM inference with production features."""
    
    # Model IDs for different providers
    CLAUDE_3_SONNET = "anthropic.claude-3-sonnet-20240229-v1:0"
    CLAUDE_3_HAIKU = "anthropic.claude-3-haiku-20240307-v1:0"
    TITAN_TEXT = "amazon.titan-text-express-v1"
    LLAMA_2_70B = "meta.llama2-70b-chat-v1"
    
    # Cost per 1000 tokens (approximate, in USD)
    MODEL_COSTS = {
        CLAUDE_3_SONNET: {"input": 0.003, "output": 0.015},
        CLAUDE_3_HAIKU: {"input": 0.00025, "output": 0.00125},
        TITAN_TEXT: {"input": 0.0008, "output": 0.0016},
        LLAMA_2_70B: {"input": 0.00195, "output": 0.00256},
    }
    
    def __init__(
        self,
        region: str = "us-east-1",
        default_model: str = CLAUDE_3_SONNET,
        max_retries: int = 3,
        enable_circuit_breaker: bool = True,
        track_usage: bool = True,
    ):
        """
        Initialize Bedrock client.
        
        Args:
            region: AWS region
            default_model: Default model to use
            max_retries: Number of retries on failure
            enable_circuit_breaker: Enable circuit breaker pattern
            track_usage: Enable token usage tracking
        """
        self.region = region
        self.default_model = default_model
        self.max_retries = max_retries
        self.track_usage = track_usage
        self.client = boto3.client("bedrock-runtime", region_name=region)
        
        # Circuit breaker for fault tolerance
        self.circuit_breaker = CircuitBreaker() if enable_circuit_breaker else None
        
        # Token usage tracking
        self.usage_history: List[TokenUsage] = []
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get token usage statistics."""
        if not self.usage_history:
            return {
                "total_calls": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_cost": 0.0,
                "by_model": {}
            }
        
        total_input = sum(u.input_tokens for u in self.usage_history)
        total_output = sum(u.output_tokens for u in self.usage_history)
        total_cost = sum(u.cost_estimate or 0.0 for u in self.usage_history)
        
        by_model = {}
        for usage in self.usage_history:
            if usage.model_id not in by_model:
                by_model[usage.model_id] = {
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost": 0.0
                }
            by_model[usage.model_id]["calls"] += 1
            by_model[usage.model_id]["input_tokens"] += usage.input_tokens
            by_model[usage.model_id]["output_tokens"] += usage.output_tokens
            by_model[usage.model_id]["cost"] += usage.cost_estimate or 0.0
        
        return {
            "total_calls": len(self.usage_history),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_cost": total_cost,
            "by_model": by_model
        }
    
    def _calculate_cost(self, model_id: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost estimate for token usage."""
        if model_id not in self.MODEL_COSTS:
            return 0.0
        
        costs = self.MODEL_COSTS[model_id]
        input_cost = (input_tokens / 1000) * costs["input"]
        output_cost = (output_tokens / 1000) * costs["output"]
        return input_cost + output_cost
    
    def _track_usage(self, model_id: str, input_tokens: int, output_tokens: int):
        """Track token usage."""
        if not self.track_usage:
            return
        
        total_tokens = input_tokens + output_tokens
        cost = self._calculate_cost(model_id, input_tokens, output_tokens)
        
        usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            model_id=model_id,
            cost_estimate=cost
        )
        
        self.usage_history.append(usage)
    
    def _validate_response(self, response_text: str, model_id: str, token_usage: TokenUsage) -> BedrockResponse:
        """Validate response against schema."""
        try:
            return BedrockResponse(
                text=response_text,
                model_id=model_id,
                token_usage=token_usage
            )
        except ValidationError as e:
            raise ValueError(f"Response validation failed: {e}")
    
    async def invoke(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        system_prompt: Optional[str] = None,
        validate_response: bool = True,
    ) -> BedrockResponse:
        """
        Invoke LLM with prompt.
        
        Args:
            prompt: Input prompt
            model: Model ID (uses default if not specified)
            temperature: Temperature for sampling (0-1)
            max_tokens: Maximum tokens in response
            system_prompt: System prompt for Claude models
            validate_response: Whether to validate response schema
            
        Returns:
            Validated Bedrock response
            
        Raises:
            RuntimeError: If all retries fail or circuit breaker is open
            ValueError: If response validation fails
        """
        model = model or self.default_model
        
        # Use circuit breaker if enabled
        if self.circuit_breaker:
            return await self.circuit_breaker.call(
                self._invoke_internal,
                prompt,
                model,
                temperature,
                max_tokens,
                system_prompt,
                validate_response
            )
        else:
            return await self._invoke_internal(
                prompt,
                model,
                temperature,
                max_tokens,
                system_prompt,
                validate_response
            )
    
    async def _invoke_internal(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
        system_prompt: Optional[str],
        validate_response: bool,
    ) -> BedrockResponse:
        """Internal invoke method."""
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._invoke_sync,
            prompt,
            model,
            temperature,
            max_tokens,
            system_prompt,
            validate_response,
        )
    
    def _invoke_sync(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
        system_prompt: Optional[str],
        validate_response: bool,
    ) -> BedrockResponse:
        """Synchronous invoke (runs in thread pool)."""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                response_text, input_tokens, output_tokens = self._call_model(
                    prompt, model, temperature, max_tokens, system_prompt
                )
                
                # Track usage
                self._track_usage(model, input_tokens, output_tokens)
                
                # Create token usage object
                token_usage = TokenUsage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=input_tokens + output_tokens,
                    model_id=model,
                    cost_estimate=self._calculate_cost(model, input_tokens, output_tokens)
                )
                
                # Validate response if requested
                if validate_response:
                    return self._validate_response(response_text, model, token_usage)
                else:
                    return BedrockResponse(
                        text=response_text,
                        model_id=model,
                        token_usage=token_usage
                    )
            
            except ClientError as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                    continue
        
        raise RuntimeError(
            f"Failed to invoke {model} after {self.max_retries} attempts: {last_error}"
        )
    
    def _call_model(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
        system_prompt: Optional[str],
    ) -> tuple[str, int, int]:
        """Call the appropriate model and return (text, input_tokens, output_tokens)."""
        if "claude" in model.lower():
            return self._invoke_claude(
                prompt, model, temperature, max_tokens, system_prompt
            )
        elif "titan" in model.lower():
            return self._invoke_titan(prompt, model, temperature, max_tokens)
        elif "llama" in model.lower():
            return self._invoke_llama(prompt, model, temperature, max_tokens)
        else:
            raise ValueError(f"Unsupported model: {model}")
    
    def _invoke_claude(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
        system_prompt: Optional[str],
    ) -> tuple[str, int, int]:
        """Invoke Claude model and return (text, input_tokens, output_tokens)."""
        messages = [{"role": "user", "content": prompt}]
        
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        
        if system_prompt:
            body["system"] = system_prompt
        
        response = self.client.invoke_model(
            modelId=model,
            body=json.dumps(body),
        )
        
        response_body = json.loads(response["body"].read())
        
        # Extract text and token usage
        text = ""
        if "content" in response_body:
            content = response_body["content"]
            if isinstance(content, list) and len(content) > 0:
                text = content[0].get("text", "")
        
        # Extract token usage
        usage = response_body.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        
        return text, input_tokens, output_tokens
    
    def _invoke_titan(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> tuple[str, int, int]:
        """Invoke Titan model and return (text, input_tokens, output_tokens)."""
        body = {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": max_tokens,
                "temperature": temperature,
                "topP": 0.9,
            },
        }
        
        response = self.client.invoke_model(
            modelId=model,
            body=json.dumps(body),
        )
        
        response_body = json.loads(response["body"].read())
        
        # Extract text from response
        text = ""
        if "results" in response_body and len(response_body["results"]) > 0:
            text = response_body["results"][0].get("outputText", "")
        
        # Estimate token usage (Titan doesn't always provide exact counts)
        input_tokens = len(prompt.split()) * 1.3  # Rough estimate
        output_tokens = len(text.split()) * 1.3
        
        return text, int(input_tokens), int(output_tokens)
    
    def _invoke_llama(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> tuple[str, int, int]:
        """Invoke Llama model and return (text, input_tokens, output_tokens)."""
        body = {
            "prompt": prompt,
            "max_gen_len": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
        }
        
        response = self.client.invoke_model(
            modelId=model,
            body=json.dumps(body),
        )
        
        response_body = json.loads(response["body"].read())
        
        # Extract text from response
        text = response_body.get("generation", "")
        
        # Estimate token usage
        input_tokens = len(prompt.split()) * 1.3
        output_tokens = len(text.split()) * 1.3
        
        return text, int(input_tokens), int(output_tokens)
    
    async def invoke_with_fallback(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        system_prompt: Optional[str] = None,
        fallback_models: Optional[List[str]] = None,
    ) -> BedrockResponse:
        """
        Invoke LLM with automatic fallback to alternative models.
        
        Args:
            prompt: Input prompt
            temperature: Temperature for sampling
            max_tokens: Maximum tokens
            system_prompt: System prompt
            fallback_models: List of models to try (uses default if None)
            
        Returns:
            Validated Bedrock response
            
        Raises:
            RuntimeError: If all fallback models fail
        """
        if fallback_models is None:
            fallback_models = [
                self.CLAUDE_3_SONNET,
                self.CLAUDE_3_HAIKU,
                self.TITAN_TEXT,
            ]
        
        last_error = None
        for model in fallback_models:
            try:
                return await self.invoke(
                    prompt,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    system_prompt=system_prompt,
                )
            except Exception as e:
                last_error = e
                continue
        
        raise RuntimeError(f"All fallback models failed. Last error: {last_error}")

