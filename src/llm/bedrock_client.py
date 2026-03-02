"""AWS Bedrock LLM client integration."""

import json
import asyncio
from typing import Optional, Dict, Any
import boto3
from botocore.exceptions import ClientError


class BedrockClient:
    """Client for AWS Bedrock LLM inference."""
    
    # Model IDs for different providers
    CLAUDE_3_SONNET = "anthropic.claude-3-sonnet-20240229-v1:0"
    CLAUDE_3_HAIKU = "anthropic.claude-3-haiku-20240307-v1:0"
    TITAN_TEXT = "amazon.titan-text-express-v1"
    LLAMA_2_70B = "meta.llama2-70b-chat-v1"
    
    def __init__(
        self,
        region: str = "us-east-1",
        default_model: str = CLAUDE_3_SONNET,
        max_retries: int = 3,
    ):
        """
        Initialize Bedrock client.
        
        Args:
            region: AWS region
            default_model: Default model to use
            max_retries: Number of retries on failure
        """
        self.region = region
        self.default_model = default_model
        self.max_retries = max_retries
        self.client = boto3.client("bedrock-runtime", region_name=region)
    
    async def invoke(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Invoke LLM with prompt.
        
        Args:
            prompt: Input prompt
            model: Model ID (uses default if not specified)
            temperature: Temperature for sampling (0-1)
            max_tokens: Maximum tokens in response
            system_prompt: System prompt for Claude models
            
        Returns:
            LLM response text
            
        Raises:
            RuntimeError: If all retries fail
        """
        model = model or self.default_model
        
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
        )
    
    def _invoke_sync(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
        system_prompt: Optional[str],
    ) -> str:
        """Synchronous invoke (runs in thread pool)."""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
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
            
            except ClientError as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    wait_time = 2 ** attempt
                    asyncio.run(asyncio.sleep(wait_time))
                    continue
        
        raise RuntimeError(
            f"Failed to invoke {model} after {self.max_retries} attempts: {last_error}"
        )
    
    def _invoke_claude(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
        system_prompt: Optional[str],
    ) -> str:
        """Invoke Claude model."""
        messages = [{"role": "user", "content": prompt}]
        
        body = {
            "model": model,
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
        
        # Extract text from response
        if "content" in response_body:
            content = response_body["content"]
            if isinstance(content, list) and len(content) > 0:
                return content[0].get("text", "")
        
        return response_body.get("text", "")
    
    def _invoke_titan(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Invoke Titan model."""
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
        if "results" in response_body and len(response_body["results"]) > 0:
            return response_body["results"][0].get("outputText", "")
        
        return ""
    
    def _invoke_llama(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Invoke Llama model."""
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
        if "generation" in response_body:
            return response_body["generation"]
        
        return ""
    
    async def invoke_with_fallback(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Invoke LLM with automatic fallback to alternative models.
        
        Args:
            prompt: Input prompt
            temperature: Temperature for sampling
            max_tokens: Maximum tokens
            system_prompt: System prompt
            
        Returns:
            LLM response text
        """
        models = [
            self.CLAUDE_3_SONNET,
            self.CLAUDE_3_HAIKU,
            self.TITAN_TEXT,
        ]
        
        for model in models:
            try:
                return await self.invoke(
                    prompt,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    system_prompt=system_prompt,
                )
            except Exception as e:
                continue
        
        raise RuntimeError("All fallback models failed")
