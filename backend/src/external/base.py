"""Base class for FastMCP tool wrappers."""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
from enum import Enum


logger = logging.getLogger(__name__)


class ToolError(Exception):
    """Exception raised when a tool execution fails."""
    pass


@dataclass
class ToolResult:
    """Result from a tool execution."""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    tool_name: str = ""
    execution_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    retry_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "tool_name": self.tool_name,
            "execution_time_ms": self.execution_time_ms,
            "timestamp": self.timestamp.isoformat(),
            "retry_count": self.retry_count,
        }


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker for external API calls."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout_seconds: int = 60,
        half_open_max_calls: int = 3
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            timeout_seconds: Seconds to wait before attempting reset
            half_open_max_calls: Max calls allowed in half-open state
        """
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.half_open_max_calls = half_open_max_calls
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = CircuitState.CLOSED
        self.half_open_calls = 0
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return False
        return time.time() - self.last_failure_time >= self.timeout_seconds
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Async function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            ToolError: If circuit is open or function fails
        """
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                logger.info("Circuit breaker attempting reset to HALF_OPEN")
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
            else:
                raise ToolError("Circuit breaker is OPEN - service unavailable")
        
        if self.state == CircuitState.HALF_OPEN:
            if self.half_open_calls >= self.half_open_max_calls:
                raise ToolError("Circuit breaker HALF_OPEN - max calls reached")
            self.half_open_calls += 1
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        """Handle successful call."""
        if self.state == CircuitState.HALF_OPEN:
            logger.info("Circuit breaker reset to CLOSED after successful call")
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.half_open_calls = 0
        elif self.state == CircuitState.CLOSED:
            self.failure_count = max(0, self.failure_count - 1)
    
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            logger.warning("Circuit breaker reopening to OPEN after failure in HALF_OPEN")
            self.state = CircuitState.OPEN
        elif self.failure_count >= self.failure_threshold:
            logger.warning(f"Circuit breaker opening after {self.failure_count} failures")
            self.state = CircuitState.OPEN


class RateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(self, requests_per_second: float = 10.0):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_second: Maximum requests per second
        """
        self.requests_per_second = requests_per_second
        self.tokens = requests_per_second
        self.last_update = time.time()
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire a token, waiting if necessary."""
        async with self.lock:
            now = time.time()
            elapsed = now - self.last_update
            self.tokens = min(
                self.requests_per_second,
                self.tokens + elapsed * self.requests_per_second
            )
            self.last_update = now
            
            if self.tokens < 1:
                wait_time = (1 - self.tokens) / self.requests_per_second
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= 1


class FastMCPTool(ABC):
    """Base class for FastMCP tool wrappers."""
    
    def __init__(
        self,
        tool_name: str,
        max_retries: int = 3,
        retry_base_delay_ms: int = 100,
        retry_max_delay_ms: int = 5000,
        circuit_breaker_threshold: int = 5,
        rate_limit_rps: float = 10.0,
    ):
        """
        Initialize FastMCP tool.
        
        Args:
            tool_name: Name of the tool
            max_retries: Maximum retry attempts
            retry_base_delay_ms: Base delay for exponential backoff
            retry_max_delay_ms: Maximum delay between retries
            circuit_breaker_threshold: Failures before circuit opens
            rate_limit_rps: Requests per second limit
        """
        self.tool_name = tool_name
        self.max_retries = max_retries
        self.retry_base_delay_ms = retry_base_delay_ms
        self.retry_max_delay_ms = retry_max_delay_ms
        self.circuit_breaker = CircuitBreaker(failure_threshold=circuit_breaker_threshold)
        self.rate_limiter = RateLimiter(requests_per_second=rate_limit_rps)
        self.logger = logging.getLogger(f"{__name__}.{tool_name}")
    
    @abstractmethod
    async def _execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the tool logic.
        
        Args:
            **kwargs: Tool-specific parameters
            
        Returns:
            Tool execution result
            
        Raises:
            ToolError: If execution fails
        """
        pass
    
    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """
        Get the tool schema for registration.
        
        Returns:
            JSON schema for the tool
        """
        pass
    
    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute tool with retry, circuit breaker, and rate limiting.
        
        Args:
            **kwargs: Tool-specific parameters
            
        Returns:
            ToolResult with execution details
        """
        start_time = time.time()
        retry_count = 0
        last_error = None
        
        # Apply rate limiting
        await self.rate_limiter.acquire()
        
        for attempt in range(self.max_retries + 1):
            try:
                # Execute with circuit breaker protection
                data = await self.circuit_breaker.call(self._execute, **kwargs)
                
                execution_time_ms = (time.time() - start_time) * 1000
                
                self.logger.info(
                    f"Tool {self.tool_name} executed successfully",
                    extra={
                        "tool_name": self.tool_name,
                        "execution_time_ms": execution_time_ms,
                        "retry_count": retry_count,
                        "params": kwargs,
                    }
                )
                
                return ToolResult(
                    success=True,
                    data=data,
                    tool_name=self.tool_name,
                    execution_time_ms=execution_time_ms,
                    retry_count=retry_count,
                )
            
            except Exception as e:
                last_error = str(e)
                retry_count = attempt
                
                self.logger.warning(
                    f"Tool {self.tool_name} failed (attempt {attempt + 1}/{self.max_retries + 1})",
                    extra={
                        "tool_name": self.tool_name,
                        "error": last_error,
                        "attempt": attempt + 1,
                        "params": kwargs,
                    }
                )
                
                # Don't retry if circuit is open
                if "Circuit breaker" in last_error:
                    break
                
                # Don't retry on last attempt
                if attempt < self.max_retries:
                    delay_ms = self._calculate_backoff_delay(attempt)
                    await asyncio.sleep(delay_ms / 1000.0)
        
        # All retries exhausted
        execution_time_ms = (time.time() - start_time) * 1000
        
        self.logger.error(
            f"Tool {self.tool_name} failed after {retry_count + 1} attempts",
            extra={
                "tool_name": self.tool_name,
                "error": last_error,
                "retry_count": retry_count,
                "params": kwargs,
            }
        )
        
        return ToolResult(
            success=False,
            error=last_error,
            tool_name=self.tool_name,
            execution_time_ms=execution_time_ms,
            retry_count=retry_count,
        )
    
    def _calculate_backoff_delay(self, attempt: int) -> float:
        """
        Calculate exponential backoff delay with jitter.
        
        Args:
            attempt: Retry attempt number (0-indexed)
            
        Returns:
            Delay in milliseconds
        """
        import random
        
        delay = min(
            self.retry_base_delay_ms * (2 ** attempt),
            self.retry_max_delay_ms
        )
        
        # Add jitter (±25%)
        jitter = random.uniform(0.75, 1.25)
        return delay * jitter
    
    async def execute_with_fallback(
        self,
        fallback_func: Optional[Callable] = None,
        **kwargs
    ) -> ToolResult:
        """
        Execute tool with fallback function.
        
        Args:
            fallback_func: Async function to call if tool fails
            **kwargs: Tool-specific parameters
            
        Returns:
            ToolResult from tool or fallback
        """
        result = await self.execute(**kwargs)
        
        if not result.success and fallback_func:
            self.logger.info(f"Executing fallback for {self.tool_name}")
            try:
                fallback_data = await fallback_func(**kwargs)
                return ToolResult(
                    success=True,
                    data=fallback_data,
                    tool_name=f"{self.tool_name}_fallback",
                    execution_time_ms=0.0,
                    retry_count=0,
                )
            except Exception as e:
                self.logger.error(f"Fallback failed for {self.tool_name}: {e}")
        
        return result
