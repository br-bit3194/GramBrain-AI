"""
Rate limiting middleware using Redis
"""
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from .redis_client import CacheClient
from typing import Dict
import time

class RateLimiter:
    def __init__(self, cache_client: CacheClient):
        self.cache = cache_client
        # Rate limits per role (requests per minute)
        self.rate_limits: Dict[str, int] = {
            "farmer": 100,
            "village_leader": 200,
            "policymaker": 150,
            "consumer": 100,
            "admin": 1000,
            "anonymous": 20,
        }
    
    async def check_rate_limit(self, user_id: str, role: str = "anonymous") -> tuple[bool, int]:
        """
        Check if user has exceeded rate limit
        Returns: (is_allowed, remaining_requests)
        """
        limit = self.rate_limits.get(role, 100)
        key = f"rate_limit:{user_id}:{int(time.time() // 60)}"
        
        current = await self.cache.increment(key, 1, 60)
        
        if current > limit:
            return False, 0
        
        return True, limit - current
    
    def get_retry_after(self) -> int:
        """Get seconds until rate limit resets"""
        return 60 - (int(time.time()) % 60)


async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware"""
    # Skip rate limiting for health check
    if request.url.path == "/health":
        return await call_next(request)
    
    # Get user info from request state (set by auth middleware)
    user_id = getattr(request.state, "user_id", request.client.host)
    role = getattr(request.state, "role", "anonymous")
    
    # Check rate limit
    cache = CacheClient()
    limiter = RateLimiter(cache)
    is_allowed, remaining = await limiter.check_rate_limit(user_id, role)
    
    if not is_allowed:
        retry_after = limiter.get_retry_after()
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "status": "error",
                "detail": "Rate limit exceeded",
                "retry_after": retry_after
            },
            headers={"Retry-After": str(retry_after)}
        )
    
    # Add rate limit headers
    response = await call_next(request)
    response.headers["X-RateLimit-Remaining"] = str(remaining)
    response.headers["X-RateLimit-Reset"] = str(limiter.get_retry_after())
    
    return response
