"""
Redis cache client for caching and rate limiting
"""
import json
from typing import Optional, Any
from datetime import timedelta
import os

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

class CacheClient:
    def __init__(self):
        self.enabled = REDIS_AVAILABLE and os.getenv("REDIS_ENABLED", "false").lower() == "true"
        self.client = None
        
        if self.enabled:
            redis_host = os.getenv("REDIS_HOST", "localhost")
            redis_port = int(os.getenv("REDIS_PORT", "6379"))
            try:
                self.client = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    decode_responses=True
                )
            except Exception as e:
                print(f"Redis connection failed: {e}")
                self.enabled = False
    
    async def get(self, key: str) -> Optional[str]:
        """Get value from cache"""
        if not self.enabled or not self.client:
            return None
        try:
            return await self.client.get(key)
        except Exception:
            return None
    
    async def set(self, key: str, value: str, ttl: int = 3600) -> bool:
        """Set value in cache with TTL (seconds)"""
        if not self.enabled or not self.client:
            return False
        try:
            await self.client.setex(key, ttl, value)
            return True
        except Exception:
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if not self.enabled or not self.client:
            return False
        try:
            await self.client.delete(key)
            return True
        except Exception:
            return False
    
    async def increment(self, key: str, amount: int = 1, ttl: int = 60) -> int:
        """Increment counter (for rate limiting)"""
        if not self.enabled or not self.client:
            return 0
        try:
            pipe = self.client.pipeline()
            pipe.incr(key, amount)
            pipe.expire(key, ttl)
            results = await pipe.execute()
            return results[0]
        except Exception:
            return 0
    
    async def get_json(self, key: str) -> Optional[Any]:
        """Get JSON value from cache"""
        value = await self.get(key)
        if value:
            try:
                return json.loads(value)
            except Exception:
                return None
        return None
    
    async def set_json(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set JSON value in cache"""
        try:
            json_str = json.dumps(value)
            return await self.set(key, json_str, ttl)
        except Exception:
            return False
    
    async def close(self):
        """Close Redis connection"""
        if self.client:
            await self.client.close()
