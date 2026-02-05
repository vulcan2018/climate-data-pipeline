"""Redis caching layer for API responses.

Provides caching utilities for expensive computations and frequently
accessed data to achieve <2 second response times.
"""

import hashlib
import json
import os
from functools import wraps
from typing import Any, Callable, TypeVar

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


# Type variable for generic caching
T = TypeVar("T")

# Default cache settings
DEFAULT_TTL = 3600  # 1 hour
DEFAULT_REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")


class CacheManager:
    """Manages caching operations with Redis backend."""

    def __init__(
        self,
        redis_url: str | None = None,
        default_ttl: int = DEFAULT_TTL,
        prefix: str = "climate:",
    ):
        """Initialize cache manager.

        Args:
            redis_url: Redis connection URL (None = use env var or localhost)
            default_ttl: Default time-to-live in seconds
            prefix: Key prefix for all cached items
        """
        self.default_ttl = default_ttl
        self.prefix = prefix
        self._client: Any | None = None

        if redis_url is None:
            redis_url = DEFAULT_REDIS_URL

        self._redis_url = redis_url

    @property
    def client(self) -> Any:
        """Get Redis client, creating if needed."""
        if self._client is None:
            if not REDIS_AVAILABLE:
                raise RuntimeError("Redis package not installed. Run: pip install redis")
            self._client = redis.from_url(self._redis_url)
        return self._client

    def _make_key(self, key: str) -> str:
        """Create prefixed cache key."""
        return f"{self.prefix}{key}"

    def get(self, key: str) -> Any | None:
        """Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        try:
            data = self.client.get(self._make_key(key))
            if data:
                return json.loads(data)
        except Exception:
            pass
        return None

    def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
    ) -> bool:
        """Set value in cache.

        Args:
            key: Cache key
            value: Value to cache (must be JSON-serializable)
            ttl: Time-to-live in seconds (None = use default)

        Returns:
            True if successful
        """
        if ttl is None:
            ttl = self.default_ttl

        try:
            self.client.setex(
                self._make_key(key),
                ttl,
                json.dumps(value),
            )
            return True
        except Exception:
            return False

    def delete(self, key: str) -> bool:
        """Delete value from cache.

        Args:
            key: Cache key

        Returns:
            True if deleted
        """
        try:
            return bool(self.client.delete(self._make_key(key)))
        except Exception:
            return False

    def clear_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern.

        Args:
            pattern: Glob pattern (e.g., "dataset:*")

        Returns:
            Number of keys deleted
        """
        try:
            full_pattern = self._make_key(pattern)
            keys = self.client.keys(full_pattern)
            if keys:
                return self.client.delete(*keys)
        except Exception:
            pass
        return 0

    def exists(self, key: str) -> bool:
        """Check if key exists in cache.

        Args:
            key: Cache key

        Returns:
            True if key exists
        """
        try:
            return bool(self.client.exists(self._make_key(key)))
        except Exception:
            return False

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        try:
            info = self.client.info()
            return {
                "connected": True,
                "used_memory_mb": info.get("used_memory", 0) / (1024 * 1024),
                "total_keys": info.get("db0", {}).get("keys", 0),
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
            }
        except Exception as e:
            return {
                "connected": False,
                "error": str(e),
            }


# Global cache instance (lazy initialization)
_cache: CacheManager | None = None


def get_cache() -> CacheManager:
    """Get the global cache manager instance."""
    global _cache
    if _cache is None:
        _cache = CacheManager()
    return _cache


def cache_key(*args: Any, **kwargs: Any) -> str:
    """Generate a cache key from arguments.

    Args:
        *args: Positional arguments to include in key
        **kwargs: Keyword arguments to include in key

    Returns:
        Deterministic hash-based cache key
    """
    key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True)
    return hashlib.sha256(key_data.encode()).hexdigest()[:16]


def cached(
    ttl: int = DEFAULT_TTL,
    key_prefix: str = "",
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for caching function results.

    Args:
        ttl: Cache time-to-live in seconds
        key_prefix: Prefix for cache key

    Returns:
        Decorator function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Generate cache key
            key = f"{key_prefix}{func.__name__}:{cache_key(*args, **kwargs)}"

            # Try to get from cache
            cache = get_cache()
            cached_result = cache.get(key)
            if cached_result is not None:
                return cached_result

            # Compute result
            result = func(*args, **kwargs)

            # Store in cache
            cache.set(key, result, ttl)

            return result

        return wrapper

    return decorator


def cached_async(
    ttl: int = DEFAULT_TTL,
    key_prefix: str = "",
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for caching async function results.

    Args:
        ttl: Cache time-to-live in seconds
        key_prefix: Prefix for cache key

    Returns:
        Decorator function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            # Generate cache key
            key = f"{key_prefix}{func.__name__}:{cache_key(*args, **kwargs)}"

            # Try to get from cache
            cache = get_cache()
            cached_result = cache.get(key)
            if cached_result is not None:
                return cached_result

            # Compute result
            result = await func(*args, **kwargs)

            # Store in cache
            cache.set(key, result, ttl)

            return result

        return wrapper  # type: ignore

    return decorator


class InMemoryCache:
    """Simple in-memory cache for when Redis is not available."""

    def __init__(self, max_size: int = 1000):
        """Initialize in-memory cache.

        Args:
            max_size: Maximum number of items to cache
        """
        self._cache: dict[str, Any] = {}
        self._max_size = max_size

    def get(self, key: str) -> Any | None:
        """Get value from cache."""
        return self._cache.get(key)

    def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set value in cache (ttl ignored in memory cache)."""
        # Simple LRU-like eviction: remove oldest if at capacity
        if len(self._cache) >= self._max_size:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[key] = value
        return True

    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> None:
        """Clear all cached items."""
        self._cache.clear()
