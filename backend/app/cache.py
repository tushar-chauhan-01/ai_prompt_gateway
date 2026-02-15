"""
In-memory response cache with LRU eviction and TTL expiry.

Caches full RouteResponse objects keyed by (prompt, classifier_mode)
so duplicate prompts skip the real API call entirely.
"""

import hashlib
import threading
import time
import traceback
from collections import OrderedDict
from typing import Optional

from backend.app.models import ClassifierMode, RouteResponse

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_MAX_SIZE = 100       # max entries before LRU eviction
_DEFAULT_TTL_SECONDS = 1800   # 30 minutes


# ---------------------------------------------------------------------------
# Cache entry wrapper
# ---------------------------------------------------------------------------

class _CacheEntry:
    """Wraps a cached value with its creation timestamp."""

    __slots__ = ("value", "created_at")

    def __init__(self, value: RouteResponse) -> None:
        self.value = value
        self.created_at = time.monotonic()

    def is_expired(self, ttl: float) -> bool:
        try:
            return (time.monotonic() - self.created_at) > ttl
        except Exception:
            traceback.print_exc()
            raise


# ---------------------------------------------------------------------------
# ResponseCache
# ---------------------------------------------------------------------------

class ResponseCache:
    """
    Thread-safe, in-memory LRU cache with TTL for RouteResponse objects.

    - Max size:  evicts least-recently-used entry when full.
    - TTL:       entries older than `ttl_seconds` are treated as misses
                 and lazily removed.
    """

    def __init__(
        self,
        max_size: int = _DEFAULT_MAX_SIZE,
        ttl_seconds: float = _DEFAULT_TTL_SECONDS,
    ) -> None:
        try:
            self._max_size = max_size
            self._ttl = ttl_seconds
            self._store: OrderedDict[str, _CacheEntry] = OrderedDict()
            self._lock = threading.Lock()
            self._hits = 0
            self._misses = 0
        except Exception:
            traceback.print_exc()
            raise

    # ------------------------------------------------------------------
    # Key generation
    # ------------------------------------------------------------------

    @staticmethod
    def _make_key(prompt: str, classifier_mode: ClassifierMode) -> str:
        """Create a deterministic cache key from prompt + classifier mode."""
        try:
            raw = f"{prompt.strip().lower()}::{classifier_mode.value}"
            return hashlib.sha256(raw.encode()).hexdigest()
        except Exception:
            traceback.print_exc()
            raise

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(
        self,
        prompt: str,
        classifier_mode: ClassifierMode,
    ) -> Optional[RouteResponse]:
        """
        Look up a cached response.

        Returns the RouteResponse on hit, or None on miss / expiry.
        On hit the entry is moved to the end (most-recently-used).
        """
        try:
            key = self._make_key(prompt, classifier_mode)
            with self._lock:
                entry = self._store.get(key)
                if entry is None:
                    self._misses += 1
                    return None

                if entry.is_expired(self._ttl):
                    del self._store[key]
                    self._misses += 1
                    return None

                # Move to end → most recently used
                self._store.move_to_end(key)
                self._hits += 1
                return entry.value
        except Exception:
            traceback.print_exc()
            raise

    def put(
        self,
        prompt: str,
        classifier_mode: ClassifierMode,
        response: RouteResponse,
    ) -> None:
        """
        Store a response in the cache.

        If the cache is at capacity, the least-recently-used entry
        is evicted first.
        """
        try:
            key = self._make_key(prompt, classifier_mode)
            with self._lock:
                # If key already exists, remove it so we can re-insert at end
                if key in self._store:
                    del self._store[key]

                # Evict LRU if at capacity
                while len(self._store) >= self._max_size:
                    self._store.popitem(last=False)

                self._store[key] = _CacheEntry(response)
        except Exception:
            traceback.print_exc()
            raise

    def clear(self) -> None:
        """Remove all cached entries and reset stats."""
        try:
            with self._lock:
                self._store.clear()
                self._hits = 0
                self._misses = 0
        except Exception:
            traceback.print_exc()
            raise

    def stats(self) -> dict:
        """Return cache statistics."""
        try:
            with self._lock:
                total = self._hits + self._misses
                hit_rate = (self._hits / total * 100) if total > 0 else 0.0
                return {
                    "size": len(self._store),
                    "max_size": self._max_size,
                    "ttl_seconds": self._ttl,
                    "hits": self._hits,
                    "misses": self._misses,
                    "hit_rate_percent": round(hit_rate, 2),
                }
        except Exception:
            traceback.print_exc()
            raise
