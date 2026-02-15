"""Webhook delivery deduplication using an in-memory LRU cache."""

from collections import OrderedDict

DEFAULT_MAX_SIZE = 10_000


class DeliveryTracker:
    """Track seen webhook delivery UUIDs to prevent duplicate processing.

    Uses an OrderedDict as an LRU cache. When max_size is reached,
    the oldest entries are evicted.
    """

    def __init__(self, max_size: int = DEFAULT_MAX_SIZE) -> None:
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._max_size = max_size

    def is_duplicate(self, delivery_id: str) -> bool:
        """Check if a delivery ID has been seen. If new, record it.

        Returns True if this ID was already seen (duplicate).
        Returns False if this is a new ID (now recorded).
        """
        if delivery_id in self._seen:
            # Move to end so it's treated as recently seen
            self._seen.move_to_end(delivery_id)
            return True

        # Record new delivery
        self._seen[delivery_id] = None

        # Evict oldest if over capacity
        while len(self._seen) > self._max_size:
            self._seen.popitem(last=False)

        return False

    @property
    def size(self) -> int:
        return len(self._seen)
