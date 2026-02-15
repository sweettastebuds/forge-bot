"""Tests for forge_bot.utils.dedup."""

from forge_bot.utils.dedup import DeliveryTracker


def test_new_delivery_is_not_duplicate():
    tracker = DeliveryTracker()
    assert tracker.is_duplicate("uuid-1") is False


def test_seen_delivery_is_duplicate():
    tracker = DeliveryTracker()
    tracker.is_duplicate("uuid-1")
    assert tracker.is_duplicate("uuid-1") is True


def test_different_deliveries_are_not_duplicates():
    tracker = DeliveryTracker()
    tracker.is_duplicate("uuid-1")
    assert tracker.is_duplicate("uuid-2") is False


def test_eviction_when_over_max_size():
    tracker = DeliveryTracker(max_size=3)
    tracker.is_duplicate("a")
    tracker.is_duplicate("b")
    tracker.is_duplicate("c")
    # Adding a 4th should evict "a" (oldest)
    tracker.is_duplicate("d")

    assert tracker.size == 3
    # "a" was evicted — but calling is_duplicate re-inserts it as new
    # So check size stays at 3 and "a" is not found without re-adding
    assert "a" not in tracker._seen  # directly verify eviction
    assert "b" in tracker._seen
    assert "c" in tracker._seen
    assert "d" in tracker._seen


def test_duplicate_check_refreshes_entry():
    tracker = DeliveryTracker(max_size=3)
    tracker.is_duplicate("a")
    tracker.is_duplicate("b")
    tracker.is_duplicate("c")

    # Access "a" so it moves to end (most recent)
    assert tracker.is_duplicate("a") is True

    # Adding "d" should now evict "b" (oldest), not "a"
    tracker.is_duplicate("d")

    assert "a" in tracker._seen  # refreshed, still here
    assert "b" not in tracker._seen  # evicted
    assert "c" in tracker._seen
    assert "d" in tracker._seen
