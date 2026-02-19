"""Tests for forge_bot.retrieval.token_budget."""

from forge_bot.retrieval.token_budget import (
    TokenBudget,
    estimate_tokens,
    truncate_to_tokens,
)


def test_default_budget_available():
    b = TokenBudget(context_window=8192)
    # 8192 - 1024 (output) - 512 (system) = 6656
    assert b.available == 6656


def test_consume_reduces_available():
    b = TokenBudget(context_window=4096)
    initial = b.available
    b.consume(500)
    assert b.available == initial - 500


def test_can_fit():
    b = TokenBudget(context_window=4096)
    assert b.can_fit(100)
    assert b.can_fit(b.available)
    assert not b.can_fit(b.available + 1)


def test_reset_restores_budget():
    b = TokenBudget(context_window=4096)
    initial = b.available
    b.consume(1000)
    assert b.available < initial
    b.reset()
    assert b.available == initial


def test_available_never_negative():
    b = TokenBudget(context_window=100, reserve_output=80, reserve_system=80)
    assert b.available == 0


def test_level1_chunk_tokens():
    b = TokenBudget(context_window=8192)
    assert b.level1_chunk_tokens == 512  # min(512, 6656//4)


def test_level2_chunk_tokens():
    b = TokenBudget(context_window=8192)
    assert b.level2_chunk_tokens == 1000  # min(1000, 6656)


def test_level2_chunk_tokens_small_context():
    b = TokenBudget(context_window=2048)
    # 2048 - 1024 - 512 = 512 available
    assert b.level2_chunk_tokens == 512  # min(1000, 512)


def test_level2_refine_chunk_tokens():
    b = TokenBudget(context_window=8192)
    assert b.level2_refine_chunk_tokens == 300  # min(300, 6656//8)


def test_estimate_tokens():
    assert estimate_tokens("") == 1  # min 1
    assert estimate_tokens("a" * 400) == 100  # 400 / 4


def test_truncate_to_tokens_no_change():
    text = "short"
    assert truncate_to_tokens(text, 100) == text


def test_truncate_to_tokens_truncates():
    text = "a" * 1000
    result = truncate_to_tokens(text, 10)
    assert len(result) == 43  # 10 * 4 = 40 chars + "..."
    assert result.endswith("...")
