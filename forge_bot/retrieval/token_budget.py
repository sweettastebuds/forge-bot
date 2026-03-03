"""Token budget tracking for retrieval operations.

Each LLM call in the retrieval hierarchy targets ~4K tokens.  The budget
is derived from ``llm_context_window`` so operators with larger models
get proportionally bigger per-call budgets while the structure stays the
same.
"""

from __future__ import annotations

# Rough chars-per-token ratio (conservative for code).
CHARS_PER_TOKEN = 4


class TokenBudget:
    """Track and enforce a token budget for a single LLM call.

    Parameters
    ----------
    context_window:
        The model's total context window in tokens.
    reserve_output:
        Tokens reserved for the model's response.
    reserve_system:
        Tokens reserved for the system prompt / instructions.
    """

    def __init__(
        self,
        context_window: int = 8192,
        *,
        reserve_output: int = 1024,
        reserve_system: int = 512,
    ) -> None:
        self.context_window = context_window
        self.reserve_output = reserve_output
        self.reserve_system = reserve_system
        self._used = 0

    # -- Derived budgets --------------------------------------------------

    @property
    def available(self) -> int:
        """Tokens available for retrieval content in a single call."""
        return max(
            0,
            self.context_window - self.reserve_output - self.reserve_system - self._used,
        )

    @property
    def available_chars(self) -> int:
        return self.available * CHARS_PER_TOKEN

    # -- Per-level defaults ------------------------------------------------

    @property
    def level1_chunk_tokens(self) -> int:
        """Chunk size for Level-1 BM25 retrieval (default 512)."""
        return max(1, min(512, self.available // 4))

    @property
    def level2_chunk_tokens(self) -> int:
        """Chunk size for Level-2 parallel scanning (default 1000)."""
        return max(1, min(1000, self.available))

    @property
    def level2_refine_chunk_tokens(self) -> int:
        """Chunk size for Level-2 refinement pass (default 300)."""
        return max(1, min(300, self.available // 8))

    # -- Accounting --------------------------------------------------------

    def consume(self, tokens: int) -> None:
        self._used += tokens

    def can_fit(self, tokens: int) -> bool:
        return tokens <= self.available

    def reset(self) -> None:
        self._used = 0

    def __repr__(self) -> str:
        return (
            f"TokenBudget(window={self.context_window}, "
            f"used={self._used}, available={self.available})"
        )


def estimate_tokens(text: str) -> int:
    """Quick token estimate from character count."""
    return max(1, len(text) // CHARS_PER_TOKEN)


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate *text* to fit within *max_tokens*."""
    max_chars = max_tokens * CHARS_PER_TOKEN
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."
