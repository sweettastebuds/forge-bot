"""Estimated token budget tracking for multi-round LLM sessions."""

from __future__ import annotations
from dataclasses import dataclass, field

# Conservative estimate: 1 token ≈ 4 chars for mixed code/natural-language content.
_CHARS_PER_TOKEN = 4


@dataclass
class TokenBudget:
    """Track estimated token consumption against a fixed context window.
    ``reserved_tokens`` covers all fixed overhead (system prompt, tool
    schemas, output reserve) so callers don't need to account for each
    piece separately.
    """

    context_window: int
    reserved_tokens: int
    _consumed: int = field(default=0, init=False, repr=False)

    # -- Convenience constructor -------------------------------------------
    @classmethod
    def for_context_window(cls, context_window: int) -> TokenBudget:
        """Create a budget with a sensible default reserve (25 % of window)."""
        return cls(
            context_window=context_window,
            reserved_tokens=context_window // 4,
        )

    # -- Read-only properties ----------------------------------------------
    @property
    def consumed(self) -> int:
        """Tokens consumed so far."""
        return self._consumed

    @property
    def remaining(self) -> int:
        """Tokens still available for new content."""
        return max(0, self.context_window - self.reserved_tokens - self._consumed)

    @property
    def remaining_chars(self) -> int:
        """Approximate characters that still fit."""
        return self.remaining * _CHARS_PER_TOKEN

    @property
    def usage_ratio(self) -> float:
        """Fraction of the *usable* budget (window minus reserve) consumed.
        Returns a value in [0.0, 1.0].
        """
        usable = self.context_window - self.reserved_tokens
        if usable <= 0:
            return 1.0
        return min(1.0, self._consumed / usable)

    # -- Mutation -----------------------------------------------------------
    def consume(self, text: str = "", *, tokens: int = 0) -> None:
        """Record token consumption.
        Pass *text* to estimate from character length, or *tokens* for a
        known count.  If both are given, *text* takes precedence.
        """
        if text:
            self._consumed += len(text) // _CHARS_PER_TOKEN
        else:
            self._consumed += tokens

    def can_fit(
        self,
        text: str = "",
        *,
        tokens: int = 0,
        reserve: int = 0,
    ) -> bool:
        """Check whether *text* (or *tokens*) fits with optional extra reserve."""
        cost = len(text) // _CHARS_PER_TOKEN if text else tokens
        return cost + reserve <= self.remaining
