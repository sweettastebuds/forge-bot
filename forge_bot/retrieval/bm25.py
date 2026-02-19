"""Pure-Python BM25 scoring (no external dependencies).

Implements the Okapi BM25 ranking function for a small, in-memory corpus
of text chunks.  Designed for the Level-1 retrieval pass where we need
fast keyword-based ranking without requiring numpy or rank-bm25.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field


def _tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric, drop short tokens."""
    return [t for t in re.split(r"[^a-zA-Z0-9_]+", text.lower()) if len(t) > 1]


@dataclass
class BM25Index:
    """An in-memory BM25 index over a list of text documents.

    Build once with ``BM25Index.build(docs)``, then call
    ``index.query(keywords, top_k)`` to get ranked results.
    """

    # Tuning knobs (sensible defaults for code search).
    k1: float = 1.5
    b: float = 0.75

    # Internal state set by ``build()``.
    _docs: list[str] = field(default_factory=list, repr=False)
    _doc_tokens: list[list[str]] = field(default_factory=list, repr=False)
    _doc_lens: list[int] = field(default_factory=list, repr=False)
    _avgdl: float = 0.0
    _n: int = 0
    # term → set of doc indices containing it
    _df: dict[str, int] = field(default_factory=dict, repr=False)

    @classmethod
    def build(
        cls,
        documents: list[str],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> BM25Index:
        """Create a BM25 index from a list of raw text documents."""
        idx = cls(k1=k1, b=b)
        idx._docs = documents
        idx._n = len(documents)

        for doc in documents:
            tokens = _tokenize(doc)
            idx._doc_tokens.append(tokens)
            idx._doc_lens.append(len(tokens))

        idx._avgdl = sum(idx._doc_lens) / idx._n if idx._n > 0 else 1.0

        # Build document-frequency table.
        df: dict[str, int] = {}
        for tokens in idx._doc_tokens:
            seen: set[str] = set()
            for t in tokens:
                if t not in seen:
                    df[t] = df.get(t, 0) + 1
                    seen.add(t)
        idx._df = df
        return idx

    def query(
        self,
        keywords: list[str],
        *,
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        """Return ``(doc_index, score)`` pairs sorted by descending score.

        Parameters
        ----------
        keywords:
            Pre-tokenized query terms (should already be lowercased).
        top_k:
            Maximum number of results to return.
        """
        if not self._docs:
            return []

        scores = [0.0] * self._n

        for term in keywords:
            if term not in self._df:
                continue
            df = self._df[term]
            # IDF with smoothing to avoid negatives.
            idf = math.log((self._n - df + 0.5) / (df + 0.5) + 1.0)

            for i, tokens in enumerate(self._doc_tokens):
                tf = tokens.count(term)
                if tf == 0:
                    continue
                dl = self._doc_lens[i]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
                scores[i] += idf * numerator / denominator

        # Rank and return top-k non-zero scores.
        ranked = sorted(
            ((i, s) for i, s in enumerate(scores) if s > 0),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_k]

    def query_text(self, query: str, *, top_k: int = 10) -> list[tuple[int, float]]:
        """Convenience: tokenize *query* then rank."""
        return self.query(_tokenize(query), top_k=top_k)
