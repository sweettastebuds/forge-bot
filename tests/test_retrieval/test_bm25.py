"""Tests for forge_bot.retrieval.bm25."""

from forge_bot.retrieval.bm25 import BM25Index, _tokenize


def test_tokenize_basic():
    tokens = _tokenize("Hello World foo_bar")
    assert tokens == ["hello", "world", "foo_bar"]


def test_tokenize_strips_short_tokens():
    tokens = _tokenize("a bb ccc")
    assert tokens == ["bb", "ccc"]


def test_tokenize_splits_on_punctuation():
    tokens = _tokenize("def my_func(arg): pass")
    assert "my_func" in tokens
    assert "arg" in tokens
    assert "def" in tokens
    assert "pass" in tokens


def test_build_empty_corpus():
    index = BM25Index.build([])
    assert index._n == 0
    result = index.query(["anything"])
    assert result == []


def test_build_and_query_single_doc():
    docs = ["the quick brown fox"]
    index = BM25Index.build(docs)
    results = index.query(["fox"])
    assert len(results) == 1
    assert results[0][0] == 0  # doc index
    assert results[0][1] > 0  # positive score


def test_query_ranks_relevant_higher():
    docs = [
        "python is a programming language used for web development",
        "java is also a programming language",
        "cooking recipes for pasta and pizza",
    ]
    index = BM25Index.build(docs)
    results = index.query(["python", "programming"])
    # Python doc should rank first
    assert results[0][0] == 0
    # Cooking doc should not appear
    doc_indices = [r[0] for r in results]
    assert 2 not in doc_indices


def test_query_text_convenience():
    docs = ["function that handles errors and logging", "database connection pool"]
    index = BM25Index.build(docs)
    results = index.query_text("function errors handles")
    assert len(results) >= 1
    assert results[0][0] == 0  # errors doc


def test_top_k_limits_results():
    docs = [f"document number {i}" for i in range(20)]
    index = BM25Index.build(docs)
    results = index.query(["document", "number"], top_k=5)
    assert len(results) <= 5


def test_no_matching_terms_returns_empty():
    docs = ["alpha beta gamma"]
    index = BM25Index.build(docs)
    results = index.query(["zzzzz"])
    assert results == []


def test_multiple_term_overlap():
    docs = [
        "async def handle_request(request): return response",
        "def process_data(data): return result",
        "class RequestHandler: pass",
    ]
    index = BM25Index.build(docs)
    results = index.query(["request", "handle", "async"])
    assert results[0][0] == 0  # first doc has all terms
