"""Tests for forge_bot.retrieval.chunking."""

from forge_bot.retrieval.chunking import Chunk, chunk_diff_by_file, chunk_text


def test_chunk_text_basic():
    text = "line 1\nline 2\nline 3"
    chunks = chunk_text(text, chunk_tokens=100, source="test.py")
    assert len(chunks) == 1
    assert chunks[0].source == "test.py"
    assert chunks[0].content == text


def test_chunk_text_splits_at_boundary():
    # Create text that exceeds 512 tokens (512*4 = 2048 chars)
    lines = [f"line {i}: " + "x" * 80 for i in range(30)]
    text = "\n".join(lines)
    chunks = chunk_text(text, chunk_tokens=512, source="big.py")
    assert len(chunks) > 1
    # All chunks should have source set
    for c in chunks:
        assert c.source == "big.py"
        assert c.index >= 0


def test_chunk_text_overlap():
    lines = [f"line {i}: " + "x" * 80 for i in range(30)]
    text = "\n".join(lines)
    chunks = chunk_text(text, chunk_tokens=512, overlap_tokens=64)
    if len(chunks) >= 2:
        # Second chunk should start with some lines from the first
        last_lines_first = chunks[0].content.split("\n")[-3:]
        first_lines_second = chunks[1].content.split("\n")[:3]
        # There should be overlap
        overlap = set(last_lines_first) & set(first_lines_second)
        assert len(overlap) > 0


def test_chunk_text_empty():
    assert chunk_text("", chunk_tokens=512) == []
    assert chunk_text("   ", chunk_tokens=512) == []


def test_chunk_text_sets_line_numbers():
    text = "a\nb\nc\nd\ne"
    chunks = chunk_text(text, chunk_tokens=100)
    assert len(chunks) == 1
    assert chunks[0].start_line == 1
    assert chunks[0].end_line == 5


def test_chunk_diff_by_file():
    diff = (
        "diff --git a/foo.py b/foo.py\n"
        "--- a/foo.py\n"
        "+++ b/foo.py\n"
        "@@ -1,3 +1,4 @@\n"
        " existing\n"
        "+new line\n"
        "diff --git a/bar.py b/bar.py\n"
        "--- a/bar.py\n"
        "+++ b/bar.py\n"
        "@@ -5,3 +5,4 @@\n"
        " other\n"
        "+another new\n"
    )
    chunks = chunk_diff_by_file(diff)
    assert len(chunks) == 2
    assert chunks[0].source == "foo.py"
    assert chunks[1].source == "bar.py"


def test_chunk_diff_by_file_empty():
    assert chunk_diff_by_file("") == []
    assert chunk_diff_by_file("  ") == []


def test_chunk_diff_by_file_single():
    diff = (
        "diff --git a/only.py b/only.py\n"
        "+++ b/only.py\n"
        "+first line\n"
    )
    chunks = chunk_diff_by_file(diff)
    assert len(chunks) == 1
    assert chunks[0].source == "only.py"


def test_chunk_token_count_auto():
    chunk = Chunk(content="a" * 400, source="test", index=0)
    assert chunk.token_count == 100  # 400 chars / 4
