"""Tests for forge_bot.rag.chunker."""

from forge_bot.rag.chunker import Chunker, CodeChunk, detect_language


def test_detect_language_python():
    assert detect_language("src/main.py") == "python"


def test_detect_language_javascript():
    assert detect_language("app/index.js") == "javascript"
    assert detect_language("app/index.ts") == "javascript"


def test_detect_language_unknown():
    assert detect_language("Makefile") == "text"
    assert detect_language("data.csv") == "text"


def test_chunk_empty_file():
    chunker = Chunker()
    chunks = chunker.chunk_file("", "empty.py", "python")
    assert chunks == []


def test_chunk_whitespace_only():
    chunker = Chunker()
    chunks = chunker.chunk_file("   \n\n  ", "blank.py", "python")
    assert chunks == []


def test_chunk_fallback_small_file():
    """Small file should produce a single chunk via fallback."""
    chunker = Chunker(chunk_size=500, overlap=50)
    content = "x = 1\ny = 2\nprint(x + y)\n"
    chunks = chunker.chunk_file(content, "small.py", "python")
    assert len(chunks) >= 1
    assert chunks[0].file_path == "small.py"
    assert chunks[0].language == "python"
    assert chunks[0].content == content.strip() or content in chunks[0].content


def test_chunk_fallback_splits_large():
    """Large file should be split into multiple chunks."""
    chunker = Chunker(chunk_size=50, overlap=10)  # Very small chunks
    paragraphs = ["paragraph " + str(i) + "\n" + "x" * 100 for i in range(10)]
    content = "\n\n".join(paragraphs)
    chunks = chunker.chunk_file(content, "big.txt", "text")
    assert len(chunks) > 1


def test_chunk_metadata_line_numbers():
    chunker = Chunker(chunk_size=500)
    content = "line1\nline2\nline3\n"
    chunks = chunker.chunk_file(content, "test.txt", "text")
    assert len(chunks) >= 1
    assert chunks[0].start_line >= 1


def test_chunk_token_count():
    chunker = Chunker()
    content = "a" * 400  # 400 chars = ~100 tokens
    chunks = chunker.chunk_file(content, "test.txt", "text")
    assert len(chunks) == 1
    assert chunks[0].token_count == 100


def test_chunk_language_auto_detection():
    chunker = Chunker(chunk_size=500)
    content = "function hello() { console.log('hi'); }\n"
    chunks = chunker.chunk_file(content, "app.js")
    assert len(chunks) >= 1
    assert chunks[0].language == "javascript"


def test_code_chunk_dataclass():
    chunk = CodeChunk(
        content="def foo(): pass",
        file_path="test.py",
        language="python",
        symbol_name="foo",
        start_line=1,
        end_line=1,
    )
    assert chunk.token_count == len("def foo(): pass") // 4
    assert chunk.symbol_name == "foo"


def test_chunk_preserves_content():
    """Chunks should contain the actual source code."""
    chunker = Chunker(chunk_size=1000)
    content = "def hello():\n    return 'world'\n"
    chunks = chunker.chunk_file(content, "hello.py", "python")
    all_content = " ".join(c.content for c in chunks)
    assert "hello" in all_content
    assert "world" in all_content
