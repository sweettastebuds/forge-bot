"""Flexible text chunking at configurable sizes for retrieval levels.

Produces plain text chunks with metadata for BM25 indexing and parallel
scanning.  Re-uses the CodeChunk dataclass from the RAG module but adds
simpler splitting strategies optimised for retrieval (no tree-sitter
dependency).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from forge_bot.retrieval.token_budget import CHARS_PER_TOKEN, estimate_tokens


@dataclass
class Chunk:
    """A text chunk with provenance metadata."""

    content: str
    source: str  # file path, diff header, or arbitrary label
    index: int  # ordinal position within the source
    start_line: int = 0
    end_line: int = 0
    token_count: int = field(default=0)

    def __post_init__(self) -> None:
        if self.token_count == 0:
            self.token_count = estimate_tokens(self.content)


def chunk_text(
    text: str,
    *,
    chunk_tokens: int = 512,
    overlap_tokens: int = 64,
    source: str = "",
) -> list[Chunk]:
    """Split *text* into token-sized chunks with overlap.

    Splits on line boundaries when possible to keep code readable.
    """
    if not text.strip():
        return []

    max_chars = chunk_tokens * CHARS_PER_TOKEN
    overlap_chars = overlap_tokens * CHARS_PER_TOKEN

    lines = text.split("\n")
    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_chars = 0
    start_line = 1

    for line_num, line in enumerate(lines, 1):
        line_len = len(line) + 1  # +1 for the newline

        if buf_chars + line_len > max_chars and buf:
            # Emit current buffer as a chunk.
            chunk_text_str = "\n".join(buf)
            chunks.append(
                Chunk(
                    content=chunk_text_str,
                    source=source,
                    index=len(chunks),
                    start_line=start_line,
                    end_line=start_line + len(buf) - 1,
                )
            )

            # Compute overlap: keep trailing lines up to overlap_chars.
            overlap_buf: list[str] = []
            overlap_len = 0
            for prev_line in reversed(buf):
                if overlap_len + len(prev_line) + 1 > overlap_chars:
                    break
                overlap_buf.insert(0, prev_line)
                overlap_len += len(prev_line) + 1

            buf = overlap_buf
            buf_chars = overlap_len
            start_line = line_num - len(buf)

        buf.append(line)
        buf_chars += line_len

    # Emit remaining buffer.
    if buf:
        chunk_text_str = "\n".join(buf)
        chunks.append(
            Chunk(
                content=chunk_text_str,
                source=source,
                index=len(chunks),
                start_line=start_line,
                end_line=start_line + len(buf) - 1,
            )
        )

    return chunks


def chunk_diff_by_file(diff_text: str) -> list[Chunk]:
    """Split a unified diff into one chunk per file.

    Each chunk's ``source`` is set to the file path from the diff header.
    Chunks that exceed a reasonable size are sub-chunked.
    """
    if not diff_text.strip():
        return []

    # Split on diff file headers.
    file_chunks: list[tuple[str, str]] = []  # (path, content)
    current_path = ""
    current_lines: list[str] = []

    for line in diff_text.split("\n"):
        if line.startswith("diff --git"):
            if current_path and current_lines:
                file_chunks.append((current_path, "\n".join(current_lines)))
            # Extract path from "diff --git a/path b/path"
            parts = line.split(" b/", 1)
            current_path = parts[1] if len(parts) > 1 else line
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_path and current_lines:
        file_chunks.append((current_path, "\n".join(current_lines)))

    chunks: list[Chunk] = []
    for path, content in file_chunks:
        token_count = estimate_tokens(content)
        if token_count <= 1000:
            # Small enough to be a single chunk.
            chunks.append(
                Chunk(
                    content=content,
                    source=path,
                    index=len(chunks),
                    token_count=token_count,
                )
            )
        else:
            # Sub-chunk large file diffs.
            sub_chunks = chunk_text(
                content,
                chunk_tokens=512,
                overlap_tokens=64,
                source=path,
            )
            for sc in sub_chunks:
                sc.index = len(chunks)
                chunks.append(sc)

    return chunks
