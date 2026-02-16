"""AST-aware chunking with fallback for unsupported languages."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger("forge_bot.rag.chunker")

# Approximate tokens = chars / 4
_CHARS_PER_TOKEN = 4

# Languages we attempt tree-sitter parsing for.
_TREE_SITTER_LANGUAGES = {"python", "javascript"}

# Extensions → language mapping for auto-detection.
_EXT_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "javascript",
    ".jsx": "javascript",
    ".tsx": "javascript",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".cpp": "c",
    ".h": "c",
    ".java": "java",
}


@dataclass
class CodeChunk:
    """A chunk of code with metadata."""

    content: str
    file_path: str
    language: str
    symbol_name: str | None
    start_line: int
    end_line: int
    token_count: int = field(default=0)

    def __post_init__(self) -> None:
        if self.token_count == 0:
            self.token_count = len(self.content) // _CHARS_PER_TOKEN


def detect_language(file_path: str) -> str:
    """Detect language from file extension."""
    for ext, lang in _EXT_TO_LANG.items():
        if file_path.endswith(ext):
            return lang
    return "text"


class Chunker:
    """Split source files into semantically meaningful chunks."""

    def __init__(
        self, chunk_size: int = 1500, overlap: int = 200,
    ) -> None:
        self._chunk_size = chunk_size  # in tokens
        self._overlap = overlap  # in tokens
        self._max_chars = chunk_size * _CHARS_PER_TOKEN
        self._overlap_chars = overlap * _CHARS_PER_TOKEN
        self._ts_available: bool | None = None

    def chunk_file(
        self, content: str, file_path: str, language: str = "",
    ) -> list[CodeChunk]:
        """Chunk a file into a list of CodeChunks.

        Tries AST-based chunking for supported languages, falls back
        to sliding-window text splitting.
        """
        if not content.strip():
            return []

        if not language:
            language = detect_language(file_path)

        # Try AST-based chunking for supported languages.
        if language in _TREE_SITTER_LANGUAGES:
            chunks = self._chunk_with_ast(content, file_path, language)
            if chunks:
                return chunks

        return self._chunk_with_fallback(content, file_path, language)

    def _chunk_with_ast(
        self, content: str, file_path: str, language: str,
    ) -> list[CodeChunk]:
        """Extract function/class chunks using tree-sitter."""
        if self._ts_available is False:
            return []

        try:
            return self._do_tree_sitter_parse(content, file_path, language)
        except Exception:
            if self._ts_available is None:
                logger.info(
                    "tree-sitter not available, using fallback chunking",
                )
                self._ts_available = False
            return []

    def _do_tree_sitter_parse(
        self, content: str, file_path: str, language: str,
    ) -> list[CodeChunk]:
        """Actual tree-sitter parsing logic (may raise ImportError)."""
        import tree_sitter  # noqa: F401

        if language == "python":
            import tree_sitter_python as ts_python

            lang = ts_python.language()
        elif language == "javascript":
            import tree_sitter_javascript as ts_javascript

            lang = ts_javascript.language()
        else:
            return []

        self._ts_available = True

        parser = tree_sitter.Parser(tree_sitter.Language(lang))
        tree = parser.parse(content.encode("utf-8"))
        root = tree.root_node

        # Node types that represent top-level definitions.
        if language == "python":
            target_types = {"function_definition", "class_definition"}
        else:
            target_types = {
                "function_declaration", "class_declaration",
                "export_statement",
            }

        chunks: list[CodeChunk] = []
        lines = content.split("\n")

        for child in root.children:
            if child.type in target_types:
                start = child.start_point[0]
                end = child.end_point[0]
                chunk_content = "\n".join(lines[start : end + 1])

                # Extract symbol name.
                symbol = None
                for sub in child.children:
                    if sub.type == "identifier" or sub.type == "name":
                        symbol = sub.text.decode("utf-8")
                        break

                # Split oversized chunks.
                if len(chunk_content) > self._max_chars:
                    sub_chunks = self._split_large_chunk(
                        chunk_content, file_path, language, symbol, start,
                    )
                    chunks.extend(sub_chunks)
                else:
                    chunks.append(
                        CodeChunk(
                            content=chunk_content,
                            file_path=file_path,
                            language=language,
                            symbol_name=symbol,
                            start_line=start + 1,
                            end_line=end + 1,
                        ),
                    )

        return chunks

    def _split_large_chunk(
        self,
        content: str,
        file_path: str,
        language: str,
        symbol: str | None,
        base_line: int,
    ) -> list[CodeChunk]:
        """Split an oversized AST chunk into smaller pieces."""
        lines = content.split("\n")
        chunks: list[CodeChunk] = []
        i = 0

        while i < len(lines):
            end = min(i + (self._max_chars // 80), len(lines))  # ~80 chars/line
            chunk_lines = lines[i:end]
            chunk_text = "\n".join(chunk_lines)

            if len(chunk_text) > self._max_chars:
                chunk_text = chunk_text[: self._max_chars]

            part_num = len(chunks) + 1
            chunks.append(
                CodeChunk(
                    content=chunk_text,
                    file_path=file_path,
                    language=language,
                    symbol_name=f"{symbol} (part {part_num})" if symbol else None,
                    start_line=base_line + i + 1,
                    end_line=base_line + end,
                ),
            )

            # Advance with overlap.
            overlap_lines = max(1, self._overlap_chars // 80)
            i = end - overlap_lines
            if i <= chunks[-1].start_line - base_line - 1:
                i = end  # Prevent infinite loop.

        return chunks

    def _chunk_with_fallback(
        self, content: str, file_path: str, language: str,
    ) -> list[CodeChunk]:
        """Split on double-newline boundaries with sliding window."""
        # Split into paragraphs first.
        paragraphs = content.split("\n\n")

        chunks: list[CodeChunk] = []
        current_text = ""
        current_start = 1

        for para in paragraphs:
            candidate = (current_text + "\n\n" + para).strip() if current_text else para

            if len(candidate) > self._max_chars and current_text:
                # Emit current chunk.
                end_line = current_start + current_text.count("\n")
                chunks.append(
                    CodeChunk(
                        content=current_text,
                        file_path=file_path,
                        language=language,
                        symbol_name=None,
                        start_line=current_start,
                        end_line=end_line,
                    ),
                )

                # Overlap: keep tail of current text.
                if self._overlap_chars > 0:
                    overlap_text = current_text[-self._overlap_chars :]
                    overlap_lines = overlap_text.count("\n")
                    current_start = end_line - overlap_lines
                    current_text = overlap_text + "\n\n" + para
                else:
                    current_start = end_line + 1
                    current_text = para
            else:
                current_text = candidate

        # Emit the last chunk.
        if current_text.strip():
            end_line = current_start + current_text.count("\n")
            chunks.append(
                CodeChunk(
                    content=current_text,
                    file_path=file_path,
                    language=language,
                    symbol_name=None,
                    start_line=current_start,
                    end_line=end_line,
                ),
            )

        return chunks
