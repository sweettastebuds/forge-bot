"""RetrievalTool -- self-contained codebase search via the workspace container.

Registered in the tool-calling loop so the LLM can search the cloned
repository using the three-level retrieval hierarchy.  The tool discovers
and reads files from the workspace on its own -- the LLM only needs to
provide a query.
"""

from __future__ import annotations

import logging
import posixpath
import re
import shlex
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from forge_bot.retrieval.pipeline import SmartRetriever
from forge_bot.retrieval.token_budget import estimate_tokens
from forge_bot.tools.base import BaseTool, ToolParameter, ToolResult

if TYPE_CHECKING:
    from forge_bot.container.manager import ContainerManager

logger = logging.getLogger("forge_bot.retrieval.tool")

# Paths excluded from file discovery (find predicates).
_FIND_EXCLUDES = (
    "! -path '*/.git/*' ! -path '*/node_modules/*' "
    "! -path '*/__pycache__/*' ! -path '*/.venv/*' "
    "! -name '*.pyc' ! -name '*.jpg' ! -name '*.png' "
    "! -name '*.gif' ! -name '*.ico' ! -name '*.svg' "
    "! -name '*.woff' ! -name '*.woff2' ! -name '*.ttf' "
    "! -name '*.eot' ! -name '*.zip' ! -name '*.tar' "
    "! -name '*.gz' ! -name '*.so' ! -name '*.o' "
    "! -name '*.class' ! -name '*.jar' ! -name '*.bin' "
    "! -name '*.lock'"
)

# Same exclusions expressed as grep flags (--exclude / --exclude-dir).
_GREP_EXCLUDES = (
    "--exclude-dir=.git --exclude-dir=node_modules "
    "--exclude-dir=__pycache__ --exclude-dir=.venv "
    "--exclude='*.pyc' --exclude='*.jpg' --exclude='*.png' "
    "--exclude='*.gif' --exclude='*.ico' --exclude='*.svg' "
    "--exclude='*.woff' --exclude='*.woff2' --exclude='*.ttf' "
    "--exclude='*.eot' --exclude='*.zip' --exclude='*.tar' "
    "--exclude='*.gz' --exclude='*.so' --exclude='*.o' "
    "--exclude='*.class' --exclude='*.jar' --exclude='*.bin' "
    "--exclude='*.lock'"
)

# Max files to read when the repo is large and grep narrows the set.
_MAX_FILES = 30
# Cap on total characters read from the workspace.
_MAX_CONTENT_CHARS = 120_000
# Threshold: if the repo has fewer files than this, read them all.
_SMALL_REPO_THRESHOLD = 50


class RetrievalTool(BaseTool):
    """Deep search and analysis of the repository codebase.

    The tool discovers relevant files from the cloned workspace, reads
    their contents, and applies the multi-level retrieval hierarchy to
    produce an answer grounded in actual code.  The LLM only needs to
    provide a natural-language query.
    """

    name = "smart_search"
    description = (
        "Deep search and analysis of the repository codebase. "
        "Finds and reads relevant source files from the cloned repository, "
        "then analyzes them to answer your query. Returns a synthesized "
        "answer grounded in actual file contents.\n\n"
        "When to use this tool:\n"
        "- Understanding how a feature, module, or function is implemented\n"
        "- Finding where something is defined or used across the codebase\n"
        "- Answering questions that require reading and cross-referencing "
        "multiple files\n"
        "- Analyzing architecture, patterns, or dependencies\n\n"
        "When NOT to use (prefer exec with grep/find):\n"
        "- Simple pattern matching or counting occurrences\n"
        "- Listing files or checking directory structure\n"
        "- Running tests or build commands\n\n"
        "Important: The repository must be cloned to /workspace first."
    )
    parameters = [
        ToolParameter(
            name="query",
            type="string",
            description="What you want to find or understand in the codebase.",
        ),
        ToolParameter(
            name="path",
            type="string",
            description=(
                "Limit search to a specific file or directory relative to "
                "the repo root (e.g. 'src/auth', 'main.py'). "
                "Omit to search the entire repository."
            ),
            required=False,
        ),
    ]

    def __init__(
        self,
        retriever: SmartRetriever,
        container: ContainerManager,
    ) -> None:
        self._retriever = retriever
        self._cm = container

    # -- public API (BaseTool) -----------------------------------------------

    async def execute(self, **kwargs: object) -> ToolResult:
        raw_query = kwargs.get("query")
        if raw_query is None or not isinstance(raw_query, str):
            return ToolResult(
                tool_name=self.name,
                success=False,
                content="Missing required parameter: query",
            )
        query = raw_query.strip()
        path = str(kwargs.get("path", "")).strip().lstrip("/")

        if not query:
            return ToolResult(
                tool_name=self.name,
                success=False,
                content="Missing required parameter: query",
            )

        # Validate path to prevent directory traversal outside /workspace.
        if path:
            resolved = PurePosixPath("/workspace", path).as_posix()
            resolved = posixpath.normpath(resolved)
            if not resolved.startswith("/workspace"):
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    content="Invalid path: escapes the repository root.",
                )

        # Verify the repo has been cloned.
        check = await self._cm.exec(
            "test -d /workspace/.git && echo ok",
            timeout=5,
        )
        if "ok" not in check.stdout:
            return ToolResult(
                tool_name=self.name,
                success=False,
                content=(
                    "Repository not cloned yet. Clone it first using exec: "
                    "git clone --depth=1 '<clone_url>' /workspace"
                ),
            )

        # Gather file contents from the workspace.
        try:
            combined, sources = await self._gather_content(query, path)
        except Exception:
            logger.exception("Failed to gather content from workspace")
            return ToolResult(
                tool_name=self.name,
                success=False,
                content="Failed to read files from the workspace.",
            )

        if not combined:
            return ToolResult(
                tool_name=self.name,
                success=True,
                content="No relevant files found for your query.",
            )

        # Auto-select retrieval level based on content size.
        try:
            total_tokens = estimate_tokens(combined)
            budget = self._retriever._make_budget()

            if len(sources) > 5 and total_tokens > budget.available:
                # Large multi-file search -> Level 3 multi-hop.
                logger.info(
                    "smart_search: Level 3 (%d files, ~%d tokens)",
                    len(sources),
                    total_tokens,
                )
                answer = await self._retriever.answer_complex(query, sources)
            else:
                # Fits in budget -> Level 2 thorough scan.
                logger.info(
                    "smart_search: Level 2 (%d files, ~%d tokens)",
                    len(sources),
                    total_tokens,
                )
                answer = await self._retriever.scan_and_answer(
                    query,
                    combined,
                    source=path or "repository",
                )

            return ToolResult(
                tool_name=self.name,
                success=True,
                content=answer,
            )

        except Exception:
            logger.exception("Smart search retrieval failed")
            return ToolResult(
                tool_name=self.name,
                success=False,
                content="Search failed: internal retrieval error.",
            )

    # -- internals -----------------------------------------------------------

    async def _gather_content(
        self,
        query: str,
        path: str,
    ) -> tuple[str, dict[str, str]]:
        """Discover and read relevant files from the workspace.

        Returns ``(combined_text, sources_dict)`` where *combined_text* is
        all files concatenated with ``=== path ===`` headers, and
        *sources_dict* maps relative paths to their contents.
        """
        root = f"/workspace/{path}" if path else "/workspace"

        # 1. Discover files.
        find_result = await self._cm.exec(
            f"find {shlex.quote(root)} -type f {_FIND_EXCLUDES} | head -500",
            timeout=10,
        )
        all_files = [f.strip() for f in find_result.stdout.strip().split("\n") if f.strip()]

        if not all_files:
            return "", {}

        # 2. Narrow to relevant files if the repo is large.
        if len(all_files) > _SMALL_REPO_THRESHOLD:
            keywords = self._extract_keywords(query)
            if keywords:
                pattern = "|".join(re.escape(kw) for kw in keywords)
                grep_result = await self._cm.exec(
                    f"grep -rl -i -E {shlex.quote(pattern)} "
                    f"{_GREP_EXCLUDES} "
                    f"{shlex.quote(root)} "
                    f"2>/dev/null | head -{_MAX_FILES}",
                    timeout=15,
                )
                narrowed = [f.strip() for f in grep_result.stdout.strip().split("\n") if f.strip()]
                all_files = narrowed or all_files[:_MAX_FILES]
        else:
            all_files = all_files[:_MAX_FILES]

        # 3. Batch-read files with headers in a single exec call.
        escaped = " ".join(shlex.quote(f) for f in all_files)
        read_cmd = (
            f"for f in {escaped}; do "
            f"printf '=== %s ===\\n' \"$f\"; "
            f'cat "$f" 2>/dev/null; '
            f"printf '\\n'; "
            f"done"
        )
        read_result = await self._cm.exec(read_cmd, timeout=30)

        if read_result.exit_code != 0 or not read_result.stdout.strip():
            return "", {}

        # 4. Parse output back into per-file sources.
        raw = read_result.stdout
        if len(raw) > _MAX_CONTENT_CHARS:
            raw = raw[:_MAX_CONTENT_CHARS]

        sources: dict[str, str] = {}
        parts = re.split(r"^=== (.+?) ===$", raw, flags=re.MULTILINE)
        # parts: ['', path1, content1, path2, content2, ...]
        for i in range(1, len(parts), 2):
            abs_path = parts[i].strip()
            content = parts[i + 1] if i + 1 < len(parts) else ""
            rel = (
                abs_path.replace("/workspace/", "", 1)
                if abs_path.startswith("/workspace/")
                else abs_path
            )
            sources[rel] = content.strip()

        combined = "\n\n".join(f"=== {p} ===\n{c}" for p, c in sources.items())
        return combined, sources

    @staticmethod
    def _extract_keywords(query: str) -> list[str]:
        """Extract meaningful keywords from a natural-language query."""
        stop = {
            "a",
            "an",
            "the",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "can",
            "shall",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "through",
            "during",
            "before",
            "after",
            "and",
            "but",
            "or",
            "nor",
            "not",
            "so",
            "yet",
            "both",
            "either",
            "neither",
            "each",
            "every",
            "all",
            "any",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "only",
            "own",
            "same",
            "than",
            "too",
            "very",
            "just",
            "about",
            "above",
            "below",
            "between",
            "how",
            "what",
            "where",
            "when",
            "which",
            "who",
            "whom",
            "why",
            "this",
            "that",
            "these",
            "those",
            "i",
            "me",
            "my",
            "we",
            "our",
            "you",
            "your",
            "he",
            "him",
            "his",
            "she",
            "her",
            "it",
            "its",
            "they",
            "them",
            "their",
            "if",
            "then",
            "else",
            "while",
            "until",
            "up",
            "down",
            "out",
            "off",
            "over",
            "under",
            "again",
            "further",
            "there",
            "here",
            "find",
            "show",
            "tell",
            "explain",
            "describe",
            "get",
            "use",
            "using",
            "used",
        }
        words = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", query)
        return [w for w in words if w.lower() not in stop and len(w) > 1][:8]
