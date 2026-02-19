"""RetrievalTool — exposes the smart retrieval hierarchy as a tool.

Registered in the tool-calling loop so the LLM can search repository
content, diffs, and files using the three-level hierarchy.
"""

from __future__ import annotations

import logging

from forge_bot.retrieval.pipeline import SmartRetriever
from forge_bot.tools.base import BaseTool, ToolParameter, ToolResult

logger = logging.getLogger("forge_bot.retrieval.tool")


class RetrievalTool(BaseTool):
    """Search repository content using smart multi-level retrieval.

    The tool supports three modes:

    * ``keyword`` — Level 1 BM25 keyword search (fast).
    * ``thorough`` — Level 2 parallel scan + answer (comprehensive).
    * ``multi_hop`` — Level 3 decomposition for complex questions.
    """

    name = "smart_search"
    description = (
        "Search code, diffs, or documents using multi-level retrieval. "
        "Use 'keyword' mode for quick lookups, 'thorough' for complete "
        "analysis, or 'multi_hop' for complex questions spanning multiple "
        "files.  Returns relevant content or a synthesized answer."
    )
    parameters = [
        ToolParameter(
            name="question",
            type="string",
            description="The question or search query.",
        ),
        ToolParameter(
            name="content",
            type="string",
            description=(
                "The text content to search (code, diff, documentation). "
                "Provide the raw text to analyze."
            ),
        ),
        ToolParameter(
            name="mode",
            type="string",
            description=(
                "Retrieval mode: 'keyword' (fast BM25), 'thorough' "
                "(parallel scan), or 'multi_hop' (decomposed reasoning). "
                "Default: 'thorough'."
            ),
            required=False,
        ),
        ToolParameter(
            name="source_name",
            type="string",
            description=(
                "Label for the content source (e.g. file path). Used for traceability in results."
            ),
            required=False,
        ),
    ]

    def __init__(self, retriever: SmartRetriever) -> None:
        self._retriever = retriever

    async def execute(self, **kwargs: object) -> ToolResult:
        question = str(kwargs.get("question", ""))
        content = str(kwargs.get("content", ""))
        mode = str(kwargs.get("mode", "thorough")).lower()
        source_name = str(kwargs.get("source_name", ""))

        if not question:
            return ToolResult(
                tool_name=self.name,
                success=False,
                content="Missing required parameter: question",
            )

        if not content:
            return ToolResult(
                tool_name=self.name,
                success=False,
                content="Missing required parameter: content",
            )

        try:
            if mode == "keyword":
                result = await self._retriever.search_text(
                    question,
                    content,
                    source=source_name,
                )
                text = "\n---\n".join(c.content for c in result.chunks)
                if not text:
                    text = "No relevant results found."
                return ToolResult(
                    tool_name=self.name,
                    success=True,
                    content=text,
                )

            elif mode == "multi_hop":
                answer = await self._retriever.answer_complex(
                    question,
                    {source_name or "content": content},
                )
                return ToolResult(
                    tool_name=self.name,
                    success=True,
                    content=answer,
                )

            else:  # thorough (default)
                answer = await self._retriever.scan_and_answer(
                    question,
                    content,
                    source=source_name,
                )
                return ToolResult(
                    tool_name=self.name,
                    success=True,
                    content=answer,
                )

        except Exception:
            logger.exception("Smart search failed (mode=%s)", mode)
            return ToolResult(
                tool_name=self.name,
                success=False,
                content=f"Search failed: internal error in {mode} mode.",
            )
