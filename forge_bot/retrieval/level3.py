"""Level 3 -- Multi-hop reasoning agent.

The Level-2 scanner is registered as a callable tool.  A Level-3 agent
decomposes complex questions into sub-questions, calls Level-2 for each,
accumulates answers in memory, and synthesizes a final response.

This handles questions like "Does this PR introduce any inconsistency
with the patterns used in the existing codebase?"
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from forge_bot.clients.llm import LLMClient
from forge_bot.retrieval.level2 import Level2Scanner
from forge_bot.retrieval.token_budget import TokenBudget, truncate_to_tokens

logger = logging.getLogger("forge_bot.retrieval.level3")

_MAX_HOPS = 5
_MAX_TOTAL_TOOL_CALLS = 20

_AGENT_SYSTEM_PROMPT = """\
You are a multi-hop reasoning agent.  Your job is to answer complex \
questions by breaking them into simpler sub-questions and using the \
search_context tool to find answers.

PROCESS:
1. Decompose the main question into 1-{max_hops} focused sub-questions.
2. For each sub-question, call the search_context tool with the \
   sub-question and the name of the source to search.
3. After collecting answers, synthesize a final comprehensive response.

RULES:
- Only call tools when you need information you don't already have.
- Each tool call should ask a specific, focused question.
- When you have enough information, stop calling tools and provide \
  your final answer directly (without tool calls).
- Do NOT fabricate information.  If the context is insufficient, say so.

Available sources: {sources}
"""

_SEARCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_context",
        "description": (
            "Search a source for information relevant to a sub-question. "
            "The source will be scanned in parallel and relevant content "
            "extracted to answer your sub-question."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "A focused sub-question to search for.",
                },
                "source": {
                    "type": "string",
                    "description": (
                        "The name of the source to search (e.g. a file path or 'diff')."
                    ),
                },
            },
            "required": ["question", "source"],
        },
    },
}


@dataclass
class SourceDocument:
    """A named document that Level 3 can search via Level 2."""

    name: str
    content: str


@dataclass
class HopRecord:
    """Record of one sub-question and its answer."""

    question: str
    source: str
    answer: str


class Level3Agent:
    """Multi-hop reasoning agent that decomposes questions into sub-queries.

    Each hop creates a fresh ``Level2Scanner`` with its own ``TokenBudget``
    so that one hop's token consumption does not starve subsequent hops.
    """

    def __init__(
        self,
        llm: LLMClient,
        budget_factory: Callable[[], TokenBudget],
        *,
        max_parallel: int = 10,
        max_hops: int = _MAX_HOPS,
    ) -> None:
        self._llm = llm
        self._budget_factory = budget_factory
        self._max_parallel = max_parallel
        self._max_hops = max_hops

    async def answer(
        self,
        question: str,
        sources: list[SourceDocument],
    ) -> str:
        """Decompose *question*, search *sources*, and synthesize an answer.

        The agent controls the loop: it decides which sub-questions to
        ask and which sources to search.  Level 2 does the actual
        content scanning.
        """
        if not sources:
            return "No sources available to search."

        source_map = {s.name: s for s in sources}
        source_names = [s.name for s in sources]

        system_prompt = _AGENT_SYSTEM_PROMPT.format(
            max_hops=self._max_hops,
            sources=", ".join(source_names),
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        hops: list[HopRecord] = []
        total_tool_calls = 0

        for hop_num in range(self._max_hops):
            try:
                response = await self._llm.chat_with_tools(
                    messages=messages,
                    tools=[_SEARCH_TOOL_SCHEMA],
                    max_tokens=1024,
                    temperature=0.2,
                )
            except Exception:
                logger.exception("Level-3 LLM call failed at hop %d", hop_num)
                break

            message = response.choices[0].message

            # If no tool calls, the agent is done reasoning.
            if not message.tool_calls:
                final = message.content or ""
                if final:
                    return self._format_final(final, hops)
                break

            # Append ONE assistant message with all tool_calls (OpenAI chat format).
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in message.tool_calls
                    ],
                }
            )

            # Process tool calls.
            for tc in message.tool_calls:
                if tc.function.name != "search_context":
                    continue

                # Enforce global cap on total tool invocations.
                if total_tool_calls >= _MAX_TOTAL_TOOL_CALLS:
                    logger.warning(
                        "Level-3 global tool-call cap (%d) reached; skipping remaining calls",
                        _MAX_TOTAL_TOOL_CALLS,
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": "Tool call cap reached; no more tool calls allowed.",
                        }
                    )
                    continue

                total_tool_calls += 1

                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                sub_question = args.get("question", question)
                source_name = args.get("source", source_names[0])

                # Find the source document.
                source_doc = source_map.get(source_name)
                if not source_doc:
                    # Try fuzzy match.
                    for name, doc in source_map.items():
                        if source_name.lower() in name.lower():
                            source_doc = doc
                            break

                if not source_doc:
                    tool_result = f"Source '{source_name}' not found."
                else:
                    logger.info(
                        "Level-3 hop %d: searching '%s' for: %s",
                        hop_num,
                        source_name,
                        sub_question[:80],
                    )
                    # Fresh budget per hop so consumption doesn't carry over.
                    hop_budget = self._budget_factory()
                    hop_scanner = Level2Scanner(
                        self._llm, hop_budget, max_parallel=self._max_parallel
                    )
                    tool_result = await hop_scanner.retrieve_and_answer(
                        sub_question,
                        source_doc.content,
                        source=source_name,
                    )

                hops.append(
                    HopRecord(
                        question=sub_question,
                        source=source_name,
                        answer=tool_result,
                    )
                )

                # Append tool result message for this call.
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": truncate_to_tokens(tool_result, 1500),
                    }
                )

        # Exhausted hops -- synthesize from accumulated answers.
        if hops:
            return await self._synthesize(question, hops)

        return "Unable to find sufficient information to answer the question."

    async def _synthesize(
        self,
        question: str,
        hops: list[HopRecord],
    ) -> str:
        """Synthesize a final answer from accumulated hop results."""
        context_parts = []
        for i, hop in enumerate(hops, 1):
            context_parts.append(
                f"Sub-question {i} (searched '{hop.source}'): {hop.question}\nFinding: {hop.answer}"
            )

        context = "\n\n".join(context_parts)
        user_msg = (
            f"Based on these findings, provide a comprehensive answer.\n\n"
            f"FINDINGS:\n{context}\n\n"
            f"ORIGINAL QUESTION:\n{question}"
        )

        try:
            return await self._llm.chat(
                "Synthesize a clear, comprehensive answer from the "
                "provided findings. Do not fabricate information.",
                user_msg,
                max_tokens=2048,
                temperature=0.2,
            )
        except Exception:
            logger.exception("Level-3 synthesis failed")
            # Return raw findings as fallback.
            return self._format_final("", hops)

    @staticmethod
    def _format_final(text: str, hops: list[HopRecord]) -> str:
        """Append hop records to the final answer for traceability."""
        if not hops:
            return text

        parts = [text] if text else []
        if hops:
            parts.append("\n\n<details><summary>Retrieval trace</summary>\n")
            for i, hop in enumerate(hops, 1):
                parts.append(f"**Hop {i}** -- searched `{hop.source}` for: _{hop.question}_\n")
            parts.append("</details>")

        return "\n".join(parts)
