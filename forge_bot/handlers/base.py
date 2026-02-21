"""Abstract base class for all webhook event handlers.

Provides shared infrastructure:
- Template rendering
- Tool-calling loop with verification
- Error handling helpers
"""

from __future__ import annotations

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import jinja2

from forge_bot.api.client import GenericForgeClient
from forge_bot.clients.llm import LLMClient
from forge_bot.config import Settings
from forge_bot.handlers.verification import (
    ProgressTracker,
    check_hallucination,
    check_relevance,
    verify_response,
)
from forge_bot.status.formatter import ToolCallRecord, abbreviate
from forge_bot.status.manager import StatusCommentManager
from forge_bot.tools.base import ToolResult
from forge_bot.tools.registry import ToolRegistry

logger = logging.getLogger("forge_bot.handlers")

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
_template_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(str(_PROMPTS_DIR)),
    autoescape=False,
    keep_trailing_newline=True,
)

_MAX_TOOL_ROUNDS = 10
_WARNING_BANNER = "> :warning: **This response may contain inaccuracies — please verify.**\n\n"


class BaseHandler(ABC):
    """Base class injecting shared dependencies into every handler."""

    def __init__(
        self,
        api_client: GenericForgeClient,
        llm_client: LLMClient,
        settings: Settings,
        bot_username: str,
    ) -> None:
        self.api = api_client
        self.llm = llm_client
        self.settings = settings
        self.bot_username = bot_username

    def render_template(self, template_name: str, **kwargs: object) -> str:
        """Render a Jinja2 prompt template from the prompts/ directory."""
        template = _template_env.get_template(template_name)
        return template.render(**kwargs)

    @abstractmethod
    async def handle(self, event: object) -> None:
        """Process a webhook event. Subclasses must implement."""

    # -- Tool-calling loop ---------------------------------------------------

    async def _tool_loop(
        self,
        *,
        system_prompt: str,
        user_message: str,
        registry: ToolRegistry,
        status: StatusCommentManager,
    ) -> tuple[str, list[tuple[str, ToolResult]]]:
        """Run the LLM with tools, executing calls and verifying each round.

        Returns (final_reply_text, list_of_all_tool_results).
        """
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        progress = ProgressTracker()
        all_tool_results: list[tuple[str, ToolResult]] = []
        error_reply = (
            "Sorry, I encountered an error while generating a response. "
            "Please try again later."
        )
        fallback_reply = (
            "I looked into it but wasn't able to form a complete answer. "
            "Could you provide more details?"
        )

        for round_num in range(_MAX_TOOL_ROUNDS):
            if progress.is_stuck():
                logger.info("LLM stuck after %d rounds, forcing exit", round_num)
                break

            # Call LLM
            try:
                response = await self.llm.chat_with_tools(
                    messages=messages,
                    tools=registry.openai_schemas(),
                )
            except Exception:
                logger.exception(
                    "chat_with_tools failed (round %d), retrying without tools",
                    round_num,
                )
                try:
                    response = await self.llm.chat(system_prompt, user_message)
                    return str(response), all_tool_results
                except Exception:
                    logger.exception(
                        "LLM chat fallback also failed (round %d)", round_num,
                    )
                    return error_reply, all_tool_results

            # Extract tool calls from response
            tool_calls = self._extract_tool_calls(response)

            if not tool_calls:
                final_text = self._extract_text(response)
                logger.info(
                    "Round %d: LLM returned text response (%d chars)",
                    round_num,
                    len(final_text),
                )
                if final_text:
                    return final_text, all_tool_results
                break

            # Execute tool calls with verification
            had_new_calls = False
            round_results: list[tuple[str, ToolResult]] = []

            for call in tool_calls:
                tool_name = call.get("name", "")
                tool_args = call.get("arguments", {})
                if isinstance(tool_args, str):
                    try:
                        tool_args = json.loads(tool_args)
                    except json.JSONDecodeError:
                        tool_args = {"raw": tool_args}

                args_key = f"{tool_name}:{json.dumps(tool_args, sort_keys=True)}"

                # Relevance check
                if not check_relevance(tool_name, tool_args, user_message):
                    logger.info("Blocked irrelevant tool call: %s", tool_name)
                    continue

                # Progress check: skip duplicates
                if progress.is_duplicate(tool_name, args_key):
                    logger.info("Skipping duplicate tool call: %s", args_key)
                    messages.append(
                        {
                            "role": "assistant",
                            "content": f"[Skipped duplicate call to {tool_name}]",
                        }
                    )
                    continue

                progress.record(tool_name, args_key)
                had_new_calls = True

                # Execute
                logger.info(
                    "Round %d: executing %s(%s)",
                    round_num,
                    tool_name,
                    abbreviate(json.dumps(tool_args), 80),
                )
                await status.update_phase(f"Running {tool_name}...")
                start = time.monotonic()
                result = await registry.execute(tool_name, tool_args)
                duration = time.monotonic() - start

                round_results.append((tool_name, result))
                all_tool_results.append((tool_name, result))

                # Record in status
                await status.record_tool_call(
                    ToolCallRecord(
                        tool_name=tool_name,
                        arguments_summary=abbreviate(json.dumps(tool_args), 60),
                        result_summary=abbreviate(result.content, 100),
                        success=result.success,
                        duration_seconds=round(duration, 2),
                    )
                )

                # Add result to messages for next LLM round
                messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": f"call_{round_num}_{tool_name}",
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": json.dumps(tool_args),
                                },
                            }
                        ],
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": f"call_{round_num}_{tool_name}",
                        "content": result.content,
                    }
                )

            progress.record_round(had_new_calls)

            # Hallucination check after this round
            if round_results:
                last_assistant = ""
                for msg in reversed(messages):
                    if msg.get("role") == "assistant" and msg.get("content"):
                        last_assistant = msg["content"]
                        break

                if last_assistant:
                    mismatches = check_hallucination(last_assistant, round_results)
                    for mismatch in mismatches:
                        logger.warning("Hallucination detected: %s", mismatch)
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    f"CORRECTION: {mismatch}. "
                                    "Please re-check the tool output and correct "
                                    "your response."
                                ),
                            }
                        )

        # Exhausted rounds — try to extract last assistant message.
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                return msg["content"], all_tool_results

        return fallback_reply, all_tool_results

    # -- Pre-post verification -----------------------------------------------

    async def _verify_and_maybe_retry(
        self,
        reply: str,
        all_tool_results: list[tuple[str, ToolResult]],
        status: StatusCommentManager,
    ) -> str:
        """Run pre-post verification. Retry once, then warn if still failing."""
        result = verify_response(reply, all_tool_results)
        if result.passed:
            return reply

        logger.warning(
            "Pre-post verification failed: %s", "; ".join(result.failures),
        )

        await status.update_phase("Verifying response...")
        correction = (
            "Before posting your response, I found these issues:\n"
            + "\n".join(f"- {f}" for f in result.failures)
            + "\nPlease correct your response."
        )

        try:
            retry = await self.llm.chat(
                "You are correcting a response before it is posted. "
                "Fix the issues listed below and return the corrected response.",
                f"Original response:\n{reply}\n\n{correction}",
            )
            retry_result = verify_response(str(retry), all_tool_results)
            if retry_result.passed:
                return str(retry)
        except Exception:
            logger.warning("Retry LLM call failed", exc_info=True)

        return _WARNING_BANNER + reply

    # -- Response helpers ----------------------------------------------------

    @staticmethod
    def _extract_tool_calls(response: Any) -> list[dict]:
        """Extract tool calls from an LLM response.

        Handles both OpenAI native tool_calls and prompt-mode ```tool blocks.
        """
        if hasattr(response, "choices"):
            message = response.choices[0].message
            if hasattr(message, "tool_calls") and message.tool_calls:
                calls = []
                for tc in message.tool_calls:
                    args = tc.function.arguments
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {"raw": args}
                    calls.append(
                        {
                            "name": tc.function.name,
                            "arguments": args,
                        }
                    )
                return calls

        # Prompt mode: parse ```tool blocks from content text
        if hasattr(response, "choices"):
            text = response.choices[0].message.content or ""
        else:
            text = str(response)
        tool_block_re = re.compile(r"```tool\s*\n(.*?)\n```", re.DOTALL)
        blocks = tool_block_re.findall(text)
        calls = []
        for block in blocks:
            try:
                data = json.loads(block.strip())
                if "name" in data:
                    calls.append(data)
            except json.JSONDecodeError:
                continue
        return calls

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Extract plain text content from an LLM response."""
        if hasattr(response, "choices"):
            return response.choices[0].message.content or ""
        return str(response)

    async def _post_error_response(
        self,
        status: StatusCommentManager,
        message: str,
    ) -> None:
        """Post an error message and finalize the status comment."""
        try:
            await status.post_response(message)
            await status.finalize_status("Error")
        except Exception:
            logger.warning("Failed to post error response", exc_info=True)
