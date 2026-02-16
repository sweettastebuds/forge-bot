"""Handler for issue_comment webhook events (@mention replies).

The LLM drives its own context gathering via tools (exec, api_call,
search_api, todo).  The handler manages the tool-calling loop, status
comment updates, container lifecycle, and verification checks.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from forge_bot.container.manager import ContainerManager
from forge_bot.handlers.base import BaseHandler
from forge_bot.handlers.verification import (
    ProgressTracker,
    check_hallucination,
    check_relevance,
    verify_response,
)
from forge_bot.models import IssueCommentEvent
from forge_bot.status.formatter import ToolCallRecord, abbreviate
from forge_bot.status.manager import StatusCommentManager
from forge_bot.tools.api_call import ApiCallTool
from forge_bot.tools.base import ToolResult
from forge_bot.tools.exec_tool import ExecTool
from forge_bot.tools.registry import ToolRegistry
from forge_bot.tools.search_api import SearchApiTool
from forge_bot.tools.todo import TodoTool

logger = logging.getLogger("forge_bot.handlers.issue_comment")

_MAX_TOOL_ROUNDS = 10
_WARNING_BANNER = (
    "> :warning: **This response may contain inaccuracies — please verify.**\n\n"
)


class IssueCommentHandler(BaseHandler):
    """Respond to @mentions in issue/PR comments with LLM-generated answers.

    Flow:
    1. Post initial status comment
    2. Create workspace container (clone repo)
    3. Register tools and enter tool-calling loop with verification
    4. Post final response as separate comment
    5. Finalize status and destroy container
    """

    async def handle(self, event: IssueCommentEvent) -> None:
        owner, repo = event.repository.full_name.split("/", 1)
        issue_num = event.issue.number
        default_branch = event.repository.default_branch
        clone_url = event.repository.clone_url

        logger.info(
            "Handling @mention on %s#%d by %s",
            event.repository.full_name,
            issue_num,
            event.sender.login,
        )

        # 1. Post status comment
        status = StatusCommentManager(self.api, owner, repo, issue_num)
        try:
            await status.post_initial_status()
        except Exception:
            logger.warning("Failed to post initial status comment", exc_info=True)

        # 2. Create workspace container
        container: ContainerManager | None = None
        try:
            await status.update_phase("Cloning repository...")
            container = ContainerManager(
                self.settings,
                clone_url,
                default_branch,
                token=self.settings.forge_api_token,
                network_enabled=self.settings.container_network_enabled,
            )
            await container.create()
        except Exception:
            logger.exception("Failed to create workspace container")
            await status.update_phase("Error: container creation failed")
            await self._post_error_response(
                status,
                "I couldn't set up a workspace to analyze your request. "
                "Please try again later.",
            )
            if container:
                await container.destroy()
            return

        try:
            # 3. Register tools
            registry = ToolRegistry()
            registry.register(SearchApiTool(self.api))
            registry.register(ApiCallTool(self.api, owner=owner, repo=repo))
            registry.register(ExecTool(container))
            todo_tool = TodoTool(status)
            registry.register(todo_tool)

            # 4. Tool-calling loop with verification
            await status.update_phase("Thinking...")
            reply, all_tool_results = await self._tool_loop(
                event=event,
                registry=registry,
                status=status,
            )

            # 5. Pre-post verification
            reply = await self._verify_and_maybe_retry(
                reply, all_tool_results, event, status
            )

            # 6. Post response
            try:
                await status.post_response(reply)
                logger.info(
                    "Posted reply on %s#%d", event.repository.full_name, issue_num
                )
            except Exception:
                logger.exception(
                    "Failed to post response on %s#%d",
                    event.repository.full_name,
                    issue_num,
                )

            # 7. Finalize status
            await status.finalize_status("Done")

        except Exception:
            logger.exception(
                "Error processing %s#%d", event.repository.full_name, issue_num
            )
            await self._post_error_response(
                status,
                "Sorry, I encountered an error while processing your request.",
            )
        finally:
            await container.destroy()

    # -- Tool-calling loop --

    async def _tool_loop(
        self,
        *,
        event: IssueCommentEvent,
        registry: ToolRegistry,
        status: StatusCommentManager,
    ) -> tuple[str, list[tuple[str, ToolResult]]]:
        """Run the LLM with tools, executing calls and verifying each round.

        Returns (final_reply_text, list_of_all_tool_results).
        """
        user_question = event.comment.body
        system_prompt = self._build_system_prompt(event, registry)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question},
        ]

        progress = ProgressTracker()
        all_tool_results: list[tuple[str, ToolResult]] = []
        error_reply = (
            "Sorry, I encountered an error while generating a "
            "response. Please try again later."
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
                # Fallback: try simple chat without tools
                try:
                    response = await self.llm.chat(
                        system_prompt,
                        user_question,
                    )
                    return str(response), all_tool_results
                except Exception:
                    logger.exception("LLM call failed (round %d)", round_num)
                    return error_reply, all_tool_results

            # Extract tool calls from response
            tool_calls = self._extract_tool_calls(response)

            if not tool_calls:
                final_text = self._extract_text(response)
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
                if not check_relevance(tool_name, tool_args, user_question):
                    logger.info("Blocked irrelevant tool call: %s", tool_name)
                    continue

                # Progress check: skip duplicates
                if progress.is_duplicate(tool_name, args_key):
                    logger.info("Skipping duplicate tool call: %s", args_key)
                    messages.append({
                        "role": "assistant",
                        "content": f"[Skipped duplicate call to {tool_name}]",
                    })
                    continue

                progress.record(tool_name, args_key)
                had_new_calls = True

                # Execute
                await status.update_phase(f"Running {tool_name}...")
                start = time.monotonic()
                result = await registry.execute(tool_name, tool_args)
                duration = time.monotonic() - start

                round_results.append((tool_name, result))
                all_tool_results.append((tool_name, result))

                # Record in status
                await status.record_tool_call(ToolCallRecord(
                    tool_name=tool_name,
                    arguments_summary=abbreviate(
                        json.dumps(tool_args), 60
                    ),
                    result_summary=abbreviate(result.content, 100),
                    success=result.success,
                    duration_seconds=round(duration, 2),
                ))

                # Add result to messages for next LLM round
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": f"call_{round_num}_{tool_name}",
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(tool_args),
                        },
                    }],
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": f"call_{round_num}_{tool_name}",
                    "content": result.content,
                })

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
                        messages.append({
                            "role": "user",
                            "content": (
                                f"CORRECTION: {mismatch}. "
                                "Please re-check the tool output and correct "
                                "your response."
                            ),
                        })

        # Exhausted rounds — try to extract last assistant message.
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                return msg["content"], all_tool_results

        return fallback_reply, all_tool_results

    # -- Pre-post verification --

    async def _verify_and_maybe_retry(
        self,
        reply: str,
        all_tool_results: list[tuple[str, ToolResult]],
        event: IssueCommentEvent,
        status: StatusCommentManager,
    ) -> str:
        """Run pre-post verification. Retry once, then warn if still failing."""
        result = verify_response(reply, all_tool_results)
        if result.passed:
            return reply

        logger.warning(
            "Pre-post verification failed: %s", "; ".join(result.failures)
        )

        # Retry: re-prompt with specific failures
        await status.update_phase("Verifying response...")
        correction = (
            "Before posting your response, I found these issues:\n"
            + "\n".join(f"- {f}" for f in result.failures)
            + "\nPlease correct your response."
        )

        try:
            retry = await self.llm.chat(
                f"Original response:\n{reply}\n\n{correction}",
                event.comment.body,
            )
            retry_result = verify_response(str(retry), all_tool_results)
            if retry_result.passed:
                return str(retry)
        except Exception:
            logger.warning("Retry LLM call failed", exc_info=True)

        # Still failing — post with warning banner
        return _WARNING_BANNER + reply

    # -- Helpers --

    def _build_system_prompt(
        self,
        event: IssueCommentEvent,
        registry: ToolRegistry,
    ) -> str:
        """Build the system prompt with context and tool descriptions."""
        return self.render_template(
            "issue_respond.j2",
            repo_full_name=event.repository.full_name,
            issue_number=event.issue.number,
            issue_title=event.issue.title,
            bot_username=self.bot_username,
            tool_descriptions=registry.prompt_text(),
        )

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
                    calls.append({
                        "name": tc.function.name,
                        "arguments": args,
                    })
                return calls

        # Prompt mode: parse ```tool blocks
        text = str(response)
        tool_block_re = re.compile(
            r"```tool\s*\n(.*?)\n```", re.DOTALL
        )
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
        self, status: StatusCommentManager, message: str
    ) -> None:
        """Post an error message and finalize the status comment."""
        try:
            await status.post_response(message)
            await status.finalize_status("Error")
        except Exception:
            logger.warning("Failed to post error response", exc_info=True)
