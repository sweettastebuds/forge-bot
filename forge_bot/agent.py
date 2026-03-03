"""Core agent loop: LLM + shell execution in a container.

Implements the Qwen-Agent-inspired "4k-Agent" pattern:
- ``execute`` tool for shell commands + optional extra tools (e.g. smart_search)
- External memory via ``/workspace/.notes`` (persists across context trims)
- Budget-aware output truncation and message trimming
- Stuck detection and graceful fallback
- Text-embedded tool-call detection for models that don't use the tool API
- Reasoning nudge every 3 rounds to keep the agent on track
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from forge_bot.status.formatter import TodoItem, ToolCallRecord, abbreviate
from forge_bot.tools.base import BaseTool

if TYPE_CHECKING:
    from forge_bot.clients.llm import LLMClient
    from forge_bot.config import Settings
    from forge_bot.container.manager import ContainerManager
    from forge_bot.status.manager import StatusCommentManager

logger = logging.getLogger("forge_bot.agent")

# -- constants ---------------------------------------------------------------

_CHARS_PER_TOKEN = 4
_BLOCKED_PATTERNS = ["rm -rf /", "mkfs", "dd if=", "> /dev/"]
_NOTES_PATH = "/workspace/.notes"
_ARTIFACTS_DIR = "/workspace/.artifacts"

_EXECUTE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "execute",
        "description": (
            "Run a shell command in /workspace. "
            "Use bash, python, grep, git, jq. "
            "Use forge-api for Gitea API calls."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute",
                },
            },
            "required": ["command"],
        },
    },
}

# Regex for FORGE_TODO markers the LLM can echo from exec output.
_TODO_SET_RE = re.compile(r"FORGE_TODO:set:(.+)")
_TODO_CHECK_RE = re.compile(r"FORGE_TODO:check:(.+)")

# Detect text-embedded tool calls (models that output JSON instead of using API).
_TEXT_TOOL_RE = re.compile(r'\{\s*"name"\s*:\s*"execute"')

_REASONING_NUDGE = (
    "Pause and assess: What have you found so far? "
    "Do you have enough to answer, or what do you still need?"
)


# -- internal helpers --------------------------------------------------------


@dataclass
class _LoopState:
    """Lightweight stuck-detection bookkeeping."""

    _history: list[str] = field(default_factory=list)
    _stale_rounds: int = 0
    text_call_count: int = 0  # cap text-embedded tool calls at 3

    def record(self, command: str) -> bool:
        """Record a command.  Returns ``True`` if it's a duplicate (3+)."""
        self._history.append(command)
        return self._history.count(command) >= 3

    def record_round(self, had_new: bool) -> None:
        self._stale_rounds = 0 if had_new else self._stale_rounds + 1

    def is_stuck(self) -> bool:
        return self._stale_rounds >= 2


def _estimate_tokens(text: str) -> int:
    return len(text) // _CHARS_PER_TOKEN


def _messages_tokens(messages: list[dict[str, Any]]) -> int:
    total = 0
    for msg in messages:
        content = msg.get("content") or ""
        if isinstance(content, str):
            total += _estimate_tokens(content)
        # tool_calls add some overhead
        if msg.get("tool_calls"):
            total += 40  # rough per-call overhead
    return total


def _check_blocked(command: str) -> str | None:
    cmd_lower = command.lower()
    for pattern in _BLOCKED_PATTERNS:
        if pattern in cmd_lower:
            return pattern
    return None


def _parse_command(arguments: str | dict[str, Any]) -> str:
    if isinstance(arguments, dict):
        return str(arguments.get("command", ""))
    try:
        data = json.loads(arguments)
        return str(data.get("command", ""))
    except (json.JSONDecodeError, AttributeError):
        return str(arguments)


def _truncate(text: str, limit: int) -> str:
    """Truncate *text* keeping the head and a small tail."""
    if len(text) <= limit:
        return text
    head = int(limit * 0.80)
    tail = limit - head - 60  # room for the separator
    if tail < 0:
        tail = 0
    mid = "\n...(truncated, use head/tail/grep to focus)...\n"
    return text[:head] + mid + text[-tail:] if tail else text[:head] + mid


def _extract_text_tool_call(text: str) -> str | None:
    """Detect when an LLM outputs a tool call as JSON text instead of using the API.

    Returns the command string if found, else None.
    """
    if not _TEXT_TOOL_RE.search(text):
        return None
    try:
        start = text.index("{")
        # Find matching closing brace.
        depth, end = 0, start
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            if depth == 0:
                end = i + 1
                break
        data = json.loads(text[start:end])
        args = data.get("arguments", {})
        if isinstance(args, str):
            args = json.loads(args)
        return args.get("command", "")
    except (json.JSONDecodeError, ValueError, KeyError):
        return None


# -- AgentLoop ---------------------------------------------------------------


class AgentLoop:
    """Run an LLM agent loop with *execute* + optional extra tools.

    The LLM writes its own bash / Python scripts to explore the codebase,
    call the Forge API, run tests, etc.  Findings are persisted in
    ``/workspace/.notes`` so they survive context-window trimming.
    """

    def __init__(
        self,
        llm: LLMClient,
        container: ContainerManager,
        settings: Settings,
        *,
        status: StatusCommentManager | None = None,
        extra_tools: list[BaseTool] | None = None,
    ) -> None:
        self._llm = llm
        self._container = container
        self._settings = settings
        self._status = status
        self._extra_tools = extra_tools or []
        self._context_window = settings.llm_context_window

    # -- public API ----------------------------------------------------------

    async def run(self, system_prompt: str, user_message: str) -> str:
        """Execute the agent loop and return the final text response."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        state = _LoopState()
        round_num = 0

        # Build tool schemas: execute + any extra tools
        tools = [_EXECUTE_TOOL] + [t.to_openai_schema() for t in self._extra_tools]

        while True:
            if state.is_stuck():
                logger.info("Agent stuck after %d rounds, breaking", round_num)
                break

            # Inject notes & trim to fit context window
            call_messages = await self._prepare_messages(messages)

            # Call LLM
            try:
                response = await self._llm.chat_with_tools(
                    messages=call_messages,
                    tools=tools,
                )
            except Exception:
                logger.exception("chat_with_tools failed (round %d)", round_num)
                return await self._fallback(system_prompt, user_message)

            if not response.choices:
                logger.warning("LLM returned empty choices (round %d)", round_num)
                return await self._fallback(system_prompt, user_message)

            msg = response.choices[0].message

            # No tool calls -- check for text-embedded tool calls or final response
            if not msg.tool_calls:
                text_cmd = _extract_text_tool_call(msg.content or "")
                if text_cmd and state.text_call_count < 3:
                    state.text_call_count += 1
                    messages.append({"role": "assistant", "content": msg.content})
                    result = await self._handle_tool_call(text_cmd, state, round_num)
                    messages.append({"role": "user", "content": f"Tool result:\n{result}"})
                    state.record_round(True)
                    round_num += 1
                    continue
                return msg.content or ""

            # Append the assistant message (with tool_calls) to history
            messages.append(msg.model_dump())

            had_new = False
            for tc in msg.tool_calls:
                if tc.function.name == "execute":
                    command = _parse_command(tc.function.arguments)
                    result_content = await self._handle_tool_call(command, state, round_num)
                else:
                    result_content = await self._dispatch_extra_tool(
                        tc.function.name, tc.function.arguments, round_num
                    )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_content,
                    }
                )
                if (
                    not result_content.startswith("[Skipped")
                    and not result_content.startswith("Error:")
                    and not result_content.startswith("Command blocked")
                ):
                    had_new = True

            state.record_round(had_new)

            # Reasoning nudge every 3 rounds (starting from round 2)
            if round_num >= 2 and round_num % 3 == 2:
                messages.append({"role": "user", "content": _REASONING_NUDGE})

            round_num += 1

        # Stuck -- force a final response
        return await self._force_final(messages)

    async def collect_artifacts(self) -> list[tuple[str, bytes]]:
        """Read files from /workspace/.artifacts/ for attachment."""
        try:
            result = await self._container.exec(f"ls {_ARTIFACTS_DIR}/ 2>/dev/null", timeout=5)
            if result.exit_code != 0 or not result.stdout.strip():
                return []
            artifacts: list[tuple[str, bytes]] = []
            for name in result.stdout.strip().splitlines():
                name = name.strip()
                if not name:
                    continue
                # Strip directory components to prevent path traversal
                name = os.path.basename(name)
                if not name:
                    continue
                safe_path = shlex.quote(f"{_ARTIFACTS_DIR}/{name}")
                content = await self._container.exec(f"cat {safe_path}", timeout=10)
                if content.exit_code == 0:
                    artifacts.append((name, content.stdout.encode()))
            return artifacts
        except Exception:
            return []

    # -- tool execution ------------------------------------------------------

    async def _handle_tool_call(
        self,
        command: str,
        state: _LoopState,
        round_num: int,
    ) -> str:
        """Execute a single tool call, returning the result string."""
        if not command:
            return "Error: missing 'command' argument."

        # Safety
        blocked = _check_blocked(command)
        if blocked:
            return f"Command blocked for safety: contains '{blocked}'"

        # Duplicate detection
        if state.record(command):
            return "[Skipped: duplicate command already executed]"

        # Status update
        if self._status:
            short = abbreviate(command, 80)
            await self._status.update_phase(f"Running: `{short}`")

        # Execute
        start = time.monotonic()
        result = await self._container.exec(
            command,
            timeout=self._settings.container_timeout,
        )
        duration = time.monotonic() - start

        # Format output with dynamic truncation
        output = self._format_result(result, duration)

        # Record in status comment
        if self._status:
            await self._status.record_tool_call(
                ToolCallRecord(
                    tool_name="execute",
                    arguments_summary=abbreviate(command, 60),
                    result_summary=abbreviate(
                        result.stdout or result.stderr or "(no output)",
                        100,
                    ),
                    success=result.exit_code == 0,
                    duration_seconds=round(duration, 2),
                )
            )

        # Parse FORGE_TODO markers from output
        if self._status:
            await self._parse_todo_markers(result.stdout)

        logger.info(
            "Round %d: exec [exit=%d, %.1fs] %s",
            round_num,
            result.exit_code,
            duration,
            abbreviate(command, 80),
        )
        return output

    async def _dispatch_extra_tool(
        self,
        name: str,
        arguments: str | dict[str, Any],
        round_num: int,
    ) -> str:
        """Dispatch a call to a registered BaseTool."""
        tool = next((t for t in self._extra_tools if t.name == name), None)
        if not tool:
            return f"Error: unknown tool '{name}'"

        if isinstance(arguments, str):
            try:
                args = json.loads(arguments)
            except json.JSONDecodeError:
                args = {}
        else:
            args = arguments

        if self._status:
            await self._status.update_phase(f"Running: `{name}`")

        start = time.monotonic()
        try:
            result = await tool.execute(**args)
            duration = time.monotonic() - start
            output = _truncate(result.content, self._max_output_chars())

            if self._status:
                await self._status.record_tool_call(
                    ToolCallRecord(
                        tool_name=name,
                        arguments_summary=abbreviate(str(args), 60),
                        result_summary=abbreviate(result.content, 100),
                        success=result.success,
                        duration_seconds=round(duration, 2),
                    )
                )

            logger.info(
                "Round %d: %s [success=%s, %.1fs]",
                round_num,
                name,
                result.success,
                duration,
            )
            return output
        except Exception:
            logger.exception("Tool %s failed", name)
            return f"{name} failed. Try a different approach."

    # -- context management --------------------------------------------------

    async def _prepare_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Build the message list for the next LLM call.

        1. Read ``/workspace/.notes`` and inject as a system message.
        2. Trim older rounds so the total fits within the context window.
        """
        # Read notes from container
        notes = await self._read_notes()

        # Build: [system, notes?, user, ...history...]
        result: list[dict[str, Any]] = []

        # Always include the system prompt (messages[0]) and user msg (messages[1])
        result.append(messages[0])  # system
        if notes:
            notes_limit = self._context_window * _CHARS_PER_TOKEN // 4
            if len(notes) > notes_limit:
                notes = notes[-notes_limit:]
            result.append(
                {
                    "role": "system",
                    "content": f"## Your accumulated findings:\n{notes}",
                }
            )
        result.append(messages[1])  # user

        # History = everything after system + user (index 2+)
        history = messages[2:]

        # Budget: context_window minus reserve for response
        reserve = min(self._settings.llm_max_tokens, self._context_window // 4)
        budget = self._context_window - reserve
        pinned_tokens = _messages_tokens(result)

        # Keep as much recent history as fits
        # Walk backwards, accumulating until budget is spent
        kept: list[dict[str, Any]] = []
        used = pinned_tokens
        for msg in reversed(history):
            msg_tokens = _messages_tokens([msg])
            if used + msg_tokens > budget:
                break
            kept.append(msg)
            used += msg_tokens
        kept.reverse()

        result.extend(kept)
        return result

    async def _read_notes(self) -> str:
        """Read the contents of /workspace/.notes, or '' if absent."""
        try:
            result = await self._container.exec(
                f"cat {_NOTES_PATH} 2>/dev/null || true",
                timeout=5,
            )
            return result.stdout.strip()
        except Exception:
            return ""

    # -- output formatting ---------------------------------------------------

    def _format_result(self, result: Any, duration: float) -> str:
        """Format an ExecResult for the LLM, with dynamic truncation."""
        limit = self._max_output_chars()
        parts: list[str] = []
        if result.exit_code != 0:
            parts.append(f"Exit code: {result.exit_code}")
        if result.stdout:
            parts.append(_truncate(result.stdout, limit))
        if result.stderr:
            parts.append(f"stderr: {_truncate(result.stderr, limit // 3)}")
        if not parts:
            parts.append("(no output)")
        parts.append(f"[{duration:.1f}s]")
        return "\n".join(parts)

    def _max_output_chars(self) -> int:
        """Compute output char limit based on remaining context budget."""
        # Use at most half the context window for a single output
        half_window = self._context_window * _CHARS_PER_TOKEN // 2
        return max(500, min(8000, half_window))

    # -- fallbacks -----------------------------------------------------------

    async def _fallback(self, system_prompt: str, user_message: str) -> str:
        """Try simple chat without tools, then canned error."""
        try:
            return await self._llm.chat(system_prompt, user_message)
        except Exception:
            logger.exception("Fallback LLM call also failed")
            return (
                "Sorry, I encountered an error while processing your request. "
                "Please try again later."
            )

    async def _force_final(self, messages: list[dict[str, Any]]) -> str:
        """Ask the LLM for a final response without tools."""
        messages.append(
            {
                "role": "user",
                "content": (
                    "Please provide your final response now based on everything "
                    "you have gathered. Do not make any more tool calls."
                ),
            }
        )
        call_messages = await self._prepare_messages(messages)
        try:
            response = await self._llm.chat_with_tools(
                messages=call_messages,
                tools=[],
            )
            if not response.choices:
                return "I was unable to complete my analysis. Please try again."
            return response.choices[0].message.content or ""
        except Exception:
            logger.exception("Force-final LLM call failed")
            # Try to find the last assistant text
            for msg in reversed(messages):
                if msg.get("role") == "assistant":
                    content = msg.get("content")
                    if content:
                        return content
            return "I was unable to complete my analysis. Please try again."

    # -- todo markers --------------------------------------------------------

    async def _parse_todo_markers(self, output: str) -> None:
        """Scan exec output for ``FORGE_TODO:`` markers and update status."""
        if not self._status or not output:
            return

        for line in output.splitlines():
            m = _TODO_SET_RE.search(line)
            if m:
                try:
                    items = json.loads(m.group(1))
                    if isinstance(items, list):
                        todos = [TodoItem(text=str(t)) for t in items]
                        await self._status.update_todos(todos)
                except (json.JSONDecodeError, TypeError):
                    pass
                continue

            m = _TODO_CHECK_RE.search(line)
            if m:
                check_text = m.group(1).strip()
                current = self._status.todos
                for todo in current:
                    if todo.text == check_text:
                        todo.done = True
                await self._status.update_todos(current)
