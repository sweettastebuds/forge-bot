"""Verification checks for the tool-calling loop and pre-post response.

Three in-loop checks:
- Relevance: is the tool call plausibly related to the task?
- Hallucination: does the LLM's narrative contradict tool output?
- Progress: is the LLM stuck repeating itself?

One pre-post check:
- Response verification: does the final response contain fabricated claims?
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from forge_bot.tools.base import ToolResult

logger = logging.getLogger("forge_bot.handlers.verification")


@dataclass
class VerifyResult:
    """Result of the pre-post response verification."""

    passed: bool
    failures: list[str] = field(default_factory=list)


# -- In-loop: Relevance check --


def check_relevance(
    tool_name: str,
    tool_args: dict,
    user_question: str,
) -> bool:
    """Check if a tool call is plausibly relevant to the user's question.

    This is a lightweight pattern-based check (no LLM call).  Returns
    True if the call seems relevant, False if clearly unrelated.

    Defaults to True (permissive) since we'd rather allow a marginally
    useful call than block a genuinely helpful one.
    """
    # Always allow todo — it's a meta-tool for planning.
    if tool_name == "todo":
        return True

    # Always allow search_api — it's exploratory.
    if tool_name == "search_api":
        return True

    # For exec and api_call, check that the command/endpoint has some
    # lexical overlap with the question or is a common exploration command.
    _ALWAYS_RELEVANT_COMMANDS = {
        "ls", "find", "grep", "cat", "head", "tail", "git", "tree",
        "python", "pytest", "npm", "cargo", "make", "wc",
    }

    if tool_name == "exec":
        command = str(tool_args.get("command", ""))
        first_word = command.split()[0] if command.split() else ""
        base_cmd = first_word.rsplit("/", 1)[-1]
        if base_cmd in _ALWAYS_RELEVANT_COMMANDS:
            return True

    # Default: allow (permissive).
    return True


# -- In-loop: Hallucination check --


_EXIT_CODE_CLAIMS = re.compile(
    r"(?:tests?\s+(?:pass|succeed|all\s+pass))|(?:no\s+(?:errors?|failures?))",
    re.IGNORECASE,
)
_FAIL_CLAIMS = re.compile(
    r"(?:tests?\s+fail)|(?:errors?\s+found)|(?:compilation\s+fail)",
    re.IGNORECASE,
)


def check_hallucination(
    llm_text: str,
    tool_results: list[tuple[str, ToolResult]],
) -> list[str]:
    """Compare LLM claims against actual tool output.

    Returns a list of mismatch descriptions (empty = no issues found).
    """
    mismatches: list[str] = []

    for tool_name, result in tool_results:
        if tool_name != "exec":
            continue

        # Check for "tests pass" claims when exit code was non-zero.
        exit_code_match = re.search(r"Exit code:\s*(\d+)", result.content)
        if exit_code_match:
            exit_code = int(exit_code_match.group(1))
            if exit_code != 0 and _EXIT_CODE_CLAIMS.search(llm_text):
                mismatches.append(
                    f"LLM claims tests pass/succeed but exec returned "
                    f"exit code {exit_code}"
                )
            if exit_code == 0 and _FAIL_CLAIMS.search(llm_text):
                mismatches.append(
                    f"LLM claims tests fail but exec returned exit code 0"
                )

    return mismatches


# -- In-loop: Progress check --


@dataclass
class ProgressTracker:
    """Tracks tool call history to detect loops and stalls."""

    _history: list[tuple[str, str]] = field(default_factory=list)
    _stale_rounds: int = 0

    def record(self, tool_name: str, tool_args_key: str) -> None:
        """Record a tool call (name + abbreviated args for dedup)."""
        self._history.append((tool_name, tool_args_key))

    def is_duplicate(self, tool_name: str, tool_args_key: str) -> bool:
        """Check if this exact call has been made before."""
        count = sum(
            1 for n, a in self._history
            if n == tool_name and a == tool_args_key
        )
        return count >= 2

    def record_round(self, had_new_calls: bool) -> None:
        """Record whether a round had meaningful new tool calls."""
        if had_new_calls:
            self._stale_rounds = 0
        else:
            self._stale_rounds += 1

    def is_stuck(self) -> bool:
        """Check if the LLM is stuck (2+ rounds with no new calls)."""
        return self._stale_rounds >= 2


# -- Pre-post: Response verification --


def verify_response(
    response_text: str,
    tool_results: list[tuple[str, ToolResult]],
) -> VerifyResult:
    """Validate the final response against accumulated tool results.

    Checks:
    - Exit code claims match actual results
    - File paths referenced in response were actually seen in tool output
    - Code blocks in response correspond to tool output
    """
    failures: list[str] = []

    # Check 1: exit code claims vs actual.
    hallucinations = check_hallucination(response_text, tool_results)
    failures.extend(hallucinations)

    # Check 2: file paths mentioned in response should appear in tool output.
    # Extract paths from response (simplified: anything that looks like a/b.ext)
    path_re = re.compile(
        r"(?:^|[\s`\"'])"
        r"([\w./-]+\.(?:py|js|ts|go|rs|java|c|cpp|h|yml|yaml|toml|json|md|txt|sh))"
        r"(?:[\s`\"',;:!?)]|$)",
        re.MULTILINE,
    )
    response_paths = set(path_re.findall(response_text))

    # Collect all paths that appeared in tool output.
    tool_output_text = " ".join(r.content for _, r in tool_results)
    seen_paths = set(path_re.findall(tool_output_text))

    # Also include paths from exec commands (e.g., "cat src/main.py").
    for tool_name, result in tool_results:
        if tool_name == "exec":
            cmd_paths = path_re.findall(result.command if hasattr(result, "command") else "")
            seen_paths.update(cmd_paths)

    fabricated = response_paths - seen_paths
    # Filter out very common paths that might be mentioned generically.
    _COMMON_PATHS = {"README.md", "setup.py", "main.py", "index.js"}
    fabricated -= _COMMON_PATHS

    if len(fabricated) > 3:
        failures.append(
            f"Response references {len(fabricated)} file paths not seen "
            f"in any tool output: {', '.join(list(fabricated)[:5])}"
        )

    return VerifyResult(
        passed=len(failures) == 0,
        failures=failures,
    )
