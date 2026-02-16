"""ExecTool: run commands in the per-event workspace container."""

from __future__ import annotations

from typing import TYPE_CHECKING

from forge_bot.tools.base import BaseTool, ToolParameter, ToolResult

if TYPE_CHECKING:
    from forge_bot.container.manager import ContainerManager


class ExecTool(BaseTool):
    """Run a shell command in the workspace container.

    The repo is cloned at ``/workspace``.  Supports git, bash, python,
    curl, grep, find, and other standard tools.
    """

    name = "exec"
    description = (
        "Run a shell command in the workspace container. "
        "The repo is cloned at /workspace. You can run git, bash, python, "
        "curl, and other commands. Use this to run tests, read files, "
        "search code, or analyze the repository."
    )
    parameters = [
        ToolParameter(
            "command",
            "string",
            "Shell command to execute (e.g. 'python -m pytest tests/', "
            "'grep -r \"TODO\" --include=\"*.py\"', 'git log --oneline -10')",
        ),
        ToolParameter(
            "timeout",
            "integer",
            "Timeout in seconds (default: 60, max: 120)",
            required=False,
        ),
    ]

    _BLOCKED_PATTERNS = [
        "rm -rf /",
        "mkfs",
        "dd if=",
        "> /dev/",
    ]

    def __init__(self, container_manager: ContainerManager) -> None:
        self._cm = container_manager

    async def execute(self, **kwargs: object) -> ToolResult:
        command = str(kwargs.get("command", ""))
        timeout = int(kwargs.get("timeout", 60))
        timeout = min(timeout, 120)

        if not command:
            return ToolResult(self.name, False, "Missing command")

        cmd_lower = command.lower()
        for blocked in self._BLOCKED_PATTERNS:
            if blocked in cmd_lower:
                return ToolResult(
                    self.name,
                    False,
                    f"Command blocked for safety: contains '{blocked}'",
                )

        try:
            result = await self._cm.exec(command, timeout=timeout)
            parts: list[str] = []
            if result.exit_code != 0:
                parts.append(f"Exit code: {result.exit_code}")
            if result.stdout:
                parts.append(f"stdout:\n{result.stdout}")
            if result.stderr:
                parts.append(f"stderr:\n{result.stderr}")
            if not parts:
                parts.append("(no output)")
            parts.append(f"[{result.duration_seconds:.1f}s]")
            return ToolResult(
                self.name, result.exit_code == 0, "\n".join(parts)
            )
        except Exception as exc:
            return ToolResult(self.name, False, f"Exec error: {exc}")
