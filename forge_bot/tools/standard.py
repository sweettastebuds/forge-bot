"""Standard local tools for the agent."""

from __future__ import annotations

import asyncio
import glob as globlib
import re
import sys
from asyncio.subprocess import PIPE
from pathlib import Path
from typing import Iterable

from forge_bot.tools.base import BaseTool, ToolParameter, ToolResult


class _PathBoundTool(BaseTool):
    """Base class for tools operating on the local filesystem."""

    def __init__(self, base_dir: str | Path | None = None) -> None:
        self._base_dir = Path(base_dir).resolve() if base_dir else None

    def _resolve_path(self, raw_path: str) -> Path:
        if not raw_path:
            raise ValueError("Missing required parameter: path")
        path = Path(raw_path)
        if not path.is_absolute():
            path = (self._base_dir or Path.cwd()) / path
        path = path.resolve()
        if self._base_dir and self._base_dir not in path.parents and path != self._base_dir:
            raise ValueError("Path escapes the base directory.")
        return path

    def _rel_path(self, path: Path) -> str:
        if self._base_dir:
            try:
                return str(path.relative_to(self._base_dir))
            except ValueError:
                return str(path)
        return str(path)


class ReadFileTool(_PathBoundTool):
    """Read the contents of a file."""

    name = "read_file"
    description = "Read a file from disk"
    parameters = [
        ToolParameter(
            name="path",
            type="string",
            description="File path",
        )
    ]

    async def execute(self, **kwargs: object) -> ToolResult:
        raw_path = kwargs.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            return ToolResult(self.name, False, "Missing required parameter: path")
        try:
            path = self._resolve_path(raw_path)
            content = await asyncio.to_thread(path.read_text)
            return ToolResult(self.name, True, content)
        except FileNotFoundError:
            return ToolResult(self.name, False, f"File not found: {raw_path}")
        except UnicodeDecodeError:
            try:
                data = await asyncio.to_thread(path.read_bytes)
                return ToolResult(self.name, True, data.decode(errors="replace"))
            except Exception as exc:  # pragma: no cover - unlikely
                return ToolResult(self.name, False, f"Failed to read file: {exc}")
        except Exception as exc:
            return ToolResult(self.name, False, f"Failed to read file: {exc}")


class WriteFileTool(_PathBoundTool):
    """Write content to a file, creating parents as needed."""

    name = "write_file"
    description = "Write text to a file on disk"
    parameters = [
        ToolParameter(
            name="path",
            type="string",
            description="File path",
        ),
        ToolParameter(
            name="content",
            type="string",
            description="File contents to write",
        ),
    ]

    async def execute(self, **kwargs: object) -> ToolResult:
        raw_path = kwargs.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            return ToolResult(self.name, False, "Missing required parameter: path")
        if "content" not in kwargs:
            return ToolResult(self.name, False, "Missing required parameter: content")
        content = str(kwargs.get("content", ""))
        try:
            path = self._resolve_path(raw_path)
            await asyncio.to_thread(self._write, path, content)
            return ToolResult(
                self.name,
                True,
                f"Wrote {len(content)} characters to {self._rel_path(path)}",
            )
        except Exception as exc:
            return ToolResult(self.name, False, f"Failed to write file: {exc}")

    @staticmethod
    def _write(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


class EditFileTool(_PathBoundTool):
    """Perform a string replacement within a file."""

    name = "edit_file"
    description = "Replace text in a file"
    parameters = [
        ToolParameter(
            name="path",
            type="string",
            description="File path",
        ),
        ToolParameter(
            name="old",
            type="string",
            description="Text to replace",
        ),
        ToolParameter(
            name="new",
            type="string",
            description="Replacement text",
        ),
    ]

    async def execute(self, **kwargs: object) -> ToolResult:
        raw_path = kwargs.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            return ToolResult(self.name, False, "Missing required parameter: path")
        if "old" not in kwargs or "new" not in kwargs:
            return ToolResult(self.name, False, "Missing required parameters: old, new")
        old = str(kwargs.get("old", ""))
        new = str(kwargs.get("new", ""))
        if not old:
            return ToolResult(self.name, False, "Parameter 'old' must not be empty.")

        try:
            path = self._resolve_path(raw_path)
            replaced = await asyncio.to_thread(self._replace, path, old, new)
            if replaced == 0:
                return ToolResult(self.name, False, "No occurrences found to replace.")
            return ToolResult(
                self.name,
                True,
                f"Replaced {replaced} occurrence(s) in {self._rel_path(path)}",
            )
        except FileNotFoundError:
            return ToolResult(self.name, False, f"File not found: {raw_path}")
        except Exception as exc:
            return ToolResult(self.name, False, f"Failed to edit file: {exc}")

    @staticmethod
    def _replace(path: Path, old: str, new: str) -> int:
        text = path.read_text()
        count = text.count(old)
        if count:
            path.write_text(text.replace(old, new))
        return count


class BashTool(_PathBoundTool):
    """Run a shell command locally."""

    name = "bash"
    description = "Execute a shell command"
    parameters = [
        ToolParameter(
            name="cmd",
            type="string",
            description="Command to run",
        )
    ]

    async def execute(self, **kwargs: object) -> ToolResult:
        cmd = kwargs.get("cmd")
        if not isinstance(cmd, str) or not cmd.strip():
            return ToolResult(self.name, False, "Missing required parameter: cmd")

        process = await asyncio.create_subprocess_shell(
            cmd,
            cwd=str(self._base_dir) if self._base_dir else None,
            stdout=PIPE,
            stderr=PIPE,
        )
        stdout, stderr = await process.communicate()
        combined = (stdout or b"").decode() + (stderr or b"").decode()
        if not combined.strip():
            combined = f"(exit code {process.returncode})"
        return ToolResult(self.name, process.returncode == 0, combined)


class GlobTool(_PathBoundTool):
    """Find files matching a glob pattern."""

    name = "glob"
    description = "List files matching a glob pattern"
    parameters = [
        ToolParameter(
            name="pattern",
            type="string",
            description="Glob pattern (supports **)",
        )
    ]

    async def execute(self, **kwargs: object) -> ToolResult:
        pattern = kwargs.get("pattern")
        if not isinstance(pattern, str) or not pattern.strip():
            return ToolResult(self.name, False, "Missing required parameter: pattern")

        root = self._base_dir or Path.cwd()

        def _run_glob() -> list[str]:
            matches = globlib.glob(pattern, recursive=True, root_dir=root)
            return [str((root / Path(m)).resolve()) for m in matches]

        matches = await asyncio.to_thread(_run_glob)
        if not matches:
            return ToolResult(self.name, True, "No matches found.")
        content = "\n".join(sorted(matches))
        return ToolResult(self.name, True, content)


class GrepTool(_PathBoundTool):
    """Regex search within files."""

    name = "grep"
    description = "Search for a regex pattern in files"
    parameters = [
        ToolParameter(
            name="pattern",
            type="string",
            description="Regex pattern to search for",
        ),
        ToolParameter(
            name="path",
            type="string",
            description="File path, directory, or glob pattern to search",
        ),
    ]

    async def execute(self, **kwargs: object) -> ToolResult:
        pattern = kwargs.get("pattern")
        target = kwargs.get("path")
        if not isinstance(pattern, str) or not pattern.strip():
            return ToolResult(self.name, False, "Missing required parameter: pattern")
        if not isinstance(target, str) or not target.strip():
            return ToolResult(self.name, False, "Missing required parameter: path")

        try:
            regex = re.compile(pattern)
        except re.error as exc:
            return ToolResult(self.name, False, f"Invalid regex: {exc}")

        try:
            base_path = self._resolve_path(target)
        except ValueError as exc:
            return ToolResult(self.name, False, str(exc))

        matches = await asyncio.to_thread(
            self._search,
            regex,
            base_path,
            target,
        )
        if not matches:
            return ToolResult(self.name, True, "No matches found.")
        content = "\n".join(matches)
        return ToolResult(self.name, True, content)

    def _search(self, regex: re.Pattern[str], base_path: Path, target: str) -> list[str]:
        files = self._candidate_files(base_path, target)
        results: list[str] = []
        for file_path in files:
            try:
                text = file_path.read_text(errors="ignore")
            except Exception:
                continue
            for idx, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    results.append(f"{self._rel_path(file_path)}:{idx}:{line.strip()}")
                if len(results) >= 50:
                    return results
        return results

    def _candidate_files(self, base_path: Path, target: str) -> Iterable[Path]:
        if any(ch in target for ch in "*?[]"):
            root = self._base_dir or Path.cwd()
            matches = globlib.glob(target, recursive=True, root_dir=root)
            return [
                (root / Path(m)).resolve()
                for m in matches
                if (root / Path(m)).resolve().is_file()
            ]
        if base_path.is_dir():
            return (p for p in base_path.rglob("*") if p.is_file())
        if base_path.is_file():
            return [base_path]
        return []


class AskUserTool(BaseTool):
    """Prompt the user for input (best-effort)."""

    name = "ask_user"
    description = "Ask the user a question and return their response"
    parameters = [
        ToolParameter(
            name="question",
            type="string",
            description="Question to present to the user",
        )
    ]

    async def execute(self, **kwargs: object) -> ToolResult:
        question = kwargs.get("question")
        if not isinstance(question, str):
            return ToolResult(self.name, False, "Missing required parameter: question")

        stdin = sys.stdin
        if not stdin or not stdin.isatty():
            return ToolResult(
                self.name,
                False,
                "ask_user is unavailable in non-interactive environments.",
            )

        try:
            answer = await asyncio.to_thread(input, question)
            return ToolResult(self.name, True, answer)
        except EOFError:
            return ToolResult(self.name, False, "No input available.")
        except Exception as exc:  # pragma: no cover - defensive
            return ToolResult(self.name, False, f"Failed to read input: {exc}")


def default_tools(base_dir: str | Path | None = None) -> list[BaseTool]:
    """Return the standard toolset."""
    return [
        ReadFileTool(base_dir),
        WriteFileTool(base_dir),
        EditFileTool(base_dir),
        BashTool(base_dir),
        GlobTool(base_dir),
        GrepTool(base_dir),
        AskUserTool(),
    ]
