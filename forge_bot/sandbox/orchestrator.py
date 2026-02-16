"""Sandbox orchestrator: ephemeral container lifecycle via Docker SDK."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

import docker

from forge_bot.config import Settings
from forge_bot.sandbox.images import ImageRegistry
from forge_bot.sandbox.parser import RunCommand

logger = logging.getLogger("forge_bot.sandbox.orchestrator")

# Extension map: language key → (file extension, run command template)
_LANG_CONFIG: dict[str, tuple[str, str]] = {
    "python": (".py", "python /tmp/code.py"),
    "node": (".js", "node /tmp/code.js"),
    "go": (".go", "go run /tmp/code.go"),
    "rust": (".rs", "rustc /tmp/code.rs -o /tmp/code && /tmp/code"),
    "c": (".c", "gcc /tmp/code.c -o /tmp/code && /tmp/code"),
}

_MAX_OUTPUT_CHARS = 8_000


@dataclass
class ExecutionResult:
    """Result of a sandbox code execution."""

    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    image: str
    oom_killed: bool


class SandboxOrchestrator:
    """Create, execute, and destroy ephemeral sandbox containers."""

    def __init__(self, settings: Settings, registry: ImageRegistry) -> None:
        self._settings = settings
        self._registry = registry
        self._client: docker.DockerClient | None = None

    async def connect(self) -> None:
        """Connect to the Docker daemon (DinD)."""
        self._client = await asyncio.to_thread(
            docker.from_env,
        )
        logger.info("Connected to Docker daemon")

    async def close(self) -> None:
        """Close the Docker client."""
        if self._client:
            await asyncio.to_thread(self._client.close)
            self._client = None

    async def execute(self, command: RunCommand) -> ExecutionResult:
        """Run code in an ephemeral container and return the result."""
        if not self._client:
            raise RuntimeError("Not connected to Docker daemon — call connect() first")

        image = self._registry.resolve(command.language)
        if not image:
            image = self._registry.resolve("default")
        if not image:
            raise ValueError(
                f"Unknown language '{command.language}'. "
                f"Available: {', '.join(self._registry.available_languages())}"
            )

        lang_config = _LANG_CONFIG.get(command.language)
        if lang_config:
            ext, run_cmd = lang_config
        else:
            # Fallback: assume script-style execution
            ext = f".{command.language}"
            run_cmd = f"/tmp/code{ext}"

        filename = f"code{ext}"

        # Build shell command: write code to file, then execute
        shell_cmd = (
            f"cat > /tmp/{filename} << 'FORGE_EOF'\n"
            f"{command.code}\n"
            f"FORGE_EOF\n"
            f"{run_cmd}"
        )

        container = None
        start = time.monotonic()
        try:
            container = await asyncio.to_thread(
                self._client.containers.run,
                image,
                ["sh", "-c", shell_cmd],
                detach=True,
                mem_limit=self._settings.sandbox_memory,
                nano_cpus=int(self._settings.sandbox_cpus * 1e9),
                pids_limit=256,
                network_mode="bridge" if command.network_enabled else "none",
                read_only=True,
                tmpfs={"/tmp": "size=100m"},
                environment={},  # No host env leaked
            )

            # Wait for completion or timeout
            result: dict[str, Any] = await asyncio.to_thread(
                container.wait,
                timeout=self._settings.sandbox_timeout,
            )
            duration = time.monotonic() - start
            exit_code = result.get("StatusCode", -1)

            stdout = await asyncio.to_thread(
                container.logs, stdout=True, stderr=False,
            )
            stderr = await asyncio.to_thread(
                container.logs, stdout=False, stderr=True,
            )

            # Check for OOM — refresh container state to get latest attrs
            await asyncio.to_thread(container.reload)
            oom_killed = container.attrs.get("State", {}).get("OOMKilled", False)

            stdout_str = stdout.decode("utf-8", errors="replace")
            stderr_str = stderr.decode("utf-8", errors="replace")

            if len(stdout_str) > _MAX_OUTPUT_CHARS:
                stdout_str = (
                    stdout_str[:_MAX_OUTPUT_CHARS] + "\n... (output truncated)"
                )
            if len(stderr_str) > _MAX_OUTPUT_CHARS:
                stderr_str = (
                    stderr_str[:_MAX_OUTPUT_CHARS] + "\n... (output truncated)"
                )

            return ExecutionResult(
                exit_code=exit_code,
                stdout=stdout_str,
                stderr=stderr_str,
                duration_seconds=round(duration, 2),
                image=image,
                oom_killed=oom_killed,
            )

        except Exception as exc:
            duration = time.monotonic() - start
            # Check if it's a timeout-related error
            err_str = str(exc).lower()
            if "timeout" in err_str or "timed out" in err_str:
                return ExecutionResult(
                    exit_code=-1,
                    stdout="",
                    stderr=f"Execution timed out after {self._settings.sandbox_timeout} seconds.",
                    duration_seconds=round(duration, 2),
                    image=image or "unknown",
                    oom_killed=False,
                )
            raise

        finally:
            if container:
                try:
                    await asyncio.to_thread(container.remove, force=True)
                except Exception:
                    logger.warning("Failed to remove container", exc_info=True)
