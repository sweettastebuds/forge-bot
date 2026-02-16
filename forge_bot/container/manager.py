"""Per-event persistent container lifecycle manager.

Replaces the fire-and-forget sandbox with a container that persists for the
entire webhook event processing.  The repo is cloned on creation, and the
handler can run arbitrary commands via ``exec()``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

import docker

from forge_bot.config import Settings

logger = logging.getLogger("forge_bot.container.manager")

_MAX_OUTPUT_CHARS = 8_000
_INIT_POLL_INTERVAL = 1.0  # seconds


@dataclass
class ExecResult:
    """Result of a command execution in the container."""

    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    command: str


class ContainerManager:
    """Per-event persistent container with repo clone and exec support.

    Usage::

        async with ContainerManager(settings, clone_url, "main", token="...") as cm:
            result = await cm.exec("git log --oneline -5")
            result = await cm.exec("python -m pytest tests/")
    """

    def __init__(
        self,
        settings: Settings,
        repo_clone_url: str,
        repo_ref: str,
        *,
        token: str = "",
        network_enabled: bool = True,
        image: str | None = None,
    ) -> None:
        self._settings = settings
        self._clone_url = repo_clone_url
        self._ref = repo_ref
        self._token = token
        self._network = network_enabled
        self._image = image or settings.container_workspace_image
        self._docker: docker.DockerClient | None = None
        self._container: Any = None

    # -- async context manager --

    async def __aenter__(self) -> ContainerManager:
        await self.create()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.destroy()

    # -- lifecycle --

    async def create(self) -> None:
        """Create the container, clone the repo, and wait until ready."""
        self._docker = await asyncio.to_thread(docker.from_env)

        authed_url = self._inject_token(self._clone_url, self._token)

        init_script = (
            f"git clone --depth=50 --no-single-branch '{authed_url}' /workspace"
            f" && cd /workspace"
            f" && git checkout '{self._ref}'"
            f" && echo 'FORGE_READY'"
        )

        self._container = await asyncio.to_thread(
            self._docker.containers.run,
            self._image,
            ["sh", "-c", f"{init_script} && sleep infinity"],
            detach=True,
            mem_limit=self._settings.sandbox_memory,
            nano_cpus=int(self._settings.sandbox_cpus * 1e9),
            pids_limit=256,
            network_mode="bridge" if self._network else "none",
            working_dir="/workspace",
            environment={"GIT_TERMINAL_PROMPT": "0"},
            tmpfs={"/tmp": "size=200m"},
        )

        logger.info("Container %s created, cloning repo...", self._container.short_id)
        await self._wait_for_ready(timeout=self._settings.sandbox_timeout)
        logger.info("Container %s ready", self._container.short_id)

    async def exec(
        self,
        command: str,
        *,
        timeout: int | None = None,
        workdir: str = "/workspace",
    ) -> ExecResult:
        """Execute a command inside the running container.

        Returns an :class:`ExecResult` with exit code, stdout, stderr, and
        wall-clock duration.
        """
        if not self._container:
            raise RuntimeError("Container not created — call create() first")

        effective_timeout = min(
            timeout or self._settings.sandbox_timeout,
            120,
        )

        start = time.monotonic()
        try:
            exec_handle = await asyncio.wait_for(
                asyncio.to_thread(
                    self._container.exec_run,
                    ["sh", "-c", command],
                    workdir=workdir,
                    demux=True,
                ),
                timeout=effective_timeout,
            )
        except asyncio.TimeoutError:
            duration = time.monotonic() - start
            return ExecResult(
                exit_code=-1,
                stdout="",
                stderr=f"Command timed out after {effective_timeout}s.",
                duration_seconds=round(duration, 2),
                command=command,
            )

        duration = time.monotonic() - start
        exit_code = exec_handle.exit_code
        stdout_raw, stderr_raw = exec_handle.output

        stdout = (stdout_raw or b"").decode("utf-8", errors="replace")
        stderr = (stderr_raw or b"").decode("utf-8", errors="replace")

        if len(stdout) > _MAX_OUTPUT_CHARS:
            stdout = stdout[:_MAX_OUTPUT_CHARS] + "\n... (truncated)"
        if len(stderr) > _MAX_OUTPUT_CHARS:
            stderr = stderr[:_MAX_OUTPUT_CHARS] + "\n... (truncated)"

        return ExecResult(
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=round(duration, 2),
            command=command,
        )

    async def destroy(self) -> None:
        """Force-remove the container and close the Docker client."""
        if self._container:
            try:
                cid = self._container.short_id
                await asyncio.to_thread(self._container.remove, force=True)
                logger.info("Container %s destroyed", cid)
            except Exception:
                logger.warning("Failed to remove container", exc_info=True)
            self._container = None
        if self._docker:
            await asyncio.to_thread(self._docker.close)
            self._docker = None

    # -- internals --

    async def _wait_for_ready(self, timeout: int = 60) -> None:
        """Poll container logs until ``FORGE_READY`` marker appears."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            logs: bytes = await asyncio.to_thread(
                self._container.logs, stdout=True, stderr=False,
            )
            if b"FORGE_READY" in logs:
                return
            await asyncio.sleep(_INIT_POLL_INTERVAL)
        raise TimeoutError(
            f"Container init did not complete within {timeout}s"
        )

    @staticmethod
    def _inject_token(clone_url: str, token: str) -> str:
        """Inject API token into git clone URL for authentication.

        ``https://gitea.example.com/owner/repo.git``
        → ``https://token@gitea.example.com/owner/repo.git``
        """
        if not token:
            return clone_url
        if "://" in clone_url:
            scheme, rest = clone_url.split("://", 1)
            return f"{scheme}://{token}@{rest}"
        return clone_url
