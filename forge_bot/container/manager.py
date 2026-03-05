"""Per-event persistent container lifecycle manager.

Replaces the fire-and-forget sandbox with a container that persists for the
entire webhook event processing.  The container starts with an empty
``/workspace``; the LLM decides how to clone the repo via the exec tool.
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

_INIT_SCRIPT = (
    "if command -v git >/dev/null 2>&1; then "
    "echo 'FORGE_INIT: tools present, skipping install'; "
    "else "
    "echo 'FORGE_INIT: installing git curl jq...' && "
    "apt-get update -qq && "
    "apt-get install -y -qq --no-install-recommends git curl jq >/dev/null 2>&1 && "
    "echo 'FORGE_INIT: install complete'; "
    "fi && "
    # Install forge-api helper — lets the LLM call the Gitea/Forgejo API
    # with `forge-api GET /repos/owner/repo/...` instead of a full curl command.
    "printf '%s\\n' '#!/bin/sh' "
    "'METHOD=\"${1:-GET}\"' "
    "'ENDPOINT=\"$2\"' "
    "'shift 2 2>/dev/null' "
    "'exec curl -sf "
    '-H "Authorization: token $FORGE_TOKEN" '
    '-H "Content-Type: application/json" '
    '-X "$METHOD" "$FORGE_URL/api/v1$ENDPOINT" "$@"\' '
    "> /usr/local/bin/forge-api && "
    "chmod +x /usr/local/bin/forge-api && "
    "echo FORGE_READY && "
    "sleep infinity"
)


@dataclass
class ExecResult:
    """Result of a command execution in the container."""

    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    command: str


class ContainerManager:
    """Per-event persistent container with exec support.

    The container starts with an empty ``/workspace``.  The LLM clones the
    repo itself via the exec tool, choosing the right depth/flags for the
    task at hand.

    Usage::

        async with ContainerManager(settings, clone_url, "main", token="...") as cm:
            result = await cm.exec("git clone ... /workspace")
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
        forge_url: str = "",
        owner: str = "",
        repo: str = "",
    ) -> None:
        self._settings = settings
        self._clone_url = repo_clone_url
        self._ref = repo_ref
        self._token = token
        self._network = network_enabled
        self._image = image or settings.container_workspace_image
        self._forge_url = forge_url
        self._owner = owner
        self._repo = repo
        self._docker: docker.DockerClient | None = None
        self._container: Any = None
        self._authed_url: str = ""

    # -- async context manager --

    async def __aenter__(self) -> ContainerManager:
        await self.create()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.destroy()

    # -- public properties --

    @property
    def clone_url(self) -> str:
        """Authenticated clone URL (token already injected)."""
        return self._authed_url

    @property
    def default_branch(self) -> str:
        """Default branch from the webhook payload."""
        return self._ref

    # -- lifecycle --

    async def build_image(self, dockerfile_path: str) -> None:
        """Build a custom Docker image from a Dockerfile.

        This can be used to prepare a custom workspace image with specific
        tools or dependencies pre-installed.  The image is tagged as
        ``forge-bot-workspace:latest`` by default.
        """
        self._docker = await asyncio.to_thread(docker.from_env)
        logger.info("Building Docker image from %s...", dockerfile_path)
        await asyncio.to_thread(
            self._docker.images.build,
            path=".",
            dockerfile=dockerfile_path,
            tag=self._image,
        )
        logger.info("Docker image %s built successfully", self._image)

    async def create(self) -> None:
        """Create the container and wait until ready.

        The container starts with an empty ``/workspace``.  The LLM is
        responsible for cloning the repo via the exec tool.
        """
        self._docker = await asyncio.to_thread(docker.from_env)
        self._authed_url = self._inject_token(self._clone_url, self._token)

        self._container = await asyncio.to_thread(
            self._docker.containers.run,
            self._image,
            ["sh", "-c", _INIT_SCRIPT],
            detach=True,
            mem_limit=self._settings.container_memory,
            nano_cpus=int(self._settings.container_cpus * 1e9),
            pids_limit=256,
            network_mode="bridge" if self._network else "none",
            working_dir="/workspace",
            environment={
                "GIT_TERMINAL_PROMPT": "0",
                "FORGE_URL": self._forge_url,
                "FORGE_TOKEN": self._token,
                "FORGE_OWNER": self._owner,
                "FORGE_REPO": self._repo,
            },
            tmpfs={"/tmp": "size=200m"},
        )

        logger.info("Container %s created", self._container.short_id)
        await self._wait_for_ready(timeout=self._settings.container_timeout)
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
            timeout or self._settings.container_timeout,
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
        except TimeoutError:
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
                self._container.logs,
                stdout=True,
                stderr=False,
            )
            if b"FORGE_READY" in logs:
                return
            await asyncio.sleep(_INIT_POLL_INTERVAL)
        raise TimeoutError(f"Container init did not complete within {timeout}s")

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
