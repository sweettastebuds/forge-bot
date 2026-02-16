"""Tests for the per-event container manager."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from forge_bot.config import Settings
from forge_bot.container.manager import ContainerManager, ExecResult


@pytest.fixture
def settings() -> Settings:
    return Settings(
        forge_instance_url="https://gitea.example.com",
        forge_api_token="test-token",
        forge_webhook_secret="test-secret",
        llm_api_key="test-llm-key",
        sandbox_timeout=60,
        sandbox_memory="512m",
        sandbox_cpus=1.0,
        container_workspace_image="forge-bot-workspace:latest",
    )


def _make_mock_container(*, logs: bytes = b"FORGE_READY") -> MagicMock:
    """Create a mock container that looks like docker SDK container."""
    container = MagicMock()
    container.short_id = "abc123"
    container.logs.return_value = logs
    container.remove.return_value = None

    # exec_run returns (exit_code, (stdout, stderr))
    exec_result = MagicMock()
    exec_result.exit_code = 0
    exec_result.output = (b"hello world\n", b"")
    container.exec_run.return_value = exec_result

    return container


def _make_mock_docker_client(container: MagicMock | None = None) -> MagicMock:
    """Create a mock docker.DockerClient."""
    client = MagicMock()
    mock_container = container or _make_mock_container()
    client.containers.run.return_value = mock_container
    client.close.return_value = None
    return client


class TestTokenInjection:
    def test_https_url(self) -> None:
        result = ContainerManager._inject_token(
            "https://gitea.example.com/owner/repo.git", "mytoken"
        )
        assert result == "https://mytoken@gitea.example.com/owner/repo.git"

    def test_http_url(self) -> None:
        result = ContainerManager._inject_token(
            "http://gitea.local/owner/repo.git", "tok123"
        )
        assert result == "http://tok123@gitea.local/owner/repo.git"

    def test_empty_token(self) -> None:
        url = "https://gitea.example.com/owner/repo.git"
        result = ContainerManager._inject_token(url, "")
        assert result == url

    def test_no_scheme(self) -> None:
        url = "git@gitea.example.com:owner/repo.git"
        result = ContainerManager._inject_token(url, "tok")
        assert result == url  # unchanged, no :// to split on


class TestCreate:
    @pytest.mark.asyncio
    async def test_creates_container(self, settings: Settings) -> None:
        mock_container = _make_mock_container()
        mock_client = _make_mock_docker_client(mock_container)

        cm = ContainerManager(
            settings,
            "https://gitea.example.com/owner/repo.git",
            "main",
            token="test-token",
        )

        with patch("forge_bot.container.manager.docker") as mock_docker:
            mock_docker.from_env.return_value = mock_client
            await cm.create()

        # Verify container was created with expected args
        mock_client.containers.run.assert_called_once()
        call_kwargs = mock_client.containers.run.call_args
        assert call_kwargs[0][0] == "forge-bot-workspace:latest"
        assert call_kwargs[1]["detach"] is True
        assert call_kwargs[1]["mem_limit"] == "512m"
        assert call_kwargs[1]["working_dir"] == "/workspace"

        await cm.destroy()

    @pytest.mark.asyncio
    async def test_init_timeout_raises(self, settings: Settings) -> None:
        mock_container = _make_mock_container(logs=b"still cloning...")
        mock_client = _make_mock_docker_client(mock_container)

        cm = ContainerManager(
            settings,
            "https://gitea.example.com/owner/repo.git",
            "main",
        )
        # Use very short timeout
        settings_short = Settings(
            forge_instance_url="https://gitea.example.com",
            forge_api_token="test-token",
            forge_webhook_secret="test-secret",
            llm_api_key="test-llm-key",
            sandbox_timeout=1,
        )
        cm._settings = settings_short

        with patch("forge_bot.container.manager.docker") as mock_docker:
            mock_docker.from_env.return_value = mock_client
            with pytest.raises(TimeoutError, match="did not complete"):
                await cm.create()

        await cm.destroy()


class TestExec:
    @pytest.mark.asyncio
    async def test_exec_success(self, settings: Settings) -> None:
        mock_container = _make_mock_container()
        mock_client = _make_mock_docker_client(mock_container)

        cm = ContainerManager(
            settings,
            "https://gitea.example.com/owner/repo.git",
            "main",
        )

        with patch("forge_bot.container.manager.docker") as mock_docker:
            mock_docker.from_env.return_value = mock_client
            await cm.create()

        result = await cm.exec("echo hello")
        assert isinstance(result, ExecResult)
        assert result.exit_code == 0
        assert result.stdout == "hello world\n"
        assert result.stderr == ""
        assert result.command == "echo hello"
        assert result.duration_seconds >= 0

        await cm.destroy()

    @pytest.mark.asyncio
    async def test_exec_failure_exit_code(self, settings: Settings) -> None:
        mock_container = _make_mock_container()
        exec_result = MagicMock()
        exec_result.exit_code = 1
        exec_result.output = (b"", b"error: not found\n")
        mock_container.exec_run.return_value = exec_result

        mock_client = _make_mock_docker_client(mock_container)

        cm = ContainerManager(
            settings,
            "https://gitea.example.com/owner/repo.git",
            "main",
        )

        with patch("forge_bot.container.manager.docker") as mock_docker:
            mock_docker.from_env.return_value = mock_client
            await cm.create()

        result = await cm.exec("false")
        assert result.exit_code == 1
        assert "error: not found" in result.stderr

        await cm.destroy()

    @pytest.mark.asyncio
    async def test_exec_without_create_raises(self, settings: Settings) -> None:
        cm = ContainerManager(
            settings,
            "https://gitea.example.com/owner/repo.git",
            "main",
        )
        with pytest.raises(RuntimeError, match="not created"):
            await cm.exec("echo hello")

    @pytest.mark.asyncio
    async def test_exec_truncates_long_output(self, settings: Settings) -> None:
        mock_container = _make_mock_container()
        long_output = b"x" * 10_000
        exec_result = MagicMock()
        exec_result.exit_code = 0
        exec_result.output = (long_output, b"")
        mock_container.exec_run.return_value = exec_result

        mock_client = _make_mock_docker_client(mock_container)

        cm = ContainerManager(
            settings,
            "https://gitea.example.com/owner/repo.git",
            "main",
        )

        with patch("forge_bot.container.manager.docker") as mock_docker:
            mock_docker.from_env.return_value = mock_client
            await cm.create()

        result = await cm.exec("cat big_file")
        assert len(result.stdout) <= 8_000 + 50  # truncation + marker
        assert "truncated" in result.stdout

        await cm.destroy()


class TestDestroy:
    @pytest.mark.asyncio
    async def test_destroy_removes_container(self, settings: Settings) -> None:
        mock_container = _make_mock_container()
        mock_client = _make_mock_docker_client(mock_container)

        cm = ContainerManager(
            settings,
            "https://gitea.example.com/owner/repo.git",
            "main",
        )

        with patch("forge_bot.container.manager.docker") as mock_docker:
            mock_docker.from_env.return_value = mock_client
            await cm.create()

        await cm.destroy()
        mock_container.remove.assert_called_once_with(force=True)

    @pytest.mark.asyncio
    async def test_destroy_ignores_removal_error(self, settings: Settings) -> None:
        mock_container = _make_mock_container()
        mock_container.remove.side_effect = Exception("already removed")
        mock_client = _make_mock_docker_client(mock_container)

        cm = ContainerManager(
            settings,
            "https://gitea.example.com/owner/repo.git",
            "main",
        )

        with patch("forge_bot.container.manager.docker") as mock_docker:
            mock_docker.from_env.return_value = mock_client
            await cm.create()

        # Should not raise
        await cm.destroy()

    @pytest.mark.asyncio
    async def test_destroy_without_create_is_safe(self, settings: Settings) -> None:
        cm = ContainerManager(
            settings,
            "https://gitea.example.com/owner/repo.git",
            "main",
        )
        # Should not raise
        await cm.destroy()


class TestAsyncContextManager:
    @pytest.mark.asyncio
    async def test_context_manager(self, settings: Settings) -> None:
        mock_container = _make_mock_container()
        mock_client = _make_mock_docker_client(mock_container)

        with patch("forge_bot.container.manager.docker") as mock_docker:
            mock_docker.from_env.return_value = mock_client

            async with ContainerManager(
                settings,
                "https://gitea.example.com/owner/repo.git",
                "main",
            ) as cm:
                result = await cm.exec("echo test")
                assert result.exit_code == 0

        # Container should be destroyed after context exit
        mock_container.remove.assert_called_once_with(force=True)


class TestNetworkMode:
    @pytest.mark.asyncio
    async def test_network_enabled(self, settings: Settings) -> None:
        mock_container = _make_mock_container()
        mock_client = _make_mock_docker_client(mock_container)

        cm = ContainerManager(
            settings,
            "https://gitea.example.com/owner/repo.git",
            "main",
            network_enabled=True,
        )

        with patch("forge_bot.container.manager.docker") as mock_docker:
            mock_docker.from_env.return_value = mock_client
            await cm.create()

        call_kwargs = mock_client.containers.run.call_args[1]
        assert call_kwargs["network_mode"] == "bridge"

        await cm.destroy()

    @pytest.mark.asyncio
    async def test_network_disabled(self, settings: Settings) -> None:
        mock_container = _make_mock_container()
        mock_client = _make_mock_docker_client(mock_container)

        cm = ContainerManager(
            settings,
            "https://gitea.example.com/owner/repo.git",
            "main",
            network_enabled=False,
        )

        with patch("forge_bot.container.manager.docker") as mock_docker:
            mock_docker.from_env.return_value = mock_client
            await cm.create()

        call_kwargs = mock_client.containers.run.call_args[1]
        assert call_kwargs["network_mode"] == "none"

        await cm.destroy()
