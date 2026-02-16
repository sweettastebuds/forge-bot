"""Tests for forge_bot.sandbox.orchestrator."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from forge_bot.config import Settings
from forge_bot.sandbox.images import ImageRegistry
from forge_bot.sandbox.orchestrator import ExecutionResult, SandboxOrchestrator
from forge_bot.sandbox.parser import RunCommand


@pytest.fixture()
def settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("FORGE_INSTANCE_URL", "https://gitea.example.com")
    monkeypatch.setenv("FORGE_API_TOKEN", "tok")
    monkeypatch.setenv("FORGE_WEBHOOK_SECRET", "sec")
    monkeypatch.setenv("LLM_API_KEY", "llm-key")
    monkeypatch.setenv("SANDBOX_TIMEOUT", "30")
    monkeypatch.setenv("SANDBOX_MEMORY", "256m")
    monkeypatch.setenv("SANDBOX_CPUS", "0.5")
    for var in ("LLM_BASE_URL", "LLM_MODEL", "LOG_LEVEL",
                "SANDBOX_IMAGES_FILE"):
        monkeypatch.delenv(var, raising=False)
    return Settings()


@pytest.fixture()
def registry() -> ImageRegistry:
    return ImageRegistry()


def _make_container(
    exit_code: int = 0,
    stdout: bytes = b"hello\n",
    stderr: bytes = b"",
    oom_killed: bool = False,
) -> MagicMock:
    """Create a mock Docker container."""
    container = MagicMock()
    container.wait.return_value = {"StatusCode": exit_code}
    container.logs.side_effect = (
        lambda stdout=True, stderr=True: stdout_val if stdout else stderr_val
    )
    # Need to bind using defaults
    stdout_val = stdout
    stderr_val = stderr
    container.logs.side_effect = lambda **kwargs: (
        stdout_val if kwargs.get("stdout", True) and not kwargs.get("stderr", True)
        else stderr_val if kwargs.get("stderr", True) and not kwargs.get("stdout", True)
        else stdout_val + stderr_val
    )
    container.attrs = {"State": {"OOMKilled": oom_killed}}
    container.reload.return_value = None
    container.remove.return_value = None
    return container


@pytest.mark.asyncio
async def test_execute_success(settings, registry):
    container = _make_container(exit_code=0, stdout=b"hello world\n")
    mock_client = MagicMock()
    mock_client.containers.run.return_value = container

    orch = SandboxOrchestrator(settings, registry)
    with patch("forge_bot.sandbox.orchestrator.docker") as mock_docker:
        mock_docker.from_env.return_value = mock_client
        await orch.connect()

    cmd = RunCommand(language="python", code="print('hello world')", network_enabled=False)
    result = await orch.execute(cmd)

    assert result.exit_code == 0
    assert "hello world" in result.stdout
    assert result.image == "python:3.12-slim"
    assert result.oom_killed is False
    assert result.duration_seconds >= 0
    container.remove.assert_called_once_with(force=True)


@pytest.mark.asyncio
async def test_execute_with_resource_limits(settings, registry):
    """Verify container is created with correct resource limits."""
    container = _make_container()
    mock_client = MagicMock()
    mock_client.containers.run.return_value = container

    orch = SandboxOrchestrator(settings, registry)
    orch._client = mock_client

    cmd = RunCommand(language="python", code="print('hi')", network_enabled=False)
    await orch.execute(cmd)

    call_kwargs = mock_client.containers.run.call_args
    assert call_kwargs.kwargs["mem_limit"] == "256m"
    assert call_kwargs.kwargs["nano_cpus"] == int(0.5 * 1e9)
    assert call_kwargs.kwargs["pids_limit"] == 256
    assert call_kwargs.kwargs["network_mode"] == "none"
    assert call_kwargs.kwargs["read_only"] is True


@pytest.mark.asyncio
async def test_execute_network_bridge(settings, registry):
    """Network mode should be 'bridge' when network_enabled=True."""
    container = _make_container()
    mock_client = MagicMock()
    mock_client.containers.run.return_value = container

    orch = SandboxOrchestrator(settings, registry)
    orch._client = mock_client

    cmd = RunCommand(language="node", code="console.log('hi')", network_enabled=True)
    await orch.execute(cmd)

    call_kwargs = mock_client.containers.run.call_args
    assert call_kwargs.kwargs["network_mode"] == "bridge"


@pytest.mark.asyncio
async def test_execute_nonzero_exit(settings, registry):
    container = _make_container(exit_code=1, stderr=b"NameError: name 'x' is not defined\n")
    mock_client = MagicMock()
    mock_client.containers.run.return_value = container

    orch = SandboxOrchestrator(settings, registry)
    orch._client = mock_client

    cmd = RunCommand(language="python", code="print(x)", network_enabled=False)
    result = await orch.execute(cmd)

    assert result.exit_code == 1
    assert "NameError" in result.stderr


@pytest.mark.asyncio
async def test_execute_oom_detected(settings, registry):
    container = _make_container(exit_code=137, oom_killed=True)
    mock_client = MagicMock()
    mock_client.containers.run.return_value = container

    orch = SandboxOrchestrator(settings, registry)
    orch._client = mock_client

    cmd = RunCommand(language="python", code="x = 'a' * 10**10", network_enabled=False)
    result = await orch.execute(cmd)

    assert result.oom_killed is True
    assert result.exit_code == 137


@pytest.mark.asyncio
async def test_execute_timeout(settings, registry):
    """Timeout exception should return a timeout result."""
    mock_client = MagicMock()
    container = MagicMock()
    container.wait.side_effect = Exception("read timeout")
    container.remove.return_value = None
    mock_client.containers.run.return_value = container

    orch = SandboxOrchestrator(settings, registry)
    orch._client = mock_client

    cmd = RunCommand(language="python", code="import time; time.sleep(999)", network_enabled=False)
    result = await orch.execute(cmd)

    assert result.exit_code == -1
    assert "timed out" in result.stderr.lower()


@pytest.mark.asyncio
async def test_execute_unknown_language(settings):
    """Unknown language with no default should raise ValueError."""
    registry = ImageRegistry(defaults={"only_known": "some:image"})
    mock_client = MagicMock()

    orch = SandboxOrchestrator(settings, registry)
    orch._client = mock_client

    cmd = RunCommand(language="brainfuck", code="++++", network_enabled=False)
    with pytest.raises(ValueError, match="Unknown language"):
        await orch.execute(cmd)


@pytest.mark.asyncio
async def test_execute_not_connected(settings, registry):
    """Calling execute() without connect() should raise RuntimeError."""
    orch = SandboxOrchestrator(settings, registry)

    cmd = RunCommand(language="python", code="print(1)", network_enabled=False)
    with pytest.raises(RuntimeError, match="Not connected"):
        await orch.execute(cmd)


@pytest.mark.asyncio
async def test_connect_and_close(settings, registry):
    with patch("forge_bot.sandbox.orchestrator.docker") as mock_docker:
        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client

        orch = SandboxOrchestrator(settings, registry)
        await orch.connect()
        assert orch._client is not None

        await orch.close()
        assert orch._client is None
        mock_client.close.assert_called_once()


@pytest.mark.asyncio
async def test_execute_container_removed_on_failure(settings, registry):
    """Container should be force-removed even if execution fails."""
    container = MagicMock()
    container.wait.side_effect = Exception("unexpected error")
    container.remove.return_value = None
    mock_client = MagicMock()
    mock_client.containers.run.return_value = container

    orch = SandboxOrchestrator(settings, registry)
    orch._client = mock_client

    cmd = RunCommand(language="python", code="print(1)", network_enabled=False)
    # "unexpected error" doesn't contain "timeout", so it should re-raise
    with pytest.raises(Exception, match="unexpected error"):
        await orch.execute(cmd)

    container.remove.assert_called_once_with(force=True)


@pytest.mark.asyncio
async def test_execute_node_language(settings, registry):
    """Node language should use correct file extension and command."""
    container = _make_container(stdout=b"hi\n")
    mock_client = MagicMock()
    mock_client.containers.run.return_value = container

    orch = SandboxOrchestrator(settings, registry)
    orch._client = mock_client

    cmd = RunCommand(language="node", code="console.log('hi')", network_enabled=False)
    result = await orch.execute(cmd)

    assert result.image == "node:22-slim"
    # Check that the shell command passed to containers.run includes node
    shell_cmd = mock_client.containers.run.call_args.args[1]
    assert "node /tmp/code.js" in " ".join(shell_cmd)
