"""Tests for the multi-agent system: definition, registry, and spawn tool."""

from __future__ import annotations

import textwrap
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from forge_bot.agents.definition import (
    AgentDefinition,
    load_agent_md,
    load_agents_from_directory,
    parse_agent_md,
)
from forge_bot.agents.registry import AgentRegistry
from forge_bot.agents.spawn_tool import SpawnAgentTool
from forge_bot.tools.base import BaseTool, ToolParameter, ToolResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_VALID_AGENT_MD = textwrap.dedent("""\
    ---
    name: test-agent
    description: A test agent for unit tests.
    purpose: Run unit tests
    tools: execute, smart_search
    model: gpt-4o
    ---

    You are a test agent. Do your thing.
""")

_MINIMAL_AGENT_MD = textwrap.dedent("""\
    ---
    description: Minimal agent.
    purpose: Minimal purpose
    tools: execute
    ---

    Minimal prompt.
""")

_NO_PURPOSE_MD = textwrap.dedent("""\
    ---
    name: bad-agent
    description: Missing purpose.
    tools: execute
    ---

    Some prompt.
""")


class _FakeTool(BaseTool):
    name = "fake_search"
    description = "A fake search tool."
    parameters = [ToolParameter(name="query", type="string", description="Search query")]

    async def execute(self, **kwargs: object) -> ToolResult:
        return ToolResult(tool_name=self.name, success=True, content="result")


# ---------------------------------------------------------------------------
# AgentDefinition / parsing
# ---------------------------------------------------------------------------


class TestParseAgentMd:
    def test_valid(self) -> None:
        defn = parse_agent_md(_VALID_AGENT_MD, source="test")
        assert defn.name == "test-agent"
        assert defn.description == "A test agent for unit tests."
        assert defn.purpose == "Run unit tests"
        assert defn.tools == ["execute", "smart_search"]
        assert defn.model == "gpt-4o"
        assert "test agent" in defn.prompt.lower()
        assert defn.source_path == "test"

    def test_minimal_no_name(self) -> None:
        defn = parse_agent_md(_MINIMAL_AGENT_MD)
        assert defn.name == ""  # no name in frontmatter
        assert defn.purpose == "Minimal purpose"
        assert defn.tools == ["execute"]

    def test_missing_purpose_raises(self) -> None:
        with pytest.raises(ValueError, match="purpose"):
            parse_agent_md(_NO_PURPOSE_MD)

    def test_missing_frontmatter_raises(self) -> None:
        with pytest.raises(ValueError, match="frontmatter"):
            parse_agent_md("Just some text without frontmatter.")

    def test_incomplete_frontmatter_raises(self) -> None:
        with pytest.raises(ValueError, match="incomplete"):
            parse_agent_md("---\nname: oops\n")

    def test_tools_yaml_list(self) -> None:
        md = textwrap.dedent("""\
            ---
            name: list-tools
            description: Agent with YAML list tools.
            purpose: Test list tools
            tools:
              - execute
              - smart_search
            ---

            Prompt.
        """)
        defn = parse_agent_md(md)
        assert defn.tools == ["execute", "smart_search"]

    def test_tools_empty(self) -> None:
        md = textwrap.dedent("""\
            ---
            name: no-tools
            description: Agent with no tools.
            purpose: Do nothing
            ---

            Prompt.
        """)
        defn = parse_agent_md(md)
        assert defn.tools == []

    def test_model_defaults_empty(self) -> None:
        defn = parse_agent_md(_MINIMAL_AGENT_MD)
        assert defn.model == ""


class TestLoadAgentMd:
    def test_load_from_file(self, tmp_path: Path) -> None:
        md_path = tmp_path / "my-agent.md"
        md_path.write_text(_VALID_AGENT_MD)
        defn = load_agent_md(md_path)
        assert defn.name == "test-agent"  # from frontmatter, not filename

    def test_name_from_filename(self, tmp_path: Path) -> None:
        md_path = tmp_path / "custom-name.md"
        md_path.write_text(_MINIMAL_AGENT_MD)
        defn = load_agent_md(md_path)
        assert defn.name == "custom-name"  # derived from filename stem


class TestLoadAgentsFromDirectory:
    def test_loads_multiple(self, tmp_path: Path) -> None:
        (tmp_path / "a.md").write_text(_VALID_AGENT_MD)
        (tmp_path / "b.md").write_text(_MINIMAL_AGENT_MD)
        agents = load_agents_from_directory(tmp_path)
        assert len(agents) == 2

    def test_skips_invalid(self, tmp_path: Path) -> None:
        (tmp_path / "good.md").write_text(_VALID_AGENT_MD)
        (tmp_path / "bad.md").write_text("not valid frontmatter")
        agents = load_agents_from_directory(tmp_path)
        assert len(agents) == 1
        assert agents[0].name == "test-agent"

    def test_missing_directory(self, tmp_path: Path) -> None:
        agents = load_agents_from_directory(tmp_path / "nonexistent")
        assert agents == []


# ---------------------------------------------------------------------------
# AgentRegistry
# ---------------------------------------------------------------------------


class TestAgentRegistry:
    def test_load_builtin(self) -> None:
        registry = AgentRegistry()
        registry.load_builtin()
        agents = registry.list_agents()
        names = {a.name for a in agents}
        # Built-in agents should be discovered.
        assert "code-reviewer" in names
        assert "test-runner" in names
        assert "planner" in names

    def test_get(self) -> None:
        registry = AgentRegistry()
        defn = AgentDefinition(
            name="my-agent",
            description="Test",
            purpose="Testing",
            prompt="Hello",
        )
        registry.register(defn)
        assert registry.get("my-agent") is defn
        assert registry.get("nonexistent") is None

    def test_override(self) -> None:
        registry = AgentRegistry()
        defn1 = AgentDefinition(name="agent-a", description="v1", purpose="v1", prompt="v1")
        defn2 = AgentDefinition(name="agent-a", description="v2", purpose="v2", prompt="v2")
        registry.register(defn1)
        registry.register(defn2)
        assert registry.get("agent-a").description == "v2"

    def test_describe_agents(self) -> None:
        registry = AgentRegistry()
        registry.register(
            AgentDefinition(
                name="test",
                description="Test agent.",
                purpose="Testing things",
                prompt="p",
                tools=["execute"],
            )
        )
        desc = registry.describe_agents()
        assert "test" in desc
        assert "Testing things" in desc
        assert "execute" in desc

    def test_describe_agents_empty(self) -> None:
        registry = AgentRegistry()
        assert "No agents" in registry.describe_agents()

    async def test_load_repo_agents_container(self) -> None:
        registry = AgentRegistry()
        container = AsyncMock()
        # Simulate no repo agents found.
        container.exec = AsyncMock(return_value=SimpleNamespace(exit_code=1, stdout="", stderr=""))
        await registry.load_repo_agents_container(container)
        assert registry.repo_loaded is True

    async def test_load_repo_agents_api(self) -> None:
        registry = AgentRegistry()
        api = AsyncMock()
        # Simulate directory not found.
        api.call = AsyncMock(side_effect=Exception("404"))
        await registry.load_repo_agents_api(api, "owner", "repo")
        assert registry.repo_loaded is True


# ---------------------------------------------------------------------------
# SpawnAgentTool
# ---------------------------------------------------------------------------


class TestSpawnAgentTool:
    def _make_spawn_tool(
        self,
        registry: AgentRegistry | None = None,
        available_tools: list[BaseTool] | None = None,
    ) -> tuple[SpawnAgentTool, AsyncMock, AsyncMock]:
        if registry is None:
            registry = AgentRegistry()
            registry.register(
                AgentDefinition(
                    name="test-agent",
                    description="Test.",
                    purpose="Test purpose",
                    prompt="You are a test agent.",
                    tools=["execute", "fake_search"],
                )
            )
            # Mark repo as loaded to skip lazy loading.
            registry._repo_loaded = True

        llm = AsyncMock()
        container = AsyncMock()
        settings = MagicMock()
        settings.llm_context_window = 8192
        settings.llm_max_tokens = 1024
        settings.container_timeout = 60
        status = AsyncMock()

        tool = SpawnAgentTool(
            registry,
            llm,
            container,
            settings,
            available_tools or [_FakeTool()],
            status=status,
        )
        return tool, llm, status

    def test_schema(self) -> None:
        tool, _, _ = self._make_spawn_tool()
        schema = tool.to_openai_schema()
        assert schema["type"] == "function"
        fn = schema["function"]
        assert fn["name"] == "spawn_agent"
        props = fn["parameters"]["properties"]
        assert "agent" in props
        assert "task" in props
        assert "model" in props
        assert "agent" in fn["parameters"]["required"]
        assert "task" in fn["parameters"]["required"]
        assert "model" not in fn["parameters"]["required"]

    async def test_unknown_agent(self) -> None:
        tool, _, _ = self._make_spawn_tool()
        result = await tool.execute(agent="nonexistent", task="do stuff")
        assert not result.success
        assert "unknown agent" in result.content

    async def test_missing_agent_param(self) -> None:
        tool, _, _ = self._make_spawn_tool()
        result = await tool.execute(task="do stuff")
        assert not result.success
        assert "'agent' parameter" in result.content

    async def test_missing_task_param(self) -> None:
        tool, _, _ = self._make_spawn_tool()
        result = await tool.execute(agent="test-agent")
        assert not result.success
        assert "'task' parameter" in result.content

    async def test_executes_subagent(self) -> None:
        tool, _, status = self._make_spawn_tool()
        mock_loop = AsyncMock()
        mock_loop.run = AsyncMock(return_value="Sub-agent result")
        factory = MagicMock(return_value=mock_loop)
        tool._create_agent_loop = factory

        result = await tool.execute(agent="test-agent", task="analyze this")

        assert result.success
        assert "Sub-agent result" in result.content
        # Verify AgentLoop was created with agent_name.
        factory.assert_called_once()
        call_kwargs = factory.call_args
        assert call_kwargs.kwargs["agent_name"] == "test-agent"

    async def test_filters_tools(self) -> None:
        """Sub-agent only receives tools in its allowlist."""
        registry = AgentRegistry()
        registry.register(
            AgentDefinition(
                name="restricted",
                description="Only execute.",
                purpose="Test",
                prompt="Prompt",
                tools=["execute"],  # Only execute, not fake_search
            )
        )
        registry._repo_loaded = True

        tool, _, _ = self._make_spawn_tool(
            registry=registry,
            available_tools=[_FakeTool()],  # fake_search, not in allowlist
        )
        mock_loop = AsyncMock()
        mock_loop.run = AsyncMock(return_value="Done")
        factory = MagicMock(return_value=mock_loop)
        tool._create_agent_loop = factory

        await tool.execute(agent="restricted", task="do it")

        call_kwargs = factory.call_args
        # fake_search should be filtered out since it's not in the allowlist.
        assert call_kwargs.kwargs["extra_tools"] == []

    async def test_no_recursive_spawn(self) -> None:
        """spawn_agent tool should never appear in sub-agent's tools."""
        registry = AgentRegistry()
        registry.register(
            AgentDefinition(
                name="agent-with-spawn",
                description="Wants spawn.",
                purpose="Test",
                prompt="Prompt",
                tools=["execute", "spawn_agent"],  # explicitly requests spawn_agent
            )
        )
        registry._repo_loaded = True

        # Create a spawn tool and include it in available_tools.
        tool, _, _ = self._make_spawn_tool(registry=registry)
        mock_loop = AsyncMock()
        mock_loop.run = AsyncMock(return_value="Done")
        factory = MagicMock(return_value=mock_loop)
        tool._create_agent_loop = factory

        await tool.execute(agent="agent-with-spawn", task="do it")

        call_kwargs = factory.call_args
        sub_tool_names = [t.name for t in call_kwargs.kwargs["extra_tools"]]
        assert "spawn_agent" not in sub_tool_names

    async def test_subagent_failure(self) -> None:
        tool, _, _ = self._make_spawn_tool()
        mock_loop = AsyncMock()
        mock_loop.run = AsyncMock(side_effect=RuntimeError("boom"))
        tool._create_agent_loop = MagicMock(return_value=mock_loop)

        result = await tool.execute(agent="test-agent", task="fail")

        assert not result.success
        assert "failed" in result.content.lower()

    async def test_status_update(self) -> None:
        tool, _, status = self._make_spawn_tool()
        mock_loop = AsyncMock()
        mock_loop.run = AsyncMock(return_value="Done")
        tool._create_agent_loop = MagicMock(return_value=mock_loop)

        await tool.execute(agent="test-agent", task="work")

        status.update_phase.assert_called_with("Spawning agent: **test-agent**")
