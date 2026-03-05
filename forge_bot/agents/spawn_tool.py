"""SpawnAgentTool: allows the orchestrator to delegate tasks to sub-agents."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from forge_bot.agents.registry import AgentRegistry
from forge_bot.tools.base import BaseTool, ToolParameter, ToolResult

if TYPE_CHECKING:
    from forge_bot.clients.llm import LLMClient
    from forge_bot.config import Settings
    from forge_bot.container.manager import ContainerManager
    from forge_bot.status.manager import StatusCommentManager

logger = logging.getLogger("forge_bot.agents.spawn_tool")


class SpawnAgentTool(BaseTool):
    """Spawn a specialized sub-agent to handle a focused subtask.

    The orchestrator uses this tool to delegate work to agents defined in
    ``agent.md`` files.  Sub-agents share the parent's container and status
    manager but run their own tool-calling loop.  Sub-agents **cannot** spawn
    further sub-agents (flat nesting).
    """

    name = "spawn_agent"
    description = (
        "Spawn a specialized sub-agent to handle a focused subtask. "
        "The sub-agent runs autonomously with its own tools and returns a result."
    )
    parameters = [
        ToolParameter(
            name="agent",
            type="string",
            description="Name of the agent to spawn (from the available agents list)",
        ),
        ToolParameter(
            name="task",
            type="string",
            description="Clear, specific task description for the sub-agent",
        ),
        ToolParameter(
            name="model",
            type="string",
            description="Override the LLM model for this agent (optional)",
            required=False,
        ),
    ]

    def __init__(
        self,
        registry: AgentRegistry,
        llm: LLMClient,
        container: ContainerManager,
        settings: Settings,
        available_tools: list[BaseTool],
        *,
        status: StatusCommentManager | None = None,
    ) -> None:
        self._registry = registry
        self._llm = llm
        self._container = container
        self._settings = settings
        self._available_tools = available_tools
        self._status = status

    async def execute(self, **kwargs: object) -> ToolResult:
        """Spawn a sub-agent and return its result."""
        agent_name = str(kwargs.get("agent", ""))
        task = str(kwargs.get("task", ""))
        model_override = str(kwargs.get("model", ""))

        if not agent_name:
            return ToolResult(self.name, False, "Error: 'agent' parameter is required.")
        if not task:
            return ToolResult(self.name, False, "Error: 'task' parameter is required.")

        # Lazy-load repo agents from the container if not yet loaded.
        if not self._registry.repo_loaded:
            try:
                await self._registry.load_repo_agents_container(self._container)
            except Exception:
                logger.warning("Failed to lazy-load repo agents", exc_info=True)

        defn = self._registry.get(agent_name)
        if not defn:
            available = ", ".join(a.name for a in self._registry.list_agents())
            return ToolResult(
                self.name,
                False,
                f"Error: unknown agent '{agent_name}'. Available: {available}",
            )

        # Filter tools to the agent's allowlist (never include spawn_agent).
        sub_tools = [
            t for t in self._available_tools if t.name in defn.tools and t.name != self.name
        ]

        # Determine LLM client — use model override if specified.
        llm = self._llm
        if model_override or defn.model:
            effective_model = model_override or defn.model
            llm = self._create_llm_with_model(effective_model)

        if self._status:
            await self._status.update_phase(f"Spawning agent: **{agent_name}**")

        sub_agent = self._create_agent_loop(
            llm,
            self._container,
            self._settings,
            status=self._status,
            extra_tools=sub_tools,
            agent_name=agent_name,
        )

        try:
            result = await sub_agent.run(defn.prompt, task)
            logger.info("Sub-agent '%s' completed successfully", agent_name)
            return ToolResult(self.name, True, result)
        except Exception:
            logger.exception("Sub-agent '%s' failed", agent_name)
            return ToolResult(
                self.name, False, f"Agent '{agent_name}' failed. Try a different approach."
            )

    def _create_agent_loop(self, *args: object, **kwargs: object) -> object:
        """Create an AgentLoop instance.  Deferred import avoids circular deps."""
        from forge_bot.agent import AgentLoop

        return AgentLoop(*args, **kwargs)

    def _create_llm_with_model(self, model: str) -> LLMClient:
        """Create a new LLMClient with a different model."""
        from forge_bot.clients.llm import LLMClient
        from forge_bot.config import Settings

        # Build a modified settings with the overridden model.
        overrides = {"llm_model": model}
        modified = Settings(
            **{
                field: getattr(self._settings, field)
                for field in self._settings.model_fields
                if field not in overrides
            },
            **overrides,
        )
        return LLMClient(modified)
