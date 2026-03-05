"""Multi-agent system: definitions, registry, and spawn tool."""

from forge_bot.agents.definition import AgentDefinition, load_agent_md, parse_agent_md
from forge_bot.agents.registry import AgentRegistry
from forge_bot.agents.spawn_tool import SpawnAgentTool

__all__ = [
    "AgentDefinition",
    "AgentRegistry",
    "SpawnAgentTool",
    "load_agent_md",
    "parse_agent_md",
]
