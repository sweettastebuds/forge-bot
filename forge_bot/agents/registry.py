"""Agent registry: discovers and manages available agent definitions."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from forge_bot.agents.definition import (
    AgentDefinition,
    load_agents_from_directory,
    parse_agent_md,
)

if TYPE_CHECKING:
    from forge_bot.api.client import GenericForgeClient
    from forge_bot.container.manager import ContainerManager

logger = logging.getLogger("forge_bot.agents.registry")

# Directories inside a repository that may contain agent definitions.
_REPO_AGENT_DIRS = [".forge-bot/agents", ".forgejo/agents"]


class AgentRegistry:
    """Discover and manage available agent definitions.

    Built-in agents ship with forge-bot itself (``forge_bot/agents/*.md``).
    Repository agents live in ``.forge-bot/agents/`` or ``.forgejo/agents/``
    inside the target repo and override built-in agents with the same name.
    """

    def __init__(self) -> None:
        self._agents: dict[str, AgentDefinition] = {}
        self._repo_loaded: bool = False

    # -- loading -------------------------------------------------------------

    def load_builtin(self) -> None:
        """Load built-in agent definitions shipped with forge-bot."""
        builtin_dir = Path(__file__).parent
        for defn in load_agents_from_directory(builtin_dir):
            self._agents[defn.name] = defn
            logger.debug("Loaded built-in agent: %s", defn.name)

    async def load_repo_agents_api(
        self,
        api_client: GenericForgeClient,
        owner: str,
        repo: str,
    ) -> None:
        """Load repo agent definitions via the Forge API (no clone needed).

        Checks both ``.forge-bot/agents/`` and ``.forgejo/agents/`` in the
        repo.  Faster than container-based loading when the repo hasn't been
        cloned yet.
        """
        for agent_dir in _REPO_AGENT_DIRS:
            try:
                contents = await api_client.call(
                    "get_contents",
                    owner=owner,
                    repo=repo,
                    filepath=agent_dir,
                )
            except Exception:
                logger.debug("Repo dir %s not found via API (expected)", agent_dir)
                continue

            if not isinstance(contents, list):
                continue

            for entry in contents:
                name = entry.get("name", "")
                if not name.endswith(".md"):
                    continue
                filepath = f"{agent_dir}/{name}"
                try:
                    file_resp = await api_client.call(
                        "get_file_content",
                        owner=owner,
                        repo=repo,
                        filepath=filepath,
                    )
                    raw = file_resp.get("content", "")
                    if file_resp.get("encoding") == "base64":
                        import base64

                        raw = base64.b64decode(raw).decode("utf-8")
                    defn = parse_agent_md(raw, source=f"api:{owner}/{repo}/{filepath}")
                    if not defn.name:
                        defn.name = Path(name).stem
                    self._agents[defn.name] = defn
                    logger.info("Loaded repo agent via API: %s", defn.name)
                except Exception:
                    logger.warning("Failed to load repo agent %s", filepath, exc_info=True)

        self._repo_loaded = True

    async def load_repo_agents_container(self, container: ContainerManager) -> None:
        """Load repo agent definitions from the workspace container.

        Searches both ``.forge-bot/agents/`` and ``.forgejo/agents/`` inside
        ``/workspace``.  Use this after the repo has been cloned.
        """
        if self._repo_loaded:
            return

        for agent_dir in _REPO_AGENT_DIRS:
            ws_dir = f"/workspace/{agent_dir}"
            try:
                result = await container.exec(
                    f"ls {ws_dir}/*.md 2>/dev/null || true",
                    timeout=5,
                )
                if result.exit_code != 0 or not result.stdout.strip():
                    continue

                for md_path in result.stdout.strip().splitlines():
                    md_path = md_path.strip()
                    if not md_path:
                        continue
                    cat_result = await container.exec(f"cat '{md_path}'", timeout=5)
                    if cat_result.exit_code != 0:
                        continue
                    try:
                        defn = parse_agent_md(cat_result.stdout, source=f"container:{md_path}")
                        if not defn.name:
                            defn.name = Path(md_path).stem
                        self._agents[defn.name] = defn
                        logger.info("Loaded repo agent from container: %s", defn.name)
                    except Exception:
                        logger.warning("Failed to parse repo agent %s", md_path, exc_info=True)
            except Exception:
                logger.debug("Could not list %s in container", ws_dir, exc_info=True)

        self._repo_loaded = True

    # -- access --------------------------------------------------------------

    def register(self, defn: AgentDefinition) -> None:
        """Manually register an agent definition."""
        self._agents[defn.name] = defn

    def get(self, name: str) -> AgentDefinition | None:
        """Look up an agent by name."""
        return self._agents.get(name)

    def list_agents(self) -> list[AgentDefinition]:
        """Return all registered agents."""
        return list(self._agents.values())

    def describe_agents(self) -> str:
        """Format agent descriptions for inclusion in the orchestrator prompt."""
        if not self._agents:
            return "No agents available."
        lines = ["Available agents:"]
        for defn in self._agents.values():
            tools = ", ".join(defn.tools) if defn.tools else "none"
            lines.append(
                f"- **{defn.name}**: {defn.description} Purpose: {defn.purpose}. Tools: {tools}"
            )
        return "\n".join(lines)

    @property
    def repo_loaded(self) -> bool:
        """Whether repo agents have been loaded."""
        return self._repo_loaded
