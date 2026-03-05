"""Agent definition: dataclass and loaders for agent.md files.

Agent definitions use YAML frontmatter + markdown body (Claude Code subagent
standard).  The frontmatter carries metadata; the markdown body becomes the
agent's system prompt.

Example agent.md::

    ---
    name: code-reviewer
    description: Expert code review specialist.
    purpose: Analyze code for bugs, security, and style
    tools: execute, smart_search
    model:
    ---

    You are an expert code reviewer...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger("forge_bot.agents.definition")


@dataclass
class AgentDefinition:
    """Parsed agent definition from an agent.md file."""

    name: str
    description: str
    purpose: str
    prompt: str
    tools: list[str] = field(default_factory=list)
    model: str = ""
    source_path: str = ""


def _parse_tools(raw: str | list[str] | None) -> list[str]:
    """Normalise the ``tools`` field to a list of strings."""
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(t).strip() for t in raw if str(t).strip()]
    return [t.strip() for t in raw.split(",") if t.strip()]


def parse_agent_md(content: str, source: str = "") -> AgentDefinition:
    """Parse an agent.md string into an :class:`AgentDefinition`.

    Parameters
    ----------
    content:
        Full file content (YAML frontmatter delimited by ``---``).
    source:
        Optional label for debugging (e.g. file path or API URL).

    Raises
    ------
    ValueError
        If the frontmatter is missing or required fields are absent.
    """
    content = content.strip()
    if not content.startswith("---"):
        raise ValueError(f"Agent definition missing YAML frontmatter: {source}")

    # Split on the second '---' delimiter.
    parts = content.split("---", 2)
    if len(parts) < 3:
        raise ValueError(f"Agent definition has incomplete frontmatter: {source}")

    front_raw = parts[1].strip()
    body = parts[2].strip()

    try:
        meta = yaml.safe_load(front_raw) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in agent frontmatter ({source}): {exc}") from exc

    if not isinstance(meta, dict):
        raise ValueError(f"Agent frontmatter is not a mapping: {source}")

    name = str(meta.get("name") or "")
    description = str(meta.get("description") or "")
    purpose = str(meta.get("purpose") or "")
    model = str(meta.get("model") or "")

    if not purpose:
        raise ValueError(f"Agent definition missing required 'purpose' field: {source}")

    tools = _parse_tools(meta.get("tools"))

    return AgentDefinition(
        name=name,
        description=description,
        purpose=purpose,
        prompt=body,
        tools=tools,
        model=model,
        source_path=source,
    )


def load_agent_md(path: Path) -> AgentDefinition:
    """Load and parse a single agent.md file.

    The ``name`` field defaults to the filename stem if not set in the
    frontmatter.
    """
    content = path.read_text(encoding="utf-8")
    defn = parse_agent_md(content, source=str(path))
    if not defn.name:
        defn.name = path.stem
    return defn


def load_agents_from_directory(directory: Path) -> list[AgentDefinition]:
    """Discover and load all ``*.md`` files in *directory*.

    Invalid files are logged at WARNING level and skipped.
    """
    if not directory.is_dir():
        return []

    agents: list[AgentDefinition] = []
    for md_path in sorted(directory.glob("*.md")):
        try:
            agents.append(load_agent_md(md_path))
        except Exception:
            logger.warning("Failed to load agent definition %s", md_path, exc_info=True)
    return agents
