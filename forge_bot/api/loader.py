"""YAML API definition loader with caching."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from forge_bot.api.schema import ApiDefinitionFile

logger = logging.getLogger("forge_bot.api.loader")

_cache: dict[str, ApiDefinitionFile] = {}

DEFINITIONS_DIR = Path(__file__).parent / "definitions"


def load_api_definition(path: Path) -> ApiDefinitionFile:
    """Load and validate a YAML API definition file.

    Results are cached by resolved file path to avoid re-parsing.
    """
    key = str(path.resolve())
    if key in _cache:
        return _cache[key]

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    api_def = ApiDefinitionFile.model_validate(raw)
    _cache[key] = api_def
    logger.info(
        "Loaded API definition %s: %d endpoints",
        path.name,
        len(api_def.endpoints),
    )
    return api_def


def load_provider_definition(provider: str) -> ApiDefinitionFile:
    """Load the API definition file for a given provider.

    Args:
        provider: Provider name, e.g. ``"gitea"`` or ``"forgejo"``.

    Returns:
        The parsed API definition.

    Raises:
        FileNotFoundError: If no definition file exists for the provider.
    """
    path = DEFINITIONS_DIR / f"{provider}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No API definition file for provider '{provider}' at {path}")
    return load_api_definition(path)


def clear_cache() -> None:
    """Clear the definition cache (useful for tests)."""
    _cache.clear()
