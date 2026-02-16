"""Sandbox image registry and pre-pull logic."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("forge_bot.sandbox.images")

DEFAULT_IMAGES: dict[str, str] = {
    "python": "python:3.12-slim",
    "node": "node:22-slim",
    "go": "golang:1.23-alpine",
    "rust": "rust:1.77-slim",
    "c": "gcc:14",
    "default": "python:3.12-slim",
}


class ImageRegistry:
    """Map language keys to Docker image names, with optional overrides."""

    def __init__(self, defaults: dict[str, str] | None = None) -> None:
        self._images: dict[str, str] = dict(defaults or DEFAULT_IMAGES)

    def load_override_file(self, path: str) -> None:
        """Merge an external sandbox-images.json into the registry."""
        file_path = Path(path)
        if not file_path.is_file():
            logger.warning("Sandbox images file not found: %s", path)
            return
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self._images.update(data)
                logger.info(
                    "Loaded %d image overrides from %s",
                    len(data),
                    path,
                )
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load sandbox images file %s: %s", path, exc)

    def resolve(self, language: str) -> str | None:
        """Resolve a language key to a Docker image name.

        Returns *None* if the language is not in the registry.
        """
        return self._images.get(language.lower())

    def available_languages(self) -> list[str]:
        """Return sorted list of registered language keys (excluding 'default')."""
        return sorted(k for k in self._images if k != "default")

    async def prepull(
        self, docker_client: Any, keys: str, *, max_concurrent: int = 3,
    ) -> None:
        """Pre-pull sandbox images based on the SANDBOX_PREPULL_IMAGES value.

        *keys* is a comma-separated string of image keys, or "all" / "none".
        """
        if keys.strip().lower() == "none":
            logger.info("Sandbox image pre-pull disabled")
            return

        if keys.strip().lower() == "all":
            to_pull = list(self._images.values())
        else:
            requested = [k.strip() for k in keys.split(",") if k.strip()]
            to_pull = []
            for key in requested:
                image = self._images.get(key)
                if image:
                    to_pull.append(image)
                else:
                    logger.warning("Unknown image key '%s', skipping", key)

        # Deduplicate
        to_pull = list(dict.fromkeys(to_pull))
        if not to_pull:
            logger.info("No sandbox images to pre-pull")
            return

        sem = asyncio.Semaphore(max_concurrent)

        async def _pull_one(image: str) -> None:
            async with sem:
                try:
                    logger.info("Pre-pulling %s ...", image)
                    await asyncio.to_thread(docker_client.images.pull, image)
                    logger.info("Pre-pulled %s", image)
                except Exception:
                    logger.warning("Failed to pre-pull %s", image, exc_info=True)

        await asyncio.gather(*[_pull_one(img) for img in to_pull])
        logger.info(
            "Sandbox pre-pull complete: %d images requested", len(to_pull),
        )
