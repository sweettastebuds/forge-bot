"""Thin async wrapper around the OpenAI-compatible chat completions API."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from openai import AsyncOpenAI

from forge_bot.config import Settings

logger = logging.getLogger("forge_bot.clients.llm")


class LLMClient:
    """AsyncOpenAI wrapper with concurrency control via asyncio.Semaphore."""

    def __init__(self, settings: Settings) -> None:
        self._client = AsyncOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            timeout=float(settings.llm_timeout),
        )
        self._model = settings.llm_model
        self._temperature = settings.llm_temperature
        self._max_tokens = settings.llm_max_tokens
        self._semaphore = asyncio.Semaphore(settings.llm_max_concurrent)

    async def chat(
        self,
        system_prompt: str,
        user_message: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Send a chat completion request and return the assistant's reply.

        Blocks on the semaphore to respect LLM_MAX_CONCURRENT.
        """
        async with self._semaphore:
            effective_tokens = max_tokens or self._max_tokens
            logger.debug("LLM request: model=%s tokens=%d", self._model, effective_tokens)
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=temperature if temperature is not None else self._temperature,
                max_tokens=max_tokens or self._max_tokens,
            )
            if not response.choices:
                logger.warning("LLM returned empty choices")
                return ""
            content = response.choices[0].message.content or ""
            logger.debug("LLM response: %d chars", len(content))
            return content

    async def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Any:
        """Send a chat completion with tool definitions.

        Returns the raw response object so callers can inspect tool_calls.
        """
        async with self._semaphore:
            kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": messages,
                "temperature": temperature if temperature is not None else self._temperature,
                "max_tokens": max_tokens or self._max_tokens,
            }
            if tools:
                kwargs["tools"] = tools
            response = await self._client.chat.completions.create(**kwargs)
            return response

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.close()
