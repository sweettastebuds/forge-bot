"""Tests for forge_bot.retrieval.tool."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from forge_bot.retrieval.pipeline import SmartRetriever
from forge_bot.retrieval.tool import RetrievalTool


def _make_retriever() -> SmartRetriever:
    llm = AsyncMock()
    llm.chat.return_value = "keyword1\nkeyword2"
    return SmartRetriever(llm, context_window=8192)


async def test_tool_keyword_mode():
    retriever = _make_retriever()
    tool = RetrievalTool(retriever)

    result = await tool.execute(
        question="find error handling",
        content="def handle_error(e): log(e)\ndef other(): pass",
        mode="keyword",
        source_name="app.py",
    )

    assert result.success
    assert result.tool_name == "smart_search"


async def test_tool_missing_question():
    retriever = _make_retriever()
    tool = RetrievalTool(retriever)

    result = await tool.execute(content="some code")
    assert not result.success
    assert "question" in result.content.lower()


async def test_tool_missing_content():
    retriever = _make_retriever()
    tool = RetrievalTool(retriever)

    result = await tool.execute(question="find something")
    assert not result.success
    assert "content" in result.content.lower()


async def test_tool_openai_schema():
    retriever = _make_retriever()
    tool = RetrievalTool(retriever)
    schema = tool.to_openai_schema()

    assert schema["type"] == "function"
    assert schema["function"]["name"] == "smart_search"
    params = schema["function"]["parameters"]
    assert "question" in params["properties"]
    assert "content" in params["properties"]
    assert "question" in params["required"]
