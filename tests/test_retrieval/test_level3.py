"""Tests for forge_bot.retrieval.level3."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from forge_bot.retrieval.level2 import Level2Scanner
from forge_bot.retrieval.level3 import Level3Agent, SourceDocument
from forge_bot.retrieval.token_budget import TokenBudget


def _make_scanner(answer: str = "Found relevant pattern.") -> Level2Scanner:
    """Mock Level2Scanner."""
    scanner = AsyncMock(spec=Level2Scanner)
    scanner.retrieve_and_answer.return_value = answer
    return scanner


def _make_llm_response(
    *,
    text: str | None = None,
    tool_calls: list[dict] | None = None,
) -> MagicMock:
    """Create a mock LLM response object."""
    response = MagicMock()
    message = MagicMock()

    if tool_calls:
        mock_tool_calls = []
        for tc in tool_calls:
            mock_tc = MagicMock()
            mock_tc.id = tc.get("id", "call_1")
            mock_tc.function.name = tc["name"]
            mock_tc.function.arguments = json.dumps(tc["arguments"])
            mock_tool_calls.append(mock_tc)
        message.tool_calls = mock_tool_calls
        message.content = text or ""
    else:
        message.tool_calls = None
        message.content = text or ""

    response.choices = [MagicMock(message=message)]
    return response


def _make_llm(responses: list) -> AsyncMock:
    """Mock LLM that returns a sequence of responses."""
    llm = AsyncMock()
    llm.chat_with_tools = AsyncMock(side_effect=responses)
    llm.chat = AsyncMock(return_value="Synthesized answer.")
    return llm


async def test_agent_decomposes_and_answers():
    # First call: agent makes a tool call
    # Second call: agent produces final answer
    responses = [
        _make_llm_response(tool_calls=[{
            "id": "call_1",
            "name": "search_context",
            "arguments": {
                "question": "What error handling patterns exist?",
                "source": "app.py",
            },
        }]),
        _make_llm_response(text="Based on my analysis, errors are logged."),
    ]
    llm = _make_llm(responses)
    scanner = _make_scanner("handle_error logs exceptions")
    budget = TokenBudget(context_window=8192)

    agent = Level3Agent(llm, scanner, budget)
    result = await agent.answer(
        "Does this code handle errors consistently?",
        [SourceDocument(name="app.py", content="def handle_error(e): log(e)")],
    )

    assert "errors" in result.lower() or "analysis" in result.lower()
    scanner.retrieve_and_answer.assert_awaited_once()


async def test_agent_no_sources():
    llm = AsyncMock()
    scanner = _make_scanner()
    budget = TokenBudget(context_window=8192)

    agent = Level3Agent(llm, scanner, budget)
    result = await agent.answer("anything", [])
    assert "no sources" in result.lower()


async def test_agent_direct_answer_no_tools():
    # LLM answers directly without tool calls
    responses = [
        _make_llm_response(text="The code looks consistent."),
    ]
    llm = _make_llm(responses)
    scanner = _make_scanner()
    budget = TokenBudget(context_window=8192)

    agent = Level3Agent(llm, scanner, budget)
    result = await agent.answer(
        "Is the code consistent?",
        [SourceDocument(name="a.py", content="code")],
    )

    assert "consistent" in result.lower()
    scanner.retrieve_and_answer.assert_not_awaited()


async def test_agent_handles_llm_failure():
    llm = AsyncMock()
    llm.chat_with_tools = AsyncMock(side_effect=RuntimeError("LLM down"))
    llm.chat = AsyncMock(return_value="fallback")
    scanner = _make_scanner()
    budget = TokenBudget(context_window=8192)

    agent = Level3Agent(llm, scanner, budget)
    result = await agent.answer(
        "anything",
        [SourceDocument(name="a.py", content="code")],
    )
    assert "unable" in result.lower() or "insufficient" in result.lower()


async def test_agent_max_hops_respected():
    # Agent always makes tool calls — should stop after max_hops
    def make_tool_response():
        return _make_llm_response(tool_calls=[{
            "id": "call_1",
            "name": "search_context",
            "arguments": {"question": "sub-q", "source": "a.py"},
        }])

    responses = [make_tool_response() for _ in range(10)]
    llm = _make_llm(responses)
    scanner = _make_scanner()
    budget = TokenBudget(context_window=8192)

    agent = Level3Agent(llm, scanner, budget, max_hops=3)
    result = await agent.answer(
        "complex question",
        [SourceDocument(name="a.py", content="code")],
    )

    # Should have called scanner at most 3 times
    assert scanner.retrieve_and_answer.await_count <= 3
