"""Tests for forge_bot.retrieval.level3."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

from forge_bot.retrieval.level3 import Level3Agent, SourceDocument
from forge_bot.retrieval.token_budget import TokenBudget


def _make_budget_factory() -> tuple[MagicMock, TokenBudget]:
    """Return a budget factory and a sample budget."""
    budget = TokenBudget(context_window=8192)
    factory = MagicMock(return_value=budget)
    return factory, budget


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
        _make_llm_response(
            tool_calls=[
                {
                    "id": "call_1",
                    "name": "search_context",
                    "arguments": {
                        "question": "What error handling patterns exist?",
                        "source": "app.py",
                    },
                }
            ]
        ),
        _make_llm_response(text="Based on my analysis, errors are logged."),
    ]
    llm = _make_llm(responses)
    factory, _ = _make_budget_factory()

    # Patch Level2Scanner so the agent's per-hop scanner is mocked.
    with patch("forge_bot.retrieval.level3.Level2Scanner", autospec=True) as mock_scanner_cls:
        mock_instance = AsyncMock()
        mock_instance.retrieve_and_answer.return_value = "handle_error logs exceptions"
        mock_scanner_cls.return_value = mock_instance

        agent = Level3Agent(llm, budget_factory=factory)
        result = await agent.answer(
            "Does this code handle errors consistently?",
            [SourceDocument(name="app.py", content="def handle_error(e): log(e)")],
        )

    assert "errors" in result.lower() or "analysis" in result.lower()
    mock_instance.retrieve_and_answer.assert_awaited_once()
    # Budget factory should have been called once (one hop)
    factory.assert_called_once()


async def test_agent_no_sources():
    llm = AsyncMock()
    factory, _ = _make_budget_factory()

    agent = Level3Agent(llm, budget_factory=factory)
    result = await agent.answer("anything", [])
    assert "no sources" in result.lower()


async def test_agent_direct_answer_no_tools():
    # LLM answers directly without tool calls
    responses = [
        _make_llm_response(text="The code looks consistent."),
    ]
    llm = _make_llm(responses)
    factory, _ = _make_budget_factory()

    agent = Level3Agent(llm, budget_factory=factory)
    result = await agent.answer(
        "Is the code consistent?",
        [SourceDocument(name="a.py", content="code")],
    )

    assert "consistent" in result.lower()
    # No hops -- budget factory should not have been called
    factory.assert_not_called()


async def test_agent_handles_llm_failure():
    llm = AsyncMock()
    llm.chat_with_tools = AsyncMock(side_effect=RuntimeError("LLM down"))
    llm.chat = AsyncMock(return_value="fallback")
    factory, _ = _make_budget_factory()

    agent = Level3Agent(llm, budget_factory=factory)
    result = await agent.answer(
        "anything",
        [SourceDocument(name="a.py", content="code")],
    )
    assert "unable" in result.lower() or "insufficient" in result.lower()


async def test_agent_max_hops_respected():
    # Agent always makes tool calls -- should stop after max_hops
    def make_tool_response():
        return _make_llm_response(
            tool_calls=[
                {
                    "id": "call_1",
                    "name": "search_context",
                    "arguments": {"question": "sub-q", "source": "a.py"},
                }
            ]
        )

    responses = [make_tool_response() for _ in range(10)]
    llm = _make_llm(responses)
    factory, _ = _make_budget_factory()

    with patch("forge_bot.retrieval.level3.Level2Scanner", autospec=True) as mock_scanner_cls:
        mock_instance = AsyncMock()
        mock_instance.retrieve_and_answer.return_value = "some answer"
        mock_scanner_cls.return_value = mock_instance

        agent = Level3Agent(llm, budget_factory=factory, max_hops=3)
        await agent.answer(
            "complex question",
            [SourceDocument(name="a.py", content="code")],
        )

    # Should have called scanner at most 3 times
    assert mock_instance.retrieve_and_answer.await_count <= 3
    # Budget factory called once per hop
    assert factory.call_count <= 3


async def test_agent_creates_fresh_budget_per_hop():
    """Each hop should get its own fresh budget (not a shared one)."""
    responses = [
        _make_llm_response(
            tool_calls=[
                {
                    "id": "call_1",
                    "name": "search_context",
                    "arguments": {"question": "q1", "source": "a.py"},
                }
            ]
        ),
        _make_llm_response(
            tool_calls=[
                {
                    "id": "call_2",
                    "name": "search_context",
                    "arguments": {"question": "q2", "source": "a.py"},
                }
            ]
        ),
        _make_llm_response(text="Final answer."),
    ]
    llm = _make_llm(responses)

    budgets_created: list[TokenBudget] = []

    def budget_factory() -> TokenBudget:
        b = TokenBudget(context_window=8192)
        budgets_created.append(b)
        return b

    with patch("forge_bot.retrieval.level3.Level2Scanner", autospec=True) as mock_scanner_cls:
        mock_instance = AsyncMock()
        mock_instance.retrieve_and_answer.return_value = "hop result"
        mock_scanner_cls.return_value = mock_instance

        agent = Level3Agent(llm, budget_factory=budget_factory)
        await agent.answer(
            "multi-hop question",
            [SourceDocument(name="a.py", content="code")],
        )

    # Two hops means two distinct budget instances
    assert len(budgets_created) == 2
    assert budgets_created[0] is not budgets_created[1]
