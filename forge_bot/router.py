"""Event router: dispatches webhook payloads to the appropriate handler."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from forge_bot.handlers.issue_comment import IssueCommentHandler
from forge_bot.handlers.pull_request import PullRequestHandler
from forge_bot.models import (
    IssueCommentEvent,
    IssuesEvent,
    PullRequestEvent,
)

if TYPE_CHECKING:
    from forge_bot.clients.forge import ForgeClient
    from forge_bot.clients.llm import LLMClient
    from forge_bot.config import Settings

logger = logging.getLogger("forge_bot.router")

# Actions we care about for each event type.
_PR_ACTIONS = {"opened", "synchronized"}
_COMMENT_ACTIONS = {"created"}
_ISSUE_ACTIONS = {"opened", "assigned"}


def _mentions_user(text: str, username: str) -> bool:
    """Return True if *text* contains an @mention of *username*."""
    pattern = rf"(?:^|\s)@{re.escape(username)}(?:\s|$|[.,;:!?\)])"
    return bool(re.search(pattern, text, re.MULTILINE))


async def dispatch(
    event_type: str,
    payload: dict[str, Any],
    bot_username: str,
    forge_client: ForgeClient | None = None,
    llm_client: LLMClient | None = None,
    settings: Settings | None = None,
) -> None:
    """Route a webhook event to its handler.

    1. Self-loop guard — drop events sent by the bot itself.
    2. Parse into the appropriate Pydantic model.
    3. Dispatch to the matching handler.
    """
    sender_login = payload.get("sender", {}).get("login")

    # --- Self-loop guard ---
    if sender_login and sender_login == bot_username:
        logger.debug("Ignoring self-sent event from %s", sender_login)
        return

    # --- Pull request events ---
    if event_type == "pull_request":
        action = payload.get("action", "")
        if action not in _PR_ACTIONS:
            logger.debug("Ignoring pull_request action=%s", action)
            return
        event = PullRequestEvent.model_validate(payload)
        logger.info(
            "Routing pull_request/%s #%d in %s",
            event.action,
            event.number,
            event.repository.full_name,
        )
        if forge_client and llm_client and settings:
            handler = PullRequestHandler(forge_client, llm_client, settings, bot_username)
            await handler.handle(event)
        return

    # --- Issue comment events ---
    if event_type == "issue_comment":
        action = payload.get("action", "")
        if action not in _COMMENT_ACTIONS:
            logger.debug("Ignoring issue_comment action=%s", action)
            return
        event = IssueCommentEvent.model_validate(payload)

        # Only act if the bot is @mentioned in the comment body.
        if not _mentions_user(event.comment.body, bot_username):
            logger.debug("Comment #%d does not mention bot, skipping", event.comment.id)
            return

        logger.info(
            "Routing issue_comment/%s on #%d in %s",
            event.action,
            event.issue.number,
            event.repository.full_name,
        )
        if forge_client and llm_client and settings:
            handler = IssueCommentHandler(forge_client, llm_client, settings, bot_username)
            await handler.handle(event)
        return

    # --- Issue events (opened, assigned) ---
    if event_type == "issues":
        action = payload.get("action", "")
        if action not in _ISSUE_ACTIONS:
            logger.debug("Ignoring issues action=%s", action)
            return
        event = IssuesEvent.model_validate(payload)

        # For "assigned" events, only act if the bot is the assignee.
        if event.action == "assigned":
            assignee_logins = {a.login for a in event.issue.assignees}
            if bot_username not in assignee_logins:
                logger.debug("Issue #%d not assigned to bot, skipping", event.issue.number)
                return

        logger.info(
            "Routing issues/%s #%d in %s",
            event.action,
            event.issue.number,
            event.repository.full_name,
        )
        # TODO Phase 3: await issue_handler.handle(event)
        return

    logger.debug("Unhandled event type: %s", event_type)
