"""Handler for issue_comment webhook events (@mention replies).

Thin wrapper around BaseHandler._run_with_tools — extracts event-specific
data and provides a system prompt builder callback.
"""

from __future__ import annotations

import logging

from forge_bot.handlers.base import BaseHandler
from forge_bot.models import IssueCommentEvent

logger = logging.getLogger("forge_bot.handlers.issue_comment")


class IssueCommentHandler(BaseHandler):
    """Respond to @mentions in issue/PR comments with LLM-generated answers."""

    async def handle(self, event: IssueCommentEvent) -> None:
        owner, repo = event.repository.full_name.split("/", 1)
        issue_num = event.issue.number
        default_branch = event.repository.default_branch
        clone_url = event.repository.clone_url

        logger.info(
            "Handling @mention on %s#%d by %s",
            event.repository.full_name,
            issue_num,
            event.sender.login,
        )

        def _build_system_prompt(registry, clone_url, default_branch):  # noqa: ANN001
            return self.render_template(
                "issue_respond.j2",
                repo_full_name=event.repository.full_name,
                issue_number=event.issue.number,
                issue_title=event.issue.title,
                bot_username=self.bot_username,
                tool_descriptions=registry.prompt_text(),
                clone_url=clone_url,
                default_branch=default_branch,
            )

        await self._run_with_tools(
            owner=owner,
            repo=repo,
            issue_index=issue_num,
            clone_url=clone_url,
            default_branch=default_branch,
            build_system_prompt=_build_system_prompt,
            user_message=event.comment.body,
            initial_phase="Thinking...",
            error_message="Sorry, I encountered an error while processing your request.",
        )
