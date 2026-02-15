"""Handler for issue_comment webhook events (@mention replies)."""

import logging

from forge_bot.handlers.base import BaseHandler
from forge_bot.models import IssueCommentEvent

logger = logging.getLogger("forge_bot.handlers.issue_comment")


class IssueCommentHandler(BaseHandler):
    """Respond to @mentions in issue/PR comments with LLM-generated answers."""

    async def handle(self, event: IssueCommentEvent) -> None:
        owner, repo = event.repository.full_name.split("/", 1)
        issue_num = event.issue.number

        logger.info(
            "Handling @mention on %s#%d by %s",
            event.repository.full_name,
            issue_num,
            event.sender.login,
        )

        # Fetch the full conversation thread for context.
        raw_comments = await self.forge.get_issue_comments(owner, repo, issue_num)
        thread_comments = [
            {
                "user": c.get("user", {}).get("login", "unknown"),
                "body": c.get("body", ""),
                "created_at": c.get("created_at", ""),
            }
            for c in raw_comments
        ]

        # Build the prompt from the Jinja2 template.
        system_prompt = self.render_template(
            "issue_respond.j2",
            repo_full_name=event.repository.full_name,
            issue_number=issue_num,
            issue_title=event.issue.title,
            rag_context=None,
            thread_comments=thread_comments,
        )

        # The user message is the triggering comment itself.
        user_message = event.comment.body

        # Call the LLM.
        try:
            reply = await self.llm.chat(system_prompt, user_message)
        except Exception:
            logger.exception("LLM call failed for %s#%d", event.repository.full_name, issue_num)
            reply = (
                "Sorry, I encountered an error while generating a response. "
                "Please try again later."
            )

        # Post the reply back to the issue/PR.
        try:
            await self.forge.post_comment(owner, repo, issue_num, reply)
            logger.info("Posted reply on %s#%d", event.repository.full_name, issue_num)
        except Exception:
            logger.exception(
                "Failed to post comment on %s#%d",
                event.repository.full_name,
                issue_num,
            )
