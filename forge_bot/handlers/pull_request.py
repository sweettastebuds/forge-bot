"""Handler for pull_request webhook events (code review)."""

import logging

from forge_bot.handlers.base import BaseHandler
from forge_bot.models import PullRequestEvent

logger = logging.getLogger("forge_bot.handlers.pull_request")

# Diffs larger than this are truncated to stay within LLM context limits.
_MAX_DIFF_CHARS = 30_000


class PullRequestHandler(BaseHandler):
    """Review PRs on opened/synchronized by posting an LLM-generated summary."""

    async def handle(self, event: PullRequestEvent) -> None:
        owner, repo = event.repository.full_name.split("/", 1)
        pr_num = event.pull_request.number

        logger.info(
            "Reviewing PR %s#%d (%s) by %s",
            event.repository.full_name,
            pr_num,
            event.action,
            event.sender.login,
        )

        # Fetch diff and changed file list in parallel-safe order.
        try:
            diff_text = await self.forge.get_pull_diff(owner, repo, pr_num)
        except Exception:
            logger.exception("Failed to fetch diff for %s#%d", event.repository.full_name, pr_num)
            diff_text = ""

        try:
            changed_files = await self.forge.get_pull_files(owner, repo, pr_num)
        except Exception:
            logger.exception("Failed to fetch files for %s#%d", event.repository.full_name, pr_num)
            changed_files = []

        if not diff_text:
            logger.warning(
                "Empty diff for %s#%d, skipping review",
                event.repository.full_name, pr_num,
            )
            return

        # Truncate very large diffs to avoid exceeding LLM context.
        if len(diff_text) > _MAX_DIFF_CHARS:
            diff_text = diff_text[:_MAX_DIFF_CHARS] + "\n\n... (diff truncated)"

        # Build a concise file summary for the prompt.
        file_summary = "\n".join(
            f"- {f.get('filename', '?')} (+{f.get('additions', 0)}/{-f.get('deletions', 0)})"
            for f in changed_files
        )

        # Render the system prompt from the Jinja2 template.
        system_prompt = self.render_template(
            "pr_review.j2",
            repo_full_name=event.repository.full_name,
            pr_title=event.pull_request.title,
            rag_context=None,
        )

        # User message = PR description + file list + diff.
        user_message = self._build_user_message(event, file_summary, diff_text)

        # Call the LLM.
        try:
            review = await self.llm.chat(system_prompt, user_message)
        except Exception:
            logger.exception("LLM call failed for PR %s#%d", event.repository.full_name, pr_num)
            review = (
                "Sorry, I encountered an error while reviewing this PR. "
                "Please try again later."
            )

        # Post the review as a regular comment (inline reviews are unreliable).
        try:
            await self.forge.post_comment(owner, repo, pr_num, review)
            logger.info("Posted review on %s#%d", event.repository.full_name, pr_num)
        except Exception:
            logger.exception(
                "Failed to post review on %s#%d",
                event.repository.full_name,
                pr_num,
            )

    @staticmethod
    def _build_user_message(
        event: PullRequestEvent,
        file_summary: str,
        diff_text: str,
    ) -> str:
        pr = event.pull_request
        parts = [
            f"## PR #{pr.number}: {pr.title}",
            f"**Branch:** {pr.head.ref} → {pr.base.ref}",
        ]
        if pr.body:
            parts.append(f"\n**Description:**\n{pr.body}")
        if file_summary:
            parts.append(f"\n**Changed files:**\n{file_summary}")
        parts.append(f"\n**Diff:**\n```diff\n{diff_text}\n```")
        return "\n".join(parts)
