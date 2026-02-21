"""Handler for pull_request webhook events (code review).

When the diff exceeds the token budget, the handler dynamically switches
to the three-level retrieval hierarchy to review the diff in chunks
rather than truncating it.
"""

from __future__ import annotations

import logging

from forge_bot.handlers.base import BaseHandler
from forge_bot.models import PullRequestEvent
from forge_bot.retrieval.pipeline import SmartRetriever
from forge_bot.retrieval.token_budget import estimate_tokens

logger = logging.getLogger("forge_bot.handlers.pull_request")

# Diffs larger than this are truncated in legacy (non-retrieval) mode.
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

        # Fetch diff and changed file list.
        try:
            diff_text = await self.api.call("get_pull_diff", owner=owner, repo=repo, index=pr_num)
        except Exception:
            logger.exception(
                "Failed to fetch diff for %s#%d",
                event.repository.full_name,
                pr_num,
            )
            diff_text = ""

        try:
            changed_files = await self.api.call(
                "get_pull_files", owner=owner, repo=repo, index=pr_num
            )
        except Exception:
            logger.exception(
                "Failed to fetch files for %s#%d",
                event.repository.full_name,
                pr_num,
            )
            changed_files = []

        if not diff_text:
            logger.warning(
                "Empty diff for %s#%d, skipping review",
                event.repository.full_name,
                pr_num,
            )
            return

        # Ensure diff_text is a string (call() may return dict for json endpoints).
        diff_text = str(diff_text)

        # Build a concise file summary for the prompt.
        if isinstance(changed_files, list):
            file_summary = "\n".join(
                f"- {f.get('filename', '?')} (+{f.get('additions', 0)}/{-f.get('deletions', 0)})"
                for f in changed_files
            )
        else:
            file_summary = ""

        # Decide: smart retrieval vs. legacy single-prompt.
        # estimate_tokens uses chars/4 — a rough heuristic. The threshold
        # at context_window/2 leaves room for the system prompt + response.
        # Dynamically switch to retrieval when the diff is too large
        # to fit comfortably alongside the system prompt + response.
        use_retrieval = (
            estimate_tokens(diff_text) > self.settings.llm_context_window // 2
        )

        if use_retrieval:
            review = await self._review_with_retrieval(
                event,
                diff_text,
                file_summary,
            )
        else:
            review = await self._review_legacy(
                event,
                diff_text,
                file_summary,
            )

        # Post the review as a regular comment (inline reviews are unreliable).
        try:
            await self.api.call(
                "post_issue_comment",
                owner=owner,
                repo=repo,
                index=pr_num,
                body=review,
            )
            logger.info("Posted review on %s#%d", event.repository.full_name, pr_num)
        except Exception:
            logger.exception(
                "Failed to post review on %s#%d",
                event.repository.full_name,
                pr_num,
            )

    async def _review_with_retrieval(
        self,
        event: PullRequestEvent,
        diff_text: str,
        file_summary: str,
    ) -> str:
        """Review using the smart retrieval hierarchy (Level 2 or 3)."""
        pr = event.pull_request
        question = (
            f"Review this pull request for bugs, security issues, "
            f"performance problems, and error handling.\n\n"
            f"PR #{pr.number}: {pr.title}\n"
            f"Branch: {pr.head.ref} → {pr.base.ref}\n"
        )
        if pr.body:
            question += f"Description: {pr.body}\n"
        if file_summary:
            question += f"\nChanged files:\n{file_summary}\n"

        retriever = SmartRetriever(
            self.llm,
            context_window=self.settings.llm_context_window,
            max_parallel=self.settings.smart_retrieval_max_parallel,
        )

        try:
            return await retriever.review_diff(question, diff_text)
        except Exception:
            logger.exception(
                "Smart retrieval failed for PR %s#%d, falling back to legacy",
                event.repository.full_name,
                event.pull_request.number,
            )
            return await self._review_legacy(event, diff_text, file_summary)

    async def _review_legacy(
        self,
        event: PullRequestEvent,
        diff_text: str,
        file_summary: str,
    ) -> str:
        """Legacy single-prompt review (truncates large diffs)."""
        # Truncate very large diffs to avoid exceeding LLM context.
        if len(diff_text) > _MAX_DIFF_CHARS:
            diff_text = diff_text[:_MAX_DIFF_CHARS] + "\n\n... (diff truncated)"

        system_prompt = self.render_template(
            "pr_review.j2",
            repo_full_name=event.repository.full_name,
            pr_title=event.pull_request.title,
        )

        user_message = self._build_user_message(event, file_summary, diff_text)

        try:
            return await self.llm.chat(system_prompt, user_message)
        except Exception:
            logger.exception(
                "LLM call failed for PR %s#%d",
                event.repository.full_name,
                event.pull_request.number,
            )
            return "Sorry, I encountered an error while reviewing this PR. Please try again later."

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
