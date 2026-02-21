"""Handler for pull_request webhook events (code review).

Thin wrapper around BaseHandler._run_with_tools — fetches the diff and
changed file list, builds the user message, then delegates to the shared
container + tool-calling loop.
"""

from __future__ import annotations

import logging

from forge_bot.handlers.base import BaseHandler
from forge_bot.models import PullRequestEvent

logger = logging.getLogger("forge_bot.handlers.pull_request")

# Diffs larger than this are truncated in the user message.  The LLM
# can always fetch the full diff via tools.
_MAX_DIFF_CHARS = 60_000


class PullRequestHandler(BaseHandler):
    """Review PRs on opened/synchronized via the tool-calling loop."""

    async def handle(self, event: PullRequestEvent) -> None:
        owner, repo = event.repository.full_name.split("/", 1)
        pr_num = event.pull_request.number
        default_branch = event.repository.default_branch
        clone_url = event.repository.clone_url

        logger.info(
            "Reviewing PR %s#%d (%s) by %s",
            event.repository.full_name,
            pr_num,
            event.action,
            event.sender.login,
        )

        # Fetch diff and changed file list.
        try:
            diff_text = await self.api.call(
                "get_pull_diff", owner=owner, repo=repo, index=pr_num,
            )
        except Exception:
            logger.exception(
                "Failed to fetch diff for %s#%d",
                event.repository.full_name,
                pr_num,
            )
            diff_text = ""

        try:
            changed_files = await self.api.call(
                "get_pull_files", owner=owner, repo=repo, index=pr_num,
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

        # Ensure diff_text is a string.
        diff_text = str(diff_text)

        # Build a concise file summary for the prompt.
        if isinstance(changed_files, list):
            file_summary = "\n".join(
                f"- {f.get('filename', '?')} "
                f"(+{f.get('additions', 0)}/{-f.get('deletions', 0)})"
                for f in changed_files
            )
        else:
            file_summary = ""

        user_message = self._build_user_message(event, file_summary, diff_text)

        def _build_system_prompt(registry, clone_url, default_branch):  # noqa: ANN001
            return self.render_template(
                "pr_review_tools.j2",
                repo_full_name=event.repository.full_name,
                pr_number=event.pull_request.number,
                pr_title=event.pull_request.title,
                bot_username=self.bot_username,
                tool_descriptions=registry.prompt_text(),
                clone_url=clone_url,
                default_branch=default_branch,
            )

        await self._run_with_tools(
            owner=owner,
            repo=repo,
            issue_index=pr_num,
            clone_url=clone_url,
            default_branch=default_branch,
            build_system_prompt=_build_system_prompt,
            user_message=user_message,
            initial_phase="Reviewing PR...",
            error_message="Sorry, I encountered an error while reviewing this PR.",
        )

    # -- Helpers --

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

        if len(diff_text) > _MAX_DIFF_CHARS:
            truncated = diff_text[:_MAX_DIFF_CHARS]
            parts.append(
                f"\n**Diff (truncated — use tools to view full diff):**"
                f"\n```diff\n{truncated}\n```"
            )
            parts.append(
                "\n> The diff was truncated. Clone the repo and run "
                "`git diff origin/main` to see the full diff, "
                "or use `smart_search` to analyze specific parts."
            )
        else:
            parts.append(f"\n**Diff:**\n```diff\n{diff_text}\n```")

        return "\n".join(parts)
