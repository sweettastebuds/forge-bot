"""Handler for pull_request webhook events (code review).

Uses the same container + tool-calling loop as the issue comment handler.
The LLM receives the diff in the user message and can use tools (exec,
smart_search, api_call) for deeper analysis and cross-referencing.
"""

from __future__ import annotations

import logging

from forge_bot.container.manager import ContainerManager
from forge_bot.handlers.base import BaseHandler
from forge_bot.models import PullRequestEvent
from forge_bot.status.manager import StatusCommentManager
from forge_bot.tools.api_call import ApiCallTool
from forge_bot.tools.exec_tool import ExecTool
from forge_bot.tools.registry import ToolRegistry
from forge_bot.tools.search_api import SearchApiTool
from forge_bot.tools.todo import TodoTool

# Smart retrieval — always available, guarded import.
_RETRIEVAL_AVAILABLE = True
try:
    from forge_bot.retrieval.pipeline import SmartRetriever
    from forge_bot.retrieval.tool import RetrievalTool
except Exception:  # noqa: BLE001
    _RETRIEVAL_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "Smart retrieval import failed — tool will be unavailable",
        exc_info=True,
    )

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

        # Post status comment.
        status = StatusCommentManager(self.api, owner, repo, pr_num)
        try:
            await status.post_initial_status()
        except Exception:
            logger.warning("Failed to post initial status comment", exc_info=True)

        # Create workspace container.
        container: ContainerManager | None = None
        try:
            await status.update_phase("Starting workspace...")
            container = ContainerManager(
                self.settings,
                clone_url,
                default_branch,
                token=self.settings.forge_api_token,
                network_enabled=self.settings.container_network_enabled,
            )
            await container.create()
        except Exception:
            logger.exception("Failed to create workspace container")
            await self._post_error_response(
                status,
                "I couldn't set up a workspace to review this PR. "
                "Please try again later.",
            )
            if container:
                await container.destroy()
            return

        try:
            # Register tools.
            registry = ToolRegistry()
            registry.register(SearchApiTool(self.api))
            registry.register(ApiCallTool(self.api, owner=owner, repo=repo))
            registry.register(ExecTool(container))
            registry.register(TodoTool(status))

            if _RETRIEVAL_AVAILABLE:
                retriever = SmartRetriever(
                    self.llm,
                    context_window=self.settings.llm_context_window,
                    max_parallel=self.settings.smart_retrieval_max_parallel,
                )
                registry.register(RetrievalTool(retriever, container))

            # Build prompts and run tool loop.
            system_prompt = self._build_system_prompt(
                event,
                registry,
                clone_url=container.clone_url,
                default_branch=default_branch,
            )
            user_message = self._build_user_message(
                event, file_summary, diff_text,
            )

            await status.update_phase("Reviewing PR...")
            review, all_tool_results = await self._tool_loop(
                system_prompt=system_prompt,
                user_message=user_message,
                registry=registry,
                status=status,
            )

            # Pre-post verification.
            review = await self._verify_and_maybe_retry(
                review, all_tool_results, status,
            )

            # Post the review.
            try:
                await status.post_response(review)
                logger.info(
                    "Posted review on %s#%d",
                    event.repository.full_name,
                    pr_num,
                )
            except Exception:
                logger.exception(
                    "Failed to post review on %s#%d",
                    event.repository.full_name,
                    pr_num,
                )

            await status.finalize_status("Done")

        except Exception:
            logger.exception(
                "Error reviewing %s#%d",
                event.repository.full_name,
                pr_num,
            )
            await self._post_error_response(
                status,
                "Sorry, I encountered an error while reviewing this PR.",
            )
        finally:
            await container.destroy()

    # -- Helpers --

    def _build_system_prompt(
        self,
        event: PullRequestEvent,
        registry: ToolRegistry,
        *,
        clone_url: str,
        default_branch: str,
    ) -> str:
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
