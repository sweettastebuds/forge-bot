"""Handler for pull_request webhook events (code review).

Uses the AgentLoop so the LLM can clone the repo, generate diffs,
run linters / tests, and produce a thorough review autonomously.
"""

from __future__ import annotations

import logging

from forge_bot.agent import AgentLoop
from forge_bot.container.manager import ContainerManager
from forge_bot.handlers.base import BaseHandler
from forge_bot.models import PullRequestEvent
from forge_bot.retrieval.pipeline import SmartRetriever
from forge_bot.retrieval.tool import RetrievalTool
from forge_bot.status.manager import StatusCommentManager

logger = logging.getLogger("forge_bot.handlers.pull_request")


class PullRequestHandler(BaseHandler):
    """Review PRs using the agent loop with a workspace container."""

    async def handle(self, event: PullRequestEvent) -> None:
        owner, repo = event.repository.full_name.split("/", 1)
        pr = event.pull_request
        pr_num = pr.number

        logger.info(
            "Reviewing PR %s#%d (%s) by %s",
            event.repository.full_name,
            pr_num,
            event.action,
            event.sender.login,
        )

        # Post status comment for real-time observability.
        status = StatusCommentManager(self.api, owner, repo, pr_num)
        try:
            await status.post_initial_status()
        except Exception:
            logger.warning("Failed to post initial status", exc_info=True)

        # Create workspace container.
        container = ContainerManager(
            self.settings,
            event.repository.clone_url,
            event.repository.default_branch,
            token=self.settings.forge_api_token,
            network_enabled=self.settings.container_network_enabled,
            forge_url=self.settings.forge_instance_url,
            owner=owner,
            repo=repo,
        )
        try:
            await status.update_phase("Starting workspace...")
            await container.create()

            system_prompt = self.render_template(
                "agent_pr_review.j2",
                repo_full_name=event.repository.full_name,
                pr_number=pr_num,
                pr_title=pr.title,
                pr_body=pr.body or "",
                head_branch=pr.head.ref,
                base_branch=pr.base.ref,
                bot_username=self.bot_username,
                clone_url=container.clone_url,
                default_branch=event.repository.default_branch,
            )

            user_message = f"Review PR #{pr_num}: {pr.title}"

            # Set up smart retrieval tool.
            retriever = SmartRetriever(self.llm, context_window=self.settings.llm_context_window)
            search_tool = RetrievalTool(retriever, container)

            agent = AgentLoop(
                self.llm,
                container,
                self.settings,
                status=status,
                extra_tools=[search_tool],
            )
            review = await agent.run(system_prompt, user_message)

            # Collect and attach artifacts.
            artifacts = await agent.collect_artifacts()
            await status.post_response(review)
            for filename, content in artifacts:
                await status.attach_file(filename, content)

            await status.finalize_status("Done")
            logger.info("Posted review on %s#%d", event.repository.full_name, pr_num)
        except Exception:
            logger.exception(
                "Error reviewing %s#%d",
                event.repository.full_name,
                pr_num,
            )
            await self._post_error(status)
        finally:
            await container.destroy()

    # -- helpers -------------------------------------------------------------

    async def _post_error(self, status: StatusCommentManager) -> None:
        try:
            await status.post_response("Sorry, I encountered an error while reviewing this PR.")
            await status.finalize_status("Error")
        except Exception:
            logger.warning("Failed to post error response", exc_info=True)
