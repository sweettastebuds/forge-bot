"""Handler for issue_comment webhook events (@mention replies).

Uses the AgentLoop to give the LLM a shell-execution tool so it can
explore the codebase, call APIs, and write its own scripts at runtime.
"""

from __future__ import annotations

import logging

from forge_bot.agent import AgentLoop
from forge_bot.container.manager import ContainerManager
from forge_bot.handlers.base import BaseHandler
from forge_bot.models import IssueCommentEvent
from forge_bot.retrieval.pipeline import SmartRetriever
from forge_bot.retrieval.tool import RetrievalTool
from forge_bot.status.manager import StatusCommentManager

logger = logging.getLogger("forge_bot.handlers.issue_comment")


class IssueCommentHandler(BaseHandler):
    """Respond to @mentions using the agent loop with a workspace container."""

    async def handle(self, event: IssueCommentEvent) -> None:
        full_name = event.repository.full_name
        if "/" not in full_name:
            logger.error("Invalid repository full_name: %s", full_name)
            return
        owner, repo = full_name.split("/", 1)
        issue_num = event.issue.number

        logger.info(
            "Handling @mention on %s#%d by %s",
            event.repository.full_name,
            issue_num,
            event.sender.login,
        )

        # Post status comment for real-time observability.
        status = StatusCommentManager(self.api, owner, repo, issue_num)
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
                "agent_system.j2",
                repo_full_name=event.repository.full_name,
                issue_number=issue_num,
                issue_title=event.issue.title,
                bot_username=self.bot_username,
                clone_url=container.clone_url,
                default_branch=event.repository.default_branch,
            )

            # Build user message with conversation context.
            user_message = await self._build_user_message(owner, repo, issue_num, event)

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
            reply = await agent.run(system_prompt, user_message)

            # Collect and attach artifacts.
            artifacts = await agent.collect_artifacts()
            await status.post_response(reply)
            for filename, content in artifacts:
                await status.attach_file(filename, content)

            await status.finalize_status("Done")
        except Exception:
            logger.exception(
                "Error processing %s#%d",
                event.repository.full_name,
                issue_num,
            )
            await self._post_error(status)
        finally:
            await container.destroy()

    # -- helpers -------------------------------------------------------------

    async def _build_user_message(
        self,
        owner: str,
        repo: str,
        issue_num: int,
        event: IssueCommentEvent,
    ) -> str:
        """Build user message with conversation context from the issue thread."""
        context = await self._fetch_conversation(owner, repo, issue_num, event)
        if context:
            return (
                f"## Conversation so far:\n{context}\n\n## Current request:\n{event.comment.body}"
            )
        return event.comment.body

    async def _fetch_conversation(
        self,
        owner: str,
        repo: str,
        issue_num: int,
        event: IssueCommentEvent,
    ) -> str:
        """Fetch issue thread and build a condensed context string."""
        parts: list[str] = []

        # Include issue body if present.
        if event.issue.body:
            parts.append(f"**Issue description:**\n{event.issue.body[:500]}")

        # Fetch all comments.
        try:
            comments = await self.api.call(
                "get_issue_comments", owner=owner, repo=repo, index=issue_num
            )
        except Exception:
            logger.warning("Failed to fetch issue comments", exc_info=True)
            return "\n\n".join(parts) if parts else ""

        if not isinstance(comments, list):
            return "\n\n".join(parts) if parts else ""

        # Filter out the triggering comment, build condensed thread.
        char_budget = self.settings.llm_context_window * 4 // 6
        used = sum(len(p) for p in parts)
        for c in comments:
            if c.get("id") == event.comment.id:
                continue
            user = c.get("user", {}).get("login", "unknown")
            body = c.get("body", "")[:300]
            entry = f"@{user}: {body}"
            if used + len(entry) > char_budget:
                break
            parts.append(entry)
            used += len(entry)

        return "\n\n".join(parts)

    async def _post_error(self, status: StatusCommentManager) -> None:
        try:
            await status.post_response("Sorry, I encountered an error processing your request.")
            await status.finalize_status("Error")
        except Exception:
            logger.warning("Failed to post error response", exc_info=True)
