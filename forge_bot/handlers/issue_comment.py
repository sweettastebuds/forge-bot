"""Handler for issue_comment webhook events (@mention replies).

The LLM drives its own context gathering via tools (exec, api_call,
search_api, todo).  The handler manages the tool-calling loop, status
comment updates, container lifecycle, and verification checks.
"""

from __future__ import annotations

import logging

from forge_bot.container.manager import ContainerManager
from forge_bot.handlers.base import BaseHandler
from forge_bot.models import IssueCommentEvent
from forge_bot.status.manager import StatusCommentManager
from forge_bot.tools.api_call import ApiCallTool
from forge_bot.tools.exec_tool import ExecTool
from forge_bot.tools.registry import ToolRegistry
from forge_bot.tools.search_api import SearchApiTool
from forge_bot.tools.todo import TodoTool

# Smart retrieval — always available (no optional deps), but guarded
# so a broken import surfaces as a warning rather than crashing the handler.
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

logger = logging.getLogger("forge_bot.handlers.issue_comment")


class IssueCommentHandler(BaseHandler):
    """Respond to @mentions in issue/PR comments with LLM-generated answers.

    Flow:
    1. Post initial status comment
    2. Create workspace container (clone repo)
    3. Register tools and enter tool-calling loop with verification
    4. Post final response as separate comment
    5. Finalize status and destroy container
    """

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

        # 1. Post status comment
        status = StatusCommentManager(self.api, owner, repo, issue_num)
        try:
            await status.post_initial_status()
        except Exception:
            logger.warning("Failed to post initial status comment", exc_info=True)

        # 2. Create workspace container
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
            await status.update_phase("Error: container creation failed")
            await self._post_error_response(
                status,
                "I couldn't set up a workspace to analyze your request. Please try again later.",
            )
            if container:
                await container.destroy()
            return

        try:
            # 3. Register tools
            registry = ToolRegistry()
            registry.register(SearchApiTool(self.api))
            registry.register(ApiCallTool(self.api, owner=owner, repo=repo))
            registry.register(ExecTool(container))
            todo_tool = TodoTool(status)
            registry.register(todo_tool)

            # Register smart retrieval tool when available — the LLM
            # dynamically decides whether to invoke it per request.
            if _RETRIEVAL_AVAILABLE:
                retriever = SmartRetriever(
                    self.llm,
                    context_window=self.settings.llm_context_window,
                    max_parallel=self.settings.smart_retrieval_max_parallel,
                )
                registry.register(RetrievalTool(retriever, container))

            # 4. Tool-calling loop with verification
            system_prompt = self._build_system_prompt(
                event,
                registry,
                clone_url=container.clone_url,
                default_branch=default_branch,
            )
            await status.update_phase("Thinking...")
            reply, all_tool_results = await self._tool_loop(
                system_prompt=system_prompt,
                user_message=event.comment.body,
                registry=registry,
                status=status,
            )

            # 5. Pre-post verification
            reply = await self._verify_and_maybe_retry(reply, all_tool_results, status)

            # 6. Post response
            try:
                await status.post_response(reply)
                logger.info("Posted reply on %s#%d", event.repository.full_name, issue_num)
            except Exception:
                logger.exception(
                    "Failed to post response on %s#%d",
                    event.repository.full_name,
                    issue_num,
                )

            # 7. Finalize status
            await status.finalize_status("Done")

        except Exception:
            logger.exception("Error processing %s#%d", event.repository.full_name, issue_num)
            await self._post_error_response(
                status,
                "Sorry, I encountered an error while processing your request.",
            )
        finally:
            await container.destroy()

    # -- Helpers --

    def _build_system_prompt(
        self,
        event: IssueCommentEvent,
        registry: ToolRegistry,
        *,
        clone_url: str,
        default_branch: str,
    ) -> str:
        """Build the system prompt with context and tool descriptions."""
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
