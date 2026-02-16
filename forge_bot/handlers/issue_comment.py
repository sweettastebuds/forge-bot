"""Handler for issue_comment webhook events (@mention replies)."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from forge_bot.handlers.base import BaseHandler
from forge_bot.models import IssueCommentEvent
from forge_bot.sandbox.orchestrator import ExecutionResult
from forge_bot.sandbox.parser import RunCommand

logger = logging.getLogger("forge_bot.handlers.issue_comment")

# Models that have been observed to NOT support native tool calling.
# Populated at runtime by auto-fallback so we avoid repeating the failed
# native attempt on every subsequent request for the same model.
_models_without_tool_support: set[str] = set()

# --- Fixed limits (not model-dependent) ---
_MAX_FILE_FETCHES = 5
_MAX_GROUNDING_FILES = 3
_MAX_COMMIT_FETCHES = 3
_MAX_ATTACHMENT_DOWNLOADS = 3
_MAX_ATTACHMENT_CONTENT_CHARS = 6_000
_MAX_TOOL_ROUNDS = 5
_MAX_BOT_COMMENT_CHARS = 500


def _context_limits(context_window: int) -> dict[str, int]:
    """Derive context budgets from the model's context window size.

    All values scale with the context window so small models get tighter
    budgets and large models can use more context.
    """
    return {
        "max_recent_comments": min(max(context_window // 2000, 2), 10),
        "summary_max_tokens": min(max(context_window // 16, 128), 512),
        "max_tree_entries": min(max(context_window // 40, 50), 200),
        "max_file_chars": min(max(context_window * 2, 2000), 8000),
        "max_grounding_file_chars": min(max(context_window, 1000), 4000),
        "max_pr_diff_chars": min(max(context_window * 4, 4000), 30_000),
    }

# --- Grounding files to fetch proactively (in priority order) ---
_GROUNDING_FILES = [
    "README.md",
    "readme.md",
    "README",
    "pyproject.toml",
    "package.json",
    "Cargo.toml",
    "go.mod",
    "Makefile",
    "CLAUDE.md",
    ".forgejo/config.yaml",
    ".gitea/config.yaml",
]

# --- Text attachment extensions ---
_TEXT_EXTENSIONS = {
    ".md", ".txt", ".py", ".js", ".ts", ".go", ".rs", ".java", ".c", ".cpp",
    ".h", ".yml", ".yaml", ".toml", ".json", ".cfg", ".sh", ".csv", ".log",
    ".xml", ".html", ".css", ".sql", ".rb", ".pl",
}

# --- Regexes ---
_FILE_PATH_RE = re.compile(
    r"(?:^|[\s`'\"])"
    r"([\w./-]+\.(?:py|js|ts|go|rs|java|c|cpp|h|yml|yaml|toml|json|md|txt|cfg|sh))"
    r"(?:[\s`'\",;:!?)]|$)",
    re.MULTILINE,
)
_ATTACHMENT_RE = re.compile(
    r"!\[([^\]]*)\]\((/attachments/[^\)]+)\)",
)
_COMMIT_SHA_RE = re.compile(
    r"(?:^|\s)(?:commit\s+)?([0-9a-f]{7,40})(?:\s|$|[.,;:!?\)])",
    re.IGNORECASE | re.MULTILINE,
)
_FETCH_REQUEST_RE = re.compile(r"\[FETCH:\s*([^\]]+)\]")
_TOOL_CALL_RE = re.compile(r"```tool\s*\n(.*?)\n```", re.DOTALL)


# --- Helpers ---


def _extract_file_paths(text: str) -> list[str]:
    """Extract plausible file paths from text."""
    return list(dict.fromkeys(_FILE_PATH_RE.findall(text)))


def _extract_attachments(text: str, instance_url: str) -> list[dict[str, str]]:
    """Extract attachment markdown images/links from text."""
    results = []
    for alt, path in _ATTACHMENT_RE.findall(text):
        results.append({
            "alt": alt or "attachment",
            "url": f"{instance_url}{path}",
        })
    return results


def _extract_commit_shas(text: str) -> list[str]:
    """Extract plausible commit SHAs from text."""
    return list(dict.fromkeys(_COMMIT_SHA_RE.findall(text)))


def _get_extension(url: str) -> str:
    """Extract the lowercase file extension from a URL or path."""
    path = url.split("?")[0].split("#")[0]
    # Use only the last path segment to avoid matching dots in the domain.
    segment = path.rsplit("/", 1)[-1]
    dot_pos = segment.rfind(".")
    if dot_pos == -1:
        return ""
    return segment[dot_pos:].lower()


_MAX_OUTPUT_CHARS = 8_000

_FALLBACK_REPLY = (
    "I looked into it but wasn't able to form a complete "
    "answer. Could you provide more details?"
)


class IssueCommentHandler(BaseHandler):
    """Respond to @mentions in issue/PR comments with LLM-generated answers."""

    # --- Sandbox /run handling ---

    async def handle_run(
        self, event: IssueCommentEvent, command: RunCommand,
    ) -> None:
        """Execute code in a sandbox and post results."""
        from forge_bot.sandbox.images import ImageRegistry
        from forge_bot.sandbox.orchestrator import SandboxOrchestrator

        owner, repo = event.repository.full_name.split("/", 1)
        issue_num = event.issue.number

        if not self.settings.sandbox_enabled:
            await self.forge.post_comment(
                owner, repo, issue_num,
                "Sandbox execution is disabled on this instance.",
            )
            return

        registry = ImageRegistry()
        if self.settings.sandbox_images_file:
            registry.load_override_file(self.settings.sandbox_images_file)

        image = registry.resolve(command.language)
        if image is None:
            available = ", ".join(registry.available_languages())
            await self.forge.post_comment(
                owner, repo, issue_num,
                f"Unknown language `{command.language}`. "
                f"Supported languages: {available}",
            )
            return

        orchestrator = SandboxOrchestrator(self.settings, registry)
        try:
            await orchestrator.connect()
            result = await orchestrator.execute(command)
        except Exception as exc:
            logger.exception(
                "Sandbox execution failed for %s#%d",
                event.repository.full_name, issue_num,
            )
            await self.forge.post_comment(
                owner, repo, issue_num,
                f"Sandbox error: `{exc}`",
            )
            return
        finally:
            await orchestrator.close()

        reply = self._format_execution_result(result, command)
        try:
            await self.forge.post_comment(owner, repo, issue_num, reply)
            logger.info(
                "Posted sandbox result on %s#%d (exit=%d)",
                event.repository.full_name, issue_num, result.exit_code,
            )
        except Exception:
            logger.exception(
                "Failed to post sandbox result on %s#%d",
                event.repository.full_name, issue_num,
            )

    @staticmethod
    def _format_execution_result(
        result: ExecutionResult, command: RunCommand,
    ) -> str:
        """Format an ExecutionResult as a markdown comment."""
        lines: list[str] = []

        if result.oom_killed:
            lines.append(
                "**Out of Memory** — the program exceeded the memory limit "
                "and was killed.",
            )
        elif result.exit_code != 0:
            lines.append(f"**Exit code {result.exit_code}**")
        else:
            lines.append("**Execution succeeded**")

        # stdout
        stdout = result.stdout.strip()
        if stdout:
            if len(stdout) > _MAX_OUTPUT_CHARS:
                stdout = stdout[:_MAX_OUTPUT_CHARS] + "\n... (truncated)"
            lines.append(f"\n**stdout**\n```\n{stdout}\n```")

        # stderr
        stderr = result.stderr.strip()
        if stderr:
            if len(stderr) > _MAX_OUTPUT_CHARS:
                stderr = stderr[:_MAX_OUTPUT_CHARS] + "\n... (truncated)"
            lines.append(f"\n**stderr**\n```\n{stderr}\n```")

        if not stdout and not stderr:
            lines.append("\n*(no output)*")

        # Footer
        lines.append(
            f"\n---\n"
            f"*Language: `{command.language}` · "
            f"Image: `{result.image}` · "
            f"Duration: {result.duration_seconds:.1f}s*"
        )

        return "\n".join(lines)

    # --- RAG /index handling ---

    async def handle_index(self, event: IssueCommentEvent) -> None:
        """Re-index the repository and post a status comment."""
        owner, repo = event.repository.full_name.split("/", 1)
        issue_num = event.issue.number
        default_branch = event.repository.default_branch

        if not self.settings.rag_enabled:
            await self.forge.post_comment(
                owner, repo, issue_num,
                "RAG indexing is disabled on this instance "
                "(set `RAG_ENABLED=true` to enable).",
            )
            return

        try:
            from forge_bot.rag.pipeline import RAGPipeline

            pipeline = RAGPipeline(self.settings, self.forge)
            count = await pipeline.ensure_indexed(
                owner, repo, default_branch, force=True,
            )
            await self.forge.post_comment(
                owner, repo, issue_num,
                f"Re-indexed repository — **{count}** chunks stored.",
            )
            logger.info(
                "Re-indexed %s: %d chunks", event.repository.full_name, count,
            )
        except Exception as exc:
            logger.exception(
                "Failed to index %s", event.repository.full_name,
            )
            await self.forge.post_comment(
                owner, repo, issue_num,
                f"Indexing failed: `{exc}`",
            )

    # --- @mention Q&A handling ---

    async def handle(self, event: IssueCommentEvent) -> None:
        owner, repo = event.repository.full_name.split("/", 1)
        issue_num = event.issue.number
        default_branch = event.repository.default_branch
        limits = _context_limits(self.settings.llm_context_window)

        logger.info(
            "Handling @mention on %s#%d by %s (context_window=%d)",
            event.repository.full_name,
            issue_num,
            event.sender.login,
            self.settings.llm_context_window,
        )

        # Fetch the full conversation thread for context.
        raw_comments = await self.forge.get_issue_comments(
            owner, repo, issue_num
        )
        thread_comments = [
            {
                "user": c.get("user", {}).get("login", "unknown"),
                "body": c.get("body", ""),
                "created_at": c.get("created_at", ""),
            }
            for c in raw_comments
        ]

        # Truncate bot's own comments to avoid long hallucinated responses
        # from dominating the context window on subsequent requests.
        for comment in thread_comments:
            if (
                comment["user"] == self.bot_username
                and len(comment["body"]) > _MAX_BOT_COMMENT_CHARS
            ):
                comment["body"] = (
                    comment["body"][:_MAX_BOT_COMMENT_CHARS]
                    + "\n\n*(response truncated)*"
                )

        # --- Conversation trimming: summarize old comments ---
        conversation_summary: str | None = None
        recent_comments = thread_comments
        max_recent = limits["max_recent_comments"]

        if len(thread_comments) > max_recent:
            split_index = len(thread_comments) - max_recent
            old_comments = thread_comments[:split_index]
            recent_comments = thread_comments[split_index:]
            conversation_summary = await self._summarize_old_comments(
                old_comments,
                event.repository.full_name,
                event.issue.title,
                self.bot_username,
                max_tokens=limits["summary_max_tokens"],
            )
            logger.info(
                "Summarized %d old comments into %d chars for %s#%d",
                len(old_comments),
                len(conversation_summary),
                event.repository.full_name,
                issue_num,
            )

        # --- Repo context: tree + referenced files ---
        repo_tree_text, tree_paths = await self._fetch_repo_tree(
            owner, repo, default_branch, limits
        )
        file_context = await self._fetch_referenced_files(
            owner, repo, event, thread_comments, default_branch, limits
        )

        # Proactively fetch grounding files if few/no referenced files.
        if len(file_context) < _MAX_GROUNDING_FILES and tree_paths:
            already_fetched = {f["path"] for f in file_context}
            grounding = await self._fetch_grounding_files(
                owner, repo, tree_paths, already_fetched, default_branch,
                limits,
            )
            file_context.extend(grounding)

        # --- Commit context ---
        commit_context = await self._fetch_referenced_commits(
            owner, repo, event, thread_comments
        )

        # --- Attachment awareness ---
        instance_url = self.settings.forge_instance_url.rstrip("/")
        attachments = await self._collect_attachments(
            event, raw_comments, instance_url
        )

        # --- PR context (when comment is on a pull request) ---
        pr_diff = ""
        pr_files_summary = ""
        is_pull = event.is_pull or event.issue.is_pull
        if is_pull:
            pr_diff, pr_files_summary = await self._fetch_pr_context(
                owner, repo, issue_num, limits,
            )

        # --- RAG context (optional) ---
        rag_context = None
        if self.settings.rag_enabled:
            try:
                from forge_bot.rag.pipeline import RAGPipeline

                pipeline = RAGPipeline(self.settings, self.forge)
                rag_context = await pipeline.retrieve(
                    owner, repo, event.comment.body,
                    top_k=self.settings.rag_top_k,
                )
                if rag_context:
                    logger.info(
                        "RAG context retrieved for %s#%d (%d chars)",
                        event.repository.full_name, issue_num, len(rag_context),
                    )
            except Exception:
                logger.warning(
                    "RAG retrieval failed for %s#%d, continuing without",
                    event.repository.full_name, issue_num,
                    exc_info=True,
                )

        # --- Build prompt and run LLM with tool-calling loop ---
        reply = await self._run_with_tool_loop(
            owner=owner,
            repo=repo,
            event=event,
            issue_num=issue_num,
            thread_comments=recent_comments,
            conversation_summary=conversation_summary,
            repo_tree_text=repo_tree_text,
            tree_paths=tree_paths,
            file_context=file_context,
            commit_context=commit_context,
            attachments=attachments,
            rag_context=rag_context,
            default_branch=default_branch,
            limits=limits,
            pr_diff=pr_diff,
            pr_files_summary=pr_files_summary,
        )

        # Post the reply back to the issue/PR.
        try:
            await self.forge.post_comment(owner, repo, issue_num, reply)
            logger.info(
                "Posted reply on %s#%d",
                event.repository.full_name,
                issue_num,
            )
        except Exception:
            logger.exception(
                "Failed to post comment on %s#%d",
                event.repository.full_name,
                issue_num,
            )

    # --- LLM call with tool-calling loop ---

    async def _run_with_tool_loop(
        self,
        *,
        owner: str,
        repo: str,
        event: IssueCommentEvent,
        issue_num: int,
        thread_comments: list[dict[str, str]],
        conversation_summary: str | None,
        repo_tree_text: str,
        tree_paths: set[str],
        file_context: list[dict[str, str]],
        commit_context: list[dict[str, str]],
        attachments: list[dict[str, str]],
        default_branch: str,
        limits: dict[str, int],
        rag_context: str | None = None,
        pr_diff: str = "",
        pr_files_summary: str = "",
    ) -> str:
        """Call the LLM with tool support, executing tool calls in a loop."""
        from forge_bot.tools.fetch_file import FetchFileTool
        from forge_bot.tools.get_commit import GetCommitTool
        from forge_bot.tools.registry import ToolRegistry
        from forge_bot.tools.search_code import SearchCodeTool

        max_file_chars = limits["max_file_chars"]

        # Build per-request tool registry.
        registry = ToolRegistry()
        registry.register(FetchFileTool(
            self.forge, owner, repo, default_branch,
            max_chars=max_file_chars,
        ))
        registry.register(GetCommitTool(self.forge, owner, repo))
        registry.register(SearchCodeTool(
            self.forge, owner, repo, default_branch,
            tree_paths=sorted(tree_paths),
        ))

        # Determine tool mode.
        tool_mode = self.settings.llm_tool_mode
        use_native = tool_mode in ("native", "auto")

        # Skip native if we already know this model doesn't support tools.
        model_name = self.settings.llm_model
        if use_native and model_name in _models_without_tool_support:
            logger.debug(
                "Skipping native tool calling for model %s (cached)",
                model_name,
            )
            use_native = False

        # For prompt mode, inject tool descriptions into the system prompt.
        tool_descriptions = "" if use_native else registry.prompt_text()

        system_prompt = self._build_system_prompt(
            event=event,
            issue_num=issue_num,
            thread_comments=thread_comments,
            conversation_summary=conversation_summary,
            repo_tree_text=repo_tree_text,
            file_context=file_context,
            commit_context=commit_context,
            attachments=attachments,
            rag_context=rag_context,
            tool_descriptions=tool_descriptions,
            pr_diff=pr_diff,
            pr_files_summary=pr_files_summary,
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": event.comment.body},
        ]

        # --- Trace state ---
        prompt_chars = len(system_prompt)
        tool_calls_log: list[str] = []
        fallback_triggered = False
        final_mode = "native" if use_native else "prompt"
        completed_rounds = 0

        for round_num in range(_MAX_TOOL_ROUNDS):
            try:
                if use_native:
                    content = await self._native_tool_round(
                        messages, registry, event, issue_num, round_num,
                    )
                else:
                    content = await self._prompt_tool_round(
                        messages, registry, event, issue_num, round_num,
                    )

                completed_rounds = round_num + 1

                if content is not None:
                    # Tool round returned final content — done.
                    self._log_trace(
                        event, issue_num, final_mode, completed_rounds,
                        tool_calls_log, fallback_triggered, prompt_chars,
                        len(content),
                    )
                    return content

                # None means tool calls were made — record them.
                # Scan the last messages for tool results.
                for msg in reversed(messages):
                    if msg.get("role") == "tool":
                        tool_calls_log.append(msg.get("tool_call_id", "?"))
                        break
                    if msg.get("role") == "user" and "**" in msg.get(
                        "content", "",
                    ):
                        # Prompt-mode tool results.
                        tool_calls_log.append("prompt-tool")
                        break
                continue

            except Exception:
                if use_native and tool_mode == "auto":
                    # Auto mode: try falling back to prompt mode.
                    logger.warning(
                        "Native tool calling failed for %s#%d, "
                        "falling back to prompt mode",
                        event.repository.full_name, issue_num,
                        exc_info=True,
                    )
                    use_native = False
                    fallback_triggered = True
                    final_mode = "prompt"
                    _models_without_tool_support.add(model_name)
                    tool_descriptions = registry.prompt_text()
                    system_prompt = self._build_system_prompt(
                        event=event,
                        issue_num=issue_num,
                        thread_comments=thread_comments,
                        conversation_summary=conversation_summary,
                        repo_tree_text=repo_tree_text,
                        file_context=file_context,
                        commit_context=commit_context,
                        attachments=attachments,
                        rag_context=rag_context,
                        tool_descriptions=tool_descriptions,
                        pr_diff=pr_diff,
                        pr_files_summary=pr_files_summary,
                    )
                    prompt_chars = len(system_prompt)
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": event.comment.body},
                    ]
                    continue

                logger.exception(
                    "LLM call failed for %s#%d (round %d)",
                    event.repository.full_name, issue_num, round_num,
                )
                reply = (
                    "Sorry, I encountered an error while generating a "
                    "response. Please try again later."
                )
                self._log_trace(
                    event, issue_num, final_mode, round_num + 1,
                    tool_calls_log, fallback_triggered, prompt_chars,
                    len(reply),
                )
                return reply

        # Exhausted all rounds — extract the last assistant content.
        reply = self._extract_last_reply(messages)
        self._log_trace(
            event, issue_num, final_mode, _MAX_TOOL_ROUNDS,
            tool_calls_log, fallback_triggered, prompt_chars, len(reply),
        )
        return reply

    @staticmethod
    def _log_trace(
        event: IssueCommentEvent,
        issue_num: int,
        mode: str,
        rounds: int,
        tool_calls: list[str],
        fallback: bool,
        prompt_chars: int,
        reply_chars: int,
    ) -> None:
        """Log a structured per-request summary for debugging."""
        logger.info(
            "TRACE %s#%d | mode=%s rounds=%d tools=%d "
            "fallback=%s prompt=%dc reply=%dc",
            event.repository.full_name,
            issue_num,
            mode,
            rounds,
            len(tool_calls),
            fallback,
            prompt_chars,
            reply_chars,
        )

    def _build_system_prompt(
        self,
        *,
        event: IssueCommentEvent,
        issue_num: int,
        thread_comments: list[dict[str, str]],
        conversation_summary: str | None,
        repo_tree_text: str,
        file_context: list[dict[str, str]],
        commit_context: list[dict[str, str]],
        attachments: list[dict[str, str]],
        rag_context: str | None,
        tool_descriptions: str,
        pr_diff: str = "",
        pr_files_summary: str = "",
    ) -> str:
        """Render the system prompt with optional tool descriptions."""
        rendered = self.render_template(
            "issue_respond.j2",
            repo_full_name=event.repository.full_name,
            issue_number=issue_num,
            issue_title=event.issue.title,
            rag_context=rag_context,
            thread_comments=thread_comments,
            conversation_summary=conversation_summary,
            repo_tree=repo_tree_text,
            file_context=file_context,
            attachments=attachments,
            commit_context=commit_context,
            tool_descriptions=tool_descriptions,
            pr_diff=pr_diff,
            pr_files_summary=pr_files_summary,
        )
        # Collapse 3+ consecutive newlines into 2 (one blank line max).
        return re.sub(r"\n{3,}", "\n\n", rendered).strip()

    async def _native_tool_round(
        self,
        messages: list[dict[str, Any]],
        registry: Any,
        event: IssueCommentEvent,
        issue_num: int,
        round_num: int,
    ) -> str | None:
        """Execute one round of native tool calling.

        Returns the final reply text, or None if tool calls were made
        and the loop should continue.
        """
        response = await self.llm.chat_with_tools(
            messages, tools=registry.openai_schemas(),
        )
        choice = response.choices[0]
        message = choice.message

        if message.tool_calls:
            # Append the assistant message with tool_calls.
            messages.append(message.model_dump())

            for tc in message.tool_calls:
                args = (
                    json.loads(tc.function.arguments)
                    if isinstance(tc.function.arguments, str)
                    else tc.function.arguments
                )
                result = await registry.execute(tc.function.name, args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result.content,
                })
                logger.info(
                    "Tool round %d: %s(%s) → %s [%s#%d]",
                    round_num + 1, tc.function.name, args,
                    "OK" if result.success else "FAIL",
                    event.repository.full_name, issue_num,
                )
            return None  # Continue loop.

        # No tool calls — return the final content.
        content = message.content or ""
        clean = self._clean_reply(content)
        return clean or _FALLBACK_REPLY

    async def _prompt_tool_round(
        self,
        messages: list[dict[str, Any]],
        registry: Any,
        event: IssueCommentEvent,
        issue_num: int,
        round_num: int,
    ) -> str | None:
        """Execute one round of prompt-based tool calling.

        Returns the final reply text, or None if tool calls were parsed
        and the loop should continue.
        """
        response = await self.llm.chat_with_tools(messages)
        content = response.choices[0].message.content or ""

        tool_blocks = _TOOL_CALL_RE.findall(content)
        if tool_blocks:
            messages.append({"role": "assistant", "content": content})
            tool_results: list[str] = []
            for tc_json in tool_blocks:
                try:
                    tc_data = json.loads(tc_json)
                    result = await registry.execute(
                        tc_data["name"], tc_data.get("arguments", {}),
                    )
                    tool_results.append(
                        f"**{result.tool_name}**: {result.content}"
                    )
                    logger.info(
                        "Tool round %d: %s → %s [%s#%d]",
                        round_num + 1, tc_data["name"],
                        "OK" if result.success else "FAIL",
                        event.repository.full_name, issue_num,
                    )
                except (json.JSONDecodeError, KeyError) as exc:
                    tool_results.append(f"Error parsing tool call: {exc}")
            messages.append({
                "role": "user",
                "content": "Tool results:\n" + "\n".join(tool_results),
            })
            return None  # Continue loop.

        # No tool calls — clean and return.
        clean = self._clean_reply(content)
        return clean or _FALLBACK_REPLY

    @staticmethod
    def _clean_reply(text: str) -> str:
        """Remove stray tool/fetch markers from final reply text."""
        text = _TOOL_CALL_RE.sub("", text)
        text = _FETCH_REQUEST_RE.sub("", text)
        return text.strip()

    @staticmethod
    def _extract_last_reply(messages: list[dict[str, Any]]) -> str:
        """Extract the last assistant content from a message list."""
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if isinstance(content, str) and content.strip():
                    clean = _TOOL_CALL_RE.sub("", content)
                    clean = _FETCH_REQUEST_RE.sub("", clean).strip()
                    if clean:
                        return clean
        return _FALLBACK_REPLY

    # --- Context gathering methods ---

    async def _fetch_repo_tree(
        self, owner: str, repo: str, default_branch: str,
        limits: dict[str, int],
    ) -> tuple[str, set[str]]:
        """Fetch the repo file tree and format as a compact listing."""
        try:
            tree = await self.forge.get_repo_tree(
                owner, repo, ref=default_branch
            )
            files = [
                e["path"]
                for e in tree
                if e.get("type") == "blob"
            ][:limits["max_tree_entries"]]
            if files:
                return "\n".join(files), set(files)
        except Exception:
            logger.warning(
                "Could not fetch repo tree for %s/%s (ref=%s)",
                owner,
                repo,
                default_branch,
            )
        return "", set()

    async def _fetch_referenced_files(
        self,
        owner: str,
        repo: str,
        event: IssueCommentEvent,
        thread_comments: list[dict[str, str]],
        default_branch: str,
        limits: dict[str, int],
    ) -> list[dict[str, str]]:
        """Find file paths mentioned in the conversation and fetch content."""
        max_file_chars = limits["max_file_chars"]
        all_text = event.issue.body or ""
        for c in thread_comments:
            all_text += "\n" + c.get("body", "")
        all_text += "\n" + event.comment.body

        paths = _extract_file_paths(all_text)
        if not paths:
            return []

        fetched: list[dict[str, str]] = []
        for path in paths[:_MAX_FILE_FETCHES]:
            try:
                content = await self.forge.get_file_content(
                    owner, repo, path, ref=default_branch
                )
                if len(content) > max_file_chars:
                    content = (
                        content[:max_file_chars] + "\n... (truncated)"
                    )
                fetched.append({"path": path, "content": content})
                logger.info("Fetched referenced file %s", path)
            except Exception:
                logger.warning("Could not fetch %s (may not exist)", path)
        return fetched

    async def _fetch_grounding_files(
        self,
        owner: str,
        repo: str,
        tree_paths: set[str],
        already_fetched: set[str],
        default_branch: str,
        limits: dict[str, int],
    ) -> list[dict[str, str]]:
        """Proactively fetch key project files to ground the LLM."""
        max_chars = limits["max_grounding_file_chars"]
        targets = []
        for candidate in _GROUNDING_FILES:
            if candidate in tree_paths and candidate not in already_fetched:
                targets.append(candidate)
            if len(targets) >= _MAX_GROUNDING_FILES:
                break

        fetched: list[dict[str, str]] = []
        for path in targets:
            try:
                content = await self.forge.get_file_content(
                    owner, repo, path, ref=default_branch
                )
                if len(content) > max_chars:
                    content = (
                        content[:max_chars]
                        + "\n... (truncated)"
                    )
                fetched.append({"path": path, "content": content})
                logger.info("Fetched grounding file %s", path)
            except Exception:
                logger.warning("Could not fetch grounding file %s", path)
        return fetched

    async def _fetch_referenced_commits(
        self,
        owner: str,
        repo: str,
        event: IssueCommentEvent,
        thread_comments: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """Find commit SHAs mentioned in conversation and fetch info."""
        all_text = event.issue.body or ""
        for c in thread_comments:
            all_text += "\n" + c.get("body", "")
        all_text += "\n" + event.comment.body

        shas = _extract_commit_shas(all_text)
        if not shas:
            return []

        fetched: list[dict[str, str]] = []
        for sha in shas[:_MAX_COMMIT_FETCHES]:
            try:
                data = await self.forge.get_commit(owner, repo, sha)
                commit_info = data.get("commit", {})
                message = commit_info.get("message", "")
                author = commit_info.get("author", {}).get("name", "unknown")
                date = commit_info.get("author", {}).get("date", "")
                full_sha = data.get("sha", sha)
                summary = (
                    f"Commit {full_sha[:12]} by {author} ({date}):\n{message}"
                )
                fetched.append({"sha": full_sha, "summary": summary})
                logger.info("Fetched commit %s", sha)
            except Exception:
                logger.warning("Could not fetch commit %s", sha)
        return fetched

    async def _fetch_pr_context(
        self,
        owner: str,
        repo: str,
        pr_num: int,
        limits: dict[str, int],
    ) -> tuple[str, str]:
        """Fetch the PR diff and changed file summary.

        Returns (diff_text, files_summary). Failures return empty strings.
        """
        max_diff_chars = limits["max_pr_diff_chars"]
        diff_text = ""
        files_summary = ""

        try:
            diff_text = await self.forge.get_pull_diff(owner, repo, pr_num)
            if len(diff_text) > max_diff_chars:
                diff_text = (
                    diff_text[:max_diff_chars] + "\n\n... (diff truncated)"
                )
            logger.info(
                "Fetched PR diff for %s/%s#%d (%d chars)",
                owner, repo, pr_num, len(diff_text),
            )
        except Exception:
            logger.warning(
                "Could not fetch PR diff for %s/%s#%d",
                owner, repo, pr_num,
            )

        try:
            changed_files = await self.forge.get_pull_files(
                owner, repo, pr_num,
            )
            files_summary = "\n".join(
                f"- {f.get('filename', '?')} "
                f"(+{f.get('additions', 0)}/{-f.get('deletions', 0)})"
                for f in changed_files
            )
        except Exception:
            logger.warning(
                "Could not fetch PR files for %s/%s#%d",
                owner, repo, pr_num,
            )

        return diff_text, files_summary

    async def _collect_attachments(
        self,
        event: IssueCommentEvent,
        raw_comments: list[dict[str, Any]],
        instance_url: str,
    ) -> list[dict[str, str]]:
        """Extract attachment references and download text-based ones."""
        all_text = event.issue.body or ""
        for c in raw_comments:
            all_text += "\n" + c.get("body", "")
        all_text += "\n" + event.comment.body

        attachments = _extract_attachments(all_text, instance_url)

        downloads_remaining = _MAX_ATTACHMENT_DOWNLOADS
        for att in attachments:
            if downloads_remaining <= 0:
                break
            ext = _get_extension(att["url"])
            if ext not in _TEXT_EXTENSIONS:
                continue
            try:
                content = await self.forge.download_url(att["url"])
                if len(content) > _MAX_ATTACHMENT_CONTENT_CHARS:
                    content = (
                        content[:_MAX_ATTACHMENT_CONTENT_CHARS]
                        + "\n... (truncated)"
                    )
                att["content"] = content
                downloads_remaining -= 1
                logger.info("Downloaded text attachment: %s", att["alt"])
            except Exception:
                logger.warning(
                    "Could not download attachment %s", att["url"]
                )

        return attachments

    # --- Conversation summarization ---

    async def _summarize_old_comments(
        self,
        old_comments: list[dict[str, str]],
        repo_full_name: str,
        issue_title: str,
        bot_username: str,
        *,
        max_tokens: int = 512,
    ) -> str:
        """Summarize older conversation comments into a compact paragraph.

        Uses a dedicated template that instructs the LLM to discard
        unconfirmed bot claims, breaking the hallucination feedback loop.
        """
        summary_prompt = self.render_template(
            "conversation_summary.j2",
            repo_full_name=repo_full_name,
            issue_title=issue_title,
            bot_username=bot_username,
            comments=old_comments,
        )
        try:
            summary = await self.llm.chat(
                summary_prompt,
                "Summarize the conversation above.",
                temperature=0.1,
                max_tokens=max_tokens,
            )
            return summary.strip()
        except Exception:
            logger.warning(
                "Failed to summarize old comments for %s; using truncation",
                repo_full_name,
            )
            # Fallback: simple truncation of first 2 + last comment.
            parts: list[str] = []
            for c in old_comments[:2]:
                parts.append(f"{c['user']}: {c['body'][:200]}")
            if len(old_comments) > 2:
                parts.append(f"... ({len(old_comments) - 2} more comments) ...")
                last = old_comments[-1]
                parts.append(f"{last['user']}: {last['body'][:200]}")
            return "\n".join(parts)
