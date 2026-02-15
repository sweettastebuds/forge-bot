"""Handler for issue_comment webhook events (@mention replies)."""

import logging
import re

from forge_bot.handlers.base import BaseHandler
from forge_bot.models import IssueCommentEvent

logger = logging.getLogger("forge_bot.handlers.issue_comment")

# Cap how many files we'll inline to avoid blowing up the context window.
_MAX_FILE_FETCHES = 5
_MAX_FILE_CHARS = 8_000
_MAX_TREE_ENTRIES = 200

# Patterns for extracting file paths and attachment URLs from text.
_FILE_PATH_RE = re.compile(
    r"(?:^|[\s`'\"])"  # boundary
    r"([\w./-]+\.(?:py|js|ts|go|rs|java|c|cpp|h|yml|yaml|toml|json|md|txt|cfg|sh))"
    r"(?:[\s`'\",;:!?)]|$)",
    re.MULTILINE,
)
_ATTACHMENT_RE = re.compile(
    r"!\[([^\]]*)\]\((/attachments/[^\)]+)\)",
)


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

        # --- Repo context: tree + referenced files ---
        repo_tree_text = await self._fetch_repo_tree(owner, repo)
        file_context = await self._fetch_referenced_files(
            owner, repo, event, thread_comments
        )

        # --- Attachment awareness ---
        instance_url = self.settings.forge_instance_url.rstrip("/")
        attachments = self._collect_attachments(event, raw_comments, instance_url)

        # Build the prompt from the Jinja2 template.
        system_prompt = self.render_template(
            "issue_respond.j2",
            repo_full_name=event.repository.full_name,
            issue_number=issue_num,
            issue_title=event.issue.title,
            rag_context=None,
            thread_comments=thread_comments,
            repo_tree=repo_tree_text,
            file_context=file_context,
            attachments=attachments,
        )

        # The user message is the triggering comment itself.
        user_message = event.comment.body

        # Call the LLM.
        try:
            reply = await self.llm.chat(system_prompt, user_message)
        except Exception:
            logger.exception(
                "LLM call failed for %s#%d",
                event.repository.full_name,
                issue_num,
            )
            reply = (
                "Sorry, I encountered an error while generating a response. "
                "Please try again later."
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

    async def _fetch_repo_tree(self, owner: str, repo: str) -> str:
        """Fetch the repo file tree and format as a compact listing."""
        try:
            tree = await self.forge.get_repo_tree(owner, repo)
            # Filter to blobs (files) only and limit size.
            files = [
                e["path"] for e in tree
                if e.get("type") == "blob"
            ][:_MAX_TREE_ENTRIES]
            if files:
                return "\n".join(files)
        except Exception:
            logger.debug("Could not fetch repo tree for %s/%s", owner, repo)
        return ""

    async def _fetch_referenced_files(
        self,
        owner: str,
        repo: str,
        event: IssueCommentEvent,
        thread_comments: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """Find file paths mentioned in the conversation and fetch their content."""
        # Gather text from issue body + all comments + triggering comment.
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
                content = await self.forge.get_file_content(owner, repo, path)
                if len(content) > _MAX_FILE_CHARS:
                    content = content[:_MAX_FILE_CHARS] + "\n... (truncated)"
                fetched.append({"path": path, "content": content})
                logger.debug("Fetched %s for context", path)
            except Exception:
                logger.debug("Could not fetch %s (may not exist)", path)
        return fetched

    @staticmethod
    def _collect_attachments(
        event: IssueCommentEvent,
        raw_comments: list[dict[str, str]],
        instance_url: str,
    ) -> list[dict[str, str]]:
        """Extract attachment references from issue body and comments."""
        all_text = event.issue.body or ""
        for c in raw_comments:
            all_text += "\n" + c.get("body", "")
        all_text += "\n" + event.comment.body
        return _extract_attachments(all_text, instance_url)
