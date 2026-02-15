"""Handler for issue_comment webhook events (@mention replies)."""

import logging
import re
from typing import Any

from forge_bot.handlers.base import BaseHandler
from forge_bot.models import IssueCommentEvent

logger = logging.getLogger("forge_bot.handlers.issue_comment")

# --- Limits ---
_MAX_FILE_FETCHES = 5
_MAX_FILE_CHARS = 8_000
_MAX_TREE_ENTRIES = 200
_MAX_GROUNDING_FILES = 3
_MAX_GROUNDING_FILE_CHARS = 4_000
_MAX_COMMIT_FETCHES = 3
_MAX_ATTACHMENT_DOWNLOADS = 3
_MAX_ATTACHMENT_CONTENT_CHARS = 6_000
_MAX_FETCH_ROUNDS = 2

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


def _parse_fetch_request(raw: str) -> tuple[str, str]:
    """Parse 'path/to/file@branch' into (path, ref).

    Returns (path, "") if no branch is specified.
    """
    raw = raw.strip()
    if "@" in raw:
        path, ref = raw.rsplit("@", 1)
        return path.strip(), ref.strip()
    return raw, ""


class IssueCommentHandler(BaseHandler):
    """Respond to @mentions in issue/PR comments with LLM-generated answers."""

    async def handle(self, event: IssueCommentEvent) -> None:
        owner, repo = event.repository.full_name.split("/", 1)
        issue_num = event.issue.number
        default_branch = event.repository.default_branch

        logger.info(
            "Handling @mention on %s#%d by %s",
            event.repository.full_name,
            issue_num,
            event.sender.login,
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

        # --- Repo context: tree + referenced files ---
        repo_tree_text, tree_paths = await self._fetch_repo_tree(
            owner, repo, default_branch
        )
        file_context = await self._fetch_referenced_files(
            owner, repo, event, thread_comments, default_branch
        )

        # Proactively fetch grounding files if few/no referenced files.
        if len(file_context) < _MAX_GROUNDING_FILES and tree_paths:
            already_fetched = {f["path"] for f in file_context}
            grounding = await self._fetch_grounding_files(
                owner, repo, tree_paths, already_fetched, default_branch
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

        # --- Build prompt and run LLM with fetch-loop ---
        reply = await self._run_with_fetch_loop(
            owner=owner,
            repo=repo,
            event=event,
            issue_num=issue_num,
            thread_comments=thread_comments,
            repo_tree_text=repo_tree_text,
            tree_paths=tree_paths,
            file_context=file_context,
            commit_context=commit_context,
            attachments=attachments,
            default_branch=default_branch,
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

    # --- LLM call with fetch loop ---

    async def _run_with_fetch_loop(
        self,
        *,
        owner: str,
        repo: str,
        event: IssueCommentEvent,
        issue_num: int,
        thread_comments: list[dict[str, str]],
        repo_tree_text: str,
        tree_paths: set[str],
        file_context: list[dict[str, str]],
        commit_context: list[dict[str, str]],
        attachments: list[dict[str, str]],
        default_branch: str,
    ) -> str:
        """Call the LLM, and if it requests files via [FETCH:], fetch and re-prompt."""
        for round_num in range(_MAX_FETCH_ROUNDS + 1):
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
                commit_context=commit_context,
            )

            user_message = event.comment.body

            try:
                reply = await self.llm.chat(system_prompt, user_message)
            except Exception:
                logger.exception(
                    "LLM call failed for %s#%d (round %d)",
                    event.repository.full_name,
                    issue_num,
                    round_num,
                )
                return (
                    "Sorry, I encountered an error while generating a "
                    "response. Please try again later."
                )

            # Check for [FETCH:] requests in the reply.
            fetch_requests = _FETCH_REQUEST_RE.findall(reply)
            if not fetch_requests or round_num >= _MAX_FETCH_ROUNDS:
                # No more fetches or max rounds reached — return the reply.
                # Strip any leftover [FETCH:] markers from the final reply.
                clean = _FETCH_REQUEST_RE.sub("", reply).strip()
                return clean if clean else (
                    "I looked into it but wasn't able to form a complete "
                    "answer. Could you provide more details?"
                )

            # Fetch the requested files and add to context.
            already = {f["path"] for f in file_context}
            new_files = 0
            for raw_request in fetch_requests:
                path, ref = _parse_fetch_request(raw_request)
                if path in already:
                    continue
                effective_ref = ref or default_branch
                try:
                    content = await self.forge.get_file_content(
                        owner, repo, path, ref=effective_ref
                    )
                    if len(content) > _MAX_FILE_CHARS:
                        content = (
                            content[:_MAX_FILE_CHARS] + "\n... (truncated)"
                        )
                    label = f"{path}@{ref}" if ref else path
                    file_context.append({"path": label, "content": content})
                    already.add(path)
                    new_files += 1
                    logger.info(
                        "Fetch-loop: fetched %s (ref=%s)", path, effective_ref
                    )
                except Exception:
                    logger.warning(
                        "Fetch-loop: could not fetch %s (ref=%s)",
                        path,
                        effective_ref,
                    )

            if new_files == 0:
                # All requests failed or were duplicates — return as-is.
                clean = _FETCH_REQUEST_RE.sub("", reply).strip()
                return clean if clean else (
                    "I looked into it but wasn't able to form a complete "
                    "answer. Could you provide more details?"
                )

            logger.info(
                "Fetch-loop round %d: fetched %d new files, re-prompting",
                round_num + 1,
                new_files,
            )

        return reply  # pragma: no cover — safety fallback

    # --- Context gathering methods ---

    async def _fetch_repo_tree(
        self, owner: str, repo: str, default_branch: str
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
            ][:_MAX_TREE_ENTRIES]
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
    ) -> list[dict[str, str]]:
        """Find file paths mentioned in the conversation and fetch content."""
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
                if len(content) > _MAX_FILE_CHARS:
                    content = (
                        content[:_MAX_FILE_CHARS] + "\n... (truncated)"
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
    ) -> list[dict[str, str]]:
        """Proactively fetch key project files to ground the LLM."""
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
                if len(content) > _MAX_GROUNDING_FILE_CHARS:
                    content = (
                        content[:_MAX_GROUNDING_FILE_CHARS]
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
