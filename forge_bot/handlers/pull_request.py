"""Handler for pull_request webhook events (code review)."""

from __future__ import annotations

import logging

from forge_bot.handlers.base import BaseHandler
from forge_bot.models import PullRequestEvent
from forge_bot.utils.token_budget import TokenBudget

logger = logging.getLogger("forge_bot.handlers.pull_request")

# Diffs larger than this are truncated to stay within LLM context limits.
_MAX_DIFF_CHARS = 30_000
# Max chars per chunk (leave room for system prompt + PR metadata).
_CHUNK_CHARS = 25_000
# Maximum number of chunks to review (to bound cost/time).
_MAX_CHUNKS = 8


def _split_diff_by_file(diff_text: str) -> list[str]:
    """Split a unified diff into per-file segments."""
    segments: list[str] = []
    current: list[str] = []
    for line in diff_text.splitlines(keepends=True):
        if line.startswith("diff --git ") and current:
            segments.append("".join(current))
            current = []
        current.append(line)
    if current:
        segments.append("".join(current))
    return segments


def _pack_chunks(file_diffs: list[str], max_chars: int) -> list[str]:
    """Pack per-file diffs into chunks that fit within *max_chars*.

    Each file stays whole unless it alone exceeds the limit (then it's
    hard-truncated).  Files are grouped greedily.
    """
    chunks: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    for fd in file_diffs:
        fd_len = len(fd)
        # Single file exceeds limit — truncate it into its own chunk.
        if fd_len > max_chars:
            if current_parts:
                chunks.append("".join(current_parts))
                current_parts, current_len = [], 0
            chunks.append(fd[:max_chars] + "\n\n... (file diff truncated)")
            continue
        # Would overflow current chunk — flush.
        if current_len + fd_len > max_chars and current_parts:
            chunks.append("".join(current_parts))
            current_parts, current_len = [], 0
        current_parts.append(fd)
        current_len += fd_len

    if current_parts:
        chunks.append("".join(current_parts))
    return chunks


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

        # # Truncate very large diffs to avoid exceeding LLM context.
        # if len(diff_text) > _MAX_DIFF_CHARS:
        #     diff_text = diff_text[:_MAX_DIFF_CHARS] + "\n\n... (diff truncated)"

        # Build a concise file summary for the prompt.
        if isinstance(changed_files, list):
            file_summary = "\n".join(
                f"- {f.get('filename', '?')} (+{f.get('additions', 0)}/{-f.get('deletions', 0)})"
                for f in changed_files
            )
        else:
            file_summary = ""

        # Decide: single-pass or chunked review.
        if len(diff_text) <= _MAX_DIFF_CHARS:
            review = await self._review_single(event, file_summary, diff_text)
        else:
            review = await self._review_chunked(event, file_summary, diff_text)

        # # Render the system prompt from the Jinja2 template.
        # system_prompt = self.render_template(
        #     "pr_review.j2",
        #     repo_full_name=event.repository.full_name,
        #     pr_title=event.pull_request.title,
        # )

        # User message = PR description + file list + diff.
        # user_message = self._build_user_message(event, file_summary, diff_text)

        # # Call the LLM.
        # try:
        #     review = await self.llm.chat(system_prompt, user_message)
        # except Exception:
        #     logger.exception(
        #         "LLM call failed for PR %s#%d",
        #         event.repository.full_name,
        #         pr_num,
        #     )
        #     review = (
        #         "Sorry, I encountered an error while reviewing this PR. Please try again later."
        #     )

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

    async def _review_single(
        self,
        event: PullRequestEvent,
        file_summary: str,
        diff_text: str,
    ) -> str:
        """Review the PR in a single pass."""
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

    async def _review_chunked(
        self,
        event: PullRequestEvent,
        file_summary: str,
        diff_text: str,
    ) -> str:
        """Review the PR in multiple chunks and aggregate the feedback."""
        file_diffs = _split_diff_by_file(diff_text)
        chunks = _pack_chunks(file_diffs, _CHUNK_CHARS)
        total = len(chunks)

        if len(chunks) > _MAX_CHUNKS:
            logger.warning(
                "PR %s#%d has %d chunks, exceeding the max of %d. Truncating.",
                event.repository.full_name,
                event.pull_request.number,
                total,
                _MAX_CHUNKS,
            )
            chunks = chunks[:_MAX_CHUNKS]

        logger.info(
            "Chunked review for %s#%d: %d chunks",
            event.repository.full_name,
            event.pull_request.number,
            total,
        )

        # Phase 1: review each chunk independently.
        chunk_reviews = []
        for i, chunk in enumerate(chunks):
            system_prompt = self.render_template(
                "pr_review_chunk.j2",
                repo_full_name=event.repository.full_name,
                pr_title=event.pull_request.title,
                chunk_index=i + 1,
                total_chunks=len(chunks),
            )
            # system_prompt = self.render_template(
            #     "pr_review.j2",
            #     repo_full_name=event.repository.full_name,
            #     pr_title=event.pull_request.title,
            # )
            user_message = self._build_chunk_message(
                event, file_summary, chunk, part=i, total=total
            )
            try:
                review = await self.llm.chat(system_prompt, user_message)
                chunk_reviews.append(review)
            except Exception:
                logger.exception(
                    "LLM chunk %d/%d failed for PR %s#%d",
                    i,
                    total,
                    event.repository.full_name,
                    event.pull_request.number,
                )
                chunk_reviews.append(
                    f"### Part {i}/{total}\n\n_Failed to review this portion of the diff._"
                )

        # Phase 2: synthesise a unified summary from chunk reviews.
        combined = "\n\n---\n\n".join(chunk_reviews)
        synthesis = await self._synthesise(event, combined)

        return synthesis

    async def _synthesise(
        self,
        event: PullRequestEvent,
        combined_reviews: str,
    ) -> str:
        system_prompt = self.render_template(
            "pr_review_synthesis.j2",
            repo_full_name=event.repository.full_name,
            pr_title=event.pull_request.title,
            combined_reviews=combined_reviews,
        )
        user_message = (
            f"## PR #{event.pull_request.number}: {event.pull_request.title}\n\n{combined_reviews}"
        )
        try:
            return await self.llm.chat(system_prompt, user_message)
        except Exception:
            logger.exception(
                "Synthesis LLM call failed for PR %s#%d",
                event.repository.full_name,
                event.pull_request.number,
            )
            # Fallback: return the raw chunk reviews without synthesis.
            return (
                "_⚠️ Failed to synthesise a unified review. "
                "Showing per-part reviews:_\n\n" + combined_reviews
            )

    # ------------------------------------------------------------------
    # Message builders
    # ------------------------------------------------------------------

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

    @staticmethod
    def _build_chunk_message(
        event: PullRequestEvent,
        file_summary: str,
        diff_chunk: str,
        *,
        part: int,
        total: int,
    ) -> str:
        pr = event.pull_request
        parts = [
            f"## PR #{pr.number}: {pr.title} - Part {part}/{total}",
            f"**Branch:** {pr.head.ref} → {pr.base.ref}",
        ]

        if pr.body and part == 1:
            parts.append(f"\n**Description:**\n{pr.body}")
        if file_summary and part == 1:
            parts.append(f"\n**All changed files:**\n{file_summary}")
        parts.append(f"\n**Diff (part {part}/{total}):**\n```diff\n{diff_chunk}\n```")
        return "\n".join(parts)
