"""Pydantic models for Gitea/Forgejo webhook payloads."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class WebhookUser(BaseModel):
    """User object as embedded in webhook payloads."""

    id: int
    login: str


class Repository(BaseModel):
    """Repository object from webhook payloads."""

    full_name: str
    clone_url: str = ""
    default_branch: str = "main"


class BranchRef(BaseModel):
    """Branch reference (head/base) in a pull request."""

    ref: str
    sha: str


class PullRequest(BaseModel):
    """Pull request object embedded in PullRequestEvent."""

    id: int
    number: int
    title: str
    body: str = ""
    state: str
    user: WebhookUser
    head: BranchRef
    base: BranchRef
    assignees: list[WebhookUser] = Field(default_factory=list)
    requested_reviewers: list[WebhookUser] = Field(default_factory=list)

    @field_validator("assignees", "requested_reviewers", mode="before")
    @classmethod
    def _null_to_list(cls, v: object) -> object:
        return v if v is not None else []


class Comment(BaseModel):
    """Comment object embedded in IssueCommentEvent."""

    id: int
    body: str
    user: WebhookUser


class Issue(BaseModel):
    """Issue object embedded in IssueCommentEvent and IssuesEvent."""

    number: int
    title: str
    body: str = ""
    pull_request: object | None = None
    assignees: list[WebhookUser] = Field(default_factory=list)

    @field_validator("assignees", mode="before")
    @classmethod
    def _null_to_list(cls, v: object) -> object:
        return v if v is not None else []

    @property
    def is_pull(self) -> bool:
        """True if this issue is actually a pull request."""
        return self.pull_request is not None


class PullRequestEvent(BaseModel):
    """Webhook payload for pull_request events."""

    action: str
    number: int
    pull_request: PullRequest
    repository: Repository
    sender: WebhookUser


class IssueCommentEvent(BaseModel):
    """Webhook payload for issue_comment events."""

    action: str
    comment: Comment
    issue: Issue
    repository: Repository
    sender: WebhookUser
    is_pull: bool = False


class IssuesEvent(BaseModel):
    """Webhook payload for issues events (opened, assigned, etc.)."""

    action: str
    issue: Issue
    repository: Repository
    sender: WebhookUser
