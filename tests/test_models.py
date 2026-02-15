"""Tests for forge_bot.models."""

from forge_bot.models import (
    IssueCommentEvent,
    IssuesEvent,
    PullRequestEvent,
    WebhookUser,
)


def test_pull_request_event_from_payload(sample_pr_payload: dict):
    event = PullRequestEvent.model_validate(sample_pr_payload)
    assert event.action == "opened"
    assert event.number == 1
    assert event.pull_request.title == "Test PR"
    assert event.sender.login == "developer"
    assert event.repository.full_name == "owner/repo"
    assert event.pull_request.head.ref == "feature"
    assert event.pull_request.base.ref == "main"


def test_issue_comment_event_from_payload(sample_comment_payload: dict):
    event = IssueCommentEvent.model_validate(sample_comment_payload)
    assert event.action == "created"
    assert event.comment.body == "@forge-bot what does this function do?"
    assert event.comment.user.login == "developer"
    assert event.issue.number == 5
    assert event.sender.login == "developer"


def test_issues_event_assigned():
    payload = {
        "action": "assigned",
        "issue": {
            "number": 10,
            "title": "Bug report",
            "body": "Something is broken",
            "pull_request": None,
            "assignees": [{"id": 2, "login": "forge-bot"}],
        },
        "repository": {"full_name": "owner/repo"},
        "sender": {"id": 3, "login": "admin"},
    }
    event = IssuesEvent.model_validate(payload)
    assert event.action == "assigned"
    assert event.issue.number == 10
    assert len(event.issue.assignees) == 1
    assert event.issue.assignees[0].login == "forge-bot"


def test_issue_is_pull_property():
    """Issue.is_pull returns True when pull_request field is not None."""
    event = IssuesEvent.model_validate(
        {
            "action": "opened",
            "issue": {
                "number": 1,
                "title": "PR as issue",
                "pull_request": {"url": "http://example.com"},
                "assignees": [],
            },
            "repository": {"full_name": "owner/repo"},
            "sender": {"id": 1, "login": "dev"},
        }
    )
    assert event.issue.is_pull is True

    event2 = IssuesEvent.model_validate(
        {
            "action": "opened",
            "issue": {
                "number": 2,
                "title": "Real issue",
                "pull_request": None,
                "assignees": [],
            },
            "repository": {"full_name": "owner/repo"},
            "sender": {"id": 1, "login": "dev"},
        }
    )
    assert event2.issue.is_pull is False


def test_webhook_user_minimal():
    user = WebhookUser(id=42, login="test-user")
    assert user.id == 42
    assert user.login == "test-user"


def test_pr_event_optional_fields():
    """PR event works with minimal optional fields."""
    payload = {
        "action": "opened",
        "number": 1,
        "pull_request": {
            "id": 1,
            "number": 1,
            "title": "Test",
            "state": "open",
            "user": {"id": 1, "login": "dev"},
            "head": {"ref": "feat", "sha": "aaa"},
            "base": {"ref": "main", "sha": "bbb"},
        },
        "repository": {"full_name": "owner/repo"},
        "sender": {"id": 1, "login": "dev"},
    }
    event = PullRequestEvent.model_validate(payload)
    assert event.pull_request.body == ""
    assert event.pull_request.assignees == []
    assert event.pull_request.requested_reviewers == []
