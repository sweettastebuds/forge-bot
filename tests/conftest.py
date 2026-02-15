import pytest


@pytest.fixture
def webhook_secret() -> str:
    return "test-secret-key"


@pytest.fixture
def bot_username() -> str:
    return "forge-bot"


@pytest.fixture
def sample_pr_payload() -> dict:
    return {
        "action": "opened",
        "number": 1,
        "pull_request": {
            "id": 1,
            "number": 1,
            "title": "Test PR",
            "body": "Test description",
            "state": "open",
            "user": {"id": 1, "login": "developer"},
            "head": {"ref": "feature", "sha": "abc123"},
            "base": {"ref": "main", "sha": "def456"},
            "assignees": [{"id": 2, "login": "forge-bot"}],
            "requested_reviewers": [],
        },
        "repository": {
            "full_name": "owner/repo",
            "clone_url": "https://gitea.example.com/owner/repo.git",
            "default_branch": "main",
        },
        "sender": {"id": 1, "login": "developer"},
    }


@pytest.fixture
def sample_comment_payload() -> dict:
    return {
        "action": "created",
        "comment": {
            "id": 1,
            "body": "@forge-bot what does this function do?",
            "user": {"id": 1, "login": "developer"},
        },
        "issue": {
            "number": 5,
            "title": "Question about auth module",
            "body": "I need help understanding the auth flow.",
            "pull_request": None,
            "assignees": [],
        },
        "is_pull": False,
        "repository": {"full_name": "owner/repo", "default_branch": "master"},
        "sender": {"id": 1, "login": "developer"},
    }
