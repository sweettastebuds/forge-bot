"""Tests for forge_bot.router."""


from forge_bot.router import _mentions_user, dispatch

BOT = "forge-bot"


# --- @mention detection ---


class TestMentionsUser:
    def test_mention_at_start(self):
        assert _mentions_user("@forge-bot please review", BOT) is True

    def test_mention_mid_sentence(self):
        assert _mentions_user("hey @forge-bot can you help?", BOT) is True

    def test_mention_at_end(self):
        assert _mentions_user("take a look @forge-bot", BOT) is True

    def test_mention_with_punctuation(self):
        assert _mentions_user("thanks @forge-bot!", BOT) is True
        assert _mentions_user("cc @forge-bot.", BOT) is True

    def test_mention_after_open_paren(self):
        """Opening paren is not whitespace, so direct (@ doesn't match."""
        assert _mentions_user("(@forge-bot)", BOT) is False
        # But with a space after the paren it works.
        assert _mentions_user("( @forge-bot )", BOT) is True

    def test_no_mention(self):
        assert _mentions_user("no mention here", BOT) is False

    def test_partial_mention_no_match(self):
        assert _mentions_user("email@forge-bot-extra", BOT) is False

    def test_mention_on_second_line(self):
        text = "first line\n@forge-bot second line"
        assert _mentions_user(text, BOT) is True

    def test_special_chars_in_username_escaped(self):
        """Regex metacharacters in username should be escaped."""
        assert _mentions_user("@user.name hi", "user.name") is True
        assert _mentions_user("@username hi", "user.name") is False


# --- Self-loop guard ---


async def test_self_loop_drops_event(sample_pr_payload: dict):
    """Events sent by the bot itself should be silently ignored."""
    sample_pr_payload["sender"]["login"] = BOT
    # Should return without error — no handler called.
    await dispatch("pull_request", sample_pr_payload, BOT)


# --- Pull request dispatch ---


async def test_dispatch_pr_opened(sample_pr_payload: dict):
    """PR opened event should be parsed and routed (handler is a no-op stub)."""
    # Should not raise.
    await dispatch("pull_request", sample_pr_payload, BOT)


async def test_dispatch_pr_ignored_action(sample_pr_payload: dict):
    """PR actions other than opened/synchronized should be ignored."""
    sample_pr_payload["action"] = "closed"
    await dispatch("pull_request", sample_pr_payload, BOT)


# --- Issue comment dispatch ---


async def test_dispatch_comment_with_mention(sample_comment_payload: dict):
    """Comment mentioning the bot should be routed."""
    await dispatch("issue_comment", sample_comment_payload, BOT)


async def test_dispatch_comment_without_mention(sample_comment_payload: dict):
    """Comment not mentioning the bot should be skipped."""
    sample_comment_payload["comment"]["body"] = "some random comment"
    await dispatch("issue_comment", sample_comment_payload, BOT)


async def test_dispatch_comment_ignored_action(sample_comment_payload: dict):
    """Non-created comment actions should be ignored."""
    sample_comment_payload["action"] = "edited"
    await dispatch("issue_comment", sample_comment_payload, BOT)


# --- Issues dispatch ---


async def test_dispatch_issue_assigned_to_bot():
    payload = {
        "action": "assigned",
        "issue": {
            "number": 10,
            "title": "Bug",
            "body": "",
            "pull_request": None,
            "assignees": [{"id": 2, "login": BOT}],
        },
        "repository": {"full_name": "owner/repo"},
        "sender": {"id": 3, "login": "admin"},
    }
    await dispatch("issues", payload, BOT)


async def test_dispatch_issue_assigned_to_other():
    """Issues assigned to someone else should be skipped."""
    payload = {
        "action": "assigned",
        "issue": {
            "number": 10,
            "title": "Bug",
            "body": "",
            "pull_request": None,
            "assignees": [{"id": 3, "login": "other-user"}],
        },
        "repository": {"full_name": "owner/repo"},
        "sender": {"id": 3, "login": "admin"},
    }
    await dispatch("issues", payload, BOT)


async def test_dispatch_unknown_event():
    """Unknown event types should be silently ignored."""
    await dispatch("push", {"sender": {"login": "dev"}}, BOT)
