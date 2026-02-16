"""Tests for forge_bot.sandbox.parser."""

from forge_bot.sandbox.parser import parse_run_command


def test_parse_run_python():
    body = (
        "@bot /run python\n"
        "```python\n"
        "print('hello')\n"
        "```"
    )
    cmd = parse_run_command(body)
    assert cmd is not None
    assert cmd.language == "python"
    assert cmd.code == "print('hello')"
    assert cmd.network_enabled is False


def test_parse_run_with_net():
    body = (
        "@bot /run --net node\n"
        "```javascript\n"
        "console.log('hi')\n"
        "```"
    )
    cmd = parse_run_command(body)
    assert cmd is not None
    assert cmd.language == "node"
    assert cmd.code == "console.log('hi')"
    assert cmd.network_enabled is True


def test_parse_run_no_code_block():
    body = "@bot /run python\nno code block here"
    cmd = parse_run_command(body)
    assert cmd is None


def test_parse_run_no_command():
    body = "@bot please help me with this issue"
    cmd = parse_run_command(body)
    assert cmd is None


def test_parse_run_unknown_language_still_parses():
    """Parser doesn't validate language — orchestrator handles rejection."""
    body = (
        "/run haskell\n"
        "```haskell\n"
        "main = putStrLn \"hello\"\n"
        "```"
    )
    cmd = parse_run_command(body)
    assert cmd is not None
    assert cmd.language == "haskell"


def test_parse_run_multiline_code():
    body = (
        "/run python\n"
        "```python\n"
        "x = 1\n"
        "y = 2\n"
        "print(x + y)\n"
        "```"
    )
    cmd = parse_run_command(body)
    assert cmd is not None
    assert "x = 1" in cmd.code
    assert "print(x + y)" in cmd.code


def test_parse_run_code_fence_no_language_tag():
    """Code fence without a language tag should still match."""
    body = (
        "/run python\n"
        "```\n"
        "print('bare fence')\n"
        "```"
    )
    cmd = parse_run_command(body)
    assert cmd is not None
    assert cmd.code == "print('bare fence')"


def test_parse_run_extra_text_before():
    body = (
        "Hey @bot can you /run python\n"
        "```python\n"
        "print('test')\n"
        "```"
    )
    cmd = parse_run_command(body)
    assert cmd is not None
    assert cmd.language == "python"
