"""Tests for forge_bot.sandbox.images."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from forge_bot.sandbox.images import DEFAULT_IMAGES, ImageRegistry


def test_default_images_resolve():
    reg = ImageRegistry()
    assert reg.resolve("python") == "python:3.12-slim"
    assert reg.resolve("node") == "node:22-slim"
    assert reg.resolve("go") == "golang:1.23-alpine"
    assert reg.resolve("rust") == "rust:1.77-slim"
    assert reg.resolve("c") == "gcc:14"


def test_resolve_case_insensitive():
    reg = ImageRegistry()
    assert reg.resolve("Python") == "python:3.12-slim"
    assert reg.resolve("NODE") == "node:22-slim"


def test_resolve_unknown_returns_none():
    reg = ImageRegistry()
    assert reg.resolve("fortran") is None


def test_resolve_default_key():
    reg = ImageRegistry()
    assert reg.resolve("default") == "python:3.12-slim"


def test_available_languages_excludes_default():
    reg = ImageRegistry()
    langs = reg.available_languages()
    assert "default" not in langs
    assert "python" in langs
    assert "node" in langs
    assert langs == sorted(langs)


def test_load_override_file():
    override = {"ruby": "ruby:3.3-slim", "python": "python:3.13-slim"}
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False,
    ) as f:
        json.dump(override, f)
        f.flush()
        path = f.name

    reg = ImageRegistry()
    reg.load_override_file(path)
    assert reg.resolve("ruby") == "ruby:3.3-slim"
    assert reg.resolve("python") == "python:3.13-slim"
    # Original non-overridden keys still work
    assert reg.resolve("node") == "node:22-slim"
    Path(path).unlink()


def test_load_override_file_missing(caplog):
    reg = ImageRegistry()
    reg.load_override_file("/nonexistent/path.json")
    assert "not found" in caplog.text


def test_load_override_file_invalid_json(tmp_path):
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("not valid json{{{")
    reg = ImageRegistry()
    reg.load_override_file(str(bad_file))
    # Should not crash, should still have defaults
    assert reg.resolve("python") == "python:3.12-slim"


def test_custom_defaults():
    reg = ImageRegistry(defaults={"custom": "my-image:latest"})
    assert reg.resolve("custom") == "my-image:latest"
    assert reg.resolve("python") is None  # No defaults loaded


@pytest.mark.asyncio
async def test_prepull_none():
    """Pre-pull 'none' should skip all pulls."""
    mock_client = MagicMock()
    reg = ImageRegistry()
    await reg.prepull(mock_client, "none")
    mock_client.images.pull.assert_not_called()


@pytest.mark.asyncio
async def test_prepull_specific_keys():
    """Pre-pull specific keys should pull only those images."""
    mock_client = MagicMock()
    reg = ImageRegistry()
    await reg.prepull(mock_client, "python,node")
    assert mock_client.images.pull.call_count == 2
    pulled = {call.args[0] for call in mock_client.images.pull.call_args_list}
    assert pulled == {"python:3.12-slim", "node:22-slim"}


@pytest.mark.asyncio
async def test_prepull_all():
    """Pre-pull 'all' should pull all registered images (deduplicated)."""
    mock_client = MagicMock()
    reg = ImageRegistry()
    await reg.prepull(mock_client, "all")
    # python and default share the same image, so one less than total keys
    unique_images = set(DEFAULT_IMAGES.values())
    assert mock_client.images.pull.call_count == len(unique_images)


@pytest.mark.asyncio
async def test_prepull_unknown_key_skipped(caplog):
    """Unknown keys in pre-pull string should be logged and skipped."""
    mock_client = MagicMock()
    reg = ImageRegistry()
    await reg.prepull(mock_client, "python,unknown_lang")
    assert mock_client.images.pull.call_count == 1
    assert "Unknown image key" in caplog.text
