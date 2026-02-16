"""Tests for API definition YAML loader."""

from pathlib import Path

import pytest

from forge_bot.api.loader import (
    clear_cache,
    load_api_definition,
    load_provider_definition,
)


@pytest.fixture(autouse=True)
def _clear_loader_cache() -> None:
    """Ensure each test starts with a clean cache."""
    clear_cache()


class TestLoadApiDefinition:
    def test_loads_gitea_yaml(self) -> None:
        defs_dir = Path(__file__).parent.parent / "forge_bot" / "api" / "definitions"
        api_def = load_api_definition(defs_dir / "gitea.yaml")
        assert api_def.version == "1"
        assert api_def.base_path == "/api/v1"
        assert len(api_def.endpoints) > 0

    def test_loads_forgejo_yaml(self) -> None:
        defs_dir = Path(__file__).parent.parent / "forge_bot" / "api" / "definitions"
        api_def = load_api_definition(defs_dir / "forgejo.yaml")
        assert len(api_def.endpoints) > 0

    def test_caching(self) -> None:
        defs_dir = Path(__file__).parent.parent / "forge_bot" / "api" / "definitions"
        path = defs_dir / "gitea.yaml"
        first = load_api_definition(path)
        second = load_api_definition(path)
        assert first is second  # Same object = cache hit

    def test_nonexistent_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_api_definition(Path("/nonexistent/file.yaml"))

    def test_gitea_has_expected_endpoints(self) -> None:
        defs_dir = Path(__file__).parent.parent / "forge_bot" / "api" / "definitions"
        api_def = load_api_definition(defs_dir / "gitea.yaml")
        names = {ep.name for ep in api_def.endpoints}
        expected = {
            "get_authenticated_user",
            "get_repo",
            "get_file_content",
            "get_repo_tree",
            "get_issue_comments",
            "post_issue_comment",
            "edit_issue_comment",
            "get_pull_diff",
            "get_pull_files",
            "get_commit",
            "list_branches",
            "create_pull_request",
        }
        assert expected.issubset(names), f"Missing: {expected - names}"


class TestLoadProviderDefinition:
    def test_gitea_provider(self) -> None:
        api_def = load_provider_definition("gitea")
        assert len(api_def.endpoints) > 0

    def test_forgejo_provider(self) -> None:
        api_def = load_provider_definition("forgejo")
        assert len(api_def.endpoints) > 0

    def test_unknown_provider(self) -> None:
        with pytest.raises(FileNotFoundError, match="github"):
            load_provider_definition("github")


class TestClearCache:
    def test_clear_cache_forces_reload(self) -> None:
        defs_dir = Path(__file__).parent.parent / "forge_bot" / "api" / "definitions"
        path = defs_dir / "gitea.yaml"
        first = load_api_definition(path)
        clear_cache()
        second = load_api_definition(path)
        assert first is not second  # Different objects after cache clear
