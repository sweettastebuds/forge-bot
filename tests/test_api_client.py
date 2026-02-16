"""Tests for the GenericForgeClient."""

from unittest.mock import patch

import httpx
import pytest
import pytest_httpx

from forge_bot.api.client import GenericForgeClient
from forge_bot.api.loader import clear_cache
from forge_bot.config import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings(
        forge_instance_url="https://gitea.example.com",
        forge_api_token="test-token",
        forge_webhook_secret="test-secret",
        forge_provider="gitea",
        llm_api_key="test-llm-key",
    )


@pytest.fixture(autouse=True)
def _clear_loader_cache() -> None:
    clear_cache()


@pytest.fixture
def client(settings: Settings) -> GenericForgeClient:
    return GenericForgeClient(settings)


class TestClientInit:
    def test_loads_endpoints(self, client: GenericForgeClient) -> None:
        names = client.list_endpoints()
        assert len(names) > 0
        assert "get_authenticated_user" in names

    def test_forgejo_provider(self) -> None:
        settings = Settings(
            forge_instance_url="https://forgejo.example.com",
            forge_api_token="tok",
            forge_webhook_secret="sec",
            forge_provider="forgejo",
            llm_api_key="key",
        )
        c = GenericForgeClient(settings)
        assert len(c.list_endpoints()) > 0


class TestSearch:
    def test_search_by_name(self, client: GenericForgeClient) -> None:
        results = client.search("pull")
        assert len(results) > 0
        names = [ep.name for ep in results]
        assert any("pull" in name for name in names)

    def test_search_by_tag(self, client: GenericForgeClient) -> None:
        results = client.search("diff")
        assert len(results) > 0

    def test_search_by_description(self, client: GenericForgeClient) -> None:
        results = client.search("comment")
        assert len(results) > 0

    def test_search_no_results(self, client: GenericForgeClient) -> None:
        results = client.search("xyznonexistent999")
        assert results == []

    def test_search_case_insensitive(self, client: GenericForgeClient) -> None:
        lower = client.search("pull")
        upper = client.search("PULL")
        assert len(lower) == len(upper)

    def test_search_ranking(self, client: GenericForgeClient) -> None:
        results = client.search("branch")
        # Endpoints with "branch" in name should rank higher
        if len(results) >= 2:
            top = results[0]
            assert "branch" in top.name.lower() or any(
                "branch" in t for t in top.tags
            )


class TestGetEndpoint:
    def test_existing(self, client: GenericForgeClient) -> None:
        ep = client.get_endpoint("get_authenticated_user")
        assert ep is not None
        assert ep.method == "GET"
        assert ep.path == "/user"

    def test_nonexistent(self, client: GenericForgeClient) -> None:
        assert client.get_endpoint("nonexistent") is None


class TestCall:
    @pytest.mark.asyncio
    async def test_get_json(
        self, client: GenericForgeClient, httpx_mock: pytest_httpx.HTTPXMock
    ) -> None:
        httpx_mock.add_response(
            url="https://gitea.example.com/api/v1/user",
            json={"login": "bot", "id": 42},
        )
        result = await client.call("get_authenticated_user")
        assert result == {"login": "bot", "id": 42}

    @pytest.mark.asyncio
    async def test_get_text(
        self, client: GenericForgeClient, httpx_mock: pytest_httpx.HTTPXMock
    ) -> None:
        httpx_mock.add_response(
            url="https://gitea.example.com/api/v1/repos/owner/repo/pulls/1.diff",
            text="diff --git a/file.py b/file.py\n",
        )
        result = await client.call(
            "get_pull_diff", owner="owner", repo="repo", index=1
        )
        assert "diff --git" in result

    @pytest.mark.asyncio
    async def test_path_params(
        self, client: GenericForgeClient, httpx_mock: pytest_httpx.HTTPXMock
    ) -> None:
        httpx_mock.add_response(
            url="https://gitea.example.com/api/v1/repos/myowner/myrepo/git/commits/abc123",
            json={"sha": "abc123", "message": "test"},
        )
        result = await client.call(
            "get_commit", owner="myowner", repo="myrepo", sha="abc123"
        )
        assert result["sha"] == "abc123"

    @pytest.mark.asyncio
    async def test_query_params(
        self, client: GenericForgeClient, httpx_mock: pytest_httpx.HTTPXMock
    ) -> None:
        httpx_mock.add_response(
            json={"tree": []},
        )
        await client.call(
            "get_repo_tree",
            owner="owner",
            repo="repo",
            ref="main",
            recursive=True,
        )
        request = httpx_mock.get_request()
        assert request is not None
        assert "recursive=True" in str(request.url)

    @pytest.mark.asyncio
    async def test_post_body(
        self, client: GenericForgeClient, httpx_mock: pytest_httpx.HTTPXMock
    ) -> None:
        httpx_mock.add_response(
            json={"id": 99, "body": "hello"},
        )
        result = await client.call(
            "post_issue_comment",
            owner="owner",
            repo="repo",
            index=5,
            body="hello",
        )
        assert result["id"] == 99
        request = httpx_mock.get_request()
        assert request is not None
        assert request.method == "POST"

    @pytest.mark.asyncio
    async def test_patch_body(
        self, client: GenericForgeClient, httpx_mock: pytest_httpx.HTTPXMock
    ) -> None:
        httpx_mock.add_response(json={"id": 99, "body": "updated"})
        await client.call(
            "edit_issue_comment",
            owner="owner",
            repo="repo",
            id=99,
            body="updated",
        )
        request = httpx_mock.get_request()
        assert request is not None
        assert request.method == "PATCH"

    @pytest.mark.asyncio
    async def test_unknown_endpoint(
        self, client: GenericForgeClient
    ) -> None:
        with pytest.raises(ValueError, match="Unknown API endpoint"):
            await client.call("nonexistent_endpoint")

    @pytest.mark.asyncio
    async def test_missing_required_param(
        self, client: GenericForgeClient
    ) -> None:
        with pytest.raises(ValueError, match="Missing required parameter"):
            await client.call("get_commit", owner="owner")
            # Missing repo and sha

    @pytest.mark.asyncio
    async def test_http_error_propagates(
        self, client: GenericForgeClient, httpx_mock: pytest_httpx.HTTPXMock
    ) -> None:
        httpx_mock.add_response(
            url="https://gitea.example.com/api/v1/user",
            status_code=401,
        )
        with pytest.raises(httpx.HTTPStatusError):
            await client.call("get_authenticated_user")

    @pytest.mark.asyncio
    async def test_auth_header_sent(
        self, client: GenericForgeClient, httpx_mock: pytest_httpx.HTTPXMock
    ) -> None:
        httpx_mock.add_response(json={"login": "bot"})
        await client.call("get_authenticated_user")
        request = httpx_mock.get_request()
        assert request is not None
        assert request.headers["authorization"] == "token test-token"

    @pytest.mark.asyncio
    async def test_optional_param_omitted(
        self, client: GenericForgeClient, httpx_mock: pytest_httpx.HTTPXMock
    ) -> None:
        httpx_mock.add_response(text="file content here")
        await client.call(
            "get_file_content",
            owner="owner",
            repo="repo",
            filepath="README.md",
            # ref is optional, omitted
        )
        request = httpx_mock.get_request()
        assert request is not None
        assert "ref" not in str(request.url)
