"""Tests for API definition schema models."""

import pytest
from pydantic import ValidationError

from forge_bot.api.schema import ApiDefinitionFile, EndpointDef, EndpointParam


class TestEndpointParam:
    def test_required_fields(self) -> None:
        p = EndpointParam(
            name="owner", type="string", location="path"
        )
        assert p.name == "owner"
        assert p.type == "string"
        assert p.location == "path"
        assert p.required is True
        assert p.description == ""
        assert p.default is None

    def test_optional_param(self) -> None:
        p = EndpointParam(
            name="ref",
            type="string",
            location="query",
            required=False,
            default="main",
            description="Branch name",
        )
        assert p.required is False
        assert p.default == "main"
        assert p.description == "Branch name"


class TestEndpointDef:
    def test_minimal(self) -> None:
        ep = EndpointDef(
            name="get_user",
            method="GET",
            path="/user",
            description="Get current user",
        )
        assert ep.name == "get_user"
        assert ep.method == "GET"
        assert ep.path == "/user"
        assert ep.tags == []
        assert ep.params == []
        assert ep.response_type == "json"
        assert ep.headers == {}

    def test_full(self) -> None:
        ep = EndpointDef(
            name="get_file_content",
            method="GET",
            path="/repos/{owner}/{repo}/raw/{filepath}",
            description="Get raw file content",
            tags=["file", "content"],
            params=[
                EndpointParam(
                    name="owner", type="string", location="path"
                ),
                EndpointParam(
                    name="ref",
                    type="string",
                    location="query",
                    required=False,
                ),
            ],
            response_type="text",
            headers={"Accept": "text/plain"},
        )
        assert len(ep.params) == 2
        assert ep.response_type == "text"
        assert ep.headers["Accept"] == "text/plain"
        assert ep.tags == ["file", "content"]


class TestApiDefinitionFile:
    def test_valid(self) -> None:
        api_def = ApiDefinitionFile(
            version="1",
            base_path="/api/v1",
            endpoints=[
                EndpointDef(
                    name="get_user",
                    method="GET",
                    path="/user",
                    description="Get user",
                ),
            ],
        )
        assert len(api_def.endpoints) == 1
        assert api_def.base_path == "/api/v1"

    def test_defaults(self) -> None:
        api_def = ApiDefinitionFile(
            endpoints=[
                EndpointDef(
                    name="x",
                    method="GET",
                    path="/x",
                    description="x",
                ),
            ],
        )
        assert api_def.version == "1"
        assert api_def.base_path == "/api/v1"

    def test_empty_endpoints_allowed(self) -> None:
        api_def = ApiDefinitionFile(endpoints=[])
        assert api_def.endpoints == []

    def test_model_validate_from_dict(self) -> None:
        raw = {
            "version": "1",
            "base_path": "/api/v1",
            "endpoints": [
                {
                    "name": "get_user",
                    "method": "GET",
                    "path": "/user",
                    "description": "Get user",
                    "tags": ["identity"],
                    "params": [
                        {
                            "name": "token",
                            "type": "string",
                            "location": "query",
                            "required": False,
                        }
                    ],
                    "response_type": "json",
                }
            ],
        }
        api_def = ApiDefinitionFile.model_validate(raw)
        assert api_def.endpoints[0].name == "get_user"
        assert api_def.endpoints[0].params[0].name == "token"
        assert api_def.endpoints[0].params[0].required is False

    def test_missing_endpoint_name_fails(self) -> None:
        raw = {
            "endpoints": [
                {"method": "GET", "path": "/x", "description": "x"}
            ],
        }
        with pytest.raises(ValidationError):
            ApiDefinitionFile.model_validate(raw)
