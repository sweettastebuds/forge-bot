"""Pydantic models for YAML API definition files."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class EndpointParam(BaseModel):
    """Describes a single parameter for an API endpoint."""

    name: str
    type: str  # "string", "integer", "boolean", "array"
    location: str  # "path", "query", "body"
    required: bool = True
    description: str = ""
    default: Any = None


class EndpointDef(BaseModel):
    """Describes a single API endpoint."""

    name: str
    method: str  # GET, POST, PATCH, PUT, DELETE
    path: str  # URL template with {placeholders}
    description: str
    tags: list[str] = Field(default_factory=list)
    params: list[EndpointParam] = Field(default_factory=list)
    response_type: str = "json"  # "json" or "text"
    headers: dict[str, str] = Field(default_factory=dict)
    body_template: dict[str, str] = Field(default_factory=dict)


class ApiDefinitionFile(BaseModel):
    """Top-level schema for a YAML API definition file."""

    version: str = "1"
    base_path: str = "/api/v1"
    endpoints: list[EndpointDef]
