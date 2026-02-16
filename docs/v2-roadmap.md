# V2 Roadmap: Tool/Function-Calling System

## Context

V1 of forge-bot uses a fragile `[FETCH: path]` marker system where the LLM outputs text markers and the handler regex-parses them to fetch files. This is unreliable — smaller models (gemma3:12b) frequently forget they can request files, leading to inconsistent responses.

Additionally, file fetching is the only dynamic capability. There's no structured way to add more tools (code search, commit lookup, etc.) or tell the model what it can do.

## Goal

Replace the `[FETCH:]` regex system with a proper tool-calling architecture that:

1. Uses native OpenAI tool calling when the model supports it (gpt-4o, llama3.1, mistral)
2. Falls back to prompt-based tool descriptions for models that don't (gemma3)
3. Makes it easy to add new tools
4. Ensures the model always knows what capabilities it has

## Architecture

### Tool Base Classes — `forge_bot/tools/base.py`

```python
@dataclass
class ToolResult:
    tool_name: str
    success: bool
    content: str

@dataclass
class ToolParameter:
    name: str
    type: str          # "string", "integer", "boolean"
    description: str
    required: bool = True

class BaseTool(ABC):
    name: str           # e.g. "fetch_file"
    description: str    # human-readable
    parameters: list[ToolParameter]

    async def execute(self, **kwargs) -> ToolResult
    def to_openai_schema(self) -> dict      # OpenAI tools format
    def to_prompt_text(self) -> str          # plain-text for prompt fallback
```

### Tool Registry — `forge_bot/tools/registry.py`

```python
class ToolRegistry:
    def register(self, tool: BaseTool) -> None
    def get(self, name: str) -> BaseTool | None
    def openai_schemas(self) -> list[dict]     # for native mode
    def prompt_text(self) -> str               # for prompt fallback mode
    async def execute(self, name: str, arguments: dict) -> ToolResult
```

Created per-request (not singleton) because each tool needs the repo context (owner, repo, branch) from that specific webhook event.

### Tool Implementations

**`forge_bot/tools/fetch_file.py`** — `FetchFileTool`
- Replaces `[FETCH:]` marker system
- Parameters: `path` (required), `ref` (optional)
- Calls `forge.get_file_content()`, truncates to `max_file_chars`

**`forge_bot/tools/get_commit.py`** — `GetCommitTool`
- Parameters: `sha` (required)
- Calls `forge.get_commit()`, formats as summary

**`forge_bot/tools/search_code.py`** — `SearchCodeTool`
- Parameters: `query` (required), `file_pattern` (optional)
- Iterates repo tree paths, fetches files, case-insensitive text search
- Returns matching file:line snippets (max 10 results, max 50 files searched)

### LLM Client Changes — `forge_bot/clients/llm.py`

Add a new `chat_with_tools()` method alongside the existing `chat()`:

```python
async def chat_with_tools(
    self,
    messages: list[dict],
    tools: list[dict] | None = None,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> ChatCompletion:
    """Chat completion with optional tool definitions.
    Returns the full response object so the caller can inspect
    tool_calls. Caller handles the tool-call loop.
    """
```

### Config — `forge_bot/config.py`

```python
llm_tool_mode: str = Field(
    default="auto",
    description="Tool calling strategy: 'native', 'prompt', or 'auto'",
)
```

- `"native"`: Always use OpenAI `tools` parameter
- `"prompt"`: Inject tool descriptions into system prompt, model uses ```tool JSON blocks
- `"auto"` (default): Try native first, fall back to prompt if unsupported

### Handler Changes — `forge_bot/handlers/issue_comment.py`

Replace `_run_with_fetch_loop()` with `_run_with_tool_loop()`:

1. Build system prompt (with `tool_descriptions` for prompt mode, empty for native mode)
2. Construct messages: `[system, user]`
3. Loop (max 5 iterations):
   - **Native mode**: Call `chat_with_tools(messages, tools=...)`. If `tool_calls` in response, execute via registry, append results, continue.
   - **Prompt mode**: Parse ```tool JSON blocks from content. If found, execute, append results, continue.
   - **Auto mode**: Try native first, on error switch to prompt mode.
4. Clean stray markers from final output

### Prompt Template — `prompts/issue_respond.j2`

Replace `[FETCH:]` block with:

```jinja2
{% if tool_descriptions %}
=== AVAILABLE TOOLS ===
{{ tool_descriptions }}
IMPORTANT: If you need information not provided below, USE a tool to get it.
{% elif repo_tree %}
If you need to read a file, use the tools provided to you.
{% endif %}
```

## File Summary

| File | Change |
|------|--------|
| `forge_bot/tools/__init__.py` | **NEW** — empty package init |
| `forge_bot/tools/base.py` | **NEW** — BaseTool, ToolResult, ToolParameter |
| `forge_bot/tools/registry.py` | **NEW** — ToolRegistry |
| `forge_bot/tools/fetch_file.py` | **NEW** — FetchFileTool |
| `forge_bot/tools/get_commit.py` | **NEW** — GetCommitTool |
| `forge_bot/tools/search_code.py` | **NEW** — SearchCodeTool |
| `forge_bot/clients/llm.py` | Add `chat_with_tools()` method |
| `forge_bot/config.py` | Add `llm_tool_mode` field |
| `forge_bot/handlers/issue_comment.py` | Replace fetch-loop with tool-loop |
| `prompts/issue_respond.j2` | Replace `[FETCH:]` block with tool awareness |
| `.env.example` | Add `LLM_TOOL_MODE` |
| `tests/test_tools.py` | **NEW** — tool unit tests |
| `tests/test_llm_client.py` | Add chat_with_tools tests |
| `tests/test_issue_comment_handler.py` | Adapt fetch-loop tests to tool-loop |

## Design Decisions

- **Per-request registry**: Each tool instance needs repo context from the webhook event
- **Keep proactive context-fetching**: Tools are a safety net, not a replacement
- **Dual mode**: Native tool calling for capable models, prompt-based for others
- **Auto-detection**: Try native first, cache fallback decision per-request
- **```tool JSON blocks**: Structured format for prompt-based fallback (better than `[FETCH:]` markers)
