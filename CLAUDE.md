# CLAUDE.md

## Project

forge-bot: Python webhook service for Gitea/Forgejo AI bot. FastAPI receives webhooks, routes to handlers, calls OpenAI-compatible LLM with tool-calling, posts results back via REST API. Per-event workspace containers for sandboxed code execution. Optional RAG with ChromaDB and tree-sitter for context retrieval. Designed for reliability, security, and maintainability.

See [docs/TODO.md](docs/TODO.md) for current work items and backlog.

## Stack

- Python 3.12, FastAPI, Uvicorn, httpx, pydantic-settings
- openai SDK (AsyncOpenAI) for any OpenAI-compatible endpoint
- docker SDK for per-event workspace containers
- Jinja2 for prompt templates
- PyYAML for YAML-driven API endpoint definitions
- Optional: ChromaDB + sentence-transformers + tree-sitter for RAG
- Tests: pytest, pytest-asyncio, pytest-httpx

## Architecture

```
Webhook POST → HMAC verify → return 200 → background task
  → Event Router (pull_request|issue_comment|issues)
    → Handler
      → IssueCommentHandler: create container → register tools → tool-calling loop → verify → post
      → PullRequestHandler: fetch diff → build prompt → call LLM → post comment
```

Key constraint: Gitea webhook timeout is 5s. Must return 200 before processing.

### Tool-calling Loop (IssueCommentHandler)

The LLM drives its own context-gathering via a tool-calling loop (max 10 rounds):

1. Post initial status comment → create workspace container (repo clone)
2. Register tools: `search_api`, `api_call`, `exec`, `todo`
3. LLM calls tools → handler executes → results fed back to LLM
4. In-loop verification: relevance check, hallucination detection, progress tracking
5. Pre-post verification: validate response against tool output, retry once if failed
6. Post final response as separate comment, finalize status

## Module Map

```
forge_bot/
  server.py       — FastAPI app, POST /webhook + GET /health, HMAC-SHA256 verification, lifespan
  config.py       — pydantic-settings Settings class, all env vars, validated at startup
  router.py       — event type + action → handler dispatch, self-loop guard, @mention detection
  models.py       — Pydantic models: PullRequestEvent, IssueCommentEvent, IssuesEvent, etc.
  api/
    client.py     — GenericForgeClient: YAML-driven HTTP client, call any defined endpoint
    loader.py     — YAML API definition loader with caching
    schema.py     — Pydantic: EndpointDef, EndpointParam, ApiDefinitionFile
    definitions/
      gitea.yaml    — ~20 Gitea API endpoint definitions
      forgejo.yaml  — ~20 Forgejo API endpoint definitions
  clients/
    forge.py      — ForgeClient (DEPRECATED: replaced by GenericForgeClient in api/)
    llm.py        — LLMClient: AsyncOpenAI wrapper, chat() + chat_with_tools(), Semaphore concurrency
  container/
    manager.py    — ContainerManager: per-event persistent containers, exec support, async context mgr
  handlers/
    base.py       — BaseHandler: api_client, llm_client, settings, bot_username injected; Jinja2 loader
    pull_request.py — PR opened/synchronized: fetch diff, build review prompt, post comment
    issue_comment.py — @mention reply: tool-calling loop with verification, status updates, container lifecycle
    verification.py  — In-loop checks (relevance, hallucination, progress) + pre-post response verification
  tools/
    base.py       — BaseTool ABC, ToolResult, ToolParameter; OpenAI schema generation
    registry.py   — ToolRegistry: register, lookup, execute tools; OpenAI schemas for all tools
    exec_tool.py  — ExecTool: run shell commands in workspace container, blocked patterns
    api_call.py   — ApiCallTool: execute YAML-defined API endpoints, auto-inject owner/repo
    search_api.py — SearchApiTool: search API endpoints by keyword
    todo.py       — TodoTool: visible progress checklist via status comment
  status/
    manager.py    — StatusCommentManager: two-comment system (status + response), real-time updates
    formatter.py  — Markdown rendering for todos, tool calls; TodoItem, ToolCallRecord dataclasses
  sandbox/          — DEPRECATED (replaced by container/manager.py)
    orchestrator.py — SandboxOrchestrator: old fire-and-forget containers
    images.py       — ImageRegistry: language → Docker image mapping, pre-pull
    parser.py       — Parse /run commands + code blocks from comment body
  rag/              — Optional (RAG_ENABLED=false by default)
    chunker.py    — AST-aware chunking (tree-sitter for Python/JS), sliding-window fallback
    embedder.py   — Dual-backend: local sentence-transformers or remote OpenAI-compatible API
    store.py      — ChromaDB persistent vector store wrapper
    ingester.py   — Fetch repo files via Forge API, chunk, embed, store
    pipeline.py   — RAGPipeline: lazy-initialized orchestrator
    retriever.py  — High-level query interface with context truncation
  prompts/
    issue_respond.j2 — System prompt for @mention replies (anti-hallucination rules, tool instructions)
    pr_review.j2     — System prompt for PR reviews (severity prefixes 🔴🟡💡)
  utils/
    dedup.py      — DeliveryTracker: LRU OrderedDict (max 10k) for X-Gitea-Delivery UUIDs
```

## API Client (YAML-driven)

The `GenericForgeClient` replaces the old hardcoded `ForgeClient`. API endpoints are defined in YAML files (`forge_bot/api/definitions/gitea.yaml` and `forgejo.yaml`):

```python
# Call any defined endpoint by name
result = await api_client.call("get_file_content", owner="o", repo="r", filepath="README.md")
 
# Search endpoints by keyword
matches = api_client.search("pull")
 
# List all available endpoints
names = api_client.list_endpoints()
```

Provider is selected via `FORGE_PROVIDER` env var (default: `gitea`).

## Tool System

Four tools are registered per interaction:

| Tool | Purpose | Key details |
|------|---------|-------------|
| `exec` | Run shell commands in workspace | Blocked: `rm -rf /`, `mkfs`, `dd if=`, `> /dev/`; max 120s timeout |
| `api_call` | Call any YAML-defined API endpoint | Auto-injects owner/repo from context |
| `search_api` | Search API endpoints by keyword | Fuzzy matching over endpoint names/descriptions |
| `todo` | Visible progress checklist | Actions: set, check, add; rendered in status comment |

Tools implement `BaseTool` ABC with `to_openai_schema()` for native tool-calling support.

## Status Comment System

`StatusCommentManager` maintains two Gitea comments per interaction:

1. **Status comment** — edited in real-time with: current phase, todo list, tool call history (collapsible)
2. **Response comment** — posted once at the end with the LLM's final answer

## Verification System

### In-loop checks (each tool-calling round)

- **Relevance**: pattern-based check that tool calls relate to the user's question (permissive default)
- **Hallucination**: compares LLM claims (e.g. "tests pass") against actual exec exit codes
- **Progress**: `ProgressTracker` detects stuck loops (2+ rounds with no new calls) and duplicate calls

### Pre-post check (before posting response)

- `verify_response()`: validates exit code claims, checks that file paths in response were seen in tool output
- On failure: retry once with correction prompt, then post with warning banner if still failing

## Container Lifecycle

`ContainerManager` creates a persistent container per webhook event:

- Image: `CONTAINER_WORKSPACE_IMAGE` (default: `forge-bot-workspace:latest`, built from `Dockerfile.workspace`)
- Starts with empty `/workspace`; LLM clones the repo via `exec` tool
- Resource limits: `--memory=512m`, `--cpus=1.0`, `--pids-limit=256`
- Network: `bridge` (configurable) or `none`
- Token injection: API token embedded in clone URL for authenticated access
- Readiness: polls for `FORGE_READY` marker in container logs
- Destroyed after event processing completes (in `finally` block)

## Gitea/Forgejo API Patterns

- All endpoints defined in YAML: `/api/v1/repos/{owner}/{repo}/...`
- Webhook headers: check X-Forgejo-*first, fallback X-Gitea-*
- Signature: HMAC-SHA256 hex digest over raw body bytes
- Delivery ID: X-Gitea-Delivery / X-Forgejo-Delivery UUID for dedup
- Bot identity: `get_authenticated_user` endpoint on startup to learn own username
- Null coercion: Gitea sends null for empty lists (assignees, requested_reviewers) — use field_validator(mode="before")

## Critical Implementation Details

1. **HMAC verification**: Read raw body bytes BEFORE json parsing. Use `hmac.compare_digest()`.
2. **Self-loop prevention**: Drop events where `sender.login == bot_username`. Without this, bot comments trigger infinite webhook loops.
3. **@mention detection**: `r'(?:^|\s)@{username}(?:\s|$|[.,;:!?\)])'` with `re.MULTILINE`
4. **LLM concurrency**: `asyncio.Semaphore(LLM_MAX_CONCURRENT)` on all LLM calls.
5. **Container resource limits**: `--memory=512m`, `--cpus=1.0`, `--pids-limit=256`, timeout configurable.
6. **Prompt templates**: Jinja2 `.j2` files in `forge_bot/prompts/`. Low temperature (0.2). PR reviews use severity prefixes (🔴🟡💡).
7. **Anti-hallucination**: `issue_respond.j2` opens with MANDATORY RULES forbidding fabrication. In-loop hallucination checks compare LLM claims vs exec exit codes. Pre-post verification catches fabricated file paths.
8. **Graceful degradation**: All context-fetching and tool execution wrapped in try/except — failures logged at WARNING level, never crash the server.
9. **Tool-calling fallback**: If `chat_with_tools` fails, falls back to simple `chat()` without tools. If that also fails, returns a canned error message.
10. **Exec safety**: `ExecTool` blocks destructive patterns (`rm -rf /`, `mkfs`, `dd if=`, `> /dev/`). Max timeout 120s.

## Bot Commands (in Gitea comments)

- `@bot <question>` — Ask a question about the codebase (LLM uses tools to explore and answer)
- `@bot /run <language>` + fenced code block — Execute code in a sandboxed container (legacy sandbox)
- `@bot /index` — Re-index the repository for RAG context (requires RAG_ENABLED=true)

## Commands

```bash
# Run
uvicorn forge_bot.server:app --host 0.0.0.0 --port 8080
 
# Test
pytest
pytest tests/test_server.py -v
pytest -x --tb=short
 
# Lint
ruff check forge_bot/
ruff format forge_bot/
 
# Docker
docker compose up -d
docker compose logs -f forge-bot
 
# Build workspace image
docker build -f Dockerfile.workspace -t forge-bot-workspace:latest .
```

## Env Vars (Required)

FORGE_INSTANCE_URL, FORGE_API_TOKEN, FORGE_WEBHOOK_SECRET, LLM_API_KEY

## Env Vars (Key Optional)

| Variable | Default | Description |
|----------|---------|-------------|
| LLM_BASE_URL | `https://api.openai.com/v1` | LLM API base URL |
| LLM_MODEL | `gpt-4o` | Model name |
| LLM_TEMPERATURE | `0.2` | Generation temperature |
| LLM_MAX_TOKENS | `4096` | Max response tokens |
| LLM_TIMEOUT | `120` | Request timeout (seconds) |
| LLM_MAX_CONCURRENT | `3` | Max parallel LLM requests |
| LLM_CONTEXT_WINDOW | `8192` | Context window (tokens) — match your model |
| FORGE_PROVIDER | `gitea` | API provider: `gitea` or `forgejo` |
| SANDBOX_ENABLED | `true` | Enable sandbox/container features |
| SANDBOX_TIMEOUT | `60` | Container command timeout (seconds) |
| SANDBOX_MEMORY | `512m` | Container memory limit |
| SANDBOX_CPUS | `1.0` | Container CPU limit |
| CONTAINER_WORKSPACE_IMAGE | `forge-bot-workspace:latest` | Docker image for workspace containers |
| CONTAINER_NETWORK_ENABLED | `true` | Allow network in workspace containers |
| RAG_ENABLED | `false` | Enable RAG pipeline |
| LOG_LEVEL | `INFO` | Logging level |
| BOT_COMMAND_PREFIX | `/` | Command prefix |

## Test Structure

```
tests/
  conftest.py                   — Shared fixtures (settings, clients, mock events)
  test_server.py                — Webhook endpoint, HMAC verification
  test_router.py                — Event routing, self-loop guard, @mention
  test_models.py                — Pydantic webhook models
  test_config.py                — Settings parsing and validation
  test_api_client.py            — GenericForgeClient calls and search
  test_api_loader.py            — YAML definition loading
  test_api_schema.py            — Schema validation
  test_forge_client.py          — ForgeClient (deprecated)
  test_llm_client.py            — LLM chat and tool calling
  test_pull_request_handler.py  — PR review handler
  test_issue_comment_handler.py — Tool-calling loop, verification
  test_container_manager.py     — Container lifecycle
  test_tools.py                 — Tool implementations
  test_status.py                — Status comment manager
  test_verification.py          — Relevance, hallucination, progress
  test_dedup.py                 — Delivery deduplication
  test_sandbox/                 — Sandbox subsystem (parser, images, orchestrator)
  test_rag/                     — RAG subsystem (chunker, embedder, ingester, pipeline)
```

Pytest config: `asyncio_mode = "auto"`, `testpaths = ["tests"]`.

## Style

- Async everywhere (`async def`, `await`). No sync blocking calls except via `asyncio.to_thread()`.
- Type hints on all function signatures.
- Pydantic models for all external data (webhook payloads, API responses).
- Errors: log + post user-facing comment on Gitea, never crash the server.
- Tests: pytest-asyncio for async, pytest-httpx for mocking HTTP.
- Ruff: target Python 3.12, line length 100, rules: E, F, I, N, W, UP, B, SIM.
