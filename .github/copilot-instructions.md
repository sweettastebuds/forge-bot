# GitHub Copilot Instructions for forge-bot

forge-bot is a Python 3.12 webhook service that connects Gitea/Forgejo repositories to an
OpenAI-compatible LLM. It receives webhook events, routes them to handlers, runs an agentic
tool-calling loop in a per-event Docker workspace container, and posts results back via the
Forge REST API. Every PR should be reviewed through this lens.

---

## Architecture

```
Webhook POST → HMAC-SHA256 verify → return 200 → background task
  → router.py (event type + action → handler)
    → PullRequestHandler  — AgentLoop reviews the diff
    → IssueCommentHandler — AgentLoop answers @mentions
```

Key invariants:

- **Return 200 before processing.** Gitea/Forgejo has a 5 s webhook timeout. The handler
  must be dispatched as a `BackgroundTask`; never `await` it on the request path.
- **Self-loop guard.** `router.py` must drop any event where `sender.login == bot_username`.
  Removing or weakening this check causes infinite webhook loops.
- **HMAC verification.** `server.py` reads raw bytes *before* JSON parsing and uses
  `hmac.compare_digest()`. Any shortcut (e.g. comparing strings directly, parsing first)
  is a security regression.
- **Container destroyed in `finally`.** Every handler creates a `ContainerManager` and must
  destroy it in a `finally` block regardless of success or failure.

---

## Stack & Style Rules

- **Python 3.12** — use modern syntax: `X | Y` unions, `match`, `f`-strings with `=`.
- **Async everywhere.** Every function that touches the network, filesystem, or Docker
  must be `async def` / `await`. Never call sync-blocking code on the event loop directly;
  use `asyncio.to_thread()` if unavoidable.
- **Type hints on all public signatures.** Return types included. Avoid `Any` unless
  unavoidable (e.g. JSON blobs).
- **Pydantic v2 models** for all external data (webhook payloads, API responses, config).
  Use `model_validate()`, `Field()`, and `field_validator(mode="before")` for null-coercion
  (Gitea sends `null` for empty lists like `assignees` and `requested_reviewers`).
- **`pydantic-settings` `Settings` class** (`config.py`) for all environment variables.
  Never read `os.environ` directly elsewhere.
- **Jinja2 `.j2` templates** in `forge_bot/prompts/` for every LLM prompt. Never build
  prompts by string concatenation in handler code.
- **YAML-driven API client.** Endpoint definitions live in
  `forge_bot/api/definitions/gitea.yaml` (and `forgejo.yaml`). Do not hard-code API URLs
  or add new `httpx` calls outside `GenericForgeClient`. New endpoints belong in the YAML.
- **Ruff** is the linter/formatter (target Python 3.12, line length 100, rules E F I N W UP
  B SIM). All new code must pass `ruff check` and `ruff format --check`.

---

## AgentLoop & Tool System

- The `AgentLoop` (`forge_bot/agent.py`) is the core agentic primitive. It drives an
  LLM in a tool-calling loop (max rounds until stuck), with budget-aware context trimming,
  `/workspace/.notes` external memory, `FORGE_TODO` progress markers, and a stuck-detection
  fallback.
- Tools are `BaseTool` subclasses (`forge_bot/tools/base.py`) with `name`, `description`,
  `parameters: list[ToolParameter]`, and `async execute(**kwargs) -> ToolResult`.
  `to_openai_schema()` must remain in sync with `parameters`.
- The `execute` built-in tool is defined inline in `agent.py` as an OpenAI schema dict;
  it runs shell commands in the workspace container. Blocked patterns (`rm -rf /`, `mkfs`,
  `dd if=`, `> /dev/`) must not be removed or weakened.
- Extra tools (e.g. `RetrievalTool`) are injected at construction time; they must implement
  `BaseTool` and be passed as `extra_tools=[...]` to `AgentLoop`.

---

## ContainerManager

- One container per webhook event, destroyed in `finally`.
- Resource limits (`--memory=512m`, `--cpus=1.0`, `--pids-limit=256`) must not be
  increased without justification.
- The container's network can be disabled via `CONTAINER_NETWORK_ENABLED=false`; code
  must not assume network is always available.
- Readiness is detected by polling for `FORGE_READY` in container logs; do not skip this
  poll or replace it with a fixed `sleep`.
- The API token is embedded in the clone URL for authenticated access. Never log the
  clone URL at INFO or above.

---

## StatusCommentManager

- Maintains **two** Gitea comments per interaction: a live-updating status comment and a
  final response comment. Never collapse them into one.
- `post_initial_status()` → `update_phase(msg)` → `record_tool_call(record)` →
  `post_response(text)` → `finalize_status(label)` is the canonical lifecycle. Do not
  call these out of order.
- Status updates are best-effort: all calls must be wrapped in `try/except` and failures
  logged at WARNING, never re-raised.

---

## Testing Conventions

- **pytest-asyncio** with `asyncio_mode = "auto"` — async tests run without extra
  configuration; `@pytest.mark.asyncio` is acceptable and commonly used in this repo.
- **pytest-httpx** for mocking outbound HTTP (`httpx.AsyncClient`).
- Use `unittest.mock.AsyncMock` for coroutine dependencies (`api.call`, `llm.chat`, etc.).
  Use `MagicMock` for synchronous objects (settings, container, status).
- Handler tests patch `ContainerManager`, `StatusCommentManager`, and `AgentLoop` with
  `patch(...)` context managers — see `tests/test_pull_request_handler.py` for the
  canonical pattern.
- Settings in tests are `MagicMock()` objects with only the attributes under test set
  explicitly. Do not instantiate the real `Settings` class in unit tests.
- Test files live in `tests/` and mirror the module they test
  (e.g. `test_pull_request_handler.py` ↔ `forge_bot/handlers/pull_request.py`).

---

## What to Look For in PRs

### Correctness & Reliability
- Any `await` on the webhook request path (before `BackgroundTasks.add_task`) will block
  the 200 response and risk Gitea timeouts — flag as critical.
- Missing `finally: await container.destroy()` in a handler leaks Docker containers.
- Swallowed exceptions in the agent loop (`except Exception: pass`) hide failures from
  operators; require at least `logger.warning(..., exc_info=True)`.
- New LLM calls must go through `LLMClient` with its `asyncio.Semaphore` concurrency
  guard — never construct `AsyncOpenAI` directly in handler code.

### Security
- HMAC verification must remain on raw bytes before JSON parsing. Flag any PR that moves
  the signature check after `json.loads()`.
- The `execute` tool's blocked-pattern list (`_BLOCKED_PATTERNS` in `agent.py`) must not
  shrink.
- API tokens and secrets must never be logged. Check any new `logger.*` lines that
  reference `settings`, `clone_url`, or `token`.
- New endpoints added to `gitea.yaml`/`forgejo.yaml` that expose write operations
  (POST/PATCH/DELETE) warrant extra scrutiny.

### Context Window & LLM Budget
- Every new field added to a Jinja2 prompt template increases the system-prompt size.
  Large fields (diffs, file contents) must be truncated before injection.
- Context trimming in `AgentLoop._prepare_messages()` assumes the system prompt is at
  index 0 and the initial user message at index 1 — changes to message ordering break it.

### API Client / YAML Definitions
- New API calls must be added as YAML endpoint definitions, not as raw `httpx` calls.
- Path parameters use `{name}` placeholders; they must have a matching `params` entry
  with `location: path`.
- `response_type: text` endpoints return `str`; `response_type: json` return `dict|list`.
  Callers must handle both types correctly.

### Async & Concurrency
- Docker SDK calls (`docker.from_env()`, `container.exec_run()`) are synchronous and must
  be wrapped with `asyncio.to_thread()`.
- Shared mutable state accessed from multiple background tasks must be protected (e.g.
  `DeliveryTracker` uses an `OrderedDict` — confirm thread-safety for any new shared state).

### Prompt Templates
- Templates must not concatenate untrusted user input without escaping; the `autoescape`
  is disabled in the Jinja2 environment (prompts are not HTML), so Markdown injection is
  possible. Severity: 🟡 Suggestion unless the content can break tool-calling JSON.
- Low temperature (0.2) is intentional for determinism. PRs raising it above 0.5 need
  justification.

---

## File Map (quick reference)

| Path | Purpose |
|------|---------|
| `forge_bot/server.py` | FastAPI app, HMAC verification, background dispatch |
| `forge_bot/router.py` | Event routing, self-loop guard, @mention detection |
| `forge_bot/agent.py` | AgentLoop: tool-calling loop, context trimming, stuck detection |
| `forge_bot/config.py` | All settings via pydantic-settings |
| `forge_bot/models.py` | Pydantic webhook payload models |
| `forge_bot/api/client.py` | YAML-driven HTTP client |
| `forge_bot/api/definitions/` | Gitea & Forgejo endpoint YAML definitions |
| `forge_bot/handlers/pull_request.py` | PR review handler |
| `forge_bot/handlers/issue_comment.py` | @mention reply handler |
| `forge_bot/container/manager.py` | Per-event Docker workspace |
| `forge_bot/status/manager.py` | Two-comment status system |
| `forge_bot/tools/base.py` | BaseTool ABC, ToolResult, ToolParameter |
| `forge_bot/prompts/` | Jinja2 prompt templates |
| `forge_bot/retrieval/` | SmartRetriever: BM25 + LLM-ranked context retrieval |
| `forge_bot/rag/` | Optional ChromaDB-based RAG pipeline (disabled by default) |
