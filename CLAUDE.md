# CLAUDE.md

## Project

forge-bot: Python webhook service for Gitea/Forgejo AI bot. FastAPI receives webhooks, routes to handlers, calls OpenAI-compatible LLM, posts results back via REST API. Docker-in-Docker for sandboxed code execution.

## Stack

- Python 3.12, FastAPI, Uvicorn, httpx, pydantic-settings
- openai SDK (AsyncOpenAI) for any OpenAI-compatible endpoint
- docker SDK for DinD sandbox orchestration
- Jinja2 for prompt templates
- Optional: ChromaDB + sentence-transformers + tree-sitter for RAG
- Tests: pytest, pytest-asyncio, pytest-httpx

## Architecture

```
Webhook POST → HMAC verify → return 200 → background task
  → Event Router (pull_request|issue_comment|issues)
    → Handler (fetch context via Forge API → build prompt → call LLM → post comment)
```

Key constraint: Gitea webhook timeout is 5s. Must return 200 before processing.

## Module Map

```
forge_bot/
  server.py       — FastAPI app, single POST /webhook endpoint, HMAC-SHA256 verification
  config.py       — pydantic-settings, all env vars, validated at startup
  router.py       — event type + action → handler dispatch, self-loop guard, @mention detection
  models.py       — Pydantic models: PullRequestEvent, IssueCommentEvent, WebhookUser, etc.
  handlers/
    base.py           — Abstract base: forge_client, llm_client, config injected
    pull_request.py   — PR opened/synchronized: fetch diff, build review prompt, post comment
    issue_comment.py  — Comment created: @mention reply with repo context
                        Dynamic context budgets via _context_limits(context_window),
                        conversation summarization for long threads,
                        fetches repo tree, grounding files, referenced files/commits,
                        downloads text attachments, runs LLM with tool-calling loop
                        (supports native OpenAI tool_calls and prompt-based ```tool
                        JSON blocks, controlled by LLM_TOOL_MODE: native/prompt/auto)
                        Tools: fetch_file, get_commit, search_code (max 5 rounds)
                        handle_run(): sandbox code execution via /run command
                        handle_index(): on-demand RAG re-indexing via /index command
    issue_assign.py   — Issue assigned to bot: greeting/triage (not yet implemented)
  clients/
    forge.py      — httpx.AsyncClient wrapper for Gitea/Forgejo API v1
                    Implemented: get_self, get_pull_diff, get_pull_files,
                    get_issue_comments, post_comment, get_file_content,
                    get_repo_tree, get_commit, download_url
                    Not yet: get_issue, get_pull_request, post_review
                    Auth: Authorization: token {FORGE_API_TOKEN}
                    PRs and issues share index namespace for comments
    llm.py        — AsyncOpenAI wrapper, asyncio.Semaphore for concurrency control
                    chat() returns string, chat_with_tools() returns ChatCompletion
  tools/
    base.py       — BaseTool ABC, ToolResult/ToolParameter dataclasses,
                    to_openai_schema() and to_prompt_text() converters
    registry.py   — ToolRegistry: register/get/execute tools, openai_schemas(),
                    prompt_text() for prompt-based fallback
    fetch_file.py — FetchFileTool: read file contents via Forge API
    get_commit.py — GetCommitTool: retrieve commit info by SHA
    search_code.py — SearchCodeTool: grep-like search across repo files
  sandbox/
    orchestrator.py — Create/destroy ephemeral containers via DinD, resource limits
    images.py       — Image registry, pre-pull logic, sandbox-images.json loading
    parser.py       — Parse /run commands + code blocks from comment body
  rag/              — Optional (RAG_ENABLED=false by default)
    chunker.py    — AST-aware chunking (tree-sitter for Python/JS) with
                    sliding-window fallback for other languages
    embedder.py   — Dual-backend: local sentence-transformers or remote
                    OpenAI-compatible /v1/embeddings endpoint
    store.py      — ChromaDB persistent vector store wrapper
    ingester.py   — Fetch repo files via Forge API, chunk, embed, store
    pipeline.py   — Orchestrates lazy-initialized RAG workflow
    retriever.py  — High-level query interface with context truncation
  utils/
    dedup.py      — LRU OrderedDict (max 10k) tracking X-Gitea-Delivery UUIDs
    diff.py       — Unified diff parser
    formatting.py — Markdown formatting helpers
```

## Gitea/Forgejo API Patterns

- All endpoints: /api/v1/repos/{owner}/{repo}/...
- Diff: GET /pulls/{index}.diff (raw text)
- Comments: POST /issues/{index}/comments (works for PRs too)
- Reviews: POST /pulls/{index}/reviews (inline comments unreliable — fallback to summary)
- Webhook headers: check X-Forgejo-*first, fallback X-Gitea-*
- Signature: HMAC-SHA256 hex digest over raw body bytes
- Delivery ID: X-Gitea-Delivery / X-Forgejo-Delivery UUID for dedup
- Bot identity: GET /api/v1/user on startup to learn own username
- File content: GET /repos/{owner}/{repo}/raw/{filepath}?ref={ref}
- Repo tree: GET /repos/{owner}/{repo}/git/trees/{ref}?recursive=true
- Commits: GET /repos/{owner}/{repo}/git/commits/{sha}
- Attachments: Gitea markdown images use /attachments/{uuid}/{filename} paths — no API, parse from body
- Null coercion: Gitea sends null for empty lists (assignees, requested_reviewers) — use field_validator(mode="before")

## Critical Implementation Details

1. **HMAC verification**: Read raw body bytes BEFORE json parsing. Use hmac.compare_digest().
2. **Self-loop prevention**: Drop events where sender.login == bot_username. Without this, bot comments trigger infinite webhook loops.
3. **@mention detection**: `r'(?:^|\s)@{username}(?:\s|$|[.,;:!?\)])'` with re.MULTILINE
4. **LLM concurrency**: asyncio.Semaphore(LLM_MAX_CONCURRENT) on all LLM calls.
5. **Sandbox defaults**: --network=none, --memory=512m, --cpus=1.0, --pids-limit=256, 60s timeout.
6. **Sandbox images**: Pre-pull on startup (SANDBOX_PREPULL_IMAGES env), on-demand pull for rest.
7. **Prompt templates**: Jinja2 .j2 files in prompts/ dir. Low temperature (0.2). Severity prefixes (🔴🟡💡).
8. **Repo context in issues**: IssueCommentHandler fetches repo tree (using default_branch from webhook payload), proactively fetches grounding files (README.md, pyproject.toml, etc.), auto-fetches files mentioned in thread (max 5, 8k chars each), detects commit SHAs and fetches commit info, downloads text-based attachments. LLM uses tool-calling loop (fetch_file, get_commit, search_code) to request additional context — up to 5 rounds. Supports native OpenAI tool_calls and prompt-based ```tool JSON blocks (LLM_TOOL_MODE: native/prompt/auto).
9. **Graceful degradation**: All context-fetching (tree, files, commits, attachments) is wrapped in try/except — failures logged at WARNING level, never block the reply.
10. **Dynamic context limits**: `_context_limits(context_window)` derives max_recent_comments, summary_max_tokens, max_tree_entries, max_file_chars, max_grounding_file_chars from LLM_CONTEXT_WINDOW. Small models get fewer comments and smaller context; large models get more.
11. **Conversation summarization**: When thread exceeds max_recent_comments, older comments are summarized via a dedicated LLM call (conversation_summary.j2, temp=0.1). Summary explicitly filters bot hallucinations. Fallback: naive truncation (first 2 + last comment) on LLM failure.
12. **Anti-hallucination prompt**: issue_respond.j2 opens with MANDATORY RULES forbidding fabrication of files, commits, or configs. Ground truth data placed at end of prompt for recency attention bias.

## Bot Commands (in Gitea comments)

- `@bot /run <language>` + fenced code block — Execute code in a sandboxed container
- `@bot /run --net <language>` + code block — Execute with network access enabled
- `@bot /index` — Re-index the repository for RAG context (requires RAG_ENABLED=true)
- `@bot <question>` — Ask a question about the codebase (uses repo context + optional RAG)

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
```

## Env Vars (Required)

FORGE_INSTANCE_URL, FORGE_API_TOKEN, FORGE_WEBHOOK_SECRET, LLM_API_KEY

## Env Vars (Key Optional)

LLM_BASE_URL (default: <https://api.openai.com/v1>), LLM_MODEL (gpt-4o),
LLM_TEMPERATURE (0.2), LLM_MAX_TOKENS (4096), LLM_TIMEOUT (120),
LLM_MAX_CONCURRENT (3), LLM_CONTEXT_WINDOW (8192 — match your model),
LLM_TOOL_MODE (auto — native/prompt/auto),
SANDBOX_ENABLED (true), SANDBOX_TIMEOUT (60),
SANDBOX_PREPULL_IMAGES (python,node), RAG_ENABLED (false), LOG_LEVEL (INFO)

## Style

- Async everywhere (async def, await). No sync blocking calls.
- Type hints on all function signatures.
- Pydantic models for all external data (webhook payloads, API responses).
- Errors: log + post user-facing comment on Gitea, never crash the server.
- Tests: pytest-asyncio for async, pytest-httpx for mocking HTTP.

## Commit Practices

- Make small, focused commits along the way — don't batch all changes into one giant commit.
- Each commit should be a logical unit (e.g., "add tool base classes", "add tool tests", "update handler").
- Write descriptive commit messages: `feat:`, `fix:`, `test:`, `docs:` prefixes.
- Run tests before committing to verify nothing is broken.
