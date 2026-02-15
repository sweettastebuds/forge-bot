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
                        Fetches repo tree, auto-fetches files referenced in thread,
                        extracts attachment URLs from markdown, passes all to LLM
    issue_assign.py   — Issue assigned to bot: greeting/triage (not yet implemented)
  clients/
    forge.py      — httpx.AsyncClient wrapper for Gitea/Forgejo API v1
                    Implemented: get_self, get_pull_diff, get_pull_files,
                    get_issue_comments, post_comment, get_file_content, get_repo_tree
                    Not yet: get_issue, get_pull_request, post_review
                    Auth: Authorization: token {FORGE_API_TOKEN}
                    PRs and issues share index namespace for comments
    llm.py        — AsyncOpenAI wrapper, asyncio.Semaphore for concurrency control
  sandbox/
    orchestrator.py — Create/destroy ephemeral containers via DinD, resource limits
    images.py       — Image registry, pre-pull logic, sandbox-images.json loading
    parser.py       — Parse /run commands + code blocks from comment body
  rag/              — Optional (RAG_ENABLED=false by default)
    pipeline.py, 
    ingester.py, 
    chunker.py, 
    embedder.py, 
    store.py, 
    retriever.py
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
- Attachments: Gitea markdown images use /attachments/{uuid}/{filename} paths
- Null coercion: Gitea sends null for empty lists (assignees, requested_reviewers) — use field_validator(mode="before")

## Critical Implementation Details

1. **HMAC verification**: Read raw body bytes BEFORE json parsing. Use hmac.compare_digest().
2. **Self-loop prevention**: Drop events where sender.login == bot_username. Without this, bot comments trigger infinite webhook loops.
3. **@mention detection**: `r'(?:^|\s)@{username}(?:\s|$|[.,;:!?\)])'` with re.MULTILINE
4. **LLM concurrency**: asyncio.Semaphore(LLM_MAX_CONCURRENT) on all LLM calls.
5. **Sandbox defaults**: --network=none, --memory=512m, --cpus=1.0, --pids-limit=256, 60s timeout.
6. **Sandbox images**: Pre-pull on startup (SANDBOX_PREPULL_IMAGES env), on-demand pull for rest.
7. **Prompt templates**: Jinja2 .j2 files in prompts/ dir. Low temperature (0.2). Severity prefixes (🔴🟡💡).
8. **Repo context in issues**: IssueCommentHandler fetches repo tree + auto-fetches files mentioned in thread (max 5 files, 8k chars each). Attachment URLs extracted from markdown and surfaced to LLM.
9. **Graceful degradation**: All context-fetching (tree, files, attachments) is wrapped in try/except — failures are logged but never block the reply.

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
LLM_MAX_CONCURRENT (3), SANDBOX_ENABLED (true), SANDBOX_TIMEOUT (60),
SANDBOX_PREPULL_IMAGES (python,node), RAG_ENABLED (false), LOG_LEVEL (INFO)

## Style

- Async everywhere (async def, await). No sync blocking calls.
- Type hints on all function signatures.
- Pydantic models for all external data (webhook payloads, API responses).
- Errors: log + post user-facing comment on Gitea, never crash the server.
- Tests: pytest-asyncio for async, pytest-httpx for mocking HTTP.
