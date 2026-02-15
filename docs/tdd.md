# Technical Design Document: AI Bot Service for Gitea/Forgejo

**Project Codename:** `forge-bot`
**Version:** 0.1.0 (MVP)
**Status:** Draft
**Date:** 2026-02-14

---

## 1. Overview

forge-bot is a standalone, self-hosted Python service that acts as an AI-powered collaborator on Gitea and Forgejo instances. It operates as a regular bot user account, receives webhook events for pull requests and issues, and uses any OpenAI-compatible LLM to generate contextual responses — code reviews, issue triage, Q&A, and code execution.

The service runs as a Docker container with Docker-in-Docker (DinD) capabilities, enabling isolated sandbox environments for code execution and testing. An optional RAG (Retrieval-Augmented Generation) context pipeline provides repository-aware responses by ingesting, chunking, and embedding codebases.

### 1.1 Design Goals

- **Platform-agnostic LLM backend:** Consume any OpenAI-compatible endpoint (OpenAI, Ollama, vLLM, LiteLLM, Anthropic via proxy, etc.)
- **Gitea + Forgejo compatibility:** Single codebase supporting both platforms (shared API surface, normalized header handling)
- **Isolation:** Docker-in-Docker for sandboxed code execution per review/comment — no shared state, no host contamination
- **Optional intelligence layer:** RAG context pipeline that can be enabled/disabled without affecting core functionality
- **Operational simplicity:** Single Docker image, environment variable configuration, no external database required for MVP

### 1.2 MVP Scope

The first release targets the following user stories:

| # | As a... | I want to... | Trigger |
|---|---------|-------------|---------|
| 1 | Developer | Get an AI code review when I open a PR | Bot is assigned to PR, or PR is opened with bot as reviewer |
| 2 | Developer | Get an AI code review when I push new commits | `pull_request` → `synchronized` event, bot is assigned |
| 3 | Developer | Ask the bot questions on an issue | `issue_comment` → `created`, bot is @mentioned or assigned |
| 4 | Developer | Ask the bot to run/test code in a comment | `issue_comment` with code fence and `/run` command |
| 5 | Maintainer | Have the bot auto-respond to issues it's assigned to | `issues` → `assigned` event where assignee is bot |

### 1.3 Out of Scope (MVP)

- Web UI / dashboard
- Multi-instance management (one bot per Gitea/Forgejo instance)
- PR auto-merge or approval workflows
- Fine-tuning or model training pipelines
- Git push/commit capabilities (bot is read + comment only)

---

## 2. Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Gitea / Forgejo Instance              │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌────────────────────────┐ │
│  │  Issues  │  │   PRs    │  │  Webhook Delivery      │ │
│  │          │  │          │  │  POST /webhook         │ │
│  └──────────┘  └──────────┘  └───────────┬────────────┘ │
└──────────────────────────────────────────┼──────────────┘
                                           │ HTTPS + HMAC-SHA256
                                           ▼
┌──────────────────────────────────────────────────────────┐
│                forge-bot Docker Container                │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │              FastAPI Webhook Server                │  │
│  │                                                    │  │
│  │  ┌──────────────┐  ┌───────────────────────────┐   │  │
│  │  │ HMAC Verify  │→ │ Event Router              │   │  │
│  │  │              │  │                           │   │  │
│  │  │ Return 200   │  │  pull_request → PR Handler│   │  │
│  │  │ immediately  │  │  issue_comment → Comment  │   │  │
│  │  └──────────────┘  │  issues (assigned) → Issue│   │  │
│  │                    └───────────┬───────────────┘   │  │
│  └────────────────────────────────┼───────────────────┘  │
│                                   ▼                      │
│  ┌────────────────────────────────────────────────────┐  │
│  │           Background Task Processor                │  │
│  │           (asyncio task queue)                     │  │
│  │                                                    │  │
│  │  ┌──────────────┐ ┌──────────┐ ┌────────────────┐  │  │
│  │  │ Gitea/Forgejo│ │  LLM     │ │ Sandbox        │  │  │
│  │  │ API Client   │ │  Client  │ │ Orchestrator   │  │  │
│  │  │              │ │(OpenAI)  │ │ (DinD)         │  │  │
│  │  └──────┬───────┘ └────┬─────┘ └───────┬────────┘  │  │
│  │         │              │               │           │  │
│  └─────────┼──────────────┼───────────────┼───────────┘  │
│            │              │               │              │
│            │              │               ▼              │
│            │              │    ┌────────────────────┐    │
│            │              │    │ Ephemeral Sandbox  │    │
│            │              │    │ Container (DinD)   │    │
│            │              │    │                    │    │
│            │              │    │ • Clone repo       │    │
│            │              │    │ • Run tests        │    │
│            │              │    │ • Execute code     │    │
│            │              │    │ • Destroyed after  │    │
│            │              │    └────────────────────┘    │
│            │              │                              │
│  ┌─────────┼──────────────┼────────────────────────────┐ │
│  │         │    Optional RAG Context Pipeline          │ │
│  │         │              │                            │ │
│  │  ┌──────┴───────┐ ┌───┴──────────┐ ┌────────────┐   │ │
│  │  │ Repo Ingester│ │ Chunker +    │ │ Vector     │   │ │
│  │  │ (git clone)  │ │ Embedder     │ │ Store      │   │ │
│  │  │              │ │              │ │ (ChromaDB) │   │ │
│  │  └──────────────┘ └──────────────┘ └────────────┘   │ │
│  └─────────────────────────────────────────────────────┘ │
│                                                          │
│            ▼                        ▼                    │
│   Gitea/Forgejo API         LLM Endpoint                 │
│   (post comments)           (OpenAI-compat)              │
└──────────────────────────────────────────────────────────┘
```

### 2.2 Component Inventory

| Component | Responsibility | Technology |
|-----------|---------------|------------|
| **Webhook Server** | Receive events, verify HMAC, route to handlers | FastAPI + Uvicorn |
| **Event Router** | Parse event type + action, filter relevant events, dispatch | Python (pure logic) |
| **Background Processor** | Async task execution, deduplication, concurrency control | `asyncio` task queue (MVP), ARQ/Redis (future) |
| **Gitea/Forgejo Client** | All REST API interactions (read issues/PRs/diffs, post comments) | `httpx.AsyncClient` |
| **LLM Client** | Chat completions against any OpenAI-compatible endpoint | `openai.AsyncOpenAI` |
| **Sandbox Orchestrator** | Create/destroy ephemeral Docker containers for code execution | `docker` SDK (Python) |
| **RAG Pipeline** (optional) | Ingest repos, chunk files, compute embeddings, serve context | ChromaDB + sentence-transformers |
| **Configuration** | Environment variables, validation, defaults | `pydantic-settings` |

---

## 3. Detailed Component Design

### 3.1 Webhook Server

The webhook server is a minimal FastAPI application with a single endpoint that handles all incoming Gitea/Forgejo webhook deliveries.

**Endpoint:** `POST /webhook`

**Request flow:**

1. Read raw request body (bytes — never parse before HMAC verification)
2. Extract signature from `X-Forgejo-Signature` or `X-Gitea-Signature` header
3. Compute HMAC-SHA256 over raw body using configured secret
4. Compare digests using `hmac.compare_digest()` (constant-time)
5. Return `HTTP 200` immediately
6. Parse JSON body and dispatch to event router as a background task

**Why return 200 before processing:** Gitea's default `DELIVER_TIMEOUT` is 5 seconds. LLM calls take 10–120+ seconds. If the webhook endpoint doesn't respond quickly, Gitea marks the delivery as failed and may retry, causing duplicate processing.

**Deduplication:** Track the `X-Gitea-Delivery` / `X-Forgejo-Delivery` UUID in an in-memory LRU dict (`OrderedDict`, max 10,000 entries). Reject duplicates before dispatching.

```
forge_bot/
├── server.py           # FastAPI app, webhook endpoint, HMAC verification
├── router.py           # Event type + action routing logic
├── handlers/
│   ├── pull_request.py # PR opened/synchronized handler
│   ├── issue_comment.py# Comment created handler (issues + PRs)
│   └── issue_assign.py # Issue/PR assigned handler
```

### 3.2 Event Router

The router examines the event type header and the `action` field in the payload, then decides whether the bot should act. The key filtering logic:

```
EVENT: pull_request
  ACTION: opened, synchronized
    → IF bot is in assignees OR bot is in requested_reviewers
      → Dispatch to PR review handler

EVENT: issue_comment
  ACTION: created
    → GUARD: sender is NOT the bot itself (prevent self-reply loops)
    → IF bot is @mentioned in comment body OR bot is in issue assignees
      → IF comment contains a slash command (/run, /review, /index, /help)
        → Dispatch to command handler (sandbox, review, etc.)
      → ELSE IF "is_pull" is true in payload
        → Dispatch to PR comment handler (can access diff context)
      → ELSE
        → Dispatch to issue comment handler

EVENT: issues
  ACTION: assigned
    → IF assigned user is bot
      → Dispatch to issue greeting/triage handler
```

**Bot identity detection:** On startup, the service calls `GET /api/v1/user` with its API token to learn its own username and user ID. These are used for three purposes:

1. **Assignee matching** — check if the bot's username appears in `assignees` or `requested_reviewers` lists
2. **@mention detection** — scan comment bodies for `@{bot_username}` using a regex pattern that handles word boundaries: `r'(?:^|\s)@{username}(?:\s|$|[.,;:!?])'`
3. **Self-loop prevention** — if `sender.login == bot_username`, the event is silently dropped. Without this, the bot's own comments trigger new webhook events, which trigger new LLM calls, which post new comments, creating an infinite loop

**@mention behavior:** When the bot is @mentioned in a comment, it treats the rest of the comment body (minus the @mention itself and any slash commands) as a natural language question or instruction. If the mention appears on a PR, the bot has access to the PR diff as additional context. If on an issue, it uses the issue title, body, and preceding comment thread.

### 3.3 Gitea/Forgejo API Client

A unified async client that abstracts all platform API interactions. Uses `httpx.AsyncClient` with connection pooling, automatic retries, and the bot's API token.

**Core methods:**

```python
class ForgeClient:
    """Unified client for Gitea and Forgejo REST API v1."""

    async def get_self() -> User
    # GET /api/v1/user — resolve bot identity on startup

    async def get_issue(owner, repo, index) -> Issue
    # GET /api/v1/repos/{owner}/{repo}/issues/{index}

    async def get_pull_request(owner, repo, index) -> PullRequest
    # GET /api/v1/repos/{owner}/{repo}/pulls/{index}

    async def get_pull_diff(owner, repo, index) -> str
    # GET /api/v1/repos/{owner}/{repo}/pulls/{index}.diff
    # Returns raw unified diff as plain text

    async def get_pull_files(owner, repo, index) -> list[ChangedFile]
    # GET /api/v1/repos/{owner}/{repo}/pulls/{index}/files
    # Returns list of changed files with stats

    async def get_issue_comments(owner, repo, index) -> list[Comment]
    # GET /api/v1/repos/{owner}/{repo}/issues/{index}/comments
    # Paginated — auto-follows pagination

    async def post_comment(owner, repo, index, body: str) -> Comment
    # POST /api/v1/repos/{owner}/{repo}/issues/{index}/comments
    # Works for both issues and PRs (shared index namespace)

    async def post_review(owner, repo, index, body, event, comments) -> Review
    # POST /api/v1/repos/{owner}/{repo}/pulls/{index}/reviews
    # event: "COMMENT" | "APPROVED" | "REQUEST_CHANGES"
    # NOTE: inline line comments unreliable in some versions;
    #       fall back to summary comment if review post fails

    async def get_file_content(owner, repo, filepath, ref) -> str
    # GET /api/v1/repos/{owner}/{repo}/raw/{filepath}?ref={ref}
    # For fetching specific files (e.g., for RAG context)

    async def get_repo_tree(owner, repo, ref, recursive=True) -> list[TreeEntry]
    # GET /api/v1/repos/{owner}/{repo}/git/trees/{ref}?recursive=true
    # For repository structure discovery
```

**Authentication:** All requests include `Authorization: token {FORGE_API_TOKEN}` header.

**Platform normalization:** The client auto-detects whether the target is Gitea or Forgejo by checking the `/api/v1/version` endpoint response on startup and adjusts header handling accordingly.

### 3.4 LLM Client

Wraps the OpenAI Python SDK's `AsyncOpenAI` client, configured via environment variables. Initialized once at startup and shared across all background tasks.

**Configuration:**

| Env Var | Purpose | Default |
|---------|---------|---------|
| `LLM_BASE_URL` | OpenAI-compatible API base URL | `https://api.openai.com/v1` |
| `LLM_API_KEY` | API key for the LLM endpoint | (required) |
| `LLM_MODEL` | Model name to use | `gpt-4o` |
| `LLM_TEMPERATURE` | Sampling temperature | `0.2` |
| `LLM_MAX_TOKENS` | Max response tokens | `4096` |
| `LLM_TIMEOUT` | Request timeout in seconds | `120` |
| `LLM_MAX_CONCURRENT` | Max concurrent LLM calls | `3` |

**Concurrency control:** An `asyncio.Semaphore(LLM_MAX_CONCURRENT)` gates all LLM calls to prevent overwhelming local models or hitting rate limits.

**Prompt architecture:** System prompts are stored as Jinja2 templates in a `prompts/` directory, enabling customization without code changes:

```
prompts/
├── pr_review.j2         # System prompt for PR code review
├── issue_respond.j2     # System prompt for issue Q&A
├── code_execution.j2    # System prompt for sandboxed code tasks
└── summarize.j2         # System prompt for PR/issue summarization
```

Each handler assembles the full message list:

1. **System prompt** (from template, with repo metadata injected)
2. **RAG context** (optional — retrieved code chunks relevant to the diff/question)
3. **Primary content** (diff for PR review, comment thread for issues)
4. **User instruction** (the triggering comment body, or implicit "review this PR")

#### 3.4.1 Default Prompt Templates

The bot ships with opinionated default prompts designed for actionable, low-noise output. These defaults can be overridden by placing custom `.j2` files in a mounted volume or (post-MVP) via a per-repo `.forgebot.yml` configuration.

**`pr_review.j2` — Code Review (default):**

```
You are a code reviewer for the repository "{{ repo_full_name }}".
You are reviewing a pull request titled "{{ pr_title }}".

Your review should follow these rules:

PRIORITIES (in order):
1. Bugs and correctness issues — logic errors, off-by-one, null/nil dereference, race conditions
2. Security vulnerabilities — injection, auth bypass, secret exposure, unsafe deserialization
3. Performance problems — O(n²) where O(n) is possible, unnecessary allocations, missing indexes
4. Error handling — uncaught exceptions, swallowed errors, missing edge cases
5. Suggestions — better approaches, simpler alternatives, readability improvements

DO NOT comment on:
- Code style, formatting, or naming conventions (that is the linter's job)
- Missing documentation or comments (unless a public API is genuinely unclear)
- Trivial changes (import reordering, whitespace, etc.)
- Things that are correct but you would have done differently

OUTPUT FORMAT:
- Use markdown with severity prefixes: 🔴 (critical), 🟡 (warning), 💡 (suggestion)
- Reference specific file paths and line numbers from the diff (e.g., `src/main.py:42`)
- Keep each comment to 2–3 sentences maximum
- If there are no significant issues, say: "Looks good — no major issues found." and optionally note 1–2 minor observations

{% if rag_context %}
CODEBASE CONTEXT (retrieved from repository — use this to understand the broader codebase):
{{ rag_context }}
{% endif %}
```

**`issue_respond.j2` — Issue Q&A (default):**

```
You are a helpful assistant for the repository "{{ repo_full_name }}".
You are responding to a conversation on issue #{{ issue_number }}: "{{ issue_title }}".

Your role is to help the developer with their question or problem. Be conversational
and direct. If you are unsure about something, say so rather than guessing.

When referencing code, use markdown code blocks with the appropriate language tag.
When referencing files in the repository, use the full relative path.

{% if rag_context %}
REPOSITORY CONTEXT (relevant code from the codebase):
{{ rag_context }}
{% endif %}

CONVERSATION HISTORY:
{% for comment in thread_comments %}
**{{ comment.user }}** ({{ comment.created_at }}):
{{ comment.body }}
{% endfor %}
```

**`code_execution.j2` — Sandbox Task (default):**

```
You are a code execution assistant. A user has requested code execution in
{{ language }} for the repository "{{ repo_full_name }}".

Your job is to explain the execution results. If the code succeeded, briefly
summarize what it did. If it failed, explain the error and suggest a fix.

Do not re-run the code or suggest modifications unless asked.
```

**Design rationale for defaults:**

- **Low temperature (0.2):** Produces consistent, deterministic reviews. Higher temperatures cause the model to "discover" phantom issues or vary wildly between re-reviews of the same diff.
- **Severity prefixes over structured JSON:** Markdown with emoji prefixes renders cleanly in Gitea/Forgejo comment UIs without post-processing. JSON-based structured output is fragile with local models that don't follow schemas precisely.
- **Explicit "do not" instructions:** Without these, LLMs reliably comment on style, formatting, and naming — generating noise that developers learn to ignore, undermining trust in the bot's real findings.
- **"Looks good" escape hatch:** Without this, the model fabricates issues to fill the response. Explicitly permitting a short positive review prevents false positives.
- **Thread context in issue prompts:** Including the full comment thread (up to a token budget) lets the bot participate in multi-turn conversations rather than responding to each comment in isolation.

### 3.5 Sandbox Orchestrator (Docker-in-Docker)

The sandbox system provides ephemeral, isolated Linux containers for code execution. Each sandbox is a fresh container that is created on demand, executes a task, and is destroyed — no state persists.

**Why DinD:** The bot itself runs in a Docker container. To spawn child containers for code execution, it needs access to a Docker daemon. There are two approaches:

| Approach | How | Tradeoffs |
|----------|-----|-----------|
| **DinD (Docker-in-Docker)** | Run a `dockerd` inside the bot container (privileged) | Full isolation, independent daemon, no host socket exposure |
| **DooD (Docker-outside-of-Docker)** | Mount host's `/var/run/docker.sock` | Simpler setup, but child containers share host daemon and can see host resources |

**MVP uses DinD** via the official `docker:dind` sidecar pattern in docker-compose, giving the bot its own isolated Docker daemon.

**Sandbox lifecycle:**

```
1. TRIGGER: Handler identifies a code execution request
      │
2. CREATE: Orchestrator pulls/creates a sandbox image
      │      (e.g., python:3.12-slim, node:22-slim, gcc:14)
      │
3. CONFIGURE: Mount task payload as a tmpfs volume
      │         Set resource limits: --memory=512m --cpus=1.0
      │         Set network: --network=none (default) or bridge
      │         Set timeout: 60 seconds max
      │
4. EXECUTE: Run the task inside the container
      │       Capture stdout, stderr, exit code
      │
5. COLLECT: Read output from the container
      │
6. DESTROY: Force-remove the container + volumes
      │
7. RESPOND: Format results and post as a comment
```

**Resource constraints (per sandbox):**

| Resource | Limit | Rationale |
|----------|-------|-----------|
| Memory | 512 MB | Prevents OOM on host |
| CPU | 1.0 cores | Fair scheduling |
| Disk (tmpfs) | 100 MB | No persistent writes |
| Network | `none` (default) | Prevent exfiltration; opt-in `bridge` for package installs |
| Execution time | 60 seconds | Kill after timeout |
| PIDs | 256 | Prevent fork bombs |

**Sandbox images:** The orchestrator maintains a configurable allowlist of base images. Users can request a language runtime in their comment (e.g., `/run python`, `/run node`), and the orchestrator maps it to a pre-pulled image.

**Default image registry:**

```python
SANDBOX_IMAGES = {
    "python": "python:3.12-slim",
    "node": "node:22-slim",
    "go": "golang:1.23-alpine",
    "rust": "rust:1.77-slim",
    "c": "gcc:14",
    "default": "python:3.12-slim",
}
```

**Image pre-pulling:** On startup, the orchestrator pre-pulls sandbox images so that the first `/run` command doesn't incur a multi-minute image download. This is controlled by two mechanisms:

1. **`SANDBOX_PREPULL_IMAGES`** (env var) — a comma-separated list of image keys from the registry to pre-pull on startup. Defaults to `python,node` (the most commonly requested runtimes). Set to `all` to pre-pull every registered image, or `none` to disable pre-pulling entirely.

2. **`sandbox-images.json`** (optional config file) — a JSON file mounted at `/app/config/sandbox-images.json` that overrides and extends the default image registry. This allows operators to add custom images (e.g., a project-specific image with pre-installed dependencies) without modifying the bot's source code.

```json
{
  "python": "python:3.12-slim",
  "python3.11": "python:3.11-slim",
  "node": "node:22-slim",
  "go": "golang:1.23-alpine",
  "rust": "rust:1.77-slim",
  "c": "gcc:14",
  "dotnet": "mcr.microsoft.com/dotnet/sdk:9.0",
  "custom-ml": "registry.example.com/ml-sandbox:latest",
  "default": "python:3.12-slim"
}
```

**Pre-pull startup behavior:**

```
1. STARTUP: Bot connects to DinD daemon
      │
2. LOAD: Read sandbox-images.json (if mounted), merge with defaults
      │
3. RESOLVE: Determine which images to pre-pull from SANDBOX_PREPULL_IMAGES
      │
4. PULL: Pull images in parallel (max 3 concurrent pulls)
      │      Log progress: "Pre-pulling python:3.12-slim... done (234 MB)"
      │      Non-blocking — bot accepts webhooks while pulling
      │
5. READY: Log summary: "Sandbox ready: 4/6 images pre-pulled"
```

If an image is requested via `/run` but hasn't been pre-pulled, the orchestrator pulls it on demand. The first execution for that image will be slower, but subsequent runs use the cached image. Images not in the registry are rejected with an error comment listing available runtimes.

### 3.6 RAG Context Pipeline (Optional)

The RAG pipeline provides repository-aware context to the LLM, improving response quality for large codebases. It is entirely optional — when disabled, the bot still functions using only the PR diff or issue context.

**Pipeline stages:**

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ 1. Ingest   │────▶│ 2. Chunk     │────▶│ 3. Embed     │────▶│ 4. Store     │
│             │     │              │     │              │     │              │
│ git clone   │     │ AST-aware    │     │ Sentence     │     │ ChromaDB     │
│ or API tree │     │ splitting    │     │ Transformers │     │ (persistent) │
│ walk        │     │              │     │ or LLM embed │     │              │
└─────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
                                                                     │
                                                                     ▼
                                                              ┌──────────────┐
                                                              │ 5. Retrieve  │
                                                              │              │
                                                              │ Query with   │
                                                              │ diff/comment │
                                                              │ as input     │
                                                              └──────────────┘
```

#### 3.6.1 Ingestion

Triggered on first interaction with a repository, or on-demand via a `/index` bot command. The ingester clones the repository (shallow clone, `--depth=1`) into a temporary directory and walks the file tree.

**File filtering:** Skip binary files, vendored dependencies (`vendor/`, `node_modules/`), generated files (`*.min.js`, `*.pb.go`), and files exceeding a size threshold (default 100 KB). A `.forgebotignore` file in the repo root can specify additional exclusions (gitignore syntax).

#### 3.6.2 Chunking Strategy

Naive line-count splitting destroys semantic meaning. The chunker uses a **language-aware, AST-based** approach with fallback:

**For supported languages (Python, JS/TS, Go, Rust, C/C++, Java):**

- Parse using `tree-sitter` to extract AST nodes
- Chunk at function/method/class boundaries
- Each chunk contains one complete semantic unit (a function, a class, a struct + its methods)
- Include the file path and import context as chunk metadata
- If a single function exceeds the max chunk size (default 2000 tokens), split at logical sub-boundaries (nested blocks, sequential statements)

**For unsupported languages / plain text:**

- Split on double-newline (paragraph) boundaries
- Use sliding window with overlap (chunk size: 1500 tokens, overlap: 200 tokens)
- Preserve file path as metadata

**Chunk metadata schema:**

```python
@dataclass
class CodeChunk:
    content: str              # The actual code/text
    file_path: str            # Relative path in repo
    repo_full_name: str       # "owner/repo"
    ref: str                  # Git ref (branch/commit SHA)
    language: str             # Detected language
    symbol_name: str | None   # Function/class name if AST-parsed
    start_line: int           # Starting line number
    end_line: int             # Ending line number
    token_count: int          # Approximate token count
```

#### 3.6.3 Embedding

Embeddings are computed using one of two backends, configurable via environment:

| Backend | Env Config | Use Case |
|---------|-----------|----------|
| **Sentence Transformers (local)** | `RAG_EMBED_MODEL=all-MiniLM-L6-v2` | Self-hosted, no external calls, fast |
| **OpenAI-compatible embeddings** | `RAG_EMBED_URL` + `RAG_EMBED_MODEL` | Higher quality, uses same LLM infra |

The embedding client shares the same OpenAI-compatible pattern as the LLM client — point `RAG_EMBED_URL` at any `/v1/embeddings` endpoint.

#### 3.6.4 Vector Store

**ChromaDB** (persistent, file-backed) is the MVP vector store. It runs in-process (no separate server), stores data in a mounted Docker volume, and supports metadata filtering.

**Collection structure:** One collection per repository, named `{owner}__{repo}` (double underscore as separator).

**Retrieval at query time:**

1. For PR reviews: extract changed file paths from the diff, query the vector store for chunks from related files (not just changed files — also callers, importers, tests)
2. For issue comments: embed the comment text and retrieve top-K most similar chunks
3. Filter by `ref` metadata to match the PR's base branch
4. Inject top-K results (default K=10, max ~4000 tokens) into the LLM context as a "Relevant codebase context" section

**Re-indexing:** When a PR is merged (detected via `pull_request` → `closed` + `merged: true` event), queue a background re-index of the changed files against the target branch.

#### 3.6.5 RAG Configuration

| Env Var | Purpose | Default |
|---------|---------|---------|
| `RAG_ENABLED` | Enable/disable entire RAG pipeline | `false` |
| `RAG_EMBED_MODEL` | Embedding model name | `all-MiniLM-L6-v2` |
| `RAG_EMBED_URL` | OpenAI-compatible embedding endpoint | (empty = local sentence-transformers) |
| `RAG_CHUNK_SIZE` | Max tokens per chunk | `1500` |
| `RAG_CHUNK_OVERLAP` | Overlap tokens between chunks | `200` |
| `RAG_TOP_K` | Number of chunks to retrieve | `10` |
| `RAG_MAX_CONTEXT_TOKENS` | Max total tokens from RAG context | `4000` |
| `RAG_STORE_PATH` | ChromaDB persistence directory | `/data/chromadb` |

---

## 4. Docker & Deployment Architecture

### 4.1 Container Structure

```
┌──────────────────────────────────────────────────────────────┐
│                    docker-compose stack                        │
│                                                               │
│  ┌─────────────────────────────┐  ┌────────────────────────┐ │
│  │       forge-bot             │  │    docker-dind          │ │
│  │                             │  │                         │ │
│  │  Python 3.12                │  │  docker:27-dind         │ │
│  │  FastAPI + Uvicorn          │  │                         │ │
│  │                             │  │  Provides Docker daemon │ │
│  │  Connects to DinD daemon    │──│  for sandbox containers │ │
│  │  via DOCKER_HOST=           │  │                         │ │
│  │  tcp://dind:2376            │  │  TLS certs shared via   │ │
│  │                             │  │  volume                 │ │
│  │  Ports: 8080 (webhook)     │  │                         │ │
│  └─────────────────────────────┘  └────────────────────────┘ │
│                                                               │
│  Volumes:                                                     │
│    forge-data    → /data          (ChromaDB, state)           │
│    dind-certs    → /certs         (TLS certs for DinD)        │
│    dind-storage  → /var/lib/docker (DinD image/container store│
│                                                               │
│  ┌─────────────────────────────┐  (optional)                 │
│  │       ollama                │                              │
│  │                             │                              │
│  │  Local LLM inference        │                              │
│  │  Ports: 11434              │                              │
│  └─────────────────────────────┘                              │
└──────────────────────────────────────────────────────────────┘
```

### 4.2 docker-compose.yml

```yaml
services:
  forge-bot:
    build: .
    container_name: forge-bot
    restart: unless-stopped
    ports:
      - "${WEBHOOK_PORT:-8080}:8080"
    environment:
      # Gitea/Forgejo connection
      - FORGE_INSTANCE_URL=${FORGE_INSTANCE_URL}
      - FORGE_API_TOKEN=${FORGE_API_TOKEN}
      - FORGE_WEBHOOK_SECRET=${FORGE_WEBHOOK_SECRET}
      # LLM connection
      - LLM_BASE_URL=${LLM_BASE_URL:-https://api.openai.com/v1}
      - LLM_API_KEY=${LLM_API_KEY}
      - LLM_MODEL=${LLM_MODEL:-gpt-4o}
      # Docker-in-Docker
      - DOCKER_HOST=tcp://dind:2376
      - DOCKER_TLS_VERIFY=1
      - DOCKER_CERT_PATH=/certs/client
      # RAG (optional)
      - RAG_ENABLED=${RAG_ENABLED:-false}
      # Sandbox
      - SANDBOX_PREPULL_IMAGES=${SANDBOX_PREPULL_IMAGES:-python,node}
    volumes:
      - forge-data:/data
      - dind-certs:/certs/client:ro
      # Optional: custom sandbox image registry
      # - ./sandbox-images.json:/app/config/sandbox-images.json:ro
    depends_on:
      dind:
        condition: service_healthy

  dind:
    image: docker:27-dind
    container_name: forge-bot-dind
    privileged: true
    restart: unless-stopped
    environment:
      - DOCKER_TLS_CERTDIR=/certs
    volumes:
      - dind-certs:/certs
      - dind-storage:/var/lib/docker
    healthcheck:
      test: ["CMD", "docker", "info"]
      interval: 10s
      timeout: 5s
      retries: 3

  # Optional: local LLM
  ollama:
    image: ollama/ollama:latest
    container_name: forge-bot-ollama
    profiles: ["local-llm"]
    restart: unless-stopped
    ports:
      - "11434:11434"
    volumes:
      - ollama-models:/root/.ollama

volumes:
  forge-data:
  dind-certs:
  dind-storage:
  ollama-models:
```

### 4.3 Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    docker.io \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY forge_bot/ ./forge_bot/
COPY prompts/ ./prompts/

# Non-root user (still needs docker group for DinD client)
RUN groupadd -g 999 docker-host && \
    useradd -m -u 1000 -G docker-host forgebot
USER forgebot

EXPOSE 8080

CMD ["uvicorn", "forge_bot.server:app", "--host", "0.0.0.0", "--port", "8080"]
```

### 4.4 Network & Security

- **Webhook ingress:** The bot listens on a single port (8080). In production, place behind a reverse proxy (Caddy, nginx, Traefik) for TLS termination.
- **DinD communication:** TLS-encrypted TCP between bot and DinD daemon via shared certificate volume.
- **Sandbox networking:** Sandbox containers default to `--network=none` (no network access). An explicit `/run --net` flag in the comment enables bridge networking for tasks requiring package installation.
- **No host Docker socket exposure:** The bot never mounts `/var/run/docker.sock` from the host — all container operations go through the isolated DinD daemon.

---

## 5. Project Structure

```
forge-bot/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
├── README.md
├── sandbox-images.json         # Default sandbox image registry (override via mount)
│
├── forge_bot/
│   ├── __init__.py
│   ├── server.py               # FastAPI app, webhook endpoint
│   ├── config.py               # Pydantic Settings (all env vars)
│   ├── router.py               # Event type → handler dispatch
│   ├── models.py               # Pydantic models for webhook payloads
│   │
│   ├── handlers/
│   │   ├── __init__.py
│   │   ├── base.py             # Abstract base handler
│   │   ├── pull_request.py     # PR review handler
│   │   ├── issue_comment.py    # Issue/PR comment handler
│   │   └── issue_assign.py     # Assignment event handler
│   │
│   ├── clients/
│   │   ├── __init__.py
│   │   ├── forge.py            # Gitea/Forgejo API client
│   │   └── llm.py              # OpenAI-compatible LLM client
│   │
│   ├── sandbox/
│   │   ├── __init__.py
│   │   ├── orchestrator.py     # Container lifecycle management
│   │   ├── images.py           # Sandbox image registry
│   │   └── parser.py           # Parse /run commands from comments
│   │
│   ├── rag/                    # Optional — entire directory ignored if RAG_ENABLED=false
│   │   ├── __init__.py
│   │   ├── pipeline.py         # Orchestrates ingest → chunk → embed → store
│   │   ├── ingester.py         # Git clone + file tree walk
│   │   ├── chunker.py          # AST-aware + fallback chunking
│   │   ├── embedder.py         # Embedding client (local or API)
│   │   ├── store.py            # ChromaDB wrapper
│   │   └── retriever.py        # Query interface for handlers
│   │
│   └── utils/
│       ├── __init__.py
│       ├── dedup.py            # Delivery ID deduplication (LRU)
│       ├── diff.py             # Unified diff parser
│       └── formatting.py       # Markdown formatting helpers
│
├── prompts/
│   ├── pr_review.j2
│   ├── issue_respond.j2
│   ├── code_execution.j2
│   └── summarize.j2
│
└── tests/
    ├── conftest.py
    ├── test_server.py
    ├── test_router.py
    ├── test_handlers/
    ├── test_clients/
    ├── test_sandbox/
    └── test_rag/
```

---

## 6. Configuration Reference

All configuration is via environment variables, validated at startup by Pydantic Settings.

### 6.1 Required

| Variable | Description | Example |
|----------|-------------|---------|
| `FORGE_INSTANCE_URL` | Base URL of Gitea/Forgejo instance | `https://gitea.example.com` |
| `FORGE_API_TOKEN` | Bot account API token | `abc123...` |
| `FORGE_WEBHOOK_SECRET` | Shared HMAC secret for webhook verification | `super-secret-key` |
| `LLM_API_KEY` | API key for the LLM endpoint | `sk-...` or `ollama` |

### 6.2 Optional

| Variable | Default | Description |
|----------|---------|-------------|
| `WEBHOOK_PORT` | `8080` | Port for the webhook server |
| `LOG_LEVEL` | `INFO` | Logging level |
| `LLM_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible base URL |
| `LLM_MODEL` | `gpt-4o` | Model identifier |
| `LLM_TEMPERATURE` | `0.2` | Sampling temperature |
| `LLM_MAX_TOKENS` | `4096` | Max response tokens |
| `LLM_TIMEOUT` | `120` | Request timeout (seconds) |
| `LLM_MAX_CONCURRENT` | `3` | Max parallel LLM requests |
| `SANDBOX_ENABLED` | `true` | Enable code execution sandboxes |
| `SANDBOX_TIMEOUT` | `60` | Max sandbox execution time (seconds) |
| `SANDBOX_MEMORY` | `512m` | Memory limit per sandbox |
| `SANDBOX_CPUS` | `1.0` | CPU limit per sandbox |
| `SANDBOX_PREPULL_IMAGES` | `python,node` | Images to pre-pull on startup (`all`, `none`, or comma-separated keys) |
| `SANDBOX_IMAGES_FILE` | (empty) | Path to custom `sandbox-images.json` override file |
| `RAG_ENABLED` | `false` | Enable RAG context pipeline |
| `RAG_EMBED_MODEL` | `all-MiniLM-L6-v2` | Embedding model |
| `RAG_EMBED_URL` | (empty) | External embedding API URL |
| `RAG_TOP_K` | `10` | Chunks to retrieve |
| `RAG_STORE_PATH` | `/data/chromadb` | Vector store directory |
| `BOT_COMMAND_PREFIX` | `/` | Prefix for bot commands (e.g., `/run`, `/index`) |

---

## 7. Request Flow: PR Code Review

This is the primary flow, traced end-to-end:

```
Developer opens PR and assigns bot as reviewer
        │
        ▼
Gitea fires pull_request webhook (action: "opened")
        │
        ▼
┌─ Webhook Server ────────────────────────────────┐
│  1. Read raw body                                │
│  2. Verify HMAC-SHA256 signature                 │
│  3. Parse X-Gitea-Delivery for dedup check       │
│  4. Return HTTP 200                              │
│  5. Dispatch to background task                  │
└──────────────────────────┬───────────────────────┘
                           ▼
┌─ Event Router ──────────────────────────────────┐
│  1. Read X-Gitea-Event: "pull_request"           │
│  2. Read action: "opened"                        │
│  3. Check: is bot in assignees or reviewers?     │
│  4. Yes → dispatch to PullRequestHandler         │
└──────────────────────────┬───────────────────────┘
                           ▼
┌─ PullRequestHandler ────────────────────────────┐
│  1. Fetch PR details:                            │
│     GET /repos/{owner}/{repo}/pulls/{index}      │
│                                                  │
│  2. Fetch raw diff:                              │
│     GET /repos/{owner}/{repo}/pulls/{index}.diff │
│                                                  │
│  3. Fetch changed file list:                     │
│     GET /repos/{owner}/{repo}/pulls/{index}/files│
│                                                  │
│  4. [If RAG enabled] Retrieve relevant context:  │
│     - Extract file paths from diff               │
│     - Query vector store for related code         │
│     - Assemble context window                    │
│                                                  │
│  5. Build LLM prompt:                            │
│     - System: pr_review.j2 template              │
│     - Context: RAG chunks (if enabled)           │
│     - User: PR title + description + diff        │
│                                                  │
│  6. Call LLM:                                    │
│     POST {LLM_BASE_URL}/chat/completions         │
│     (gated by asyncio.Semaphore)                 │
│                                                  │
│  7. Parse LLM response                           │
│                                                  │
│  8. Post review comment:                         │
│     POST /repos/{owner}/{repo}/issues/{index}/   │
│          comments                                │
│     Body: formatted review in Markdown           │
└──────────────────────────────────────────────────┘
```

---

## 8. Request Flow: Sandboxed Code Execution

```
Developer comments on issue/PR:
  "@forge-bot /run python
  ```python
  print('hello world')
  ```"
        │
        ▼
[Webhook → Router → IssueCommentHandler]
        │
        ▼
┌─ Command Parser ────────────────────────────────┐
│  1. Detect /run command in comment body           │
│  2. Extract language: "python"                    │
│  3. Extract code block content                    │
│  4. Validate against allowlist                    │
└──────────────────────────┬───────────────────────┘
                           ▼
┌─ Sandbox Orchestrator ──────────────────────────┐
│  1. Select image: python:3.12-slim               │
│  2. Create container via DinD daemon:             │
│     - --memory=512m --cpus=1.0                   │
│     - --pids-limit=256                           │
│     - --network=none                             │
│     - --read-only (tmpfs for /tmp)               │
│     - Timeout: 60s                               │
│                                                  │
│  3. Write code to tmpfs mount                     │
│  4. Execute: python /tmp/code.py                  │
│  5. Capture stdout + stderr + exit code           │
│  6. Force-remove container                        │
└──────────────────────────┬───────────────────────┘
                           ▼
┌─ Response Formatter ────────────────────────────┐
│  Format as Markdown:                             │
│                                                  │
│  **Execution Result** ✅ (exit code 0)           │
│  ```                                             │
│  hello world                                     │
│  ```                                             │
│  *Ran in python:3.12-slim | 0.3s | 2.1 MB RAM*  │
└──────────────────────────┬───────────────────────┘
                           ▼
        Post comment via Forge API Client
```

---

## 9. Data Models

### 9.1 Webhook Payload Models (Pydantic)

```python
class WebhookUser(BaseModel):
    id: int
    login: str

class WebhookRepository(BaseModel):
    full_name: str        # "owner/repo"
    clone_url: str
    default_branch: str

class WebhookPullRequest(BaseModel):
    id: int
    number: int
    title: str
    body: str | None
    state: str
    user: WebhookUser
    head: BranchRef       # {ref, sha, repo}
    base: BranchRef
    assignees: list[WebhookUser]
    requested_reviewers: list[WebhookUser]

class PullRequestEvent(BaseModel):
    action: str           # opened, synchronized, closed, assigned, ...
    number: int
    pull_request: WebhookPullRequest
    repository: WebhookRepository
    sender: WebhookUser

class WebhookComment(BaseModel):
    id: int
    body: str
    user: WebhookUser

class WebhookIssue(BaseModel):
    number: int
    title: str
    body: str | None
    pull_request: dict | None   # non-null means this is a PR
    assignees: list[WebhookUser]

class IssueCommentEvent(BaseModel):
    action: str           # created, edited, deleted
    comment: WebhookComment
    issue: WebhookIssue
    is_pull: bool         # true if comment is on a PR
    repository: WebhookRepository
    sender: WebhookUser
```

---

## 10. Bot Command Interface

Users interact with the bot through **@mentions** and **slash commands** in issue and PR comments.

### 10.1 @Mention Interaction

The primary way to interact with the bot is by @mentioning its username. The bot treats everything after the @mention as a natural language query:

```markdown
@forge-bot What does the `processQueue` function do?
@forge-bot Can you explain the error handling in this PR?
@forge-bot Is there a race condition between these two goroutines?
```

**@mention behavior by context:**

| Context | Bot has access to | Behavior |
|---------|-------------------|----------|
| Issue comment | Issue title, body, full comment thread, RAG context (if enabled) | Conversational Q&A about the issue topic |
| PR comment | All of the above + PR diff, changed files, branch info | Q&A with full code change awareness |
| PR comment (bot is also assigned reviewer) | All of the above | May reference its own prior review |

**Self-loop prevention:** The bot ignores any webhook event where the sender is the bot itself. This is critical — without it, the bot's own comment triggers a new `issue_comment` webhook, which the bot processes as a new mention, posting another comment, ad infinitum.

**Mention detection pattern:** The router uses a regex that handles common edge cases:

```python
import re

def is_mentioned(body: str, bot_username: str) -> bool:
    """Check if the bot is @mentioned in a comment body."""
    pattern = rf'(?:^|\s)@{re.escape(bot_username)}(?:\s|$|[.,;:!?\)])'
    return bool(re.search(pattern, body, re.MULTILINE))
```

This matches `@forge-bot` at the start of a line, mid-sentence, and followed by punctuation, but avoids false positives from email addresses or other @-prefixed strings.

### 10.2 Slash Commands

Slash commands trigger specific bot actions. They can be combined with @mentions or used standalone (if the bot is assigned to the issue/PR):

| Command | Description | Context |
|---------|-------------|---------|
| `@bot` (plain mention) | Ask a question or request help | Issues & PRs |
| `/review` | Request a code review (or re-review after new commits) | PRs only |
| `/run <lang>` | Execute a code block in a sandbox | Issues & PRs |
| `/run --net <lang>` | Execute with network access (for package installs) | Issues & PRs |
| `/index` | Trigger RAG re-indexing of this repo | Issues & PRs |
| `/help` | Show available commands and bot status | Issues & PRs |

**Command parsing:** Commands are detected by `sandbox/parser.py`, which extracts the command name, arguments, and any associated code blocks (fenced with triple backticks) from the comment body. Multiple commands in a single comment are processed sequentially.

**Example interaction combining @mention and slash command:**

```markdown
@forge-bot /run python
```python
import json
data = {"key": "value"}
print(json.dumps(data, indent=2))
```​
```

The parser extracts: command=`run`, language=`python`, code block content, and the implicit @mention triggers the bot to respond with execution results.

---

## 11. Error Handling & Resilience

| Failure Mode | Handling Strategy |
|-------------|-------------------|
| HMAC verification fails | Return 403, log warning, do not process |
| Duplicate delivery UUID | Return 200, skip processing |
| Gitea API returns 4xx/5xx | Retry with exponential backoff (3 attempts, 2/4/8s delays) |
| LLM endpoint timeout | Post a comment: "⚠️ I'm having trouble reaching my AI backend. I'll retry shortly." Retry once. |
| LLM returns empty/malformed response | Post a comment acknowledging the issue, log for debugging |
| Sandbox container timeout (>60s) | Kill container, post result: "⏱️ Execution timed out after 60 seconds." |
| Sandbox OOM kill | Detect via exit code 137, post: "💥 Out of memory (limit: 512 MB)." |
| DinD daemon unavailable | Disable sandbox features gracefully, respond to `/run` with explanation |
| RAG vector store corrupted | Log error, fall back to non-RAG mode for that request |
| Webhook payload missing expected fields | Log warning, skip event gracefully |

---

## 12. Security Considerations

| Concern | Mitigation |
|---------|-----------|
| **Webhook spoofing** | HMAC-SHA256 verification on every request; constant-time comparison |
| **Arbitrary code execution** | Sandboxes are ephemeral, resource-limited, network-isolated, and destroyed after use |
| **Prompt injection via PR/issue content** | System prompts instruct the LLM to treat all user content as untrusted data to review/discuss, not as instructions to follow |
| **Secret leakage** | API tokens and secrets are env vars only, never logged or included in comments. Sandbox containers do not receive any host environment variables |
| **DinD privilege escalation** | DinD runs in its own container with a separate Docker daemon. Sandbox containers cannot access the bot's filesystem or the DinD daemon's control socket |
| **Resource exhaustion (DoS)** | Concurrency semaphore on LLM calls, resource limits on sandboxes, dedup prevents reprocessing |
| **Repository data exfiltration** | Sandbox containers default to `--network=none`. RAG data is stored in a Docker volume accessible only to the bot container |

---

## 13. Dependencies

### 13.1 Python Packages (requirements.txt)

```
# Core
fastapi>=0.115.0
uvicorn[standard]>=0.34.0
httpx>=0.28.0
pydantic>=2.10.0
pydantic-settings>=2.7.0

# LLM
openai>=1.60.0

# Templating
jinja2>=3.1.0

# Docker (sandbox orchestration)
docker>=7.0.0

# RAG (optional - install with pip install -r requirements-rag.txt)
# chromadb>=0.5.0
# sentence-transformers>=3.0.0
# tree-sitter>=0.23.0
# tree-sitter-python>=0.23.0
# tree-sitter-javascript>=0.23.0

# Testing
pytest>=8.0.0
pytest-asyncio>=0.24.0
pytest-httpx>=0.34.0
```

### 13.2 External Services

| Service | Required? | Purpose |
|---------|-----------|---------|
| Gitea or Forgejo instance | Yes | Source of webhooks, target for API calls |
| OpenAI-compatible LLM endpoint | Yes | AI inference (OpenAI, Ollama, vLLM, etc.) |
| Docker daemon (DinD) | Yes (for sandbox) | Ephemeral code execution containers |
| Redis | No (future) | Task queue backend for production scale |

---

## 14. Testing Strategy

| Layer | Approach | Tools |
|-------|----------|-------|
| **Unit tests** | Test routing logic, command parsing, diff parsing, chunking, formatting in isolation | pytest |
| **Integration tests** | Test ForgeClient against a real Gitea instance (docker-compose test profile) | pytest + testcontainers |
| **Webhook simulation** | Send crafted webhook payloads to the running server | pytest-httpx, httpx |
| **Sandbox tests** | Verify container creation, execution, cleanup, resource limits | pytest + docker SDK |
| **End-to-end** | Open a real PR on a test Gitea instance, verify bot comments | Shell script / CI pipeline |

---

## 15. Future Roadmap (Post-MVP)

| Phase | Feature | Description |
|-------|---------|-------------|
| **v0.2** | Redis task queue | Replace in-memory asyncio queue with ARQ + Redis for persistence and crash recovery |
| **v0.2** | Multi-repo webhook registration | API endpoint to register/unregister the bot on repos programmatically |
| **v0.3** | PR approval workflows | Bot can approve PRs based on configurable criteria (all checks pass, review score threshold) |
| **v0.3** | Conversation memory | Maintain context across multiple comments in the same issue/PR thread |
| **v0.4** | Web dashboard | Simple status page: recent reviews, processing queue, error rates |
| **v0.4** | Per-repo configuration | `.forgebot.yml` file in repo root for custom prompts, review rules, sandbox settings |
| **v0.5** | Multi-instance support | Single bot serving multiple Gitea/Forgejo instances |
| **v0.5** | MCP tool integration | Expose bot capabilities as MCP tools for broader agent ecosystems |
