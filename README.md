# forge-bot

AI-powered bot for Gitea and Forgejo. Receives webhook events, reviews PRs, and answers questions on issues — driven by an LLM with tool access to your repository via per-event Docker containers.

## Getting Started

### Prerequisites

- **Gitea or Forgejo instance** with a bot user account
- **Docker** running on the host (for workspace containers)
- **LLM endpoint** — OpenAI, or any OpenAI-compatible API (Ollama, llama.cpp, vLLM, etc.)

### 1. Create a bot account on Gitea/Forgejo

1. Create a new user account (e.g. `forge-bot`)
2. Generate an API token: **Settings > Applications > Generate Token** (select all scopes)
3. Note the token — you'll need it for `FORGE_API_TOKEN`

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` with your values:

```bash
FORGE_INSTANCE_URL=https://your-gitea.example.com
FORGE_API_TOKEN=<bot account API token>
FORGE_WEBHOOK_SECRET=<any random secret string>
LLM_API_KEY=<your LLM API key>

# Optional — defaults shown:
# LLM_BASE_URL=https://api.openai.com/v1
# LLM_MODEL=gpt-4o
# FORGE_PROVIDER=gitea   # or "forgejo"
```

### 3. Build the workspace image

The bot spins up a Docker container per event to give the LLM access to `git`, `bash`, `python`, and the cloned repo:

```bash
docker build -t forge-bot-workspace:latest -f Dockerfile.workspace .
```

### 4. Start the bot

**With Docker Compose (recommended):**

```bash
docker compose up -d
```

**Or run directly:**

```bash
pip install -r requirements.txt
uvicorn forge_bot.server:app --host 0.0.0.0 --port 8080
```

### 5. Configure the webhook

In your Gitea/Forgejo repo (or org-wide), add a webhook:

| Setting | Value |
|---------|-------|
| **Target URL** | `http://<bot-host>:8080/webhook` |
| **Secret** | Same as `FORGE_WEBHOOK_SECRET` |
| **Content type** | `application/json` |
| **Events** | Pull Request, Issue Comment |

### 6. Test it

Create an issue and comment:

```
@forge-bot what does this repo do?
```

The bot will post a status comment (showing progress), gather context from the repo using tools, and then post its response.

## How It Works

```
Gitea/Forgejo webhook
       |
       v
  FastAPI /webhook (HMAC verify, return 200 immediately)
       |
       v  (background task)
  Event Router --> Handler
       |
       |-- 1. Post status comment ("Working...")
       |-- 2. Create workspace container (clone repo)
       |-- 3. LLM tool-calling loop:
       |       - search_api: discover available API endpoints
       |       - api_call:   call any Gitea/Forgejo API endpoint
       |       - exec:       run commands in the workspace (git, grep, cat, python, etc.)
       |       - todo:       update visible progress checklist
       |-- 4. Verification checks (hallucination, relevance, progress)
       |-- 5. Post response comment
       |-- 6. Destroy container
```

The LLM drives its own context gathering — instead of pre-fetching everything, it uses tools to explore the repo, run commands, and call APIs as needed.

## Configuration

All config via environment variables (or `.env` file).

**Required:**

| Variable | Description |
|----------|-------------|
| `FORGE_INSTANCE_URL` | Gitea/Forgejo base URL |
| `FORGE_API_TOKEN` | Bot account API token |
| `FORGE_WEBHOOK_SECRET` | HMAC secret for webhook verification |
| `LLM_API_KEY` | API key for LLM endpoint |

**Key optional:**

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_BASE_URL` | `https://api.openai.com/v1` | Any OpenAI-compatible endpoint |
| `LLM_MODEL` | `gpt-4o` | Model name |
| `LLM_CONTEXT_WINDOW` | `8192` | Match your model's context window |
| `FORGE_PROVIDER` | `gitea` | `gitea` or `forgejo` |
| `CONTAINER_WORKSPACE_IMAGE` | `forge-bot-workspace:latest` | Docker image for workspaces |
| `CONTAINER_NETWORK_ENABLED` | `true` | Allow network in workspace containers |
| `RAG_ENABLED` | `false` | Enable codebase-aware vector retrieval |

## Trigger Conditions

| Event | When bot acts |
|-------|--------------|
| PR opened/updated | Posts an LLM-generated code review |
| Comment on issue/PR | Responds when @mentioned |

## Local Models (Ollama)

forge-bot works with any OpenAI-compatible endpoint, including [Ollama](https://ollama.com) for local/self-hosted models.

### Setup

```bash
# .env
LLM_BASE_URL=http://ollama:11434/v1
LLM_MODEL=gemma3:12b
LLM_API_KEY=ollama              # Ollama ignores this but the field is required
LLM_CONTEXT_WINDOW=8192         # Match your model's context window
```

### Choosing a Model

| Size | Examples | VRAM | Best For | Limitations |
|------|----------|------|----------|-------------|
| 1-3B | gemma3:1b, phi-4-mini | 2-4 GB | Simple Q&A, fast responses | Poor instruction following, frequent hallucination, limited context |
| 7-8B | mistral:7b, llama3.1:8b, gemma3:4b | 6-8 GB | General use, good balance | May struggle with complex multi-file context |
| 12-14B | gemma3:12b, qwen2.5:14b | 10-16 GB | Code review, detailed answers | Needs adequate VRAM, slower |
| 27B+ | gemma3:27b, llama3.1:70b, qwen2.5:72b | 20-48 GB | Best local quality | High hardware requirements |

### Recommended Settings

| Model Size | `LLM_CONTEXT_WINDOW` | `LLM_MAX_TOKENS` | `LLM_TEMPERATURE` | Notes |
|------------|----------------------|-------------------|--------------------|-------|
| 1-3B | 2048 | 1024 | 0.1 | Lower temperature reduces hallucination |
| 7-8B | 4096 | 2048 | 0.15 | Good starting point for most setups |
| 12-14B | 8192 | 4096 | 0.2 | Default settings work well |
| 27B+ | 32768 | 4096 | 0.2 | Can handle larger context comfortably |
| Cloud API | 32768 | 4096 | 0.2 | GPT-4o, Claude, etc. |

### Tips for Small Models

- **Set `LLM_CONTEXT_WINDOW` accurately** — the bot adjusts how much context it sends based on this value
- **Lower the temperature** — `0.1` reduces creative hallucination on smaller models
- **Reduce `LLM_MAX_TOKENS`** — smaller models produce better output with lower limits
- **Use Ollama's `num_ctx` parameter** — ensure Ollama allocates enough context: `ollama run gemma3:12b --num_ctx 8192`

## Project Structure

```
forge_bot/
  server.py              # FastAPI app, webhook endpoint, HMAC verification
  config.py              # pydantic-settings, all env vars
  router.py              # Event type -> handler dispatch, self-loop guard
  models.py              # Pydantic models for webhook payloads

  api/                   # YAML-driven API client (replaces hardcoded methods)
    definitions/
      gitea.yaml         # ~20 Gitea API endpoint definitions
      forgejo.yaml       # ~20 Forgejo API endpoint definitions
    schema.py            # EndpointParam, EndpointDef, ApiDefinitionFile
    loader.py            # YAML loader with caching
    client.py            # GenericForgeClient: call(name, **params), search(keyword)

  container/             # Per-event persistent containers
    manager.py           # ContainerManager: create, exec, destroy

  tools/                 # LLM tool implementations
    search_api.py        # Search API definitions by keyword
    api_call.py          # Execute any YAML-defined endpoint
    exec_tool.py         # Run commands in workspace container
    todo.py              # Manage visible todo list in status comment
    registry.py          # Tool registry, OpenAI schema generation

  status/                # Real-time observability via Gitea comments
    manager.py           # Two-comment system (status + response)
    formatter.py         # Markdown rendering for todos, tool calls

  handlers/              # Webhook event handlers
    base.py              # BaseHandler with shared dependencies
    issue_comment.py     # @mention replies with tool-calling loop
    pull_request.py      # PR review on opened/synchronized
    verification.py      # Relevance, hallucination, progress checks

  prompts/               # Jinja2 prompt templates
    issue_respond.j2
    pr_review.j2

  clients/
    llm.py               # AsyncOpenAI wrapper with tool calling support

  rag/                   # Optional: vector-based code retrieval
  utils/                 # Dedup, diff parsing, formatting
```

## Development

```bash
# Install
pip install -r requirements.txt -r requirements-dev.txt

# Test
pytest

# Lint
ruff check forge_bot/
ruff format forge_bot/
```

## License

MIT
