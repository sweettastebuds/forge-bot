# forge-bot

AI-powered bot for Gitea and Forgejo. Receives webhook events, reviews PRs, answers questions on issues, and executes code in sandboxed containers — all via any OpenAI-compatible LLM.

## Quick Start

```bash
cp .env.example .env
# Edit .env with your Gitea/Forgejo URL, API token, webhook secret, and LLM config
docker compose up -d
```

Then add a webhook on your Gitea/Forgejo repo:
- **URL:** `https://your-bot-host:8080/webhook`
- **Secret:** same as `FORGE_WEBHOOK_SECRET` in `.env`
- **Events:** Pull Request, Issue Comment

## How It Works

1. Bot runs as a regular Gitea/Forgejo user account with an API token
2. Receives `pull_request` and `issue_comment` webhooks
3. Returns HTTP 200 immediately (Gitea has a 5s delivery timeout)
4. Processes events in background: fetches diffs/context, calls LLM, posts comments
5. Optionally runs code in ephemeral Docker containers via DinD

## Trigger Conditions

| Event | When bot acts |
|-------|--------------|
| PR opened/updated | Bot is in assignees or requested reviewers |
| Comment on issue/PR | Bot is @mentioned or assigned to the issue |
| `/run python` in comment | Executes code block in sandbox, posts result |
| `/review` in PR comment | Re-runs code review on current diff |
| Issue assigned to bot | Posts greeting/triage response |

## Architecture

```
Gitea/Forgejo → webhook → FastAPI (HMAC verify, return 200)
                              ↓ background task
                         Event Router → Handler
                              ↓
                    ┌─────────┼──────────┐
                    ↓         ↓          ↓
              Forge API    LLM API    Sandbox
              (httpx)     (openai)    (DinD)
```

**Optional RAG pipeline:** Ingests repo code → AST-aware chunking → embeddings → ChromaDB → retrieved context injected into LLM prompts.

## Configuration

All config via environment variables. See `.env.example` for the full list.

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
| `SANDBOX_ENABLED` | `true` | Enable `/run` code execution |
| `RAG_ENABLED` | `false` | Enable codebase-aware context |

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

- **Set `LLM_CONTEXT_WINDOW` accurately** — the bot auto-adjusts how much conversation history and file context it sends to the model based on this value
- **Lower the temperature** — `0.1` reduces creative hallucination on smaller models
- **Reduce `LLM_MAX_TOKENS`** — smaller models produce better, more focused output with lower limits
- **Use Ollama's `num_ctx` parameter** — ensure Ollama allocates enough context: `ollama run gemma3:12b --num_ctx 8192`
- **Monitor VRAM** — if the model runs out of VRAM it falls back to CPU, causing extreme slowdowns

## Project Structure

```
forge_bot/
├── server.py          # FastAPI app, webhook endpoint, HMAC
├── config.py          # pydantic-settings config
├── router.py          # Event type → handler dispatch
├── models.py          # Pydantic models for webhook payloads
├── handlers/          # PR review, issue comment, assignment
├── clients/           # Gitea/Forgejo API + LLM clients
├── sandbox/           # DinD container orchestration
├── rag/               # Optional: ingest, chunk, embed, retrieve
└── utils/             # Dedup, diff parsing, formatting
```

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
pytest
```

## License

MIT
