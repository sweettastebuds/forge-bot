# TODO

## Branch: `v2/tool-calling`

### Action Items

- [x] **PR context in issue_comment handler** — Fetch PR diff and changed
  files when the bot is @mentioned on a PR. Includes truncation and graceful
  degradation on API failure.

- [x] **Slash command awareness in LLM prompt** — Added YOUR CAPABILITIES
  section to the prompt template listing `/run` and `/index` with usage
  examples. LLM can now guide users on available commands.

- [x] **Per-request trace logging** — Structured INFO-level TRACE log line
  at the end of each tool loop: mode, rounds, tool count, fallback flag,
  prompt size, reply length.

- [x] **Reduce whitespace bloat in prompts** — Added Jinja2 whitespace
  control (`{%- -%}`) to all block tags in `issue_respond.j2` and added
  post-render collapsing of consecutive blank lines (3+ → 2) in
  `_build_system_prompt`.

### Action Items (new)

- [x] **Silence verbose third-party DEBUG logging** — Pinned `httpcore`,
  `openai`, `urllib3`, `docker`, and `httpx` loggers to WARNING in
  `server.py` lifespan. Our TRACE/INFO output is now visible even at
  DEBUG level.

- [x] **Add trace logging to PR review handler** — Added TRACE line to
  `PullRequestHandler.handle()` with prompt/reply/diff size and file count.

- [x] **Add missing env vars to docker-compose.yml** — Already present
  (added by user).

- [ ] **PR inline review comments** — The PR handler currently posts a plain
  comment via `post_comment()`. Gitea supports `POST /pulls/{index}/reviews`
  with per-line comments and suggestions. The LLM output format (severity
  prefixes + file:line references) already targets this. Parse the LLM review
  output and map findings to inline review comments. Fall back to summary
  comment on parse failure.

### Backlog

- [ ] **Repo cloning / git operations in sandbox** — The bot cannot clone the
  repo, run git commands, execute tests, or do any analysis that requires a
  working copy. It's limited to reading individual files via the Forge API.
  This blocks use cases like "run the tests", "check if this compiles",
  or deeper multi-file analysis. *Backlog because*: requires significant
  architecture work — sandbox integration for git clone, credential passing,
  workspace lifecycle management. Should be its own feature branch.

- [ ] **Diagnostic bot command (`/debug`)** — A slash command the user can
  trigger that makes the bot reply with its last request trace (mode, tools
  used, prompt size, model, response time). *Backlog because*: the per-request
  trace logging covers most debugging needs; this is a convenience feature
  for live debugging on Gitea without access to server logs.

- [ ] **JSON trace files** — Write structured request traces to a directory
  (prompt, tool calls/results, final reply, timings). *Backlog because*:
  this is heavyweight infrastructure; log-based debugging should be tried
  first and only if it proves insufficient should we invest in file-based
  traces.

- [ ] **Write tools for LLM (post_comment, create_branch, commit_file)** —
  Let the LLM take write actions on the Forge instance (post comments,
  create branches, commit files) when approved by the user. *Backlog
  because*: requires careful trust/safety boundary design — should the
  LLM auto-commit, or require user approval? How to prevent infinite loops
  (bot comments triggering its own webhooks)? Should be its own feature
  branch with a clear permission model.
