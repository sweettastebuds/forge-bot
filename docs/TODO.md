# TODO

## Branch: `v2/tool-calling`

### Action Items

- [ ] **PR context in issue_comment handler** — When the bot is @mentioned on
  a PR (detected via `event.is_pull` or `issue.pull_request is not None`),
  the handler currently has NO access to the PR diff, changed files, or PR
  metadata. The user says "review this PR" and the bot says "I don't have
  access to that information." Fix: when `is_pull` is true, fetch the PR diff
  and changed files (like `PullRequestHandler` does) and include them in
  the prompt context. Also detect PR references like `#2` in comments and
  fetch the linked PR's diff/details via a new tool or pre-fetch.

- [ ] **Slash command awareness in LLM prompt** — The LLM has no knowledge of
  the bot's available slash commands (`/run`, `/index`, future `/debug`).
  If a user asks "what can you do?" or types an unknown command, the bot
  can't describe its own capabilities. The router dispatches `/run` and
  `/index` before the LLM is invoked, but the prompt template should include
  a capabilities section so the LLM can guide users. Commands should be
  discoverable — listed, described, with usage examples. Example failure:
  user types `/debug` and bot says "The /debug command is not directly
  supported by the available tools."

- [ ] **Per-request trace logging** — Add a structured INFO-level summary log
  line at the end of each tool loop. Should include: mode used (native/prompt),
  rounds taken, tools invoked (names + args), whether fallback triggered,
  prompt size (chars), reply length (chars). Makes it easy to grep and share
  for debugging.

- [ ] **Reduce whitespace bloat in prompts** — The Jinja2 template
  (`issue_respond.j2`) doesn't use whitespace control (`{%- -%}`), so every
  `{% if %}` / `{% for %}` / `{% endif %}` block tag emits an extra blank
  line. Between sections this compounds to 3-4 blank lines. File content
  included in prompts also preserves original blank lines. For small-context
  models these wasted tokens matter. Fix: add Jinja2 whitespace trimming
  and optionally collapse consecutive blank lines in file content.

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
