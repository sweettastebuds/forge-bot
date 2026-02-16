# TODO

## Branch: `v2/tool-calling`

### Action Items

- [ ] **Per-request trace logging** — Add a structured INFO-level summary log
  line at the end of each tool loop. Should include: mode used (native/prompt),
  rounds taken, tools invoked (names + args), whether fallback triggered,
  prompt size (chars), reply length (chars). Makes it easy to grep and share
  for debugging.

- [ ] **Slash command awareness in LLM prompt** — The LLM has no knowledge of
  the bot's available slash commands (`/run`, `/index`). If a user asks
  "what can you do?" or "help", the bot can't describe its own capabilities.
  The router dispatches `/run` and `/index` before the LLM is invoked, but
  the prompt template should include a capabilities section so the LLM can
  guide users. Commands should be discoverable like Claude Code's slash
  commands — listed, described, with usage examples.

- [ ] **Reduce whitespace bloat in prompts** — The Jinja2 template
  (`issue_respond.j2`) doesn't use whitespace control (`{%- -%}`), so every
  `{% if %}` / `{% for %}` / `{% endif %}` block tag emits an extra blank
  line. Between sections this compounds to 3-4 blank lines. File content
  included in prompts also preserves original blank lines. For small-context
  models these wasted tokens matter. Fix: add Jinja2 whitespace trimming
  and optionally collapse consecutive blank lines in file content.

### Backlog

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
