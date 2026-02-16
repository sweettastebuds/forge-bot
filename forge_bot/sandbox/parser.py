"""Parse /run commands and code blocks from comment bodies."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class RunCommand:
    """Parsed /run command from a comment body."""

    language: str
    code: str
    network_enabled: bool


# Match /run [--net] <language> followed by a fenced code block.
# The code block language tag is optional (may differ from the /run language).
_RUN_CMD_RE = re.compile(
    r"(?:^|\s)/run"               # /run command
    r"(?:\s+--net)?"              # optional --net flag
    r"\s+(\w+)"                   # language argument
    r"\s*\n"                      # newline before code block
    r"```\w*\s*\n"                # opening fence (optional language tag)
    r"(.*?)"                      # code content (non-greedy)
    r"\n```",                     # closing fence
    re.DOTALL | re.MULTILINE,
)

_NET_FLAG_RE = re.compile(r"(?:^|\s)/run\s+--net\s", re.MULTILINE)


def parse_run_command(body: str, prefix: str = "/") -> RunCommand | None:
    """Parse a /run command from a comment body.

    Supported formats::

        @bot /run python
        ```python
        print("hello")
        ```

        @bot /run --net node
        ```javascript
        const http = require('http');
        ```

    Returns *None* if no valid /run command is found.
    """
    match = _RUN_CMD_RE.search(body)
    if not match:
        return None

    language = match.group(1).lower()
    code = match.group(2)
    network_enabled = bool(_NET_FLAG_RE.search(body))

    return RunCommand(
        language=language,
        code=code,
        network_enabled=network_enabled,
    )
