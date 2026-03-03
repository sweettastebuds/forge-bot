import logging
import re

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
logger = logging.getLogger("forge_bot.thinking")


def extract_thinking(text: str) -> str:
    """Extract all <think>...</think> sections from the text."""
    thoughts = _THINK_RE.findall(text)
    if thoughts:
        for i, thought in enumerate(thoughts, 1):
            logger.debug("Thinking block %d: %s", i, thought[:200])

    return _THINK_RE.sub("", text).strip()
