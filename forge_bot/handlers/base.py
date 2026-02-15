"""Abstract base class for all webhook event handlers."""

import logging
from abc import ABC, abstractmethod

import jinja2

from forge_bot.clients.forge import ForgeClient
from forge_bot.clients.llm import LLMClient
from forge_bot.config import Settings

logger = logging.getLogger("forge_bot.handlers")

_template_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader("prompts"),
    autoescape=False,
    keep_trailing_newline=True,
)


class BaseHandler(ABC):
    """Base class injecting shared dependencies into every handler."""

    def __init__(
        self,
        forge_client: ForgeClient,
        llm_client: LLMClient,
        settings: Settings,
        bot_username: str,
    ) -> None:
        self.forge = forge_client
        self.llm = llm_client
        self.settings = settings
        self.bot_username = bot_username

    def render_template(self, template_name: str, **kwargs: object) -> str:
        """Render a Jinja2 prompt template from the prompts/ directory."""
        template = _template_env.get_template(template_name)
        return template.render(**kwargs)

    @abstractmethod
    async def handle(self, event: object) -> None:
        """Process a webhook event. Subclasses must implement."""
