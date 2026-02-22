"""FastAPI webhook server for forge-bot."""

import hashlib
import hmac
import json
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Response

from forge_bot.api.client import GenericForgeClient
from forge_bot.clients.llm import LLMClient
from forge_bot.config import Settings
from forge_bot.router import dispatch
from forge_bot.utils.dedup import DeliveryTracker

logger = logging.getLogger("forge_bot")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize application state on startup."""
    app.state.settings = Settings()
    app.state.dedup = DeliveryTracker()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, app.state.settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    logger.info("forge-bot starting up")
    logger.info("Forge instance: %s", app.state.settings.forge_instance_url)
    logger.info(
        "LLM endpoint: %s (model: %s)",
        app.state.settings.llm_base_url,
        app.state.settings.llm_model,
    )

    # Initialize API client (YAML-driven generic client)
    api_client = GenericForgeClient(app.state.settings)
    app.state.api_client = api_client

    # Resolve bot identity via Forge API
    try:
        bot_user = await api_client.call("get_authenticated_user")
        app.state.bot_username = bot_user["login"]
        logger.info(
            "Bot identity resolved: %s (id=%d)",
            bot_user["login"],
            bot_user["id"],
        )
    except Exception as e:
        logger.critical(
            "FATAL: Could not resolve bot identity"
            "Self-loop guard cannot function without bot identity."
            "Check FORGE_INSTANCE_URL and FORGE_API_TOKEN."
        )
        raise RuntimeError(
            "Bot identity resolution failed - cannot start server safely."
        ) from e

    # Initialize LLM client
    llm_client = LLMClient(app.state.settings)
    app.state.llm_client = llm_client

    yield

    await llm_client.close()
    await api_client.close()
    logger.info("forge-bot shutting down")


app = FastAPI(title="forge-bot", version="0.2.0", lifespan=lifespan)


def _get_header(headers: dict[str, str], *names: str) -> str | None:
    """Get the first matching header value, case-insensitive.

    Checks Forgejo headers first, then Gitea, per TDD spec.
    """
    for name in names:
        value = headers.get(name)
        if value is not None:
            return value
    return None


def verify_hmac(body: bytes, signature: str | None, secret: str) -> None:
    """Verify HMAC-SHA256 signature over raw body bytes.

    Raises HTTPException(403) on failure.
    """
    if not signature:
        raise HTTPException(status_code=403, detail="Missing signature header")

    expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()

    if not hmac.compare_digest(signature, expected):
        raise HTTPException(status_code=403, detail="Invalid signature")


def _extract_comment_target(
    event_type: str, payload: dict[str, Any]
) -> tuple[str, str, int] | None:
    """
    Extract (owner, repo, issue_number) from the payload for error comments.

    Returns None if target cannot be determined.
    """
    repo_data = payload.get("repository", {})
    if not repo_data:
        return None

    owner = repo_data.get("owner", {}).get("login")
    repo = repo_data.get("name")

    if not owner or not repo:
        return None

    # Extract issue/PR number based on event type
    issue_number = None
    if event_type == "issue_comment":
        issue_number = payload.get("issue", {}).get("number")
    elif event_type == "pull_request":
        issue_number = payload.get("pull_request", {}).get("number")
    elif event_type == "issues":
        issue_number = payload.get("issue", {}).get("number")

    if issue_number is None:
        return None

    return (owner, repo, issue_number)


async def process_webhook(
    event_type: str,
    payload: dict[str, Any],
    bot_username: str,
    api_client: GenericForgeClient,
    llm_client: LLMClient,
    settings: Settings,
) -> None:
    """Background task: route the webhook event to the appropriate handler.

    This runs after the HTTP 200 has already been returned to Gitea.
    """
    action = payload.get("action", "")
    repo = payload.get("repository", {}).get("full_name", "unknown")
    logger.info("Processing event=%s action=%s repo=%s", event_type, action, repo)

    try:
        await dispatch(
            event_type,
            payload,
            bot_username,
            api_client=api_client,
            llm_client=llm_client,
            settings=settings,
        )
    except Exception as e:
        logger.exception(
            "Error processing event=%s action=%s repo=%s",
            event_type,
            action,
            repo,
        )

        # Attempt to post error comment to notify user
        target = _extract_comment_target(event_type, payload)
        if not target:
            return

        owner, repo_name, issue_number = target
        error_body = (
            "⚠️ **Processing Error**\n\n"
            f"Failed to process this event due to an internal error:\n"
            f"```\n{type(e).__name__}: {str(e)}\n```\n\n"
            f"Please check the bot logs or contact your administrator."
        )
        try:
            await api_client.call(
                "create_issue_comment",
                owner=owner,
                repo=repo_name,
                index=issue_number,
                body=error_body,
            )
            logger.info(
                "Posted error comment to %s/%s#%d",
                owner,
                repo_name,
                issue_number,
            )
        except Exception as ex:
            logger.exception(
                "Failed to post error comment to %s/%s#%d",
                owner,
                repo_name,
                issue_number,
            )
            logger.debug("Exception details:", exc_info=ex)


@app.post("/webhook")
async def webhook(request: Request, background_tasks: BackgroundTasks) -> Response:
    """Receive and verify a Gitea/Forgejo webhook delivery.

    1. Read raw body bytes (before JSON parsing)
    2. Verify HMAC-SHA256 signature
    3. Check delivery UUID for deduplication
    4. Return 200 immediately
    5. Dispatch processing as a background task
    """
    settings: Settings = request.app.state.settings
    dedup: DeliveryTracker = request.app.state.dedup

    # Step 1: Read raw body
    body = await request.body()

    # Step 2: Verify HMAC — check Forgejo header first, then Gitea
    signature = _get_header(
        request.headers,
        "x-forgejo-signature",
        "x-gitea-signature",
    )
    verify_hmac(body, signature, settings.forge_webhook_secret)

    # Step 3: Deduplication via delivery UUID
    delivery_id = _get_header(
        request.headers,
        "x-forgejo-delivery",
        "x-gitea-delivery",
    )
    if delivery_id and dedup.is_duplicate(delivery_id):
        logger.debug("Duplicate delivery %s, skipping", delivery_id)
        return Response(status_code=200, content="OK (duplicate)")

    # Step 4: Parse event type
    event_type = _get_header(
        request.headers,
        "x-forgejo-event",
        "x-gitea-event",
    )
    if not event_type:
        logger.warning("Webhook missing event type header")
        return Response(status_code=200, content="OK (no event type)")

    # Step 5: Parse body and dispatch as background task
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        logger.warning("Webhook body is not valid JSON")
        return Response(status_code=400, content="Invalid JSON")

    logger.info(
        "Received webhook: event=%s delivery=%s",
        event_type,
        delivery_id or "unknown",
    )

    bot_username: str = request.app.state.bot_username
    background_tasks.add_task(
        process_webhook,
        event_type,
        payload,
        bot_username,
        request.app.state.api_client,
        request.app.state.llm_client,
        settings,
    )

    return Response(status_code=200, content="OK")


@app.get("/health")
async def health() -> dict[str, str]:
    """Simple health check endpoint."""
    return {"status": "ok"}
