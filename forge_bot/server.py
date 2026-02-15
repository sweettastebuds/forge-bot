"""FastAPI webhook server for forge-bot."""

import hashlib
import hmac
import json
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Response

from forge_bot.config import Settings
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

    # TODO Phase 2: Call GET /api/v1/user to resolve bot identity
    # TODO Phase 5: Initialize sandbox orchestrator, pre-pull images

    yield

    logger.info("forge-bot shutting down")


app = FastAPI(title="forge-bot", version="0.1.0", lifespan=lifespan)


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


async def process_webhook(event_type: str, payload: dict[str, Any]) -> None:
    """Background task: route the webhook event to the appropriate handler.

    This runs after the HTTP 200 has already been returned to Gitea.
    """
    action = payload.get("action", "")
    repo = payload.get("repository", {}).get("full_name", "unknown")
    logger.info("Processing event=%s action=%s repo=%s", event_type, action, repo)

    # TODO Phase 2: Dispatch to router.py
    # router.dispatch(event_type, payload, settings, bot_username)


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

    background_tasks.add_task(process_webhook, event_type, payload)

    return Response(status_code=200, content="OK")


@app.get("/health")
async def health() -> dict[str, str]:
    """Simple health check endpoint."""
    return {"status": "ok"}
