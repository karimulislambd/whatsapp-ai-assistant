"""FastAPI entrypoint: WhatsApp webhook, web demo API, health and demo page routes."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import httpx
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse, RedirectResponse
from pydantic import BaseModel

from app import __version__
from app.config import Settings, get_settings
from app.handlers import Assistant, HandlerResult, IncomingMessage, Media
from app.llm import LLM, GroqLLM
from app.memory import Store, hash_user_id
from app.whatsapp import (
    WhatsAppClient,
    WhatsAppError,
    WhatsAppMessage,
    parse_webhook,
    verify_signature,
)

logger = logging.getLogger("app")

STATIC_DIR = Path(__file__).resolve().parent / "static"
SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
SUPPORTED_KINDS = {"text", "image", "audio"}


class ChatResponse(BaseModel):
    replies: list[str]
    mode: str


def create_app(
    settings: Settings | None = None,
    *,
    llm: LLM | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> FastAPI:
    """Application factory. Tests inject a fake LLM; production builds the real Groq client."""
    settings = settings or get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        http = http_client or httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0))
        store = Store(settings.db_path)
        app.state.settings = settings
        app.state.store = store
        app.state.assistant = Assistant(settings, store, llm or GroqLLM(settings))
        app.state.whatsapp = WhatsAppClient(
            http, settings.whatsapp_token, settings.whatsapp_phone_number_id, settings.graph_url
        )
        if not settings.groq_api_key and llm is None:
            logger.warning("GROQ_API_KEY is not set — every model call will fail")
        if not (settings.whatsapp_token and settings.whatsapp_app_secret):
            logger.warning("WhatsApp is not fully configured — only the web demo will work")
        try:
            yield
        finally:
            if http_client is None:
                await http.aclose()
            store.close()

    app = FastAPI(
        title="WhatsApp AI Assistant",
        version=__version__,
        description="Multimodal WhatsApp assistant (text, voice, image) powered by Groq.",
        lifespan=lifespan,
    )

    # ------------------------------------------------------------------ basics
    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse("/demo")

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {
            "status": "ok",
            "version": __version__,
            "groq_configured": bool(settings.groq_api_key) or llm is not None,
            "whatsapp_configured": bool(
                settings.whatsapp_token
                and settings.whatsapp_phone_number_id
                and settings.whatsapp_app_secret
            ),
        }

    @app.get("/demo", include_in_schema=False)
    async def demo() -> FileResponse:
        return FileResponse(STATIC_DIR / "demo.html", media_type="text/html")

    # ----------------------------------------------------------------- webhook
    @app.get("/webhook", response_class=PlainTextResponse)
    async def verify_webhook(
        mode: Annotated[str | None, Query(alias="hub.mode")] = None,
        token: Annotated[str | None, Query(alias="hub.verify_token")] = None,
        challenge: Annotated[str | None, Query(alias="hub.challenge")] = None,
    ) -> str:
        """Meta's one-time verification handshake when the webhook URL is configured."""
        expected = settings.whatsapp_verify_token
        if mode == "subscribe" and expected and token == expected and challenge is not None:
            logger.info("Webhook verified")
            return challenge
        raise HTTPException(status_code=403, detail="Verification failed")

    @app.post("/webhook")
    async def receive_webhook(request: Request, background: BackgroundTasks) -> dict[str, str]:
        raw = await request.body()
        if not verify_signature(
            settings.whatsapp_app_secret, raw, request.headers.get("x-hub-signature-256")
        ):
            logger.warning("Rejected webhook with missing/invalid signature")
            raise HTTPException(status_code=403, detail="Invalid signature")
        try:
            payload = json.loads(raw)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid JSON") from exc

        store: Store = request.app.state.store
        for message in parse_webhook(payload):
            # Meta retries deliveries it thinks failed — process each message id once.
            if store.mark_processed(message.id):
                background.add_task(
                    process_whatsapp_message,
                    request.app.state.assistant,
                    request.app.state.whatsapp,
                    message,
                )
            else:
                logger.info("Skipping duplicate message %s", message.id)
        # Acknowledge immediately; the model calls happen in the background task.
        return {"status": "ok"}

    # ------------------------------------------------------------ web demo API
    @app.post("/api/chat", response_model=ChatResponse)
    async def api_chat(
        request: Request,
        session_id: Annotated[str, Form()],
        message: Annotated[str, Form()] = "",
        file: Annotated[UploadFile | None, File()] = None,
    ) -> ChatResponse:
        if not SESSION_ID_RE.match(session_id):
            raise HTTPException(status_code=422, detail="Invalid session_id")

        incoming: IncomingMessage
        if file is not None and file.filename:
            content_type = (file.content_type or "").split(";")[0].strip().lower()
            if content_type.startswith("image/"):
                kind, limit = "image", settings.max_media_bytes
            elif content_type.startswith("audio/") or content_type == "video/webm":
                kind, limit = "audio", settings.max_audio_bytes
                content_type = content_type.replace("video/", "audio/")
            else:
                raise HTTPException(status_code=415, detail="Only images and audio are supported")
            data = await file.read(limit + 1)
            if len(data) > limit:
                raise HTTPException(status_code=413, detail="File too large")
            media = Media(data=data, mime_type=content_type)

            async def load_media() -> Media:
                return media

            incoming = IncomingMessage(
                user_id=f"web:{session_id}", kind=kind, text=message, load_media=load_media
            )
        elif message.strip():
            incoming = IncomingMessage(user_id=f"web:{session_id}", kind="text", text=message)
        else:
            raise HTTPException(status_code=422, detail="Send a message or a file")

        result: HandlerResult = await request.app.state.assistant.handle(incoming)
        return ChatResponse(replies=result.replies, mode=result.mode)

    return app


async def process_whatsapp_message(
    assistant: Assistant, whatsapp: WhatsAppClient, message: WhatsAppMessage
) -> None:
    """Background task: run one WhatsApp message through the core handler and send replies."""
    user_ref = hash_user_id(message.wa_id)[:10]  # never log raw phone numbers
    try:
        await whatsapp.mark_read(message.id)
    except WhatsAppError as exc:
        logger.warning("Could not mark %s as read: %s", message.id, exc)

    load_media = None
    if message.media_id:
        media_id, fallback_mime = message.media_id, message.mime_type or ""

        async def load_media() -> Media:
            data, mime = await whatsapp.download_media(media_id)
            return Media(data=data, mime_type=mime or fallback_mime.split(";")[0])

    kind = message.type if message.type in SUPPORTED_KINDS else "unsupported"
    incoming = IncomingMessage(
        user_id=f"wa:{message.wa_id}",
        kind=kind,  # type: ignore[arg-type]
        text=message.text or message.caption,
        load_media=load_media,
        raw_type=message.type,
    )
    try:
        result = await assistant.handle(incoming)
        for reply in result.replies:
            await whatsapp.send_text(message.wa_id, reply)
        logger.info("Handled %s message for user %s", message.type, user_ref)
    except Exception:
        logger.exception("Failed to handle message %s for user %s", message.id, user_ref)


app = create_app()
