"""WhatsApp Cloud API: webhook signature check, payload parsing and Graph API client."""

from __future__ import annotations

import hashlib
import hmac
import logging
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)

MAX_TEXT_LENGTH = 4096  # WhatsApp's limit for a text message body


class WhatsAppError(Exception):
    """Raised when a Graph API call fails."""


# --------------------------------------------------------------------------- signature
def verify_signature(app_secret: str, raw_body: bytes, signature_header: str | None) -> bool:
    """Validate Meta's ``X-Hub-Signature-256`` header (HMAC-SHA256 of the raw request body).

    Uses a constant-time comparison so the check does not leak timing information.
    """
    if not app_secret or not signature_header or not signature_header.startswith("sha256="):
        return False
    expected = hmac.new(app_secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
    received = signature_header.removeprefix("sha256=").strip()
    return hmac.compare_digest(expected, received)


# ----------------------------------------------------------------------------- parsing
@dataclass(frozen=True)
class WhatsAppMessage:
    id: str
    wa_id: str
    type: str
    text: str | None = None
    media_id: str | None = None
    mime_type: str | None = None
    caption: str | None = None


def parse_webhook(payload: dict[str, Any]) -> list[WhatsAppMessage]:
    """Extract user messages from a webhook payload.

    Status callbacks (sent / delivered / read / failed) arrive on the same ``messages`` field
    but under ``value.statuses`` — they are ignored here, as are reactions.
    """
    messages: list[WhatsAppMessage] = []
    for entry in payload.get("entry") or []:
        for change in entry.get("changes") or []:
            if change.get("field") != "messages":
                continue
            for msg in (change.get("value") or {}).get("messages") or []:
                parsed = _parse_message(msg)
                if parsed is not None:
                    messages.append(parsed)
    return messages


def _parse_message(msg: dict[str, Any]) -> WhatsAppMessage | None:
    msg_id, wa_id, msg_type = msg.get("id"), msg.get("from"), msg.get("type", "unknown")
    if not msg_id or not wa_id:
        return None
    base = {"id": msg_id, "wa_id": wa_id}

    match msg_type:
        case "text":
            return WhatsAppMessage(**base, type="text", text=(msg.get("text") or {}).get("body"))
        case "image" | "audio":
            media = msg.get(msg_type) or {}
            return WhatsAppMessage(
                **base,
                type=msg_type,
                media_id=media.get("id"),
                mime_type=media.get("mime_type"),
                caption=media.get("caption"),
            )
        case "button":  # quick-reply button on a template message
            return WhatsAppMessage(**base, type="text", text=(msg.get("button") or {}).get("text"))
        case "interactive":
            interactive = msg.get("interactive") or {}
            reply = interactive.get("button_reply") or interactive.get("list_reply") or {}
            return WhatsAppMessage(**base, type="text", text=reply.get("title"))
        case "reaction" | "system":
            return None
        case _:
            return WhatsAppMessage(**base, type=msg_type)


# --------------------------------------------------------------------------- splitting
def split_message(text: str, limit: int = MAX_TEXT_LENGTH) -> list[str]:
    """Split ``text`` into chunks of at most ``limit`` characters.

    Prefers paragraph breaks, then line breaks, then spaces, and only hard-cuts as a last resort.
    """
    text = text.strip()
    if not text:
        return []
    chunks: list[str] = []
    while len(text) > limit:
        window = text[:limit]
        cut = -1
        for sep in ("\n\n", "\n", " "):
            cut = window.rfind(sep)
            if cut > limit // 2:
                break
        if cut <= 0:
            cut = limit
        chunks.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    if text:
        chunks.append(text)
    return chunks


# ------------------------------------------------------------------------------ client
class WhatsAppClient:
    """Minimal async client for the Graph API endpoints this bot needs."""

    def __init__(
        self,
        http: httpx.AsyncClient,
        token: str,
        phone_number_id: str,
        graph_url: str,
    ) -> None:
        self._http = http
        self._token = token
        self._graph_url = graph_url.rstrip("/")
        self._messages_url = f"{self._graph_url}/{phone_number_id}/messages"

    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token}"}

    async def _post_message(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            resp = await self._http.post(self._messages_url, json=payload, headers=self._headers)
        except httpx.HTTPError as exc:
            raise WhatsAppError(f"Graph API request failed: {exc}") from exc
        if resp.is_error:
            raise WhatsAppError(f"Graph API error {resp.status_code}: {resp.text[:500]}")
        return resp.json()

    async def send_text(self, to: str, text: str) -> None:
        """Send a text reply, split into several messages if it exceeds 4096 characters."""
        for chunk in split_message(text):
            await self._post_message(
                {
                    "messaging_product": "whatsapp",
                    "recipient_type": "individual",
                    "to": to,
                    "type": "text",
                    "text": {"preview_url": False, "body": chunk},
                }
            )

    async def mark_read(self, message_id: str, typing_indicator: bool = True) -> None:
        """Mark an incoming message as read (blue ticks) and show "typing…" while we work."""
        payload: dict[str, Any] = {
            "messaging_product": "whatsapp",
            "status": "read",
            "message_id": message_id,
        }
        if typing_indicator:
            payload["typing_indicator"] = {"type": "text"}
        await self._post_message(payload)

    async def download_media(self, media_id: str) -> tuple[bytes, str]:
        """Resolve a media id to its short-lived URL, then download the bytes.

        Returns ``(content, mime_type)``. WhatsApp caps media size (images 5 MB, audio 16 MB),
        so the download is bounded; the handler enforces the model-specific limits.
        """
        try:
            meta_resp = await self._http.get(f"{self._graph_url}/{media_id}", headers=self._headers)
            if meta_resp.is_error:
                raise WhatsAppError(f"Media lookup failed {meta_resp.status_code}")
            meta = meta_resp.json()
            file_resp = await self._http.get(meta["url"], headers=self._headers)
            if file_resp.is_error:
                raise WhatsAppError(f"Media download failed {file_resp.status_code}")
        except httpx.HTTPError as exc:
            raise WhatsAppError(f"Media request failed: {exc}") from exc
        except (KeyError, ValueError) as exc:
            raise WhatsAppError(f"Unexpected media response: {exc}") from exc
        mime = (meta.get("mime_type") or file_resp.headers.get("content-type") or "").split(";")[0]
        return file_resp.content, mime.strip()
