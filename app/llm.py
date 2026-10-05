"""Groq client wrapper: chat (GPT-OSS 120B), vision (Qwen 3.8) and speech-to-text (Whisper).

Every call is retried once on transient failures (timeouts, connection errors, 429, 5xx);
if the retry also fails an :class:`LLMError` is raised so the handler can apologise gracefully.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from collections.abc import Awaitable, Callable
from typing import Protocol, TypeVar

import groq

from app.config import Settings

logger = logging.getLogger(__name__)

T = TypeVar("T")


class LLMError(Exception):
    """Raised when the language model could not produce a result."""


class LLM(Protocol):
    """Interface the transport-agnostic handler depends on (makes testing trivial)."""

    async def chat(self, messages: list[dict[str, str]]) -> str: ...

    async def analyze_image(
        self, image: bytes, mime_type: str, prompt: str, system_prompt: str
    ) -> str: ...

    async def transcribe(self, audio: bytes, filename: str) -> str: ...


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, groq.APIConnectionError):  # includes APITimeoutError
        return True
    if isinstance(exc, groq.APIStatusError):
        return exc.status_code == 429 or exc.status_code >= 500
    return False


class GroqLLM:
    def __init__(
        self,
        settings: Settings,
        client: groq.AsyncGroq | None = None,
        retry_delay: float = 1.0,
    ) -> None:
        self._settings = settings
        # We own the retry policy, so disable the SDK's built-in retries.
        self._client = client or groq.AsyncGroq(
            api_key=settings.groq_api_key or "missing-key",
            max_retries=0,
            timeout=settings.llm_timeout_seconds,
        )
        self._retry_delay = retry_delay

    async def _call(self, what: str, fn: Callable[[], Awaitable[T]]) -> T:
        for attempt in (1, 2):
            try:
                return await fn()
            except groq.APIError as exc:
                if attempt == 1 and _is_retryable(exc):
                    logger.warning("Groq %s failed (%s); retrying once", what, type(exc).__name__)
                    await asyncio.sleep(self._retry_delay)
                    continue
                logger.error("Groq %s failed: %s", what, exc)
                raise LLMError(f"{what} failed") from exc
        raise LLMError(f"{what} failed")  # pragma: no cover - loop always returns or raises

    async def chat(self, messages: list[dict[str, str]]) -> str:
        async def run() -> str:
            resp = await self._client.chat.completions.create(
                model=self._settings.chat_model,
                messages=messages,
                temperature=0.5,
                max_tokens=1024,
            )
            return (resp.choices[0].message.content or "").strip()

        return await self._call("chat", run)

    async def analyze_image(
        self, image: bytes, mime_type: str, prompt: str, system_prompt: str
    ) -> str:
        data_url = f"data:{mime_type};base64,{base64.b64encode(image).decode('ascii')}"

        async def run() -> str:
            resp = await self._client.chat.completions.create(
                model=self._settings.vision_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    },
                ],
                temperature=0.3,
                # Groq's free tier caps Qwen output at 1,000 tokens/min, so stay under it.
                max_tokens=900,
                # Skip Qwen's thinking step: faster and cheaper on that budget.
                extra_body=(
                    {"reasoning_effort": "none"}
                    if self._settings.vision_model.startswith("qwen/")
                    else None
                ),
            )
            return (resp.choices[0].message.content or "").strip()

        return await self._call("vision", run)

    async def transcribe(self, audio: bytes, filename: str) -> str:
        async def run() -> str:
            resp = await self._client.audio.transcriptions.create(
                model=self._settings.whisper_model,
                file=(filename, audio),
                response_format="json",
            )
            return (resp.text or "").strip()

        return await self._call("transcription", run)
