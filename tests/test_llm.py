"""Groq wrapper: request shape and the retry-once-then-fail policy (Groq API mocked with respx)."""

import json

import httpx
import pytest
import respx

from app.config import Settings
from app.llm import GroqLLM, LLMError

GROQ = "https://api.groq.com/openai/v1"


def completion(content: str) -> dict:
    return {
        "id": "c1",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


@pytest.fixture
def llm() -> GroqLLM:
    return GroqLLM(Settings(_env_file=None, groq_api_key="k"), retry_delay=0)


@respx.mock
async def test_chat_returns_content_and_uses_chat_model(llm):
    route = respx.post(f"{GROQ}/chat/completions").respond(200, json=completion(" Hi! "))
    assert await llm.chat([{"role": "user", "content": "hey"}]) == "Hi!"
    assert json.loads(route.calls.last.request.content)["model"] == "llama-3.3-70b-versatile"


@respx.mock
async def test_chat_retries_once_on_server_error(llm):
    route = respx.post(f"{GROQ}/chat/completions").mock(
        side_effect=[httpx.Response(503, json={}), httpx.Response(200, json=completion("ok"))]
    )
    assert await llm.chat([{"role": "user", "content": "hey"}]) == "ok"
    assert route.call_count == 2


@respx.mock
async def test_chat_gives_up_after_second_failure(llm):
    route = respx.post(f"{GROQ}/chat/completions").respond(500, json={})
    with pytest.raises(LLMError):
        await llm.chat([{"role": "user", "content": "hey"}])
    assert route.call_count == 2


@respx.mock
async def test_chat_does_not_retry_client_errors(llm):
    route = respx.post(f"{GROQ}/chat/completions").respond(401, json={"error": {}})
    with pytest.raises(LLMError):
        await llm.chat([{"role": "user", "content": "hey"}])
    assert route.call_count == 1


@respx.mock
async def test_vision_sends_base64_image(llm):
    route = respx.post(f"{GROQ}/chat/completions").respond(200, json=completion("a cat"))
    out = await llm.analyze_image(b"\x89PNG", "image/png", "What is it?", "sys")
    assert out == "a cat"
    body = json.loads(route.calls.last.request.content)
    assert body["model"] == "meta-llama/llama-4-scout-17b-16e-instruct"
    parts = body["messages"][1]["content"]
    assert parts[0] == {"type": "text", "text": "What is it?"}
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")


@respx.mock
async def test_transcribe_uses_whisper(llm):
    route = respx.post(f"{GROQ}/audio/transcriptions").respond(200, json={"text": " hello "})
    assert await llm.transcribe(b"OggS", "audio.ogg") == "hello"
    assert b"whisper-large-v3" in route.calls.last.request.content
