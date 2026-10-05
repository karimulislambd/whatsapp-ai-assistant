"""Shared fixtures. No test touches the network: Groq is replaced by a fake LLM and the
Graph API is mocked with respx."""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Iterator
from typing import Any

import pytest
import respx
from fastapi.testclient import TestClient

from app.config import Settings
from app.handlers import Assistant
from app.llm import LLMError
from app.main import create_app
from app.memory import Store

APP_SECRET = "test-app-secret"
VERIFY_TOKEN = "test-verify-token"
PHONE_NUMBER_ID = "1234567890"
GRAPH = "https://graph.facebook.com/v21.0"


class FakeLLM:
    """Records calls and returns canned answers; set ``fail=True`` to simulate Groq outages."""

    def __init__(self) -> None:
        self.chat_calls: list[list[dict[str, str]]] = []
        self.vision_calls: list[dict[str, Any]] = []
        self.transcribe_calls: list[dict[str, Any]] = []
        self.chat_reply = "fake chat reply"
        self.vision_reply = "fake image analysis"
        self.transcript = "hello from a voice note"
        self.fail = False

    async def chat(self, messages: list[dict[str, str]]) -> str:
        self.chat_calls.append(messages)
        if self.fail:
            raise LLMError("boom")
        return self.chat_reply

    async def analyze_image(
        self, image: bytes, mime_type: str, prompt: str, system_prompt: str
    ) -> str:
        self.vision_calls.append({"image": image, "mime_type": mime_type, "prompt": prompt})
        if self.fail:
            raise LLMError("boom")
        return self.vision_reply

    async def transcribe(self, audio: bytes, filename: str) -> str:
        self.transcribe_calls.append({"audio": audio, "filename": filename})
        if self.fail:
            raise LLMError("boom")
        return self.transcript


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        groq_api_key="test-groq-key",
        whatsapp_token="test-wa-token",
        whatsapp_phone_number_id=PHONE_NUMBER_ID,
        whatsapp_verify_token=VERIFY_TOKEN,
        whatsapp_app_secret=APP_SECRET,
        db_path=str(tmp_path / "test.db"),
        rate_limit_max=100,
        rate_limit_window_seconds=600,
    )


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def store(settings: Settings) -> Iterator[Store]:
    s = Store(settings.db_path)
    yield s
    s.close()


@pytest.fixture
def assistant(settings: Settings, store: Store, fake_llm: FakeLLM) -> Assistant:
    return Assistant(settings, store, fake_llm)


@pytest.fixture
def graph() -> Iterator[respx.MockRouter]:
    """Mock every Graph API call; individual tests add routes for media downloads."""
    with respx.mock(assert_all_called=False) as router:
        router.post(f"{GRAPH}/{PHONE_NUMBER_ID}/messages").respond(
            200, json={"messages": [{"id": "wamid.out"}]}
        )
        yield router


@pytest.fixture
def client(settings: Settings, fake_llm: FakeLLM, graph: respx.MockRouter) -> Iterator[TestClient]:
    app = create_app(settings, llm=fake_llm)
    with TestClient(app) as c:
        yield c


# ------------------------------------------------------------------------- helpers
def sign(body: bytes, secret: str = APP_SECRET) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def webhook_payload(*messages: dict[str, Any], statuses: list[dict] | None = None) -> dict:
    value: dict[str, Any] = {
        "messaging_product": "whatsapp",
        "metadata": {"display_phone_number": "15550000000", "phone_number_id": PHONE_NUMBER_ID},
    }
    if messages:
        value["contacts"] = [{"profile": {"name": "Tester"}, "wa_id": messages[0]["from"]}]
        value["messages"] = list(messages)
    if statuses:
        value["statuses"] = statuses
    return {
        "object": "whatsapp_business_account",
        "entry": [{"id": "WABA_ID", "changes": [{"field": "messages", "value": value}]}],
    }


def text_message(body: str, msg_id: str = "wamid.1", sender: str = "8801700000000") -> dict:
    return {
        "from": sender,
        "id": msg_id,
        "timestamp": "1700000000",
        "type": "text",
        "text": {"body": body},
    }


def post_webhook(client: TestClient, payload: dict, signature: str | None = "auto"):
    body = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if signature == "auto":
        headers["X-Hub-Signature-256"] = sign(body)
    elif signature is not None:
        headers["X-Hub-Signature-256"] = signature
    return client.post("/webhook", content=body, headers=headers)


def sent_texts(graph: respx.MockRouter) -> list[str]:
    """Bodies of all text messages sent to the Graph /messages endpoint."""
    texts = []
    for call in graph.calls:
        if call.request.method == "POST" and call.request.url.path.endswith("/messages"):
            payload = json.loads(call.request.content)
            if payload.get("type") == "text":
                texts.append(payload["text"]["body"])
    return texts
