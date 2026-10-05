"""Graph API client, signature helper, payload parsing and message splitting."""

import hashlib
import hmac
import json

import httpx
import pytest
import respx

from app.whatsapp import (
    MAX_TEXT_LENGTH,
    WhatsAppClient,
    WhatsAppError,
    parse_webhook,
    split_message,
    verify_signature,
)
from tests.conftest import GRAPH, PHONE_NUMBER_ID, text_message, webhook_payload


def _sig(body: bytes, secret: str = "s3cret") -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


# ------------------------------------------------------------------------ signature
def test_verify_signature_valid():
    assert verify_signature("s3cret", b'{"a":1}', _sig(b'{"a":1}'))


@pytest.mark.parametrize(
    "header",
    [None, "", "sha256=", "sha1=abc", "sha256=" + "0" * 64, _sig(b'{"a":1}', "wrong")],
)
def test_verify_signature_invalid(header):
    assert not verify_signature("s3cret", b'{"a":1}', header)


def test_verify_signature_requires_secret():
    assert not verify_signature("", b"{}", _sig(b"{}", ""))


# -------------------------------------------------------------------------- parsing
def test_parse_text_image_audio_and_interactive():
    payload = webhook_payload(
        text_message("hi", msg_id="m1"),
        {"from": "1", "id": "m2", "type": "image", "image": {"id": "I", "caption": "cap"}},
        {"from": "1", "id": "m3", "type": "audio", "audio": {"id": "A", "mime_type": "audio/ogg"}},
        {
            "from": "1",
            "id": "m4",
            "type": "interactive",
            "interactive": {"type": "button_reply", "button_reply": {"id": "b", "title": "Yes"}},
        },
        {"from": "1", "id": "m5", "type": "reaction", "reaction": {"emoji": "👍"}},
        {"from": "1", "id": "m6", "type": "sticker", "sticker": {"id": "S"}},
    )
    msgs = parse_webhook(payload)
    assert [(m.id, m.type) for m in msgs] == [
        ("m1", "text"),
        ("m2", "image"),
        ("m3", "audio"),
        ("m4", "text"),
        ("m6", "sticker"),
    ]
    assert msgs[1].media_id == "I" and msgs[1].caption == "cap"
    assert msgs[3].text == "Yes"


def test_parse_ignores_statuses_and_other_fields():
    payload = webhook_payload(statuses=[{"id": "x", "status": "read"}])
    payload["entry"][0]["changes"].append({"field": "account_update", "value": {}})
    assert parse_webhook(payload) == []
    assert parse_webhook({}) == []


# ------------------------------------------------------------------------ splitting
def test_split_short_message_is_untouched():
    assert split_message("hello") == ["hello"]
    assert split_message("   ") == []


def test_split_prefers_paragraph_boundaries():
    para = "a" * 3000
    chunks = split_message(f"{para}\n\n{para}")
    assert chunks == [para, para]


def test_split_hard_cuts_when_no_whitespace():
    chunks = split_message("x" * (MAX_TEXT_LENGTH * 2 + 10))
    assert [len(c) for c in chunks] == [MAX_TEXT_LENGTH, MAX_TEXT_LENGTH, 10]


def test_split_preserves_all_words():
    text = " ".join(f"w{i}" for i in range(3000))
    chunks = split_message(text, limit=500)
    assert all(len(c) <= 500 for c in chunks)
    assert " ".join(chunks).split() == text.split()


# --------------------------------------------------------------------------- client
@pytest.fixture
async def wa():
    async with httpx.AsyncClient() as http:
        yield WhatsAppClient(http, "tok", PHONE_NUMBER_ID, GRAPH)


@respx.mock
async def test_send_text_posts_expected_payload(wa):
    route = respx.post(f"{GRAPH}/{PHONE_NUMBER_ID}/messages").respond(200, json={})
    await wa.send_text("8801700000000", "Hello!")
    request = route.calls.last.request
    assert request.headers["Authorization"] == "Bearer tok"
    assert json.loads(request.content) == {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": "8801700000000",
        "type": "text",
        "text": {"preview_url": False, "body": "Hello!"},
    }


@respx.mock
async def test_send_text_splits_long_messages(wa):
    route = respx.post(f"{GRAPH}/{PHONE_NUMBER_ID}/messages").respond(200, json={})
    await wa.send_text("1", "y" * 9000)
    assert route.call_count == 3


@respx.mock
async def test_send_text_raises_on_graph_error(wa):
    respx.post(f"{GRAPH}/{PHONE_NUMBER_ID}/messages").respond(401, json={"error": {}})
    with pytest.raises(WhatsAppError):
        await wa.send_text("1", "hi")


@respx.mock
async def test_mark_read_payload(wa):
    route = respx.post(f"{GRAPH}/{PHONE_NUMBER_ID}/messages").respond(200, json={})
    await wa.mark_read("wamid.9")
    body = json.loads(route.calls.last.request.content)
    assert body["status"] == "read" and body["message_id"] == "wamid.9"


@respx.mock
async def test_download_media_two_step(wa):
    respx.get(f"{GRAPH}/MID").respond(
        200, json={"url": "https://lookaside.fbsbx.com/x", "mime_type": "audio/ogg; codecs=opus"}
    )
    file_route = respx.get("https://lookaside.fbsbx.com/x").respond(200, content=b"bytes")
    data, mime = await wa.download_media("MID")
    assert data == b"bytes" and mime == "audio/ogg"
    assert file_route.calls.last.request.headers["Authorization"] == "Bearer tok"


@respx.mock
async def test_download_media_errors(wa):
    respx.get(f"{GRAPH}/BAD").respond(200, json={"no_url": True})
    with pytest.raises(WhatsAppError):
        await wa.download_media("BAD")
    respx.get(f"{GRAPH}/NET").mock(side_effect=httpx.ConnectError("down"))
    with pytest.raises(WhatsAppError):
        await wa.download_media("NET")
