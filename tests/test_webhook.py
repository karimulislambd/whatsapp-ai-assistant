"""WhatsApp webhook: verification handshake, signature checks, routing and idempotency."""

import json

import httpx

from tests.conftest import (
    GRAPH,
    PHONE_NUMBER_ID,
    VERIFY_TOKEN,
    post_webhook,
    sent_texts,
    sign,
    text_message,
    webhook_payload,
)


# ------------------------------------------------------------------ verification handshake
def test_verification_succeeds_with_correct_token(client):
    resp = client.get(
        "/webhook",
        params={"hub.mode": "subscribe", "hub.verify_token": VERIFY_TOKEN, "hub.challenge": "42"},
    )
    assert resp.status_code == 200
    assert resp.text == "42"


def test_verification_fails_with_wrong_token(client):
    resp = client.get(
        "/webhook",
        params={"hub.mode": "subscribe", "hub.verify_token": "nope", "hub.challenge": "42"},
    )
    assert resp.status_code == 403


def test_verification_fails_with_wrong_mode(client):
    resp = client.get(
        "/webhook",
        params={"hub.mode": "unsubscribe", "hub.verify_token": VERIFY_TOKEN, "hub.challenge": "1"},
    )
    assert resp.status_code == 403


# ------------------------------------------------------------------------ signatures
def test_valid_signature_is_accepted_and_replied(client, graph, fake_llm):
    resp = post_webhook(client, webhook_payload(text_message("Hi there")))
    assert resp.status_code == 200
    assert sent_texts(graph) == ["fake chat reply"]
    assert fake_llm.chat_calls[0][-1] == {"role": "user", "content": "Hi there"}


def test_invalid_signature_is_rejected(client, graph, fake_llm):
    resp = post_webhook(client, webhook_payload(text_message("Hi")), signature="sha256=deadbeef")
    assert resp.status_code == 403
    assert fake_llm.chat_calls == []
    assert not graph.calls


def test_missing_signature_is_rejected(client, fake_llm):
    resp = post_webhook(client, webhook_payload(text_message("Hi")), signature=None)
    assert resp.status_code == 403
    assert fake_llm.chat_calls == []


def test_signature_over_tampered_body_is_rejected(client, fake_llm):
    original = json.dumps(webhook_payload(text_message("Hi"))).encode()
    tampered = original.replace(b"Hi", b"Yo")
    resp = client.post(
        "/webhook",
        content=tampered,
        headers={"X-Hub-Signature-256": sign(original), "Content-Type": "application/json"},
    )
    assert resp.status_code == 403
    assert fake_llm.chat_calls == []


# ------------------------------------------------------------------- idempotency etc.
def test_duplicate_deliveries_are_processed_once(client, graph, fake_llm):
    payload = webhook_payload(text_message("Hello", msg_id="wamid.dup"))
    assert post_webhook(client, payload).status_code == 200
    assert post_webhook(client, payload).status_code == 200
    assert len(fake_llm.chat_calls) == 1
    assert sent_texts(graph) == ["fake chat reply"]


def test_status_updates_are_ignored(client, graph, fake_llm):
    payload = webhook_payload(
        statuses=[{"id": "wamid.out", "status": "delivered", "recipient_id": "8801700000000"}]
    )
    assert post_webhook(client, payload).status_code == 200
    assert fake_llm.chat_calls == []
    assert not graph.calls


def test_incoming_message_is_marked_read(client, graph):
    post_webhook(client, webhook_payload(text_message("Hi", msg_id="wamid.read")))
    read_calls = [
        json.loads(c.request.content)
        for c in graph.calls
        if json.loads(c.request.content).get("status") == "read"
    ]
    assert read_calls and read_calls[0]["message_id"] == "wamid.read"


def test_mark_read_failure_does_not_block_reply(client, graph):
    def respond(request: httpx.Request) -> httpx.Response:
        is_read = json.loads(request.content).get("status") == "read"
        return httpx.Response(500 if is_read else 200, json={})

    graph.post(f"{GRAPH}/{PHONE_NUMBER_ID}/messages").mock(side_effect=respond)
    post_webhook(client, webhook_payload(text_message("Hi", msg_id="wamid.x")))
    assert sent_texts(graph) == ["fake chat reply"]


def test_image_message_downloads_media_and_replies(client, graph, fake_llm):
    graph.get(f"{GRAPH}/MEDIA123").respond(
        200,
        json={"url": "https://lookaside.fbsbx.com/media/abc", "mime_type": "image/jpeg"},
    )
    graph.get("https://lookaside.fbsbx.com/media/abc").respond(200, content=b"\xff\xd8jpegbytes")
    msg = {
        "from": "8801700000000",
        "id": "wamid.img",
        "type": "image",
        "image": {"id": "MEDIA123", "mime_type": "image/jpeg", "caption": "What is this?"},
    }
    post_webhook(client, webhook_payload(msg))
    assert fake_llm.vision_calls[0]["image"] == b"\xff\xd8jpegbytes"
    assert fake_llm.vision_calls[0]["prompt"] == "What is this?"
    assert sent_texts(graph) == ["fake image analysis"]


def test_voice_note_is_transcribed_then_answered(client, graph, fake_llm):
    graph.get(f"{GRAPH}/AUDIO1").respond(
        200, json={"url": "https://lookaside.fbsbx.com/audio/1", "mime_type": "audio/ogg"}
    )
    graph.get("https://lookaside.fbsbx.com/audio/1").respond(200, content=b"OggS...")
    msg = {
        "from": "8801700000000",
        "id": "wamid.voice",
        "type": "audio",
        "audio": {"id": "AUDIO1", "mime_type": "audio/ogg; codecs=opus", "voice": True},
    }
    post_webhook(client, webhook_payload(msg))
    assert fake_llm.transcribe_calls[0]["filename"] == "audio.ogg"
    texts = sent_texts(graph)
    assert texts[0].startswith("🎙️ *Transcript:* hello from a voice note")
    assert texts[1] == "fake chat reply"


def test_media_download_failure_sends_friendly_error(client, graph, fake_llm):
    graph.get(f"{GRAPH}/BROKEN").respond(404)
    msg = {
        "from": "8801700000000",
        "id": "wamid.broken",
        "type": "image",
        "image": {"id": "BROKEN", "mime_type": "image/jpeg"},
    }
    post_webhook(client, webhook_payload(msg))
    assert fake_llm.vision_calls == []
    assert "couldn't download" in sent_texts(graph)[0]


def test_unsupported_type_gets_friendly_message(client, graph, fake_llm):
    msg = {"from": "8801700000000", "id": "wamid.loc", "type": "location", "location": {}}
    post_webhook(client, webhook_payload(msg))
    assert fake_llm.chat_calls == []
    assert "can't handle location messages" in sent_texts(graph)[0]


def test_long_replies_are_split_into_4096_char_messages(client, graph, fake_llm):
    fake_llm.chat_reply = ("word " * 2000).strip()  # ~10k chars
    post_webhook(client, webhook_payload(text_message("long please", msg_id="wamid.long")))
    texts = sent_texts(graph)
    assert len(texts) == 3
    assert all(len(t) <= 4096 for t in texts)


def test_invalid_json_with_valid_signature_returns_400(client):
    body = b"not json"
    resp = client.post("/webhook", content=body, headers={"X-Hub-Signature-256": sign(body)})
    assert resp.status_code == 400
