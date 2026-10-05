"""Web demo API and basic routes — /api/chat must share the core handler with WhatsApp."""

from app import __version__

SESSION = "demo-session-123"


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok" and body["version"] == __version__
    assert body["whatsapp_configured"] is True


def test_root_redirects_to_demo(client):
    resp = client.get("/", follow_redirects=False)
    assert resp.status_code in (302, 307)
    assert resp.headers["location"] == "/demo"


def test_demo_page_is_served(client):
    resp = client.get("/demo")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "/api/chat" in resp.text


def test_api_chat_text(client, fake_llm):
    resp = client.post("/api/chat", data={"session_id": SESSION, "message": "Hello"})
    assert resp.status_code == 200
    assert resp.json() == {"replies": ["fake chat reply"], "mode": "chat"}
    assert fake_llm.chat_calls[0][-1]["content"] == "Hello"


def test_api_chat_shares_commands_modes_and_memory(client, fake_llm):
    assert (
        client.post("/api/chat", data={"session_id": SESSION, "message": "/about"}).json()["mode"]
        == "about"
    )
    resp = client.post("/api/chat", data={"session_id": SESSION, "message": "His GPA?"})
    assert resp.json()["mode"] == "about"
    assert "<profile>" in fake_llm.chat_calls[0][0]["content"]
    # A different session is independent.
    other = client.post("/api/chat", data={"session_id": "another-session", "message": "/help"})
    assert other.json()["mode"] == "chat"


def test_api_chat_image_upload(client, fake_llm):
    resp = client.post(
        "/api/chat",
        data={"session_id": SESSION, "message": "What's in it?"},
        files={"file": ("cat.png", b"\x89PNGdata", "image/png")},
    )
    assert resp.status_code == 200
    assert resp.json()["replies"] == ["fake image analysis"]
    assert fake_llm.vision_calls[0]["prompt"] == "What's in it?"
    assert fake_llm.vision_calls[0]["mime_type"] == "image/png"


def test_api_chat_voice_upload(client, fake_llm):
    resp = client.post(
        "/api/chat",
        data={"session_id": SESSION},
        files={"file": ("voice-note.webm", b"webmdata", "audio/webm")},
    )
    replies = resp.json()["replies"]
    assert replies[0].startswith("🎙️ *Transcript:*")
    assert fake_llm.transcribe_calls[0]["filename"] == "audio.webm"


def test_api_chat_rejects_bad_input(client):
    assert (
        client.post("/api/chat", data={"session_id": "bad id!", "message": "x"}).status_code == 422
    )
    assert (
        client.post("/api/chat", data={"session_id": SESSION, "message": "  "}).status_code == 422
    )
    resp = client.post(
        "/api/chat",
        data={"session_id": SESSION},
        files={"file": ("doc.pdf", b"%PDF", "application/pdf")},
    )
    assert resp.status_code == 415


def test_api_chat_rejects_oversized_upload(client, settings):
    big = b"x" * (settings.max_media_bytes + 1)
    resp = client.post(
        "/api/chat",
        data={"session_id": SESSION},
        files={"file": ("big.jpg", big, "image/jpeg")},
    )
    assert resp.status_code == 413


def test_api_chat_rate_limited(client, settings):
    settings.rate_limit_max = 2
    for _ in range(2):
        client.post("/api/chat", data={"session_id": SESSION, "message": "hi"})
    resp = client.post("/api/chat", data={"session_id": SESSION, "message": "hi"})
    assert "limit" in resp.json()["replies"][0]
