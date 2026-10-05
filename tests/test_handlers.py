"""Transport-agnostic core: routing by message type, commands, modes and error handling."""

from app.config import Settings
from app.handlers import (
    APOLOGY,
    MODE_ABOUT,
    MODE_CHAT,
    Assistant,
    IncomingMessage,
    Media,
)
from app.memory import Store, hash_user_id

USER = "wa:8801700000000"


def text(body: str, user: str = USER) -> IncomingMessage:
    return IncomingMessage(user_id=user, kind="text", text=body)


def media_msg(kind, data: bytes, mime: str, caption: str | None = None) -> IncomingMessage:
    async def load() -> Media:
        return Media(data=data, mime_type=mime)

    return IncomingMessage(user_id=USER, kind=kind, text=caption, load_media=load)


# ------------------------------------------------------------------------- routing
async def test_text_routes_to_chat_with_system_prompt(assistant, fake_llm):
    result = await assistant.handle(text("What is FastAPI?"))
    assert result.replies == ["fake chat reply"]
    assert result.mode == MODE_CHAT
    messages = fake_llm.chat_calls[0]
    assert messages[0]["role"] == "system"
    assert messages[-1] == {"role": "user", "content": "What is FastAPI?"}


async def test_text_includes_previous_turns(assistant, fake_llm):
    await assistant.handle(text("My name is Rafi"))
    await assistant.handle(text("What is my name?"))
    history = fake_llm.chat_calls[1][1:-1]
    assert history == [
        {"role": "user", "content": "My name is Rafi"},
        {"role": "assistant", "content": "fake chat reply"},
    ]


async def test_image_routes_to_vision_with_caption(assistant, fake_llm):
    result = await assistant.handle(media_msg("image", b"img", "image/png", "Read the sign"))
    assert result.replies == ["fake image analysis"]
    call = fake_llm.vision_calls[0]
    assert call == {"image": b"img", "mime_type": "image/png", "prompt": "Read the sign"}
    assert fake_llm.chat_calls == []


async def test_image_without_caption_uses_default_prompt(assistant, fake_llm):
    await assistant.handle(media_msg("image", b"img", "image/jpeg"))
    assert "Describe this image" in fake_llm.vision_calls[0]["prompt"]


async def test_image_turn_is_remembered_for_follow_ups(assistant, fake_llm):
    await assistant.handle(media_msg("image", b"img", "image/jpeg", "What breed is this dog?"))
    await assistant.handle(text("How big does it get?"))
    history = fake_llm.chat_calls[0][1:-1]
    assert history[0]["content"] == "[Sent an image] What breed is this dog?"
    assert history[1]["content"] == "fake image analysis"


async def test_oversized_image_is_rejected(settings, store, fake_llm):
    settings.max_media_bytes = 10
    bot = Assistant(settings, store, fake_llm)
    result = await bot.handle(media_msg("image", b"x" * 11, "image/jpeg"))
    assert "too large" in result.replies[0]
    assert fake_llm.vision_calls == []


async def test_audio_is_transcribed_then_answered(assistant, fake_llm):
    result = await assistant.handle(media_msg("audio", b"OggS", "audio/ogg"))
    assert result.replies == ["🎙️ *Transcript:* hello from a voice note", "fake chat reply"]
    assert fake_llm.transcribe_calls[0]["filename"] == "audio.ogg"
    assert fake_llm.chat_calls[0][-1]["content"] == "hello from a voice note"


async def test_empty_transcript_gets_friendly_reply(assistant, fake_llm):
    fake_llm.transcript = ""
    result = await assistant.handle(media_msg("audio", b"OggS", "audio/webm"))
    assert "couldn't make out any speech" in result.replies[0]
    assert fake_llm.chat_calls == []


async def test_unsupported_type(assistant, fake_llm):
    msg = IncomingMessage(user_id=USER, kind="unsupported", raw_type="sticker")
    result = await assistant.handle(msg)
    assert "can't handle sticker messages" in result.replies[0]
    assert fake_llm.chat_calls == []


async def test_llm_failure_returns_apology(assistant, fake_llm):
    fake_llm.fail = True
    result = await assistant.handle(text("hello?"))
    assert result.replies == [APOLOGY]


async def test_failed_turn_is_not_saved_to_memory(assistant, fake_llm, store):
    fake_llm.fail = True
    await assistant.handle(text("lost message"))
    assert store.get_history(hash_user_id(USER), MODE_CHAT, 10) == []


# ------------------------------------------------------------------------ commands
async def test_help_command(assistant, fake_llm):
    result = await assistant.handle(text("/help"))
    assert "/about" in result.replies[0] and "/reset" in result.replies[0]
    assert fake_llm.chat_calls == []


async def test_reset_clears_memory(assistant, fake_llm, store):
    await assistant.handle(text("remember this"))
    assert store.get_history(hash_user_id(USER), MODE_CHAT, 10)
    result = await assistant.handle(text("/reset"))
    assert "cleared" in result.replies[0].lower()
    assert store.get_history(hash_user_id(USER), MODE_CHAT, 10) == []


async def test_about_mode_uses_profile_only_prompt(assistant, fake_llm):
    result = await assistant.handle(text("/about"))
    assert result.mode == MODE_ABOUT
    result = await assistant.handle(text("Where does he study?"))
    assert result.mode == MODE_ABOUT
    system = fake_llm.chat_calls[0][0]["content"]
    assert "<profile>" in system and "Varendra University" in system
    assert "ONLY" in system


async def test_chat_command_switches_back(assistant, fake_llm):
    await assistant.handle(text("/about"))
    result = await assistant.handle(text("/chat"))
    assert result.mode == MODE_CHAT
    await assistant.handle(text("hi"))
    assert "<profile>" not in fake_llm.chat_calls[0][0]["content"]


async def test_modes_have_separate_histories(assistant, fake_llm):
    await assistant.handle(text("general question"))
    await assistant.handle(text("/about"))
    await assistant.handle(text("about question"))
    about_history = fake_llm.chat_calls[1][1:-1]
    assert about_history == []


async def test_mode_is_per_user(assistant):
    await assistant.handle(text("/about", user="wa:111"))
    result = await assistant.handle(text("/help", user="wa:222"))
    assert result.mode == MODE_CHAT


async def test_unknown_command(assistant, fake_llm):
    result = await assistant.handle(text("/dance"))
    assert "Unknown command" in result.replies[0]
    assert fake_llm.chat_calls == []


async def test_commands_are_case_insensitive(assistant):
    result = await assistant.handle(text("/ABOUT"))
    assert result.mode == MODE_ABOUT


async def test_missing_profile_file_falls_back(tmp_path, store, fake_llm):
    settings = Settings(_env_file=None, profile_path=str(tmp_path / "missing.md"))
    bot = Assistant(settings, store, fake_llm)
    await bot.handle(text("/about"))
    await bot.handle(text("who is he?"))
    assert "Profile unavailable" in fake_llm.chat_calls[0][0]["content"]


# ---------------------------------------------------------------------- rate limit
async def test_rate_limit_blocks_after_max_messages(tmp_path, fake_llm):
    settings = Settings(
        _env_file=None,
        db_path=str(tmp_path / "rl.db"),
        rate_limit_max=3,
        rate_limit_window_seconds=600,
    )
    store = Store(settings.db_path)
    bot = Assistant(settings, store, fake_llm)
    for _ in range(3):
        assert (await bot.handle(text("hi"))).replies == ["fake chat reply"]
    blocked = await bot.handle(text("hi again"))
    assert "limit of 3 messages per 10 minutes" in blocked.replies[0]
    assert len(fake_llm.chat_calls) == 3
    # Other users are unaffected.
    assert (await bot.handle(text("hi", user="wa:999"))).replies == ["fake chat reply"]
    store.close()
