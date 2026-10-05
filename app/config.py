"""Application settings, loaded from environment variables (and an optional .env file)."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Groq
    groq_api_key: str = ""
    chat_model: str = "openai/gpt-oss-120b"
    vision_model: str = "qwen/qwen3.8-27b"
    whisper_model: str = "whisper-large-v3"
    llm_timeout_seconds: float = 60.0

    # WhatsApp Cloud API
    whatsapp_token: str = ""
    whatsapp_phone_number_id: str = ""
    whatsapp_verify_token: str = ""
    whatsapp_app_secret: str = ""
    graph_api_version: str = "v21.0"
    graph_api_base_url: str = "https://graph.facebook.com"

    # Storage & behaviour
    db_path: str = "data/assistant.db"
    profile_path: str = str(ROOT_DIR / "data" / "profile.md")
    memory_turns: int = 10
    rate_limit_max: int = 20
    rate_limit_window_seconds: int = 600
    max_media_bytes: int = 4 * 1024 * 1024  # Groq's limit for base64-encoded images
    max_audio_bytes: int = 16 * 1024 * 1024  # WhatsApp's audio size limit
    log_level: str = "INFO"

    @property
    def graph_url(self) -> str:
        return f"{self.graph_api_base_url.rstrip('/')}/{self.graph_api_version}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
