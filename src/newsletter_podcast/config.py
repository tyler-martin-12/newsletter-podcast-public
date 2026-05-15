from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, HttpUrl, PositiveInt, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ImapConfig(BaseModel):
    host: str
    port: PositiveInt = 993
    username: str
    password_env: str
    folder: str = "Newsletters"
    move_to_folder: str | None = None

    @property
    def password(self) -> str:
        value = os.environ.get(self.password_env)
        if not value:
            raise RuntimeError(f"Missing required environment variable: {self.password_env}")
        return value


class TtsConfig(BaseModel):
    voice: str = "af_bella"
    speed: float = Field(default=1.0, gt=0.25, le=3.0)
    device: str = "cpu"


class OutputConfig(BaseModel):
    episodes_dir: Path
    feed_path: Path
    base_url: HttpUrl
    state_db: Path
    lock_path: Path

    @field_validator("episodes_dir", "feed_path", "state_db", "lock_path", mode="before")
    @classmethod
    def expand_path(cls, value: str | Path) -> Path:
        return Path(value).expanduser()


class FeedConfig(BaseModel):
    title: str
    description: str
    cover_image: Path | None = None
    retention_days: PositiveInt = 30

    @field_validator("cover_image", mode="before")
    @classmethod
    def expand_optional_path(cls, value: str | Path | None) -> Path | None:
        return Path(value).expanduser() if value else None


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(extra="forbid")

    imap: ImapConfig
    tts: TtsConfig
    output: OutputConfig
    feed: FeedConfig


def load_config(path: Path | None = None) -> AppConfig:
    config_path = path or _config_path_from_env()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")
    return AppConfig.model_validate(raw)


def _config_path_from_env() -> Path:
    value = os.environ.get("NEWSLETTER_PODCAST_CONFIG")
    if not value:
        raise RuntimeError("Set NEWSLETTER_PODCAST_CONFIG to the config.yaml path")
    return Path(value).expanduser()
