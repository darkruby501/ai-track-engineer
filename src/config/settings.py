"""Application configuration.

Settings are layered by priority (highest first): constructor arguments,
environment variables, ``.env``, then ``config/settings.yaml``, then hard-coded
defaults. Secrets (tokens) only ever come from the environment, never from the
versioned YAML file.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

# Repository root: <repo>/src/config/settings.py -> parents[2] == <repo>
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_YAML = _REPO_ROOT / "config" / "settings.yaml"


class InfluxSettings(BaseModel):
    """Connection settings for the InfluxDB time-series store."""

    url: str = "http://localhost:8086"
    org: str = "ai-track-engineer"
    bucket: str = "telemetry"
    token: str = Field(default="", repr=False)  # never echoed in logs/reprs


class OllamaSettings(BaseModel):
    """Connection settings for the local Ollama LLM server."""

    url: str = "http://localhost:11434"
    model: str = "mistral"


class CoachSettings(BaseModel):
    """Settings for the Claude-powered driving coach (``coach`` command).

    The Anthropic API key is never configured here: the SDK reads
    ``ANTHROPIC_API_KEY`` (or an ``ant auth login`` profile) from the environment.
    """

    model: str = "claude-opus-5-5"
    # Effort for the between-laps corner calls (latency-sensitive) and for the
    # post-session debrief (quality-sensitive).
    lap_effort: str = "low"
    debrief_effort: str = "high"
    # "standard" or "fast" (fast mode costs 2x but answers sooner).
    speed: str = "standard"
    request_timeout_s: float = 45.0
    # Reference laps live in ``<reference_dir>/<track>__<layout>__<car>.json``.
    reference_dir: str = "data/local/reference"
    # Markdown files folded into the coach's (cached) system prompt.
    context_files: list[str] = Field(
        default_factory=lambda: [
            "docs/tracks/buttonwillow-13cw-coach-brief.md",
            "data/local/driver-brief-bw13.md",
        ]
    )
    sessions_dir: str = "data/sessions"
    voice: bool = True
    voice_rate: int = 2
    # Seconds of warning before the reference braking point for a corner call.
    call_lead_s: float = 4.5
    max_calls_per_lap: int = 2


class AppSettings(BaseModel):
    """General application tunables."""

    log_level: str = "INFO"
    telemetry_poll_hz: int = 60
    dashboard_stream_hz: int = 30
    ac_install_path: str | None = None
    database_path: str = "data/ai-track-engineer.db"


class Settings(BaseSettings):
    """Root settings object.

    Environment variables use a double-underscore delimiter for nested values,
    e.g. ``INFLUXDB__TOKEN`` or ``OLLAMA__MODEL``. Values from the environment
    and ``.env`` override the YAML defaults.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    app: AppSettings = Field(default_factory=AppSettings)
    influxdb: InfluxSettings = Field(default_factory=InfluxSettings)
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    coach: CoachSettings = Field(default_factory=CoachSettings)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Order sources so env/.env override YAML, which overrides defaults."""
        sources: list[PydanticBaseSettingsSource] = [
            init_settings,
            env_settings,
            dotenv_settings,
        ]
        # ``_DEFAULT_YAML`` is read at call time so tests can patch the path.
        if _DEFAULT_YAML.is_file():
            sources.append(YamlConfigSettingsSource(settings_cls, yaml_file=_DEFAULT_YAML))
        sources.append(file_secret_settings)
        return tuple(sources)

    @classmethod
    def load(cls) -> Settings:
        """Build settings from the layered sources."""
        return cls()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached, process-wide :class:`Settings` instance."""
    return Settings.load()
