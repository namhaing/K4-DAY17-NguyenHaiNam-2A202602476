from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig, normalize_provider

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "custom": "gpt-4o-mini",
    "gemini": "gemini-2.0-flash",
    "anthropic": "claude-haiku-4-5-20251001",
    "ollama": "llama3.1",
    "openrouter": "openai/gpt-4o-mini",
}

API_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "custom": "CUSTOM_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "ollama": None,
    "openrouter": "OPENROUTER_API_KEY",
}

DEFAULT_COMPACT_THRESHOLD_TOKENS = 800
DEFAULT_COMPACT_KEEP_MESSAGES = 4


@dataclass
class LabConfig:
    """Shared configuration for the lab: paths, compact-memory knobs, and model providers."""

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _api_key_for(provider: str) -> str | None:
    env_name = API_KEY_ENV.get(provider)
    if env_name is None:
        return None
    if provider == "gemini":
        return _env("GEMINI_API_KEY") or _env("GOOGLE_API_KEY")
    return _env(env_name)


def _base_url_for(provider: str) -> str | None:
    if provider == "custom":
        return _env("CUSTOM_BASE_URL")
    if provider == "ollama":
        return _env("OLLAMA_BASE_URL", "http://localhost:11434")
    if provider == "openrouter":
        return _env("OPENROUTER_BASE_URL")
    return None


def _provider_config(prefix: str, fallback_provider: str, fallback_model: str | None = None) -> ProviderConfig:
    provider = normalize_provider(_env(f"{prefix}_PROVIDER", fallback_provider))
    model_name = _env(f"{prefix}_MODEL", fallback_model or DEFAULT_MODELS[provider])
    return ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=_env_float(f"{prefix}_TEMPERATURE", 0.0),
        api_key=_api_key_for(provider),
        base_url=_base_url_for(provider),
    )


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load `.env` (if present) and return a populated LabConfig."""

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()

    try:
        from dotenv import load_dotenv

        load_dotenv(root / ".env")
    except ImportError:
        pass

    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    model = _provider_config("LLM", "openai")
    # The judge defaults to the main provider/model unless overridden.
    judge_provider = normalize_provider(_env("JUDGE_PROVIDER", model.provider))
    judge_model = _provider_config(
        "JUDGE",
        judge_provider,
        model.model_name if judge_provider == model.provider else None,
    )

    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=state_dir,
        compact_threshold_tokens=_env_int("COMPACT_THRESHOLD_TOKENS", DEFAULT_COMPACT_THRESHOLD_TOKENS),
        compact_keep_messages=_env_int("COMPACT_KEEP_MESSAGES", DEFAULT_COMPACT_KEEP_MESSAGES),
        model=model,
        judge_model=judge_model,
    )
