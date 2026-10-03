from __future__ import annotations

from dataclasses import dataclass

SUPPORTED_PROVIDERS = ("openai", "custom", "gemini", "anthropic", "ollama", "openrouter")

_PROVIDER_ALIASES = {
    "anthorpic": "anthropic",
    "claude": "anthropic",
    "google": "gemini",
    "google_genai": "gemini",
    "google-genai": "gemini",
    "openai_compatible": "custom",
    "openai-compatible": "custom",
    "open_router": "openrouter",
    "open-router": "openrouter",
}

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass
class ProviderConfig:
    """Provider configuration shared by the agents.

    Supported providers: openai, custom (OpenAI-compatible base URL), gemini,
    anthropic, ollama, openrouter.
    """

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None

    def is_live_ready(self) -> bool:
        """True when this provider has enough settings to call a real model."""

        if self.provider == "ollama":
            return bool(self.model_name)
        if self.provider == "custom":
            return bool(self.base_url and self.model_name)
        return bool(self.api_key and self.model_name)


def normalize_provider(value: str) -> str:
    """Map raw provider names and common aliases (e.g. `anthorpic`) to a supported provider."""

    provider = (value or "").strip().lower()
    provider = _PROVIDER_ALIASES.get(provider, provider)
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(f"Unsupported provider {value!r}. Expected one of: {', '.join(SUPPORTED_PROVIDERS)}")
    return provider


def build_chat_model(config: ProviderConfig):
    """Instantiate the real chat model for the selected provider.

    Imports are lazy so offline mode works without every provider SDK installed.
    """

    provider = normalize_provider(config.provider)

    if provider in ("openai", "custom"):
        from langchain_openai import ChatOpenAI

        kwargs = {"model": config.model_name, "temperature": config.temperature, "api_key": config.api_key}
        if provider == "custom":
            kwargs["base_url"] = config.base_url
            kwargs["api_key"] = config.api_key or "not-needed"
        return ChatOpenAI(**kwargs)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=config.model_name,
            temperature=config.temperature,
            google_api_key=config.api_key,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=config.model_name, temperature=config.temperature, api_key=config.api_key)

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        kwargs = {"model": config.model_name, "temperature": config.temperature}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatOllama(**kwargs)

    # openrouter
    try:
        from langchain_openrouter import ChatOpenRouter

        return ChatOpenRouter(
            model_name=config.model_name,
            temperature=config.temperature,
            openrouter_api_key=config.api_key,
        )
    except ImportError:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=config.api_key,
            base_url=config.base_url or OPENROUTER_BASE_URL,
        )
