from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass
class ProviderConfig:
    """Provider configuration shared by the agents."""

    provider: str
    model_name: str
    temperature: float = 0.0
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    """Map provider aliases to canonical names."""
    if not value:
        return "openai"
    val = value.strip().lower()
    mapping = {
        "anthorpic": "anthropic",
        "claude": "anthropic",
        "google": "gemini",
        "google-genai": "gemini",
        "gpt": "openai",
        "openai-compatible": "custom",
    }
    return mapping.get(val, val)


def build_chat_model(config: ProviderConfig) -> Any:
    """Instantiate the real chat model for the selected provider.

    Supported providers:
    - openai -> ChatOpenAI
    - custom -> ChatOpenAI with base_url
    - gemini -> ChatGoogleGenerativeAI
    - anthropic -> ChatAnthropic
    - ollama -> ChatOllama
    - openrouter -> ChatOpenRouter
    """
    provider = normalize_provider(config.provider)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=api_key,
        )

    if provider == "custom":
        from langchain_openai import ChatOpenAI

        api_key = config.api_key or os.getenv("CUSTOM_API_KEY") or "EMPTY"
        base_url = config.base_url or os.getenv("CUSTOM_BASE_URL")
        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=api_key,
            base_url=base_url,
        )

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = config.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        return ChatGoogleGenerativeAI(
            model=config.model_name,
            temperature=config.temperature,
            google_api_key=api_key,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        api_key = config.api_key or os.getenv("ANTHROPIC_API_KEY")
        return ChatAnthropic(
            model=config.model_name,
            temperature=config.temperature,
            api_key=api_key,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        base_url = config.base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(
            model=config.model_name,
            temperature=config.temperature,
            base_url=base_url,
        )

    if provider == "openrouter":
        try:
            from langchain_openrouter import ChatOpenRouter

            api_key = config.api_key or os.getenv("OPENROUTER_API_KEY")
            return ChatOpenRouter(
                model=config.model_name,
                temperature=config.temperature,
                api_key=api_key,
            )
        except ImportError:
            from langchain_openai import ChatOpenAI

            api_key = config.api_key or os.getenv("OPENROUTER_API_KEY")
            return ChatOpenAI(
                model=config.model_name,
                temperature=config.temperature,
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
            )

    raise ValueError(f"Unsupported provider: {config.provider}")
