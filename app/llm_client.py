"""Provider selection for OpenAI-compatible chat completion APIs."""

import os
from collections.abc import Mapping
from dataclasses import dataclass

from groq import Groq
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from openai import OpenAI


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    api_key: str
    model: str
    base_url: str | None = None


def settings_from_env(environ: Mapping[str, str] | None = None) -> LLMSettings:
    env = os.environ if environ is None else environ
    provider = env.get("LLM_PROVIDER", "groq").strip().lower()

    if provider == "groq":
        api_key = env.get("GROQ_API_KEY", "").strip()
        model = env.get("GROQ_MODEL", "qwen/qwen3.8-27b").strip()
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is required when LLM_PROVIDER=groq")
        if not model:
            raise RuntimeError("GROQ_MODEL cannot be empty")
        return LLMSettings(provider=provider, api_key=api_key, model=model)

    if provider == "deepseek":
        api_key = env.get("DEEPSEEK_API_KEY", "").strip()
        model = env.get("DEEPSEEK_MODEL", "").strip()
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is required when LLM_PROVIDER=deepseek")
        if not model:
            raise RuntimeError("DEEPSEEK_MODEL is required when LLM_PROVIDER=deepseek")
        return LLMSettings(
            provider=provider,
            api_key=api_key,
            model=model,
            base_url="https://api.deepseek.com",
        )

    raise RuntimeError(f"Unsupported LLM_PROVIDER {provider!r}; use 'groq' or 'deepseek'")


def create_llm_client(settings: LLMSettings):
    if settings.provider == "groq":
        return Groq(api_key=settings.api_key)
    if settings.provider == "deepseek":
        return OpenAI(api_key=settings.api_key, base_url=settings.base_url)
    raise ValueError(f"Unsupported LLM provider: {settings.provider}")


def create_langchain_llm(settings: LLMSettings):
    if settings.provider == "groq":
        return ChatGroq(
            model=settings.model,
            api_key=settings.api_key,
            request_timeout=120,
            max_tokens=2048,
        )
    if settings.provider == "deepseek":
        return ChatOpenAI(
            model=settings.model,
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=120,
            max_tokens=2048,
        )
    raise ValueError(f"Unsupported LLM provider: {settings.provider}")
