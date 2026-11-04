from unittest.mock import MagicMock

import pytest

from app.llm_client import create_llm_client, settings_from_env


def test_groq_is_the_default_provider():
    settings = settings_from_env({"GROQ_API_KEY": "test-key"})

    assert settings.provider == "groq"
    assert settings.model == "qwen/qwen3.8-27b"
    assert settings.base_url is None


def test_deepseek_requires_explicit_model_id():
    with pytest.raises(RuntimeError, match="DEEPSEEK_MODEL is required"):
        settings_from_env({"LLM_PROVIDER": "deepseek", "DEEPSEEK_API_KEY": "test-key"})


def test_deepseek_settings_use_official_api_base_url():
    settings = settings_from_env(
        {
            "LLM_PROVIDER": "deepseek",
            "DEEPSEEK_API_KEY": "test-key",
            "DEEPSEEK_MODEL": "deepseek-chat-test",
        }
    )

    assert settings.provider == "deepseek"
    assert settings.model == "deepseek-chat-test"
    assert settings.base_url == "https://api.deepseek.com"


def test_client_factory_selects_sdk(monkeypatch):
    import app.llm_client as llm

    groq_instance = MagicMock()
    groq_factory = MagicMock(return_value=groq_instance)
    monkeypatch.setattr(llm, "Groq", groq_factory)
    groq_settings = settings_from_env({"GROQ_API_KEY": "test-groq"})

    assert create_llm_client(groq_settings) is groq_instance
    groq_factory.assert_called_once_with(api_key="test-groq")

    openai_instance = MagicMock()
    openai_factory = MagicMock(return_value=openai_instance)
    monkeypatch.setattr(llm, "OpenAI", openai_factory)
    deepseek_settings = settings_from_env(
        {
            "LLM_PROVIDER": "deepseek",
            "DEEPSEEK_API_KEY": "test-deepseek",
            "DEEPSEEK_MODEL": "deepseek-chat-test",
        }
    )

    assert create_llm_client(deepseek_settings) is openai_instance
    openai_factory.assert_called_once_with(
        api_key="test-deepseek", base_url="https://api.deepseek.com"
    )


def test_unknown_provider_is_rejected():
    with pytest.raises(RuntimeError, match="Unsupported LLM_PROVIDER"):
        settings_from_env({"LLM_PROVIDER": "unknown"})
