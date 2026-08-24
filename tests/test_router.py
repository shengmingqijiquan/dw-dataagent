"""模型路由测试（不真正调 API，只验证路由逻辑与降级）。"""
import yaml
import pytest
from dataagent.config import load_config
from dataagent.llm.router import LLMRouter


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = {
        "llm": {
            "providers": {
                "deepseek": {"type": "api", "base_url": "https://api.deepseek.com",
                             "model": "deepseek-chat", "api_key_env": "DEEPSEEK_API_KEY"},
                "ollama_local": {"type": "local",
                                 "base_url": "http://localhost:11434",
                                 "model": "qwen3:8b"},
            },
            "routing": {
                "task_parse": "deepseek",
                "sql_generate": "deepseek",
                "critic_review": "deepseek",
                "fallback": "ollama_local",
            },
        }
    }
    with open("config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    return load_config("config.yaml")


def test_get_model_returns_chat_model(settings):
    from langchain_core.language_models import BaseChatModel
    router = LLMRouter(settings)
    model = router.get_model("sql_generate")
    assert isinstance(model, BaseChatModel)


def test_unknown_task_uses_sql_generate(settings):
    router = LLMRouter(settings)
    assert router.get_model("unknown_task") is not None


def test_fallback_when_primary_missing(settings, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY")
    router = LLMRouter(settings)
    model = router.get_model_with_fallback("sql_generate")
    assert model.model == "qwen3:8b"
