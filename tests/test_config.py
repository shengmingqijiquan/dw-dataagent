"""配置加载测试。"""
import os
import pytest
import yaml
from nl2insight.config import ConfigError, load_config, Settings


def test_load_defaults_when_file_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings = load_config("nonexistent.yaml")
    assert isinstance(settings, Settings)
    assert settings.executor.default == "duckdb"
    assert settings.milvus.port == 19530


def test_load_yaml_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = {"executor": {"default": "starrocks"},
           "services": {"milvus": {"host": "milvus-host", "port": 19531}}}
    with open("config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    settings = load_config("config.yaml")
    assert settings.executor.default == "starrocks"
    assert settings.milvus.host == "milvus-host"
    assert settings.milvus.port == 19531


def test_env_overrides_executor(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DW_EXECUTOR", "duckdb")
    with open("config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"executor": {"default": "starrocks"}}, f)
    assert load_config("config.yaml").executor.default == "duckdb"


def test_malformed_yaml_raises_config_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yaml").write_text("llm: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="config.yaml"):
        load_config("config.yaml")
