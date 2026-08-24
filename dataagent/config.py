"""配置加载：config.yaml + 环境变量覆盖。"""
from dataclasses import dataclass, field
from pathlib import Path
import os
import yaml


class ConfigError(Exception):
    """配置加载失败。"""


@dataclass
class ProviderConfig:
    type: str                    # api | local
    base_url: str = ""
    model: str = ""
    api_key_env: str = ""


@dataclass
class LLMConfig:
    providers: dict[str, ProviderConfig] = field(default_factory=dict)
    routing: dict[str, str] = field(default_factory=dict)


@dataclass
class MilvusConfig:
    host: str = "localhost"
    port: int = 19530
    uri: str = ""
    collection: str = "dw_sql_cases"


@dataclass
class StarRocksConfig:
    host: str = "localhost"
    port: int = 9030
    user: str = "root"
    password: str = ""


@dataclass
class ExecutorConfig:
    default: str = "duckdb"      # duckdb | starrocks
    starrocks: StarRocksConfig = field(default_factory=StarRocksConfig)


@dataclass
class Settings:
    llm: LLMConfig = field(default_factory=LLMConfig)
    mcp_server_url: str = "http://localhost:8001/mcp/sse"
    milvus: MilvusConfig = field(default_factory=MilvusConfig)
    executor: ExecutorConfig = field(default_factory=ExecutorConfig)
    warehouse_path: str = "data/warehouse.duckdb"
    metadata_dir: str = "data/metadata"
    cases_path: str = "data/cases/cases.yaml"


def load_config(path: str = "config.yaml") -> Settings:
    raw = {}
    if Path(path).exists():
        with open(path, encoding="utf-8") as f:
            try:
                raw = yaml.safe_load(f) or {}
            except yaml.YAMLError as e:
                raise ConfigError(f"配置文件解析失败: {path}: {e}") from e

    llm_raw = raw.get("llm", {})
    providers = {
        name: ProviderConfig(**cfg)
        for name, cfg in llm_raw.get("providers", {}).items()
    }
    exec_raw = raw.get("executor", {}) or {}
    sr_raw = exec_raw.get("starrocks", {}) or {}
    milvus_raw = (raw.get("services") or {}).get("milvus", {}) or {}

    settings = Settings(
        llm=LLMConfig(providers=providers, routing=llm_raw.get("routing", {})),
        mcp_server_url=(raw.get("services") or {}).get("mcp_server_url",
                         "http://localhost:8001/mcp/sse"),
        milvus=MilvusConfig(**milvus_raw),
        executor=ExecutorConfig(
            default=exec_raw.get("default", "duckdb"),
            starrocks=StarRocksConfig(**sr_raw),
        ),
    )
    # 环境变量覆盖（模型 key 按需读取，不在此校验存在性）
    env_executor = os.environ.get("DW_EXECUTOR")
    if env_executor in ("duckdb", "starrocks"):
        settings.executor.default = env_executor
    return settings
