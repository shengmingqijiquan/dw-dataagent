# dw-dataagent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建生产级对标的数仓取数 DataAgent 服务：自然语言需求 → LangGraph 多步推理 → 生成 SQL → 校验 → StarRocks/DuckDB 执行 → 返回可解释结果，含 MCP SSE 服务、Milvus RAG、权限过滤、Langfuse 可观测、Golden Set 评测。

**Architecture:** 三个独立服务进程：MCP Server（FastAPI+SSE，元数据查询+权限过滤）→ Agent Service（FastAPI，LangGraph ReAct + 模型路由 + RAG + 护栏）→ 执行引擎（QueryExecutor 抽象：DuckDB 开发兜底 / StarRocks 生产对齐）。开发全程 TDD：纯逻辑组件 pytest 单测先行，集成组件以可运行脚本验收。

**Tech Stack:** Python 3.10+ / LangGraph / langchain-deepseek / langchain-ollama / MCP Python SDK + langchain-mcp-adapters / pymilvus (Milvus 2.5 standalone) / sentence-transformers (BGE-large-zh) / rank-bm25 / duckdb / starrocks / sqlglot / FastAPI / Langfuse / pytest

**Spec:** [../design.md](../design.md)（v3.0，生产级对标）

## Global Constraints

- Python ≥ 3.10（macOS 16GB，Docker Desktop 可用）
- Milvus standalone 镜像 ≥ 2.5（Docker Compose：etcd+minio+milvus，端口 19530）
- StarRocks allin1 镜像（端口 9030），**按需启动**：16GB 内存下不与 Ollama 同时运行
- LLM 主力 DeepSeek API（`DEEPSEEK_API_KEY` 环境变量）；本地兜底 Ollama `qwen3:8b`（`http://localhost:11434`）
- HuggingFace 下载慢时使用镜像 `HF_ENDPOINT=https://hf-mirror.com`
- BM25 关键词检索在应用层用 rank-bm25 实现（与 Milvus 向量检索 RRF 融合；Milvus 原生 BM25 作为话术提及）
- Rerank 跳过（design.md §5 风险表备案：bge-reranker-v2-m3 太大，RRF 融合够用）
- 所有 SQL 案例/Golden Set 均针对 §schema.py 的 30 张表，表名不得超出注册表
- 数据确定性生成：`random.Random(42)` 固定种子，评测可复现
- 权限规则：无权限的表在元数据检索层完全不可见（含表清单），不是生成后拦截

---

## Day 1 · 骨架与数据底座

### Task 1: 项目骨架与配置

**Files:**
- Create: `dw-dataagent/.gitignore`
- Create: `dw-dataagent/requirements.txt`
- Create: `dw-dataagent/config.yaml`
- Create: `dw-dataagent/dataagent/__init__.py`
- Create: `dw-dataagent/dataagent/config.py`
- Create: `dw-dataagent/tests/__init__.py`
- Create: `dw-dataagent/pytest.ini`
- Test: `dw-dataagent/tests/test_config.py`

**Interfaces:**
- Produces: `dataagent.config.Settings`（dataclass，字段见下）与 `load_config(path="config.yaml") -> Settings`；后续所有模块通过 `load_config()` 读取配置

- [ ] **Step 1: git init + 骨架目录**

```bash
cd /Users/liyu/Curser/职业专题/dw-dataagent
git init
mkdir -p dataagent data/metadata data/cases tests evals scripts deploy docs
```

- [ ] **Step 2: 写 .gitignore**

```gitignore
.venv/
__pycache__/
*.pyc
data/*.duckdb
data/metadata/*.yaml
.pytest_cache/
.env
```

- [ ] **Step 3: 写 requirements.txt**

```
langgraph>=0.2.60
langchain-core>=0.3.0
langchain-deepseek>=0.1.3
langchain-ollama>=0.2.0
langchain-mcp-adapters>=0.1.9
mcp>=1.6.0
pymilvus>=2.5.0
sentence-transformers>=3.3.0
rank-bm25>=0.2.2
jieba>=0.42.1
duckdb>=1.1.0
starrocks>=1.0.0
sqlglot>=25.30.0
fastapi>=0.115.0
uvicorn>=0.32.0
pydantic>=2.9.0
pydantic-settings>=2.6.0
PyYAML>=6.0.2
pytest>=8.3.0
langfuse>=2.60.0
requests>=2.32.0
```

- [ ] **Step 4: 写 config.yaml**

```yaml
llm:
  providers:
    deepseek:
      type: api
      base_url: https://api.deepseek.com
      model: deepseek-chat
      api_key_env: DEEPSEEK_API_KEY
    ollama_local:
      type: local
      base_url: http://localhost:11434
      model: qwen3:8b
  routing:
    task_parse: deepseek
    sql_generate: deepseek
    critic_review: deepseek
    fallback: ollama_local

services:
  mcp_server_url: http://localhost:8001/mcp/sse
  milvus:
    host: localhost
    port: 19530
    collection: dw_sql_cases

executor:
  default: duckdb
  starrocks:
    host: localhost
    port: 9030
    user: root
    password: ""

observability:
  langfuse:
    public_key_env: LANGFUSE_PUBLIC_KEY
    secret_key_env: LANGFUSE_SECRET_KEY
    host: https://cloud.langfuse.com
```

- [ ] **Step 5: 写配置加载器 `dataagent/config.py`（含加载失败自检）**

```python
"""配置加载：config.yaml + 环境变量覆盖。"""
from dataclasses import dataclass, field
from pathlib import Path
import os
import yaml


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
            raw = yaml.safe_load(f) or {}

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
```

- [ ] **Step 6: 写测试 `tests/test_config.py`**

```python
"""配置加载测试。"""
import os
import yaml
from dataagent.config import load_config, Settings


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
```

- [ ] **Step 7: 写 pytest.ini**

```ini
[pytest]
testpaths = tests
markers =
    integration: 需要外部服务（Milvus/StarRocks/LLM API）的测试
```

- [ ] **Step 8: 建 venv + 安装 + 跑测试**

```bash
cd /Users/liyu/Curser/职业专题/dw-dataagent
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest tests/test_config.py -v
```

预期：3 passed

- [ ] **Step 9: Commit**

```bash
git add -A && git commit -m "chore: 项目骨架、配置加载器与基础测试"
```

### Task 2: 数仓 Schema 注册表（30 表 + 元数据生成）

**Files:**
- Create: `dw-dataagent/dataagent/warehouse/__init__.py`
- Create: `dw-dataagent/dataagent/warehouse/schema.py`
- Test: `dw-dataagent/tests/test_schema.py`

**Interfaces:**
- Produces: `dataagent.warehouse.schema.TABLES: dict[str, TableSpec]`；`TableSpec` 字段 `name/domain/layer/grain/description/columns/partition_col`；`ColumnSpec` 字段 `name/data_type/comment`
- Produces: `dataagent.warehouse.schema.LINEAGE: list[LineageEdge]`（source/target/relation）
- Produces: `dataagent.warehouse.schema.METRICS: dict[str, MetricSpec]`（name/definition/formula）
- Produces: `generate_metadata_yaml(out_dir) -> None`：生成 tables.yaml/columns.yaml/lineage.yaml/metrics.yaml

- [ ] **Step 1: 写失败测试 `tests/test_schema.py`**

```python
"""Schema 注册表测试。"""
from dataagent.warehouse.schema import TABLES, LINEAGE, METRICS, TableSpec


def test_exactly_30_tables():
    assert len(TABLES) == 30


def test_all_domains_covered():
    domains = {t.domain for t in TABLES.values()}
    assert domains == {"订单域", "用户域", "商品域", "支付域", "物流域"}


def test_every_table_has_partition_col():
    for t in TABLES.values():
        assert t.partition_col in {c.name for c in t.columns}, t.name


def test_layer_naming_convention():
    for name, t in TABLES.items():
        prefix = name.split("_")[0]
        assert prefix in {"dwd", "dws", "ads", "dim"}, name


def test_lineage_references_existing_tables():
    for edge in LINEAGE:
        assert edge.source in TABLES, edge.source
        assert edge.target in TABLES, edge.target


def test_metrics_have_formula():
    for m in METRICS.values():
        assert m.definition and m.formula
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_schema.py -v`
预期：FAIL（ModuleNotFoundError: dataagent.warehouse.schema）

- [ ] **Step 3: 写 `dataagent/warehouse/schema.py`**

```python
"""数仓 Schema 注册表：30 张表的单一事实源。

生产对标：真实环境中此注册表对应元数据中心（DataHub/Atlas）的 API 返回；
MCP 工具只依赖本模块的查询函数，对接真实元数据中心时接口不变。
"""
from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    data_type: str
    comment: str


@dataclass(frozen=True)
class TableSpec:
    name: str
    domain: str            # 主题域
    layer: str             # DWD/DWS/ADS/DIM
    grain: str             # 粒度：日/全量
    description: str
    columns: tuple[ColumnSpec, ...]
    partition_col: str = "dt"


def C(name, dtype, comment=""):
    return ColumnSpec(name, dtype, comment)


def T(name, domain, layer, grain, desc, cols):
    return TableSpec(name, domain, layer, grain, desc, tuple(cols))


def _common_order_cols():
    return [
        C("order_id", "BIGINT", "订单ID"),
        C("user_id", "BIGINT", "用户ID"),
        C("product_id", "BIGINT", "商品ID"),
        C("dt", "DATE", "分区日期"),
    ]


TABLES: dict[str, TableSpec] = {}

def _reg(t: TableSpec):
    TABLES[t.name] = t


# ===== 订单域（6 表） =====
_reg(T("dwd_order_detail_di", "订单域", "DWD", "日", "订单明细日表（含支付状态）", [
    C("order_id", "BIGINT", "订单ID"),
    C("user_id", "BIGINT", "用户ID"),
    C("product_id", "BIGINT", "商品ID"),
    C("category_id", "BIGINT", "品类ID"),
    C("platform", "VARCHAR", "平台(iOS/Android/Web)"),
    C("pay_status", "VARCHAR", "支付状态(paid/unpaid/refunded)"),
    C("order_amount", "DECIMAL(20,2)", "订单金额"),
    C("item_count", "INT", "商品件数"),
    C("dt", "DATE", "分区日期"),
]))
_reg(T("dwd_order_created_di", "订单域", "DWD", "日", "订单创建日志日表", [
    C("order_id", "BIGINT", "订单ID"),
    C("user_id", "BIGINT", "用户ID"),
    C("product_id", "BIGINT", "商品ID"),
    C("order_status", "VARCHAR", "订单状态(created/paid/shipped/signed/cancelled)"),
    C("create_time", "DATETIME", "创建时间"),
    C("dt", "DATE", "分区日期"),
]))
_reg(T("dws_order_summary_di", "订单域", "DWS", "日", "订单汇总日表", [
    C("order_cnt", "BIGINT", "下单数"),
    C("order_user_cnt", "BIGINT", "下单用户数"),
    C("gmv_amount", "DECIMAL(20,2)", "成交总额（GMV，支付成功口径）"),
    C("pay_order_cnt", "BIGINT", "支付成功订单数"),
    C("pay_rate", "DECIMAL(10,4)", "支付率=pay_order_cnt/order_cnt"),
    C("dt", "DATE", "分区日期"),
]))
_reg(T("dws_category_order_di", "订单域", "DWS", "日", "品类订单汇总日表", [
    C("category_id", "BIGINT", "品类ID"),
    C("order_cnt", "BIGINT", "下单数"),
    C("pay_order_cnt", "BIGINT", "支付成功订单数"),
    C("gmv_amount", "DECIMAL(20,2)", "成交总额"),
    C("dt", "DATE", "分区日期"),
]))
_reg(T("dws_platform_order_di", "订单域", "DWS", "日", "平台订单汇总日表", [
    C("platform", "VARCHAR", "平台"),
    C("order_cnt", "BIGINT", "下单数"),
    C("gmv_amount", "DECIMAL(20,2)", "成交总额"),
    C("dt", "DATE", "分区日期"),
]))
_reg(T("ads_order_daily_report_di", "订单域", "ADS", "日", "订单日报表", [
    C("gmv_amount", "DECIMAL(20,2)", "成交总额"),
    C("order_cnt", "BIGINT", "下单数"),
    C("avg_order_amount", "DECIMAL(20,2)", "客单价=GMV/支付订单数"),
    C("dt", "DATE", "分区日期"),
]))

# ===== 用户域（6 表） =====
_reg(T("dwd_user_behavior_di", "用户域", "DWD", "日", "用户行为明细日表", [
    C("user_id", "BIGINT", "用户ID"),
    C("behavior_type", "VARCHAR", "行为类型(view/cart/buy)"),
    C("product_id", "BIGINT", "商品ID"),
    C("behavior_time", "DATETIME", "行为时间"),
    C("dt", "DATE", "分区日期"),
]))
_reg(T("dwd_user_register_di", "用户域", "DWD", "日", "用户注册明细日表", [
    C("user_id", "BIGINT", "用户ID"),
    C("register_channel", "VARCHAR", "注册渠道(organic/ad/wechat/appstore)"),
    C("register_time", "DATETIME", "注册时间"),
    C("city", "VARCHAR", "城市"),
    C("dt", "DATE", "分区日期"),
]))
_reg(T("dws_user_active_di", "用户域", "DWS", "日", "用户活跃汇总日表", [
    C("active_user_cnt", "BIGINT", "活跃用户数（当日有行为）"),
    C("new_user_cnt", "BIGINT", "新增用户数"),
    C("dau", "BIGINT", "日活跃用户数"),
    C("dt", "DATE", "分区日期"),
]))
_reg(T("dws_user_behavior_summary_di", "用户域", "DWS", "日", "用户行为汇总日表", [
    C("behavior_type", "VARCHAR", "行为类型"),
    C("user_cnt", "BIGINT", "行为用户数"),
    C("behavior_cnt", "BIGINT", "行为次数"),
    C("dt", "DATE", "分区日期"),
]))
_reg(T("dws_user_retention_di", "用户域", "DWS", "日", "用户留存日表", [
    C("register_dt", "DATE", "注册日期"),
    C("retain_d1", "DECIMAL(10,4)", "次日留存率"),
    C("retain_d7", "DECIMAL(10,4)", "7日留存率"),
    C("retain_d30", "DECIMAL(10,4)", "30日留存率"),
    C("dt", "DATE", "分区日期"),
]))
_reg(T("ads_user_growth_report_di", "用户域", "ADS", "日", "用户增长日报表", [
    C("new_user_cnt", "BIGINT", "新增用户数"),
    C("active_user_cnt", "BIGINT", "活跃用户数"),
    C("retention_rate", "DECIMAL(10,4)", "留存率（次日）"),
    C("dt", "DATE", "分区日期"),
]))

# ===== 商品域（6 表） =====
_reg(T("dim_product_info", "商品域", "DIM", "全量", "商品维度表", [
    C("product_id", "BIGINT", "商品ID"),
    C("product_name", "VARCHAR", "商品名称"),
    C("category_id", "BIGINT", "品类ID"),
    C("brand", "VARCHAR", "品牌"),
    C("price", "DECIMAL(20,2)", "标价"),
    C("status", "VARCHAR", "状态(on/off)"),
]))
_reg(T("dim_category_info", "商品域", "DIM", "全量", "品类维度表", [
    C("category_id", "BIGINT", "品类ID"),
    C("category_name", "VARCHAR", "品类名称"),
    C("parent_category_id", "BIGINT", "父品类ID"),
]))
_reg(T("dwd_product_view_di", "商品域", "DWD", "日", "商品浏览明细日表", [
    C("user_id", "BIGINT", "用户ID"),
    C("product_id", "BIGINT", "商品ID"),
    C("view_time", "DATETIME", "浏览时间"),
    C("dt", "DATE", "分区日期"),
]))
_reg(T("dws_product_gmv_di", "商品域", "DWS", "日", "商品GMV汇总日表", [
    C("product_id", "BIGINT", "商品ID"),
    C("gmv_amount", "DECIMAL(20,2)", "成交总额"),
    C("order_cnt", "BIGINT", "支付订单数"),
    C("pay_user_cnt", "BIGINT", "支付用户数"),
    C("dt", "DATE", "分区日期"),
]))
_reg(T("dws_product_view_di", "商品域", "DWS", "日", "商品浏览汇总日表", [
    C("product_id", "BIGINT", "商品ID"),
    C("view_cnt", "BIGINT", "浏览量"),
    C("view_user_cnt", "BIGINT", "浏览用户数"),
    C("dt", "DATE", "分区日期"),
]))
_reg(T("ads_product_ranking_di", "商品域", "ADS", "日", "商品排行榜日表", [
    C("product_id", "BIGINT", "商品ID"),
    C("gmv_rank", "INT", "GMV排名"),
    C("view_rank", "INT", "浏览量排名"),
    C("dt", "DATE", "分区日期"),
]))

# ===== 支付域（6 表） =====
_reg(T("dwd_payment_detail_di", "支付域", "DWD", "日", "支付明细日表", [
    C("payment_id", "BIGINT", "支付ID"),
    C("order_id", "BIGINT", "订单ID"),
    C("user_id", "BIGINT", "用户ID"),
    C("pay_channel", "VARCHAR", "支付渠道(alipay/wechat/card)"),
    C("pay_amount", "DECIMAL(20,2)", "支付金额"),
    C("pay_time", "DATETIME", "支付时间"),
    C("pay_status", "VARCHAR", "支付状态(success/failed)"),
    C("dt", "DATE", "分区日期"),
]))
_reg(T("dws_payment_summary_di", "支付域", "DWS", "日", "支付汇总日表", [
    C("pay_amount", "DECIMAL(20,2)", "支付总额"),
    C("pay_order_cnt", "BIGINT", "支付订单数"),
    C("pay_user_cnt", "BIGINT", "支付用户数"),
    C("refund_amount", "DECIMAL(20,2)", "退款总额"),
    C("dt", "DATE", "分区日期"),
]))
_reg(T("dws_payment_channel_di", "支付域", "DWS", "日", "支付渠道汇总日表", [
    C("pay_channel", "VARCHAR", "支付渠道"),
    C("pay_amount", "DECIMAL(20,2)", "支付总额"),
    C("pay_order_cnt", "BIGINT", "支付订单数"),
    C("dt", "DATE", "分区日期"),
]))
_reg(T("dwd_refund_detail_di", "支付域", "DWD", "日", "退款明细日表", [
    C("refund_id", "BIGINT", "退款ID"),
    C("order_id", "BIGINT", "订单ID"),
    C("refund_amount", "DECIMAL(20,2)", "退款金额"),
    C("refund_time", "DATETIME", "退款时间"),
    C("dt", "DATE", "分区日期"),
]))
_reg(T("dws_refund_summary_di", "支付域", "DWS", "日", "退款汇总日表", [
    C("refund_amount", "DECIMAL(20,2)", "退款总额"),
    C("refund_order_cnt", "BIGINT", "退款订单数"),
    C("dt", "DATE", "分区日期"),
]))
_reg(T("ads_payment_daily_report_di", "支付域", "ADS", "日", "支付日报表", [
    C("pay_amount", "DECIMAL(20,2)", "支付总额"),
    C("refund_amount", "DECIMAL(20,2)", "退款总额"),
    C("net_amount", "DECIMAL(20,2)", "净额=支付-退款"),
    C("dt", "DATE", "分区日期"),
]))

# ===== 物流域（6 表） =====
_reg(T("dwd_logistics_tracking_di", "物流域", "DWD", "日", "物流轨迹日表", [
    C("order_id", "BIGINT", "订单ID"),
    C("logistics_company", "VARCHAR", "物流公司(sf/jd/zt)"),
    C("status", "VARCHAR", "状态(shipped/transit/signed)"),
    C("ship_time", "DATETIME", "发货时间"),
    C("sign_time", "DATETIME", "签收时间"),
    C("dt", "DATE", "分区日期"),
]))
_reg(T("dwd_logistics_shipped_di", "物流域", "DWD", "日", "发货明细日表", [
    C("order_id", "BIGINT", "订单ID"),
    C("warehouse_id", "BIGINT", "仓库ID"),
    C("ship_time", "DATETIME", "发货时间"),
    C("dt", "DATE", "分区日期"),
]))
_reg(T("dws_logistics_summary_di", "物流域", "DWS", "日", "物流汇总日表", [
    C("ship_order_cnt", "BIGINT", "发货订单数"),
    C("sign_order_cnt", "BIGINT", "签收订单数"),
    C("avg_delivery_days", "DECIMAL(10,2)", "平均配送时长（天）"),
    C("dt", "DATE", "分区日期"),
]))
_reg(T("dws_logistics_company_di", "物流域", "DWS", "日", "物流公司汇总日表", [
    C("logistics_company", "VARCHAR", "物流公司"),
    C("ship_cnt", "BIGINT", "发货数"),
    C("sign_cnt", "BIGINT", "签收数"),
    C("dt", "DATE", "分区日期"),
]))
_reg(T("dim_warehouse_info", "物流域", "DIM", "全量", "仓库维度表", [
    C("warehouse_id", "BIGINT", "仓库ID"),
    C("warehouse_name", "VARCHAR", "仓库名称"),
    C("city", "VARCHAR", "城市"),
]))
_reg(T("ads_logistics_daily_report_di", "物流域", "ADS", "日", "物流日报表", [
    C("ship_cnt", "BIGINT", "发货数"),
    C("sign_cnt", "BIGINT", "签收数"),
    C("on_time_rate", "DECIMAL(10,4)", "准时率"),
    C("dt", "DATE", "分区日期"),
]))


@dataclass(frozen=True)
class LineageEdge:
    source: str
    target: str
    relation: str


LINEAGE: list[LineageEdge] = [
    LineageEdge("dwd_order_detail_di", "dws_order_summary_di", "ETL_AGG"),
    LineageEdge("dws_order_summary_di", "ads_order_daily_report_di", "ETL_JOIN"),
    LineageEdge("dwd_order_detail_di", "dws_category_order_di", "ETL_AGG"),
    LineageEdge("dwd_order_detail_di", "dws_platform_order_di", "ETL_AGG"),
    LineageEdge("dwd_user_behavior_di", "dws_user_active_di", "ETL_AGG"),
    LineageEdge("dwd_user_register_di", "dws_user_active_di", "ETL_AGG"),
    LineageEdge("dwd_user_behavior_di", "dws_user_behavior_summary_di", "ETL_AGG"),
    LineageEdge("dwd_user_register_di", "dws_user_retention_di", "ETL_JOIN"),
    LineageEdge("dws_user_active_di", "ads_user_growth_report_di", "ETL_JOIN"),
    LineageEdge("dwd_order_detail_di", "dws_product_gmv_di", "ETL_AGG"),
    LineageEdge("dwd_product_view_di", "dws_product_view_di", "ETL_AGG"),
    LineageEdge("dws_product_gmv_di", "ads_product_ranking_di", "ETL_JOIN"),
    LineageEdge("dws_product_view_di", "ads_product_ranking_di", "ETL_JOIN"),
    LineageEdge("dwd_payment_detail_di", "dws_payment_summary_di", "ETL_AGG"),
    LineageEdge("dwd_payment_detail_di", "dws_payment_channel_di", "ETL_AGG"),
    LineageEdge("dwd_refund_detail_di", "dws_refund_summary_di", "ETL_AGG"),
    LineageEdge("dws_payment_summary_di", "ads_payment_daily_report_di", "ETL_JOIN"),
    LineageEdge("dws_refund_summary_di", "ads_payment_daily_report_di", "ETL_JOIN"),
    LineageEdge("dwd_logistics_tracking_di", "dws_logistics_summary_di", "ETL_AGG"),
    LineageEdge("dwd_logistics_tracking_di", "dws_logistics_company_di", "ETL_AGG"),
    LineageEdge("dwd_logistics_shipped_di", "dws_logistics_summary_di", "ETL_AGG"),
    LineageEdge("dws_logistics_summary_di", "ads_logistics_daily_report_di", "ETL_JOIN"),
]


@dataclass(frozen=True)
class MetricSpec:
    name: str
    definition: str
    formula: str


METRICS: dict[str, MetricSpec] = {
    "GMV": MetricSpec(
        name="GMV",
        definition="支付成功订单的成交总额",
        formula="SUM(gmv_amount) WHERE pay_status='paid'",
    ),
    "支付率": MetricSpec(
        name="支付率",
        definition="支付成功订单数占下单数的比例",
        formula="pay_order_cnt / order_cnt",
    ),
    "客单价": MetricSpec(
        name="客单价",
        definition="平均每个支付订单的成交金额",
        formula="GMV / pay_order_cnt",
    ),
    "DAU": MetricSpec(
        name="DAU",
        definition="当日有任意行为的独立用户数",
        formula="COUNT(DISTINCT user_id) FROM dwd_user_behavior_di",
    ),
    "新增用户": MetricSpec(
        name="新增用户",
        definition="当日注册的独立用户数",
        formula="COUNT(DISTINCT user_id) FROM dwd_user_register_di",
    ),
    "次日留存率": MetricSpec(
        name="次日留存率",
        definition="注册次日仍活跃用户占注册用户比例",
        formula="retain_d1 FROM dws_user_retention_di",
    ),
    "退款率": MetricSpec(
        name="退款率",
        definition="退款订单数占支付订单数的比例",
        formula="refund_order_cnt / pay_order_cnt",
    ),
    "履约准时率": MetricSpec(
        name="履约准时率",
        definition="准时签收订单占发货订单的比例",
        formula="on_time_rate FROM ads_logistics_daily_report_di",
    ),
}


def generate_metadata_yaml(out_dir: str | Path) -> None:
    """从注册表生成元数据 YAML 文件（tables/columns/lineage/metrics）。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tables = [
        {
            "table_name": t.name, "domain": t.domain, "layer": t.layer,
            "granularity": t.grain, "description": t.description,
            "partition_col": t.partition_col,
        }
        for t in TABLES.values()
    ]
    (out / "tables.yaml").write_text(
        yaml.safe_dump(tables, allow_unicode=True, sort_keys=False),
        encoding="utf-8")

    columns = [
        {"table_name": t.name, "column_name": c.name,
         "data_type": c.data_type, "comment": c.comment}
        for t in TABLES.values() for c in t.columns
    ]
    (out / "columns.yaml").write_text(
        yaml.safe_dump(columns, allow_unicode=True, sort_keys=False),
        encoding="utf-8")

    lineage = [
        {"source_table": e.source, "target_table": e.target,
         "relation": e.relation}
        for e in LINEAGE
    ]
    (out / "lineage.yaml").write_text(
        yaml.safe_dump(lineage, allow_unicode=True, sort_keys=False),
        encoding="utf-8")

    metrics = [
        {"metric_name": m.name, "definition": m.definition, "formula": m.formula}
        for m in METRICS.values()
    ]
    (out / "metrics.yaml").write_text(
        yaml.safe_dump(metrics, allow_unicode=True, sort_keys=False),
        encoding="utf-8")
```

- [ ] **Step 4: 跑测试确认通过**

Run: `pytest tests/test_schema.py -v`
预期：6 passed

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: 数仓 Schema 注册表（30 表 + 血缘 + 指标 + 元数据生成）"
```

### Task 3: 权限模型（RBAC 表级权限）

**Files:**
- Create: `dw-dataagent/dataagent/permissions.py`
- Test: `dw-dataagent/tests/test_permissions.py`

**Interfaces:**
- Produces: `dataagent.permissions.ROLES: dict[str, list[str]]`（角色→主题域白名单，`*` 为全部）
- Produces: `filter_tables_by_role(role: str, tables: dict[str, TableSpec]) -> dict[str, TableSpec]`（返回角色可见表）
- Produces: `resolve_role(user: str) -> str`（用户→角色映射，默认 `data_analyst`）

- [ ] **Step 1: 写失败测试 `tests/test_permissions.py`**

```python
"""RBAC 权限过滤测试。"""
from dataagent.permissions import ROLES, filter_tables_by_role, resolve_role
from dataagent.warehouse.schema import TABLES


def test_analyst_sees_three_domains():
    visible = filter_tables_by_role("data_analyst", TABLES)
    domains = {t.domain for t in visible.values()}
    assert domains == {"订单域", "用户域", "商品域"}
    # 支付域、物流域完全不可见（源头阻断）
    assert not any(t.domain in ("支付域", "物流域") for t in visible.values())


def test_finance_sees_payment():
    visible = filter_tables_by_role("finance_analyst", TABLES)
    domains = {t.domain for t in visible.values()}
    assert domains == {"订单域", "支付域"}


def test_admin_sees_all():
    visible = filter_tables_by_role("admin", TABLES)
    assert len(visible) == len(TABLES)


def test_unknown_role_falls_back_to_analyst():
    visible = filter_tables_by_role("nonexistent_role", TABLES)
    assert len(visible) == len(filter_tables_by_role("data_analyst", TABLES))


def test_resolve_role_default():
    assert resolve_role("any_user") == "data_analyst"
    assert resolve_role("finance_wang") == "finance_analyst"
    assert resolve_role("admin_li") == "admin"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_permissions.py -v`
预期：FAIL（ModuleNotFoundError: dataagent.permissions）

- [ ] **Step 3: 写 `dataagent/permissions.py`**

```python
"""RBAC 表级权限：角色 → 主题域白名单。

生产对标：对应大厂数据权限中心（如 DataWorks 的 RBAC 模型）。
核心原则：无权限的表在元数据检索层完全不可见——从源头消除越权取数可能。
"""
ROLES: dict[str, list[str]] = {
    "data_analyst": ["订单域", "用户域", "商品域"],
    "finance_analyst": ["订单域", "支付域"],
    "ops_analyst": ["物流域"],
    "admin": ["*"],
}

# 用户 → 角色映射（演示用；生产对应 LDAP/权限中心）
_USER_ROLE_MAP: dict[str, str] = {
    "finance_wang": "finance_analyst",
    "ops_zhang": "ops_analyst",
    "admin_li": "admin",
}

DEFAULT_ROLE = "data_analyst"


def resolve_role(user: str) -> str:
    return _USER_ROLE_MAP.get(user, DEFAULT_ROLE)


def filter_tables_by_role(role: str, tables: dict) -> dict:
    allowed_domains = ROLES.get(role, ROLES[DEFAULT_ROLE])
    if allowed_domains == ["*"]:
        return dict(tables)
    return {
        name: spec
        for name, spec in tables.items()
        if spec.domain in allowed_domains
    }
```

- [ ] **Step 4: 跑测试确认通过**

Run: `pytest tests/test_permissions.py -v`
预期：5 passed

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: RBAC 表级权限（角色→主题域过滤）"
```

### Task 4: QueryExecutor 抽象 + DuckDB 实现

**Files:**
- Create: `dw-dataagent/dataagent/executor/__init__.py`
- Create: `dw-dataagent/dataagent/executor/base.py`
- Create: `dw-dataagent/dataagent/executor/duckdb_executor.py`
- Test: `dw-dataagent/tests/test_executor.py`

**Interfaces:**
- Produces: `dataagent.executor.base.QueryExecutor`（协议类：`setup()` / `execute(sql) -> list[tuple]` / `close()`）
- Produces: `dataagent.executor.base.QueryError(Exception)`（SQL 执行失败统一异常，携带原始错误信息）
- Produces: `dataagent.executor.duckdb_executor.DuckDBExecutor(path: str, tables: dict[str, TableSpec])`
- Consumes: `dataagent.warehouse.schema.TABLES`

- [ ] **Step 1: 写失败测试 `tests/test_executor.py`**

```python
"""DuckDB Executor 测试。"""
import pytest
from dataagent.executor.duckdb_executor import DuckDBExecutor
from dataagent.executor.base import QueryError
from dataagent.warehouse.schema import TABLES


@pytest.fixture
def executor(tmp_path):
    ex = DuckDBExecutor(str(tmp_path / "test.duckdb"), TABLES)
    ex.setup()
    yield ex
    ex.close()


def test_setup_creates_all_tables(executor):
    import duckdb
    con = duckdb.connect(executor.path)
    names = {r[0] for r in con.execute(
        "SELECT table_name FROM information_schema.tables").fetchall()}
    assert set(TABLES.keys()) == names
    con.close()


def test_execute_returns_rows(executor):
    executor.execute("INSERT INTO dim_category_info VALUES (1, '服饰', 0)")
    rows = executor.execute(
        "SELECT category_id, category_name FROM dim_category_info")
    assert rows == [(1, "服饰")]


def test_bad_sql_raises_query_error(executor):
    with pytest.raises(QueryError):
        executor.execute("SELECT * FROM nonexistent_table")


def test_ddl_restricted(executor):
    with pytest.raises(QueryError):
        executor.execute("DROP TABLE dim_category_info")
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_executor.py -v`
预期：FAIL（ModuleNotFoundError）

- [ ] **Step 3: 写 `dataagent/executor/base.py`**

```python
"""QueryExecutor 抽象：隔离执行引擎差异。

生产对标：开发环境 DuckDB 零成本模拟，生产环境 StarRocks 对齐；
Agent 代码只依赖本接口，引擎切换零改动。
"""
from typing import Protocol


class QueryError(Exception):
    """SQL 执行失败统一异常。message 保留原始引擎错误供 Agent 修正。"""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class QueryExecutor(Protocol):
    def setup(self) -> None:
        """建表/初始化连接。"""
        ...

    def execute(self, sql: str) -> list[tuple]:
        """执行 SELECT 返回行；DML/DDL 按引擎支持执行；失败抛 QueryError。"""
        ...

    def close(self) -> None:
        """释放连接。"""
        ...
```

- [ ] **Step 4: 写 `dataagent/executor/duckdb_executor.py`**

```python
"""DuckDB 执行器（开发兜底，零内存开销）。"""
import duckdb
from dataagent.executor.base import QueryError
from dataagent.warehouse.schema import TableSpec


class DuckDBExecutor:
    def __init__(self, path: str, tables: dict[str, TableSpec]):
        self.path = path
        self.tables = tables
        self._con = None

    def setup(self) -> None:
        import os
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._con = duckdb.connect(self.path)
        for spec in self.tables.values():
            cols = ", ".join(
                f"{c.name} {c.data_type}" for c in spec.columns)
            self._con.execute(
                f"CREATE TABLE IF NOT EXISTS {spec.name} ({cols})")
        # 只读护栏：禁止 DDL/DML（取数 Agent 只应 SELECT）
        self._con.execute("PRAGMA enable_external_access=false")

    def execute(self, sql: str) -> list[tuple]:
        try:
            return self._con.execute(sql).fetchall()
        except Exception as e:  # duckdb.Error 及一切执行异常
            raise QueryError(str(e)) from e

    def close(self) -> None:
        if self._con is not None:
            self._con.close()
            self._con = None
```

- [ ] **Step 5: 跑测试确认通过**

Run: `pytest tests/test_executor.py -v`
预期：4 passed（若 DDL 限制测试因 DuckDB 无 enforcement 失败，改为在 executor 层显式拦截 DDL——见 Step 6 修正）

- [ ] **Step 6: 如 Step 5 中 `test_ddl_restricted` 失败，在 `execute` 开头加显式拦截并重跑**

```python
    def execute(self, sql: str) -> list[tuple]:
        stripped = sql.strip().rstrip(";").upper()
        if stripped.startswith(("DROP", "DELETE", "UPDATE", "CREATE", "ALTER", "TRUNCATE", "INSERT")):
            raise QueryError(f"只读限制：不允许执行 {stripped.split()[0]} 语句")
        try:
            return self._con.execute(sql).fetchall()
        except Exception as e:
            raise QueryError(str(e)) from e
```

注意：`test_execute_returns_rows` 里用 INSERT 灌数据会与只读限制冲突——将该测试改用 `setup()` 后的直接 con 写入：

```python
def test_execute_returns_rows(executor):
    import duckdb
    con = duckdb.connect(executor.path)
    con.execute("INSERT INTO dim_category_info VALUES (1, '服饰', 0)")
    con.close()
    rows = executor.execute(
        "SELECT category_id, category_name FROM dim_category_info")
    assert rows == [(1, "服饰")]
```

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "feat: QueryExecutor 抽象 + DuckDB 实现（只读护栏）"
```

### Task 5: 模拟数仓初始化脚本（DuckDB 灌数 + StarRocks 建表）

**Files:**
- Create: `dw-dataagent/dataagent/executor/starrocks_executor.py`
- Create: `dw-dataagent/scripts/init_warehouse.py`
- Test: 手动验收（无 pytest，数据量级大）

**Interfaces:**
- Produces: `dataagent.executor.starrocks_executor.StarRocksExecutor(host, port, user, password, tables)`，接口与 DuckDBExecutor 相同
- Produces: `scripts/init_warehouse.py`：`--engine duckdb|starrocks` 参数，初始化建表 + 灌数 + 生成元数据 YAML

- [ ] **Step 1: 写 `dataagent/executor/starrocks_executor.py`**

```python
"""StarRocks 执行器（生产对齐；按需启动容器）。"""
import starrocks
from dataagent.executor.base import QueryError
from dataagent.warehouse.schema import TableSpec


class StarRocksExecutor:
    def __init__(self, host: str, port: int, user: str, password: str,
                 tables: dict[str, TableSpec], database: str = "demo"):
        self.host, self.port = host, port
        self.user, self.password = user, password
        self.tables = tables
        self.database = database
        self._con = None

    def setup(self) -> None:
        self._con = starrocks.connect(
            host=self.host, port=self.port,
            user=self.user, password=self.password)
        cur = self._con.cursor()
        cur.execute(f"CREATE DATABASE IF NOT EXISTS {self.database}")
        cur.execute(f"USE {self.database}")
        for spec in self.tables.values():
            cols = ", ".join(
                f"{c.name} {c.data_type}" for c in spec.columns)
            cur.execute(
                f"CREATE TABLE IF NOT EXISTS {spec.name} ({cols}) "
                f"ENGINE=OLAP DUPLICATE KEY({spec.columns[0].name}) "
                f"DISTRIBUTED BY HASH({spec.columns[0].name}) BUCKETS 1")
        cur.close()

    def execute(self, sql: str) -> list[tuple]:
        stripped = sql.strip().rstrip(";").upper()
        if stripped.startswith(("DROP", "DELETE", "UPDATE", "CREATE",
                                "ALTER", "TRUNCATE", "INSERT")):
            raise QueryError(f"只读限制：不允许执行 {stripped.split()[0]} 语句")
        try:
            cur = self._con.cursor()
            cur.execute(sql)
            rows = cur.fetchall()
            cur.close()
            return rows
        except Exception as e:
            raise QueryError(str(e)) from e

    def close(self) -> None:
        if self._con is not None:
            self._con.close()
            self._con = None
```

- [ ] **Step 2: 写 `scripts/init_warehouse.py`（确定性生成 90 天数据，种子 42）**

```python
"""模拟数仓初始化：建表 + 灌数 + 元数据 YAML。

用法:
  python scripts/init_warehouse.py --engine duckdb   # 默认，开发兜底
  python scripts/init_warehouse.py --engine starrocks  # 需容器已启动（按需）
"""
import argparse
import random
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataagent.config import load_config
from dataagent.executor.duckdb_executor import DuckDBExecutor
from dataagent.executor.starrocks_executor import StarRocksExecutor
from dataagent.warehouse.schema import TABLES, generate_metadata_yaml

DAYS = 90
END_DATE = date(2026, 7, 31)
rng = random.Random(42)

CATEGORIES = [(1, "服饰"), (2, "数码"), (3, "食品"), (4, "家居"), (5, "美妆")]
PLATFORMS = ["iOS", "Android", "Web"]
CHANNELS = ["alipay", "wechat", "card"]
LOGISTICS = ["sf", "jd", "zt"]
WAREHOUSES = [(1, "华东仓", "上海"), (2, "华南仓", "广州"), (3, "华北仓", "北京")]
USERS, PRODUCTS = 50_000, 5_000


def dates():
    for i in range(DAYS):
        yield END_DATE - timedelta(days=DAYS - 1 - i)


def build_dim(executor):
    """维度表灌数（全量快照）。"""
    for cid, cname in CATEGORIES:
        _insert(executor, "dim_category_info",
                [(cid, cname, 0)])
    for pid in range(1, PRODUCTS + 1):
        cid = CATEGORIES[pid % 5][0]
        _insert(executor, "dim_product_info",
                [(pid, f"商品{pid}", cid,
                  f"品牌{pid % 20}", round(rng.uniform(9.9, 999), 2),
                  "on")])
    for wid, wname, city in WAREHOUSES:
        _insert(executor, "dim_warehouse_info", [(wid, wname, city)])


def _insert(executor, table: str, rows: list[tuple]):
    """通过引擎内部连接灌数（初始化专用，绕过只读护栏）。"""
    con = executor._con
    if isinstance(executor, DuckDBExecutor):
        n = len(TABLES[table].columns)
        marks = ",".join(["?"] * n)
        con.executemany(
            f"INSERT INTO {table} VALUES ({marks})", rows)
    else:
        cur = con.cursor()
        for row in rows:
            vals = ",".join(
                f"'{v}'" if isinstance(v, str) else str(v) for v in row)
            cur.execute(f"INSERT INTO {table} VALUES ({vals})")
        cur.close()


def build_fact_daily(executor):
    """日增量事实/DWS/ADS 表灌数。"""
    for d in dates():
        dt = d.isoformat()
        active_users = rng.randint(20_000, 30_000)
        orders = rng.randint(50_000, 80_000)
        pay_orders = int(orders * rng.uniform(0.55, 0.7))

        # dwd_order_detail_di：按日抽样订单明细
        order_rows, pay_rows, refund_rows = [], [], []
        for i in range(orders):
            uid = rng.randint(1, USERS)
            pid = rng.randint(1, PRODUCTS)
            cid = CATEGORIES[pid % 5][0]
            plat = rng.choice(PLATFORMS)
            status = rng.choices(
                ["paid", "unpaid", "refunded"], weights=[55, 35, 10])[0]
            amount = round(rng.uniform(20, 500), 2)
            oid = i + 1
            order_rows.append((oid, uid, pid, cid, plat, status,
                               amount, rng.randint(1, 3), dt))
            if status in ("paid", "refunded"):
                ch = rng.choice(CHANNELS)
                pay_rows.append((oid, oid, uid, ch, amount, dt, "success", dt))
            if status == "refunded":
                refund_rows.append((oid, oid, amount * 0.6, dt, dt))

        # DWS 汇总（由明细聚合口径算出，保证口径自洽）
        paid_amount = sum(r[6] for r in order_rows
                          if r[5] in ("paid", "refunded"))
        pay_cnt = len(pay_rows)
        _insert(executor, "dws_order_summary_di",
                [(orders, len({r[1] for r in order_rows}), round(paid_amount, 2),
                  pay_cnt, round(pay_cnt / orders, 4), dt)])

        cat_agg = {}
        for r in order_rows:
            cid = r[3]
            cat_agg.setdefault(cid, [0, 0, 0.0])
            cat_agg[cid][0] += 1
            if r[5] in ("paid", "refunded"):
                cat_agg[cid][1] += 1
                cat_agg[cid][2] += r[6]
        for cid, (cnt, pcnt, amt) in cat_agg.items():
            _insert(executor, "dws_category_order_di",
                    [(cid, cnt, pcnt, round(amt, 2), dt)])

        for plat in PLATFORMS:
            plat_rows = [r for r in order_rows if r[4] == plat]
            amt = sum(r[6] for r in plat_rows
                      if r[5] in ("paid", "refunded"))
            _insert(executor, "dws_platform_order_di",
                    [(plat, len(plat_rows), round(amt, 2), dt)])

        _insert(executor, "ads_order_daily_report_di",
                [(round(paid_amount, 2), orders,
                  round(paid_amount / pay_cnt, 2) if pay_cnt else 0, dt)])

        # 用户域
        new_users = rng.randint(800, 2000)
        _insert(executor, "dws_user_active_di",
                [(active_users, new_users, active_users, dt)])
        for bt in ("view", "cart", "buy"):
            _insert(executor, "dws_user_behavior_summary_di",
                    [(bt, rng.randint(5_000, active_users),
                      rng.randint(10_000, 200_000), dt)])
        _insert(executor, "dws_user_retention_di",
                [(d - timedelta(days=1)).isoformat(),
                 round(rng.uniform(0.3, 0.5), 4),
                 round(rng.uniform(0.15, 0.3), 4),
                 round(rng.uniform(0.05, 0.15), 4), dt)])
        _insert(executor, "ads_user_growth_report_di",
                [(new_users, active_users,
                  round(rng.uniform(0.3, 0.5), 4), dt)])

        # 支付域
        total_pay = sum(r[4] for r in pay_rows)
        _insert(executor, "dws_payment_summary_di",
                [(round(total_pay, 2), len(pay_rows),
                  len({r[2] for r in pay_rows}),
                  round(sum(r[3] for r in refund_rows), 2), dt)])
        for ch in CHANNELS:
            ch_rows = [r for r in pay_rows if r[3] == ch]
            _insert(executor, "dws_payment_channel_di",
                    [(ch, round(sum(r[4] for r in ch_rows), 2),
                      len(ch_rows), dt)])
        refund_amt = sum(r[2] for r in refund_rows)
        _insert(executor, "dws_refund_summary_di",
                [(round(refund_amt, 2), len(refund_rows), dt)])
        _insert(executor, "ads_payment_daily_report_di",
                [(round(total_pay, 2), round(refund_amt, 2),
                  round(total_pay - refund_amt, 2), dt)])

        # 物流域
        ship_cnt = pay_cnt
        sign_cnt = int(ship_cnt * rng.uniform(0.7, 0.9))
        _insert(executor, "dws_logistics_summary_di",
                [(ship_cnt, sign_cnt, round(rng.uniform(1.5, 3.5), 2), dt)])
        for lg in LOGISTICS:
            _insert(executor, "dws_logistics_company_di",
                    [(lg, ship_cnt // 3, sign_cnt // 3, dt)])
        _insert(executor, "ads_logistics_daily_report_di",
                [(ship_cnt, sign_cnt, round(rng.uniform(0.85, 0.97), 4), dt)])

        print(f"[{dt}] orders={orders} pay={pay_cnt} new_users={new_users}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default="duckdb",
                        choices=["duckdb", "starrocks"])
    args = parser.parse_args()

    settings = load_config()
    if args.engine == "duckdb":
        executor = DuckDBExecutor(settings.warehouse_path, TABLES)
    else:
        sr = settings.executor.starrocks
        executor = StarRocksExecutor(
            sr.host, sr.port, sr.user, sr.password, TABLES)

    print(f"[init] 建表 30 张 ({args.engine})...")
    executor.setup()
    print("[init] 灌维度表...")
    build_dim(executor)
    print("[init] 灌日增量表（90 天）...")
    build_fact_daily(executor)
    print("[init] 生成元数据 YAML...")
    generate_metadata_yaml(settings.metadata_dir)
    executor.close()
    print("[init] 完成")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 跑 DuckDB 初始化验收**

```bash
source .venv/bin/activate
python scripts/init_warehouse.py --engine duckdb
python -c "
import duckdb
con = duckdb.connect('data/warehouse.duckdb')
print('tables:', con.execute('SELECT count(*) FROM information_schema.tables').fetchone()[0])
print('order_summary rows:', con.execute('SELECT count(*) FROM dws_order_summary_di').fetchone()[0])
print('sample:', con.execute('SELECT * FROM dws_category_order_di LIMIT 3').fetchall())
con.close()
"
```

预期：tables=30；order_summary rows=90；sample 正常返回

- [ ] **Step 4: 验证元数据文件生成**

```bash
ls data/metadata/   # tables.yaml columns.yaml lineage.yaml metrics.yaml
python -c "import yaml; print(len(yaml.safe_load(open('data/metadata/tables.yaml'))))"  # 30
python -c "import yaml; print(len(yaml.safe_load(open('data/metadata/lineage.yaml'))))"  # 22
python -c "import yaml; print(len(yaml.safe_load(open('data/metadata/metrics.yaml'))))"  # 8
```

- [ ] **Step 5: StarRocks 按需验证（内存足够时执行，否则跳过并记录）**

```bash
# 停止 Ollama（若在运行）释放内存
docker run -d --name starrocks -p 9030:9030 -p 8030:8030 \
  --memory 8g starrocks/allin1-ubuntu:3.4.3
# 等待 FE 就绪（约 1-2 分钟）
until docker exec starrocks bash -c "mysql -h127.0.0.1 -P9030 -uroot -e 'select 1'" 2>/dev/null; do sleep 5; done
python scripts/init_warehouse.py --engine starrocks
docker stop starrocks   # 用完即停，释放内存
```

若 ARM 兼容或内存问题导致失败：`docker rm starrocks`，执行器默认保持 duckdb，在 README 备注 StarRocks 验证待有条件环境。**这是已声明的兜底路径，不算失败。**

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: 模拟数仓初始化（90 天确定性数据）+ StarRocks 执行器"
```

## Day 2 · MCP 与 RAG

### Task 6: 历史案例库（50 条需求+SQL）

**Files:**
- Create: `dw-dataagent/data/cases/cases.yaml`
- Create: `dw-dataagent/dataagent/rag/__init__.py`
- Create: `dw-dataagent/dataagent/rag/cases.py`
- Test: `dw-dataagent/tests/test_cases.py`

**Interfaces:**
- Produces: `dataagent.rag.cases.load_cases(path="data/cases/cases.yaml") -> list[Case]`；`Case` 字段 `id/domain/question/sql/tables/metrics`
- Produces: `dataagent.rag.cases.Case`

- [ ] **Step 1: 写失败测试 `tests/test_cases.py`**

```python
"""案例库加载测试。"""
from dataagent.rag.cases import load_cases


def test_loads_50_cases():
    cases = load_cases()
    assert len(cases) == 50


def test_case_fields_complete():
    for c in load_cases():
        assert c.id and c.question and c.sql
        assert c.domain in {"订单域", "用户域", "商品域", "支付域", "物流域"}
        assert c.tables, c.id


def test_case_sql_references_registered_tables():
    from dataagent.warehouse.schema import TABLES
    for c in load_cases():
        for t in c.tables:
            assert t in TABLES, f"{c.id}: {t} 不在注册表"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_cases.py -v`
预期：FAIL（cases.yaml 不存在）

- [ ] **Step 3: 写 `data/cases/cases.yaml`（50 条完整内容）**

```yaml
# 历史取数案例库：需求 + SQL 配对（RAG 检索素材）
# SQL 均为只读 SELECT；日期以 2026-07 为基准区间
- id: c001
  domain: 订单域
  question: 各品类GMV Top10
  sql: SELECT category_id, SUM(gmv_amount) AS gmv FROM dws_category_order_di WHERE dt >= '2026-07-01' GROUP BY category_id ORDER BY gmv DESC LIMIT 10
  tables: [dws_category_order_di]
  metrics: [GMV]
- id: c002
  domain: 订单域
  question: 近30天各平台GMV按日趋势
  sql: SELECT dt, platform, SUM(gmv_amount) AS gmv FROM dws_platform_order_di WHERE dt >= '2026-07-02' AND dt <= '2026-07-31' GROUP BY dt, platform ORDER BY dt
  tables: [dws_platform_order_di]
  metrics: [GMV]
- id: c003
  domain: 订单域
  question: 近7天每日下单数与支付率
  sql: SELECT dt, order_cnt, pay_order_cnt, pay_rate FROM dws_order_summary_di WHERE dt >= '2026-07-25' AND dt <= '2026-07-31' ORDER BY dt
  tables: [dws_order_summary_di]
  metrics: [支付率]
- id: c004
  domain: 订单域
  question: 本月客单价变化趋势
  sql: SELECT dt, avg_order_amount FROM ads_order_daily_report_di WHERE dt >= '2026-07-01' ORDER BY dt
  tables: [ads_order_daily_report_di]
  metrics: [客单价]
- id: c005
  domain: 订单域
  question: 各平台下单用户数对比
  sql: SELECT platform, COUNT(DISTINCT user_id) AS user_cnt FROM dwd_order_detail_di WHERE dt >= '2026-07-01' GROUP BY platform
  tables: [dwd_order_detail_di]
  metrics: []
- id: c006
  domain: 订单域
  question: 退款订单的品类分布
  sql: SELECT category_id, COUNT(*) AS refund_cnt FROM dwd_order_detail_di WHERE pay_status='refunded' AND dt >= '2026-07-01' GROUP BY category_id ORDER BY refund_cnt DESC
  tables: [dwd_order_detail_di]
  metrics: []
- id: c007
  domain: 订单域
  question: 单日GMV超过100万的日期
  sql: SELECT dt, gmv_amount FROM ads_order_daily_report_di WHERE gmv_amount > 1000000 AND dt >= '2026-07-01' ORDER BY gmv_amount DESC
  tables: [ads_order_daily_report_di]
  metrics: [GMV]
- id: c008
  domain: 订单域
  question: 各品类订单量占比
  sql: SELECT category_id, order_cnt, ROUND(order_cnt * 100.0 / SUM(order_cnt) OVER (), 2) AS pct FROM dws_category_order_di WHERE dt = '2026-07-31'
  tables: [dws_category_order_di]
  metrics: []
- id: c009
  domain: 订单域
  question: 7月订单总量与GMV汇总
  sql: SELECT SUM(order_cnt) AS total_orders, SUM(gmv_amount) AS total_gmv FROM dws_order_summary_di WHERE dt >= '2026-07-01' AND dt <= '2026-07-31'
  tables: [dws_order_summary_di]
  metrics: [GMV]
- id: c010
  domain: 订单域
  question: 各平台订单转化率（支付订单/下单）
  sql: SELECT platform, SUM(CASE WHEN pay_status IN ('paid','refunded') THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS conv_rate FROM dwd_order_detail_di WHERE dt >= '2026-07-01' GROUP BY platform
  tables: [dwd_order_detail_di]
  metrics: [支付率]
- id: c011
  domain: 用户域
  question: 近30天DAU趋势
  sql: SELECT dt, dau FROM dws_user_active_di WHERE dt >= '2026-07-02' AND dt <= '2026-07-31' ORDER BY dt
  tables: [dws_user_active_di]
  metrics: [DAU]
- id: c012
  domain: 用户域
  question: 7月新增用户数按天统计
  sql: SELECT dt, new_user_cnt FROM dws_user_active_di WHERE dt >= '2026-07-01' AND dt <= '2026-07-31' ORDER BY dt
  tables: [dws_user_active_di]
  metrics: [新增用户]
- id: c013
  domain: 用户域
  question: 各注册渠道新增用户对比
  sql: SELECT register_channel, COUNT(DISTINCT user_id) AS new_users FROM dwd_user_register_di WHERE dt >= '2026-07-01' GROUP BY register_channel ORDER BY new_users DESC
  tables: [dwd_user_register_di]
  metrics: [新增用户]
- id: c014
  domain: 用户域
  question: 用户行为类型分布（view/cart/buy）
  sql: SELECT behavior_type, SUM(behavior_cnt) AS cnt FROM dws_user_behavior_summary_di WHERE dt >= '2026-07-01' GROUP BY behavior_type ORDER BY cnt DESC
  tables: [dws_user_behavior_summary_di]
  metrics: []
- id: c015
  domain: 用户域
  question: 最近一周次日留存率
  sql: SELECT dt, retain_d1 FROM dws_user_retention_di WHERE dt >= '2026-07-25' AND dt <= '2026-07-31' ORDER BY dt
  tables: [dws_user_retention_di]
  metrics: [次日留存率]
- id: c016
  domain: 用户域
  question: 各城市注册用户数 Top10
  sql: SELECT city, COUNT(DISTINCT user_id) AS users FROM dwd_user_register_di WHERE dt >= '2026-07-01' GROUP BY city ORDER BY users DESC LIMIT 10
  tables: [dwd_user_register_di]
  metrics: []
- id: c017
  domain: 用户域
  question: 7月用户增长日报（新增/活跃/留存）
  sql: SELECT dt, new_user_cnt, active_user_cnt, retention_rate FROM ads_user_growth_report_di WHERE dt >= '2026-07-01' ORDER BY dt
  tables: [ads_user_growth_report_di]
  metrics: [次日留存率, 新增用户]
- id: c018
  domain: 用户域
  question: 购物车行为用户数趋势
  sql: SELECT dt, user_cnt FROM dws_user_behavior_summary_di WHERE behavior_type='cart' AND dt >= '2026-07-01' ORDER BY dt
  tables: [dws_user_behavior_summary_di]
  metrics: []
- id: c019
  domain: 用户域
  question: 近7天活跃用户中新增用户占比
  sql: SELECT dt, ROUND(new_user_cnt * 1.0 / active_user_cnt, 4) AS new_ratio FROM dws_user_active_di WHERE dt >= '2026-07-25' AND dt <= '2026-07-31'
  tables: [dws_user_active_di]
  metrics: [新增用户, DAU]
- id: c020
  domain: 用户域
  question: 7月累计注册用户数
  sql: SELECT COUNT(DISTINCT user_id) AS total_new FROM dwd_user_register_di WHERE dt >= '2026-07-01' AND dt <= '2026-07-31'
  tables: [dwd_user_register_di]
  metrics: [新增用户]
- id: c021
  domain: 商品域
  question: 商品GMV Top20
  sql: SELECT product_id, SUM(gmv_amount) AS gmv FROM dws_product_gmv_di WHERE dt >= '2026-07-01' GROUP BY product_id ORDER BY gmv DESC LIMIT 20
  tables: [dws_product_gmv_di]
  metrics: [GMV]
- id: c022
  domain: 商品域
  question: 品类浏览热度排行
  sql: SELECT c.category_name, SUM(v.view_cnt) AS views FROM dws_product_view_di v JOIN dim_product_info p ON v.product_id=p.product_id JOIN dim_category_info c ON p.category_id=c.category_id WHERE v.dt >= '2026-07-01' GROUP BY c.category_name ORDER BY views DESC
  tables: [dws_product_view_di, dim_product_info, dim_category_info]
  metrics: []
- id: c023
  domain: 商品域
  question: 昨日商品GMV排名前50
  sql: SELECT product_id, gmv_rank, gmv_amount FROM dws_product_gmv_di WHERE dt='2026-07-31' ORDER BY gmv_amount DESC LIMIT 50
  tables: [dws_product_gmv_di]
  metrics: [GMV]
- id: c024
  domain: 商品域
  question: 品牌GMV汇总
  sql: SELECT p.brand, SUM(g.gmv_amount) AS gmv FROM dws_product_gmv_di g JOIN dim_product_info p ON g.product_id=p.product_id WHERE g.dt >= '2026-07-01' GROUP BY p.brand ORDER BY gmv DESC
  tables: [dws_product_gmv_di, dim_product_info]
  metrics: [GMV]
- id: c025
  domain: 商品域
  question: 浏览转购买率最高的商品
  sql: SELECT g.product_id, g.pay_user_cnt * 1.0 / NULLIF(v.view_user_cnt,0) AS conv FROM dws_product_gmv_di g JOIN dws_product_view_di v ON g.product_id=v.product_id AND g.dt=v.dt WHERE g.dt='2026-07-31' ORDER BY conv DESC LIMIT 10
  tables: [dws_product_gmv_di, dws_product_view_di]
  metrics: []
- id: c026
  domain: 商品域
  question: 各品类在售商品数
  sql: SELECT c.category_name, COUNT(*) AS product_cnt FROM dim_product_info p JOIN dim_category_info c ON p.category_id=c.category_id WHERE p.status='on' GROUP BY c.category_name
  tables: [dim_product_info, dim_category_info]
  metrics: []
- id: c027
  domain: 商品域
  question: 近30天浏览量趋势
  sql: SELECT dt, SUM(view_cnt) AS views FROM dws_product_view_di WHERE dt >= '2026-07-02' AND dt <= '2026-07-31' GROUP BY dt ORDER BY dt
  tables: [dws_product_view_di]
  metrics: []
- id: c028
  domain: 商品域
  question: 7月GMV最高的商品及其品牌
  sql: SELECT p.product_id, p.product_name, p.brand, SUM(g.gmv_amount) AS gmv FROM dws_product_gmv_di g JOIN dim_product_info p ON g.product_id=p.product_id WHERE g.dt >= '2026-07-01' GROUP BY 1,2,3 ORDER BY gmv DESC LIMIT 10
  tables: [dws_product_gmv_di, dim_product_info]
  metrics: [GMV]
- id: c029
  domain: 商品域
  question: 商品浏览量排名Top100
  sql: SELECT product_id, view_rank FROM ads_product_ranking_di WHERE dt='2026-07-31' AND view_rank <= 100 ORDER BY view_rank
  tables: [ads_product_ranking_di]
  metrics: []
- id: c030
  domain: 商品域
  question: 价格带商品销量分布
  sql: SELECT CASE WHEN p.price < 100 THEN '0-100' WHEN p.price < 300 THEN '100-300' WHEN p.price < 600 THEN '300-600' ELSE '600+' END AS price_band, SUM(g.order_cnt) AS orders FROM dws_product_gmv_di g JOIN dim_product_info p ON g.product_id=p.product_id WHERE g.dt >= '2026-07-01' GROUP BY 1 ORDER BY orders DESC
  tables: [dws_product_gmv_di, dim_product_info]
  metrics: []
- id: c031
  domain: 支付域
  question: 7月支付总额与订单数
  sql: SELECT SUM(pay_amount) AS total, SUM(pay_order_cnt) AS orders FROM dws_payment_summary_di WHERE dt >= '2026-07-01' AND dt <= '2026-07-31'
  tables: [dws_payment_summary_di]
  metrics: []
- id: c032
  domain: 支付域
  question: 各支付渠道金额占比
  sql: SELECT pay_channel, SUM(pay_amount) AS amount, ROUND(SUM(pay_amount)*100.0/SUM(SUM(pay_amount)) OVER (),2) AS pct FROM dws_payment_channel_di WHERE dt >= '2026-07-01' GROUP BY pay_channel ORDER BY amount DESC
  tables: [dws_payment_channel_di]
  metrics: []
- id: c033
  domain: 支付域
  question: 退款率趋势（近30天）
  sql: SELECT p.dt, r.refund_amount, p.pay_amount, ROUND(r.refund_amount/p.pay_amount, 4) AS refund_rate FROM dws_payment_summary_di p JOIN dws_refund_summary_di r ON p.dt=r.dt WHERE p.dt >= '2026-07-02' ORDER BY p.dt
  tables: [dws_payment_summary_di, dws_refund_summary_di]
  metrics: [退款率]
- id: c034
  domain: 支付域
  question: 支付失败订单明细数
  sql: SELECT COUNT(*) AS failed_cnt FROM dwd_payment_detail_di WHERE pay_status='failed' AND dt >= '2026-07-01'
  tables: [dwd_payment_detail_di]
  metrics: []
- id: c035
  domain: 支付域
  question: 各渠道支付用户数
  sql: SELECT pay_channel, COUNT(DISTINCT user_id) AS users FROM dwd_payment_detail_di WHERE pay_status='success' AND dt >= '2026-07-01' GROUP BY pay_channel
  tables: [dwd_payment_detail_di]
  metrics: []
- id: c036
  domain: 支付域
  question: 净支付额（支付-退款）日报
  sql: SELECT dt, pay_amount, refund_amount, net_amount FROM ads_payment_daily_report_di WHERE dt >= '2026-07-01' ORDER BY dt
  tables: [ads_payment_daily_report_di]
  metrics: []
- id: c037
  domain: 支付域
  question: 退款金额最高的10天
  sql: SELECT dt, refund_amount FROM dws_refund_summary_di WHERE dt >= '2026-07-01' ORDER BY refund_amount DESC LIMIT 10
  tables: [dws_refund_summary_di]
  metrics: []
- id: c038
  domain: 支付域
  question: 大额支付订单（>500元）数量
  sql: SELECT COUNT(*) AS big_orders FROM dwd_payment_detail_di WHERE pay_amount > 500 AND pay_status='success' AND dt >= '2026-07-01'
  tables: [dwd_payment_detail_di]
  metrics: []
- id: c039
  domain: 支付域
  question: 7月各渠道支付趋势（按周汇总）
  sql: SELECT DATE_TRUNC('week', dt) AS week, pay_channel, SUM(pay_amount) AS amount FROM dws_payment_channel_di WHERE dt >= '2026-07-01' GROUP BY 1,2 ORDER BY 1
  tables: [dws_payment_channel_di]
  metrics: []
- id: c040
  domain: 支付域
  question: 退款订单平均退款金额
  sql: SELECT ROUND(AVG(refund_amount), 2) AS avg_refund FROM dwd_refund_detail_di WHERE dt >= '2026-07-01'
  tables: [dwd_refund_detail_di]
  metrics: []
- id: c041
  domain: 物流域
  question: 7月发货与签收订单数
  sql: SELECT SUM(ship_order_cnt) AS shipped, SUM(sign_order_cnt) AS signed FROM dws_logistics_summary_di WHERE dt >= '2026-07-01' AND dt <= '2026-07-31'
  tables: [dws_logistics_summary_di]
  metrics: []
- id: c042
  domain: 物流域
  question: 各物流公司发货量对比
  sql: SELECT logistics_company, SUM(ship_cnt) AS shipped FROM dws_logistics_company_di WHERE dt >= '2026-07-01' GROUP BY logistics_company ORDER BY shipped DESC
  tables: [dws_logistics_company_di]
  metrics: []
- id: c043
  domain: 物流域
  question: 平均配送时长趋势
  sql: SELECT dt, avg_delivery_days FROM dws_logistics_summary_di WHERE dt >= '2026-07-01' ORDER BY dt
  tables: [dws_logistics_summary_di]
  metrics: []
- id: c044
  domain: 物流域
  question: 物流准时率日报
  sql: SELECT dt, on_time_rate FROM ads_logistics_daily_report_di WHERE dt >= '2026-07-01' ORDER BY dt
  tables: [ads_logistics_daily_report_di]
  metrics: [履约准时率]
- id: c045
  domain: 物流域
  question: 各仓库发货量
  sql: SELECT warehouse_id, COUNT(*) AS shipped FROM dwd_logistics_shipped_di WHERE dt >= '2026-07-01' GROUP BY warehouse_id ORDER BY shipped DESC
  tables: [dwd_logistics_shipped_di]
  metrics: []
- id: c046
  domain: 物流域
  question: 签收率（签收/发货）日报
  sql: SELECT dt, ROUND(sign_order_cnt * 1.0 / ship_order_cnt, 4) AS sign_rate FROM dws_logistics_summary_di WHERE dt >= '2026-07-01'
  tables: [dws_logistics_summary_di]
  metrics: []
- id: c047
  domain: 物流域
  question: 在途订单数（已发货未签收）
  sql: SELECT COUNT(*) AS in_transit FROM dwd_logistics_tracking_di WHERE status='transit' AND dt >= '2026-07-01'
  tables: [dwd_logistics_tracking_di]
  metrics: []
- id: c048
  domain: 物流域
  question: 各物流公司签收率对比
  sql: SELECT logistics_company, ROUND(SUM(sign_cnt)*1.0/SUM(ship_cnt), 4) AS sign_rate FROM dws_logistics_company_di WHERE dt >= '2026-07-01' GROUP BY logistics_company
  tables: [dws_logistics_company_di]
  metrics: []
- id: c049
  domain: 订单域
  question: 7月GMV最高的5个品类
  sql: SELECT c.category_name, SUM(o.gmv_amount) AS gmv FROM dws_category_order_di o JOIN dim_category_info c ON o.category_id=c.category_id WHERE o.dt >= '2026-07-01' GROUP BY c.category_name ORDER BY gmv DESC LIMIT 5
  tables: [dws_category_order_di, dim_category_info]
  metrics: [GMV]
- id: c050
  domain: 订单域
  question: 每日订单均价
  sql: SELECT dt, ROUND(gmv_amount/pay_order_cnt, 2) AS avg_order FROM dws_order_summary_di WHERE dt >= '2026-07-01' ORDER BY dt
  tables: [dws_order_summary_di]
  metrics: [客单价]
```

- [ ] **Step 4: 写 `dataagent/rag/cases.py`**

```python
"""历史取数案例库加载。案例 = 需求 + SQL 配对，RAG 检索素材。"""
from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass(frozen=True)
class Case:
    id: str
    domain: str
    question: str
    sql: str
    tables: list[str]
    metrics: list[str]

    def text(self) -> str:
        """入库/检索用文本：需求描述为核心语义载体。"""
        parts = [f"需求: {self.question}", f"SQL: {self.sql}"]
        if self.metrics:
            parts.append(f"指标: {', '.join(self.metrics)}")
        return "\n".join(parts)


def load_cases(path: str = "data/cases/cases.yaml") -> list[Case]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"案例文件不存在: {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    return [Case(**item) for item in raw]
```

- [ ] **Step 5: 跑测试确认通过**

Run: `pytest tests/test_cases.py -v`
预期：3 passed

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: 50 条历史取数案例库 + 加载器"
```

### Task 7: MCP 元数据服务核心（查询 + 权限过滤）

**Files:**
- Create: `dw-dataagent/dataagent/mcp_server/__init__.py`
- Create: `dw-dataagent/dataagent/mcp_server/metadata.py`
- Test: `dw-dataagent/tests/test_mcp_tools.py`

**Interfaces:**
- Produces: `dataagent.mcp_server.metadata.query_table_list(role: str, domain: str | None) -> list[dict]`
- Produces: `query_table_schema(role: str, table_name: str) -> dict`（含 columns 列表）
- Produces: `query_lineage(role: str, table_name: str) -> list[dict]`（上游+下游）
- Produces: `query_metric_definition(metric_name: str) -> dict`
- Consumes: `dataagent.warehouse.schema.{TABLES, LINEAGE, METRICS}`、`dataagent.permissions.filter_tables_by_role`

- [ ] **Step 1: 写失败测试 `tests/test_mcp_tools.py`**

```python
"""MCP 元数据查询函数测试（含权限过滤）。"""
import pytest
from dataagent.mcp_server.metadata import (
    query_table_list, query_table_schema, query_lineage,
    query_metric_definition, TableNotFoundError,
)


def test_table_list_filtered_by_role():
    tables = query_table_list("data_analyst", None)
    domains = {t["domain"] for t in tables}
    assert domains == {"订单域", "用户域", "商品域"}


def test_table_list_by_domain():
    tables = query_table_list("data_analyst", "订单域")
    assert all(t["domain"] == "订单域" for t in tables)
    assert len(tables) == 6


def test_table_schema_returns_columns():
    schema = query_table_schema("data_analyst", "dws_order_summary_di")
    assert schema["table_name"] == "dws_order_summary_di"
    cols = {c["name"] for c in schema["columns"]}
    assert {"gmv_amount", "order_cnt", "dt"} <= cols


def test_schema_denied_for_unauthorized_table():
    # data_analyst 无支付域权限 → 表不存在（源头阻断，不泄露存在性）
    with pytest.raises(TableNotFoundError):
        query_table_schema("data_analyst", "dws_payment_summary_di")


def test_schema_denied_returns_not_found_for_unknown_table():
    with pytest.raises(TableNotFoundError):
        query_table_schema("admin", "no_such_table")


def test_finance_can_access_payment():
    schema = query_table_schema("finance_analyst", "dws_payment_summary_di")
    assert schema["table_name"] == "dws_payment_summary_di"


def test_lineage_returns_upstream_and_downstream():
    result = query_lineage("data_analyst", "dws_order_summary_di")
    upstream = {e["source_table"] for e in result if e["direction"] == "upstream"}
    downstream = {e["target_table"] for e in result if e["direction"] == "downstream"}
    assert "dwd_order_detail_di" in upstream
    assert "ads_order_daily_report_di" in downstream


def test_metric_definition():
    m = query_metric_definition("GMV")
    assert m["metric_name"] == "GMV"
    assert "支付成功" in m["definition"]
    assert m["formula"]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_mcp_tools.py -v`
预期：FAIL（ModuleNotFoundError）

- [ ] **Step 3: 写 `dataagent/mcp_server/metadata.py`**

```python
"""元数据查询核心：MCP 工具的业务逻辑层（协议无关，可单测）。

生产对标：此模块对应元数据中心（DataHub/Atlas/自研）的 API 封装；
MCP Server 只是它的协议暴露层，对接真实元数据中心时此层签名不变。
"""
from dataagent.warehouse.schema import TABLES, LINEAGE, METRICS
from dataagent.permissions import filter_tables_by_role


class TableNotFoundError(Exception):
    """表不存在或无权限（两者不可区分，避免侧信道泄露）。"""


def query_table_list(role: str, domain: str | None = None) -> list[dict]:
    visible = filter_tables_by_role(role, TABLES)
    result = [
        {
            "table_name": t.name, "domain": t.domain, "layer": t.layer,
            "granularity": t.grain, "description": t.description,
            "partition_col": t.partition_col,
        }
        for t in visible.values()
        if domain is None or t.domain == domain
    ]
    return sorted(result, key=lambda r: r["table_name"])


def query_table_schema(role: str, table_name: str) -> dict:
    visible = filter_tables_by_role(role, TABLES)
    spec = visible.get(table_name)
    if spec is None:
        raise TableNotFoundError(table_name)
    return {
        "table_name": spec.name,
        "domain": spec.domain,
        "layer": spec.layer,
        "granularity": spec.grain,
        "description": spec.description,
        "partition_col": spec.partition_col,
        "columns": [
            {"name": c.name, "data_type": c.data_type, "comment": c.comment}
            for c in spec.columns
        ],
    }


def query_lineage(role: str, table_name: str) -> list[dict]:
    visible = filter_tables_by_role(role, TABLES)
    if table_name not in visible:
        raise TableNotFoundError(table_name)
    result = []
    for e in LINEAGE:
        if e.source == table_name:
            result.append({"direction": "downstream",
                           "source_table": e.source,
                           "target_table": e.target,
                           "relation": e.relation})
        if e.target == table_name:
            result.append({"direction": "upstream",
                           "source_table": e.source,
                           "target_table": e.target,
                           "relation": e.relation})
    return result


def query_metric_definition(metric_name: str) -> dict:
    m = METRICS.get(metric_name)
    if m is None:
        return {"metric_name": metric_name,
                "definition": "未收录该指标", "formula": ""}
    return {"metric_name": m.name, "definition": m.definition,
            "formula": m.formula}
```

- [ ] **Step 4: 跑测试确认通过**

Run: `pytest tests/test_mcp_tools.py -v`
预期：8 passed

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: MCP 元数据查询核心（权限过滤 + 血缘 + 口径）"
```

### Task 8: MCP Server 服务化（SSE + FastAPI 挂载）

**Files:**
- Create: `dw-dataagent/dataagent/mcp_server/server.py`
- Test: `dw-dataagent/tests/test_mcp_server_sse.py`（标记 integration）

**Interfaces:**
- Produces: `dataagent.mcp_server.server.app`（FastAPI 实例，`/mcp/sse` 为 SSE 端点）
- Produces: `dataagent.mcp_server.server.mcp`（MCP Server 对象，供 stdio 模式复用）
- Consumes: `dataagent.mcp_server.metadata` 四个查询函数

- [ ] **Step 1: 写 `dataagent/mcp_server/server.py`**

```python
"""MCP Server 服务化：SSE 模式挂载 FastAPI，供 Agent 远程调用。

生产对标：SSE 支持多客户端远程访问，是企业 MCP 标准部署形态；
stdio 仅本地 CLI 调试用（`mcp run dataagent.mcp_server.server:mcp`）。
"""
import contextvars
import starlette.requests
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import TextContent, Tool
from starlette.applications import Starlette
from starlette.routing import Mount, Route

from dataagent.mcp_server.metadata import (
    query_table_list, query_table_schema, query_lineage,
    query_metric_definition, TableNotFoundError,
)
from dataagent.permissions import resolve_role

mcp = Server("dw-metadata-server")

# SSE 会话内传递用户上下文（角色）
_session_ctx = contextvars.ContextVar("session_role", default=None)


def _role() -> str:
    return _session_ctx.get() or "data_analyst"


@mcp.list_tools()
async def handle_list_tools() -> list[Tool]:
    return [
        Tool(
            name="query_table_list",
            description="按主题域查询可见表清单。当需要了解有哪些表可用、表属于哪个主题域时使用。",
            inputSchema={
                "type": "object",
                "properties": {
                    "domain": {"type": "string",
                               "description": "主题域：订单域/用户域/商品域/支付域/物流域，可空"}
                },
                "required": []
            },
        ),
        Tool(
            name="query_table_schema",
            description="查询表的字段结构。当生成 SQL 前需要确认表有哪些字段、字段类型、分区字段时使用。",
            inputSchema={
                "type": "object",
                "properties": {
                    "table_name": {"type": "string", "description": "表名"}
                },
                "required": ["table_name"]
            },
        ),
        Tool(
            name="query_lineage",
            description="查询表的血缘关系（上游+下游）。当需要了解表的依赖链路时使用。",
            inputSchema={
                "type": "object",
                "properties": {
                    "table_name": {"type": "string", "description": "表名"}
                },
                "required": ["table_name"]
            },
        ),
        Tool(
            name="query_metric_definition",
            description="查询指标口径定义。当需求中出现指标（GMV/DAU/支付率等）且需要确认计算逻辑时使用。",
            inputSchema={
                "type": "object",
                "properties": {
                    "metric_name": {"type": "string", "description": "指标名"}
                },
                "required": ["metric_name"]
            },
        ),
    ]


@mcp.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
    import json
    role = _role()
    try:
        if name == "query_table_list":
            result = query_table_list(role, arguments.get("domain"))
        elif name == "query_table_schema":
            result = query_table_schema(role, arguments["table_name"])
        elif name == "query_lineage":
            result = query_lineage(role, arguments["table_name"])
        elif name == "query_metric_definition":
            result = query_metric_definition(arguments["metric_name"])
        else:
            return [TextContent(type="text", text=f"未知工具: {name}")]
    except TableNotFoundError as e:
        return [TextContent(type="text",
                            text=f"错误: 表 {e} 不存在或无访问权限")]
    return [TextContent(type="text",
                        text=json.dumps(result, ensure_ascii=False, indent=2))]


# SSE 传输实例必须是单例：SSE 连接与 POST 消息共用同一个路由表
sse = SseServerTransport("/messages/")


async def handle_sse(request: starlette.requests.Request):
    # 从 Header 提取用户，解析角色，注入会话上下文
    user = request.headers.get("x-user", "default_user")
    role = resolve_role(user)
    _session_ctx.set(role)

    async with sse.connect_sse(
        request.scope, request.receive, request._send) as (
        read_stream, write_stream):
        await mcp.run(read_stream, write_stream,
                      mcp.create_initialization_options())


app = Starlette(
    routes=[
        Route("/health", endpoint=lambda req: starlette.responses.JSONResponse(
            {"status": "ok"})),
        Route("/mcp/sse", endpoint=handle_sse),
        Mount("/messages/", app=sse.handle_post_message),
    ]
)
```

- [ ] **Step 2: 写集成测试 `tests/test_mcp_server_sse.py`**

```python
"""MCP SSE 服务集成测试（需手动启动服务：uvicorn dataagent.mcp_server.server:app --port 8001）。"""
import json
import pytest
from langchain_mcp_adapters.client import MultiServerMCPClient

pytestmark = pytest.mark.integration


@pytest.fixture
def mcp_client():
    client = MultiServerMCPClient({
        "metadata": {
            "transport": "sse",
            "url": "http://localhost:8001/mcp/sse",
            "headers": {"x-user": "finance_wang"},
        }
    })
    return client


@pytest.mark.asyncio
async def test_tools_discovered(mcp_client):
    tools = await mcp_client.get_tools()
    names = {t.name for t in tools}
    assert names == {"query_table_list", "query_table_schema",
                     "query_lineage", "query_metric_definition"}


@pytest.mark.asyncio
async def test_call_tool_with_permission(mcp_client):
    tools = await mcp_client.get_tools()
    schema_tool = next(t for t in tools if t.name == "query_table_schema")
    result = await schema_tool.ainvoke({"table_name": "dws_payment_summary_di"})
    assert "pay_amount" in result
```

- [ ] **Step 3: 启动服务并跑集成测试**

```bash
source .venv/bin/activate
uvicorn dataagent.mcp_server.server:app --port 8001 &
pytest tests/test_mcp_server_sse.py -v -m integration
curl http://localhost:8001/health
```

预期：2 passed；health 返回 `{"status":"ok"}`

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "feat: MCP Server SSE 服务化（FastAPI 挂载 + 会话级权限）"
```

## Day 3-4 · RAG 检索链路

### Task 9: RAG 索引构建（BGE + Milvus）

**Files:**
- Create: `dw-dataagent/dataagent/rag/indexer.py`
- Create: `dw-dataagent/scripts/build_rag_index.py`
- Test: `dw-dataagent/tests/test_indexer.py`（单元：文档构造）+ 手动验收脚本

**Interfaces:**
- Produces: `dataagent.rag.indexer.build_documents(cases: list[Case]) -> list[dict]`（{id, text, metadata}）
- Produces: `dataagent.rag.indexer.MilvusIndexer(settings).index(cases)`（连接、建 collection、插入）
- Consumes: `dataagent.rag.cases.load_cases`

- [ ] **Step 1: 写失败测试 `tests/test_indexer.py`**

```python
"""索引构建单元测试（不依赖 Milvus）。"""
from dataagent.rag.indexer import build_documents
from dataagent.rag.cases import load_cases


def test_build_documents_count():
    docs = build_documents(load_cases())
    assert len(docs) == 50


def test_document_has_id_and_text():
    for doc in build_documents(load_cases()):
        assert doc["id"].startswith("c")
        assert doc["text"]
        assert doc["metadata"]["domain"]


def test_text_contains_question_and_sql():
    docs = build_documents(load_cases())
    assert "GMV" in docs[0]["text"]
    assert "SELECT" in docs[0]["text"]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_indexer.py -v`
预期：FAIL

- [ ] **Step 3: 写 `dataagent/rag/indexer.py`**

```python
"""RAG 索引构建：案例 → BGE Embedding → Milvus 入库。

分块策略：案例（需求+SQL 配对）是语义完整单元，整条入库不切碎。
生产对标：知识库万级时按 domain 字段做 Partition Key，HNSW 参数
（M=16, efConstruction=200）平衡召回与写入性能。
"""
from pymilvus import (
    Collection, CollectionSchema, DataType, FieldSchema, connections, utility,
)
from sentence_transformers import SentenceTransformer

from dataagent.rag.cases import Case

EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"
DIM = 1024
_embedder = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def build_documents(cases: list[Case]) -> list[dict]:
    return [
        {
            "id": c.id,
            "text": c.text(),
            "metadata": {
                "domain": c.domain,
                "tables": ",".join(c.tables),
                "metrics": ",".join(c.metrics),
            },
        }
        for c in cases
    ]


class MilvusIndexer:
    def __init__(self, host: str, port: int, collection: str):
        self.host, self.port = host, port
        self.collection_name = collection

    def connect(self):
        connections.connect(alias="default", host=self.host, port=self.port)

    def create_collection(self, drop_if_exists: bool = False):
        if utility.has_collection(self.collection_name):
            if drop_if_exists:
                utility.drop_collection(self.collection_name)
            else:
                return Collection(self.collection_name)
        fields = [
            FieldSchema("id", DataType.VARCHAR, is_primary=True, max_length=32),
            FieldSchema("text", DataType.VARCHAR, max_length=4096),
            FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=DIM),
            FieldSchema("domain", DataType.VARCHAR, max_length=16),
            FieldSchema("tables", DataType.VARCHAR, max_length=512),
            FieldSchema("metrics", DataType.VARCHAR, max_length=256),
        ]
        schema = CollectionSchema(fields, description="数仓取数 SQL 案例库")
        collection = Collection(self.collection_name, schema)
        index_params = {
            "metric_type": "IP",  # 内积（embedding 已 normalize → 等价余弦）
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 200},
        }
        collection.create_index("embedding", index_params)
        return collection

    def index(self, cases: list[Case]) -> int:
        self.connect()
        collection = self.create_collection(drop_if_exists=True)
        docs = build_documents(cases)
        embedder = get_embedder()
        embeddings = embedder.encode(
            [d["text"] for d in docs],
            normalize_embeddings=True, show_progress_bar=True)
        collection.insert([
            [d["id"] for d in docs],
            [d["text"] for d in docs],
            embeddings.tolist(),
            [d["metadata"]["domain"] for d in docs],
            [d["metadata"]["tables"] for d in docs],
            [d["metadata"]["metrics"] for d in docs],
        ])
        collection.flush()
        collection.load()
        return len(docs)
```

- [ ] **Step 4: 写 `scripts/build_rag_index.py`**

```python
"""构建 RAG 案例索引：python scripts/build_rag_index.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataagent.config import load_config
from dataagent.rag.cases import load_cases
from dataagent.rag.indexer import MilvusIndexer


def main():
    settings = load_config()
    cases = load_cases(settings.cases_path)
    indexer = MilvusIndexer(
        settings.milvus.host, settings.milvus.port, settings.milvus.collection)
    n = indexer.index(cases)
    print(f"[rag] 已入库 {n} 条案例到 Milvus collection={settings.milvus.collection}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: 跑单元测试**

Run: `pytest tests/test_indexer.py -v`
预期：3 passed

- [ ] **Step 6: 启动 Milvus 并构建索引（手动验收）**

```bash
# Milvus standalone compose（部署文件见 Task 17；此处先手写快速启动）
docker compose -f deploy/infra-compose.yml up -d milvus 2>/dev/null || \
  (mkdir -p /tmp/milvus && cat > /tmp/milvus/compose.yml <<'EOF'
services:
  etcd:
    image: quay.io/coreos/etcd:v3.5.16
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
    command: etcd -advertise-client-urls=http://etcd:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd
  minio:
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    command: minio server /minio_data
  standalone:
    image: milvusdb/milvus:v2.5.4
    command: ["milvus", "run", "standalone"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    depends_on:
      - etcd
      - minio
    ports:
      - "19530:19530"
      - "9091:9091"
EOF
  docker compose -f /tmp/milvus/compose.yml up -d)

# 等待 Milvus 就绪
until curl -s http://localhost:9091/healthz | grep -q OK; do sleep 3; done

HF_ENDPOINT=https://hf-mirror.com python scripts/build_rag_index.py
```

预期：`[rag] 已入库 50 条案例`（首次运行含 BGE 模型下载，约 1.3GB，耐心等待）

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "feat: RAG 索引构建（BGE-large-zh + Milvus HNSW）"
```

### Task 10: 混合检索（向量 + BM25 + RRF）

**Files:**
- Create: `dw-dataagent/dataagent/rag/retriever.py`
- Test: `dw-dataagent/tests/test_retriever.py`（RRF 纯函数单测）+ 手动检索质量评估

**Interfaces:**
- Produces: `dataagent.rag.retriever.rrf_merge(ranked_lists: list[list[str]], k: int = 60) -> list[str]`（RRF 融合排名）
- Produces: `dataagent.rag.retriever.HybridRetriever.search(question: str, top_k: int = 5) -> list[dict]`（{id, text, domain, score}）
- Consumes: `dataagent.rag.indexer.{MilvusIndexer, get_embedder}`

- [ ] **Step 1: 写失败测试 `tests/test_retriever.py`**

```python
"""RRF 融合纯函数测试。"""
from dataagent.rag.retriever import rrf_merge


def test_rrf_merges_two_lists():
    merged = rrf_merge([["a", "b", "c"], ["b", "a", "d"]], k=60)
    # b 在两个列表中都排第 1-2 → 融合后应排第一
    assert merged[0] == "b"
    assert set(merged) == {"a", "b", "c", "d"}


def test_rrf_handles_empty_list():
    assert rrf_merge([[], []], k=60) == []


def test_rrf_stable_for_single_list():
    merged = rrf_merge([["x", "y", "z"]], k=60)
    assert merged == ["x", "y", "z"]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_retriever.py -v`
预期：FAIL

- [ ] **Step 3: 写 `dataagent/rag/retriever.py`**

```python
"""混合检索：Milvus 向量 ANN + BM25 关键词 + RRF 融合。

生产对标：向量检索找"语义相似案例"，BM25 找"专有名词精确匹配"
（表名/指标名），RRF 融合两者互补——这是企业 RAG 检索层的标准形态。
"""
from collections import defaultdict

from pymilvus import Collection, connections

from dataagent.rag.cases import Case, load_cases
from dataagent.rag.indexer import get_embedder


def rrf_merge(ranked_lists: list[list[str]], k: int = 60) -> list[str]:
    """Reciprocal Rank Fusion：score(d) = Σ 1/(k + rank_i(d))"""
    scores: dict[str, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] += 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda d: scores[d], reverse=True)


class HybridRetriever:
    def __init__(self, host: str, port: int, collection: str,
                 cases: list[Case] | None = None):
        self.host, self.port, self.collection_name = host, port, collection
        self.cases = cases or load_cases()
        self._bm25 = None
        self._case_ids = [c.id for c in self.cases]
        self._corpus = [c.text() for c in self.cases]

    def _get_bm25(self):
        if self._bm25 is None:
            from rank_bm25 import BM25Okapi
            import jieba  # 中文分词
            tokenized = [list(jieba.cut(t)) for t in self._corpus]
            self._bm25 = BM25Okapi(tokenized)
        return self._bm25

    def _vector_search(self, question: str, top_k: int) -> list[str]:
        connections.connect(alias="default", host=self.host, port=self.port)
        collection = Collection(self.collection_name)
        collection.load()  # 幂等防御：服务重启后确保 collection 已加载
        query_vec = get_embedder().encode(
            [question], normalize_embeddings=True).tolist()
        hits = collection.search(
            query_vec, "embedding",
            {"metric_type": "IP", "params": {"ef": 128}},
            limit=top_k)
        return [h.id for h in hits[0]]

    def _bm25_search(self, question: str, top_k: int) -> list[str]:
        import jieba
        scores = self._get_bm25().get_scores(list(jieba.cut(question)))
        ranked = sorted(zip(self._case_ids, scores),
                        key=lambda x: x[1], reverse=True)[:top_k]
        return [cid for cid, _ in ranked]

    def search(self, question: str, top_k: int = 5) -> list[dict]:
        """返回 [{id, question, sql, domain, score}]，score 为 RRF 分。"""
        vec_rank = self._vector_search(question, top_k * 4)
        bm25_rank = self._bm25_search(question, top_k * 4)
        merged = rrf_merge([vec_rank, bm25_rank])[:top_k]
        by_id = {c.id: c for c in self.cases}
        return [
            {"id": cid, "question": by_id[cid].question,
             "sql": by_id[cid].sql, "domain": by_id[cid].domain}
            for cid in merged if cid in by_id
        ]
```

- [ ] **Step 4: 跑单测确认通过**

Run: `pytest tests/test_retriever.py -v`
预期：3 passed

- [ ] **Step 5: 检索质量人工评估（手动验收脚本）**

```bash
source .venv/bin/activate
python - <<'EOF'
from dataagent.config import load_config
from dataagent.rag.retriever import HybridRetriever
s = load_config()
r = HybridRetriever(s.milvus.host, s.milvus.port, s.milvus.collection)
tests = [
    "统计近30天各平台GMV趋势",
    "各支付渠道的支付金额对比",
    "商品浏览转购买率分析",
    "物流签收率统计",
    "用户留存率周报",
]
for q in tests:
    print("=" * 60)
    print("需求:", q)
    for i, hit in enumerate(r.search(q, top_k=5), 1):
        print(f"  Top{i}: [{hit['domain']}] {hit['question']}")
EOF
```

预期：每条需求的 Top1 案例域与需求匹配（如物流问题 Top1 是物流域案例）。若 Top1 域不匹配率 > 2/5，检查 Milvus collection 是否已 load（`build_rag_index.py` 已 load）。

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: 混合检索（Milvus ANN + BM25 + RRF 融合）"
```

## Day 5 · Agent 核心

### Task 11: 模型路由层

**Files:**
- Create: `dw-dataagent/dataagent/llm/__init__.py`
- Create: `dw-dataagent/dataagent/llm/router.py`
- Test: `dw-dataagent/tests/test_router.py`

**Interfaces:**
- Produces: `dataagent.llm.router.LLMRouter(settings).get_model(task: str) -> BaseChatModel`（task ∈ task_parse/sql_generate/critic_review；返回 langchain ChatModel）
- Produces: `dataagent.llm.router.LLMRouter.get_model_with_fallback(task)`（主模型构造失败时回退 fallback provider）

- [ ] **Step 1: 写失败测试 `tests/test_router.py`**

```python
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
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_router.py -v`
预期：FAIL

- [ ] **Step 3: 写 `dataagent/llm/router.py`**

```python
"""模型路由层（Tiered Model Stack）。

生产对标：大厂模型网关的简化实现——按任务类型路由到不同模型，
成本与质量分层；provider 抽象隔离 API 与本地部署差异，
生产私有化部署（vLLM）时业务代码零改动。
"""
import os

from langchain_core.language_models import BaseChatModel

from dataagent.config import Settings


class LLMRouter:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._cache: dict[str, BaseChatModel] = {}

    def get_model(self, task: str) -> BaseChatModel:
        """按任务类型返回模型（实例缓存）。task ∈ task_parse/sql_generate/critic_review"""
        routing = self.settings.llm.routing
        provider_name = routing.get(task) or routing.get("sql_generate") or "deepseek"
        return self._get_provider(provider_name)

    def get_model_with_fallback(self, task: str) -> BaseChatModel:
        """主模型不可用时降级到 fallback provider。"""
        routing = self.settings.llm.routing
        provider_name = routing.get(task) or routing.get("sql_generate") or "deepseek"
        try:
            return self._get_provider(provider_name)
        except Exception:
            fallback = routing.get("fallback")
            if fallback and fallback != provider_name:
                return self._get_provider(fallback)
            raise

    def _get_provider(self, name: str) -> BaseChatModel:
        if name in self._cache:
            return self._cache[name]
        provider = self.settings.llm.providers[name]
        if provider.type == "api":
            api_key = os.environ.get(provider.api_key_env)
            if not api_key:
                raise ValueError(
                    f"环境变量 {provider.api_key_env} 未设置，无法使用 {name}")
            from langchain_deepseek import ChatDeepSeek
            model = ChatDeepSeek(
                model=provider.model,
                api_base=provider.base_url,
                api_key=api_key,
                temperature=0.1,
            )
        elif provider.type == "local":
            from langchain_ollama import ChatOllama
            model = ChatOllama(
                model=provider.model,
                base_url=provider.base_url,
                temperature=0.1,
            )
        else:
            raise ValueError(f"未知 provider 类型: {provider.type}")
        self._cache[name] = model
        return model
```

- [ ] **Step 4: 跑测试确认通过**

Run: `pytest tests/test_router.py -v`
预期：3 passed（需设置 DEEPSEEK_API_KEY 环境变量；fixture 已 monkeypatch）

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: 模型路由层（Tiered Model Stack + 降级）"
```

### Task 12: SQL 校验器（SQLGlot 规则引擎）

**Files:**
- Create: `dw-dataagent/dataagent/guardrails/__init__.py`
- Create: `dw-dataagent/dataagent/guardrails/sql_validator.py`
- Test: `dw-dataagent/tests/test_sql_validator.py`

**Interfaces:**
- Produces: `dataagent.guardrails.sql_validator.ValidationResult`（passed: bool, errors: list[str]）
- Produces: `validate_sql(sql: str, role: str) -> ValidationResult`（语法/只读/表存在/权限/分区）

- [ ] **Step 1: 写失败测试 `tests/test_sql_validator.py`**

```python
"""SQL 规则校验测试。"""
from dataagent.guardrails.sql_validator import validate_sql


def test_valid_select_passes():
    result = validate_sql(
        "SELECT dt, SUM(gmv_amount) FROM dws_order_summary_di "
        "WHERE dt >= '2026-07-01' GROUP BY dt", "data_analyst")
    assert result.passed, result.errors


def test_syntax_error_rejected():
    result = validate_sql("SELEC a FROM b", "data_analyst")
    assert not result.passed
    assert any("语法" in e for e in result.errors)


def test_unknown_table_rejected():
    result = validate_sql("SELECT * FROM nonexistent_table", "data_analyst")
    assert not result.passed
    assert any("不存在" in e for e in result.errors)


def test_unauthorized_table_rejected():
    result = validate_sql("SELECT * FROM dws_payment_summary_di", "data_analyst")
    assert not result.passed
    assert any("无权限" in e for e in result.errors)


def test_non_select_rejected():
    for sql in ["DROP TABLE dim_category_info",
                "DELETE FROM dim_category_info",
                "UPDATE dim_category_info SET status='off'",
                "INSERT INTO dim_category_info VALUES (1,'x',0)"]:
        result = validate_sql(sql, "admin")
        assert not result.passed, sql
        assert any("只读" in e for e in result.errors)


def test_fact_table_requires_partition_filter():
    result = validate_sql(
        "SELECT COUNT(*) FROM dwd_order_detail_di", "data_analyst")
    assert not result.passed
    assert any("分区" in e for e in result.errors)


def test_partition_filter_passes():
    result = validate_sql(
        "SELECT COUNT(*) FROM dwd_order_detail_di WHERE dt >= '2026-07-01'",
        "data_analyst")
    assert result.passed, result.errors
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_sql_validator.py -v`
预期：FAIL

- [ ] **Step 3: 写 `dataagent/guardrails/sql_validator.py`**

```python
"""SQL 规则校验器：确定性护栏第一层（不依赖 LLM）。

生产对标：规则引擎 100% 拦截非法 SQL；LLM Critic 只做语义审查——
确定性检查与语义检查分层，是护栏体系的核心设计。
"""
from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp

from dataagent.permissions import filter_tables_by_role
from dataagent.warehouse.schema import TABLES


@dataclass
class ValidationResult:
    passed: bool
    errors: list[str] = field(default_factory=list)


def validate_sql(sql: str, role: str) -> ValidationResult:
    errors: list[str] = []

    # 1. 语法解析
    try:
        parsed = sqlglot.parse_one(sql)
    except Exception:
        return ValidationResult(False, [f"语法错误: 无法解析 SQL"])

    # 2. 只读检查
    if not isinstance(parsed, (exp.Select, exp.Union)):
        errors.append(f"只读限制: 仅允许 SELECT 查询")

    # 3. 表存在性 + 权限
    tables = {t.name for t in parsed.find_all(exp.Table)}
    visible = filter_tables_by_role(role, TABLES)
    for t in tables:
        if t not in TABLES:
            errors.append(f"表不存在: {t}")
        elif t not in visible:
            errors.append(f"无权限: {t}")

    # 4. 分区检查：DWD 明细大表必须带分区过滤
    for t in tables:
        if t not in visible:
            continue
        spec = TABLES[t]
        if spec.layer == "DWD":
            where = parsed.find(exp.Where)
            has_partition = (
                where is not None
                and spec.partition_col in
                "".join(c.sql() for c in where.find_all(exp.Column))
            )
            if not has_partition:
                errors.append(
                    f"分区检查: {t} 为 DWD 明细表，WHERE 必须包含分区字段 {spec.partition_col}")

    return ValidationResult(len(errors) == 0, errors)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `pytest tests/test_sql_validator.py -v`
预期：7 passed

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: SQL 规则校验器（语法/只读/权限/分区）"
```

### Task 13: LangGraph Agent 状态图

**Files:**
- Create: `dw-dataagent/dataagent/agent/__init__.py`
- Create: `dw-dataagent/dataagent/agent/state.py`
- Create: `dw-dataagent/dataagent/agent/prompts.py`
- Create: `dw-dataagent/dataagent/agent/graph.py`
- Create: `dw-dataagent/dataagent/cli.py`
- Test: `dw-dataagent/tests/test_graph.py`（FakeChatModel 构造测试）

**Interfaces:**
- Produces: `dataagent.agent.state.DWState`（TypedDict：question/role/parsed/context/sql/validation_errors/critic_feedback/result/generate_attempts）
- Produces: `dataagent.agent.graph.build_agent(settings) -> CompiledStateGraph`（返回编译后的图）
- Produces: `dataagent.agent.graph.run_agent(question, role) -> dict`（便利入口：同步执行并返回 {sql, result, explanation}）
- Consumes: `dataagent.llm.router.LLMRouter`、MCP 工具（SSE 客户端）、`HybridRetriever`、`validate_sql`、`QueryExecutor`（DuckDB/StarRocks 按 config）

- [ ] **Step 1: 写 `dataagent/agent/state.py`**

```python
"""LangGraph 状态定义。"""
from typing import TypedDict


class DWState(TypedDict, total=False):
    question: str          # 用户自然语言需求
    role: str              # 用户角色（权限过滤用）
    parsed: dict           # 需求解析结果 {domain, metrics, grain, time_range}
    context: str           # 已收集的上下文（元数据 + 案例）
    sql: str               # 生成的 SQL
    validation_errors: list  # 规则校验错误
    critic_feedback: str   # Critic 审查反馈（空 = 通过）
    result: str            # 执行结果（文本）
    generate_attempts: int  # SQL 生成重试次数
    execute_attempts: int   # 执行重试次数
```

- [ ] **Step 2: 写 `dataagent/agent/prompts.py`**

```python
"""各节点 Prompt 模板。"""

PARSE_PROMPT = """你是数仓取数需求解析器。从用户需求中提取结构化信息，只输出 JSON：

需求: {question}

输出 JSON 格式（字段: domain 主题域(订单域/用户域/商品域/支付域/物流域/未知)、
metrics 涉及的指标名列表、grain 时间粒度(日/周/月/未知)、
time_range 时间范围描述）:"""

GENERATE_PROMPT = """你是资深数仓工程师。基于以下信息生成取数 SQL。

## 用户需求
{question}

## 已收集上下文
{context}

## 硬性约束（违反任何一条都是错误）
1. 只能使用上下文中出现过的表名，禁止编造表
2. 只输出一条 SELECT 语句（Markdown 代码块内）
3. DWD 明细表必须在 WHERE 中使用分区字段 dt 过滤
4. 指标计算遵循上下文中给出的口径定义
5. 输出 SQL 后另起一行「口径说明:」简述计算口径

## 校验反馈（如非空，说明上次 SQL 有错，必须修正）
{feedback}

生成 SQL:"""

CRITIC_PROMPT = """你是数仓 SQL 审查员。检查以下 SQL 是否满足用户需求。

## 用户需求
{question}

## 候选 SQL
{sql}

## 已收集上下文
{context}

审查要点：指标口径是否与上下文一致、JOIN 是否合理、聚合逻辑是否正确、分区过滤是否合规。
只输出两类结果之一：
- "PASS"
- "FAIL: <具体问题与修改建议>"（指出问题，不要直接写 SQL）"""
```

- [ ] **Step 3: 写 `dataagent/agent/graph.py`**

```python
"""LangGraph Agent 状态图：parse → tool loop → generate ⇄ validate → execute。

生产对标：状态机护栏——parse/generate 等节点各有确定的工具权限，
校验失败回到 generate（最多 2 次），执行失败同样受限重试。
"""
import json
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from dataagent.agent.prompts import GENERATE_PROMPT, PARSE_PROMPT, CRITIC_PROMPT
from dataagent.agent.state import DWState
from dataagent.config import Settings, load_config
from dataagent.executor.base import QueryError
from dataagent.executor.duckdb_executor import DuckDBExecutor
from dataagent.executor.starrocks_executor import StarRocksExecutor
from dataagent.guardrails.sql_validator import validate_sql
from dataagent.llm.router import LLMRouter
from dataagent.rag.retriever import HybridRetriever
from dataagent.warehouse.schema import TABLES

MAX_GENERATE_ATTEMPTS = 2
MAX_EXECUTE_ATTEMPTS = 2

_mcp_tools_cache: list | None = None


async def _load_mcp_tools(settings: Settings):
    """通过 langchain-mcp-adapters 从 SSE MCP Server 加载工具（缓存）。"""
    global _mcp_tools_cache
    if _mcp_tools_cache is not None:
        return _mcp_tools_cache
    from langchain_mcp_adapters.client import MultiServerMCPClient
    client = MultiServerMCPClient({
        "metadata": {
            "transport": "sse",
            "url": settings.mcp_server_url,
        }
    })
    _mcp_tools_cache = await client.get_tools()
    return _mcp_tools_cache


def build_agent(settings: Settings):
    router = LLMRouter(settings)
    retriever = HybridRetriever(
        settings.milvus.host, settings.milvus.port, settings.milvus.collection)

    if settings.executor.default == "starrocks":
        sr = settings.executor.starrocks
        executor = StarRocksExecutor(sr.host, sr.port, sr.user, sr.password, TABLES)
    else:
        executor = DuckDBExecutor(settings.warehouse_path, TABLES)
    executor.setup()

    @tool
    def search_cases(question: str) -> str:
        """检索相似的历史取数案例。当需要参考类似需求的 SQL 写法时使用。"""
        hits = retriever.search(question, top_k=3)
        return json.dumps(hits, ensure_ascii=False, indent=2)

    def parse_node(state: DWState) -> DWState:
        model = router.get_model("task_parse")
        raw = model.invoke([
            SystemMessage("你只输出合法 JSON，不要输出其他内容。"),
            HumanMessage(PARSE_PROMPT.format(question=state["question"])),
        ]).content
        try:
            parsed = json.loads(re.sub(r"```json|```", "", raw).strip())
        except json.JSONDecodeError:
            parsed = {"domain": "未知", "metrics": [], "grain": "未知",
                      "time_range": ""}
        return {"parsed": parsed}

    async def collect_context_node(state: DWState) -> DWState:
        """工具调用循环：Agent 自主决定查哪些元数据/案例。"""
        mcp_tools = await _load_mcp_tools(settings)
        tools = mcp_tools + [search_cases]
        from langgraph.prebuilt import create_react_agent
        mini_agent = create_react_agent(router.get_model("sql_generate"), tools)
        tool_names = "\n".join(f"- {t.name}: {t.description}" for t in tools)
        task = (
            f"用户需求: {state['question']}\n"
            f"解析结果: {state['parsed']}\n\n"
            f"可用工具:\n{tool_names}\n\n"
            f"请查询生成该 SQL 所需的全部信息：涉及的候选表结构、指标口径、"
            f"相似历史案例。用工具返回结果整理成上下文摘要（含表名、关键字段、"
            f"口径定义、参考 SQL）。")
        result = mini_agent.invoke({"messages": [HumanMessage(task)]})
        context = result["messages"][-1].content
        return {"context": context}

    def generate_node(state: DWState) -> DWState:
        model = router.get_model("sql_generate")
        feedback = "\n".join(state.get("validation_errors", []) or [])
        if state.get("critic_feedback"):
            feedback += f"\n审查反馈: {state['critic_feedback']}"
        raw = model.invoke([
            HumanMessage(GENERATE_PROMPT.format(
                question=state["question"],
                context=state.get("context", ""),
                feedback=feedback or "（无）",
            )),
        ]).content
        match = re.search(r"```(?:sql)?\s*(SELECT[\s\S]*?)```", raw, re.IGNORECASE)
        sql = match.group(1).strip() if match else raw.strip()
        return {"sql": sql,
                "generate_attempts": state.get("generate_attempts", 0) + 1}

    def validate_node(state: DWState) -> DWState:
        result = validate_sql(state["sql"], state["role"])
        return {"validation_errors": result.errors}

    def critic_node(state: DWState) -> DWState:
        model = router.get_model("critic_review")
        raw = model.invoke([
            HumanMessage(CRITIC_PROMPT.format(
                question=state["question"], sql=state["sql"],
                context=state.get("context", ""))),
        ]).content
        feedback = "" if raw.strip().startswith("PASS") else raw.strip()
        return {"critic_feedback": feedback}

    def execute_node(state: DWState) -> DWState:
        try:
            rows = executor.execute(state["sql"])
        except QueryError as e:
            return {"result": f"执行失败: {e.message}",
                    "execute_attempts": state.get("execute_attempts", 0) + 1}
        preview = "\n".join(str(r) for r in rows[:20])
        return {"result": f"执行成功，共 {len(rows)} 行。前 20 行:\n{preview}",
                "execute_attempts": state.get("execute_attempts", 0)}

    def route_after_validate(state: DWState) -> str:
        if state.get("validation_errors"):
            if state.get("generate_attempts", 0) < MAX_GENERATE_ATTEMPTS:
                return "generate"
            return "fail"
        return "critic"

    def route_after_critic(state: DWState) -> str:
        if state.get("critic_feedback"):
            if state.get("generate_attempts", 0) < MAX_GENERATE_ATTEMPTS:
                return "generate"
            return "fail"
        return "execute"

    def route_after_execute(state: DWState) -> str:
        if state["result"].startswith("执行失败"):
            if state.get("execute_attempts", 0) < MAX_EXECUTE_ATTEMPTS:
                return "generate"
            return "fail"
        return "done"

    workflow = StateGraph(DWState)
    workflow.add_node("parse", parse_node)
    workflow.add_node("collect", collect_context_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("validate", validate_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("execute", execute_node)
    workflow.add_node("fail", lambda s: {"result": "流程失败: 重试次数用尽"})

    workflow.set_entry_point("parse")
    workflow.add_edge("parse", "collect")
    workflow.add_edge("collect", "generate")
    workflow.add_edge("generate", "validate")
    workflow.add_conditional_edges(
        "validate", route_after_validate,
        {"generate": "generate", "critic": "critic", "fail": "fail"})
    workflow.add_conditional_edges(
        "critic", route_after_critic,
        {"generate": "generate", "execute": "execute", "fail": "fail"})
    workflow.add_conditional_edges(
        "execute", route_after_execute,
        {"generate": "generate", "done": END, "fail": "fail"})
    workflow.add_edge("fail", END)

    return workflow.compile(checkpointer=MemorySaver())


def run_agent(question: str, role: str = "data_analyst",
              settings: Settings | None = None) -> dict:
    """同步便利入口。"""
    import asyncio
    settings = settings or load_config()
    agent = build_agent(settings)
    state = asyncio.run(_invoke(agent, question, role))
    explanation = (f"SQL: {state.get('sql', '')}\n结果: {state.get('result', '')}")
    return {"sql": state.get("sql", ""), "result": state.get("result", ""),
            "explanation": explanation}


async def _invoke(agent, question: str, role: str) -> dict:
    return await agent.ainvoke(
        {"question": question, "role": role,
         "generate_attempts": 0, "execute_attempts": 0},
        config={"configurable": {"thread_id": "cli"}})
```

- [ ] **Step 4: 写 `dataagent/cli.py`**

```python
"""CLI 入口：python -m dataagent.cli "需求" [--role data_analyst]"""
import argparse

from dataagent.agent.graph import run_agent


def main():
    parser = argparse.ArgumentParser(description="数仓取数 DataAgent")
    parser.add_argument("question", help="自然语言取数需求")
    parser.add_argument("--role", default="data_analyst",
                        choices=["data_analyst", "finance_analyst",
                                 "ops_analyst", "admin"])
    args = parser.parse_args()

    result = run_agent(args.question, args.role)
    print("=" * 60)
    print(result["explanation"])
    print("=" * 60)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: 写图结构测试 `tests/test_graph.py`**

```python
"""LangGraph 图结构测试（不调真实 LLM）。"""
from dataagent.agent.graph import build_agent
from dataagent.config import load_config


def test_graph_has_all_nodes():
    agent = build_agent(load_config())
    nodes = set(agent.get_graph().nodes.keys())
    assert {"parse", "collect", "generate", "validate", "critic",
            "execute", "fail"} <= nodes
```

- [ ] **Step 6: 跑测试 + 端到端验收**

```bash
source .venv/bin/activate
pytest tests/test_graph.py -v  # 1 passed（不调 LLM）

# 端到端（需 MCP Server 已在 8001 运行、Milvus 已就绪、DEEPSEEK_API_KEY 已设置）
python -m dataagent.cli "统计最近30天各平台GMV按日趋势" --role data_analyst
```

预期：输出 SQL + 执行结果；若首条需求失败，检查：①MCP Server 是否在运行 ②Milvus 是否 load ③`DEEPSEEK_API_KEY`。

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "feat: LangGraph Agent 状态图（parse/collect/generate/validate/critic/execute）"
```

## Day 6-7 · 护栏补全与评测

### Task 14: FastAPI 服务化 + Langfuse 可观测

**Files:**
- Create: `dw-dataagent/dataagent/api.py`
- Create: `dw-dataagent/dataagent/observability.py`
- Test: 手动验收（curl + Langfuse 面板）

**Interfaces:**
- Produces: `dataagent.api.app`（FastAPI：POST /query、POST /query/stream、GET /health）
- Produces: `dataagent.observability.get_langfuse_handler() -> CallbackHandler | None`（未配置 key 时返回 None 不报错）

- [ ] **Step 1: 写 `dataagent/observability.py`**

```python
"""Langfuse 可观测性集成：Trace / Token 成本 / 反馈闭环。

未配置 LANGFUSE_PUBLIC_KEY 时优雅降级（返回 None），开发环境零依赖。
"""
import os
from typing import Optional

from langfuse.callback import CallbackHandler

_instance: Optional[CallbackHandler] = None
_checked = False


def get_langfuse_handler() -> Optional[CallbackHandler]:
    global _instance, _checked
    if _checked:
        return _instance
    _checked = True
    if os.environ.get("LANGFUSE_PUBLIC_KEY"):
        _instance = CallbackHandler()
    return _instance
```

- [ ] **Step 2: 写 `dataagent/api.py`**

```python
"""Agent Service：FastAPI 服务化。

生产对标：业务系统（BI/IM 机器人）通过 REST/SSE 调用 Agent，
Agent 作为独立服务部署、独立扩缩容。
"""
import asyncio
import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from dataagent.agent.graph import build_agent, _invoke
from dataagent.config import load_config
from dataagent.observability import get_langfuse_handler

app = FastAPI(title="dw-dataagent", version="1.0.0")
settings = load_config()
_agent = build_agent(settings)


class QueryRequest(BaseModel):
    question: str
    role: str = "data_analyst"


class QueryResponse(BaseModel):
    sql: str
    result: str
    explanation: str


@app.get("/health")
async def health():
    return {"status": "ok", "executor": settings.executor.default}


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    langfuse = get_langfuse_handler()
    config = {"callbacks": [langfuse]} if langfuse else None
    try:
        state = await _agent.ainvoke(
            {"question": req.question, "role": req.role,
             "generate_attempts": 0, "execute_attempts": 0},
            config=config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent 执行失败: {e}")
    return QueryResponse(
        sql=state.get("sql", ""),
        result=state.get("result", ""),
        explanation=f"SQL: {state.get('sql', '')}\n结果: {state.get('result', '')}")


@app.post("/query/stream")
async def query_stream(req: QueryRequest):
    """SSE 流式返回（事件: done 携带最终 JSON）。"""
    async def event_stream():
        state = await _agent.ainvoke(
            {"question": req.question, "role": req.role,
             "generate_attempts": 0, "execute_attempts": 0})
        payload = json.dumps({
            "sql": state.get("sql", ""),
            "result": state.get("result", ""),
        }, ensure_ascii=False)
        yield f"data: {payload}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

- [ ] **Step 3: 启动服务验收**

```bash
source .venv/bin/activate
uvicorn dataagent.api:app --port 8000 &

curl http://localhost:8000/health
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "7月各品类GMV Top5", "role": "data_analyst"}'
```

预期：health 返回 executor=duckdb；query 返回 {sql, result, explanation} 且 result 以「执行成功」开头

- [ ] **Step 4: Langfuse 验证（可选，配置 key 后）**

```bash
export LANGFUSE_PUBLIC_KEY=pk-... LANGFUSE_SECRET_KEY=sk-...
uvicorn dataagent.api:app --port 8000 &
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" \
  -d '{"question": "近7天DAU趋势", "role": "data_analyst"}'
# 打开 Langfuse 面板确认 Trace 出现（parse/collect/generate/validate/critic/execute 各节点）
```

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: FastAPI 服务化 + Langfuse 可观测"
```

### Task 15: Golden Set 与评测

**Files:**
- Create: `dw-dataagent/evals/golden_set.yaml`
- Create: `dw-dataagent/evals/eval_runner.py`
- Create: `dw-dataagent/scripts/run_evals.py`
- Test: `dw-dataagent/tests/test_eval_runner.py`（判定函数单测）

**Interfaces:**
- Produces: `dataagent.eval_runner.judge_sql(sql, expected) -> dict`（passed, checks）
- Consumes: `validate_sql`（执行前置校验）、`DuckDBExecutor`

- [ ] **Step 1: 写 `evals/golden_set.yaml`（30 条，含预期关键要素）**

```yaml
# Golden Set：30 条取数任务（简单聚合 40% / 多表 JOIN 30% / 口径 20% / 复杂嵌套 10%）
# 判定标准：执行成功 + 预期表全部出现 + 预期 SQL 关键字出现
- id: g001
  question: 7月各品类GMV汇总
  role: data_analyst
  difficulty: simple
  expected:
    tables: [dws_category_order_di]
    keywords: [SUM, gmv_amount, GROUP BY]
- id: g002
  question: 近30天每日下单数
  role: data_analyst
  difficulty: simple
  expected:
    tables: [dws_order_summary_di]
    keywords: [order_cnt, dt]
- id: g003
  question: 各平台支付订单数对比
  role: data_analyst
  difficulty: simple
  expected:
    tables: [dws_platform_order_di]
    keywords: [platform, order_cnt]
- id: g004
  question: 昨日商品浏览量Top10
  role: data_analyst
  difficulty: simple
  expected:
    tables: [dws_product_view_di]
    keywords: [view_cnt, ORDER BY, LIMIT]
- id: g005
  question: 7月新增用户总数
  role: data_analyst
  difficulty: simple
  expected:
    tables: [dws_user_active_di]
    keywords: [new_user_cnt, SUM]
- id: g006
  question: 各物流公司发货量统计
  role: ops_analyst
  difficulty: simple
  expected:
    tables: [dws_logistics_company_di]
    keywords: [logistics_company, ship_cnt]
- id: g007
  question: 近30天DAU趋势
  role: data_analyst
  difficulty: simple
  expected:
    tables: [dws_user_active_di]
    keywords: [dau, dt]
- id: g008
  question: 各注册渠道新增用户数
  role: data_analyst
  difficulty: simple
  expected:
    tables: [dwd_user_register_di]
    keywords: [register_channel, COUNT]
- id: g009
  question: 7月支付总额
  role: finance_analyst
  difficulty: simple
  expected:
    tables: [dws_payment_summary_di]
    keywords: [pay_amount, SUM]
- id: g010
  question: 各品类订单量排名
  role: data_analyst
  difficulty: simple
  expected:
    tables: [dws_category_order_di]
    keywords: [category_id, order_cnt, ORDER BY]
- id: g011
  question: 7月退款金额最高的10天
  role: finance_analyst
  difficulty: simple
  expected:
    tables: [dws_refund_summary_di]
    keywords: [refund_amount, ORDER BY, LIMIT]
- id: g012
  question: 物流准时率日报
  role: ops_analyst
  difficulty: simple
  expected:
    tables: [ads_logistics_daily_report_di]
    keywords: [on_time_rate, dt]
- id: g013
  question: 品牌GMV排行（关联商品维度）
  role: data_analyst
  difficulty: join
  expected:
    tables: [dws_product_gmv_di, dim_product_info]
    keywords: [brand, JOIN]
- id: g014
  question: 品类浏览热度排行（关联品类维度）
  role: data_analyst
  difficulty: join
  expected:
    tables: [dws_product_view_di, dim_product_info, dim_category_info]
    keywords: [category_name, JOIN]
- id: g015
  question: 退款率趋势（支付与退款汇总关联）
  role: finance_analyst
  difficulty: join
  expected:
    tables: [dws_payment_summary_di, dws_refund_summary_di]
    keywords: [refund_amount, pay_amount, JOIN]
- id: g016
  question: 商品浏览转购买率
  role: data_analyst
  difficulty: join
  expected:
    tables: [dws_product_gmv_di, dws_product_view_di]
    keywords: [pay_user_cnt, view_user_cnt, JOIN]
- id: g017
  question: 各品类GMV Top5（关联品类名称）
  role: data_analyst
  difficulty: join
  expected:
    tables: [dws_category_order_di, dim_category_info]
    keywords: [category_name, gmv_amount, JOIN]
- id: g018
  question: 商品GMV最高的商品及其品牌
  role: data_analyst
  difficulty: join
  expected:
    tables: [dws_product_gmv_di, dim_product_info]
    keywords: [gmv_amount, brand, ORDER BY, LIMIT]
- id: g019
  question: 各城市注册用户Top10
  role: data_analyst
  difficulty: join
  expected:
    tables: [dwd_user_register_di]
    keywords: [city, COUNT, GROUP BY]
- id: g020
  question: 各仓库发货量（关联仓库维度）
  role: ops_analyst
  difficulty: join
  expected:
    tables: [dwd_logistics_shipped_di, dim_warehouse_info]
    keywords: [warehouse_name, JOIN]
- id: g021
  question: 近30天GMV日均值
  role: data_analyst
  difficulty: metric
  expected:
    tables: [dws_order_summary_di]
    keywords: [gmv_amount, AVG]
- id: g022
  question: 7月支付率
  role: data_analyst
  difficulty: metric
  expected:
    tables: [dws_order_summary_di]
    keywords: [pay_order_cnt, order_cnt]
- id: g023
  question: 客单价趋势
  role: data_analyst
  difficulty: metric
  expected:
    tables: [ads_order_daily_report_di]
    keywords: [avg_order_amount, dt]
- id: g024
  question: 退款率（退款订单/支付订单）
  role: finance_analyst
  difficulty: metric
  expected:
    tables: [dws_payment_summary_di, dws_refund_summary_di]
    keywords: [refund_order_cnt, pay_order_cnt]
- id: g025
  question: 次日留存率趋势
  role: data_analyst
  difficulty: metric
  expected:
    tables: [dws_user_retention_di]
    keywords: [retain_d1, dt]
- id: g026
  question: 各品类GMV占比（窗口函数）
  role: data_analyst
  difficulty: metric
  expected:
    tables: [dws_category_order_di]
    keywords: [OVER, gmv_amount]
- id: g027
  question: 支付成功率（成功笔数/总笔数）
  role: finance_analyst
  difficulty: complex
  expected:
    tables: [dwd_payment_detail_di]
    keywords: [pay_status, CASE, COUNT]
- id: g028
  question: 各平台订单转化率（支付订单/下单）
  role: data_analyst
  difficulty: complex
  expected:
    tables: [dwd_order_detail_di]
    keywords: [pay_status, CASE, platform]
- id: g029
  question: 价格带商品销量分布（关联商品维度 + 条件分桶）
  role: data_analyst
  difficulty: complex
  expected:
    tables: [dws_product_gmv_di, dim_product_info]
    keywords: [price, CASE WHEN, JOIN]
- id: g030
  question: 每周GMV趋势（时间聚合 + 平台维度）
  role: data_analyst
  difficulty: complex
  expected:
    tables: [dws_platform_order_di]
    keywords: [platform, gmv_amount, GROUP BY]
```

- [ ] **Step 2: 写判定函数单测 `tests/test_eval_runner.py`**

```python
"""评测判定函数测试。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.eval_runner import judge_sql, check_execution


def test_judge_sql_passes_when_all_present():
    sql = "SELECT category_id, SUM(gmv_amount) AS gmv FROM dws_category_order_di WHERE dt >= '2026-07-01' GROUP BY category_id"
    expected = {"tables": ["dws_category_order_di"], "keywords": ["SUM", "gmv_amount", "GROUP BY"]}
    result = judge_sql(sql, expected)
    assert result["passed"], result


def test_judge_sql_fails_on_missing_table():
    sql = "SELECT * FROM dws_order_summary_di"
    expected = {"tables": ["dws_category_order_di"], "keywords": []}
    result = judge_sql(sql, expected)
    assert not result["passed"]
    assert any("表" in c["desc"] for c in result["checks"] if not c["ok"])


def test_judge_sql_fails_on_missing_keyword():
    sql = "SELECT * FROM dws_category_order_di"
    expected = {"tables": ["dws_category_order_di"], "keywords": ["SUM"]}
    result = judge_sql(sql, expected)
    assert not result["passed"]
```

- [ ] **Step 3: 写 `evals/eval_runner.py`**

```python
"""Golden Set 评测执行器。

指标：
- 执行成功率 = 通过校验且 DuckDB 执行成功的比例
- 要素准确率 = SQL 包含全部预期表与关键字且执行成功的比例（主指标）
- 失败原因分类：校验失败 / 执行失败 / 要素缺失
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from dataagent.agent.graph import run_agent
from dataagent.executor.duckdb_executor import DuckDBExecutor
from dataagent.guardrails.sql_validator import validate_sql
from dataagent.warehouse.schema import TABLES


def load_golden_set(path: str = "evals/golden_set.yaml") -> list[dict]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def judge_sql(sql: str, expected: dict) -> dict:
    checks = []
    for t in expected.get("tables", []):
        checks.append({"check": f"表 {t}", "ok": t in sql,
                       "desc": f"SQL 使用了表 {t}"})
    for kw in expected.get("keywords", []):
        checks.append({"check": f"关键字 {kw}", "ok": kw.lower() in sql.lower(),
                       "desc": f"SQL 包含 {kw}"})
    passed = all(c["ok"] for c in checks)
    return {"passed": passed, "checks": checks}


def check_execution(sql: str, role: str) -> tuple[bool, str]:
    """校验 + DuckDB 真实执行。"""
    validation = validate_sql(sql, role)
    if not validation.passed:
        return False, "校验失败: " + "; ".join(validation.errors)
    executor = DuckDBExecutor("data/warehouse.duckdb", TABLES)
    executor.setup()
    try:
        executor.execute(sql)
        return True, ""
    except Exception as e:
        return False, f"执行失败: {e}"
    finally:
        executor.close()


def run_all(golden_set_path: str = "evals/golden_set.yaml",
            report_path: str = "evals/report.yaml") -> dict:
    cases = load_golden_set(golden_set_path)
    results = []
    for case in cases:
        t0 = time.time()
        try:
            agent_result = run_agent(case["question"], case["role"])
            sql = agent_result["sql"]
            exec_ok, exec_msg = check_execution(sql, case["role"])
            judge = judge_sql(sql, case["expected"])
            passed = exec_ok and judge["passed"]
            failure = "" if passed else (
                exec_msg or "要素缺失: " + "; ".join(
                    c["check"] for c in judge["checks"] if not c["ok"]))
        except Exception as e:
            sql, exec_ok, judge, passed, failure = "", False, {"passed": False, "checks": []}, False, f"Agent 异常: {e}"
        results.append({
            "id": case["id"], "question": case["question"],
            "difficulty": case["difficulty"], "passed": passed,
            "sql": sql, "failure": failure,
            "elapsed_s": round(time.time() - t0, 1),
        })
        print(f"[{case['id']}] {'PASS' if passed else 'FAIL'} "
              f"({case['difficulty']}) {failure[:80]}")

    total = len(results)
    passed_n = sum(1 for r in results if r["passed"])
    accuracy = passed_n / total if total else 0
    exec_ok_n = sum(1 for r in results if r["passed"] or "要素缺失" in r["failure"])

    by_difficulty = {}
    for diff in ("simple", "join", "metric", "complex"):
        sub = [r for r in results if r["difficulty"] == diff]
        by_difficulty[diff] = (sum(1 for r in sub if r["passed"]) / len(sub)) if sub else None

    failures = [r for r in results if not r["passed"]]
    report = {
        "total": total, "passed": passed_n, "accuracy": round(accuracy, 4),
        "exec_success_rate": round(exec_ok_n / total, 4) if total else 0,
        "by_difficulty": by_difficulty,
        "failures": [
            {"id": r["id"], "question": r["question"], "failure": r["failure"],
             "sql": r["sql"]}
            for r in failures
        ],
    }
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(
        yaml.safe_dump(report, allow_unicode=True, sort_keys=False),
        encoding="utf-8")
    print(f"\n准确率: {accuracy:.1%} ({passed_n}/{total})")
    print(f"报告: {report_path}")
    return report


if __name__ == "__main__":
    run_all()
```

- [ ] **Step 4: 写 `scripts/run_evals.py`**

```python
"""Golden Set 评测入口：python scripts/run_evals.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.eval_runner import run_all

if __name__ == "__main__":
    run_all()
```

- [ ] **Step 5: 跑单测 + 首次全量评测**

```bash
source .venv/bin/activate
pytest tests/test_eval_runner.py -v    # 3 passed

# 全量评测（需 MCP + Milvus + DEEPSEEK_API_KEY；30 条约 10-20 分钟）
python scripts/run_evals.py
```

预期：产出 `evals/report.yaml`；首轮准确率可能 60-80%，记录失败原因

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: Golden Set(30条) + 评测执行器（准确率/失败分类/难度分层）"
```

## Day 8 · 迭代收尾

### Task 16: 准确率迭代（基于失败原因）

**Files:**
- Modify: `dw-dataagent/dataagent/agent/prompts.py`（按失败模式优化）
- Modify: `dw-dataagent/dataagent/rag/retriever.py`（按需）
- Test: 重跑全量评测对比

- [ ] **Step 1: 分析失败原因，按类别处理**

读 `evals/report.yaml` 的 failures，按以下映射处理（每轮只改一类）：

| 失败类别 | 处理动作 |
|---------|---------|
| 校验失败-表不存在 | 在 GENERATE_PROMPT 约束 1 处追加「生成前确认上下文已包含该表结构；上下文没有的表一律不用」 |
| 校验失败-无权限 | 无需改（说明 Agent 未遵循权限约束）→ 在 GENERATE_PROMPT 追加「用户角色只能查询其可见表，上下文外的表视为无权限」 |
| 校验失败-分区缺失 | 在 GENERATE_PROMPT 约束 3 追加「所有 DWD 表 WHERE 必须含 dt 分区过滤（如 dt >= '2026-07-01'）」 |
| 执行失败 | 将报错注入 generate 的 feedback（已实现），观察第二轮回合；仍失败则检查 SQLGlot 解析的 SQL 提取是否完整 |
| 要素缺失-JOIN | 在 GENERATE_PROMPT 追加「涉及多表时必须显式写 JOIN ... ON 条件」 |
| 要素缺失-指标 | 确认 collect 阶段是否查到口径：在 collect task 文案追加「指标口径必须从 query_metric_definition 工具获取，不得臆测」 |

- [ ] **Step 2: 每轮改完 Prompt 后重跑评测并对比**

```bash
python scripts/run_evals.py
# 对比 evals/report.yaml 的 accuracy 变化；单类修复后准确率应单调上升
```

- [ ] **Step 3: 达到 ≥80% 后 Commit**

```bash
git add -A && git commit -m "fix: Prompt 迭代（表约束/分区约束/口径约束）准确率达标"
```

### Task 17: 部署收尾 + 简历条目

**Files:**
- Create: `dw-dataagent/deploy/infra-compose.yml`
- Create: `dw-dataagent/Dockerfile`
- Create: `dw-dataagent/docs/resume-entry.md`（简历条目 + 面试话术）
- Modify: `dw-dataagent/README.md`（架构图 + 评测数据）

- [ ] **Step 1: 写 `deploy/infra-compose.yml`**

```yaml
# 基础设施编排：Milvus 向量库（etcd + minio + milvus standalone）
# StarRocks 因内存占用大，按需单独启动（见 design.md 风险表）
services:
  etcd:
    container_name: milvus-etcd
    image: quay.io/coreos/etcd:v3.5.16
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
    command: etcd -advertise-client-urls=http://etcd:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd
    volumes:
      - milvus-etcd:/etcd

  minio:
    container_name: milvus-minio
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    command: minio server /minio_data
    volumes:
      - milvus-minio:/minio_data

  milvus:
    container_name: milvus-standalone
    image: milvusdb/milvus:v2.5.4
    command: ["milvus", "run", "standalone"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    depends_on:
      - etcd
      - minio
    ports:
      - "19530:19530"
      - "9091:9091"

volumes:
  milvus-etcd:
  milvus-minio:
```

- [ ] **Step 2: 写 `Dockerfile`**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY dataagent/ dataagent/
COPY data/ data/
COPY config.yaml .

EXPOSE 8000

CMD ["uvicorn", "dataagent.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 3: 写 `docs/resume-entry.md`**

```markdown
# 简历条目 + 面试话术

## 简历 · 个人项目板块

**DataAgent 取数智能体（LangGraph + MCP + Milvus + StarRocks）** · 个人项目
- 从零实现生产级取数 Agent：LangGraph 状态图编排（解析→元数据查询→案例检索→生成→校验→执行），Checkpoint 持久化支持断点恢复
- 自研 MCP Server（Python SDK，SSE 服务化部署），4 个元数据工具 + RBAC 表级权限过滤（无权限表在检索层不可见，源头阻断越权）
- RAG 案例库：Milvus HNSW + BGE-large-zh Embedding + BM25 关键词检索 + RRF 融合，50 条历史 SQL 案例检索
- 双层护栏：SQLGlot 规则引擎（语法/只读/表存在/权限/分区 100% 拦截非法 SQL）+ LLM Critic 语义审查
- 模型路由层（Tiered Model Stack）：DeepSeek API + 本地 Ollama Qwen3，生产私有化部署零改动切换
- Golden Set 30 条评测驱动迭代，取数准确率 XX%（执行成功率 XX%）〔数字在 Task 17 Step 4 从 evals/report.yaml 实测回填〕
- 可观测：Langfuse 全链路 Trace + Token 成本统计
技术栈：Python / LangGraph / MCP / Milvus / BGE / StarRocks / DuckDB / SQLGlot / FastAPI / Langfuse

## 面试话术（对应各组件）

### Agent 框架
> "我选 LangGraph 而不是裸 LangChain Agent，因为它有显式状态管理——State 定义每一步的输入输出，Checkpoint 让长任务可以暂停恢复。这在生产环境意味着：Agent 跑挂了能从断点继续，而不是从头再来。我的图有 7 个节点：parse/collect/generate/validate/critic/execute/fail，校验失败回到 generate 重试（上限 2 次），这是状态机护栏的设计。"

### MCP
> "MCP Server 我做了 SSE 服务化部署而不是 stdio，因为生产环境 Agent 是独立服务，MCP 需要支持多客户端远程访问。工具层我实现了权限过滤——不是 SQL 生成后再拦，而是元数据检索阶段无权限的表对 Agent 完全不可见，从源头消除越权。"

### RAG
> "案例检索我用了混合检索：Milvus 向量检索找语义相似的案例，BM25 找表名、指标名这类专有名词的精确匹配，RRF 融合两者。纯向量检索在数仓场景会漏掉专有名词，纯关键词又理解不了语义，融合后 Top-5 命中率明显提升。"

### 护栏
> "校验分两层：SQLGlot 规则引擎做确定性检查（语法、只读、表存在、权限、分区），100% 拦截非法 SQL；LLM Critic 做语义审查（口径一致性、JOIN 合理性）。规则能兜底的绝不用 LLM——LLM 审查有幻觉风险，只能做补充。"

### 评测
> "30 条 Golden Set 按难度分层：简单聚合 40%、多表 JOIN 30%、口径 20%、复杂嵌套 10%。评测主指标是要素准确率（执行成功 + 预期表 + 关键字），副指标是执行成功率。每次 Prompt 迭代都跑全量回归，失败原因分类（校验失败/执行失败/要素缺失）驱动迭代方向。"

### 模型路由
> "模型路由层解决两个问题：成本分层（需求解析用轻量模型，SQL 生成用主力模型）和合规隔离（生产环境数据不出域，必须私有化部署 Qwen/DeepSeek，路由层切换零代码改动）。这对应大厂模型网关的设计。"
```

- [ ] **Step 4: 更新 README（补充评测数据与架构图；把「准确率 ≥ 80%」替换为实测数字）**

- [ ] **Step 5: 最终全量回归 + Commit**

```bash
pytest tests/ -v                     # 全部单测
python scripts/run_evals.py          # 最终评测数据
git add -A && git commit -m "docs: 部署收尾、简历条目与面试话术"
```

## 验证清单（全项目完成标准）

- [ ] `pytest tests/ -v` 全绿（integration 标记测试需服务运行）
- [ ] `python -m dataagent.cli "统计最近30天各平台GMV按日趋势"` 端到端输出 SQL + 执行结果
- [ ] `curl http://localhost:8000/query` 返回正确 JSON
- [ ] `python scripts/run_evals.py` 准确率 ≥ 80%，report.yaml 有据可查
- [ ] Langfuse 面板可见全链路 Trace
- [ ] 越权场景验证：`data_analyst` 角色查询支付域表被拒绝
- [ ] StarRocks 模式验证完成或已记录兜底说明（design.md 风险表）
