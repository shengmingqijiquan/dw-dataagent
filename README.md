# dw-dataagent · 数仓取数 DataAgent（生产级对标）

**[English README](README_EN.md) · [设计文档](docs/design.md)**

> 面向数仓取数场景的生产级 DataAgent 服务：自然语言需求 → 多步 Agent 推理 → 生成并执行 SQL → 返回可解释结果，全程可观测、可评测、可审计。
>
> **技术栈**：Python / LangGraph / MCP(SSE) / Milvus / BGE / StarRocks / SQLGlot / Langfuse / FastAPI
> **Python 版本**：≥ 3.10（推荐 3.11+）

## 项目定位

这是一个面向数仓取数场景的生产级 DataAgent 示例项目，**技术选型与部署形态对齐主流互联网企业的 AI 数据应用生产实践**：

> 本项目代码由 AI 辅助生成，人工负责架构设计、代码审查与测试。

| 组件 | 技术 | 设计要点 |
|------|------|---------|
| Agent 循环 | LangGraph（ReAct + State + Checkpoint） | Agent 框架选型、状态管理、循环控制 |
| Agent 服务 | FastAPI + SSE 流式接口 + Docker | 服务化部署、异步任务模式 |
| 模型路由 | DeepSeek API + Ollama Qwen3 | Tiered Model Stack、数据合规、成本控制 |
| MCP Server | Python MCP SDK + SSE 服务化 | MCP 协议、工具标准化、服务化部署 |
| 数据安全 | 角色 → 表级权限过滤（元数据层） | RBAC、数据合规、越权防护 |
| RAG 案例库 | Milvus + BGE-large-zh + 混合检索 | 向量库选型、Embedding、检索策略 |
| SQL 校验 | SQLGlot + 规则引擎 + Critic | 护栏设计、双层校验 |
| 执行引擎 | StarRocks（QueryExecutor 抽象） | OLAP 选型、引擎可替换设计 |
| Evals | Golden Set + 自动化评测 | 评测体系、迭代方法论 |
| 可观测 | Langfuse（Trace/成本/反馈） | 全链路追踪、成本分析 |

## 快速开始

```bash
# 0. 前置依赖
#    - Python ≥ 3.10（推荐 3.11+）
#    - Docker + Docker Compose（infra 服务）
#    - Docker Hub / 国内镜像网络畅通

# 1. 克隆并安装依赖
git clone <repo-url> && cd dw-dataagent
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. 复制环境变量模板并配置
cp .env.example .env
# 填写 .env 中的 DEEPSEEK_API_KEY（其他按需配置）

# 3. 启动基础设施（Milvus 向量库：etcd+minio+milvus standalone）
docker compose -f deploy/infra-compose.yml up -d

# 4. LLM 配置（模型路由）
#    A. DeepSeek API（开发主力）—— 在 .env 中设置 DEEPSEEK_API_KEY
#    B. 本地 Ollama（数据不出域兜底）
ollama pull qwen3:8b

# 5. 初始化模拟数仓（建表 + 数据 + 元数据 + 权限）
python scripts/init_warehouse.py                 # DuckDB 引擎（默认，开发兜底）
# python scripts/init_warehouse.py --engine starrocks  # StarRocks 引擎（按需，需容器已启动）
#   —— 注：需先启动 StarRocks 容器；无容器环境时使用 DuckDB 兜底（接口相同）

# 6. 构建 RAG 案例库（BGE Embedding + Milvus 入库）
python scripts/build_rag_index.py
#   —— 本机开发环境经 config.yaml services.milvus.uri 常驻使用 Milvus Lite
#      （data/milvus.db）；生产可切换 Milvus standalone（deploy/infra-compose.yml）；
#      BGE 模型约 1.3GB，国内网络下可用镜像
#      `HF_ENDPOINT=https://hf-mirror.com python scripts/build_rag_index.py`

# 7. 启动 MCP Server（SSE 服务）
python -m dataagent.mcp_server.server --port 8001

# 8. 启动 Agent Service（FastAPI）
python -m dataagent.api --port 8000

# 9. 调用取数接口
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "统计最近30天各品类GMV，按日趋势输出", "role": "data_analyst"}'

# 10. 跑 Golden Set 评测
python scripts/run_evals.py
```

## 目录结构

```
dw-dataagent/
├── README.md                    # 本文档（中文）
├── README_EN.md                 # 本文档（英文）
├── .env.example                 # 环境变量模板
├── requirements.txt
├── config.yaml                  # 模型路由/服务地址/权限配置
├── Dockerfile                   # Agent Service 镜像
├── docs/
│   └── design.md                # 设计文档（架构/组件/开发计划）
├── deploy/
│   └── infra-compose.yml        # Milvus(etcd+minio+milvus standalone)，StarRocks 按需单独启动
├── scripts/
│   ├── init_warehouse.py        # 模拟数仓初始化（建表/数据/元数据/权限）
│   ├── build_rag_index.py       # RAG 案例库构建
│   └── run_evals.py             # Golden Set 评测
├── data/                        # 数据文件（gitignore）
│   ├── metadata/                # 元数据（tables/columns/lineage/metrics/roles）
│   └── cases/                   # 历史 SQL 案例（需求+SQL 配对）
├── dataagent/
│   ├── __init__.py
│   ├── api.py                   # FastAPI Agent 服务（REST + SSE 流式）
│   ├── config.py                # 配置加载（模型路由表/服务地址）
│   ├── agent/
│   │   ├── graph.py             # LangGraph 状态图（节点/边/路由）
│   │   ├── state.py             # State 定义
│   │   └── prompts.py           # Prompt 模板
│   ├── llm/
│   │   └── router.py            # 模型路由层（Tiered Model Stack）
│   ├── mcp_server/
│   │   ├── server.py            # MCP Server（SSE 服务化 + FastAPI 挂载）
│   │   └── metadata.py          # 元数据查询 + 权限过滤
│   ├── rag/
│   │   ├── indexer.py           # 分块 + BGE Embedding + Milvus 入库
│   │   ├── retriever.py         # 混合检索（ANN + BM25 + RRF + Rerank）
│   │   └── cases.py             # 案例数据加载
│   ├── guardrails/
│   │   ├── sql_validator.py     # SQL 规则校验（SQLGlot）
│   │   └── critic.py            # LLM 审查（Critic Pattern）
│   └── executor/
│       ├── base.py              # QueryExecutor 抽象接口
│       ├── starrocks_executor.py # StarRocks 实现（生产对齐）
│       └── duckdb_executor.py   # DuckDB 实现（开发兜底）
├── tests/
│   ├── test_mcp_tools.py        # MCP 工具 + 权限过滤
│   ├── test_sql_validator.py    # 护栏规则
│   └── test_retriever.py        # 检索质量
└── evals/
    ├── golden_set.yaml          # 30 条取数任务 Golden Set
    └── eval_runner.py           # 评测执行器
```

## 架构总览

```
业务调用方（Web/BI/IM 机器人）
      │  HTTP/SSE（/query、/query/stream）
      ▼
┌───────────────────────────────────────────────────────────────┐
│ Agent Service（FastAPI + Docker）                             │
│                                                               │
│   LangGraph Agent（ReAct + Checkpoint，State 显式管理）       │
│   需求解析 → 元数据查询 → 案例检索 → SQL 生成 → 校验 → 执行   │
│   （validate / critic 不通过 → 回 generate 重试，上限 2 次）  │
│                                                               │
│      │            │           │           │                   │
│      ▼            ▼           ▼           ▼                   │
│   模型路由层     MCP 客户端    RAG 案例库    护栏层           │
│   DeepSeek      (SSE)        Milvus      SQLGlot              │
│   +Ollama                    +BM25+RRF   +Critic              │
└──────────────────────┬────────────────────────────────────────┘
                       │ SSE
                       ▼
┌───────────────────────────────────────────────────────────────┐
│ MCP Server（Python MCP SDK + SSE 服务化）                     │
│ 4 个元数据工具 + RBAC 表级权限过滤（无权限表检索层不可见）    │
│ 元数据仓库：tables / columns / lineage / metrics YAML         │
│───────────────────────────────────────────────────────────────│
│ 执行引擎：QueryExecutor 抽象                                  │
│   StarRocks（生产对齐）/ DuckDB（开发兜底）                   │
│───────────────────────────────────────────────────────────────│
│ 可观测：Langfuse（全链路 Trace / Token 成本 / 反馈闭环）      │
└───────────────────────────────────────────────────────────────┘
```

## 核心闭环

```
自然语言需求（带用户角色）→ 需求解析 → MCP 查元数据（权限过滤）
    → RAG 召回案例 → SQL 生成 → 规则校验 → Critic 审查
    → StarRocks 执行 → 结果 + 解释 → Langfuse Trace → 反馈回流
```

## 评测（Golden Set）

| 项目 | 内容 |
|------|------|
| 数据集 | 30 条取数任务（`evals/golden_set.yaml`） |
| 难度分层 | 简单聚合 12 / 多表 JOIN 8 / 口径 6 / 复杂嵌套 4（约 40%/27%/20%/13%） |
| 判定标准 | 执行成功 + 预期表全部出现 + 预期 SQL 关键字出现 |
| 主指标 | 要素准确率（详见 `evals/report.yaml`） |
| 副指标 | 执行成功率（详见 `evals/report.yaml`） |
| 运行方式 | `python scripts/run_evals.py` |
| 报告 | `evals/report.yaml`（总览 / 按难度分层 / 失败原因分类：校验失败、执行失败、要素缺失） |

> 运行 `python scripts/run_evals.py` 生成 `evals/report.yaml`，包含总览、按难度分层与失败原因分类。

## 安全说明（生产部署前必读）

本项目的服务接口与凭据默认值**仅面向本地演示**：

- `/query` 与 MCP Server 默认**无鉴权**（MCP 信任 `x-user` 头解析角色），生产部署前必须接入认证与鉴权。
- StarRocks 默认 `root` 空密码、MinIO 默认 `minioadmin/minioadmin`（见 `deploy/infra-compose.yml`），生产环境必须修改。
- 所有密钥（`DEEPSEEK_API_KEY`、`LANGFUSE_PUBLIC_KEY`、`LANGFUSE_SECRET_KEY`）均通过环境变量注入，请勿写入 `config.yaml` 或提交到仓库。

## 测试

```bash
# 运行全部单测
pytest tests/

# 运行单个测试文件
pytest tests/test_executor.py

# 运行单个测试函数
pytest tests/test_executor.py::test_duckdb_execute

# 带覆盖率（可选，需安装 coverage）
pytest tests/ --cov=dataagent --cov-report=term-missing
```

## 常见问题（FAQ）

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| `HF_ENDPOINT` 未设置，BGE 模型下载失败 | HuggingFace 国内访问受限 | `export HF_ENDPOINT=https://hf-mirror.com` 后重试 |
| Milvus 容器启动后端口不通 | etcd/minio 初始化未完成 | `docker compose -f deploy/infra-compose.yml logs -f` 等待就绪，通常需 30-60s |
| `docker compose up` 报错磁盘不足 | Milvus + etcd + minio 占用较大 | 至少预留 5GB 磁盘空间；或用 Milvus Lite（仅 `milvus-lite` 包） |
| StarRocks allin1 内存不足（OOM） | allin1 镜像约需 4GB+ | 改用 DuckDB 兜底：`DW_EXECUTOR=duckdb python scripts/init_warehouse.py` |
| 启动 MCP Server 报 `ModuleNotFoundError` | 未激活虚拟环境 | `source .venv/bin/activate` 后重新执行 |
| `pip install` 报错 `setuptools>=82` 与 `pymilvus` 不兼容 | pymilvus 内部 import pkg_resources | 安装要求已包含 `setuptools<82` 约束；如仍报错请手动 `pip install 'setuptools<82'` |
| 跑评测时准确率远低于预期 | 评测依赖 LLM 质量 + DeepSeek API Key | 确保 `DEEPSEEK_API_KEY` 已正确配置；可切换 Ollama 本地模型（`config.yaml` 中修改 `routing`） |
