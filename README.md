# dw-dataagent · 数仓取数 DataAgent（生产级对标）

> 面向数仓取数场景的生产级 DataAgent 服务：自然语言需求 → 多步 Agent 推理 → 生成并执行 SQL → 返回可解释结果，全程可观测、可评测、可审计。
>
> **技术栈**：Python / LangGraph / MCP(SSE) / Milvus / BGE / StarRocks / SQLGlot / Langfuse / FastAPI

## 项目定位

这是面试准备实操项目，**技术选型与部署形态对标主流互联网企业（字节/阿里/美团系）的 AI 数据应用生产实践**，覆盖大厂 AI Data Agent 岗位面试的全部核心考点：

> 本项目代码由 AI 辅助生成，人工负责架构设计、代码审查与测试。

| 组件 | 技术 | 面试考点 |
|------|------|---------|
| Agent 循环 | LangGraph（ReAct + State + Checkpoint） | Agent 框架选型、状态管理、循环控制 |
| Agent 服务 | FastAPI + SSE 流式接口 + Docker | 服务化部署、异步任务模式 |
| 模型路由 | DeepSeek API + Ollama Qwen3（vLLM 生产叙事） | Tiered Model Stack、数据合规、成本控制 |
| MCP Server | Python MCP SDK + SSE 服务化 | MCP 协议、工具标准化、服务化部署 |
| 数据安全 | 角色 → 表级权限过滤（元数据层） | RBAC、数据合规、越权防护 |
| RAG 案例库 | Milvus + BGE-large-zh + 混合检索 | 向量库选型、Embedding、检索策略 |
| SQL 校验 | SQLGlot + 规则引擎 + Critic | 护栏设计、双层校验 |
| 执行引擎 | StarRocks（QueryExecutor 抽象） | OLAP 选型、引擎可替换设计 |
| Evals | Golden Set + 自动化评测 | 评测体系、迭代方法论 |
| 可观测 | Langfuse（Trace/成本/反馈） | 全链路追踪、成本分析 |

## 快速开始

```bash
# 1. Python 环境
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. 基础设施（Milvus 向量库：etcd+minio+milvus standalone；StarRocks 按需单独启动，见 design.md §5 风险表）
docker compose -f deploy/infra-compose.yml up -d

# 3. LLM 配置（模型路由）
#    A. DeepSeek API（开发主力）
export DEEPSEEK_API_KEY="sk-xxx"
#    B. 本地 Ollama（数据不出域兜底）
ollama pull qwen3:8b

# 4. 初始化模拟数仓（建表 + 数据 + 元数据 + 权限）
python scripts/init_warehouse.py                 # DuckDB 引擎（默认，开发兜底）
# python scripts/init_warehouse.py --engine starrocks  # StarRocks 引擎（按需，需容器已启动）
#   —— 注：StarRocks 容器验证待有条件环境（本机无 Docker 时跳过，DuckDB 兜底已验证）

# 5. 构建 RAG 案例库（BGE Embedding + Milvus 入库）
python scripts/build_rag_index.py
#   —— 本机开发环境经 config.yaml services.milvus.uri 常驻使用 Milvus Lite
#      （data/milvus.db）；生产可切换 Milvus standalone（deploy/infra-compose.yml）；
#      BGE 模型约 1.3GB，国内网络下可用镜像
#      `HF_ENDPOINT=https://hf-mirror.com python scripts/build_rag_index.py`

# 6. 启动 MCP Server（SSE 服务）
python -m dataagent.mcp_server.server --port 8001

# 7. 启动 Agent Service（FastAPI）
python -m dataagent.api --port 8000

# 8. 调用取数接口
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "统计最近30天各品类GMV，按日趋势输出", "role": "data_analyst"}'

# 9. 跑 Golden Set 评测
python scripts/run_evals.py

# 10. 跑单测
pytest tests/
```

## 目录结构

```
dw-dataagent/
├── README.md                    # 本文档
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
| 主指标 | 要素准确率（XX%，待全量评测实测回填） |
| 副指标 | 执行成功率（XX%，待全量评测实测回填） |
| 运行方式 | `python scripts/run_evals.py` |
| 报告 | `evals/report.yaml`（总览 / 按难度分层 / 失败原因分类：校验失败、执行失败、要素缺失） |

> 评测结果需在有 DeepSeek API Key 的环境执行全量评测后，从 `evals/report.yaml` 实测回填（见 `docs/resume-entry.md` 中对应的 XX% 占位）。

## 里程碑

| Day | 里程碑 | 验收标准 |
|-----|--------|---------|
| 1 | 基础设施 + 模拟数仓就绪 | StarRocks 30 张表 + 元数据 + 权限文件 |
| 2 | MCP Server 服务化可用 | 4 工具 SSE 可调 + 权限过滤 + pytest 绿 |
| 3 | RAG 案例库可用 | Milvus 50 条案例可检索 |
| 4 | 混合检索完成 | 融合检索优于单一方式 |
| 5 | Agent 服务跑通闭环 | REST → SQL → StarRocks 结果 + Langfuse Trace |
| 6 | 护栏就位 | 非法 SQL 100% 拦截 + 单测绿 |
| 7 | Golden Set 评测 | 准确率可量化 |
| 8 | 准确率 ≥ 80%（目标，待实测回填） | 30 条评测达标 + 简历条目就绪 |
