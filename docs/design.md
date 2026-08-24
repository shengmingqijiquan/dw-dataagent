# dw-dataagent 设计文档

> 版本：v3.0 | 2026-08-23
> v3.0 变更：对标生产环境——StarRocks 执行引擎 / MCP SSE 服务化 / Agent FastAPI 服务 / 表级权限 / Langfuse 全程可观测 / pytest

## 1. 项目定位与命名

**项目名**：`dw-dataagent`（数仓取数 DataAgent）

**一句话定位**：面向数仓取数场景的生产级 DataAgent 服务——自然语言需求 → 多步 Agent 推理 → 生成并执行 SQL → 返回可解释结果，全程可观测、可评测、可审计。

**项目背景**：

> 业务取数需求是数仓团队最高频的场景。本项目设计并实现了一个生产级 DataAgent 服务：LangGraph 做 Agent 循环，MCP SSE 模式标准化接入元数据服务（含表级权限过滤），Milvus 支撑 RAG 检索历史 SQL 案例，模型路由层做成本与合规控制，四层校验保障 SQL 质量，StarRocks 执行真实 OLAP 查询，Langfuse 全程 Trace 可观测，Golden Set 驱动迭代。

**成功标准**：

| 标准 | 具体指标 |
|------|---------|
| 服务闭环 | REST 接口输入需求 → 正确 SQL + 执行结果 |
| 取数准确率 | Golden Set 30 条，准确率 ≥ 80% |
| 生产要素 | 权限过滤/可观测/单测/服务化部署全部就位 |
| 设计清晰 | 每个组件都能讲清「为什么这么设计」 |

## 2. 技术架构与组件设计

### 2.1 架构总图（生产形态）

```
┌────────────────────────────────────────────────────────────┐
│                     业务调用方（Web/BI/IM 机器人）              │
└──────────────────────────┬─────────────────────────────────┘
                           ↓ HTTP/SSE
┌────────────────────────────────────────────────────────────┐
│            Agent Service（FastAPI + Docker）                 │
│  ├─ POST /query          同步取数接口                        │
│  ├─ POST /query/stream   SSE 流式接口（思考过程可见）          │
│  └─ GET  /health         健康检查                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         LangGraph Agent（ReAct 循环 + Checkpoint）      │  │
│  │  需求解析 → 元数据查询 → 案例检索 → SQL生成 → 校验 → 执行  │  │
│  └──────────────────────────────────────────────────────┘  │
│        ↓              ↓            ↓             ↓         │
│  ┌───────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ 模型路由层  │  │ MCP 客户端│  │ RAG 检索  │  │ 护栏层    │  │
│  │ DeepSeek  │  │ (SSE连接)│  │ (Milvus) │  │ 规则引擎+  │  │
│  │ + Ollama  │  └──────────┘  └──────────┘  │ Critic   │  │
│  └───────────┘       ↓                        └──────────┘  │
└──────────────────────┼──────────────────────────────────────┘
                       ↓ HTTP/SSE
┌────────────────────────────────────────────────────────────┐
│            MCP Server（FastAPI + SSE 服务化部署）              │
│  ├─ query_table_list(domain)        按主题域查表（权限过滤）   │
│  ├─ query_table_schema(table)       查字段结构（权限过滤）     │
│  ├─ query_lineage(table)            查血缘（权限过滤）         │
│  └─ query_metric_definition(metric) 查指标口径               │
│              ↓ 内部对接（生产=元数据中心 API）                   │
│       元数据仓库（tables/columns/lineage/metrics YAML）       │
└────────────────────────────────────────────────────────────┘
                       ↓
┌────────────────────────────────────────────────────────────┐
│  执行引擎：QueryExecutor 抽象                                │
│  ├─ StarRocks 实现（allin1 Docker，生产对齐）                 │
│  └─ DuckDB 实现（开发兜底，接口相同）                          │
└────────────────────────────────────────────────────────────┘
                       ↑
┌────────────────────────────────────────────────────────────┐
│  可观测：Langfuse（Trace / Token 成本 / 用户反馈闭环）          │
└────────────────────────────────────────────────────────────┘
```

### 2.2 六大组件职责与选型

| 组件 | 职责 | 技术选型 | 选型理由 |
|------|------|---------|-------------------|
| **Agent 循环** | 多步推理编排 | LangGraph + Checkpoint | 显式状态管理，可暂停/恢复；生产级持久化 |
| **模型路由** | 成本/合规控制 | Router + DeepSeek API + Ollama Qwen3 | Tiered Model Stack；生产私有化部署零改动 |
| **MCP Server** | 元数据查询工具 | Python MCP SDK + SSE 服务化 | SSE 支持多客户端远程访问，企业标准部署形态 |
| **RAG 案例库** | 历史 SQL 检索 | Milvus + BGE-large-zh | 国内最主流；2.5+ 原生 BM25 混合检索 |
| **SQL 校验** | 生成质量保障 | SQLGlot + 规则引擎 + Critic | 确定性校验 + LLM 语义审查双层 |
| **执行引擎** | SQL 真实执行 | StarRocks（生产）+ DuckDB（兜底） | 真实 OLAP 语义；抽象层隔离实现差异 |

### 2.3 模拟数仓设计（电商域）

**规模**：5 个主题域、30 张表、50 条历史 SQL 案例

| 主题域 | 表示例 | 元数据内容 |
|--------|--------|-----------|
| 订单域 | dwd_order_detail_di, dws_order_summary_di | 表结构/字段注释/分区/血缘 |
| 用户域 | dwd_user_behavior_di, dws_user_active_di | 同上 + 指标口径 |
| 商品域 | dim_product_info, dws_product_gmv_di | 同上 |
| 支付域 | dwd_payment_detail_di, dws_payment_summary_di | 同上 |
| 物流域 | dwd_logistics_tracking_di | 同上 |

**元数据模型**（四类元数据）：

```yaml
# 1. 表元数据 tables
table_name: dws_order_summary_di
domain: 订单域
layer: DWS
granularity: 日
description: 订单汇总日表

# 2. 字段元数据 columns
table_name: dws_order_summary_di
column_name: gmv_amount
data_type: DECIMAL(20,2)
comment: 成交总额（支付成功口径）

# 3. 血缘关系 lineage
source_table: dwd_order_detail_di
target_table: dws_order_summary_di
relation: ETL_JOIN

# 4. 指标口径 metrics
metric_name: GMV
definition: 支付成功订单的成交总额
formula: SUM(gmv_amount) WHERE pay_status='paid'
```

### 2.4 权限模型（生产数据合规）

```
用户 → 角色 → 表级权限
roles.yaml:
  data_analyst:      [订单域, 用户域, 商品域]   # 分析角色可见业务域
  finance_analyst:   [订单域, 支付域]           # 财务角色额外可见支付域
  admin:             [*]                        # 管理员全表可见

MCP 工具执行时按用户角色过滤返回结果：
  query_table_list("全部") by 数据分析师
  → 只返回其有权限的表，无权限表完全不可见（避免侧信道泄露）
```

**设计要点**：
> 生产 DataAgent 的第一道安全防线是权限过滤——不是 SQL 生成后拦截，而是元数据检索阶段就不可见。无权限的表对 Agent 来说"不存在"，从源头消除越权取数的可能，对应主流数据平台的 RBAC 数据权限模型。

### 2.5 核心数据流

```
1. 用户（携带角色）提交："统计最近30天各品类GMV，按日趋势输出"
2. Agent 需求解析（Router → 小模型）→ 取数场景、订单域、GMV指标、日粒度
3. Agent 调 MCP query_table_schema → 返回权限内候选表结构
4. Agent 调 MCP query_metric_definition("GMV") → 口径定义
5. Agent 调 RAG search_cases → Milvus 混合检索召回相似历史 SQL（2-3 条）
6. Agent 生成 SQL（Router → 主力模型：表结构 + 口径 + 案例）
7. 规则引擎校验：语法 ✓ 表存在性 ✓ 分区 ✓ 命名 ✓ 只读 ✓ 权限内表 ✓
8. Critic LLM 审查：JOIN 合理性、聚合逻辑、口径一致性
9. StarRocks 执行 → 返回结果 + 解释（用了哪些表、什么口径、Token 成本）
10. Langfuse 记录全链路 Trace；用户反馈回流 → Golden Set 更新
```

### 2.6 技术栈清单

```
Agent 编排：langgraph, langchain-core
LLM：langchain-deepseek（DeepSeek API）+ langchain-ollama（本地 Qwen3）
MCP：mcp (Python SDK) + langchain-mcp-adapters；SSE 服务化（FastAPI 挂载）
向量库：pymilvus + Milvus standalone（Docker Compose: etcd+minio+milvus）
Embedding：sentence-transformers + BAAI/bge-large-zh-v1.5
Rerank：BAAI/bge-reranker-v2-m3（加分项）
执行引擎：StarRocks allin1 Docker（生产对齐）+ duckdb（兜底）
校验：sqlglot
可观测：langfuse（开发 Langfuse Cloud / 生产自托管）
服务框架：FastAPI + uvicorn（Agent Service + MCP Server 双服务）
测试：pytest
部署：docker-compose 全栈编排 + Dockerfile
```

### 2.7 模型路由层设计

| Tier | 模型 | 用途 | 理由 |
|------|------|------|------|
| Tier 1（快/便宜） | Qwen3-8B（Ollama）或 DeepSeek-chat | 需求解析、格式校验 | 简单任务，成本优先 |
| Tier 2（主力） | DeepSeek-V3（API）或 Qwen3-32B（本地） | SQL 生成、Critic 审查 | 代码生成主力，质量优先 |

```yaml
# config.yaml
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
```

**设计要点**：
> 模型路由层解决两个问题：成本和质量分层。同时它隔离了模型差异——生产环境因数据合规要求 vLLM 私有化部署 Qwen/DeepSeek 时，业务代码零改动，只改路由配置。这正是主流大厂模型网关的设计思路。

## 3. 开发计划与里程碑（8 天冲刺）

### Day 1：基础设施 + 模拟数仓

**任务**：
1. Python venv + requirements.txt
2. docker-compose 启动：Milvus standalone（etcd+minio+milvus）+ StarRocks allin1
3. DeepSeek API Key 配置；Ollama + qwen3:8b（兜底）
4. `scripts/init_warehouse.py`：StarRocks 建 30 张表 + 灌模拟数据（StarRocks 跑不动则切 DuckDB，QueryExecutor 抽象保证零改动）
5. 元数据文件：tables/columns/lineage/metrics.yaml + roles.yaml（权限）
6. 50 条历史 SQL 案例

**验收**：StarRocks（或 DuckDB）可查表；元数据+权限文件完整；案例就绪

### Day 2：MCP Server 开发（SSE 服务化 + 权限）

**任务**：
1. `dataagent/mcp_server/metadata_server.py`：4 个工具（table_list/table_schema/lineage/metric_definition）
2. **权限过滤**：工具调用携带用户角色，按 roles.yaml 过滤返回
3. **SSE 服务化**：FastAPI 挂载 MCP SSE 端点（`/mcp/sse`）
4. pytest 单测：4 个工具的正确性 + 权限过滤边界（越权表不可见）

**验收**：MCP Server 作为 HTTP 服务运行；pytest 全绿；越权表被过滤

### Day 3：RAG 案例库构建

**任务**：
1. `dataagent/rag/cases.py` + `indexer.py`：BGE-large-zh Embedding + Milvus 入库
2. 案例分块：需求+SQL 配对整条入库（语义完整单元）
3. `scripts/build_rag_index.py`
4. pytest：入库数量、向量维度正确

**验收**：Milvus 中 50 条案例；相似需求召回对应案例

### Day 4：混合检索 + Rerank

**任务**：
1. `dataagent/rag/retriever.py`：向量 ANN + BM25（Milvus 原生）+ RRF 融合
2. Rerank 加分项：bge-reranker-v2-m3
3. 检索质量评估：10 条测试需求，Top-5 相关性人工评分

**验收**：混合检索 Top-5 相关性优于单一方式，评估结果留档

### Day 5：LangGraph Agent 核心 + 服务化

**任务**：
1. `dataagent/agent/state.py` + `prompts.py` + `graph.py`
2. 节点：parse → 工具调用循环（MCP + RAG）→ generate → validate → execute
3. `langchain-mcp-adapters` 通过 SSE 客户端加载 MCP 工具
4. `dataagent/llm/router.py` 模型路由
5. **Langfuse 集成**：每个节点 Trace + Token 统计
6. **FastAPI 服务**：POST /query + SSE 流式接口 + Dockerfile
7. 跑通首条取数需求

**验收**：REST 接口 → SQL → StarRocks 执行结果完整闭环；Langfuse 可见 Trace

### Day 6：护栏体系

**任务**：
1. `dataagent/guardrails/sql_validator.py`（SQLGlot）：语法/表存在/分区/命名/只读/权限内表
2. `dataagent/guardrails/critic.py`：LLM 审查（口径一致性、JOIN 合理性）
3. 状态机约束：每阶段工具白名单
4. 停止条件：最大步数、重试上限
5. pytest：非法 SQL（错误表名/DROP/越权表/缺分区）100% 拦截

**验收**：护栏单测全绿；人工构造的非法 SQL 全部被拦

### Day 7：Golden Set + Evals

**任务**：
1. `evals/golden_set.yaml`：30 条取数任务（简单聚合 40% / 多表 JOIN 30% / 口径 20% / 复杂嵌套 10%）
2. `evals/eval_runner.py`：执行成功率 + 关键要素准确率 + 失败原因分类
3. 基于失败原因迭代

**验收**：准确率可量化；失败原因分类清晰

### Day 8：收尾

**任务**：
1. 准确率优化至 ≥ 80%
2. 全栈 docker-compose 编排收尾
3. README 更新（架构图 + 准确率数据 + 成本数据）
4. 文档收尾（README / 设计文档）

**验收**：准确率 ≥ 80%；文档收尾完成

## 4. 关键设计决策（「为什么」速查）

| 决策 | 理由 |
|------|------|
| 为什么 Agent 用 LangGraph？ | 显式状态管理 + Checkpoint 持久化，生产级可恢复；条件路由可视化 |
| 为什么 MCP 用 SSE 而不是 stdio？ | SSE 支持多客户端远程访问，是服务化部署标准形态；stdio 仅限本地单进程调试 |
| 为什么 MCP Server 独立部署？ | 元数据服务与 Agent 服务解耦：权限控制集中管理，多 Agent 复用，独立扩缩容 |
| 为什么向量库选 Milvus？ | 国内最主流；2.5+ 原生 BM25；阿里云/腾讯云托管版平滑迁移路径 |
| 为什么权限在元数据层做而不是 SQL 层？ | 从源头阻断——无权限的表对 Agent 不可见，比生成后拦截更安全（消除侧信道） |
| 为什么执行层做 QueryExecutor 抽象？ | 开发/生产引擎隔离：DuckDB 兜底开发，StarRocks 生产对齐，Agent 代码零改动 |
| 为什么校验用 SQLGlot + Critic 双层？ | 确定性规则兜底（100% 拦截非法），LLM 补语义审查（口径/JOIN 合理性） |
| 为什么模型路由而不是单一模型？ | 成本分层 + 合规隔离；生产私有化部署零改动切换 |
| 为什么 Langfuse？ | 开源主流，Trace/成本/反馈三合一，自托管满足数据合规 |

## 5. 风险与兜底

| 风险 | 兜底方案 |
|------|---------|
| StarRocks allin1 内存不足/ARM 兼容问题 | 切 DuckDB（QueryExecutor 抽象保证零改动） |
| BGE-large-zh 下载失败 | 降级 bge-small-zh |
| Milvus Compose 资源不足 | 降级 Milvus Lite（API 兼容） |
| DeepSeek API 不可用 | 切 Ollama Qwen3（config.yaml 一行切换） |
| Ollama 内存不足 | 纯 API 方案，本地模型仅作可选兜底 |
| Rerank 模型太大 | 跳过，RRF 融合够用 |
| 时间不足 | 砍 Rerank + 流式接口，保核心闭环 + 权限 + 可观测 |

## 6. 技术栈与主流生产实践对齐

| 组件 | 主流生产实践 | 本项目 | 选型说明 |
|------|------------------------|--------|---------|
| LLM | 自研 + 开源私有化（vLLM 集群）+ 模型网关 | 路由层：DeepSeek API + Ollama Qwen3 | 生产因数据合规必须私有化，路由层支持无缝切换 |
| 向量库 | Milvus / 自研向量存储 | Milvus standalone | 国内生态最成熟，2.5+ 原生 BM25 省一套 ES |
| Agent | LangChain/LangGraph 或自研 | LangGraph | 通用标准，状态图模型可迁移自研框架 |
| OLAP | StarRocks/ClickHouse/Doris | StarRocks + QueryExecutor 抽象 | 执行层接口化，引擎可替换 |
| 元数据 | DataHub/Atlas/自研元数据中心 | MCP Server 对接 YAML 元数据仓库 | MCP 工具接口不变，生产直接接 DataHub API |
| 数据安全 | RBAC 表级权限 + 脱敏 | 元数据层权限过滤 | 权限前置到元数据检索，从源头消除越权 |
| 可观测 | Langfuse/自研平台 | Langfuse | Trace + Token 成本 + 反馈闭环三位一体 |
