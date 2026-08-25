<div align="center">

<p>中文 | <a href="./README_EN.md">English</a></p>
<h1>nl2insight</h1>
<p>
  <strong>面向数仓场景的 NL-to-Insight Agent</strong>
</p>
<p>
  自然语言 → SQL 生成 → 执行 → 数据洞察 &nbsp;|&nbsp; LangGraph Agent &nbsp;|&nbsp; MCP Server &nbsp;|&nbsp; RAG 检索 &nbsp;|&nbsp; SQLGlot 校验 &nbsp;|&nbsp; Langfuse 可观测
</p>

<p>
  <img src="https://img.shields.io/badge/Python-3.10+-blue" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-red" alt="License">
  <img src="https://img.shields.io/badge/LangGraph-0.2.60+-purple" alt="LangGraph">
  <img src="https://img.shields.io/badge/FastAPI-0.115.0+-green" alt="FastAPI">
  <img src="https://img.shields.io/badge/StarRocks-Prodb-blueviolet" alt="StarRocks">
</p>

<p>
  <a href="#-项目简介">项目简介</a> •
  <a href="#-核心特性">核心特性</a> •
  <a href="#-快速开始">快速开始</a> •
  <a href="#-测试与评测">测试与评测</a> •
  <a href="#-安全说明">安全说明</a> •
  <a href="#-常见问题">常见问题</a>
</p>

</div>

<br/>

## 📖 项目简介

**nl2insight** 是一个面向数仓场景的 NL-to-Insight Agent 服务。它将自然语言需求转化为可执行的 SQL，经过多步 Agent 推理、RAG 案例检索、规则校验与 Critic 审查后，最终在 StarRocks（或 DuckDB）上执行，并返回带解释的数据洞察结果。

项目全面对标主流互联网企业的 AI 数据应用生产实践，涵盖 **模型路由、元数据 MCP 服务化、表级 RBAC 权限、双层 SQL 护栏、全链路 Langfuse 追踪** 等核心生产要素。

> 本项目代码由 AI 辅助生成，人工负责架构设计、代码审查与测试。

## ✨ 核心特性

| 特性 | 说明 |
| :--- | :--- |
| **多步 Agent 推理** | 基于 LangGraph 的 ReAct 循环，显式 State 管理 + Checkpoint 持久化，支持需求解析→工具调用→SQL 生成→校验→执行的完整闭环。 |
| **模型路由层** | DeepSeek API + Ollama Qwen3 混合部署；简单任务走低成本模型，核心生成走主力模型；生产私有化部署零代码改动。 |
| **MCP 服务化** | 4 个元数据工具（表列表/表结构/血缘/指标口径）通过 SSE 协议对外暴露，支持多客户端并发访问。 |
| **表级 RBAC 权限** | 权限控制前置到元数据检索层，无权限的表对 Agent 完全不可见，从源头消除越权取数风险。 |
| **RAG 案例库** | Milvus + BGE-large-zh + BM25 混合检索，召回相似历史 SQL 案例辅助生成，RRF 融合提升召回质量。 |
| **双层 SQL 护栏** | SQLGlot 规则引擎（语法/只读/表存在/分区/命名）做确定性拦截，LLM Critic 做语义审查，双层兜底。 |
| **可观测体系** | Langfuse 全链路 Trace，记录每个节点的 Token 消耗与延迟，支持用户反馈回流驱动迭代。 |
| **引擎可替换** | QueryExecutor 抽象层隔离 StarRocks（生产）与 DuckDB（开发兜底），接口一致，切换零改动。 |
| **Golden Set 评测** | 30 条取数任务，按简单/JOIN/口径/嵌套分层（40%/27%/20%/13%），自动化评测 + 失败原因分类。 |

## 🏗️ 架构总览

```
业务调用方（Web/BI/IM 机器人）
      │  HTTP/SSE（/query、/query/stream）
      ▼
┌───────────────────────────────────────────────────────────────┐
│ Agent Service（FastAPI + Docker）                             │
│                                                               │
│   LangGraph Agent（ReAct + Checkpoint）                       │
│   需求解析 → 元数据查询 → 案例检索 → SQL 生成 → 校验 → 执行   │
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
│ 4 个元数据工具 + RBAC 表级权限过滤                            │
│───────────────────────────────────────────────────────────────│
│ 执行引擎：StarRocks / DuckDB（QueryExecutor 抽象）            │
│───────────────────────────────────────────────────────────────│
│ 可观测：Langfuse（Trace / Token / 反馈闭环）                  │
└───────────────────────────────────────────────────────────────┘
```

## 🚀 快速开始

> 详细文档见 [docs/design.md](docs/design.md)。

### 前置依赖

- Python ≥ 3.10（推荐 3.11+）
- Docker + Docker Compose（启动 Milvus 等基础设施）
- DeepSeek API Key（或本地 Ollama）

### 安装与启动

```bash
# 1. 克隆并安装依赖
git clone <repo-url> && cd nl2insight
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY

# 3. 启动基础设施（Milvus）
docker compose -f deploy/infra-compose.yml up -d

# 4. 初始化模拟数仓
python scripts/init_warehouse.py

# 5. 构建 RAG 案例库
python scripts/build_rag_index.py
# 国内网络可用镜像：
# HF_ENDPOINT=https://hf-mirror.com python scripts/build_rag_index.py

# 6. 启动服务（两个终端）
python -m nl2insight.mcp_server.server --port 8001   # MCP Server
python -m nl2insight.api --port 8000                   # Agent Service

# 7. 调用接口
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "统计最近30天各品类GMV，按日趋势输出", "role": "data_analyst"}'
```

## 🧪 测试与评测

```bash
# 运行全部单测
pytest tests/

# 运行单个测试文件
pytest tests/test_executor.py

# 运行单个测试函数
pytest tests/test_executor.py::test_duckdb_execute

# 带覆盖率
pytest tests/ --cov=nl2insight --cov-report=term-missing

# 跑 Golden Set 评测
python scripts/run_evals.py
# 报告输出至 evals/report.yaml
```

## 📚 文档导航

| 文档 | 内容 |
| :--- | :--- |
| [设计文档](docs/design.md) | 架构设计、组件选型理由、风险兜底方案 |
| [.env.example](.env.example) | 环境变量模板与说明 |

## 🛡️ 安全说明

本项目默认配置**仅面向本地演示**，生产部署前请务必阅读：

- `/query` 与 MCP Server 默认**无鉴权**（MCP 信任 `x-user` 头解析角色），生产部署前必须接入认证与鉴权。
- StarRocks 默认 `root` 空密码、MinIO 默认 `minioadmin/minioadmin`，生产环境必须修改。
- 所有密钥（`DEEPSEEK_API_KEY`、`LANGFUSE_PUBLIC_KEY`、`LANGFUSE_SECRET_KEY`）均通过环境变量注入，请勿写入 `config.yaml` 或提交到仓库。

## ❓ 常见问题（FAQ）

| 问题 | 解决方案 |
|------|---------|
| BGE 模型下载失败 | 设置 `HF_ENDPOINT=https://hf-mirror.com` 后重试 |
| Milvus 端口不通 | `docker compose logs -f` 等待 etcd/minio 初始化完成（约 30-60s） |
| Docker Compose 磁盘不足 | 预留至少 5GB；或改用 Milvus Lite（仅 `milvus-lite` 包） |
| StarRocks 内存不足（OOM） | 改用 DuckDB 兜底：`DW_EXECUTOR=duckdb python scripts/init_warehouse.py` |
| MCP Server 报 ModuleNotFoundError | 激活虚拟环境：`source .venv/bin/activate` 后重试 |
| pip install 报 setuptools 冲突 | `pip install 'setuptools<82'` 后再安装依赖 |
| 评测准确率远低于预期 | 确保 `DEEPSEEK_API_KEY` 已正确配置；可切换 Ollama 本地模型 |

## 🤝 贡献指南

欢迎所有形式的贡献！

1. Fork 本项目
2. 创建功能分支：`git checkout -b feature/your-feature`
3. 提交更改：`git commit -m 'feat: add your feature'`
4. 推送分支：`git push origin feature/your-feature`
5. 创建 Pull Request

## 📄 许可证

本项目基于 [MIT License](LICENSE) 开源协议。

---

<div align="center">

**⭐ 如果这个项目对你有帮助，请给我们一个 Star！**

</div>
