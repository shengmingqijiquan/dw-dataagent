<div align="center">

<p><a href="./README.md">中文</a> | English</p>
<h1>nl2insight</h1>
<p>
  <strong>A Production-Grade DataAgent for Data Warehouse Querying</strong>
</p>
<p>
  Text-to-SQL &nbsp;|&nbsp; LangGraph Agent &nbsp;|&nbsp; MCP Server &nbsp;|&nbsp; RAG Retrieval &nbsp;|&nbsp; SQLGlot Validation &nbsp;|&nbsp; Langfuse Observability
</p>

<p>
  <img src="https://img.shields.io/badge/Python-3.10+-blue" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-red" alt="License">
  <img src="https://img.shields.io/badge/LangGraph-0.2.60+-purple" alt="LangGraph">
  <img src="https://img.shields.io/badge/FastAPI-0.115.0+-green" alt="FastAPI">
  <img src="https://img.shields.io/badge/StarRocks-Production-blueviolet" alt="StarRocks">
</p>

<p>
  <a href="#-project-overview">Overview</a> •
  <a href="#-key-features">Features</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-testing--evaluation">Testing</a> •
  <a href="#-security-notes">Security</a> •
  <a href="#-faq">FAQ</a>
</p>

</div>

<br/>

## 📖 Project Overview

**nl2insight** is a production-grade DataAgent service for data warehouse querying scenarios. It transforms natural language requests into executable SQL through multi-step Agent reasoning, RAG case retrieval, rule validation, and Critic review, finally executing on StarRocks (or DuckDB) and returning explainable analysis results.

The project aligns with mainstream AI data application production practices, covering **model routing, MCP service deployment, table-level RBAC permissions, dual-layer SQL guardrails, and full-link Langfuse tracing** — core production requirements.

> Code generated with AI assistance; architecture design, code review, and testing are done by humans.

## ✨ Key Features

| Feature | Description |
| :--- | :--- |
| **Multi-Step Agent Reasoning** | LangGraph-based ReAct loop with explicit State management and Checkpoint persistence, supporting the full pipeline: parse → tool call → SQL generate → validate → execute. |
| **Model Routing Layer** | DeepSeek API + Ollama Qwen3 hybrid deployment; low-cost models for simple tasks, main models for core generation; zero code changes for private deployment. |
| **MCP Service** | 4 metadata tools (table list/schema/lineage/metrics) exposed via SSE protocol, supporting multi-client concurrent access. |
| **Table-Level RBAC** | Permission filtering at the metadata retrieval layer — unauthorized tables are invisible to the Agent, eliminating privilege escalation risk at the source. |
| **RAG Case Library** | Milvus + BGE-large-zh + BM25 hybrid retrieval, RRF fusion for improved recall quality, using historical SQL cases to assist generation. |
| **Dual-Layer SQL Guardrails** | SQLGlot rule engine (syntax/read-only/table existence/partition/naming) for deterministic blocking, LLM Critic for semantic review. |
| **Observability** | Langfuse full-link tracing, recording Token consumption and latency per node, supporting user feedback for iteration. |
| **Replaceable Engines** | QueryExecutor abstraction isolates StarRocks (production) and DuckDB (dev fallback) with identical interfaces. |
| **Golden Set Evaluation** | 30 data query tasks across 4 difficulty levels (40%/27%/20%/13%), automated evaluation with failure classification. |

## 🏗️ Architecture Overview

```
Business caller (Web/BI/IM bot)
      │  HTTP/SSE (/query, /query/stream)
      ▼
┌───────────────────────────────────────────────────────────────┐
│ Agent Service (FastAPI + Docker)                              │
│                                                               │
│   LangGraph Agent (ReAct + Checkpoint)                        │
│   Parse → Metadata query → Case retrieval → SQL generate      │
│         → Validate → Execute                                  │
│                                                               │
│      │            │           │           │                   │
│      ▼            ▼           ▼           ▼                   │
│   Model routing   MCP client   RAG corpus   Guardrails       │
│   DeepSeek        (SSE)        Milvus      SQLGlot            │
│   +Ollama                   +BM25+RRF   +Critic               │
└──────────────────────┬────────────────────────────────────────┘
                       │ SSE
                       ▼
┌───────────────────────────────────────────────────────────────┐
│ MCP Server (Python MCP SDK + SSE service)                     │
│ 4 metadata tools + RBAC table-level permission filtering      │
│───────────────────────────────────────────────────────────────│
│ Execution engine: StarRocks / DuckDB (QueryExecutor abstract) │
│───────────────────────────────────────────────────────────────│
│ Observability: Langfuse (Trace / Token / feedback loop)       │
└───────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

> Detailed documentation: [docs/design.md](docs/design.md).

### Prerequisites

- Python ≥ 3.10 (3.11+ recommended)
- Docker + Docker Compose (for infrastructure services)
- DeepSeek API Key (or local Ollama)

### Installation & Launch

```bash
# 1. Clone and install dependencies
git clone <repo-url> && cd nl2insight
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure environment variables
cp .env.example .env
# Fill in DEEPSEEK_API_KEY in .env

# 3. Start infrastructure (Milvus)
docker compose -f deploy/infra-compose.yml up -d

# 4. Initialize simulated data warehouse
python scripts/init_warehouse.py

# 5. Build RAG corpus
python scripts/build_rag_index.py
# For Chinese network, use mirror:
# HF_ENDPOINT=https://hf-mirror.com python scripts/build_rag_index.py

# 6. Start services (two terminals)
python -m nl2insight.mcp_server.server --port 8001   # MCP Server
python -m nl2insight.api --port 8000                   # Agent Service

# 7. Call the query API
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "统计最近30天各品类GMV，按日趋势输出", "role": "data_analyst"}'
```

## 🧪 Testing & Evaluation

```bash
# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_executor.py

# Run a single test function
pytest tests/test_executor.py::test_duckdb_execute

# With coverage
pytest tests/ --cov=nl2insight --cov-report=term-missing

# Run Golden Set evaluation
python scripts/run_evals.py
# Report output to evals/report.yaml
```

## 📚 Documentation

| Document | Content |
| :--- | :--- |
| [Design Doc](docs/design.md) | Architecture design, component selection rationale, risk mitigation |
| [.env.example](.env.example) | Environment variable template with descriptions |

## 🛡️ Security Notes

The default configuration in this project is **for local demo only**. Please read carefully before production deployment:

- `/query` and MCP Server have **no authentication by default** (MCP trusts the `x-user` header for role resolution). Production deployments must integrate authentication and authorization.
- StarRocks defaults to `root` with empty password; MinIO defaults to `minioadmin/minioadmin`. **These must be changed in production.**
- All secrets (`DEEPSEEK_API_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`) are injected via environment variables. **Do not write them into `config.yaml` or commit to the repository.**

## ❓ FAQ

| Issue | Solution |
|-------|----------|
| BGE model download fails | Set `HF_ENDPOINT=https://hf-mirror.com` then retry |
| Milvus port unreachable | `docker compose logs -f` and wait for etcd/minio to initialize (~30-60s) |
| Docker Compose disk space error | Reserve at least 5GB; or use Milvus Lite (only `milvus-lite` package) |
| StarRocks OOM | Use DuckDB fallback: `DW_EXECUTOR=duckdb python scripts/init_warehouse.py` |
| `ModuleNotFoundError` on MCP start | Activate virtual environment: `source .venv/bin/activate` then retry |
| `pip install` setuptools conflict | `pip install 'setuptools<82'` then reinstall dependencies |
| Evaluation accuracy far below expected | Ensure `DEEPSEEK_API_KEY` is set; can switch to Ollama local model |

## 🤝 Contributing

All forms of contribution are welcome!

1. Fork the project
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -m 'feat: add your feature'`
4. Push: `git push origin feature/your-feature`
5. Open a Pull Request

## 📄 License

This project is open-sourced under the [MIT License](LICENSE).

---

<div align="center">

**⭐ If this project helps you, please give us a Star!**

</div>
