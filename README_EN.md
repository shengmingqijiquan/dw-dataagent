# dw-dataagent · DataWarehouse DataAgent (Production-Grade)

**[中文 README](README.md) · [Design Doc](docs/design.md)**

> A production-grade DataAgent service for data warehouse querying: natural language requests → multi-step Agent reasoning → SQL generation & execution → explainable results, with full observability, evaluation, and auditing.
>
> **Tech stack**: Python / LangGraph / MCP(SSE) / Milvus / BGE / StarRocks / SQLGlot / Langfuse / FastAPI
> **Python version**: ≥ 3.10 (3.11+ recommended)

## Project Overview

This is a production-grade DataAgent example project for data warehouse querying scenarios, **with technology choices and deployment patterns aligned with mainstream internet companies' AI data application practices**:

> Code generated with AI assistance; architecture design, code review, and testing are done by humans.

| Component | Technology | Design Highlights |
|-----------|------------|-------------------|
| Agent Loop | LangGraph (ReAct + State + Checkpoint) | Framework selection, state management, loop control |
| Agent Service | FastAPI + SSE streaming + Docker | Service deployment, async task pattern |
| Model Routing | DeepSeek API + Ollama Qwen3 | Tiered Model Stack, data compliance, cost control |
| MCP Server | Python MCP SDK + SSE service | MCP protocol, tool standardization, service deployment |
| Data Security | Role → table-level RBAC (metadata layer) | RBAC, data compliance, unauthorized access prevention |
| RAG Corpus | Milvus + BGE-large-zh + hybrid retrieval | Vector DB selection, Embedding, retrieval strategy |
| SQL Validation | SQLGlot + rule engine + Critic | Guardrail design, two-layer validation |
| Execution Engine | StarRocks (QueryExecutor abstraction) | OLAP selection, replaceable engine design |
| Evals | Golden Set + automated evaluation | Evaluation system, iterative methodology |
| Observability | Langfuse (Trace/cost/feedback) | Full-link tracing, cost analysis |

## Quick Start

```bash
# 0. Prerequisites
#    - Python ≥ 3.10 (3.11+ recommended)
#    - Docker + Docker Compose (for infrastructure services)
#    - Network access to Docker Hub / domestic mirror

# 1. Clone and install dependencies
git clone <repo-url> && cd dw-dataagent
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure environment variables
cp .env.example .env
# Fill in DEEPSEEK_API_KEY (configure others as needed)

# 3. Start infrastructure (Milvus: etcd+minio+milvus standalone)
docker compose -f deploy/infra-compose.yml up -d

# 4. LLM configuration (model routing)
#    A. DeepSeek API (primary for development) — set DEEPSEEK_API_KEY in .env
#    B. Local Ollama (for data-resident fallback)
ollama pull qwen3:8b

# 5. Initialize simulated data warehouse (create tables + data + metadata + permissions)
python scripts/init_warehouse.py                 # DuckDB (default, dev fallback)
# python scripts/init_warehouse.py --engine starrocks  # StarRocks (optional, requires container)
#   —— Note: requires StarRocks container running first; DuckDB fallback works without it (same interface)

# 6. Build RAG corpus (BGE Embedding + Milvus indexing)
python scripts/build_rag_index.py
#   —— Dev environment uses Milvus Lite by default via config.yaml (data/milvus.db)
#      Production can switch to Milvus standalone (deploy/infra-compose.yml)
#      BGE model is ~1.3GB; use mirror for Chinese network:
#      `HF_ENDPOINT=https://hf-mirror.com python scripts/build_rag_index.py`

# 7. Start MCP Server (SSE service)
python -m dataagent.mcp_server.server --port 8001

# 8. Start Agent Service (FastAPI)
python -m dataagent.api --port 8000

# 9. Call the query API
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "統計最近30天各品类GMV，按日趋势输出", "role": "data_analyst"}'

# 10. Run Golden Set evaluation
python scripts/run_evals.py
```

## Project Structure

```
dw-dataagent/
├── README.md                    # This doc (Chinese)
├── README_EN.md                 # This doc (English)
├── .env.example                 # Environment variable template
├── requirements.txt
├── config.yaml                  # Model routing / service addresses / permission config
├── Dockerfile                   # Agent Service image
├── docs/
│   └── design.md                # Design doc (architecture / components / dev plan)
├── deploy/
│   └── infra-compose.yml        # Milvus (etcd+minio+milvus standalone); StarRocks optional
├── scripts/
│   ├── init_warehouse.py        # Simulated data warehouse initialization
│   ├── build_rag_index.py       # RAG corpus construction
│   └── run_evals.py             # Golden Set evaluation
├── data/                        # Data files (gitignored)
│   ├── metadata/                # Metadata (tables/columns/lineage/metrics/roles)
│   └── cases/                   # Historical SQL cases (request+SQL pairs)
├── dataagent/
│   ├── __init__.py
│   ├── api.py                   # FastAPI Agent service (REST + SSE streaming)
│   ├── config.py                # Config loading (model routing / service addresses)
│   ├── agent/
│   │   ├── graph.py             # LangGraph state graph (nodes/edges/routing)
│   │   ├── state.py             # State definition
│   │   └── prompts.py           # Prompt templates
│   ├── llm/
│   │   └── router.py            # Model routing layer (Tiered Model Stack)
│   ├── mcp_server/
│   │   ├── server.py            # MCP Server (SSE service + FastAPI mount)
│   │   └── metadata.py          # Metadata query + permission filtering
│   ├── rag/
│   │   ├── indexer.py           # Chunking + BGE Embedding + Milvus indexing
│   │   ├── retriever.py         # Hybrid retrieval (ANN + BM25 + RRF + Rerank)
│   │   └── cases.py             # Case data loader
│   ├── guardrails/
│   │   ├── sql_validator.py     # SQL rule validation (SQLGlot)
│   │   └── critic.py            # LLM review (Critic Pattern)
│   └── executor/
│       ├── base.py              # QueryExecutor abstract interface
│       ├── starrocks_executor.py # StarRocks implementation (production aligned)
│       └── duckdb_executor.py   # DuckDB implementation (dev fallback)
├── tests/
│   ├── test_mcp_tools.py        # MCP tools + permission filtering
│   ├── test_sql_validator.py    # Guardrail rules
│   └── test_retriever.py        # Retrieval quality
└── evals/
    ├── golden_set.yaml          # 30 data query tasks (Golden Set)
    └── eval_runner.py           # Evaluation runner
```

## Architecture Overview

```
Business caller (Web/BI/IM bot)
      │  HTTP/SSE (/query, /query/stream)
      ▼
┌───────────────────────────────────────────────────────────────┐
│ Agent Service (FastAPI + Docker)                              │
│                                                               │
│   LangGraph Agent (ReAct + Checkpoint, explicit State mgmt)   │
│   Parse → Metadata query → Case retrieval → SQL generate → Validate → Execute │
│   (validate/critic fail → retry generate, max 2 attempts)     │
│                                                               │
│         │            │           │           │                │
│         ▼            ▼           ▼           ▼                │
│   Model routing   MCP client   RAG corpus   Guardrails       │
│   DeepSeek        (SSE)        Milvus      SQLGlot            │
│   +Ollama                   +BM25+RRF   +Critic               │
└──────────────────────┬────────────────────────────────────────┘
                       │ SSE
                       ▼
┌───────────────────────────────────────────────────────────────┐
│ MCP Server (Python MCP SDK + SSE service)                     │
│ 4 metadata tools + RBAC table-level permission filtering      │
│ (unauthorized tables invisible at retrieval layer)            │
│ Metadata warehouse: tables/columns/lineage/metrics YAML       │
│───────────────────────────────────────────────────────────────│
│ Execution engine: QueryExecutor abstraction                   │
│   StarRocks (production) / DuckDB (dev fallback)              │
│───────────────────────────────────────────────────────────────│
│ Observability: Langfuse (full-link Trace / Token cost / feedback) │
└───────────────────────────────────────────────────────────────┘
```

## Core Pipeline

```
Natural language request (with user role)
  → Parse → MCP query metadata (with permission filtering)
    → RAG case retrieval → SQL generation → Rule validation → Critic review
      → StarRocks execution → Result + explanation → Langfuse Trace → Feedback loop
```

## Evaluation (Golden Set)

| Item | Details |
|------|---------|
| Dataset | 30 data query tasks (`evals/golden_set.yaml`) |
| Difficulty split | Simple aggregation 12 / Multi-table JOIN 8 / Metric definition 6 / Complex nesting 4 (~40%/27%/20%/13%) |
| Pass criteria | Execution success + all expected tables present + expected SQL keywords present |
| Primary metric | Element accuracy (see `evals/report.yaml`) |
| Secondary metric | Execution success rate (see `evals/report.yaml`) |
| Run command | `python scripts/run_evals.py` |
| Report | `evals/report.yaml` (overview / by difficulty / failure classification) |

> Run `python scripts/run_evals.py` to generate `evals/report.yaml` with overview, difficulty breakdown, and failure reason classification.

## Security Notes (Read Before Production Deployment)

Service interfaces and credential defaults in this project are **for local demo only**:

- `/query` and MCP Server have **no authentication by default** (MCP trusts the `x-user` header for role resolution). Production deployments must integrate authentication and authorization.
- StarRocks defaults to `root` with empty password; MinIO defaults to `minioadmin/minioadmin` (see `deploy/infra-compose.yml`). **These must be changed in production.**
- All secrets (`DEEPSEEK_API_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`) are injected via environment variables. **Do not write them into `config.yaml` or commit to the repository.**

## Testing

```bash
# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_executor.py

# Run a single test function
pytest tests/test_executor.py::test_duckdb_execute

# With coverage (optional, requires coverage)
pytest tests/ --cov=dataagent --cov-report=term-missing
```

## FAQ

| Issue | Cause | Solution |
|-------|-------|----------|
| BGE model download fails | HuggingFace is blocked in China | `export HF_ENDPOINT=https://hf-mirror.com` then retry |
| Milvus container starts but port is unreachable | etcd/minio initialization not complete | `docker compose -f deploy/infra-compose.yml logs -f` and wait (typically 30-60s) |
| `docker compose up` fails due to disk space | Milvus + etcd + minio require significant disk | Reserve at least 5GB; or use Milvus Lite (only `milvus-lite` package) |
| StarRocks allin1 OOM | allin1 image requires ~4GB+ memory | Use DuckDB fallback: `DW_EXECUTOR=duckdb python scripts/init_warehouse.py` |
| `ModuleNotFoundError` starting MCP Server | Virtual environment not activated | `source .venv/bin/activate` then retry |
| `pip install` fails with `setuptools>=82` + `pymilvus` conflict | pymilvus internally imports pkg_resources (removed in setuptools>=82) | Installation requirement already includes `setuptools<82` constraint; if still failing, `pip install 'setuptools<82'` |
| Evaluation accuracy far below expected | Evaluation depends on LLM quality + DeepSeek API Key | Ensure `DEEPSEEK_API_KEY` is correctly set; can switch to Ollama local model (modify `routing` in `config.yaml`) |
