# 简历条目 + 面试话术

## 简历 · 个人项目板块

**DataAgent 取数智能体（LangGraph + MCP + Milvus + StarRocks）** · 个人项目
- 从零实现生产级取数 Agent：LangGraph 状态图编排（解析→元数据查询→案例检索→生成→校验→执行），Checkpoint 支持同进程内断点恢复
- 自研 MCP Server（Python SDK，SSE 服务化部署），4 个元数据工具 + RBAC 表级权限过滤（无权限表在检索层不可见，源头阻断越权）
- RAG 案例库：Milvus HNSW + BGE-large-zh Embedding + BM25 关键词检索 + RRF 融合，50 条历史 SQL 案例检索
- 双层护栏：SQLGlot 规则引擎（语法/只读/表存在/权限/分区 100% 拦截非法 SQL）+ LLM Critic 语义审查
- 模型路由层（Tiered Model Stack）：DeepSeek API + 本地 Ollama Qwen3，生产私有化部署零改动切换
- Golden Set 30 条评测驱动迭代，取数准确率 XX%（执行成功率 XX%）〔数字在 Task 17 Step 4 从 evals/report.yaml 实测回填〕
- 可观测：Langfuse 全链路 Trace + Token 成本统计
技术栈：Python / LangGraph / MCP / Milvus / BGE / StarRocks / DuckDB / SQLGlot / FastAPI / Langfuse

## 面试话术（对应各组件）

### Agent 框架
> "我选 LangGraph 而不是裸 LangChain Agent，因为它有显式状态管理——State 定义每一步的输入输出，Checkpoint 让长任务可以在同进程内暂停恢复（当前用 MemorySaver 进程内实现，生产可换持久化存储）。这意味着：Agent 在图内重试循环（校验失败回 generate）时每一步状态都能保存，不用从头再来。我的图有 7 个节点：parse/collect/generate/validate/critic/execute/fail，校验失败回到 generate 重试（上限 2 次），这是状态机护栏的设计。"

### MCP
> "MCP Server 我做了 SSE 服务化部署而不是 stdio，因为生产环境 Agent 是独立服务，MCP 需要支持多客户端远程访问。工具层我实现了权限过滤——不是 SQL 生成后再拦，而是元数据检索阶段无权限的表对 Agent 完全不可见，从源头消除越权。"

### RAG
> "案例检索我用了混合检索：Milvus 向量检索找语义相似的案例，BM25 找表名、指标名这类专有名词的精确匹配，RRF 融合两者。纯向量检索在数仓场景会漏掉专有名词，纯关键词又理解不了语义，融合后 Top-5 命中率明显提升。"

### 护栏
> "校验分两层：SQLGlot 规则引擎做确定性检查（语法、只读、表存在、权限、分区），100% 拦截非法 SQL；LLM Critic 做语义审查（口径一致性、JOIN 合理性）。规则能兜底的绝不用 LLM——LLM 审查有幻觉风险，只能做补充。"

### 评测
> "30 条 Golden Set 按难度分层：简单聚合 12 条、多表 JOIN 8 条、口径 6 条、复杂嵌套 4 条（约 40%/27%/20%/13%）。评测主指标是要素准确率（执行成功 + 预期表 + 关键字），副指标是执行成功率。每次 Prompt 迭代都跑全量回归，失败原因分类（校验失败/执行失败/要素缺失）驱动迭代方向。"

### 模型路由
> "模型路由层解决两个问题：成本分层（需求解析用轻量模型，SQL 生成用主力模型）和合规隔离（生产环境数据不出域，必须私有化部署 Qwen/DeepSeek，路由层切换零代码改动）。这对应大厂模型网关的设计。"
