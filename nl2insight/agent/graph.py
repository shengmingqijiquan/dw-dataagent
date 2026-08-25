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

from nl2insight.agent.prompts import GENERATE_PROMPT, PARSE_PROMPT, CRITIC_PROMPT
from nl2insight.agent.state import DWState
from nl2insight.config import Settings, load_config
from nl2insight.executor.base import QueryError
from nl2insight.executor.duckdb_executor import DuckDBExecutor
from nl2insight.executor.starrocks_executor import StarRocksExecutor
from nl2insight.guardrails.sql_validator import validate_sql
from nl2insight.llm.router import LLMRouter
from nl2insight.rag.retriever import HybridRetriever
from nl2insight.warehouse.schema import TABLES

MAX_GENERATE_ATTEMPTS = 2
MAX_EXECUTE_ATTEMPTS = 2

_mcp_tools_cache: dict[str, list] = {}


async def _load_mcp_tools(settings: Settings, role: str):
    """通过 langchain-mcp-adapters 从 SSE MCP Server 加载工具（按角色分键缓存）。

    SSE 连接头携带 x-user=role，MCP Server 端以此解析角色做权限过滤；
    不同角色走不同连接/会话，故缓存按角色分键，避免角色上下文串用。
    """
    global _mcp_tools_cache
    if role in _mcp_tools_cache:
        return _mcp_tools_cache[role]
    from langchain_mcp_adapters.client import MultiServerMCPClient
    client = MultiServerMCPClient({
        "metadata": {
            "transport": "sse",
            "url": settings.mcp_server_url,
            "headers": {"x-user": role},
        }
    })
    _mcp_tools_cache[role] = await client.get_tools()
    return _mcp_tools_cache[role]


def build_agent(settings: Settings):
    router = LLMRouter(settings)
    retriever = HybridRetriever(
        settings.milvus.host, settings.milvus.port, settings.milvus.collection,
        uri=settings.milvus.uri)

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
        """工具调用循环：Agent 自主决定查哪些元数据/案例。

        MCP 不可达/工具调用异常时降级进状态机——generate 拿到显式
        失败上下文继续，而不是未处理异常逃逸出图。
        """
        try:
            mcp_tools = await _load_mcp_tools(settings, state["role"])
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
        except Exception as e:
            return {"context": f"上下文收集失败: {e}"}
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
              settings: Settings | None = None,
              agent=None) -> dict:
    """同步便利入口。agent 可复用：多次调用传同一实例避免每 call 重建全图（M4）。"""
    import asyncio
    settings = settings or load_config()
    if agent is None:
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
