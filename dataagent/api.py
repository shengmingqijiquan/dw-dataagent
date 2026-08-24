"""Agent Service：FastAPI 服务化。

生产对标：业务系统（BI/IM 机器人）通过 REST/SSE 调用 Agent，
Agent 作为独立服务部署、独立扩缩容。
"""
import asyncio
import json
import logging
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from dataagent.agent.graph import build_agent, _invoke
from dataagent.config import load_config
from dataagent.observability import get_langfuse_handler

app = FastAPI(title="dw-dataagent", version="1.0.0")
settings = load_config()
_agent = build_agent(settings)
logger = logging.getLogger(__name__)


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
    # 图带 MemorySaver checkpointer，thread_id 必传；每请求独立 checkpoint
    config = {"configurable": {"thread_id": uuid4().hex}}
    if langfuse:
        config["callbacks"] = [langfuse]
    try:
        state = await _agent.ainvoke(
            {"question": req.question, "role": req.role,
             "generate_attempts": 0, "execute_attempts": 0},
            config=config)
    except Exception:
        logger.exception("Agent 执行失败")
        raise HTTPException(status_code=500,
                            detail="Agent 执行失败，请查看服务端日志")
    return QueryResponse(
        sql=state.get("sql", ""),
        result=state.get("result", ""),
        explanation=f"SQL: {state.get('sql', '')}\n结果: {state.get('result', '')}")


@app.post("/query/stream")
async def query_stream(req: QueryRequest):
    """SSE 流式返回（事件: done 携带最终 JSON）。"""
    langfuse = get_langfuse_handler()
    async def event_stream():
        # 与 /query 一致挂载 langfuse handler，SSE 路径同样全程 Trace
        config = {"configurable": {"thread_id": uuid4().hex}}
        if langfuse:
            config["callbacks"] = [langfuse]
        state = await _agent.ainvoke(
            {"question": req.question, "role": req.role,
             "generate_attempts": 0, "execute_attempts": 0},
            config=config)
        payload = json.dumps({
            "sql": state.get("sql", ""),
            "result": state.get("result", ""),
        }, ensure_ascii=False)
        yield f"data: {payload}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
