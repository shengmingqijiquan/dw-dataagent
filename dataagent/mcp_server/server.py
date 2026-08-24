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
from starlette.responses import Response
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
    return Response()  # 会话结束需返回响应对象，否则 Starlette 会尝试调用 None


app = Starlette(
    routes=[
        Route("/health", endpoint=lambda req: starlette.responses.JSONResponse(
            {"status": "ok"})),
        Route("/mcp/sse", endpoint=handle_sse),
        Mount("/messages/", app=sse.handle_post_message),
    ]
)
