"""MCP SSE 服务集成测试（需手动启动服务：uvicorn dataagent.mcp_server.server:app --port 8001）。"""
import json
import pytest
from langchain_mcp_adapters.client import MultiServerMCPClient

pytestmark = pytest.mark.integration


@pytest.fixture
def mcp_client():
    client = MultiServerMCPClient({
        "metadata": {
            "transport": "sse",
            "url": "http://localhost:8001/mcp/sse",
            "headers": {"x-user": "finance_wang"},
        }
    })
    return client


@pytest.mark.asyncio
async def test_tools_discovered(mcp_client):
    tools = await mcp_client.get_tools()
    names = {t.name for t in tools}
    assert names == {"query_table_list", "query_table_schema",
                     "query_lineage", "query_metric_definition"}


@pytest.mark.asyncio
async def test_call_tool_with_permission(mcp_client):
    tools = await mcp_client.get_tools()
    schema_tool = next(t for t in tools if t.name == "query_table_schema")
    result = await schema_tool.ainvoke({"table_name": "dws_payment_summary_di"})
    # langchain-mcp-adapters>=0.3 的 ainvoke 返回 content block 列表，需先归一为文本
    result_text = result[0]["text"] if isinstance(result, list) else str(result)
    assert "pay_amount" in result_text
