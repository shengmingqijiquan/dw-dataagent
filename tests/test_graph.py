"""LangGraph 图结构测试（不调真实 LLM）。"""
import asyncio

from dataagent.agent import graph as graph_module
from dataagent.agent.graph import build_agent
from dataagent.config import load_config


def test_graph_has_all_nodes():
    agent = build_agent(load_config())
    nodes = set(agent.get_graph().nodes.keys())
    assert {"parse", "collect", "generate", "validate", "critic",
            "execute", "fail"} <= nodes


def test_load_mcp_tools_sends_role_header_and_caches_per_role(monkeypatch):
    # MCP 连接头带 x-user=role，且缓存按角色分键（不同角色各占一条连接）
    captured = []

    class FakeClient:
        def __init__(self, config):
            captured.append(config)

        async def get_tools(self):
            return ["fake-tool"]

    monkeypatch.setattr(
        "langchain_mcp_adapters.client.MultiServerMCPClient", FakeClient)
    settings = load_config()
    try:
        asyncio.run(graph_module._load_mcp_tools(settings, "finance_analyst"))
        asyncio.run(graph_module._load_mcp_tools(settings, "ops_analyst"))
        asyncio.run(graph_module._load_mcp_tools(settings, "finance_analyst"))
    finally:
        graph_module._mcp_tools_cache.clear()
    # finance_analyst 被缓存命中不再新建连接，共 2 条连接、各自带角色头
    assert len(captured) == 2
    assert captured[0]["metadata"]["headers"] == {"x-user": "finance_analyst"}
    assert captured[1]["metadata"]["headers"] == {"x-user": "ops_analyst"}
