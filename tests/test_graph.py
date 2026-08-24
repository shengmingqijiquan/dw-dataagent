"""LangGraph 图结构测试（不调真实 LLM）。"""
from dataagent.agent.graph import build_agent
from dataagent.config import load_config


def test_graph_has_all_nodes():
    agent = build_agent(load_config())
    nodes = set(agent.get_graph().nodes.keys())
    assert {"parse", "collect", "generate", "validate", "critic",
            "execute", "fail"} <= nodes
