"""RRF 融合纯函数测试。"""
from dataagent.rag.retriever import rrf_merge


def test_rrf_merges_two_lists():
    merged = rrf_merge([["a", "b", "c"], ["b", "c", "d"]], k=60)
    # b 在两个列表中都排第 1-2 → 融合后应排第一
    assert merged[0] == "b"
    assert set(merged) == {"a", "b", "c", "d"}


def test_rrf_handles_empty_list():
    assert rrf_merge([[], []], k=60) == []


def test_rrf_stable_for_single_list():
    merged = rrf_merge([["x", "y", "z"]], k=60)
    assert merged == ["x", "y", "z"]
