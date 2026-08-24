"""索引构建单元测试（不依赖 Milvus）。"""
from dataagent.rag.indexer import build_documents
from dataagent.rag.cases import load_cases


def test_build_documents_count():
    docs = build_documents(load_cases())
    assert len(docs) == 50


def test_document_has_id_and_text():
    for doc in build_documents(load_cases()):
        assert doc["id"].startswith("c")
        assert doc["text"]
        assert doc["metadata"]["domain"]


def test_text_contains_question_and_sql():
    docs = build_documents(load_cases())
    assert "GMV" in docs[0]["text"]
    assert "SELECT" in docs[0]["text"]
