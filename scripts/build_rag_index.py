"""构建 RAG 案例索引：python scripts/build_rag_index.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nl2insight.config import load_config
from nl2insight.rag.cases import load_cases
from nl2insight.rag.indexer import MilvusIndexer


def main():
    settings = load_config()
    cases = load_cases(settings.cases_path)
    indexer = MilvusIndexer(
        settings.milvus.host, settings.milvus.port, settings.milvus.collection,
        uri=settings.milvus.uri)
    n = indexer.index(cases)
    print(f"[rag] 已入库 {n} 条案例到 Milvus collection={settings.milvus.collection}")


if __name__ == "__main__":
    main()
