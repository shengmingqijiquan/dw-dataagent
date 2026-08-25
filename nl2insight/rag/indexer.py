"""RAG 索引构建：案例 → BGE Embedding → Milvus 入库。

分块策略：案例（需求+SQL 配对）是语义完整单元，整条入库不切碎。
生产对标：知识库万级时按 domain 字段做 Partition Key，HNSW 参数
（M=16, efConstruction=200）平衡召回与写入性能。
"""
from pymilvus import (
    Collection, CollectionSchema, DataType, FieldSchema, connections, utility,
)
from sentence_transformers import SentenceTransformer

from nl2insight.rag.cases import Case

EMBEDDING_MODEL = "BAAI/bge-large-zh-v1.5"
DIM = 1024
_embedder = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def build_documents(cases: list[Case]) -> list[dict]:
    return [
        {
            "id": c.id,
            "text": c.text(),
            "metadata": {
                "domain": c.domain,
                "tables": ",".join(c.tables),
                "metrics": ",".join(c.metrics),
            },
        }
        for c in cases
    ]


class MilvusIndexer:
    def __init__(self, host: str, port: int, collection: str, uri: str = ""):
        self.host, self.port = host, port
        self.collection_name = collection
        self.uri = uri

    def connect(self):
        # uri 非空时走 Milvus Lite（本地文件）或云 URI；否则 standalone host/port
        if self.uri:
            connections.connect(alias="default", uri=self.uri)
        else:
            connections.connect(alias="default", host=self.host, port=self.port)

    def create_collection(self, drop_if_exists: bool = False):
        if utility.has_collection(self.collection_name):
            if drop_if_exists:
                utility.drop_collection(self.collection_name)
            else:
                return Collection(self.collection_name)
        fields = [
            FieldSchema("id", DataType.VARCHAR, is_primary=True, max_length=32),
            FieldSchema("text", DataType.VARCHAR, max_length=4096),
            FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=DIM),
            FieldSchema("domain", DataType.VARCHAR, max_length=16),
            FieldSchema("tables", DataType.VARCHAR, max_length=512),
            FieldSchema("metrics", DataType.VARCHAR, max_length=256),
        ]
        schema = CollectionSchema(fields, description="数仓取数 SQL 案例库")
        collection = Collection(self.collection_name, schema)
        if self.uri:
            # Milvus Lite 本地模式不支持 HNSW，仅支持 FLAT/IVF_FLAT/AUTOINDEX
            index_params = {"metric_type": "IP", "index_type": "AUTOINDEX"}
        else:
            index_params = {
                "metric_type": "IP",  # 内积（embedding 已 normalize → 等价余弦）
                "index_type": "HNSW",
                "params": {"M": 16, "efConstruction": 200},
            }
        collection.create_index("embedding", index_params)
        return collection

    def index(self, cases: list[Case]) -> int:
        self.connect()
        collection = self.create_collection(drop_if_exists=True)
        docs = build_documents(cases)
        embedder = get_embedder()
        embeddings = embedder.encode(
            [d["text"] for d in docs],
            normalize_embeddings=True, show_progress_bar=True)
        collection.insert([
            [d["id"] for d in docs],
            [d["text"] for d in docs],
            embeddings.tolist(),
            [d["metadata"]["domain"] for d in docs],
            [d["metadata"]["tables"] for d in docs],
            [d["metadata"]["metrics"] for d in docs],
        ])
        collection.flush()
        collection.load()
        return len(docs)
