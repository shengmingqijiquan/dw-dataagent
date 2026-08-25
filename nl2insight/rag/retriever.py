"""混合检索：Milvus 向量 ANN + BM25 关键词 + RRF 融合。

生产对标：向量检索找"语义相似案例"，BM25 找"专有名词精确匹配"
（表名/指标名），RRF 融合两者互补——这是企业 RAG 检索层的标准形态。
"""
from collections import defaultdict

from pymilvus import Collection, connections

from nl2insight.rag.cases import Case, load_cases
from nl2insight.rag.indexer import get_embedder


def rrf_merge(ranked_lists: list[list[str]], k: int = 60) -> list[str]:
    """Reciprocal Rank Fusion：score(d) = Σ 1/(k + rank_i(d))"""
    scores: dict[str, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] += 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda d: scores[d], reverse=True)


class HybridRetriever:
    def __init__(self, host: str, port: int, collection: str,
                 cases: list[Case] | None = None, uri: str = ""):
        self.host, self.port, self.collection_name = host, port, collection
        self.uri = uri  # 非空 → Milvus Lite（本地文件）/云 URI；否则 standalone host/port
        self.cases = cases or load_cases()
        self._bm25 = None
        self._case_ids = [c.id for c in self.cases]
        self._corpus = [c.text() for c in self.cases]

    def _get_bm25(self):
        if self._bm25 is None:
            from rank_bm25 import BM25Okapi
            import jieba  # 中文分词
            tokenized = [list(jieba.cut(t)) for t in self._corpus]
            self._bm25 = BM25Okapi(tokenized)
        return self._bm25

    def _vector_search(self, question: str, top_k: int) -> list[str]:
        if self.uri:
            connections.connect(alias="default", uri=self.uri)
        else:
            connections.connect(alias="default", host=self.host, port=self.port)
        collection = Collection(self.collection_name)
        collection.load()  # 幂等防御：服务重启后确保 collection 已加载
        query_vec = get_embedder().encode(
            [question], normalize_embeddings=True).tolist()
        hits = collection.search(
            query_vec, "embedding",
            {"metric_type": "IP", "params": {"ef": 128}},
            limit=top_k)
        return [h.id for h in hits[0]]

    def _bm25_search(self, question: str, top_k: int) -> list[str]:
        import jieba
        scores = self._get_bm25().get_scores(list(jieba.cut(question)))
        ranked = sorted(zip(self._case_ids, scores),
                        key=lambda x: x[1], reverse=True)[:top_k]
        return [cid for cid, _ in ranked]

    def search(self, question: str, top_k: int = 5) -> list[dict]:
        """返回 [{id, question, sql, domain, score}]，score 为 RRF 分。"""
        vec_rank = self._vector_search(question, top_k * 4)
        bm25_rank = self._bm25_search(question, top_k * 4)
        merged = rrf_merge([vec_rank, bm25_rank])[:top_k]
        by_id = {c.id: c for c in self.cases}
        return [
            {"id": cid, "question": by_id[cid].question,
             "sql": by_id[cid].sql, "domain": by_id[cid].domain}
            for cid in merged if cid in by_id
        ]
