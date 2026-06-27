"""
Retrieve top-k chunks for a query, with optional cross-encoder reranking.

Usage:
    python -m src.retrieve.retriever "how do I load a model in 4-bit?"
"""
import argparse

import yaml
import chromadb
from chromadb.utils import embedding_functions

from src.utils import get_logger

log = get_logger("retriever")


class Retriever:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=cfg["embedding"]["model"]
        )
        client = chromadb.PersistentClient(path=cfg["vector_store"]["persist_dir"])
        self.coll = client.get_collection(
            name=cfg["vector_store"]["collection"], embedding_function=ef
        )
        self._reranker = None
        if cfg["retrieval"].get("reranking"):
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(cfg["retrieval"]["reranker_model"])

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        k = top_k or self.cfg["retrieval"]["top_k"]
        # When reranking, pull a wider candidate set first, then re-rank down.
        fetch_k = k if not self._reranker else max(k, self.cfg["retrieval"].get("rerank_fetch", 8))

        res = self.coll.query(query_texts=[query], n_results=fetch_k)
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        if not docs:
            log.warning("No chunks retrieved for query: %r", query[:80])
            return []

        hits = [{"text": d, "source": m.get("source", "unknown")}
                for d, m in zip(docs, metas)]

        if self._reranker:
            pairs = [[query, h["text"]] for h in hits]
            scores = self._reranker.predict(pairs)
            ranked = sorted(zip(hits, scores), key=lambda x: x[1], reverse=True)
            n = self.cfg["retrieval"].get("rerank_top_n", k)
            hits = [h for h, _ in ranked[:n]]
        return hits


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--config", default="config/default.yaml")
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    r = Retriever(cfg)
    hits = r.retrieve(args.query)
    if not hits:
        print("No results.")
    for i, h in enumerate(hits, 1):
        print(f"\n[{i}] ({h['source']})\n{h['text'][:300]}...")