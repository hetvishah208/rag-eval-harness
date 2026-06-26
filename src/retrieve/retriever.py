"""
Retrieve top-k chunks for a query, with optional cross-encoder reranking.

Usage:
    python -m src.retrieve.retriever "how do I load a model in 4-bit?"
"""
import argparse
import yaml

import chromadb
from chromadb.utils import embedding_functions


class Retriever:
    def __init__(self, cfg):
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

    def retrieve(self, query, top_k=None):
        k = top_k or self.cfg["retrieval"]["top_k"]
        # if reranking, pull a wider candidate set first
        fetch_k = k if not self._reranker else max(k, self.cfg["retrieval"].get("rerank_fetch", 8))
        res = self.coll.query(query_texts=[query], n_results=fetch_k)
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        hits = [{"text": d, "source": m["source"]} for d, m in zip(docs, metas)]

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
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    r = Retriever(cfg)
    for i, h in enumerate(r.retrieve(args.query), 1):
        print(f"\n[{i}] ({h['source']})\n{h['text'][:300]}...")
