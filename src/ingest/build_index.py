"""
Embed pre-chunked docs into ChromaDB.

Pipeline position:
    chunk_docs.py  (parse + clean + chunk -> data/processed/chunks_<size>.json)
        -> build_index.py  (embed those chunks -> ChromaDB collection)

Usage:
    # First produce chunks:
    python -m src.ingest.chunk_docs --chunk-sizes 512
    # Then index a given chunk size with a given embedding model:
    python -m src.ingest.build_index --chunk-size 512 --config config/default.yaml
"""
import argparse
import json
import os
import yaml

import chromadb
from chromadb.utils import embedding_functions


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_index(chunk_size, cfg, collection_suffix=""):
    chunks_path = os.path.join(
        cfg["corpus"]["processed_dir"], f"chunks_{chunk_size}.json"
    )
    if not os.path.exists(chunks_path):
        raise FileNotFoundError(
            f"{chunks_path} not found. Run chunk_docs.py --chunk-sizes {chunk_size} first."
        )
    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"[index] loaded {len(chunks)} chunks from {chunks_path}")

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=cfg["embedding"]["model"]
    )
    client = chromadb.PersistentClient(path=cfg["vector_store"]["persist_dir"])
    name = cfg["vector_store"]["collection"] + collection_suffix
    try:
        client.delete_collection(name)
    except Exception:
        pass
    coll = client.create_collection(name=name, embedding_function=ef)

    B = 256
    for i in range(0, len(chunks), B):
        batch = chunks[i:i + B]
        coll.add(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[
                {"source": c["metadata"]["source_file"],
                 "section": c["metadata"].get("section_header") or ""}
                for c in batch
            ],
        )
        print(f"[index] embedded {min(i + B, len(chunks))}/{len(chunks)}")
    print(f"[index] done. collection '{name}' has {coll.count()} chunks")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-size", type=int, default=512)
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--collection-suffix", default="")
    args = ap.parse_args()
    build_index(args.chunk_size, load_config(args.config), args.collection_suffix)
