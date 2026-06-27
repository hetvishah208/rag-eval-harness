"""
Embed pre-chunked docs into a ChromaDB collection.

Pipeline position:
    chunk_docs.py   (parse + clean + chunk -> data/processed/chunks_<size>.json)
        -> build_index.py   (embed those chunks -> ChromaDB collection)

Usage:
    python -m src.ingest.chunk_docs --chunk-sizes 512
    python -m src.ingest.build_index --chunk-size 512 --config config/default.yaml
"""
import argparse
import json
import os

import yaml
import chromadb
from chromadb.utils import embedding_functions

from src.utils import get_logger

log = get_logger("build_index")

EMBED_BATCH_SIZE = 256


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_index(chunk_size: int, cfg: dict, collection_suffix: str = "") -> None:
    chunks_path = os.path.join(cfg["corpus"]["processed_dir"], f"chunks_{chunk_size}.json")
    if not os.path.exists(chunks_path):
        raise FileNotFoundError(
            f"{chunks_path} not found. Run "
            f"`python -m src.ingest.chunk_docs --chunk-sizes {chunk_size}` first."
        )
    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)
    log.info("Loaded %d chunks from %s", len(chunks), chunks_path)

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=cfg["embedding"]["model"]
    )
    client = chromadb.PersistentClient(path=cfg["vector_store"]["persist_dir"])
    name = cfg["vector_store"]["collection"] + collection_suffix

    # Drop an existing collection of the same name so rebuilds are deterministic.
    # Only swallow the "does not exist" case; let anything else surface.
    try:
        client.delete_collection(name)
        log.info("Dropped existing collection '%s' before rebuild", name)
    except Exception as e:  # noqa: BLE001
        if "does not exist" not in str(e).lower():
            raise

    coll = client.create_collection(name=name, embedding_function=ef)

    for i in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[i:i + EMBED_BATCH_SIZE]
        coll.add(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[
                {"source": c["metadata"]["source_file"],
                 "section": c["metadata"].get("section_header") or ""}
                for c in batch
            ],
        )
        log.info("Embedded %d/%d", min(i + EMBED_BATCH_SIZE, len(chunks)), len(chunks))

    log.info("Done. Collection '%s' has %d chunks", name, coll.count())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-size", type=int, default=512)
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--collection-suffix", default="")
    args = ap.parse_args()
    build_index(args.chunk_size, load_config(args.config), args.collection_suffix)