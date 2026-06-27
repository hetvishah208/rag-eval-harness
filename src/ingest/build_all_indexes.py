"""
Build every vector index required by config/experiments.yaml in one command.

Deduplicates: configs sharing an (embedding, chunk_size, collection) triple are
built once. Chunk JSON is produced once per chunk size. Existing collections are
skipped unless --force is passed.

Usage:
    python -m src.ingest.build_all_indexes
    python -m src.ingest.build_all_indexes --force
    python -m src.ingest.build_all_indexes --config config/experiments.yaml

Run this AFTER cloning the corpus into data/raw/ (see README).
"""
import argparse
import copy
import os
import subprocess
import sys

import yaml
import chromadb

from src.ingest.build_index import build_index, load_config
from src.utils import get_logger

log = get_logger("build_all")


def chunk_json_path(processed_dir: str, chunk_size: int) -> str:
    return os.path.join(processed_dir, f"chunks_{chunk_size}.json")


def ensure_chunks(chunk_sizes, raw_dir, processed_dir, force) -> None:
    """Run chunk_docs once per needed chunk size (skips if JSON already present)."""
    needed = [cs for cs in sorted(chunk_sizes)
              if force or not os.path.exists(chunk_json_path(processed_dir, cs))]
    if not needed:
        log.info("Chunk JSON already present for sizes %s — skipping chunking", sorted(chunk_sizes))
        return
    log.info("Chunking corpus at sizes: %s", needed)
    cmd = [sys.executable, "-m", "src.ingest.chunk_docs",
           "--raw-dir", raw_dir, "--processed-dir", processed_dir,
           "--chunk-sizes", *map(str, needed)]
    subprocess.run(cmd, check=True)


def collection_exists(persist_dir: str, name: str) -> bool:
    client = chromadb.PersistentClient(path=persist_dir)
    return name in [c.name for c in client.list_collections()]


def main(config_path: str, force: bool) -> None:
    exp = yaml.safe_load(open(config_path, encoding="utf-8"))
    base_cfg = load_config("config/default.yaml")
    defaults = exp["defaults"]
    raw_dir = defaults.get("corpus_path", "data/raw/").rstrip("/")
    processed_dir = defaults.get("processed_path", "data/processed/").rstrip("/")
    persist_dir = base_cfg["vector_store"]["persist_dir"]

    # 1. Determine unique index builds + needed chunk sizes
    unique = {}  # (embedding, chunk_size, collection) -> [config keys]
    for k, c in exp["configs"].items():
        triple = (c["embedding_model"], c["chunk_size"], c["collection_name"])
        unique.setdefault(triple, []).append(k)
    chunk_sizes = {c["chunk_size"] for c in exp["configs"].values()}

    log.info("%d configs -> %d unique index build(s)", len(exp["configs"]), len(unique))

    # 2. Produce chunk JSON once per size
    ensure_chunks(chunk_sizes, raw_dir, processed_dir, force)

    # 3. Build each unique collection
    built, skipped = 0, 0
    for (embedding, chunk_size, collection), keys in unique.items():
        if not force and collection_exists(persist_dir, collection):
            log.info("'%s' already exists — skipping (used by %s). Use --force to rebuild.",
                     collection, keys)
            skipped += 1
            continue
        log.info("Building '%s' (emb=%s, chunk=%d) for configs %s",
                 collection, embedding, chunk_size, keys)
        cfg = copy.deepcopy(base_cfg)
        cfg["embedding"]["model"] = embedding
        cfg["vector_store"]["collection"] = collection
        build_index(chunk_size, cfg, collection_suffix="")
        built += 1

    log.info("Done. built=%d, skipped=%d, total unique=%d", built, skipped, len(unique))
    log.info("Next: python -m src.eval.run_eval --config config/experiments.yaml")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/experiments.yaml")
    ap.add_argument("--force", action="store_true", help="rebuild chunks + collections even if present")
    args = ap.parse_args()
    main(args.config, args.force)