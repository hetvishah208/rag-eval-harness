"""
Build every vector index required by config/experiments.yaml — in one command.

It deduplicates: configs that share an (embedding, chunk_size, collection) triple
are built once, not repeatedly. Chunk JSON is produced once per chunk size.

Usage:
    python -m src.ingest.build_all_indexes
    python -m src.ingest.build_all_indexes --force        # rebuild even if present
    python -m src.ingest.build_all_indexes --config config/experiments.yaml

Run this AFTER cloning the corpus into data/raw/ (see README Phase 1).
"""
import argparse
import copy
import os
import subprocess
import sys
import yaml

import chromadb

from src.ingest.build_index import build_index, load_config


def chunk_json_path(processed_dir, chunk_size):
    return os.path.join(processed_dir, f"chunks_{chunk_size}.json")


def ensure_chunks(chunk_sizes, raw_dir, processed_dir, force):
    """Run chunk_docs once per needed chunk size (skips if JSON already present)."""
    needed = []
    for cs in sorted(chunk_sizes):
        if force or not os.path.exists(chunk_json_path(processed_dir, cs)):
            needed.append(cs)
    if not needed:
        print(f"[all] chunk JSON already present for sizes {sorted(chunk_sizes)} — skipping chunking")
        return
    print(f"[all] chunking corpus at sizes: {needed}")
    cmd = [sys.executable, "-m", "src.ingest.chunk_docs",
           "--raw-dir", raw_dir, "--processed-dir", processed_dir,
           "--chunk-sizes", *map(str, needed)]
    subprocess.run(cmd, check=True)


def collection_exists(persist_dir, name):
    try:
        client = chromadb.PersistentClient(path=persist_dir)
        return name in [c.name for c in client.list_collections()]
    except Exception:
        return False


def main(config_path, force):
    exp = yaml.safe_load(open(config_path))
    base_cfg = load_config("config/default.yaml")
    defaults = exp["defaults"]
    raw_dir = defaults.get("corpus_path", "data/raw/").rstrip("/")
    processed_dir = defaults.get("processed_path", "data/processed/").rstrip("/")
    persist_dir = base_cfg["vector_store"]["persist_dir"]

    # 1. Determine unique index builds + needed chunk sizes
    unique = {}   # (embedding, chunk_size, collection) -> [config keys]
    for k, c in exp["configs"].items():
        triple = (c["embedding_model"], c["chunk_size"], c["collection_name"])
        unique.setdefault(triple, []).append(k)
    chunk_sizes = {c["chunk_size"] for c in exp["configs"].values()}

    print(f"[all] {len(exp['configs'])} configs -> {len(unique)} unique index build(s)")

    # 2. Produce chunk JSON once per size
    ensure_chunks(chunk_sizes, raw_dir, processed_dir, force)

    # 3. Build each unique collection
    built, skipped = 0, 0
    for (embedding, chunk_size, collection), keys in unique.items():
        if not force and collection_exists(persist_dir, collection):
            print(f"[all] '{collection}' already exists — skipping (used by {keys}). Use --force to rebuild.")
            skipped += 1
            continue
        print(f"\n[all] building '{collection}' (emb={embedding}, chunk={chunk_size}) for configs {keys}")
        cfg = copy.deepcopy(base_cfg)
        cfg["embedding"]["model"] = embedding
        cfg["vector_store"]["collection"] = collection
        # build_index reads chunks_<size>.json and embeds into the named collection
        build_index(chunk_size, cfg, collection_suffix="")
        built += 1

    print(f"\n[all] done. built={built}, skipped={skipped}, total unique={len(unique)}")
    print("[all] next: python -m src.eval.run_eval --config config/experiments.yaml")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/experiments.yaml")
    ap.add_argument("--force", action="store_true", help="rebuild chunks + collections even if present")
    args = ap.parse_args()
    main(args.config, args.force)
