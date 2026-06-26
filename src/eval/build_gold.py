"""
Draft gold Q&A pairs from sampled corpus chunks using a local Ollama model.
EVERY generated pair MUST be human-verified before use (see docs/GOLD_DATASET.md).

This writes a *draft* file with a "verified": false flag on each row. Your job:
review each one, fix the answer, confirm the source chunk supports it, set
"verified": true, and delete any ambiguous/unanswerable questions.

Usage:
    python -m src.eval.build_gold --n 70 --out data/gold/gold_draft.jsonl
"""
import argparse
import json
import random
import yaml
import ollama

import chromadb
from chromadb.utils import embedding_functions


DRAFT_PROMPT = """You are creating an evaluation question for a documentation QA system.
Based ONLY on the documentation chunk below, write:
1. A specific, factual question a developer would realistically ask.
2. A correct, concise reference answer grounded entirely in the chunk.

Avoid questions that are vague, opinion-based, or not answerable from the chunk alone.
Return STRICT JSON: {{"question": "...", "ground_truth": "..."}}

Chunk (from {source}):
{chunk}
"""


def sample_chunks(cfg, n):
    client = chromadb.PersistentClient(path=cfg["vector_store"]["persist_dir"])
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=cfg["embedding"]["model"]
    )
    coll = client.get_collection(cfg["vector_store"]["collection"], embedding_function=ef)
    data = coll.get(include=["documents", "metadatas"])
    items = list(zip(data["documents"], data["metadatas"]))
    # prefer longer chunks (more substantive) and de-dup sources for coverage
    items = [it for it in items if len(it[0]) > 200]
    random.shuffle(items)
    return items[:n]


def draft(chunk, source, model):
    prompt = DRAFT_PROMPT.format(source=source, chunk=chunk[:2000])
    resp = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2},
    )
    text = resp["message"]["content"].strip()
    # be forgiving about code fences
    text = text.replace("```json", "").replace("```", "").strip()
    try:
        obj = json.loads(text)
        return {"question": obj["question"], "ground_truth": obj["ground_truth"]}
    except Exception:
        return None


def main(n, out, config_path, model):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    chunks = sample_chunks(cfg, n)
    rows = []
    for i, (chunk, meta) in enumerate(chunks, 1):
        d = draft(chunk, meta.get("source", "?"), model)
        if d:
            d.update({
                "source": meta.get("source", "?"),
                "source_chunk": chunk[:1200],   # keep for your verification
                "verified": False,
            })
            rows.append(d)
        print(f"[gold] drafted {i}/{len(chunks)}")
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"\n[gold] wrote {len(rows)} DRAFT pairs to {out}")
    print("[gold] NEXT: human-verify every row, set verified=true, target ~50 keepers.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=70)   # over-generate; you'll cut to ~50
    ap.add_argument("--out", default="data/eval/gold_draft.jsonl")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--model", default="mistral:7b-instruct-q4_0")
    args = ap.parse_args()
    main(args.n, args.out, args.config, args.model)
