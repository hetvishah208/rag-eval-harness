"""
Draft gold Q&A pairs from sampled corpus chunks using a local Ollama model.
EVERY generated pair MUST be human-verified before use (see docs/GOLD_DATASET.md).

Writes a *draft* JSONL file with "verified": false on each row. Review each one,
fix the answer, confirm the source chunk supports it, set "verified": true, and
delete ambiguous/unanswerable questions. The interactive reviewer
(`python -m src.eval.review_gold`) makes that easy.

Usage:
    python -m src.eval.build_gold --n 70
    python -m src.eval.build_gold --n 70 --out data/eval/gold_draft.jsonl
"""
import argparse
import json
import random

import yaml
import chromadb
from chromadb.utils import embedding_functions

from src.utils import ollama_chat, get_logger

log = get_logger("build_gold")

MIN_CHUNK_CHARS = 200  # only draft from substantive chunks

DRAFT_PROMPT = """You are creating an evaluation question for a documentation QA system.
Based ONLY on the documentation chunk below, write:
1. A specific, factual question a developer would realistically ask.
2. A correct, concise reference answer grounded entirely in the chunk.

Avoid questions that are vague, opinion-based, or not answerable from the chunk alone.
Return STRICT JSON: {{"question": "...", "ground_truth": "..."}}

Chunk (from {source}):
{chunk}
"""


def sample_chunks(cfg: dict, n: int) -> list[tuple]:
    client = chromadb.PersistentClient(path=cfg["vector_store"]["persist_dir"])
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=cfg["embedding"]["model"]
    )
    coll = client.get_collection(cfg["vector_store"]["collection"], embedding_function=ef)
    data = coll.get(include=["documents", "metadatas"])
    items = list(zip(data["documents"], data["metadatas"]))
    items = [it for it in items if len(it[0]) > MIN_CHUNK_CHARS]
    if not items:
        log.error("No chunks longer than %d chars in collection '%s'. "
                  "Is the index built and the collection name correct?",
                  MIN_CHUNK_CHARS, cfg["vector_store"]["collection"])
        return []
    random.shuffle(items)
    return items[:n]


def draft(chunk: str, source: str, model: str):
    prompt = DRAFT_PROMPT.format(source=source, chunk=chunk[:2000])
    try:
        text = ollama_chat(model, prompt, options={"temperature": 0.2})
    except Exception as e:  # noqa: BLE001
        log.warning("Draft generation failed for %s: %s", source, e)
        return None
    text = text.strip().replace("```json", "").replace("```", "").strip()
    try:
        obj = json.loads(text)
        return {"question": obj["question"], "ground_truth": obj["ground_truth"]}
    except (json.JSONDecodeError, KeyError):
        return None


def main(n: int, out: str, config_path: str, model: str) -> None:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    chunks = sample_chunks(cfg, n)
    if not chunks:
        return

    rows = []
    for i, (chunk, meta) in enumerate(chunks, 1):
        d = draft(chunk, meta.get("source", "?"), model)
        if d:
            d.update({
                "source": meta.get("source", "?"),
                "source_chunk": chunk[:1200],  # kept for human verification only
                "verified": False,
            })
            rows.append(d)
        log.info("Drafted %d/%d", i, len(chunks))

    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    log.info("Wrote %d DRAFT pairs to %s", len(rows), out)
    log.info("NEXT: run `python -m src.eval.review_gold` to verify and build gold_qa.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=70, help="chunks to sample (over-generate, then cut)")
    ap.add_argument("--out", default="data/eval/gold_draft.jsonl")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--model", default="mistral:7b-instruct-q4_0")
    args = ap.parse_args()
    main(args.n, args.out, args.config, args.model)