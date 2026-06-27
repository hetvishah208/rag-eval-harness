"""
Evaluation harness (Windows-safe, sequential, checkpointed).

WHY THIS DOESN'T USE RAGAS's EXECUTOR
-------------------------------------
RAGAS's async job executor hangs on some Windows setups (jobs sit at 0/N and never
dispatch to a local Ollama server). This harness bypasses it and computes the metrics
with direct, sequential Ollama calls — fully local, $0, and reliable on CPU-only
Windows.

A NOTE ON THE METRICS (important — read before quoting these as "RAGAS metrics")
-------------------------------------------------------------------------------
These are RAGAS-*inspired*, single-shot LLM-judge approximations, not RAGAS's exact
algorithms:
  - faithfulness        : judge scores whether the answer's claims are supported by
                          the retrieved context (close to RAGAS faithfulness).
  - answer_relevancy    : judge scores how directly the answer addresses the question.
                          (RAGAS does this by generating questions from the answer and
                          measuring similarity; this is a simpler direct judge score.)
  - context_relevance   : judge scores whether the retrieved context is relevant to the
                          question. NOTE: this is a relevance score, NOT RAGAS's
                          rank-aware "context precision". Named honestly to avoid
                          implying the rank-aware algorithm.
  - context_recall      : judge scores how much of the reference answer is supported by
                          the retrieved context (close to RAGAS context recall).

Every score is a float in [0, 1]. The judge is a local 7B model, so treat scores as
directional; see docs/GOLD_DATASET.md for the human judge-validation.

Resilience: results are checkpointed to results/eval_results.json after every config,
so a crash partway through a multi-hour run doesn't lose completed configs.

Usage:
    python -m src.eval.run_eval --config config/experiments.yaml
"""
import argparse
import copy
import json
import os
import re
import time

import yaml

from src.retrieve.retriever import Retriever
from src.generate.answer import answer as rag_answer
from src.utils import ollama_chat, get_logger

log = get_logger("run_eval")

# Public metric names (note: context_relevance, not context_precision — see module docstring)
METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_relevance", "context_recall"]


# ---------- judge prompts (each asks for a single 0..1 score) ----------

FAITHFULNESS_PROMPT = """You are evaluating whether an ANSWER is faithful to the provided CONTEXT.
A faithful answer makes only claims that are supported by the context.

CONTEXT:
{context}

ANSWER:
{answer}

How faithful is the answer to the context?
- 1.0 = every claim is supported by the context
- 0.0 = the answer contradicts or invents information not in the context

Reply with ONLY a single number between 0.0 and 1.0 and nothing else."""

ANSWER_RELEVANCY_PROMPT = """You are evaluating whether an ANSWER is relevant to a QUESTION.

QUESTION:
{question}

ANSWER:
{answer}

How directly does the answer address the question?
- 1.0 = fully answers the question
- 0.0 = does not address the question at all

Reply with ONLY a single number between 0.0 and 1.0 and nothing else."""

CONTEXT_RELEVANCE_PROMPT = """You are evaluating whether the retrieved CONTEXT is relevant to a QUESTION.

QUESTION:
{question}

CONTEXT:
{context}

How relevant is the context for answering the question?
- 1.0 = highly relevant and useful
- 0.0 = unrelated to the question

Reply with ONLY a single number between 0.0 and 1.0 and nothing else."""

CONTEXT_RECALL_PROMPT = """You are evaluating whether the retrieved CONTEXT contains the information
needed to produce the REFERENCE ANSWER.

REFERENCE ANSWER:
{ground_truth}

CONTEXT:
{context}

How much of the reference answer is supported by the context?
- 1.0 = all information in the reference answer is present in the context
- 0.0 = none of it is present

Reply with ONLY a single number between 0.0 and 1.0 and nothing else."""


def load_gold(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fmt_gt(gt) -> str:
    return ", ".join(str(x) for x in gt) if isinstance(gt, list) else str(gt)


def parse_score(text: str):
    """
    Parse a [0,1] float from a judge reply. Prefers a leading number so replies
    like '0.8 — the answer is...' parse correctly, and rejects out-of-range noise
    like 'on a scale of 1 to 10'. Returns None if no valid score is found.
    """
    text = text.strip()
    # Prefer a number at the very start of the reply.
    m = re.match(r"\s*([01](?:\.\d+)?|0?\.\d+)", text)
    if not m:
        # Fall back to the first standalone decimal in [0,1].
        for cand in re.findall(r"\d*\.?\d+", text):
            try:
                v = float(cand)
            except ValueError:
                continue
            if 0.0 <= v <= 1.0:
                return v
        return None
    try:
        return max(0.0, min(1.0, float(m.group(1))))
    except ValueError:
        return None


def judge_score(prompt: str, judge_model: str):
    """One judge call → float in [0,1], or None if unparseable."""
    try:
        content = ollama_chat(judge_model, prompt, options={"temperature": 0.0, "num_predict": 16})
    except Exception as e:  # noqa: BLE001 — already retried inside ollama_chat
        log.error("Judge call failed permanently, recording None: %s", e)
        return None
    return parse_score(content)


def mean(xs) -> float:
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else 0.0


def pipeline_config_from_experiment(base_cfg: dict, exp: dict) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["chunking"]["chunk_size"] = exp["chunk_size"]
    cfg["chunking"]["chunk_overlap"] = exp.get("chunk_overlap", 0)
    cfg["embedding"]["model"] = exp["embedding_model"]
    cfg["retrieval"]["top_k"] = exp["top_k"]
    cfg["retrieval"]["reranking"] = exp.get("reranker") is not None
    if exp.get("reranker"):
        cfg["retrieval"]["reranker_model"] = exp["reranker"]
        cfg["retrieval"]["rerank_top_n"] = exp.get("rerank_top_n", exp["top_k"])
        cfg["retrieval"]["rerank_fetch"] = exp.get("rerank_top_n", 10)
    cfg["vector_store"]["collection"] = exp["collection_name"]
    return cfg


def run_one(exp: dict, base_cfg: dict, defaults: dict, gold: list[dict], judge_model: str) -> dict:
    pipe_cfg = pipeline_config_from_experiment(base_cfg, exp)
    pipe_cfg["generation"]["model"] = defaults["generator_model"]
    retriever = Retriever(pipe_cfg)

    faith, ans_rel, ctx_rel, ctx_rec = [], [], [], []
    n = len(gold)
    for idx, ex in enumerate(gold, 1):
        q = ex["question"]
        gt = fmt_gt(ex["ground_truth"])
        out = rag_answer(q, pipe_cfg, retriever=retriever)
        ans = out["answer"]
        context = "\n\n".join(out["contexts"]) if out["contexts"] else "(no context retrieved)"

        f = judge_score(FAITHFULNESS_PROMPT.format(context=context, answer=ans), judge_model)
        r = judge_score(ANSWER_RELEVANCY_PROMPT.format(question=q, answer=ans), judge_model)
        p = judge_score(CONTEXT_RELEVANCE_PROMPT.format(question=q, context=context), judge_model)
        c = judge_score(CONTEXT_RECALL_PROMPT.format(ground_truth=gt, context=context), judge_model)
        faith.append(f); ans_rel.append(r); ctx_rel.append(p); ctx_rec.append(c)
        log.info("[%d/%d] faith=%s ans_rel=%s ctx_rel=%s ctx_rec=%s | %s",
                 idx, n, f, r, p, c, q[:55])

    return {
        "faithfulness": mean(faith),
        "answer_relevancy": mean(ans_rel),
        "context_relevance": mean(ctx_rel),
        "context_recall": mean(ctx_rec),
    }


def write_results(results: list[dict], results_path: str) -> None:
    os.makedirs(results_path, exist_ok=True)
    out_json = os.path.join(results_path, "eval_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log.info("Checkpointed %d config(s) -> %s", len(results), out_json)


def main(config_path: str) -> None:
    matrix = yaml.safe_load(open(config_path, encoding="utf-8"))
    base_cfg = yaml.safe_load(open("config/default.yaml", encoding="utf-8"))
    defaults = matrix["defaults"]
    judge_model = defaults["judge_model"]
    results_path = defaults.get("results_path", "results")
    gold = load_gold(defaults["eval_dataset"])

    results = []
    for key, exp in matrix["configs"].items():
        log.info("Running %s: %s (%d questions)", key, exp["name"], len(gold))
        t0 = time.time()
        scores = run_one(exp, base_cfg, defaults, gold, judge_model)
        scores.update({
            "config": key,
            "name": exp["name"],
            "chunk_size": exp["chunk_size"],
            "embedding": exp["embedding_model"],
            "top_k": exp["top_k"],
            "reranker": exp.get("reranker"),
            "runtime_sec": round(time.time() - t0, 1),
        })
        results.append(scores)
        log.info("%s done in %.1fs -> faith=%s ans_rel=%s ctx_rel=%s ctx_rec=%s",
                 key, scores["runtime_sec"], scores["faithfulness"],
                 scores["answer_relevancy"], scores["context_relevance"],
                 scores["context_recall"])
        # Checkpoint after every config so a later crash can't lose completed work.
        write_results(results, results_path)

    render_table(results, results_path)


def render_table(results: list[dict], results_path: str = "results") -> None:
    cols = ["config", "name", "chunk_size", "embedding", "top_k", "reranker"] + METRIC_NAMES
    lines = ["| " + " | ".join(cols) + " |",
             "| " + " | ".join("---" for _ in cols) + " |"]
    for r in results:
        row = [f"{r.get(c):.3f}" if isinstance(r.get(c), float) else str(r.get(c, ""))
               for c in cols]
        lines.append("| " + " | ".join(row) + " |")
    table = "\n".join(lines)
    os.makedirs(results_path, exist_ok=True)
    with open(os.path.join(results_path, "results_table.md"), "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print("\n" + table)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/experiments.yaml")
    args = ap.parse_args()
    main(args.config)