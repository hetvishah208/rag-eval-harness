"""
Evaluation harness (Windows-safe, sequential).

RAGAS's async executor hangs on some Windows setups (jobs sit at 0/N and never
dispatch to Ollama). This version bypasses RAGAS entirely and computes the four
core metrics with direct, sequential Ollama calls — same judge model, same
metric definitions, full control over the loop. $0, fully local.

Metrics (all judged by a local Ollama model, scored 0..1):
  - faithfulness        : is every claim in the answer supported by the contexts?
  - answer_relevancy    : does the answer address the question?
  - context_precision   : are the retrieved contexts relevant to the question?
  - context_recall      : do the contexts contain what's needed for the ground truth?

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

import ollama

from src.retrieve.retriever import Retriever
from src.generate.answer import answer as rag_answer


METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


# ---------- judge prompts (each asks for a single 0..1 score) ----------

FAITHFULNESS_PROMPT = """You are evaluating whether an ANSWER is faithful to the provided CONTEXT.
A faithful answer makes only claims that are supported by the context.

CONTEXT:
{context}

ANSWER:
{answer}

Score from 0.0 to 1.0 how faithful the answer is to the context:
- 1.0 = every claim is supported by the context
- 0.0 = the answer contradicts or invents information not in the context

Respond with ONLY a number between 0.0 and 1.0. No explanation."""

ANSWER_RELEVANCY_PROMPT = """You are evaluating whether an ANSWER is relevant to a QUESTION.

QUESTION:
{question}

ANSWER:
{answer}

Score from 0.0 to 1.0 how directly the answer addresses the question:
- 1.0 = fully answers the question
- 0.0 = does not address the question at all

Respond with ONLY a number between 0.0 and 1.0. No explanation."""

CONTEXT_PRECISION_PROMPT = """You are evaluating whether the retrieved CONTEXT is relevant to a QUESTION.

QUESTION:
{question}

CONTEXT:
{context}

Score from 0.0 to 1.0 how relevant the context is for answering the question:
- 1.0 = the context is highly relevant and useful
- 0.0 = the context is unrelated to the question

Respond with ONLY a number between 0.0 and 1.0. No explanation."""

CONTEXT_RECALL_PROMPT = """You are evaluating whether the retrieved CONTEXT contains the information
needed to produce the REFERENCE ANSWER.

REFERENCE ANSWER:
{ground_truth}

CONTEXT:
{context}

Score from 0.0 to 1.0 how much of the reference answer is supported by the context:
- 1.0 = all information in the reference answer is present in the context
- 0.0 = none of it is present

Respond with ONLY a number between 0.0 and 1.0. No explanation."""


def load_gold(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fmt_gt(gt):
    return ", ".join(str(x) for x in gt) if isinstance(gt, list) else str(gt)


def judge_score(prompt, judge_model):
    """Single Ollama call that returns a float in [0,1]. Robust to messy output."""
    resp = ollama.chat(
        model=judge_model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.0, "num_predict": 16},
    )
    text = resp["message"]["content"].strip()
    m = re.search(r"\d*\.?\d+", text)
    if not m:
        return None
    try:
        val = float(m.group())
    except ValueError:
        return None
    return max(0.0, min(1.0, val))  # clamp to [0,1]


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else 0.0


def pipeline_config_from_experiment(base_cfg, exp):
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


def run_one(exp, base_cfg, defaults, gold, judge_model):
    pipe_cfg = pipeline_config_from_experiment(base_cfg, exp)
    pipe_cfg["generation"]["model"] = defaults["generator_model"]
    retriever = Retriever(pipe_cfg)

    faith, ans_rel, ctx_prec, ctx_rec = [], [], [], []
    n = len(gold)
    for idx, ex in enumerate(gold, 1):
        q = ex["question"]
        gt = fmt_gt(ex["ground_truth"])
        out = rag_answer(q, pipe_cfg, retriever=retriever)
        answer = out["answer"]
        context = "\n\n".join(out["contexts"])

        f = judge_score(FAITHFULNESS_PROMPT.format(context=context, answer=answer), judge_model)
        r = judge_score(ANSWER_RELEVANCY_PROMPT.format(question=q, answer=answer), judge_model)
        p = judge_score(CONTEXT_PRECISION_PROMPT.format(question=q, context=context), judge_model)
        c = judge_score(CONTEXT_RECALL_PROMPT.format(ground_truth=gt, context=context), judge_model)
        faith.append(f); ans_rel.append(r); ctx_prec.append(p); ctx_rec.append(c)
        print(f"  [{idx}/{n}] faith={f} ans_rel={r} ctx_prec={p} ctx_rec={c}  | {q[:55]}")

    return {
        "faithfulness": mean(faith),
        "answer_relevancy": mean(ans_rel),
        "context_precision": mean(ctx_prec),
        "context_recall": mean(ctx_rec),
    }


def main(config_path):
    matrix = yaml.safe_load(open(config_path))
    base_cfg = yaml.safe_load(open("config/default.yaml"))
    defaults = matrix["defaults"]
    judge_model = defaults["judge_model"]
    gold = load_gold(defaults["eval_dataset"])

    results = []
    for key, exp in matrix["configs"].items():
        print(f"\n[eval] running {key}: {exp['name']}  ({len(gold)} questions)")
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
        print(f"[eval] {key} done in {scores['runtime_sec']}s -> "
              f"faith={scores['faithfulness']} ans_rel={scores['answer_relevancy']} "
              f"ctx_prec={scores['context_precision']} ctx_rec={scores['context_recall']}")

    os.makedirs(defaults.get("results_path", "results"), exist_ok=True)
    out_json = os.path.join(defaults.get("results_path", "results"), "eval_results.json")
    json.dump(results, open(out_json, "w"), indent=2)
    print(f"\n[eval] wrote {out_json}")
    render_table(results)


def render_table(results):
    cols = ["config", "name", "chunk_size", "embedding", "top_k", "reranker"] + METRIC_NAMES
    lines = ["| " + " | ".join(cols) + " |",
             "| " + " | ".join("---" for _ in cols) + " |"]
    for r in results:
        row = [f"{r.get(c):.3f}" if isinstance(r.get(c), float) else str(r.get(c, ""))
               for c in cols]
        lines.append("| " + " | ".join(row) + " |")
    table = "\n".join(lines)
    open("results/results_table.md", "w", encoding="utf-8").write(table + "\n")
    print("\n" + table)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/experiments.yaml")
    args = ap.parse_args()
    main(args.config)