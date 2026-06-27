"""
End-to-end RAG answer: retrieve context, prompt Ollama, return answer + sources.

Usage:
    python -m src.generate.answer "How do I load a model in 4-bit?"
"""
import argparse

import yaml

from src.retrieve.retriever import Retriever
from src.utils import ollama_chat, get_logger

log = get_logger("answer")

PROMPT = """You are a precise assistant for the Hugging Face Transformers library.
Answer the question using ONLY the context below. If the context does not contain
the answer, say "I don't have enough information in the provided documentation."
Cite the source file(s) you used.

Context:
{context}

Question: {question}

Answer:"""

NO_CONTEXT_MSG = "I don't have enough information in the provided documentation."


def format_context(hits: list[dict]) -> str:
    return "\n\n---\n\n".join(f"[source: {h['source']}]\n{h['text']}" for h in hits)


def answer(question: str, cfg: dict, retriever: Retriever | None = None) -> dict:
    retriever = retriever or Retriever(cfg)
    hits = retriever.retrieve(question)

    if not hits:
        # Nothing retrieved — return a grounded refusal rather than hallucinating.
        return {
            "question": question,
            "answer": NO_CONTEXT_MSG,
            "contexts": [],
            "sources": [],
        }

    prompt = PROMPT.format(context=format_context(hits), question=question)
    content = ollama_chat(
        model=cfg["generation"]["model"],
        prompt=prompt,
        options={
            "temperature": cfg["generation"]["temperature"],
            "num_predict": cfg["generation"]["max_tokens"],
        },
    )
    return {
        "question": question,
        "answer": content,
        "contexts": [h["text"] for h in hits],
        "sources": [h["source"] for h in hits],
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--config", default="config/default.yaml")
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    out = answer(args.question, cfg)
    print("\n=== ANSWER ===\n" + out["answer"])
    print("\n=== SOURCES ===")
    for s in dict.fromkeys(out["sources"]):
        print(" -", s)