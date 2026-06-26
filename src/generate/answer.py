"""
End-to-end RAG answer: retrieve context, prompt Ollama, return answer + sources.

Usage:
    python -m src.generate.answer "How do I load a model in 4-bit?"
"""
import argparse
import yaml
import ollama

from src.retrieve.retriever import Retriever


PROMPT = """You are a precise assistant for the Hugging Face Transformers library.
Answer the question using ONLY the context below. If the context does not contain
the answer, say "I don't have enough information in the provided documentation."
Cite the source file(s) you used.

Context:
{context}

Question: {question}

Answer:"""


def format_context(hits):
    blocks = []
    for h in hits:
        blocks.append(f"[source: {h['source']}]\n{h['text']}")
    return "\n\n---\n\n".join(blocks)


def answer(question, cfg, retriever=None):
    retriever = retriever or Retriever(cfg)
    hits = retriever.retrieve(question)
    context = format_context(hits)
    prompt = PROMPT.format(context=context, question=question)
    resp = ollama.chat(
        model=cfg["generation"]["model"],
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": cfg["generation"]["temperature"],
            "num_predict": cfg["generation"]["max_tokens"],
        },
    )
    return {
        "question": question,
        "answer": resp["message"]["content"],
        "contexts": [h["text"] for h in hits],
        "sources": [h["source"] for h in hits],
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--config", default="config/default.yaml")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    out = answer(args.question, cfg)
    print("\n=== ANSWER ===\n" + out["answer"])
    print("\n=== SOURCES ===")
    for s in dict.fromkeys(out["sources"]):
        print(" -", s)
