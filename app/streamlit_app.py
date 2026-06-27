"""
Streamlit demo for HF Spaces (free tier, CPU).

- Retrieval runs LIVE (BGE-small embeddings + ChromaDB, fast on CPU).
- Generation calls the free Hugging Face Inference Providers API (chat completions).
- Cached answers serve as a fallback if the API is unavailable.

Set HF_TOKEN as a Space secret (a free read token works).
"""

import os
os.environ.setdefault("HF_HOME", "/tmp/huggingface")
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", "/tmp/huggingface")

import json

import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from huggingface_hub import InferenceClient

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
GEN_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
COLLECTION = "bge_256"
PERSIST_DIR = "src/data/processed/chroma"
CACHE_PATH = "src/answer_cache.json"
TOP_K = 4

st.set_page_config(page_title="RAG over HF Docs", page_icon="📚")


@st.cache_resource
def get_collection():
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    return client.get_collection(COLLECTION, embedding_function=ef)


@st.cache_data
def load_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def retrieve(coll, query, k=TOP_K):
    res = coll.query(query_texts=[query], n_results=k)
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    sources = [m.get("source", "unknown") for m in metas]
    return list(zip(docs, sources))


def generate(question, contexts):
    token = os.environ.get("HF_TOKEN")
    client = InferenceClient(token=token)
    ctx = "\n\n---\n\n".join(c for c, _ in contexts)
    messages = [
        {
            "role": "system",
            "content": "You answer questions about the Hugging Face Transformers library "
                       "using ONLY the provided context. If the answer isn't in the context, say so.",
        },
        {
            "role": "user",
            "content": f"Context:\n{ctx}\n\nQuestion: {question}",
        },
    ]
    completion = client.chat.completions.create(
        model=GEN_MODEL,
        messages=messages,
        max_tokens=400,
        temperature=0.0,
    )
    return completion.choices[0].message.content


st.title("📚 RAG over Hugging Face Docs — with an Eval Harness")
st.caption("Live retrieval (ChromaDB + BGE-small) · generation via free HF Inference API · "
           "eval results & methodology in the repo README")

cache = load_cache()
coll = get_collection()

q = st.text_input("Ask about the Transformers library:",
                  placeholder="How do I load a model in 4-bit?")
if st.button("Ask") and q.strip():
    with st.spinner("Retrieving…"):
        hits = retrieve(coll, q)
    key = q.strip().lower()
    answered = False
    try:
        with st.spinner("Generating live…"):
            ans = generate(q, hits)
        st.info("Answer (generated live)")
        st.write(ans)
        answered = True
    except Exception as e:
        if key in cache:
            st.success("Answer (cached)")
            st.write(cache[key])
            answered = True
        else:
            st.warning(f"Live generation unavailable ({e}). Showing retrieved context below.")
    with st.expander("Retrieved sources", expanded=not answered):
        for doc, src in hits:
            st.markdown(f"**{src}**")
            st.text(doc[:400] + "…")