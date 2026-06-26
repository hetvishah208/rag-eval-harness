"""
Streamlit demo for HF Spaces (free tier, CPU).

Design for free CPU hosting:
- Retrieval runs LIVE (embeddings + ChromaDB are fast on CPU).
- Generation calls the free Hugging Face Inference API (no local LLM in the Space).
- ~20 demo questions have CACHED answers, served instantly if the API rate-limits.
- A badge shows whether each answer was generated live or served from cache.

Set HF_TOKEN as a Space secret (free account token works for the Inference API).
"""
import json
import os

import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from huggingface_hub import InferenceClient

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
GEN_MODEL = "HuggingFaceH4/zephyr-7b-beta"   # free hosted instruct model
COLLECTION = "hf_docs"
PERSIST_DIR = "data/processed/chroma"
CACHE_PATH = "app/answer_cache.json"
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
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {}


def retrieve(coll, query, k=TOP_K):
    res = coll.query(query_texts=[query], n_results=k)
    return list(zip(res["documents"][0], [m["source"] for m in res["metadatas"][0]]))


def generate(question, contexts):
    token = os.environ.get("HF_TOKEN")
    client = InferenceClient(token=token)
    ctx = "\n\n---\n\n".join(c for c, _ in contexts)
    prompt = (
        "Answer using ONLY the context. If unknown, say so.\n\n"
        f"Context:\n{ctx}\n\nQuestion: {question}\nAnswer:"
    )
    out = client.text_generation(prompt, model=GEN_MODEL, max_new_tokens=400, temperature=0.0)
    return out


st.title("📚 RAG over Hugging Face Docs — with an Eval Harness")
st.caption("Live retrieval (ChromaDB + BGE-small) · generation via free HF Inference API · "
           "[eval results & methodology in the repo README]")

cache = load_cache()
coll = get_collection()

q = st.text_input("Ask about the Transformers library:",
                  placeholder="How do I load a model in 4-bit?")
if st.button("Ask") and q.strip():
    with st.spinner("Retrieving…"):
        hits = retrieve(coll, q)
    key = q.strip().lower()
    if key in cache:
        st.success("Answer (cached)")
        st.write(cache[key])
    else:
        try:
            with st.spinner("Generating live…"):
                ans = generate(q, hits)
            st.info("Answer (generated live)")
            st.write(ans)
        except Exception as e:
            st.warning(f"Live generation unavailable ({e}). Showing retrieved context only.")
    with st.expander("Retrieved sources"):
        for doc, src in hits:
            st.markdown(f"**{src}**")
            st.text(doc[:400] + "…")
