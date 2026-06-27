# RAG Evaluation Harness

A deployed Retrieval-Augmented Generation (RAG) system over the Hugging Face Transformers documentation, with a built-in evaluation harness that measures and compares retrieval configurations on four quality metrics — built end-to-end on free, open-source, local, CPU-only tooling. Zero budget, no paid APIs, no paid hosting.

**[Live Demo](https://huggingface.co/spaces/hetviiishahhh/rag-eval-harness)** · **[Results & Tradeoffs](docs/TRADEOFFS.md)** · **[Gold Dataset Methodology](docs/GOLD_DATASET.md)**

---

## Why this project exists

Most RAG demos prove a pipeline *runs*. They almost never prove the pipeline is *good*. In GenAI hiring right now, the rare and high-signal skill isn't wiring up a retriever and an LLM — plenty of people can do that — it's being able to measure whether the system actually works and make defensible engineering decisions from those measurements.

This project is built around that gap. It's a working RAG system, but the centerpiece is the evaluation harness that scores it across multiple configurations and produces a real comparison table with an honest tradeoffs analysis.

### The business case

Any company shipping a RAG product (internal knowledge assistant, customer support bot, docs search, etc.) faces the same questions: *Which embedding model? How big should chunks be? How many do we retrieve? Is a reranker worth the added latency and cost?* Guessing is expensive — a bad retrieval config quietly produces wrong answers that erode user trust. This harness is the apparatus that answers those questions with numbers instead of opinions, and it does it without spending a cent, which matters for prototyping, for small teams, and for anyone who can't send proprietary documents to a third-party API.

### The personal case

This patches a specific, real gap. Earlier RAG work of mine claimed "reduced hallucinated responses by 40%" — a good outcome, but with no public, reproducible proof that I can actually *measure* RAG quality. This project is that proof: a public repo where the evaluation methodology, the gold dataset construction, the judge validation, and the results are all visible and reproducible.

---

## What it does

1. **Ingests** the Hugging Face Transformers documentation (markdown), cleans it, and chunks it.
2. **Embeds** the chunks with local sentence-transformers models and stores them in a local ChromaDB vector database.
3. **Answers** questions end-to-end: retrieve relevant chunks → build a grounded prompt → generate an answer with a local LLM (Ollama), citing its sources.
4. **Evaluates** four different retrieval configurations against a hand-verified gold dataset, scoring each on faithfulness, answer relevancy, context relevance, and context recall — using a local LLM as the judge (no paid API anywhere).
5. **Produces** a results table, a tradeoffs writeup, and a documented judge-validation step.

---

## Architecture

```
Hugging Face Transformers docs (markdown, ~719 files)
        │
        ▼  chunk_docs.py   (parse + clean MDX/frontmatter + chunk)
   chunks_256.json / chunks_512.json
        │
        ▼  build_all_indexes.py → build_index.py
   ChromaDB collections: minilm_256, minilm_512, bge_256
        │
        ▼  retriever.py   (top-k similarity search, optional cross-encoder rerank)
        │
        ▼  answer.py       (retrieved context + question → Ollama Phi-3-mini → answer + sources)
        │
        ▼  run_eval.py     (per-question metric scoring via local Mistral 7B judge)
        │
        ▼  results/eval_results.json + results/results_table.md
```

---

## The tech stack (everything free and local)

| Layer | Tool | Notes |
|-------|------|-------|
| Generator LLM | Ollama — phi3:mini (3.8B, q4) | Fits in 16 GB RAM, runs on CPU |
| Judge LLM | Ollama — mistral:7b-instruct-q4_0 | Scores the four metrics |
| Judge embeddings | Ollama — nomic-embed-text | Used by the eval layer |
| Retrieval embeddings | sentence-transformers — BAAI/bge-small-en-v1.5, all-MiniLM-L6-v2 | Two models compared |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | Cross-encoder, query-time |
| Vector store | ChromaDB | Persistent, local |
| Eval metrics | Custom, RAGAS-inspired | Direct Ollama judging (see note below) |
| Demo | Streamlit on Hugging Face Spaces (Docker) | Free tier, [live](https://huggingface.co/spaces/hetviiishahhh/rag-eval-harness) |
| CI | GitHub Actions | Config exists (not yet wired) |

Tested on: Windows 11, Python 3.11.5, Intel Iris Xe (CPU-only, no usable GPU), 16 GB RAM. The hardware constraints directly shaped the design: small quantized models, a capped 4-config experiment matrix, and a 31-question gold set are all deliberate choices to keep the full evaluation runnable overnight on a CPU.

---

## Live demo

**[https://huggingface.co/spaces/hetviiishahhh/rag-eval-harness](https://huggingface.co/spaces/hetviiishahhh/rag-eval-harness)**

The deployed app runs on Hugging Face Spaces (free tier, Docker + Streamlit). Because the free tier is CPU-only and can't run Ollama, the deployment separates retrieval from generation:

- **Retrieval** runs live in the Space — ChromaDB + BGE-small, fast on CPU.
- **Generation** calls the free HF Inference API instead of local Ollama.
- **Fallback**: `app/answer_cache.json` serves cached answers if the Inference API rate-limits.
- Requires `HF_TOKEN` set as a Space secret for the Inference API.

The local evaluation harness (the real point of the project) runs entirely locally with Ollama — the HF Spaces deployment is a demo layer on top.

---

## The experiment matrix

Four configurations, each varying one or two axes from its neighbor so the comparison is interpretable:

| Config | Embedding | Chunk size | top-k | Reranking | Collection |
|--------|-----------|-----------|-------|-----------|------------|
| A | all-MiniLM-L6-v2 | 256 | 3 | no | minilm_256 |
| B | all-MiniLM-L6-v2 | 512 | 5 | no | minilm_512 |
| C | BAAI/bge-small-en-v1.5 | 256 | 3 | no | bge_256 |
| D | BAAI/bge-small-en-v1.5 | 256 | 3 | yes | bge_256 (reranked at query time) |

A→C isolates the embedding model. A→B isolates chunk size + top-k. C→D isolates reranking. Only three vector indexes are actually built — D reuses C's collection and applies the cross-encoder rerank at query time, so there's no redundant index.

---

## Results

(Generated by `src/eval/run_eval.py`, stored in `results/eval_results.json`)

| Config | Embedding | Chunk | top-k | Rerank | Faithfulness | Answer Rel. | Ctx Relevance | Ctx Recall |
|--------|-----------|-------|-------|--------|-------------|-------------|---------------|------------|
| A | MiniLM | 256 | 3 | no | 0.794 | 0.745 | 0.742 | 0.726 |
| B | MiniLM | 512 | 5 | no | 0.839 | 0.813 | 0.703 | 0.750 |
| C | BGE-small | 256 | 3 | no | 0.810 | 0.765 | 0.729 | 0.739 |
| D | BGE-small | 256 | 3 | yes | **0.860** | **0.835** | 0.694 | **0.769** |

> **Note on metric naming:** The column above reads "Ctx Relevance" (not "Ctx Precision"). These are RAGAS-inspired single-shot judge approximations, not RAGAS's exact algorithms. `context_relevance` is the honest name — it's a relevance score, not RAGAS's rank-aware precision algorithm. Earlier runs and `results/results_table.md` may still say `context_precision`; they refer to the same metric.

### What the metrics mean

- **Faithfulness** — is the answer grounded in the retrieved context, or is it making things up?
- **Answer relevancy** — does the answer actually address the question?
- **Context relevance** — are the retrieved chunks actually relevant (signal vs noise)?
- **Context recall** — did retrieval find the chunks needed to answer fully?

### Headline findings

- **Reranking (D) produces the best answers** — highest faithfulness, answer relevancy, and context recall. If shipping one config, it's D.
- **But reranking had the lowest context relevance** — it draws from a wider candidate set, so a few borderline chunks survive into the final context. The generator handles them, but the relevance metric penalizes them. A real, documented precision/recall tradeoff.
- **BGE-small beats MiniLM** (A vs C) on faithfulness and relevancy — a worthwhile swap since both models are similar in size and speed.
- **Bigger chunks + higher top-k (B)** improves recall but hurts precision — the classic precision/recall tradeoff, visible cleanly in the data.

Full analysis: [docs/TRADEOFFS.md](docs/TRADEOFFS.md).

---

## Honest limitation: the judge

The eval uses a local 7B model (Mistral) as its judge, which is much smaller than the GPT-4-class models usually used for this. It was validated by hand on 13 questions (see [docs/GOLD_DATASET.md](docs/GOLD_DATASET.md)) — agreement with human scoring was only ~46%, with the judge being too lenient on hallucinated content and occasionally too harsh on correct answers. Conclusion: the results are directional, not precise. The relative ordering of configs is trustworthy; small absolute gaps are not. This is the honest cost of doing everything on free, local, CPU-only tooling, and it's documented rather than hidden.

---

## How the gold dataset was built

The 31-question evaluation set was generated then 100% human-verified:

1. Sample substantive chunks from the corpus.
2. Have a local model draft a question + reference answer per chunk.
3. Review every draft by hand; reject anything unsupported, vague, snippet-dependent, answerable without the docs, or duplicated.

Started from 70 drafts, kept 28 after review (plus 3 hand-seeded = 31). The ~46% keep rate is itself a finding: a large fraction of LLM-drafted questions aren't good eval questions, and the value is in the filtering. Full methodology: [docs/GOLD_DATASET.md](docs/GOLD_DATASET.md).

---

## Production-grade fixes applied

A full code review was completed covering reproducibility, reliability, and correctness:

- **Reproducibility bug fixed**: `chunk_docs.py` had a skip pattern that would have excluded model card docs on a fresh clone — but the gold dataset has questions from those docs. Removed.
- **Retry + backoff on Ollama calls**: `src/utils.py` provides `ollama_chat()` with 3 retries and exponential backoff. A dropped connection during a 4-hour eval no longer kills the run.
- **Per-config checkpointing**: `run_eval.py` writes `eval_results.json` after every config completes. A crash at config D can't lose A/B/C.
- **Metric naming honesty**: `context_precision` renamed to `context_relevance` with a docstring explaining why.
- **Structured logging**: All `print()` calls replaced with `logging` module across every file.
- **Scoped exception handling**: Bare `except: pass` replaced with targeted exception handling.
- **Tighter score parsing**: Regex tightened to prevent "scale of 1 to 10" style judge responses from producing silent wrong scores.
- **Pinned dependencies**: `requirements.txt` pins the exact combination that avoids the RAGAS/langchain vertexai crash.

---

## Setup & reproduction (Windows)

Every step below was actually run on Windows 11 / Python 3.11.5 / CPU-only / 16 GB RAM. Errors that came up and their fixes are included inline.

### Prerequisites

- Python 3.10+ (`py --version`)
- Git (`git --version`)
- ~10 GB free disk (models + embeddings + indexes)

### 1. Project structure

If files arrived flat, create the folders and move files into place:

```powershell
cd "C:\Users\hetvi\Desktop\Interview Stack\RAG Evaluation Harness"

mkdir -Force src\ingest, src\retrieve, src\generate, src\eval
mkdir -Force config, data\raw, data\eval, data\processed, results
mkdir -Force app, docs, tests, .github\workflows
```

Create the `__init__.py` files Python needs for imports:

```powershell
New-Item -ItemType File -Force src\__init__.py
New-Item -ItemType File -Force src\ingest\__init__.py
New-Item -ItemType File -Force src\retrieve\__init__.py
New-Item -ItemType File -Force src\generate\__init__.py
New-Item -ItemType File -Force src\eval\__init__.py
New-Item -ItemType File -Force tests\__init__.py
```

### 2. Python environment

Windows gotcha: `python` may be hijacked by the Microsoft Store alias. Use `py`.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If activation is blocked by execution policy:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Install dependencies:

```powershell
pip install -r requirements.txt
pip install langchain-ollama langchain-huggingface
```

**Critical version note (RAGAS compatibility):** newer RAGAS (0.4.x) breaks against langchain 1.x with `ModuleNotFoundError: langchain_community.chat_models.vertexai`. Pin the working set:

```powershell
pip install "ragas==0.2.10" "langchain==0.3.27" "langchain-community==0.3.27" "langchain-core>=0.3,<0.4" "langchain-ollama==0.2.3" "langchain-huggingface==0.1.2"
```

Orphaned `langgraph` / `langchain-classic` dependency warnings after this are harmless — those packages aren't used by the eval.

### 3. Ollama + models

Install Ollama from [https://ollama.com/download/windows](https://ollama.com/download/windows).

If `ollama` isn't recognized in the terminal, its CLI isn't on PATH. Set it for the session (needed in every new terminal):

```powershell
$env:PATH += ";$env:LOCALAPPDATA\Programs\Ollama"
ollama --version
```

Pull the three models:

```powershell
ollama pull phi3:mini                    # generator  (~2.2 GB)
ollama pull mistral:7b-instruct-q4_0     # judge      (~4.1 GB)
ollama pull nomic-embed-text             # judge embeddings (~274 MB)
```

### 4. Clone the corpus (sparse — English docs only)

```powershell
git clone --depth 1 --filter=blob:none --sparse https://github.com/huggingface/transformers.git data\raw
cd data\raw
git sparse-checkout set docs/source/en
cd ..\..
```

Result: ~719 markdown files under `data\raw\docs\source\en\`.

### 5. Chunk the corpus

```powershell
python -m src.ingest.chunk_docs --chunk-sizes 256 512
```

Produces `chunks_256.json` (~24,775 chunks) and `chunks_512.json` (~11,793 chunks) in `data\processed\`.

### 6. Build the vector indexes

```powershell
python -m src.ingest.build_all_indexes
```

Builds the three unique ChromaDB collections (`minilm_256`, `minilm_512`, `bge_256`). ~15–25 min on CPU. Re-runs skip existing collections; use `--force` to rebuild.

If you hit `UnicodeDecodeError: 'charmap' codec can't decode...`: Windows defaults to `cp1252`. The corrected code already handles this with `encoding="utf-8"` on all file opens.

### 7. Configure the default collection

Open `config\default.yaml` and set the collection the single-question tool uses:

```yaml
collection: "bge_256"
```

### 8. Test end-to-end RAG

```powershell
python -m src.generate.answer "How do I load a model in 4-bit?"
```

Returns an answer plus the source files it retrieved. (Note: small local models will sometimes hallucinate API details even with correct context retrieved — that's exactly what the eval harness is built to catch and measure.)

### 9. Build the gold dataset

Generate drafts:

```powershell
python -m src.eval.build_gold --n 70
```

The chunk-length filter in `build_gold.py` must match your chunk size. For 256-char chunks, the filter is `> 200` (a `> 400` filter returns zero candidates and the run produces 0 drafts).

Then review them interactively (keep / reject / edit), which writes the verified `data\eval\gold_qa.json`:

```powershell
python -m src.eval.review_gold
```

### 10. Run the evaluation

Smoke test first (always — catches wiring bugs in minutes, not hours):

```powershell
python -m src.eval.run_eval --config config/experiments_smoke.yaml
```

Then the full 4-config run (overnight job on CPU — plug in, disable sleep, close heavy apps so it doesn't swap):

```powershell
python -m src.eval.run_eval --config config/experiments.yaml
```

Outputs `results/eval_results.json` and `results/results_table.md`.

**Critical implementation note — why the eval doesn't use RAGAS's executor:** RAGAS's async job executor hangs on Windows — every judge job sits at 0/N, Ollama stays idle, and the jobs eventually all time out. Setting `WindowsSelectorEventLoopPolicy` did not fix it. The working solution was to bypass RAGAS's `evaluate()` executor entirely and compute all four metrics with direct, sequential `ollama.chat()` calls using custom judge prompts. This is what `run_eval.py` does now. It's slower (serial) but reliable, transparent, and gives full control over the judge prompts.

---

## Common issues (all encountered and solved)

| Issue | Fix |
|-------|-----|
| `python` not found | Use `py` (Windows Store alias conflict) |
| `Activate.ps1` blocked | `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| `ollama` not recognized | `$env:PATH += ";$env:LOCALAPPDATA\Programs\Ollama"` (per terminal) |
| `ModuleNotFoundError: ...vertexai` | Pin `ragas==0.2.10` + langchain 0.3.x set (see step 2) |
| `UnicodeDecodeError` reading chunks | Fixed in code — `encoding="utf-8"` on all file opens |
| Collection `[hf_docs]` does not exist | Set `collection: "bge_256"` in `config\default.yaml` |
| `build_gold` writes 0 drafts | Lower the chunk-length filter from `>400` to `>200` |
| Eval hangs at 0/N, Ollama idle | Fixed — `run_eval.py` bypasses RAGAS's executor |
| Memory near 100% during eval | Close other apps; models + Python are tight on 16 GB |

---

## Project structure

```
RAG Evaluation Harness/
├── config/
│   ├── default.yaml              # single-query pipeline config
│   ├── experiments.yaml          # the A/B/C/D matrix (full run)
│   ├── experiments_ci.yaml       # tiny matrix for CI
│   └── experiments_smoke.yaml    # 1-config smoke test
├── src/
│   ├── utils.py                   # shared logging + ollama_chat with retries
│   ├── ingest/
│   │   ├── chunk_docs.py          # parse + clean + chunk HF docs → JSON
│   │   ├── build_index.py         # embed chunk JSON → one ChromaDB collection
│   │   └── build_all_indexes.py   # build all unique indexes from the matrix
│   ├── retrieve/retriever.py      # top-k search + optional cross-encoder rerank
│   ├── generate/answer.py         # retrieve + prompt + Ollama → answer + sources
│   └── eval/
│       ├── build_gold.py          # draft gold Q&A from corpus chunks
│       ├── review_gold.py         # interactive keep/reject/edit → gold_qa.json
│       └── run_eval.py            # sequential metric scoring via local judge
├── app/
│   ├── streamlit_app.py           # deployed demo (live retrieval + HF Inference API)
│   └── answer_cache.json          # cached demo answers (rate-limit fallback)
├── data/
│   ├── raw/                       # cloned HF docs (sparse checkout)
│   ├── processed/                 # chunk JSON + ChromaDB
│   └── eval/gold_qa.json          # 31 verified Q&A pairs
├── docs/
│   ├── GOLD_DATASET.md            # dataset methodology + judge validation
│   └── TRADEOFFS.md               # full results analysis
├── results/
│   ├── eval_results.json
│   └── results_table.md
├── tests/test_smoke.py
├── README.md
├── SETUP.md                       # step-by-step setup log
└── requirements.txt
```

---

## License

MIT