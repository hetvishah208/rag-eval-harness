# RAG Evaluation Harness — Setup Guide (Windows)

A step-by-step walkthrough to go from a fresh clone to a working end-to-end RAG pipeline with eval harness. Covers every installation, command, error you might hit, and the fix.

**Tested on:** Windows 11, Python 3.11.5, Intel Iris Xe (CPU-only), 16 GB RAM

---

## Prerequisites

Before you start, make sure you have:

- **Python 3.10+** installed (verify with `py --version`)
- **Git** installed (verify with `git --version`)
- **VS Code** (recommended editor)
- ~10 GB free disk space (for models + embeddings + vector indexes)

---

## Phase 0: Project Structure

If you downloaded the files flat (all in one folder), create the directory structure first.

### 0.1 — Create folders

```powershell
cd "C:\Users\hetvi\Desktop\Interview Stack\RAG Evaluation Harness"

mkdir -Force src\ingest
mkdir -Force src\retrieve
mkdir -Force src\generate
mkdir -Force src\eval
mkdir -Force config
mkdir -Force data\raw
mkdir -Force data\eval
mkdir -Force data\processed
mkdir -Force results
mkdir -Force app
mkdir -Force docs
mkdir -Force tests
mkdir -Force .github\workflows
```

### 0.2 — Move files to correct locations

```powershell
# config/
Move-Item default.yaml config\
Move-Item experiments.yaml config\
Move-Item experiments_ci.yaml config\

# src/ingest/
Move-Item chunk_docs.py src\ingest\
Move-Item build_index.py src\ingest\
Move-Item build_all_indexes.py src\ingest\

# src/retrieve/
Move-Item retriever.py src\retrieve\

# src/generate/
Move-Item answer.py src\generate\

# src/eval/
Move-Item run_eval.py src\eval\
Move-Item build_gold.py src\eval\

# app/
Move-Item streamlit_app.py app\
Move-Item answer_cache.json app\

# docs/
Move-Item GOLD_DATASET.md docs\
Move-Item TRADEOFFS.md docs\

# data/eval/
Move-Item gold_qa.json data\eval\

# tests/
Move-Item test_smoke.py tests\

# .github/workflows/
Move-Item ci.yml .github\workflows\

# These stay in root: .gitignore, README.md, requirements.txt
```

### 0.3 — Create `__init__.py` files (required for Python imports)

```powershell
New-Item -ItemType File -Force src\__init__.py
New-Item -ItemType File -Force src\ingest\__init__.py
New-Item -ItemType File -Force src\retrieve\__init__.py
New-Item -ItemType File -Force src\generate\__init__.py
New-Item -ItemType File -Force src\eval\__init__.py
New-Item -ItemType File -Force tests\__init__.py
```

### 0.4 — Verify structure

```powershell
tree /F /A
```

Expected output:

```
RAG Evaluation Harness/
├── .github/workflows/ci.yml
├── .gitignore
├── README.md
├── requirements.txt
├── config/
│   ├── default.yaml
│   ├── experiments.yaml
│   └── experiments_ci.yaml
├── src/
│   ├── __init__.py
│   ├── ingest/
│   │   ├── __init__.py
│   │   ├── chunk_docs.py
│   │   ├── build_index.py
│   │   └── build_all_indexes.py
│   ├── retrieve/
│   │   ├── __init__.py
│   │   └── retriever.py
│   ├── generate/
│   │   ├── __init__.py
│   │   └── answer.py
│   └── eval/
│       ├── __init__.py
│       ├── run_eval.py
│       └── build_gold.py
├── app/
│   ├── streamlit_app.py
│   └── answer_cache.json
├── data/eval/gold_qa.json
├── docs/
│   ├── GOLD_DATASET.md
│   └── TRADEOFFS.md
├── results/
└── tests/
    ├── __init__.py
    └── test_smoke.py
```

---

## Phase 1: Python Environment

### 1.1 — Create virtual environment

> **Windows gotcha:** `python` may be aliased to the Microsoft Store. Use `py` instead.

```powershell
py -m venv .venv
```

### 1.2 — Activate the venv

```powershell
.\.venv\Scripts\Activate.ps1
```

You should see `(.venv)` at the start of your prompt.

> **If you get an execution policy error:**
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```
> Then retry the activate command.

### 1.3 — Install dependencies

```powershell
pip install -r requirements.txt
```

This installs ~120 packages including PyTorch (~123 MB), ChromaDB, sentence-transformers, RAGAS, Streamlit, etc. Takes 2-5 minutes.

### 1.4 — Install two additional packages (not in requirements.txt)

These are needed by `run_eval.py` for the local Ollama judge wiring:

```powershell
pip install langchain-ollama langchain-huggingface
```

---

## Phase 2: Ollama + Models

### 2.1 — Install Ollama

Download and install from: https://ollama.com/download/windows

After installing, the Ollama app should appear in your system tray / taskbar.

### 2.2 — Add Ollama to PATH (if `ollama` command isn't recognized)

After installing Ollama, the CLI may not be on your PATH. Test it:

```powershell
ollama --version
```

> **If you get "not recognized" error:**
> ```powershell
> # Find where Ollama is installed
> & "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" --version
>
> # If that works, add it to PATH for this session
> $env:PATH += ";$env:LOCALAPPDATA\Programs\Ollama"
> ollama --version
> ```

> **Note:** This PATH fix is per-session. You'll need to run the `$env:PATH` line again each time you open a new terminal, or add Ollama to your system PATH permanently via Windows Settings > Environment Variables.

### 2.3 — Pull the three models

```powershell
ollama pull phi3:mini                    # Generator (~2.2 GB)
ollama pull mistral:7b-instruct-q4_0     # Judge for RAGAS eval (~4.1 GB)
ollama pull nomic-embed-text             # Embeddings for RAGAS judge (~274 MB)
```

Total download: ~6.6 GB. Takes 5-15 minutes depending on connection.

---

## Phase 3: Clone the Corpus

We use a sparse checkout to grab only the English documentation from the HF Transformers repo (~1.6 MB instead of the full multi-GB repo).

```powershell
git clone --depth 1 --filter=blob:none --sparse https://github.com/huggingface/transformers.git data\raw
cd data\raw
git sparse-checkout set docs/source/en
cd ..\..
```

Expected: ~719 files downloaded into `data\raw\docs\source\en\`.

---

## Phase 4: Chunk the Corpus

This parses all markdown files, cleans HF-specific artifacts (frontmatter, MDX tags, `[[autodoc]]` directives), and produces chunked JSON files.

```powershell
python -m src.ingest.chunk_docs --chunk-sizes 256 512
```

Expected output:

```
Found 724 markdown files in data/raw
Parsed 720 documents (4 skipped)
  → 24775 chunks @ size 256 → data\processed\chunks_256.json
  → 11793 chunks @ size 512 → data\processed\chunks_512.json
✅ Ingestion complete.
```

---

## Phase 5: Build Vector Indexes

This embeds all chunks with two different models (MiniLM and BGE-small) and stores them in ChromaDB collections. Builds 3 unique indexes (Config D reuses Config C's index since reranking is query-time only).

```powershell
python -m src.ingest.build_all_indexes
```

> **If you hit a UnicodeDecodeError:**
> ```
> UnicodeDecodeError: 'charmap' codec can't decode byte 0x90
> ```
> **Fix:** Open `src\ingest\build_index.py`, find this line:
> ```python
> with open(chunks_path) as f:
> ```
> Change it to:
> ```python
> with open(chunks_path, encoding="utf-8") as f:
> ```
> Then re-run the command.

Expected output (takes ~15-25 minutes on CPU):

```
[all] 4 configs -> 3 unique index build(s)
[all] building 'minilm_256' ... 24775 chunks
[all] building 'minilm_512' ... 11793 chunks
[all] building 'bge_256' ... 24775 chunks
[all] done. built=3, skipped=0, total unique=3
```

On re-runs, already-built collections are skipped automatically. Use `--force` to rebuild.

---

## Phase 6: Test End-to-End RAG

### 6.1 — Config fix (one-time)

The default config points to a collection that doesn't exist. Open `config\default.yaml` and change:

```yaml
# BEFORE
collection: "hf_docs"

# AFTER
collection: "bge_256"
```

### 6.2 — Ask a question

```powershell
python -m src.generate.answer "How do I load a model in 4-bit?"
```

Expected: a ~10-30 second wait (first run, CPU inference), then an answer citing HF docs sources. The answer may contain minor hallucinations in API details — that's expected and is exactly what the eval harness will measure.

```
=== ANSWER ===
To load the quantized version of your model with BitsAndBytesConfig for 4-bit precision...
(answer continues with cited sources)

=== SOURCES ===
 - docs\source\en\quantization\bitsandbytes.md
 - docs\source\en\model_doc\florence2.md
 ...
```

---

## Phase 7: Build Gold Dataset (Next Step)

Generate draft Q&A pairs from the corpus using Mistral 7B, then hand-verify them.

```powershell
# Generate ~70 draft pairs (takes 30-60 min on CPU — run before bed)
python -m src.eval.build_gold --n 70
```

After generation, open `data\eval\gold_draft.jsonl` and manually review every pair:
- Fix incorrect answers
- Delete ambiguous or unanswerable questions
- Set `"verified": true` on keepers
- Target ~50 verified questions

See `docs\GOLD_DATASET.md` for the full verification methodology.

---

## Phase 8: Run the Eval Harness (After Gold Set is Ready)

```powershell
python -m src.eval.run_eval --config config/experiments.yaml
```

This runs all 4 configs (A/B/C/D) through RAGAS with the local Mistral judge. On CPU with ~50 questions, expect ~2-4 hours. Run overnight.

Results land in `results/eval_results.json` and `results/results_table.md`.

---

## Phase 9: Deploy to HF Spaces (After Eval is Done)

```powershell
# Test locally first
streamlit run app/streamlit_app.py
```

Then push to Hugging Face Spaces (free tier). Details in `README.md`.

---

## Quick Reference: Common Issues

| Issue | Fix |
|-------|-----|
| `python` not found | Use `py` instead (Windows Store alias conflict) |
| `Activate.ps1` won't run | `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| `ollama` not recognized | `$env:PATH += ";$env:LOCALAPPDATA\Programs\Ollama"` |
| `UnicodeDecodeError` on `build_index.py` | Add `encoding="utf-8"` to `open(chunks_path)` |
| `Collection [hf_docs] does not exist` | Change `collection: "hf_docs"` to `collection: "bge_256"` in `config\default.yaml` |
| Model download slow | Ollama downloads are large (2-4 GB each); use a stable connection |
| `pip` not found after activating venv | Make sure `.\.venv\Scripts\Activate.ps1` ran successfully (look for `(.venv)` in prompt) |
| Re-running `build_all_indexes` | Existing collections are skipped; use `--force` to rebuild |
| Need to add Ollama PATH in every new terminal | Add Ollama to system PATH permanently via Windows Settings > Environment Variables |

---

## Stack Summary

| Component | Tool | Cost |
|-----------|------|------|
| Generator LLM | Ollama — Phi-3-mini (3.8B, q4) | Free / local |
| Judge LLM | Ollama — Mistral 7B Instruct (q4) | Free / local |
| Judge Embeddings | Ollama — nomic-embed-text | Free / local |
| Retrieval Embeddings | sentence-transformers (BGE-small, MiniLM) | Free / local |
| Vector Store | ChromaDB (persistent, local) | Free / local |
| Eval Framework | RAGAS (with local Ollama judge override) | Free |
| Demo Hosting | Streamlit on HF Spaces | Free tier |
| CI/CD | GitHub Actions | Free tier |

---

## Pipeline Flow

```
HF Transformers docs (markdown, 719 files)
    │
    ▼  chunk_docs.py
chunks_256.json / chunks_512.json
    │
    ▼  build_all_indexes.py (build_index.py x3)
ChromaDB: minilm_256, minilm_512, bge_256
    │
    ▼  retriever.py (query → top-k chunks)
    │
    ▼  answer.py (chunks + question → Ollama Phi-3-mini → answer)
    │
    ▼  run_eval.py (RAGAS + Mistral judge → metrics)
    │
    ▼  results/eval_results.json + results_table.md
```

---

## What's Next After Setup

1. **Build the gold dataset** — `python -m src.eval.build_gold --n 70`, then hand-verify
2. **Run the eval harness** — `python -m src.eval.run_eval --config config/experiments.yaml`
3. **Judge validation** — hand-score 10 examples, report agreement % (see `docs/GOLD_DATASET.md`)
4. **Write tradeoffs analysis** — fill in `docs/TRADEOFFS.md` with your findings
5. **Deploy** — push Streamlit app to HF Spaces
6. **Polish README** — insert the results table and architecture diagram
7. **Init git + push** — `git init`, commit, push to GitHub

---

*Last updated: June 24, 2026*
*Session notes: Built and tested on Windows 11, Python 3.11.5, CPU-only (Intel Iris Xe), 16 GB RAM*