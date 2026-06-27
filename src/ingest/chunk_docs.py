"""
Ingest + chunk the Hugging Face Transformers documentation.

Parses markdown files, strips MDX/frontmatter artifacts, and chunks them at
configurable sizes with metadata preserved for each chunk.

Usage:
    python -m src.ingest.chunk_docs                          # default sizes 256 and 512
    python -m src.ingest.chunk_docs --chunk-sizes 256 512 1024
    python -m src.ingest.chunk_docs --raw-dir data/raw --processed-dir data/processed
"""

import argparse
import json
import random
import re
from pathlib import Path
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from src.utils import get_logger

log = get_logger("chunk_docs")


# ─── Config ───────────────────────────────────────────────────────────────────

DEFAULT_RAW_DIR = "data/raw"
DEFAULT_PROCESSED_DIR = "data/processed"
DEFAULT_CHUNK_SIZES = [256, 512]
CHUNK_OVERLAP_RATIO = 0.2  # 20% overlap relative to chunk size
MIN_DOC_CHARS = 100        # skip docs shorter than this (stubs / index pages)

# Files/dirs to skip (non-content / build files only).
# NOTE: we deliberately do NOT skip model_doc/ — many evaluation questions are
# sourced from model cards, so they must be in the corpus for reproducibility.
SKIP_PATTERNS = [
    "_toctree.yml",
    "_config.py",
    "_redirects.yml",
]


# ─── Parsing ──────────────────────────────────────────────────────────────────

def extract_title(content: str, filename: str) -> str:
    """Extract document title from the first H1 header, or fall back to filename."""
    match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return filename.replace(".md", "").replace(".mdx", "").replace("_", " ").title()


def extract_section_header(text: str) -> Optional[str]:
    """Extract the last section header (## or ###) found within a chunk."""
    headers = re.findall(r"^#{2,3}\s+(.+)$", text, re.MULTILINE)
    return headers[-1].strip() if headers else None


def clean_markdown(content: str) -> str:
    """Strip frontmatter, HTML tags, and common MDX artifacts."""
    content = re.sub(r"^---\n.*?\n---\n", "", content, flags=re.DOTALL)      # YAML frontmatter
    content = re.sub(r"<!--.*?-->", "", content, flags=re.DOTALL)            # HTML comments
    content = re.sub(r"^import\s+.*$", "", content, flags=re.MULTILINE)      # MDX imports
    content = re.sub(r"<(?:Tip|Warning|Note|FrameworkSwitch)[^>]*>", "", content)
    content = re.sub(r"</(?:Tip|Warning|Note|FrameworkSwitch)>", "", content)
    content = re.sub(r"</?(?:br|hr|div|span|img|a)[^>]*>", "", content)      # inline HTML
    content = re.sub(r"\[\[autodoc\]\].*$", "", content, flags=re.MULTILINE) # HF autodoc markers
    content = re.sub(r"\n{3,}", "\n\n", content)                             # collapse blank lines
    return content.strip()


def should_skip(filepath: Path) -> bool:
    """Check if a file matches any skip pattern."""
    path_str = str(filepath)
    return any(pattern in path_str for pattern in SKIP_PATTERNS)


# ─── Chunking ─────────────────────────────────────────────────────────────────

def chunk_document(content: str, chunk_size: int, chunk_overlap: int,
                   source_file: str, doc_title: str) -> list[dict]:
    """Split a document into chunks, each carrying source metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,  # character-based; close enough for this corpus
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for i, text in enumerate(splitter.split_text(content)):
        chunks.append({
            "id": f"{source_file}::chunk_{i}",
            "text": text,
            "metadata": {
                "source_file": source_file,
                "doc_title": doc_title,
                "section_header": extract_section_header(text),
                "chunk_index": i,
                "chunk_size_config": chunk_size,
                "char_count": len(text),
            },
        })
    return chunks


# ─── Main pipeline ────────────────────────────────────────────────────────────

def ingest_and_chunk(raw_dir: str = DEFAULT_RAW_DIR,
                     processed_dir: str = DEFAULT_PROCESSED_DIR,
                     chunk_sizes: Optional[list[int]] = None) -> dict[int, list[dict]]:
    """
    Scan raw_dir for markdown, parse + clean each file, chunk at each requested
    size, and write data/processed/chunks_<size>.json. Returns a dict mapping
    chunk_size -> list of chunk dicts.
    """
    if chunk_sizes is None:
        chunk_sizes = DEFAULT_CHUNK_SIZES

    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)

    md_files = sorted(
        [f for f in raw_path.rglob("*.md") if not should_skip(f)]
        + [f for f in raw_path.rglob("*.mdx") if not should_skip(f)]
    )

    if not md_files:
        log.error("No .md or .mdx files found in %s", raw_dir)
        log.error("Clone the HF docs first (see README, 'Clone the corpus'):")
        log.error("  git clone --depth 1 --filter=blob:none --sparse "
                  "https://github.com/huggingface/transformers.git %s", raw_dir)
        log.error("  cd %s && git sparse-checkout set docs/source/en", raw_dir)
        return {}

    log.info("Found %d markdown files in %s", len(md_files), raw_dir)

    documents = []
    skipped = 0
    for filepath in tqdm(md_files, desc="Parsing docs"):
        try:
            content = filepath.read_text(encoding="utf-8")
            cleaned = clean_markdown(content)
            if len(cleaned) < MIN_DOC_CHARS:
                skipped += 1
                continue
            documents.append({
                "content": cleaned,
                "source_file": str(filepath.relative_to(raw_path)),
                "title": extract_title(content, filepath.name),
            })
        except Exception as e:  # noqa: BLE001
            log.warning("Failed to parse %s: %s", filepath, e)
            skipped += 1

    log.info("Parsed %d documents (%d skipped)", len(documents), skipped)

    all_results = {}
    for chunk_size in chunk_sizes:
        chunk_overlap = int(chunk_size * CHUNK_OVERLAP_RATIO)
        all_chunks = []
        for doc in tqdm(documents, desc=f"Chunking @ {chunk_size} chars"):
            all_chunks.extend(chunk_document(
                content=doc["content"],
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                source_file=doc["source_file"],
                doc_title=doc["title"],
            ))
        output_file = processed_path / f"chunks_{chunk_size}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)
        all_results[chunk_size] = all_chunks
        log.info("Wrote %d chunks @ size %d -> %s", len(all_chunks), chunk_size, output_file)

    return all_results


def print_sample_chunks(chunks: list[dict], n: int = 5) -> None:
    """Print a few random chunks for a quick sanity check."""
    for i, chunk in enumerate(random.sample(chunks, min(n, len(chunks))), 1):
        text = chunk["text"]
        print(f"\n{'='*60}")
        print(f"Sample {i} | {chunk['metadata']['source_file']}")
        print(f"Section: {chunk['metadata']['section_header']}")
        print(f"Chars: {chunk['metadata']['char_count']}")
        print("─" * 60)
        print(text[:300] + ("..." if len(text) > 300 else ""))


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest and chunk HF Transformers docs")
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    parser.add_argument("--processed-dir", default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--chunk-sizes", nargs="+", type=int, default=DEFAULT_CHUNK_SIZES)
    parser.add_argument("--sample", type=int, default=5, help="sample chunks to print per size")
    args = parser.parse_args()

    results = ingest_and_chunk(args.raw_dir, args.processed_dir, args.chunk_sizes)
    for size, chunks in results.items():
        print(f"\n{'='*60}\nSAMPLE CHUNKS @ size {size}")
        print_sample_chunks(chunks, n=args.sample)
    if results:
        log.info("Ingestion complete. Files saved to %s/", args.processed_dir)