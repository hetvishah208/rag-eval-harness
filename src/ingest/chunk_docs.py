"""
Phase 1: Ingest + Chunk HF Transformers Documentation
=====================================================
Parses markdown files from the HF Transformers docs, strips artifacts,
and chunks them at configurable sizes with metadata preservation.

Usage:
    python src/ingest.py                         # default: 256 and 512 chunk sizes
    python src/ingest.py --chunk-sizes 256 512 1024  # custom sizes
    python src/ingest.py --raw-dir /path/to/docs     # custom input path
"""

import argparse
import json
import re
from pathlib import Path
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm


# ─── Config ───────────────────────────────────────────────────────────────────

DEFAULT_RAW_DIR = "data/raw"
DEFAULT_PROCESSED_DIR = "data/processed"
DEFAULT_CHUNK_SIZES = [256, 512]
CHUNK_OVERLAP_RATIO = 0.2  # 20% overlap relative to chunk size

# Files/dirs to skip (auto-generated API refs, non-content files)
SKIP_PATTERNS = [
    "_toctree.yml",
    "_config.py",
    "model_doc/",       # auto-generated model cards — too noisy
]


# ─── Parsing ──────────────────────────────────────────────────────────────────

def extract_title(content: str, filename: str) -> str:
    """Extract document title from first H1 header, or fall back to filename."""
    match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return filename.replace(".md", "").replace(".mdx", "").replace("_", " ").title()


def extract_section_header(text: str) -> Optional[str]:
    """Extract the last section header (## or ###) found before this chunk."""
    headers = re.findall(r"^#{2,3}\s+(.+)$", text, re.MULTILINE)
    return headers[-1].strip() if headers else None


def clean_markdown(content: str) -> str:
    """Strip frontmatter, HTML tags, and common MDX artifacts."""
    # Remove YAML frontmatter
    content = re.sub(r"^---\n.*?\n---\n", "", content, flags=re.DOTALL)

    # Remove HTML comments
    content = re.sub(r"<!--.*?-->", "", content, flags=re.DOTALL)

    # Remove MDX import statements
    content = re.sub(r"^import\s+.*$", "", content, flags=re.MULTILINE)

    # Remove MDX/JSX component tags but keep inner text
    content = re.sub(r"<(?:Tip|Warning|Note|FrameworkSwitch)[^>]*>", "", content)
    content = re.sub(r"</(?:Tip|Warning|Note|FrameworkSwitch)>", "", content)

    # Remove inline HTML tags (keep content)
    content = re.sub(r"</?(?:br|hr|div|span|img|a)[^>]*>", "", content)

    # Remove [[autodoc]] directives (HF-specific auto-generated markers)
    content = re.sub(r"\[\[autodoc\]\].*$", "", content, flags=re.MULTILINE)

    # Collapse multiple blank lines
    content = re.sub(r"\n{3,}", "\n\n", content)

    return content.strip()


def should_skip(filepath: Path) -> bool:
    """Check if a file matches any skip pattern."""
    path_str = str(filepath)
    return any(pattern in path_str for pattern in SKIP_PATTERNS)


# ─── Chunking ─────────────────────────────────────────────────────────────────

def chunk_document(
    content: str,
    chunk_size: int,
    chunk_overlap: int,
    source_file: str,
    doc_title: str,
) -> list[dict]:
    """Split a document into chunks with metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,  # character-based; close enough to tokens for our purposes
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    splits = splitter.split_text(content)
    chunks = []

    for i, text in enumerate(splits):
        section_header = extract_section_header(text)
        chunks.append({
            "id": f"{source_file}::chunk_{i}",
            "text": text,
            "metadata": {
                "source_file": source_file,
                "doc_title": doc_title,
                "section_header": section_header,
                "chunk_index": i,
                "chunk_size_config": chunk_size,
                "char_count": len(text),
            },
        })

    return chunks


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def ingest_and_chunk(
    raw_dir: str = DEFAULT_RAW_DIR,
    processed_dir: str = DEFAULT_PROCESSED_DIR,
    chunk_sizes: list[int] = None,
) -> dict[int, list[dict]]:
    """
    Main ingestion pipeline.

    1. Scan raw_dir for .md/.mdx files
    2. Parse and clean each file
    3. Chunk at each specified size
    4. Save results to processed_dir

    Returns dict mapping chunk_size → list of chunk dicts.
    """
    if chunk_sizes is None:
        chunk_sizes = DEFAULT_CHUNK_SIZES

    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)

    # Collect all markdown files
    md_files = sorted(
        [f for f in raw_path.rglob("*.md") if not should_skip(f)]
        + [f for f in raw_path.rglob("*.mdx") if not should_skip(f)]
    )

    if not md_files:
        print(f"ERROR: No .md or .mdx files found in {raw_dir}")
        print(f"  Did you clone the HF docs? Run:")
        print(f"  git clone --depth 1 --filter=blob:none --sparse \\")
        print(f"    https://github.com/huggingface/transformers.git /tmp/hf-transformers")
        print(f"  cd /tmp/hf-transformers && git sparse-checkout set docs/source/en && cd -")
        print(f"  cp -r /tmp/hf-transformers/docs/source/en {raw_dir}")
        return {}

    print(f"Found {len(md_files)} markdown files in {raw_dir}")

    # Parse all documents
    documents = []
    skipped = 0
    for filepath in tqdm(md_files, desc="Parsing docs"):
        try:
            content = filepath.read_text(encoding="utf-8")
            cleaned = clean_markdown(content)

            # Skip very short docs (likely index files or stubs)
            if len(cleaned) < 100:
                skipped += 1
                continue

            relative_path = str(filepath.relative_to(raw_path))
            title = extract_title(content, filepath.name)
            documents.append({
                "content": cleaned,
                "source_file": relative_path,
                "title": title,
            })
        except Exception as e:
            print(f"  Warning: Failed to parse {filepath}: {e}")
            skipped += 1

    print(f"Parsed {len(documents)} documents ({skipped} skipped)")

    # Chunk at each size
    all_results = {}
    for chunk_size in chunk_sizes:
        chunk_overlap = int(chunk_size * CHUNK_OVERLAP_RATIO)
        all_chunks = []

        for doc in tqdm(documents, desc=f"Chunking @ {chunk_size} chars"):
            chunks = chunk_document(
                content=doc["content"],
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                source_file=doc["source_file"],
                doc_title=doc["title"],
            )
            all_chunks.extend(chunks)

        # Save to disk
        output_file = processed_path / f"chunks_{chunk_size}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)

        all_results[chunk_size] = all_chunks
        print(f"  → {len(all_chunks)} chunks @ size {chunk_size} → {output_file}")

    return all_results


def print_sample_chunks(chunks: list[dict], n: int = 5):
    """Print a few random chunks for sanity checking."""
    import random
    sample = random.sample(chunks, min(n, len(chunks)))
    for i, chunk in enumerate(sample):
        print(f"\n{'='*60}")
        print(f"Sample {i+1} | {chunk['metadata']['source_file']}")
        print(f"Section: {chunk['metadata']['section_header']}")
        print(f"Chars: {chunk['metadata']['char_count']}")
        print(f"{'─'*60}")
        print(chunk["text"][:300] + ("..." if len(chunk["text"]) > 300 else ""))


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest and chunk HF Transformers docs")
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR, help="Path to raw markdown files")
    parser.add_argument("--processed-dir", default=DEFAULT_PROCESSED_DIR, help="Output path for chunks")
    parser.add_argument(
        "--chunk-sizes", nargs="+", type=int, default=DEFAULT_CHUNK_SIZES,
        help="Chunk sizes to generate (default: 256 512)"
    )
    parser.add_argument("--sample", type=int, default=5, help="Number of sample chunks to print")
    args = parser.parse_args()

    results = ingest_and_chunk(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        chunk_sizes=args.chunk_sizes,
    )

    # Sanity check
    for size, chunks in results.items():
        print(f"\n{'='*60}")
        print(f"SAMPLE CHUNKS @ size {size}")
        print_sample_chunks(chunks, n=args.sample)

    print(f"\n✅ Ingestion complete. Files saved to {args.processed_dir}/")
