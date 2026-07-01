"""
Script ingest một lần để build VectorStore (Nhánh B).

Steps:
  1. Chunk 4 textbooks → list[VectorChunk]
  2. Load BioASQ snippets → list[VectorChunk]
  3. Merge + embed into VectorStore (FAISS + BioLORD-2023-C)
  4. Persist to data/vector_store/
  5. Print stats()

Chạy:
  python retrieval/build_vector_store.py
  python retrieval/build_vector_store.py --bioasq data/bioasq/training14b.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from RAG.bioasq_snippet_loader import load_bioasq_snippets
from RAG.textbook_chunker import TARGET_FILES, chunk_textbook, detect_chunking_strategy
from RAG.vector_store import VectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger("build_vector_store")

DEFAULT_BIOASQ = "data/bioasq/training14b.json"
DEFAULT_OUTPUT = "data/vector_store/medkg_vectorstore.faiss"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MEDKG VectorStore")
    parser.add_argument("--bioasq", default=DEFAULT_BIOASQ,
                        help=f"Path to BioASQ JSON file (default: {DEFAULT_BIOASQ})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help=f"Output path for FAISS index (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--skip-bioasq", action="store_true",
                        help="Skip BioASQ snippets (only chunk textbooks)")
    args = parser.parse_args()

    all_chunks = []

    # Step 1: Chunk textbooks
    logger.info("=== Step 1: Chunking textbooks ===")
    for name, path in TARGET_FILES.items():
        if not path.exists():
            logger.warning("Textbook not found, skipping: %s", path)
            continue
        strategy = detect_chunking_strategy(str(path))
        chunks = chunk_textbook(str(path), strategy)
        logger.info("  %s: %d chunks (strategy=%s)", name, len(chunks), strategy["strategy"])
        all_chunks.extend(chunks)
    logger.info("Textbook total: %d chunks", len(all_chunks))

    # Step 2: Load BioASQ snippets
    if not args.skip_bioasq:
        logger.info("=== Step 2: Loading BioASQ snippets from %s ===", args.bioasq)
        bioasq_path = Path(args.bioasq)
        if not bioasq_path.exists():
            logger.warning("BioASQ file not found: %s — skipping", bioasq_path)
        else:
            bioasq_chunks = load_bioasq_snippets(str(bioasq_path))
            logger.info("  BioASQ snippets: %d chunks", len(bioasq_chunks))
            all_chunks.extend(bioasq_chunks)
    else:
        logger.info("=== Step 2: Skipping BioASQ snippets (--skip-bioasq) ===")

    logger.info("Total chunks to embed: %d", len(all_chunks))

    # Step 3: Embed into VectorStore
    logger.info("=== Step 3: Embedding into VectorStore ===")
    vs = VectorStore()
    vs.add_chunks(all_chunks)

    # Step 4: Persist
    logger.info("=== Step 4: Persisting to %s ===", args.output)
    vs.persist(args.output)

    # Step 5: Stats
    logger.info("=== Step 5: Stats ===")
    stats = vs.stats()
    print("\nVectorStore stats:")
    print(f"  total_chunks   : {stats['total_chunks']}")
    for stype, count in stats["by_source_type"].items():
        print(f"  {stype:<20}: {count}")
    print(f"\nIndex saved to: {args.output}")
    print(f"Metadata saved to: {args.output}.meta.pkl")


if __name__ == "__main__":
    main()
