"""
BioASQ snippet loader for Nhánh B (VectorStore).

Reads a BioASQ Task B JSON file and produces one VectorChunk per snippet.
This is an independent read path from Agent 1 / BioASQProfile — Agent 1
builds BioASQProfile to feed Nhánh A (KG pipeline), this loader only needs
the raw text+offset to embed into the VectorStore. No AgentResult, no
EvidenceTier, no credibility_score needed here.
"""

from __future__ import annotations

import json
from pathlib import Path

from core.models import VectorChunk


def load_bioasq_snippets(json_path: str) -> list[VectorChunk]:
    """Load BioASQ snippets from a Task B JSON file.

    Each snippet in each question becomes one VectorChunk. BioASQ snippets
    are verbatim quotes from PubMed abstracts/titles — no cleaning needed.

    chunk_id format: "BIOASQ:PMID:{pmid}:snip_{i:02d}"
      where i resets to 0 for each question's snippet list.
    """
    path = Path(json_path)
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)

    chunks: list[VectorChunk] = []
    for question in data.get("questions", []):
        snippets = question.get("snippets", [])
        for i, snip in enumerate(snippets):
            doc_url = snip.get("document", "")
            pmid = doc_url.rsplit("/", 1)[-1] if doc_url else "unknown"

            chunk = VectorChunk(
                chunk_id=f"BIOASQ:PMID:{pmid}:snip_{i:02d}",
                source_type="bioasq_snippet",
                source_name=f"PMID:{pmid}",
                section_heading=snip.get("beginSection", ""),
                text=snip.get("text", ""),
                char_start=snip.get("offsetInBeginSection", 0),
                char_end=snip.get("offsetInEndSection", 0),
            )
            chunks.append(chunk)

    return chunks


if __name__ == "__main__":
    import sys

    json_path = sys.argv[1] if len(sys.argv) > 1 else "data/bioasq/training14b.json"
    chunks = load_bioasq_snippets(json_path)
    print(f"Total BioASQ chunks: {len(chunks)}")
    if chunks:
        print(f"Sample chunk[0]: id={chunks[0].chunk_id!r} section={chunks[0].section_heading!r}")
        print(f"  text preview: {chunks[0].text[:100]!r}")
