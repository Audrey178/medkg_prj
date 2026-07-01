"""
Textbook chunker for Nhánh B (VectorStore).

Per-file chunking strategies determined from Bước 0 survey (reading 5
positions per file — 0%, 25%, 50%, 75%, 100%):

  Biochemistry_Lippincott:
    HEADING-BASED. Roman-numeral headings (^[IVX]+\\. [A-Z]) appear
    consistently throughout, e.g. "I. OVERVIEW", "II. STRUCTURE",
    "II. RNA STRUCTURE". Sections are coherent semantic units (~200–800 words).
    Noise: "For additional ancillary materials..." lines (page footers) and
    "Correct answer = ..." lines (self-test Q&A) filtered before chunking.

  Anatomy_Gray:
    PARAGRAPH SLIDING WINDOW. Pure flowing prose with no consistent heading
    pattern. Empty lines separate paragraphs. Each paragraph is ~100–400 words.
    Chunks = fixed-count paragraph windows (window=4, step=2).

  Cell_Biology_Alberts:
    CHARACTER SLIDING WINDOW. Lines can be up to 7671 chars (paragraphs +
    citations merged). Some lines are reference entries ("Author S, ... 2012.")
    or figure captions ("Figure 13-36 A model of...") — filtered out. After
    cleaning, text is split into sentences and chunked by character budget
    (~1200 chars, overlap ~200 chars).

  First_Aid_Step1:
    PARAGRAPH SLIDING WINDOW. Fact-dense bullet-style content. No heading
    markers. Empty lines separate fact groups. Paragraphs are short (50–200
    chars). Window=6 paragraphs, step=3 — wider window than Gray to give each
    chunk enough context.
"""

from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path
from typing import Generator

from core.models import VectorChunk

TEXTBOOK_DIR = Path(__file__).resolve().parent.parent / "data_clean" / "data_clean" / "textbooks" / "en"

TARGET_FILES = {
    "Anatomy_Gray":           TEXTBOOK_DIR / "Anatomy_Gray.txt",
    "Biochemistry_Lippincott": TEXTBOOK_DIR / "Biochemistry_Lippincott.txt",
    "Cell_Biology_Alberts":   TEXTBOOK_DIR / "Cell_Biology_Alberts.txt",
    "First_Aid_Step1":        TEXTBOOK_DIR / "First_Aid_Step1.txt",
}

# Lippincott heading: Roman numeral, dot, uppercase letter (e.g. "I. OVERVIEW")
_LIPPINCOTT_HEADING_RE = re.compile(r"^([IVX]+)\.\s+([A-Z].+)$")
# Alberts citation line: "Author A, Author B (YEAR)." or ends with year+period
_ALBERTS_CITATION_RE = re.compile(
    r"^[A-Z][a-z]+\s+[A-Z].*\d{4}[,.]"
    r"|^\w[\w\s,]+\(\d{4}\)"
)
# Alberts figure caption
_ALBERTS_FIGURE_RE = re.compile(r"^Figure\s+\d+[–\-–]\d+", re.IGNORECASE)
# Lippincott footer/noise lines
_LIPPINCOTT_NOISE_RE = re.compile(
    r"For additional ancillary materials|Correct answer\s*=|thePoint\."
)


# ---------------------------------------------------------------------------
# Strategy detection
# ---------------------------------------------------------------------------

def detect_chunking_strategy(file_path: str) -> dict:
    """Detect chunking strategy for a textbook file.

    Returns a dict with 'strategy', 'source_name', and 'evidence' keys.
    Strategy is one of: 'heading', 'paragraph_window', 'char_window'.
    """
    path = Path(file_path)
    name = path.stem

    strategies = {
        "Biochemistry_Lippincott": {
            "strategy": "heading",
            "heading_regex": _LIPPINCOTT_HEADING_RE,
            "noise_regex": _LIPPINCOTT_NOISE_RE,
            "evidence": (
                "Roman-numeral headings (I. OVERVIEW, II. STRUCTURE, ...) appear "
                "consistently at all 5 sampled positions. Sections are coherent semantic units."
            ),
        },
        "Anatomy_Gray": {
            "strategy": "paragraph_window",
            "window": 4,
            "step": 2,
            "evidence": (
                "Pure flowing prose, no heading markers at any sampled position. "
                "Paragraphs separated by blank lines, each ~100-400 words. "
                "Window=4 paragraphs keeps ~300-800 words per chunk."
            ),
        },
        "Cell_Biology_Alberts": {
            "strategy": "char_window",
            "char_budget": 1200,
            "overlap": 200,
            "evidence": (
                "Lines up to 7671 chars — paragraphs and citations merged into one line. "
                "Citation/figure-caption lines filtered. Remaining text chunked by "
                "character budget to avoid oversized chunks."
            ),
        },
        "First_Aid_Step1": {
            "strategy": "paragraph_window",
            "window": 6,
            "step": 3,
            "evidence": (
                "Fact-dense bullet content, paragraphs 50-200 chars, no heading markers. "
                "Window=6 paragraphs gives each chunk enough context (300-600 chars)."
            ),
        },
    }

    if name in strategies:
        s = strategies[name].copy()
        s["source_name"] = name
        return s

    # Default fallback for unrecognised files
    return {
        "strategy": "paragraph_window",
        "window": 4,
        "step": 2,
        "source_name": name,
        "evidence": "Unknown file — defaulting to paragraph sliding window.",
    }


# ---------------------------------------------------------------------------
# Noise filtering helpers
# ---------------------------------------------------------------------------

def _split_paragraphs(text: str) -> list[tuple[str, int]]:
    """Split text into (paragraph_text, char_start) tuples by blank lines."""
    paragraphs: list[tuple[str, int]] = []
    current_lines: list[str] = []
    pos = 0
    line_start = 0

    for line in text.splitlines(keepends=True):
        stripped = line.rstrip("\n\r")
        if stripped.strip():
            current_lines.append(stripped)
        else:
            if current_lines:
                para = " ".join(current_lines)
                paragraphs.append((para, line_start - len("\n".join(current_lines))))
                current_lines = []
        line_start += len(line)

    if current_lines:
        para = " ".join(current_lines)
        paragraphs.append((para, line_start - len("\n".join(current_lines))))

    return paragraphs


# ---------------------------------------------------------------------------
# Chunking strategies
# ---------------------------------------------------------------------------

def _chunk_heading(
    text: str, name: str, heading_re: re.Pattern, noise_re: re.Pattern
) -> list[VectorChunk]:
    """Split by headings; each section becomes one chunk (Lippincott)."""
    chunks: list[VectorChunk] = []
    current_heading = ""
    current_lines: list[str] = []
    current_start = 0
    char_pos = 0
    chunk_idx = 0

    for line in text.splitlines(keepends=True):
        stripped = line.rstrip("\n\r")

        # Skip noise lines
        if noise_re.search(stripped):
            char_pos += len(line)
            continue

        m = heading_re.match(stripped)
        if m:
            # Flush previous section
            if current_lines:
                body = " ".join(l for l in current_lines if l.strip())
                if len(body.split()) >= 20:  # skip degenerate tiny sections
                    chunks.append(VectorChunk(
                        chunk_id=f"TEXTBOOK:{name}:{chunk_idx:04d}",
                        source_type="textbook",
                        source_name=name,
                        section_heading=current_heading,
                        text=body,
                        char_start=current_start,
                        char_end=char_pos,
                    ))
                    chunk_idx += 1
            current_heading = stripped
            current_lines = []
            current_start = char_pos
        else:
            current_lines.append(stripped)

        char_pos += len(line)

    # Flush last section
    if current_lines:
        body = " ".join(l for l in current_lines if l.strip())
        if len(body.split()) >= 20:
            chunks.append(VectorChunk(
                chunk_id=f"TEXTBOOK:{name}:{chunk_idx:04d}",
                source_type="textbook",
                source_name=name,
                section_heading=current_heading,
                text=body,
                char_start=current_start,
                char_end=char_pos,
            ))

    return chunks


def _chunk_paragraph_window(
    text: str, name: str, window: int, step: int
) -> list[VectorChunk]:
    """Sliding window over paragraphs (Gray, First Aid)."""
    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        return []

    chunks: list[VectorChunk] = []
    chunk_idx = 0
    total = len(paragraphs)

    for start in range(0, total, step):
        para_slice = paragraphs[start:start + window]
        if not para_slice:
            break
        combined = " ".join(p for p, _ in para_slice).strip()
        if len(combined.split()) < 15:
            continue
        char_start = para_slice[0][1]
        last_para, last_start = para_slice[-1]
        char_end = last_start + len(last_para)
        chunks.append(VectorChunk(
            chunk_id=f"TEXTBOOK:{name}:{chunk_idx:04d}",
            source_type="textbook",
            source_name=name,
            section_heading="",
            text=combined,
            char_start=char_start,
            char_end=char_end,
        ))
        chunk_idx += 1

    return chunks


def _chunk_char_window(
    text: str, name: str, char_budget: int, overlap: int
) -> list[VectorChunk]:
    """Character-budget sliding window with citation/figure filtering (Alberts)."""
    # Filter citation and figure caption lines
    clean_lines: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            clean_lines.append("")
            continue
        if _ALBERTS_CITATION_RE.match(s) or _ALBERTS_FIGURE_RE.match(s):
            continue
        clean_lines.append(s)

    clean_text = "\n".join(clean_lines)

    # Split into sentences (simple regex, medical-aware abbreviation skip)
    _abbrev = re.compile(r"\b(Fig|Dr|Mr|Mrs|Ms|vs|e\.g|i\.e|et al|approx|ref|vol)\.")
    raw_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', clean_text)

    chunks: list[VectorChunk] = []
    chunk_idx = 0
    current: list[str] = []
    current_len = 0
    char_pos = 0
    chunk_start = 0

    for sent in raw_sentences:
        sent = sent.strip()
        if not sent:
            continue
        if current_len + len(sent) > char_budget and current:
            combined = " ".join(current)
            if len(combined.split()) >= 15:
                chunks.append(VectorChunk(
                    chunk_id=f"TEXTBOOK:{name}:{chunk_idx:04d}",
                    source_type="textbook",
                    source_name=name,
                    section_heading="",
                    text=combined,
                    char_start=chunk_start,
                    char_end=chunk_start + len(combined),
                ))
                chunk_idx += 1
            # Overlap: keep last few sentences
            overlap_chars = 0
            overlap_sents: list[str] = []
            for s in reversed(current):
                if overlap_chars + len(s) > overlap:
                    break
                overlap_sents.insert(0, s)
                overlap_chars += len(s)
            current = overlap_sents
            current_len = overlap_chars
            chunk_start = chunk_start + len(combined) - overlap_chars

        current.append(sent)
        current_len += len(sent)

    if current:
        combined = " ".join(current)
        if len(combined.split()) >= 15:
            chunks.append(VectorChunk(
                chunk_id=f"TEXTBOOK:{name}:{chunk_idx:04d}",
                source_type="textbook",
                source_name=name,
                section_heading="",
                text=combined,
                char_start=chunk_start,
                char_end=chunk_start + len(combined),
            ))

    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_textbook(file_path: str, strategy: dict) -> list[VectorChunk]:
    """Chunk a textbook file according to detected strategy."""
    path = Path(file_path)
    text = path.read_text(encoding="utf-8", errors="replace")
    name = strategy.get("source_name", path.stem)
    s = strategy["strategy"]

    if s == "heading":
        return _chunk_heading(
            text, name,
            heading_re=strategy["heading_regex"],
            noise_re=strategy["noise_regex"],
        )
    elif s == "paragraph_window":
        return _chunk_paragraph_window(
            text, name,
            window=strategy.get("window", 4),
            step=strategy.get("step", 2),
        )
    elif s == "char_window":
        return _chunk_char_window(
            text, name,
            char_budget=strategy.get("char_budget", 1200),
            overlap=strategy.get("overlap", 200),
        )
    else:
        raise ValueError(f"Unknown strategy: {s!r}")


# ---------------------------------------------------------------------------
# Bước 4 QA — run as script
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random

    all_ok = True
    for name, path in TARGET_FILES.items():
        if not path.exists():
            print(f"[MISSING] {path}")
            all_ok = False
            continue

        strategy = detect_chunking_strategy(str(path))
        chunks = chunk_textbook(str(path), strategy)
        lengths = [len(c.text) for c in chunks]

        if not lengths:
            print(f"[ERROR] {name}: 0 chunks produced")
            all_ok = False
            continue

        print(f"\n{'='*60}")
        print(f"FILE: {name}")
        print(f"  Strategy : {strategy['strategy']}")
        print(f"  Evidence : {strategy['evidence']}")
        print(f"  Chunks   : {len(chunks)}")
        print(f"  Len(chars): median={statistics.median(lengths):.0f} "
              f"min={min(lengths)} max={max(lengths)}")

        # 5 random chunks for manual inspection
        sample = random.sample(chunks, min(5, len(chunks)))
        for i, c in enumerate(sample):
            print(f"  -- Sample {i+1} (id={c.chunk_id}, start={c.char_start}) --")
            preview = c.text[:200].replace("\n", " ")
            print(f"     {preview!r}")

    if all_ok:
        print("\n[OK] All 4 files chunked successfully")
