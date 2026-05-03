#!/usr/bin/env python3
"""
Append a structured entry to docs/decision_log.md.

Used by the audit-trail skill to enforce a paper trail of every code change,
decision, and audit finding. See .claude/skills/audit-trail/SKILL.md for
when and how to use.

Usage:
    python3 scripts/log_audit_entry.py code \
        --title "Add novelty audit script" \
        --files scripts/compute_novelty_audit.py \
        --change "New script computing per-triple novelty against Orphadata/HPOA" \
        --why "Paper needs triple-level 'exists nowhere else' quantification" \
        --verified "Ran on 1K sample, matches full_kg_novelty_stats totals" \
        --unverified "Not yet run on full 460K triples"

    python3 scripts/log_audit_entry.py decision \
        --title "PPTX vs Beamer for supervisor slides" \
        --options "LaTeX Beamer, python-pptx, Both" \
        --chosen "python-pptx" \
        --why "Supervisor needs to edit; pptx is editable, Beamer is not" \
        --rejected "Beamer rejected: compile step + read-only workflow" \
        --reversible "yes, can rebuild in Beamer from same content"

    python3 scripts/log_audit_entry.py audit \
        --title "Orphadata consistency metric" \
        --claimed "95.1%" \
        --computed "94.2%" \
        --source "deep_validation_analysis.json" \
        --status "corrected" \
        --impact "Paper draft must be updated; 95.1% inflated by 317 all-ages matches"
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_FILE = PROJECT_ROOT / "docs" / "decision_log.md"

HEADER = """# ChronoMedKG Decision & Audit Log

Append-only paper trail of every code change, architectural decision, and
metric audit. Maintained per the `audit-trail` skill.

Format: most recent entries at the bottom. Do not edit existing entries; add
corrections as new entries that reference the old one by timestamp.

---
"""


def ensure_header():
    """Create the log file with header if it doesn't exist."""
    if not LOG_FILE.exists():
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        LOG_FILE.write_text(HEADER)


def append(entry: str):
    """Append an entry to the log file, separating with a blank line."""
    ensure_header()
    with open(LOG_FILE, "a") as f:
        f.write("\n" + entry.rstrip() + "\n")


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def fmt_code(args) -> str:
    files = args.files if args.files else "—"
    lines = [
        f"## [{now()}] Code — {args.title}",
        f"- **Files**: {files}",
        f"- **Change**: {args.change}",
        f"- **Why**: {args.why}",
    ]
    if args.verified:
        lines.append(f"- **Verified**: {args.verified}")
    if args.unverified:
        lines.append(f"- **Unverified**: {args.unverified}")
    return "\n".join(lines)


def fmt_decision(args) -> str:
    lines = [
        f"## [{now()}] Decision — {args.title}",
        f"- **Options**: {args.options}",
        f"- **Chosen**: {args.chosen}",
        f"- **Why**: {args.why}",
    ]
    if args.rejected:
        lines.append(f"- **Rejected because**: {args.rejected}")
    if args.reversible:
        lines.append(f"- **Reversible**: {args.reversible}")
    return "\n".join(lines)


def fmt_audit(args) -> str:
    lines = [
        f"## [{now()}] Audit — {args.title}",
        f"- **Claimed**: {args.claimed}",
        f"- **Computed**: {args.computed}",
        f"- **Source**: {args.source}",
        f"- **Status**: {args.status}",
    ]
    if args.impact:
        lines.append(f"- **Impact**: {args.impact}")
    return "\n".join(lines)


def fmt_raw(args) -> str:
    """Free-form entry when the structured types don't fit."""
    return f"## [{now()}] Note — {args.title}\n{args.body}"


def main():
    parser = argparse.ArgumentParser(description="Append an audit-trail entry.")
    sub = parser.add_subparsers(dest="type", required=True)

    c = sub.add_parser("code", help="Log a code change")
    c.add_argument("--title", required=True)
    c.add_argument("--files", help="Comma-separated file paths")
    c.add_argument("--change", required=True)
    c.add_argument("--why", required=True)
    c.add_argument("--verified", help="How correctness was checked")
    c.add_argument("--unverified", help="Known gaps / untested assumptions")

    d = sub.add_parser("decision", help="Log a strategic decision")
    d.add_argument("--title", required=True)
    d.add_argument("--options", required=True, help="Options considered, comma-separated")
    d.add_argument("--chosen", required=True)
    d.add_argument("--why", required=True)
    d.add_argument("--rejected", help="Why the rejected options lost")
    d.add_argument("--reversible", help="yes/no plus cost to undo")

    a = sub.add_parser("audit", help="Log a metric audit finding")
    a.add_argument("--title", required=True)
    a.add_argument("--claimed", required=True)
    a.add_argument("--computed", required=True)
    a.add_argument("--source", required=True, help="File or script that produced the number")
    a.add_argument("--status", required=True, choices=["verified", "corrected", "discrepant", "stale"])
    a.add_argument("--impact", help="Does this change any paper claim?")

    n = sub.add_parser("note", help="Free-form entry")
    n.add_argument("--title", required=True)
    n.add_argument("--body", required=True)

    args = parser.parse_args()

    formatters = {
        "code": fmt_code,
        "decision": fmt_decision,
        "audit": fmt_audit,
        "note": fmt_raw,
    }
    entry = formatters[args.type](args)
    append(entry)
    print(f"Logged {args.type} entry to {LOG_FILE}")


if __name__ == "__main__":
    main()
