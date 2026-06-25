"""
Build PrimeKG SQLite index (one-time, chạy một lần).

Chuyển kg.csv (8.1M edges) → kg_index.db (~800MB disk, ~50MB RAM khi query).
Xử lý streaming row-by-row — không load toàn bộ CSV vào RAM.

Sau khi build xong, PrimeKGIndex.load() tự động dùng SQLite thay vì CSV/pickle.

Usage:
    python scripts/update/build_primekg_sqlite.py
    python scripts/update/build_primekg_sqlite.py --kg data/primekg/kg.csv
    python scripts/update/build_primekg_sqlite.py --kg /mnt/disk4/.../data/primekg/kg.csv
"""

import argparse
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger("build_primekg_sqlite")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kg", type=Path, default=None,
                        help="Path to kg.csv (default: data/primekg/kg.csv)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output path for SQLite DB (default: same dir as kg.csv)")
    args = parser.parse_args()

    from core.schema_alignment import PrimeKGIndex

    kg_path = args.kg or (PROJECT_ROOT / "data" / "primekg" / "kg.csv")
    if not kg_path.exists():
        logger.error("kg.csv not found: %s", kg_path)
        logger.error("Specify path with --kg /path/to/kg.csv")
        sys.exit(1)

    out_path = args.out or (kg_path.parent / "kg_index.db")

    if out_path.exists():
        logger.info("kg_index.db already exists at %s", out_path)
        logger.info("Delete it first to rebuild, or use --out to specify different path.")
        # Verify it works
        try:
            idx = PrimeKGIndex(kg_path)
            idx.load()
            if idx._sqlite_mode:
                logger.info("Existing SQLite index is valid — nothing to do.")
                return
        except Exception as e:
            logger.warning("Existing index invalid (%s) — rebuilding.", e)
            out_path.unlink()

    logger.info("Building PrimeKG SQLite index...")
    logger.info("  Input : %s (%.1f MB)", kg_path, kg_path.stat().st_size / 1e6)
    logger.info("  Output: %s", out_path)
    logger.info("  This takes ~3-8 minutes. RAM usage: <300MB.")

    t0 = time.monotonic()
    result_path = PrimeKGIndex.build_sqlite_index(kg_path, out_path)
    elapsed = time.monotonic() - t0

    logger.info("Done in %.0fs. SQLite index at: %s", elapsed, result_path)
    logger.info("Size: %.0f MB", result_path.stat().st_size / 1e6)

    # Verify
    logger.info("Verifying index loads correctly...")
    idx = PrimeKGIndex(kg_path)
    idx.load()
    assert idx._sqlite_mode, "SQLite mode not activated after build!"
    assert idx.is_loaded
    logger.info("Verification OK — SQLite mode active, ~50MB RAM.")
    logger.info("")
    logger.info("From now on, PrimeKGIndex.load() uses SQLite automatically.")
    logger.info("Re-run build_kg_from_bioasq.py without --skip-primekg.")


if __name__ == "__main__":
    main()
