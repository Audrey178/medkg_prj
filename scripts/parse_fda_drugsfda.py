#!/usr/bin/env python3
"""Parse FDA Drugs@FDA into a benchmark-ready pickle format.

Joins Products.txt (drug names, ingredients) with Applications.txt (ApplType)
and Submissions.txt (approval dates) from the Products.zip archive.
Saves data/validation_sources/fda_approvals_parsed.pkl.
"""

import csv
import pickle
import sys
import zipfile
from datetime import date, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FDA_ZIP = PROJECT_ROOT / "data" / "validation_sources" / "phenopackets" / "data" / "validation_sources" / "drugsfda" / "Products.zip"
FDA_ZIP_ALT = PROJECT_ROOT / "data" / "validation_sources" / "drugsfda" / "Products.zip"
OUTPUT_PATH = PROJECT_ROOT / "data" / "validation_sources" / "fda_approvals_parsed.pkl"


def find_zip():
    for path in [FDA_ZIP, FDA_ZIP_ALT]:
        if path.exists():
            return path
    for p in (PROJECT_ROOT / "data").rglob("Products.zip"):
        return p
    return None


def read_tsv_from_zip(zf, filename):
    """Read a TSV file from zip, handle encoding issues."""
    raw = zf.read(filename)
    text = raw.decode("latin-1")  # FDA files use Windows encoding
    lines = text.splitlines()
    return list(csv.DictReader(lines, delimiter="\t"))


def parse_date(date_str: str) -> date | None:
    if not date_str or date_str.strip() == "":
        return None
    date_str = date_str.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y"):
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    return None


def main():
    zip_path = find_zip()
    if zip_path is None:
        print("ERROR: Products.zip not found.")
        sys.exit(1)

    print(f"Found Products.zip at: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        print(f"Zip contents: {zf.namelist()}")

        # 1. Read Applications.txt -> {ApplNo: ApplType}
        apps_rows = read_tsv_from_zip(zf, "Applications.txt")
        print(f"Applications.txt columns: {apps_rows[0].keys() if apps_rows else 'EMPTY'}")
        appl_type_map = {}
        for row in apps_rows:
            appl_no = row.get("ApplNo", "").strip()
            appl_type = row.get("ApplType", "").strip()
            if appl_no:
                appl_type_map[appl_no] = appl_type
        print(f"  {len(appl_type_map)} applications loaded")

        # 2. Read Submissions.txt -> {ApplNo: earliest_approval_date}
        # Filter for status=AP (approved) and SubmissionType=ORIG (original)
        sub_rows = read_tsv_from_zip(zf, "Submissions.txt")
        print(f"Submissions.txt columns: {sub_rows[0].keys() if sub_rows else 'EMPTY'}")
        approval_date_map = {}  # ApplNo -> earliest AP date
        for row in sub_rows:
            appl_no = row.get("ApplNo", "").strip()
            status = row.get("SubmissionStatus", "").strip()
            date_str = row.get("SubmissionStatusDate", "").strip()
            if not appl_no or status != "AP":
                continue
            d = parse_date(date_str)
            if d is not None:
                if appl_no not in approval_date_map or d < approval_date_map[appl_no]:
                    approval_date_map[appl_no] = d
        print(f"  {len(approval_date_map)} applications with approval dates")

        # 3. Read Products.txt -> drug info
        prod_rows = read_tsv_from_zip(zf, "Products.txt")
        print(f"Products.txt columns: {prod_rows[0].keys() if prod_rows else 'EMPTY'}")
        print(f"  {len(prod_rows)} product records")

    # Join: Products + Applications + Submissions
    drug_map = {}
    for row in prod_rows:
        drug_name = row.get("DrugName", "").strip()
        active_ingredient = row.get("ActiveIngredient", "").strip()
        appl_no = row.get("ApplNo", "").strip()

        if not drug_name:
            continue

        app_type = appl_type_map.get(appl_no, "")
        approval_date = approval_date_map.get(appl_no)

        key = drug_name.lower()
        if key not in drug_map:
            drug_map[key] = {
                "approval_date": approval_date,
                "active_ingredient": active_ingredient,
                "application_type": app_type,
            }
        else:
            existing = drug_map[key]
            # Keep earliest approval date
            if approval_date is not None:
                if existing["approval_date"] is None or approval_date < existing["approval_date"]:
                    existing["approval_date"] = approval_date
            # Prefer NDA over ANDA
            if app_type == "NDA" and existing["application_type"] != "NDA":
                existing["application_type"] = app_type
                existing["active_ingredient"] = active_ingredient

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(drug_map, f)

    # Summary stats
    with_date = sum(1 for v in drug_map.values() if v["approval_date"] is not None)
    nda_count = sum(1 for v in drug_map.values() if v["application_type"] == "NDA")
    anda_count = sum(1 for v in drug_map.values() if v["application_type"] == "ANDA")
    bla_count = sum(1 for v in drug_map.values() if v["application_type"] == "BLA")

    dates = [v["approval_date"] for v in drug_map.values() if v["approval_date"]]
    min_date = min(dates) if dates else None
    max_date = max(dates) if dates else None

    # Decade distribution
    decade_counts = {}
    for d in dates:
        decade = (d.year // 10) * 10
        decade_counts[decade] = decade_counts.get(decade, 0) + 1

    print(f"\n{'='*60}")
    print(f"FDA DRUGS@FDA PARSING SUMMARY")
    print(f"{'='*60}")
    print(f"Raw product records:           {len(prod_rows)}")
    print(f"Unique drugs (by name):        {len(drug_map)}")
    print(f"With approval date:            {with_date}")
    print(f"NDA (new drugs):               {nda_count}")
    print(f"ANDA (generics):               {anda_count}")
    print(f"BLA (biologics):               {bla_count}")
    print(f"Other app types:               {len(drug_map) - nda_count - anda_count - bla_count}")
    if min_date and max_date:
        print(f"Date range:                    {min_date} to {max_date}")
    print(f"\nApprovals by decade:")
    for decade in sorted(decade_counts):
        print(f"  {decade}s: {decade_counts[decade]:,}")
    print(f"{'='*60}")
    print(f"Output saved to: {OUTPUT_PATH}")

    # Show sample entries
    print(f"\nSample NDA entries with dates:")
    shown = 0
    for name, info in drug_map.items():
        if info["application_type"] == "NDA" and info["approval_date"] is not None:
            print(f"  {name}: date={info['approval_date']}, ingredient={info['active_ingredient'][:60]}")
            shown += 1
            if shown >= 10:
                break


if __name__ == "__main__":
    main()
