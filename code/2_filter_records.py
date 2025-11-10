#!/usr/bin/env python3
"""
Standalone filter for abstract records.

Usage:
  # Filter a single JSON file
  python filter_records.py --in json_files/all_abstracts.json --out json_files/all_abstracts.filtered.json --log json_files/excluded_records.csv

  # Or filter every *.json in a directory (writes *.filtered.json alongside originals)
  python3 '2_filter_records.py' --dir json_files --log json_files/excluded_records.csv

Rules:
- Drop records with empty abstracts
- Drop commentaries/replies/letters/editorials/corrections (via doc type + regex on title/abstract)
- Drop review articles (including systematic, scoping, umbrella, narrative, rapid reviews, meta-analyses, etc.) via doc type and title regex
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

COMMENTARY_TITLE_RE = re.compile(r"\b(comment(ary)?|authors?\s*reply|response|letter)\b[:\s-]?", re.I)
COMMENTARY_ABS_RE   = re.compile(r"\b(this\s+commentary|authors?\s+reply|in\s+response\s+to)\b", re.I)
REVIEW_TITLE_RE = re.compile(
    r"(\b(systematic|scoping|umbrella|narrative|rapid)\s+review(s)?\b)"
    r"|\bmeta-?analysis(es)?\b"
    r"|\b(review|reviews)\b",
    re.I,
)

DOCTYPE_EXCLUDE = {
    "comment", "commentary", "reply", "letter", "editorial",
    "author reply", "authors reply", "authors’ reply",
    "correction", "erratum", "retraction", "news", "perspective",
    # reviews
    "review", "systematic review", "scoping review", "umbrella review",
    "narrative review", "rapid review", "meta-analysis", "meta analysis",
}

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    return re.sub(r"\s+", " ", s).strip()

def should_exclude_row(row: Dict[str, Any]) -> tuple[bool, str]:
    title = _norm(row.get("title", ""))
    abstract = _norm(row.get("abstract", ""))
    keywords = _norm(row.get("keywords", ""))
    if not abstract or len(abstract.split()) == 0:
        return True, "empty_abs"
    doc_type = _norm(row.get("document_type", row.get("doctype", row.get("type", "")))).lower()
    if any(dt in doc_type for dt in DOCTYPE_EXCLUDE):
        return True, "doc_type"
    if REVIEW_TITLE_RE.search(title) or REVIEW_TITLE_RE.search(keywords):
        return True, "review_term"
    if COMMENTARY_TITLE_RE.search(title):
        return True, "title_commentary"
    if COMMENTARY_ABS_RE.search(abstract):
        return True, "abs_commentary"
    return False, ""

def filter_df(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, Counter]:
    if df.empty:
        return df.copy(), df.copy(), Counter()
    reasons = []
    keep_mask = []
    for _, r in df.iterrows():
        drop, reason = should_exclude_row(r.to_dict())
        reasons.append(reason if drop else "")
        keep_mask.append(not drop)
    df["_excl_reason"] = reasons
    kept = df[pd.Series(keep_mask, index=df.index)].copy()
    dropped = df[~pd.Series(keep_mask, index=df.index)].copy()
    counts = Counter([r for r in reasons if r])
    return kept.drop(columns=["_excl_reason"], errors="ignore"), dropped, counts

def load_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "records" in data and isinstance(data["records"], list):
        return data["records"]
    if isinstance(data, list):
        return data
    return [data]

def save_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

def append_log_csv(dropped: pd.DataFrame, log_path: Path) -> None:
    if dropped.empty:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cols = [c for c in ["scopus_id", "title", "document_type", "abstract", "_excl_reason"] if c in dropped.columns]
    mode = "a" if log_path.exists() else "w"
    header = not log_path.exists()
    dropped[cols].to_csv(log_path, index=False, mode=mode, header=header)

def process_file(in_path: Path, out_path: Path, log_path: Path | None) -> None:
    records = load_json(in_path)
    df = pd.DataFrame.from_records(records)
    kept, dropped, counts = filter_df(df)
    # write output
    save_jsonl(kept.to_dict(orient="records"), out_path)
    if log_path is not None:
        # add reason column back temporarily for logging
        if "_excl_reason" not in dropped.columns:
            dropped["_excl_reason"] = ""
        append_log_csv(dropped, log_path)
    summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "none"
    print(f"[filter] {in_path.name}: kept={len(kept)} dropped={len(dropped)} ({summary}) -> {out_path.name}")

def main():
    ap = argparse.ArgumentParser(description="Standalone filter for abstract records.")
    ap.add_argument("--in", dest="in_file", type=Path, help="Input JSON file (list of records).")
    ap.add_argument("--out", dest="out_file", type=Path, help="Output filtered JSON file.")
    ap.add_argument("--dir", dest="in_dir", type=Path, help="Process all *.json in this directory.")
    ap.add_argument("--log", dest="log_csv", type=Path, default=None, help="CSV log file for exclusions (appended).")
    args = ap.parse_args()

    def filtered_path(p: Path) -> Path:
        # Save to /filtered subdir within the parent directory
        filtered_dir = p.parent / "filtered"
        filtered_dir.mkdir(parents=True, exist_ok=True)
        return filtered_dir / p.with_suffix(".filtered.json").name

    if args.in_dir:
        files = sorted(args.in_dir.glob("*.json"))
        print(f"Found {len(files)} .json files in {args.in_dir}:")
        for p in files:
            print(f"  Processing: {p}")
        for p in files:
            out_p = filtered_path(p)
            process_file(p, out_p, args.log_csv)
        # Combine all filtered abstracts into one file
        filtered_dir = args.in_dir / "filtered"
        combined_records = []
        for filtered_file in sorted(filtered_dir.glob("*.filtered.json")):
            with filtered_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    combined_records.extend(data)
                elif isinstance(data, dict) and "records" in data:
                    combined_records.extend(data["records"])
        combined_path = filtered_dir / "all_abstracts.json"
        with combined_path.open("w", encoding="utf-8") as f:
            json.dump(combined_records, f, ensure_ascii=False, indent=2)
        print(f"[combine] Wrote {len(combined_records)} records to {combined_path}")
    else:
        if not args.in_file or not args.out_file:
            ap.error("Either --dir or both --in and --out must be provided.")
        # Override out_file to be in /filtered subdir
        out_p = filtered_path(args.out_file)
        process_file(args.in_file, out_p, args.log_csv)

if __name__ == "__main__":
    main()
