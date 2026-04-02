#!/usr/bin/env python3
from __future__ import annotations

import argparse
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_JSON_DIR = ROOT / "data" / "json_files"
FILTER_SCRIPT = Path(__file__).resolve().parent / "2_filter_records.py"
LLM_LABELS_CSV = ROOT / "data" / "json_files" / "filtered" / "all_abstracts_LLM.csv"
ANALYSIS_DIR = ROOT / "data" / "analysis"
DERIVED_DIR = ROOT / "derived_data"

ANALYSIS_CSV = ANALYSIS_DIR / "analysis_dataset.csv"
MINIMAL_EXPORT_CSV = DERIVED_DIR / "policy_claims_minimal.csv"

EXPECTED_COLUMNS = [
    "scopus_id",
    "doi",
    "title",
    "journal",
    "publication_year",
    "keywords",
    "abstract",
    "article_type",
    "corresponding_author_country",
    "cited_by_count",
    "llm_policy_claim",
]


def load_filter_module():
    if not FILTER_SCRIPT.exists():
        raise FileNotFoundError(f"Filter script not found: {FILTER_SCRIPT}")
    return SourceFileLoader("filter_records", str(FILTER_SCRIPT)).load_module()


def load_raw_records(in_dir: Path) -> pd.DataFrame:
    mod = load_filter_module()
    files = sorted(in_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON files found under {in_dir}")

    records = []
    for path in files:
        records.extend(mod.load_json(path))
    return pd.DataFrame.from_records(records)


def filter_records(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    mod = load_filter_module()
    kept, dropped, counts = mod.filter_df(df_raw.copy())
    kept["publication_year"] = pd.to_numeric(kept["publication_year"], errors="coerce")
    kept = kept[kept["publication_year"].between(1990, 2024, inclusive="both")].copy()
    return kept, dropped, dict(counts)


def merge_llm_labels(kept: pd.DataFrame, labels_csv: Path) -> pd.DataFrame:
    lab = pd.read_csv(labels_csv, usecols=["scopus_id", "doi", "llm_policy_claim"])

    lab_scopus = lab.dropna(subset=["scopus_id"]).drop_duplicates(subset=["scopus_id"])
    lab_doi = lab.dropna(subset=["doi"]).drop_duplicates(subset=["doi"])

    merged = kept.merge(
        lab_scopus[["scopus_id", "llm_policy_claim"]],
        how="left",
        on="scopus_id",
    )

    missing_mask = merged["llm_policy_claim"].isna()
    if missing_mask.any():
        merged = merged.merge(
            lab_doi[["doi", "llm_policy_claim"]].rename(
                columns={"llm_policy_claim": "llm_policy_claim_doi"}
            ),
            how="left",
            on="doi",
        )
        merged.loc[missing_mask, "llm_policy_claim"] = merged.loc[
            missing_mask, "llm_policy_claim_doi"
        ]
        merged = merged.drop(columns=["llm_policy_claim_doi"], errors="ignore")

    return merged


def normalize_keywords(x):
    if isinstance(x, (list, tuple, set)):
        return "; ".join(map(str, x))
    if pd.isna(x):
        return pd.NA
    s = str(x).strip()
    return s if s else pd.NA


def build_analysis_dataset(merged: pd.DataFrame) -> pd.DataFrame:
    df = merged[merged["llm_policy_claim"].notna()].copy().reset_index(drop=True)

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing expected columns: {missing}")

    df["doi"] = (
        df["doi"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"^https?://(dx\.)?doi\.org/", "", regex=True)
    )
    df["doi"] = df["doi"].replace({"": pd.NA, "nan": pd.NA, "none": pd.NA})

    df["title"] = df["title"].astype(str).str.strip()
    df["journal"] = df["journal"].astype(str).str.strip().str.title()
    df["publication_year"] = pd.to_numeric(df["publication_year"], errors="coerce").astype("Int64")
    df["keywords"] = df["keywords"].apply(normalize_keywords)
    df["abstract"] = df["abstract"].astype(str)
    df["article_type"] = df["article_type"].astype(str).str.strip().str.lower()
    df["corresponding_author_country"] = (
        df["corresponding_author_country"]
        .astype(str)
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "none": pd.NA})
        .fillna("UNKNOWN")
        .str.upper()
    )
    df["cited_by_count"] = pd.to_numeric(df["cited_by_count"], errors="coerce").astype("Int64")
    df["llm_policy_claim"] = df["llm_policy_claim"].astype(bool)

    df["claim"] = df["llm_policy_claim"].astype(int)
    df["doi_norm"] = df["doi"]
    df["abstract_word_count"] = df["abstract"].fillna("").str.split().str.len().astype("Int64")

    ordered_cols = EXPECTED_COLUMNS + ["claim", "doi_norm", "abstract_word_count"]
    return df[ordered_cols].copy()


def export_minimal(df: pd.DataFrame, export_path: Path) -> pd.DataFrame:
    out = df[
        [
            "doi",
            "title",
            "journal",
            "publication_year",
            "keywords",
            "corresponding_author_country",
            "llm_policy_claim",
        ]
    ].copy()
    out = (
        out.dropna(subset=["doi", "title"])
        .query("doi != ''")
        .drop_duplicates(subset=["doi"])
        .reset_index(drop=True)
    )
    export_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(export_path, index=False)
    return out


def save_outputs(df: pd.DataFrame, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the post-screening analysis dataset.")
    parser.add_argument("--raw-json-dir", type=Path, default=RAW_JSON_DIR)
    parser.add_argument("--labels-csv", type=Path, default=LLM_LABELS_CSV)
    parser.add_argument("--analysis-csv", type=Path, default=ANALYSIS_CSV)
    parser.add_argument("--minimal-export", type=Path, default=MINIMAL_EXPORT_CSV)
    args = parser.parse_args()

    df_raw = load_raw_records(args.raw_json_dir)
    print(f"Raw records: {len(df_raw)}")

    kept, dropped, counts = filter_records(df_raw)
    print(f"Kept after content filters and year window: {len(kept)}")
    print(f"Dropped during filtering: {len(dropped)}")
    print(f"Exclusion reasons: {counts}")

    merged = merge_llm_labels(kept, args.labels_csv)
    print(f"Missing LLM label after merge: {int(merged['llm_policy_claim'].isna().sum())}")

    analysis_df = build_analysis_dataset(merged)
    print(f"Analytic N: {len(analysis_df)}")
    print(f"Years: {int(analysis_df['publication_year'].min())}–{int(analysis_df['publication_year'].max())}")
    print(f"Rows with missing abstracts: {int(analysis_df['abstract'].isna().sum())}")

    save_outputs(analysis_df, args.analysis_csv)
    minimal_df = export_minimal(analysis_df, args.minimal_export)

    print(f"Wrote analysis dataset: {args.analysis_csv}")
    print(f"Wrote minimal export: {args.minimal_export} ({len(minimal_df)} rows)")



if __name__ == "__main__":
    main()
