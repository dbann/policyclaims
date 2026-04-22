#!/usr/bin/env python3
"""
Generates samples for manual human validation of the enriched analysis dataset.

A subcommand is required. Usage:

  python 6_make_review_sample.py design-stratified
      Draw 50 papers from each study design category in `design_combined`.
      Exports a CSV with blank columns for a reviewer to fill in.
      Output: table/manual_review_by_design_50_each.csv

  python 6_make_review_sample.py stratified
      Draw a stratified-by-year sample of 400 papers across 1990-2024.
      Produces a blinded Excel file (no LLM labels, for the human reviewer)
      and an internal file (with LLM labels, for comparison). Also runs a
      Spearman trend check to confirm the sample mirrors the full dataset.

Run after: 4_build_analysis_dataset.py and 5_add_study_design_and_topics.py
Run before: 7_concordance.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = ROOT / "data" / "analysis" / "analysis_dataset_enriched_v2.csv"
TABLE_DIR = ROOT / "table"


def _display_path(path: Path) -> str:
    """Return a repo-relative path string."""
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def load_dataset(input_csv: Path) -> pd.DataFrame:
    if not input_csv.exists():
        raise FileNotFoundError(
            f"Enriched dataset not found: {input_csv}\n"
            "Run code/4_build_analysis_dataset.py and code/5_add_study_design_and_topics.py first."
        )
    df = pd.read_csv(input_csv)
    if "publication_year" in df.columns:
        df["publication_year"] = pd.to_numeric(df["publication_year"], errors="coerce").astype("Int64")
    if "llm_policy_claim" in df.columns:
        df["llm_policy_claim"] = df["llm_policy_claim"].astype(bool)
    return df


def make_design_stratified_sample(
    df: pd.DataFrame,
    n_per_design: int = 50,
    seed: int = 123,
    output_csv: Path | None = None,
) -> pd.DataFrame:
    if "design_combined" not in df.columns:
        raise KeyError("Column 'design_combined' is required.")

    design_df = df.loc[df["design_combined"].notna()].copy()
    design_df = design_df.loc[design_df["design_combined"].astype(str).str.strip() != ""].copy()

    design_counts = design_df["design_combined"].value_counts().sort_index()
    too_small = design_counts[design_counts < n_per_design]
    if not too_small.empty:
        raise ValueError(
            "Not enough rows for some design categories: "
            + ", ".join(f"{k}={v}" for k, v in too_small.items())
        )

    # Sample from each design group without using groupby.apply, which drops
    # the grouping column in recent pandas versions.
    sampled_parts = [
        sub.sample(n=n_per_design, random_state=seed)
        for _, sub in design_df.groupby("design_combined")
    ]
    sampled = (
        pd.concat(sampled_parts)
        .reset_index(drop=False)
        .rename(columns={"index": "source_row_index"})
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )

    sampled.insert(0, "review_id", [f"DES-{i:03d}" for i in range(1, len(sampled) + 1)])
    sampled["manual_check1"] = ""
    sampled["manual_check2"] = ""

    cols = [
        c for c in [
            "review_id",
            "source_row_index",
            "scopus_id",
            "doi",
            "title",
            "journal",
            "publication_year",
            "design_strict",
            "design_keywords",
            "design_combined",
            "llm_policy_claim",
            "keywords",
            "abstract",
            "manual_check1",
            "manual_check2",
        ]
        if c in sampled.columns
    ]
    sampled = sampled[cols].copy()

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        sampled.to_csv(output_csv, index=False)

    return sampled


def make_stratified_sample(
    df: pd.DataFrame,
    n: int = 400,
    seed: int = 20260327,
    blinded_xlsx: Path | None = None,
    internal_xlsx: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_df = df.dropna(subset=["llm_policy_claim", "publication_year"]).copy()
    sample_df["publication_year"] = pd.to_numeric(sample_df["publication_year"], errors="coerce")
    sample_df = sample_df.dropna(subset=["publication_year"]).copy()
    sample_df["publication_year"] = sample_df["publication_year"].astype(int)
    sample_df["claim"] = sample_df["llm_policy_claim"].astype(bool)

    years = sorted(sample_df["publication_year"].unique())
    n_years = len(years)
    base_n = n // n_years
    remainder = n % n_years

    rng = np.random.default_rng(seed)
    extra_years = set(rng.choice(years, size=remainder, replace=False)) if remainder else set()
    target_n_by_year = {y: base_n + (1 if y in extra_years else 0) for y in years}

    too_small = {
        y: int((sample_df["publication_year"] == y).sum())
        for y in years
        if (sample_df["publication_year"] == y).sum() < target_n_by_year[y]
    }
    if too_small:
        raise ValueError(f"Not enough records in some years for the requested stratified sample: {too_small}")

    stratified_parts = [
        sub.sample(n=target_n_by_year[int(year)], random_state=seed)
        for year, sub in sample_df.groupby("publication_year")
    ]
    stratified = (
        pd.concat(stratified_parts)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )
    stratified.insert(0, "review_id", [f"S400-{i:03d}" for i in range(1, len(stratified) + 1)])

    blinded_cols = [
        c for c in [
            "review_id",
            "scopus_id",
            "doi",
            "title",
            "journal",
            "keywords",
            "abstract",
        ]
        if c in stratified.columns
    ]
    internal_cols = [
        c for c in [
            "review_id",
            "scopus_id",
            "doi",
            "title",
            "journal",
            "publication_year",
            "keywords",
            "abstract",
            "llm_policy_claim",
        ]
        if c in stratified.columns
    ]

    blinded = stratified[blinded_cols].copy()
    internal = stratified[internal_cols].copy()

    if blinded_xlsx is not None:
        blinded_xlsx.parent.mkdir(parents=True, exist_ok=True)
        blinded.to_excel(blinded_xlsx, index=False)
    if internal_xlsx is not None:
        internal_xlsx.parent.mkdir(parents=True, exist_ok=True)
        internal.to_excel(internal_xlsx, index=False)

    return blinded, internal


def trend_check(df_full: pd.DataFrame, df_sample: pd.DataFrame) -> pd.DataFrame:
    full = df_full.dropna(subset=["llm_policy_claim", "publication_year"]).copy()
    full["publication_year"] = pd.to_numeric(full["publication_year"], errors="coerce")
    full = full.dropna(subset=["publication_year"]).copy()
    full["publication_year"] = full["publication_year"].astype(int)
    full["claim"] = full["llm_policy_claim"].astype(bool)

    sample = df_sample.copy()
    sample["publication_year"] = pd.to_numeric(sample["publication_year"], errors="coerce")
    sample = sample.dropna(subset=["publication_year"]).copy()
    sample["publication_year"] = sample["publication_year"].astype(int)
    sample["claim"] = sample["llm_policy_claim"].astype(bool)

    full_yearly = (
        full.groupby("publication_year")
        .agg(n_full=("claim", "size"), pct_full=("claim", lambda x: 100 * x.mean()))
        .reset_index()
    )
    sample_yearly = (
        sample.groupby("publication_year")
        .agg(n_sample=("claim", "size"), pct_sample=("claim", lambda x: 100 * x.mean()))
        .reset_index()
    )
    return full_yearly.merge(sample_yearly, on="publication_year", how="left")


def cmd_design_stratified(args: argparse.Namespace) -> None:
    df = load_dataset(args.input_csv)
    output_csv = args.output_csv or (TABLE_DIR / "manual_review_by_design_50_each.csv")
    sampled = make_design_stratified_sample(
        df,
        n_per_design=args.n_per_design,
        seed=args.seed,
        output_csv=output_csv,
    )
    print(f"Exported {len(sampled)} rows to: {_display_path(output_csv)}")
    print("\nCounts by design in exported sample:")
    print(sampled["design_combined"].value_counts().sort_index().to_string())


def cmd_stratified(args: argparse.Namespace) -> None:
    df = load_dataset(args.input_csv)
    blinded_path = args.blinded_xlsx or (TABLE_DIR / "supp_stratified_sample_400_blinded.xlsx")
    internal_path = args.internal_xlsx or (TABLE_DIR / "supp_stratified_sample_400_internal.xlsx")
    blinded, internal = make_stratified_sample(
        df,
        n=args.n,
        seed=args.seed,
        blinded_xlsx=blinded_path,
        internal_xlsx=internal_path,
    )
    trend = trend_check(df, internal)
    full_rho, full_p = spearmanr(trend["publication_year"], trend["pct_full"])
    sample_rho, sample_p = spearmanr(trend["publication_year"], trend["pct_sample"])

    trend_csv = args.trend_csv or (TABLE_DIR / "supp_stratified_sample_400_trend_check.csv")
    trend.to_csv(trend_csv, index=False)

    print(f"Saved blinded review file: {_display_path(blinded_path)}")
    print(f"Saved internal check file: {_display_path(internal_path)}")
    print(f"Saved trend check: {_display_path(trend_csv)}")
    print(f"Blinded rows: {len(blinded)}")
    print(f"Internal rows: {len(internal)}")
    print(f"Full analytic sample trend: rho={full_rho:.3f}, p={full_p:.4g}")
    print(f"Stratified sample trend: rho={sample_rho:.3f}, p={sample_p:.4g}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate review samples from the enriched analysis dataset.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser(
        "design-stratified",
        help="Create a manual review sample with 50 papers from each design_combined category.",
    )
    p1.add_argument("--input-csv", type=Path, default=INPUT_CSV)
    p1.add_argument("--output-csv", type=Path, default=None)
    p1.add_argument("--n-per-design", type=int, default=50)
    p1.add_argument("--seed", type=int, default=123)
    p1.set_defaults(func=cmd_design_stratified)

    p2 = sub.add_parser("stratified", help="Create a stratified 400-record blinded review sample.")
    p2.add_argument("--input-csv", type=Path, default=INPUT_CSV)
    p2.add_argument("--blinded-xlsx", type=Path, default=None)
    p2.add_argument("--internal-xlsx", type=Path, default=None)
    p2.add_argument("--trend-csv", type=Path, default=None)
    p2.add_argument("--n", type=int, default=400)
    p2.add_argument("--seed", type=int, default=20260327)
    p2.set_defaults(func=cmd_stratified)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()