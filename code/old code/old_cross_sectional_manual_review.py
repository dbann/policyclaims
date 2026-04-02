from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import pandas as pd


REVIEW_EXPORT_COLUMNS = [
    "review_id",
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
    "manual_is_cross_sectional",
    "manual_unclear",
    "manual_exclude",
    "manual_notes",
]


def _check_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def _parse_bool_like(series: pd.Series) -> pd.Series:
    truthy = {"1", "true", "t", "yes", "y", "confirmed"}
    falsy = {"0", "false", "f", "no", "n", "excluded"}

    def parse_one(value):
        if pd.isna(value):
            return pd.NA
        if isinstance(value, bool):
            return value

        text = str(value).strip().lower()
        if text == "":
            return pd.NA
        if text in truthy:
            return True
        if text in falsy:
            return False
        return pd.NA

    return series.apply(parse_one).astype("boolean")


def normalize_patterns(patterns: Iterable[str]) -> list[str]:
    out = []
    for p in patterns:
        p = p.replace("randomised", "randomi").replace("randomized", "randomi")
        p = p.replace("time-series", "time series").replace("meta-analysis", "meta analysis")
        out.append(p.lower())
    return out


def compile_patterns(patterns: Iterable[str]) -> list[re.Pattern]:
    return [re.compile(p, flags=re.IGNORECASE) for p in patterns]


STRICT_DESIGN_PATTERNS = {
    "Experimental": compile_patterns(
        [
            r"\brandomi[sz]ed controlled trial\b",
            r"\brandomi[sz]ed trial\b",
            r"\brandomi[sz]ed clinical trial\b",
            r"\bdouble[- ]blind\b",
            r"\bplacebo[- ]controlled trial\b",
        ]
    ),
    "Quasi-experimental": compile_patterns(
        [
            r"\bquasi[- ]experimental\b",
            r"\bquasi[- ]experimental study\b",
            r"\bnatural experiment\b",
            r"\bdifference[- ]in[- ]differences\b",
            r"\binterrupted time series\b",
            r"\bregression discontinuity\b",
            r"\bregression discontinuity design\b",
        ]
    ),
    "Cohort": compile_patterns(
        [
            r"\bcohort study\b",
            r"\bprospective cohort\b",
            r"\bretrospective cohort\b",
            r"\blongitudinal study\b",
            r"\blongitudinal analysis\b",
        ]
    ),
    "Case-control": compile_patterns(
        [
            r"\bcase[- ]control study\b",
            r"\bcase[- ]control\b",
            r"\bmatched case[- ]control\b",
        ]
    ),
    "Cross-sectional": compile_patterns(
        [
            r"\bcross[- ]sectional study\b",
            r"\bcross[- ]sectional survey\b",
            r"\bcross[- ]sectional\b",
        ]
    ),
    "Qualitative": compile_patterns(
        [
            r"\bqualitative study\b",
            r"\bqualitative analysis\b",
            r"\bqualitative approach\b",
        ]
    ),
    "Ecological / Time-series": compile_patterns(
        [
            r"\becological study\b",
            r"\becological design\b",
            r"\btime[- ]series analysis\b",
        ]
    ),
}

KEYWORD_DESIGN_PATTERNS = {
    "Experimental": [
        "randomi... controlled trial",
        "randomi... trial",
        "clinical trial",
        "rct",
        "pragmatic trial",
        "cluster randomi",
        "stepped wedge",
    ],
    "Quasi-experimental": [
        "quasi-experimental",
        "natural experiment",
        "difference in difference",
        "did design",
        "event study",
        "regression discontinuity",
        "interrupted time series",
        "its",
        "synthetic control",
    ],
    "Cohort": [
        "cohort study",
        "cohort studies",
        "prospective cohort",
        "retrospective cohort",
        "longitudinal study",
        "longitudinal",
        "longitudinal cohort",
        "follow-up study",
        "panel study",
    ],
    "Case-control": [
        "case-control",
        "matched case-control",
        "nested case-control",
    ],
    "Cross-sectional": [
        "cross-sectional",
        "prevalence study",
        "baseline survey",
    ],
    "Ecological / Time-series": [
        "ecological study",
        "time series",
        "ts analysis",
    ],
    "Qualitative": [
        "qualitative",
        "focus group",
        "thematic analysis",
        "ethnograph",
    ],
}
KEYWORD_DESIGN_PATTERNS = {
    k: normalize_patterns(v) for k, v in KEYWORD_DESIGN_PATTERNS.items()
}

DESIGN_HIERARCHY = [
    "Experimental",
    "Quasi-experimental",
    "Cohort",
    "Case-control",
    "Cross-sectional",
    "Ecological / Time-series",
    "Qualitative",
]


def parse_keywords(val) -> list[str]:
    if isinstance(val, str):
        toks = re.findall(r"'([^']+)'", val)
        if not toks:
            toks = re.split(r"[;,|]", val)
    elif isinstance(val, (list, tuple)):
        toks = [str(t) for t in val if pd.notna(t)]
    else:
        toks = []
    return [t.strip().lower() for t in toks if str(t).strip()]


def assign_design_strict(row: pd.Series) -> str:
    title = str(row.get("title", ""))
    abstract = str(row.get("abstract", ""))
    text_to_search = f"{title} {abstract}"

    for group in DESIGN_HIERARCHY:
        patterns = STRICT_DESIGN_PATTERNS.get(group, [])
        if any(p.search(text_to_search) for p in patterns):
            return group
    return "Other/None"


def assign_design_keywords(row: pd.Series) -> str:
    keywords = parse_keywords(row.get("keywords", ""))
    if not keywords:
        return "Other/None"

    for group in DESIGN_HIERARCHY:
        patterns = KEYWORD_DESIGN_PATTERNS.get(group, [])
        if any(pat in tok for tok in keywords for pat in patterns):
            return group
    return "Other/None"


def prepare_study_design_dataframe(input_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    _check_columns(df, ["title", "abstract", "keywords", "llm_policy_claim"])

    df = df.copy()
    df = df[df["llm_policy_claim"].notna()].reset_index(drop=True)
    df["design_strict"] = df.apply(assign_design_strict, axis=1)
    df["design_keywords"] = df.apply(assign_design_keywords, axis=1)
    df["design_combined"] = df["design_strict"]
    mask_other = df["design_combined"].eq("Other/None")
    df.loc[mask_other, "design_combined"] = df.loc[mask_other, "design_keywords"]
    return df


def export_cross_sectional_review_sample(
    df: pd.DataFrame,
    output_csv: str | Path = "../table/manual_review_cross_sectional_100.csv",
    n: int = 100,
    seed: int = 123,
    design_col: str = "design_combined",
    design_value: str = "Cross-sectional",
    claim_col: str = "llm_policy_claim",
) -> pd.DataFrame:
    """
    Randomly sample papers flagged as cross-sectional and export a CSV for
    manual confirmation.
    """

    required = [
        design_col,
        claim_col,
        "title",
        "abstract",
        "journal",
        "publication_year",
    ]
    _check_columns(df, required)

    cross_sectional = df.loc[df[design_col].eq(design_value)].copy()
    if cross_sectional.empty:
        raise ValueError(f"No rows found where {design_col!r} == {design_value!r}.")
    if len(cross_sectional) < n:
        raise ValueError(
            f"Requested n={n}, but only {len(cross_sectional)} rows were flagged as "
            f"{design_value!r}."
        )

    sampled = (
        cross_sectional.sample(n=n, random_state=seed)
        .reset_index(drop=False)
        .rename(columns={"index": "source_row_index"})
    )
    sampled.insert(0, "review_id", [f"CS-{i:03d}" for i in range(1, len(sampled) + 1)])

    for col in [
        "scopus_id",
        "doi",
        "keywords",
        "design_strict",
        "design_keywords",
        "design_combined",
    ]:
        if col not in sampled.columns:
            sampled[col] = pd.NA

    sampled["manual_is_cross_sectional"] = ""
    sampled["manual_unclear"] = ""
    sampled["manual_exclude"] = ""
    sampled["manual_notes"] = ""

    export_cols = [col for col in REVIEW_EXPORT_COLUMNS if col in sampled.columns]
    if "source_row_index" in sampled.columns and "source_row_index" not in export_cols:
        export_cols.insert(1, "source_row_index")

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    sampled[export_cols].to_csv(output_csv, index=False)

    claim_rate = _parse_bool_like(sampled[claim_col]).fillna(False).astype(bool).mean() * 100
    print(f"Exported {len(sampled)} papers to: {output_csv.resolve()}")
    print(f"Sampled set flagged as '{design_value}': {len(cross_sectional)} total candidates")
    print(f"Unreviewed claim rate in exported sample: {claim_rate:.1f}%")
    print(
        "Fill in 'manual_is_cross_sectional' and 'manual_unclear'. "
        "Use 'manual_exclude' for any paper you want removed from the confirmed set."
    )

    return sampled[export_cols].copy()


def calculate_confirmed_cross_sectional_claim_rate(
    reviewed_df: pd.DataFrame | None = None,
    review_csv: str | Path = "../table/manual_review_cross_sectional_100.csv",
    claim_col: str = "llm_policy_claim",
) -> tuple[pd.DataFrame, dict]:
    """
    Read the completed review CSV and estimate the claim rate using only papers
    confirmed as cross-sectional and not marked unclear/excluded.
    """

    if reviewed_df is None:
        reviewed_df = pd.read_csv(review_csv)
    else:
        reviewed_df = reviewed_df.copy()

    required = [
        claim_col,
        "manual_is_cross_sectional",
        "manual_unclear",
    ]
    _check_columns(reviewed_df, required)

    reviewed_df["manual_is_cross_sectional_bool"] = _parse_bool_like(
        reviewed_df["manual_is_cross_sectional"]
    )
    reviewed_df["manual_unclear_bool"] = _parse_bool_like(reviewed_df["manual_unclear"])

    if "manual_exclude" in reviewed_df.columns:
        reviewed_df["manual_exclude_bool"] = _parse_bool_like(reviewed_df["manual_exclude"])
    else:
        reviewed_df["manual_exclude_bool"] = False

    confirmed = reviewed_df.loc[
        reviewed_df["manual_is_cross_sectional_bool"].eq(True)
        & reviewed_df["manual_unclear_bool"].fillna(False).eq(False)
        & reviewed_df["manual_exclude_bool"].fillna(False).eq(False)
    ].copy()

    if confirmed.empty:
        raise ValueError(
            "No confirmed cross-sectional studies remain. "
            "Complete the manual review columns first."
        )

    confirmed[claim_col] = _parse_bool_like(confirmed[claim_col]).fillna(False).astype(bool)
    claim_rate = confirmed[claim_col].mean()

    summary = {
        "n_reviewed_rows": int(len(reviewed_df)),
        "n_confirmed_cross_sectional": int(len(confirmed)),
        "n_unclear": int(reviewed_df["manual_unclear_bool"].fillna(False).sum()),
        "n_excluded": int(reviewed_df["manual_exclude_bool"].fillna(False).sum()),
        "n_claims_confirmed": int(confirmed[claim_col].sum()),
        "claim_rate_confirmed": float(claim_rate),
        "claim_rate_confirmed_pct": float(claim_rate * 100),
    }

    return confirmed, summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sample 100 papers flagged as cross-sectional, export them for manual review, "
            "or recompute the claim rate from a completed review CSV."
        )
    )
    parser.add_argument(
        "--input-csv",
        default="data/json_files/filtered/all_abstracts_LLM.csv",
        help="Path to the classified abstracts CSV.",
    )
    parser.add_argument(
        "--review-csv",
        default="table/manual_review_cross_sectional_100.csv",
        help="Path to the review CSV to write or read.",
    )
    parser.add_argument(
        "--summary-csv",
        default="table/manual_review_cross_sectional_100_summary.csv",
        help="Path to the summary CSV written in summary mode.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=100,
        help="Number of cross-sectional papers to sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--mode",
        choices=["sample", "summary"],
        default="sample",
        help="Use 'sample' to export the review sheet or 'summary' after manual review is complete.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.mode == "sample":
        df = prepare_study_design_dataframe(args.input_csv)
        export_cross_sectional_review_sample(
            df=df,
            output_csv=args.review_csv,
            n=args.n,
            seed=args.seed,
            design_col="design_combined",
            design_value="Cross-sectional",
            claim_col="llm_policy_claim",
        )
        return

    confirmed, summary = calculate_confirmed_cross_sectional_claim_rate(
        review_csv=args.review_csv,
        claim_col="llm_policy_claim",
    )
    summary_df = pd.DataFrame([summary])
    summary_path = Path(args.summary_csv)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)

    print("Confirmed cross-sectional review summary:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    print(f"Summary written to: {summary_path.resolve()}")
    print(f"Confirmed rows retained: {len(confirmed)}")


if __name__ == "__main__":
    main()