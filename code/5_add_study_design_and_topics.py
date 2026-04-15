#!/usr/bin/env python3
"""
Enriches the analysis dataset with study design and topic classifications.

Two complementary methods are used to assign study design:
  - Strict (title/abstract): regex patterns matched against title and abstract
    text (e.g. "a cross-sectional study", "a randomised controlled trial").
  - Keywords: patterns matched against author-supplied keyword fields.
  A combined column prioritises the strict method and falls back to keywords.

Design categories: Experimental, Quasi-experimental, Cohort, Case-control,
Cross-sectional, Qualitative, Ecological/Time-series, Other/None.

Also adds binary topic/sensitivity flag columns (e.g. alcohol, tobacco,
diet, physical activity) derived from keyword matching, for use in
sub-group analyses.

Input:  data/analysis/analysis_dataset.csv
Output: data/analysis/analysis_dataset_enriched.csv

Run after: 4_build_analysis_dataset.py
Run before: 6_make_review_sample.py
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = ROOT / "data" / "analysis" / "analysis_dataset.csv"
OUTPUT_CSV = ROOT / "data" / "analysis" / "analysis_dataset_enriched.csv"


def normalize_patterns(patterns):
    out = []
    for p in patterns:
        p = p.replace("randomised", "randomi").replace("randomized", "randomi")
        p = p.replace("time-series", "time series").replace("meta-analysis", "meta analysis")
        out.append(p.lower())
    return out


STRICT_DESIGN_PATTERNS = {
    "Experimental": [
        "a randomi... controlled trial",
        "a randomi... trial",
        "randomi... clinical trial",
        "double-blind",
        "placebo-controlled trial",
    ],
    "Quasi-experimental": [
        "a quasi-experimental",
        "quasi-experimental study",
        "natural experiment",
        "difference-in-differences",
        "interrupted time series",
        "regression discontin...",
    ],
    "Cohort": [
        "a cohort study",
        "a prospective cohort",
        "a retrospective cohort",
        "a longitudinal study",
        "a longitudinal analysis",
    ],
    "Case-control": [
        "a case-control study",
        "case-control study",
        "matched case-control",
    ],
    "Cross-sectional": [
        "a cross-sectional study",
        "cross-sectional study",
        "cross-sectional survey",
    ],
    "Qualitative": [
        "a qualitative study",
        "qualitative analysis",
        "qualitative approach",
    ],
    "Ecological / Time-series": [
        "an ecological study",
        "ecological design",
        "time series analysis",
    ],
}
STRICT_DESIGN_PATTERNS = {
    k: normalize_patterns(v) for k, v in STRICT_DESIGN_PATTERNS.items()
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

METHOD_TERMS = ["epidemiologic methods", "simulation"]

EXCLUDE_TERMS = {
    "Policy analysis/evaluation": [
        "policy analysis",
        "policy analyses",
        "policy evaluation",
        "policy evaluations",
        "program evaluation",
        "programme evaluation",
        "impact evaluation",
        "economic evaluation",
        "cost-effectiveness",
        "cost effectiveness",
        "cost-benefit",
        "cost benefit",
    ],
    "Methods / quasi-experimental": [
        "epidemiologic methods",
        "simulation",
        "quasi-experimental",
        "quasi experimental",
        "quasi-experimental study",
        "natural experiment",
        "difference-in-differences",
        "difference in differences",
        "interrupted time series",
        "regression discontinuity",
        "instrumental variable",
        "instrumental variables",
        "propensity score",
        "propensity-score",
        "fixed effects",
        "fixed-effect",
        "synthetic control",
        "event study",
    ],
}
ALL_EXCLUDE_TERMS = [term.lower() for terms in EXCLUDE_TERMS.values() for term in terms]

POLICY_CLAIM_TERMS = [
    r"policies?\s+should",
    r"policymakers?\s+should",
    r"government\s+(must|should)",
    r"policy\s+(changes?|reforms?)\s+are\s+(needed|required|necessary)",
    r"we\s+(call\s+for|recommend|urge)\s+policy",
    r"we\s+call\s+(on|upon)\s+policymakers",
    r"we\s+call\s+for\s+government\s+action",
    r"(urgent|important)\s+need\s+for\s+policy\s+(action|change|reform)",
    r"interventions?\s+should\s+(focus|target|address|prioritize)",
    r"public\s+health\s+interventions?\s+should",
    r"policy\s+interventions?\s+(are\s+necessary|should\s+be\s+implemented)",
    r"(existing|current)\s+policies?\s+(are|is)\s+(inadequate|insufficient|outdated)",
    r"policy\s+has\s+(failed|not\s+been\s+effective)",
    r"policy\s+is\s+(failing|ineffective|inadequate)",
    r"policy\s+should\s+be\s+(reversed|changed|improved|strengthened)",
    r"policy\s+must\s+align\s+with\s+(scientific|ethical)\s+principles",
    r"policy\s+should\s+(prioritize|focus\s+on|address)",
    r"government\s+has\s+(a\s+duty|an\s+obligation)\s+to\s+implement\s+policy",
    r"government\s+should\s+(enact|pass|implement|fund)\s+policies?",
    r"legislation\s+must\s+be\s+introduced\s+to\s+(address|tackle)",
    r"(findings|results|study)\s+should\s+inform\s+policy",
    r"this\s+study\s+(shows|demonstrates|indicates)\s+(the\s+need|a\s+need)\s+for\s+policy\s+change",
    r"our\s+results\s+(support|justify|warrant)\s+policy\s+interventions",
    r"regulatory\s+framework\s+(should|must|needs\s+to)\s+be\s+(changed|revised)",
    r"stronger\s+policies\s+are\s+needed",
    r"public\s+health\s+policy\s+should\s+address",
    r"action\s+is\s+needed\s+to\s+(change|reform|improve)\s+policy",
    r"tax(ation)?\s+(policies?|measures?|reforms?)\s+(should|must|needs?\s+to)\s+be",
    r"tax(es)?\s+should\s+be\s+(raised|lowered|implemented|introduced|reconsidered)",
    r"tax\s+incentives?\s+(are|is)\s+(needed|required|necessary)",
    r"(sin|sugar|tobacco|alcohol|carbon)\s+tax(es)?\s+should\s+be\s+(implemented|increased|considered)",
    r"legal\s+framework\s+(should|must|needs\s+to)\s+be\s+(established|changed|revised)",
    r"(laws?|regulations?|rules?)\s+should\s+be\s+(changed|revised|strengthened|enacted)",
    r"(legal|regulatory)\s+changes?\s+are\s+(needed|necessary|required)",
    r"legislative\s+action\s+is\s+(needed|required|necessary)",
    r"(is|are)\s+a\s+need\s+for\s+(policy|legislative|regulatory|government)\s+(action|intervention|reform)",
    r"(clear|strong|compelling)\s+case\s+for\s+(policy|government)\s+intervention",
    r"action\s+is\s+needed",
    r"(there\s+is|it\s+is)\s+time\s+to\s+(implement|change|reconsider)\s+policies?",
    r"government\s+interventions?\s+(are|is)\s+(needed|necessary|required)",
    r"government\s+should\s+(regulate|intervene|act)",
    r"(national|state|federal|local)\s+policies?\s+should",
    r"public\s+policy\s+(must|should|needs\s+to)",
    r"(mandate|mandates|mandating|mandatory)\s+(are|is|should\s+be)\s+(needed|necessary|required|implemented)",
    r"(require|requires|requiring|requirements)\s+(are|is|should\s+be)\s+(needed|necessary|implemented)",
    r"(ban|bans|banning|banned)\s+on\s+.+\s+should\s+be\s+considered",
    r"(health\s+(system|care|policy|insurance)\s+(reform|changes?)\s+(are|is)\s+(needed|necessary|required))",
    r"health\s+system\s+should\s+be\s+(reformed|changed|improved)",
    r"(social|economic|welfare|housing|education)\s+policies?\s+should",
    r"(social|economic|welfare)\s+policy\s+reforms?\s+(are|is)\s+(needed|necessary|required)",
    r"(critical|essential|imperative|necessary|important)\s+that\s+.+(policy|government|policymakers)",
    r"(urgent|pressing|important)\s+(need|call)\s+for\s+(action|change|reform|intervention)",
    r"policy\s+efforts?\s+should\s+focus\s+on",
]


def parse_keywords(val) -> list[str]:
    if isinstance(val, str):
        toks = re.findall(r"'([^']+)'", val)
        if not toks and val.strip():
            toks = [x for x in val.split(";")]
    elif isinstance(val, (list, tuple)):
        toks = [str(t) for t in val if pd.notna(t)]
    else:
        toks = []
    return [t.strip().lower() for t in toks if str(t).strip()]


def assign_design_strict(row: pd.Series) -> str:
    title = str(row.get("title", "")).lower()
    abstract = str(row.get("abstract", "")).lower()
    text_to_search = f"{title} {abstract}"

    for group in DESIGN_HIERARCHY:
        patterns = STRICT_DESIGN_PATTERNS.get(group, [])
        if any(pat in text_to_search for pat in patterns):
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


def contains_any_term(val, terms) -> bool:
    if pd.isna(val):
        return False
    text = str(val).lower()
    return any(term in text for term in terms)


def detect_bold_policy_claims(text) -> bool:
    if pd.isna(text) or not isinstance(text, str):
        return False
    text_lower = text.lower()
    return any(re.search(pattern, text_lower) for pattern in POLICY_CLAIM_TERMS)


def add_study_design_and_topics(df: pd.DataFrame) -> pd.DataFrame:
    required = {"title", "abstract", "keywords", "llm_policy_claim", "publication_year"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    out = df.copy()
    out["llm_policy_claim"] = out["llm_policy_claim"].astype(bool)
    out["publication_year"] = pd.to_numeric(out["publication_year"], errors="coerce").astype("Int64")

    out["design_strict"] = out.apply(assign_design_strict, axis=1)
    out["design_keywords"] = out.apply(assign_design_keywords, axis=1)
    out["design_combined"] = np.where(
        out["design_strict"] != "Other/None",
        out["design_strict"],
        out["design_keywords"],
    )

    out["is_methodological"] = (
        out["title"].apply(lambda x: contains_any_term(x, METHOD_TERMS))
        | out["keywords"].apply(lambda x: contains_any_term(x, METHOD_TERMS))
    )

    out["is_excluded_sensitivity"] = (
        out["title"].apply(lambda x: contains_any_term(x, ALL_EXCLUDE_TERMS))
        | out["keywords"].apply(lambda x: contains_any_term(x, ALL_EXCLUDE_TERMS))
    )
    out["is_policy_analysis_eval"] = (
        out["title"].apply(lambda x: contains_any_term(x, EXCLUDE_TERMS["Policy analysis/evaluation"]))
        | out["keywords"].apply(lambda x: contains_any_term(x, EXCLUDE_TERMS["Policy analysis/evaluation"]))
    )
    out["is_methods_qe"] = (
        out["title"].apply(lambda x: contains_any_term(x, EXCLUDE_TERMS["Methods / quasi-experimental"]))
        | out["keywords"].apply(lambda x: contains_any_term(x, EXCLUDE_TERMS["Methods / quasi-experimental"]))
    )

    out["bold_policy_claim"] = out["abstract"].apply(detect_bold_policy_claims)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add study design and topic/sensitivity flags to the analysis dataset."
    )
    parser.add_argument("--input-csv", type=Path, default=INPUT_CSV)
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV)
    args = parser.parse_args()

    if not args.input_csv.exists():
        raise FileNotFoundError(
            f"Input analysis dataset not found: {args.input_csv}\n"
            "Run code/4_build_analysis_dataset.py first."
        )

    df = pd.read_csv(args.input_csv)
    enriched = add_study_design_and_topics(df)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(args.output_csv, index=False)

    print(f"Loaded input dataset: {len(df)} rows")
    print(f"Wrote enriched dataset: {args.output_csv}")
    print("\nStudy design counts:")
    print(enriched["design_combined"].value_counts(dropna=False))
    print("\nSensitivity/topic flags:")
    for col in [
        "is_methodological",
        "is_excluded_sensitivity",
        "is_policy_analysis_eval",
        "is_methods_qe",
        "bold_policy_claim",
    ]:
        print(f"  {col}: {int(enriched[col].sum())}")


if __name__ == "__main__":
    main()
