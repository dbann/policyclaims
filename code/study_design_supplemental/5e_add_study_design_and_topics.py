#!/usr/bin/env python3
"""
Enriches the analysis dataset with study design and topic classifications.

Two complementary methods are used to assign study design:
  - Strict (title/abstract): regex patterns matched against title and abstract
    text (e.g. "a cross-sectional study", "a randomised controlled trial").

  - Keywords: patterns matched against author-supplied keyword fields.
  A combined column prioritises the strict method and falls back to keywords.

An additional `design_combined_unambiguous` column is produced for sensitivity
analysis: it matches `design_combined` except that rows where the strict
classifier matched 2+ designs (~1-2% of rows) are blanked to "Other/None".
This lets you verify that downstream findings are not an artefact of the
design-hierarchy tiebreaker.

Design categories: Experimental, Quasi-experimental, Cohort, Case-control,
Cross-sectional, Qualitative, Ecological/Time-series, Other/None.

Input:  data/analysis/analysis_dataset.csv
Output: data/analysis/analysis_dataset_enriched_v2e.csv

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
OUTPUT_CSV = ROOT / "data" / "analysis" / "analysis_dataset_enriched_v2e.csv"


def normalize_text(text: str) -> str:
    """Normalise text for matching: lowercase, unify hyphens/dashes, collapse whitespace.

    Applied to BOTH patterns and search text so matches are consistent. This lets us
    write patterns without hyphens (e.g. "cross sectional study") and match against
    "cross-sectional", "cross–sectional" (en-dash), and "cross sectional" alike.
    """
    text = text.lower()
    # Unify hyphens, en-dashes, em-dashes to spaces
    text = re.sub(r"[-–—]", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_patterns(patterns):
    """Apply the same normalisation to patterns that we apply to search text."""
    return [normalize_text(p) for p in patterns]


TITLE_ABSTRACT_PATTERNS = {
    "Experimental": [
        "randomised controlled trial",
        "randomized controlled trial",
        "randomised trial",
        "randomized trial",
        "randomised double blind trial",
        "randomized double blind trial",
        "randomised double blinded trial",
        "randomized double blinded trial",
        "randomised double blind controlled trial",
        "randomized double blind controlled trial",
        "randomised double blinded controlled trial",
        "randomized double blinded controlled trial",
        #"controlled trial",
        #"clinical trial",
        "placebo controlled",
        "cluster randomised",
        "cluster randomized",
        "stepped wedge",
    ],
    "Quasi-experimental": [
        "quasi experimental",
        "natural experiment",
        "difference in difference",
        "interrupted time series",
        "regression discontinuity",
        #"regression discon",
        "synthetic control",
        "diff in diff",
    ],
    "Cohort": [
        "cohort study",
        "cohort analysis",
        "prospective cohort",
        "retrospective cohort",
        "birth cohort",
        "longitudinal study",
        "longitudinal analysis",
        "longitudinal cohort",
        "follow up study",
    ],
    "Case-control": [
        "case control study",
        "case control analysis",
        "matched case control",
        "nested case control",
    ],
    "Cross-sectional": [
        "cross sectional study",
        "cross sectional survey",
        "cross sectional analysis",
        "cross sectional design",
        "cross sectional association",
        "cross sectional associations",
    ],
    "Qualitative": [
        "qualitative study",
        "qualitative analysis",
        "qualitative approach",
        "qualitative research",
    ],
    "Ecological / Time-series": [
        "ecological study",
        "ecological design",
        "ecological analysis",
        "ecologic study",
        "ecologic design",
        "ecologic analysis",
        "ecologic case referent",
        "ecologic case referent study design",
        "time series",
    ],
}
TITLE_ABSTRACT_PATTERNS = {
    k: normalize_patterns(v) for k, v in TITLE_ABSTRACT_PATTERNS.items()
}

KEYWORD_DESIGN_PATTERNS = {
    "Experimental": [
        "randomised controlled trial",
        "randomized controlled trial",
        "randomised trial",
        "randomized trial",
        "clinical trial",
        "rct",
        "pragmatic trial",
        "cluster randomised",
        "cluster randomized",
        "stepped wedge",
    ],
    "Quasi-experimental": [
        "quasi experimental",
        "natural experiment",
        "difference in difference",
        "diff in diff",
        "event study",
        "regression discontinuity",
        "interrupted time series",
        "synthetic control",
    ],
    "Cohort": [
        "cohort study",
        "cohort studies",
        "cohort analysis",
        "prospective cohort",
        "retrospective cohort",
        "birth cohort",
        "longitudinal study",
        "longitudinal analysis",
        "longitudinal cohort",
        "follow up study",
        "panel study",
    ],
    "Case-control": [
        "case control",
        "matched case control",
        "nested case control",
    ],
    "Cross-sectional": [
        "cross sectional",
        "prevalence study",
        "baseline survey",
    ],
    "Ecological / Time-series": [
        "ecological study",
        "ecological analysis",
        "ecologic study",
        "ecologic analysis",
        "ecologic design",
        "ecologic case referent",
        "time series",
    ],
    "Qualitative": [
        "qualitative study",
        "qualitative research",
        "qualitative analysis",
        "focus group",
        "thematic analysis",
        "ethnograph",
        "grounded theory",
        "semi structured interview",
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


def _assign_design_from_text(text: str) -> str:
    """Return the highest-priority design group whose pattern is found in `text`.

    `text` must already be normalised via normalize_text(). Returns "Other/None"
    if no pattern matches, or if the text is empty.
    """
    if not text:
        return "Other/None"
    for group in DESIGN_HIERARCHY:
        patterns = TITLE_ABSTRACT_PATTERNS.get(group, [])
        if any(pat in text for pat in patterns):
            return group
    return "Other/None"


def _all_designs_matching_text(text: str) -> list[str]:
    """Return ALL design groups whose patterns match `text` (no hierarchy).

    Used for diagnostics only: quantifies how often the hierarchy is masking
    additional matches. Order preserved by DESIGN_HIERARCHY for stable output.
    """
    if not text:
        return []
    return [
        group
        for group in DESIGN_HIERARCHY
        if any(pat in text for pat in TITLE_ABSTRACT_PATTERNS.get(group, []))
    ]


def _all_designs_matching_keywords(keywords: list[str]) -> list[str]:
    """Return ALL design groups whose keyword patterns match any token."""
    if not keywords:
        return []
    return [
        group
        for group in DESIGN_HIERARCHY
        if any(
            pat in tok
            for tok in keywords
            for pat in KEYWORD_DESIGN_PATTERNS.get(group, [])
        )
    ]


def assign_design_title(row: pd.Series) -> str:
    """Match design patterns against the title only."""
    title_val = row.get("title")
    title = "" if pd.isna(title_val) else str(title_val)
    return _assign_design_from_text(normalize_text(title))


def assign_design_abstract(row: pd.Series) -> str:
    """Match design patterns against the abstract only."""
    abstract_val = row.get("abstract")
    abstract = "" if pd.isna(abstract_val) else str(abstract_val)
    return _assign_design_from_text(normalize_text(abstract))


def assign_design_strict(row: pd.Series) -> str:
    """Match design patterns against title and abstract combined."""
    title_val = row.get("title")
    abstract_val = row.get("abstract")
    title = "" if pd.isna(title_val) else str(title_val)
    abstract = "" if pd.isna(abstract_val) else str(abstract_val)
    return _assign_design_from_text(normalize_text(f"{title} {abstract}"))


def assign_design_keywords(row: pd.Series) -> str:
    keywords = parse_keywords(row.get("keywords", ""))
    if not keywords:
        return "Other/None"
    # Normalise each keyword token the same way we normalise patterns
    keywords = [normalize_text(k) for k in keywords]

    for group in DESIGN_HIERARCHY:
        patterns = KEYWORD_DESIGN_PATTERNS.get(group, [])
        if any(pat in tok for tok in keywords for pat in patterns):
            return group
    return "Other/None"


def contains_any_term(val, terms) -> bool:
    if pd.isna(val):
        return False
    text = normalize_text(str(val))
    norm_terms = [normalize_text(term) for term in terms]
    return any(term in text for term in norm_terms)


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
    # Robust boolean conversion: handles True/False, 1/0, "1"/"0", NaN.
    # Not strictly needed for current upstream (already bool) but guards against
    # silent breakage if 4_build_analysis_dataset.py ever writes strings.
    out["llm_policy_claim"] = pd.to_numeric(
        out["llm_policy_claim"], errors="coerce"
    ).fillna(0).astype(bool)
    out["publication_year"] = pd.to_numeric(out["publication_year"], errors="coerce").astype("Int64")

    out["design_title"] = out.apply(assign_design_title, axis=1)
    out["design_abstract"] = out.apply(assign_design_abstract, axis=1)
    out["design_strict"] = out.apply(assign_design_strict, axis=1)
    out["design_keywords"] = out.apply(assign_design_keywords, axis=1)
    out["design_combined"] = np.where(
        out["design_strict"] != "Other/None",
        out["design_strict"],
        out["design_keywords"],
    )

    # Sensitivity-analysis column: blank out rows where the strict (title+abstract)
    # classifier matched 2+ designs simultaneously. Rationale: these are the only
    # rows where the hierarchy tiebreaker actually affects design_combined, so
    # excluding them lets you check that downstream findings aren't an artefact
    # of hierarchy choices. Rows where strict returned Other/None and keywords
    # filled in are NOT flagged — keywords only contribute when strict is silent,
    # so there's no disagreement to sensitivity-test.
    strict_match_counts = out.apply(
        lambda r: len(
            _all_designs_matching_text(
                normalize_text(
                    f"{'' if pd.isna(r.get('title')) else r.get('title')} "
                    f"{'' if pd.isna(r.get('abstract')) else r.get('abstract')}"
                )
            )
        ),
        axis=1,
    )
    out["design_combined_unambiguous"] = np.where(
        strict_match_counts >= 2,
        "Other/None",
        out["design_combined"],
    )

    out["is_methodological"] = (
        out["title"].apply(lambda x: contains_any_term(x, METHOD_TERMS))
        | out["abstract"].apply(lambda x: contains_any_term(x, METHOD_TERMS))
        | out["keywords"].apply(lambda x: contains_any_term(x, METHOD_TERMS))
    )

    out["is_excluded_sensitivity"] = (
        out["title"].apply(lambda x: contains_any_term(x, ALL_EXCLUDE_TERMS))
        | out["abstract"].apply(lambda x: contains_any_term(x, ALL_EXCLUDE_TERMS))
        | out["keywords"].apply(lambda x: contains_any_term(x, ALL_EXCLUDE_TERMS))
    )
    out["is_policy_analysis_eval"] = (
        out["title"].apply(lambda x: contains_any_term(x, EXCLUDE_TERMS["Policy analysis/evaluation"]))
        | out["abstract"].apply(lambda x: contains_any_term(x, EXCLUDE_TERMS["Policy analysis/evaluation"]))
        | out["keywords"].apply(lambda x: contains_any_term(x, EXCLUDE_TERMS["Policy analysis/evaluation"]))
    )
    out["is_methods_qe"] = (
        out["title"].apply(lambda x: contains_any_term(x, EXCLUDE_TERMS["Methods / quasi-experimental"]))
        | out["abstract"].apply(lambda x: contains_any_term(x, EXCLUDE_TERMS["Methods / quasi-experimental"]))
        | out["keywords"].apply(lambda x: contains_any_term(x, EXCLUDE_TERMS["Methods / quasi-experimental"]))
    )

    out["bold_policy_claim"] = out["abstract"].apply(detect_bold_policy_claims)
    return out


def report_hierarchy_overlap(df: pd.DataFrame) -> None:
    """Report how often the DESIGN_HIERARCHY is masking additional matches.

    For each of title, abstract, title+abstract, and keywords, computes the full
    set of matching designs (no hierarchy) and reports:
      - how many rows matched 0 / 1 / 2+ designs
      - the most common multi-match combinations
      - for rows with 2+ matches, which design the hierarchy kept vs. dropped

    Purely diagnostic — does not modify the dataframe or the written CSV.
    """

    def _title(row):
        val = row.get("title")
        return "" if pd.isna(val) else str(val)

    def _abstract(row):
        val = row.get("abstract")
        return "" if pd.isna(val) else str(val)

    sources = {
        "title": df.apply(
            lambda r: _all_designs_matching_text(normalize_text(_title(r))), axis=1
        ),
        "abstract": df.apply(
            lambda r: _all_designs_matching_text(normalize_text(_abstract(r))), axis=1
        ),
        "title+abstract (strict)": df.apply(
            lambda r: _all_designs_matching_text(
                normalize_text(f"{_title(r)} {_abstract(r)}")
            ),
            axis=1,
        ),
        "keywords": df.apply(
            lambda r: _all_designs_matching_keywords(
                [normalize_text(k) for k in parse_keywords(r.get("keywords", ""))]
            ),
            axis=1,
        ),
    }

    print("\n" + "=" * 72)
    print("HIERARCHY DIAGNOSTIC: how often does the hierarchy mask extra matches?")
    print("=" * 72)

    for source_name, matches in sources.items():
        counts = matches.apply(len)
        n_zero = int((counts == 0).sum())
        n_one = int((counts == 1).sum())
        n_multi = int((counts >= 2).sum())
        total = len(counts)
        pct_multi = (n_multi / total * 100) if total else 0.0

        print(f"\n--- {source_name} ---")
        print(
            f"  0 matches: {n_zero}  |  1 match: {n_one}  |  "
            f"2+ matches: {n_multi} ({pct_multi:.1f}% of rows)"
        )

        if n_multi == 0:
            continue

        multi = matches[counts >= 2]

        # Most common multi-match combinations
        combo_counts = (
            multi.apply(lambda lst: " + ".join(lst)).value_counts().head(10)
        )
        print(f"  Top multi-match combinations (up to 10):")
        for combo, n in combo_counts.items():
            print(f"    {n:>5}  {combo}")

        # What the hierarchy kept vs. what it dropped
        kept_vs_dropped = (
            multi.apply(lambda lst: f"kept {lst[0]}  (dropped: {', '.join(lst[1:])})")
            .value_counts()
            .head(10)
        )
        print(f"  Top 'kept vs. dropped' patterns (up to 10):")
        for pattern, n in kept_vs_dropped.items():
            print(f"    {n:>5}  {pattern}")


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
    print("\nStudy design counts (title only):")
    print(enriched["design_title"].value_counts(dropna=False))
    print("\nStudy design counts (abstract only):")
    print(enriched["design_abstract"].value_counts(dropna=False))
    print("\nStudy design counts (combined):")
    print(enriched["design_combined"].value_counts(dropna=False))
    print("\nStudy design counts (combined, unambiguous only — sensitivity):")
    print(enriched["design_combined_unambiguous"].value_counts(dropna=False))
    print("\nSensitivity/topic flags:")
    for col in [
        "is_methodological",
        "is_excluded_sensitivity",
        "is_policy_analysis_eval",
        "is_methods_qe",
        "bold_policy_claim",
    ]:
        print(f"  {col}: {int(enriched[col].sum())}")

    report_hierarchy_overlap(enriched)

if __name__ == "__main__":
    main()
