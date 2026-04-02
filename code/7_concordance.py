#!/usr/bin/env python3
"""Concordance workflow for manual-review workbooks and LLM runs.

This script supports:
1. running DeepSeek on a review workbook (`infer`)
2. importing an existing workbook column as a run file (`import-column` / `bridge`)
3. evaluating a run against a manual-review column (`evaluate`)
4. comparing two run files (`compare`)
5. sweeping multiple temperatures (`sweep`)

The defaults are aligned to the current project workflow:
- workbook: `table/gold_standard.xlsx`
- sheet: `in`
- outputs: `concordance/concordance_outputs/`
- reports: `concordance/concordance_reports/`
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXCEL = ROOT / "table" / "gold_standard.xlsx"
DEFAULT_SHEET = "in"
OUTPUTS_DIR = ROOT / "concordance" / "concordance_outputs"
REPORTS_DIR = ROOT / "concordance" / "concordance_reports"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0
RETRY_AFTER_FALLBACK = 5
API_TIMEOUT_SECONDS = 90
LLM_PAUSE_SECONDS = 0.2
NUM_CONCURRENT_REQUESTS = 5
SAVE_PROGRESS_EVERY_N = 50
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

DEFAULT_PROMPT = """You are an expert policy analyst and academic researcher. Your sole function is to analyze a research abstract and determine if it makes a policy claim based on a strict set of rules and examples.

## Rules for Classification

A statement **IS** a policy claim if:
- It directly suggests or calls for action, regulation, or intervention.
- It can be **vague** (eg, "This has implications for policy") or **specific** (eg, "States should ban X").
- It may be directed at a specific body or group (eg, governments, public health organizations, healthcare professionals) or be vague (eg, "future policies should").
- It appears in the concluding sentences.

A statement **IS NOT** a policy claim if:
- It is a suggestion for **future research**.
- It is a finding or a statement of fact with **no call to action**.
- It is a background statement that **motivates the research** (usually at the start of the abstract).

## Examples

ABSTRACT: "Our analysis of traffic data from 2020-2024 revealed that the new roundabout reduced accidents by 45%. These findings have significant implications for urban planning policy and should be considered by municipal transport authorities."
OUTPUT: {"policy_claim": true}

ABSTRACT: "We conducted a randomized controlled trial of a new diabetes drug. While the drug showed promise, there was no statistically significant improvement over existing treatments. Further investigation with a larger sample size is warranted to determine its efficacy."
OUTPUT: {"policy_claim": false}

ABSTRACT: "Our study shows a strong correlation between green space exposure and reduced symptoms of anxiety. To improve public health, municipal governments should enact zoning policies that mandate the inclusion of parks and green areas in all new housing developments."
OUTPUT: {"policy_claim": true}

ABSTRACT: "This paper reviews the historical literature concerning the United Kingdom's housing crisis. The data show that housing affordability has declined steadily since the 1980s across all regions, presenting a significant challenge for young adults."
OUTPUT: {"policy_claim": false}

ABSTRACT: "These results have important policy implications..."
OUTPUT: {"policy_claim": true}

## Output Format
Your response MUST be a valid JSON object and nothing else. Follow this exact schema:
{"policy_claim": true/false}

## Task
Now, analyze the following abstract based on all the rules and examples provided.

TITLE: {title}
ABSTRACT: {abstract}
OUTPUT:"""


def now_tag() -> str:
    return time.strftime("%Y-%m-%d_%H%M%S")


def slugify(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())


def prompt_hash(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()[:12]


def load_prompt(path: Optional[str]) -> str:
    if not path:
        return DEFAULT_PROMPT
    return Path(path).read_text(encoding="utf-8")


def normalize_bool_token(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "YES" if value else "NO"
    if isinstance(value, (int, np.integer)):
        return "YES" if int(value) == 1 else ("NO" if int(value) == 0 else str(value).upper())
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return ""
        if float(value) == 1.0:
            return "YES"
        if float(value) == 0.0:
            return "NO"

    text = str(value).replace("\u00a0", " ").replace("\u200b", "").replace("\ufeff", "")
    text = re.sub(r"\s+", " ", text).strip().lower()
    truthy = {"true", "t", "1", "1.0", "yes", "y", "是", "对", "confirmed"}
    falsy = {"false", "f", "0", "0.0", "no", "n", "否", "不", "excluded"}
    if text in truthy:
        return "YES"
    if text in falsy:
        return "NO"
    return text.upper()


def parse_json_label(raw_text: str) -> tuple[str, str]:
    raw = (raw_text or "").strip()
    if raw.startswith("```"):
        raw = raw[3:].strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
        if raw.endswith("```"):
            raw = raw[:-3].strip()

    label = "OTHER"
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and "policy_claim" in parsed:
            value = parsed["policy_claim"]
            if isinstance(value, bool):
                label = "YES" if value else "NO"
            elif isinstance(value, str) and value.lower() in {"true", "false"}:
                label = "YES" if value.lower() == "true" else "NO"
    except json.JSONDecodeError:
        pass
    return raw, label


def agreement_report(y_true: Iterable[str], y_pred: Iterable[str]) -> Dict[str, Any]:
    y_true = list(y_true)
    y_pred = list(y_pred)
    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    labels = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    rng = np.random.default_rng(42)
    idx = np.arange(len(y_true))
    acc_bs, kap_bs = [], []
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    for _ in range(1000):
        sample = rng.choice(idx, size=len(idx), replace=True)
        acc_bs.append(accuracy_score(y_true_arr[sample], y_pred_arr[sample]))
        kap_bs.append(cohen_kappa_score(y_true_arr[sample], y_pred_arr[sample]))

    def pct_ci(values: list[float]) -> tuple[float, float]:
        return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))

    return {
        "percent_agreement": float(acc),
        "percent_agreement_ci": pct_ci(acc_bs),
        "cohen_kappa": float(kappa),
        "cohen_kappa_ci": pct_ci(kap_bs),
        "labels": labels,
        "confusion_matrix": cm.tolist(),
    }


def require_deepseek_key() -> str:
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY not found in .env")
    return api_key


def read_workbook(excel_path: str | Path, sheet_name: str = DEFAULT_SHEET) -> pd.DataFrame:
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name, engine="openpyxl")
    except ImportError as exc:
        raise SystemExit(
            "openpyxl is required to read the concordance workbook. "
            "Install it in the project environment with `pip install openpyxl`."
        ) from exc

    wanted = [
        "scopus_id",
        "doi",
        "title",
        "journal",
        "publication_year",
        "abstract",
        "keywords",
        "corresponding_author_country",
        "cited_by_count",
        "policy_claim_via_terms",
        "llm_policy_claim",
        "DB review",
        "DB notes",
        "EC review",
        "EC notes",
        "MW review",
        "MW notes",
    ]
    cols = [c for c in wanted if c in df.columns]
    df = df[cols].copy()
    if "scopus_id" in df.columns:
        df["id"] = df["scopus_id"].fillna("").astype(str)
    else:
        df["id"] = np.arange(1, len(df) + 1).astype(str)
    if "title" not in df.columns:
        df["title"] = ""
    if "abstract" not in df.columns:
        df["abstract"] = ""
    df["title"] = df["title"].fillna("").astype(str)
    df["abstract"] = df["abstract"].fillna("").astype(str)
    return df


def build_message(row: pd.Series, prompt_text: str) -> str:
    return (
        prompt_text.replace("{title}", row.get("title", "").strip())
        .replace("{abstract}", row.get("abstract", "").strip())
    )


def load_cached_runs() -> dict[tuple[str, str, str, float], dict[str, Any]]:
    cache: dict[tuple[str, str, str, float], dict[str, Any]] = {}
    for path in OUTPUTS_DIR.glob("run_*.csv"):
        try:
            tmp = pd.read_csv(path)
        except Exception:
            continue
        for _, row in tmp.iterrows():
            key = (
                str(row.get("id", "")),
                str(row.get("prompt_hash", "")),
                str(row.get("model", "")),
                float(row.get("temperature", np.nan)),
            )
            cache[key] = row.to_dict()
    return cache


def call_deepseek_api(
    *,
    api_key: str,
    messages: list[dict[str, str]],
    temperature: float,
) -> Optional[str]:
    import httpx

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 60,
        "response_format": {"type": "json_object"},
    }

    for attempt in range(1, MAX_RETRIES + 1):
        time.sleep(LLM_PAUSE_SECONDS)
        try:
            response = httpx.post(
                DEEPSEEK_API_URL,
                headers=headers,
                json=payload,
                timeout=API_TIMEOUT_SECONDS,
            )
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                wait_seconds = int(retry_after) if retry_after and retry_after.isdigit() else RETRY_AFTER_FALLBACK
                print(f"[429] rate-limited, waiting {wait_seconds}s (attempt {attempt}/{MAX_RETRIES})")
                time.sleep(wait_seconds)
                continue
            if 500 <= response.status_code < 600:
                print(f"[{response.status_code}] server error (attempt {attempt}/{MAX_RETRIES})")
                time.sleep(RETRY_BACKOFF_BASE ** (attempt - 1))
                continue
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as exc:
            print(f"[ERROR] API attempt {attempt}: {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_BASE ** (attempt - 1))
    return None


def infer_one(
    row_dict: dict[str, Any],
    *,
    api_key: str,
    prompt_text: str,
    prompt_name: str,
    prompt_digest: str,
    temperature: float,
) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": "You are an expert academic classifier that only returns JSON."},
        {"role": "user", "content": build_message(pd.Series(row_dict), prompt_text)},
    ]
    raw = call_deepseek_api(api_key=api_key, messages=messages, temperature=temperature)
    if raw is None:
        llm_output, llm_label = "[ERROR]", "OTHER"
    else:
        llm_output, llm_label = parse_json_label(raw)

    return {
        "id": row_dict.get("id", ""),
        "scopus_id": row_dict.get("scopus_id", ""),
        "doi": row_dict.get("doi", ""),
        "title": row_dict.get("title", ""),
        "prompt_name": prompt_name,
        "prompt_hash": prompt_digest,
        "model": DEEPSEEK_MODEL,
        "temperature": temperature,
        "llm_output": llm_output,
        "llm_label": llm_label,
    }


def find_latest_run(prompt_name: str, temperature: float) -> Optional[Path]:
    pattern = f"run_{slugify(prompt_name)}_temp{temperature}_*.csv"
    runs = sorted(OUTPUTS_DIR.glob(pattern), key=lambda p: p.stat().st_mtime)
    return runs[-1] if runs else None


def cmd_infer(args: argparse.Namespace) -> None:
    api_key = require_deepseek_key()
    df = read_workbook(args.excel, sheet_name=args.sheet)
    df = df.sort_values("id").reset_index(drop=True)
    if args.limit is not None:
        df = df.iloc[: args.limit].copy()

    prompt_text = load_prompt(args.prompt_file)
    prompt_digest = prompt_hash(prompt_text)
    cached_rows: list[dict[str, Any]] = []
    rows_to_process: list[dict[str, Any]] = []

    cache = {} if args.no_cache else load_cached_runs()
    for _, row in df.iterrows():
        key = (str(row["id"]), prompt_digest, DEEPSEEK_MODEL, float(args.temperature))
        if key in cache:
            cached_rows.append(cache[key])
        else:
            rows_to_process.append(row.to_dict())

    print(f"Loaded workbook rows: {len(df)}")
    print(f"Using cache: {len(cached_rows)} rows")
    print(f"New API calls needed: {len(rows_to_process)} rows")

    results = list(cached_rows)
    if rows_to_process:
        partial_path = OUTPUTS_DIR / f"run_{slugify(args.prompt_name)}_temp{args.temperature}_partial.csv"
        with ThreadPoolExecutor(max_workers=NUM_CONCURRENT_REQUESTS) as executor:
            futures = {
                executor.submit(
                    infer_one,
                    row_dict,
                    api_key=api_key,
                    prompt_text=prompt_text,
                    prompt_name=args.prompt_name,
                    prompt_digest=prompt_digest,
                    temperature=float(args.temperature),
                ): row_dict["id"]
                for row_dict in rows_to_process
            }
            for i, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Processing"), start=1):
                try:
                    results.append(future.result())
                except Exception as exc:
                    sample_id = futures[future]
                    results.append(
                        {
                            "id": sample_id,
                            "scopus_id": "",
                            "doi": "",
                            "title": "",
                            "prompt_name": args.prompt_name,
                            "prompt_hash": prompt_digest,
                            "model": DEEPSEEK_MODEL,
                            "temperature": float(args.temperature),
                            "llm_output": f"[ERROR: {exc}]",
                            "llm_label": "OTHER",
                        }
                    )
                if i % SAVE_PROGRESS_EVERY_N == 0:
                    pd.DataFrame(results).to_csv(partial_path, index=False)
        if partial_path.exists():
            partial_path.unlink()

    out_df = pd.DataFrame(results).sort_values("id").reset_index(drop=True)
    out_path = OUTPUTS_DIR / f"run_{slugify(args.prompt_name)}_temp{args.temperature}_{now_tag()}.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved run: {out_path}")


def import_workbook_column(df: pd.DataFrame, column: str, run_name: str) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        value = row.get(column, None)
        llm_label = ""
        llm_output = ""

        if isinstance(value, str) and "{" in value and "}" in value:
            llm_output, llm_label = parse_json_label(value)
        else:
            token = normalize_bool_token(value)
            if token in {"YES", "NO"}:
                llm_label = token
                llm_output = json.dumps({"policy_claim": token == "YES"})
            else:
                llm_label = "OTHER" if token else ""
                llm_output = "" if pd.isna(value) else str(value)

        rows.append(
            {
                "id": row.get("id", ""),
                "scopus_id": row.get("scopus_id", ""),
                "doi": row.get("doi", ""),
                "title": row.get("title", ""),
                "prompt_name": run_name,
                "prompt_hash": "excel",
                "model": "excel-column",
                "temperature": np.nan,
                "llm_output": llm_output,
                "llm_label": llm_label,
            }
        )
    return pd.DataFrame(rows)


def cmd_import_column(args: argparse.Namespace) -> None:
    df = read_workbook(args.excel, sheet_name=args.sheet)
    if args.column not in df.columns:
        raise SystemExit(f"Column '{args.column}' not found. Available columns: {list(df.columns)}")
    run_name = args.name or f"excel:{args.column}"
    out_df = import_workbook_column(df, args.column, run_name)
    out_path = OUTPUTS_DIR / f"run_{slugify(run_name)}_{now_tag()}.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved imported run: {out_path}")


def cmd_bridge(args: argparse.Namespace) -> None:
    cmd_import_column(args)


def cmd_evaluate(args: argparse.Namespace) -> None:
    run_df = pd.read_csv(args.run_csv)
    wb_df = read_workbook(args.excel, sheet_name=args.sheet)
    merged = run_df.merge(wb_df, on="id", how="left", suffixes=("", "_excel"))

    if args.ref_column not in merged.columns:
        raise SystemExit(f"Reference column '{args.ref_column}' not found in workbook sheet '{args.sheet}'.")

    merged["_ref_norm"] = merged[args.ref_column].apply(normalize_bool_token)
    merged["_pred_norm"] = merged["llm_label"].apply(normalize_bool_token)
    labeled = merged[merged["_ref_norm"].isin(["YES", "NO"])].copy()
    if labeled.empty:
        raise SystemExit("No usable YES/NO labels found in the reference column.")

    rep = agreement_report(labeled["_ref_norm"], labeled["_pred_norm"])
    mismatch_mask = labeled["_ref_norm"] != labeled["_pred_norm"]
    mismatch_cols = [
        c
        for c in ["id", "scopus_id", "doi", "title", args.ref_column, "llm_label", "llm_output", "abstract"]
        if c in labeled.columns
    ]
    mismatches = labeled.loc[mismatch_mask, mismatch_cols].copy()

    mismatches_path = REPORTS_DIR / f"mismatches_{slugify(args.ref_column)}_{now_tag()}.csv"
    mismatches.to_csv(mismatches_path, index=False)

    report_lines = [
        "# Concordance Report",
        f"- Run CSV: `{args.run_csv}`",
        f"- Workbook: `{args.excel}` (sheet: `{args.sheet}`)",
        f"- Reference column: `{args.ref_column}`",
        f"- Evaluated labeled rows: {len(labeled)} / {len(merged)}",
        "",
        f"**Percent agreement:** {rep['percent_agreement']:.3f} "
        f"(95% CI {rep['percent_agreement_ci'][0]:.3f}-{rep['percent_agreement_ci'][1]:.3f})",
        f"**Cohen's kappa:** {rep['cohen_kappa']:.3f} "
        f"(95% CI {rep['cohen_kappa_ci'][0]:.3f}-{rep['cohen_kappa_ci'][1]:.3f})",
        "",
        "## Confusion matrix",
        f"Labels: {rep['labels']}",
        "```",
        *[str(row) for row in rep["confusion_matrix"]],
        "```",
        "",
        f"Mismatches saved to `{mismatches_path}`.",
    ]
    report_path = REPORTS_DIR / f"summary_{slugify(args.ref_column)}_{now_tag()}.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Saved report: {report_path}")
    print(f"Saved mismatches: {mismatches_path}")


def cmd_compare(args: argparse.Namespace) -> None:
    run_a = pd.read_csv(args.run_a)
    run_b = pd.read_csv(args.run_b)
    merged = run_a.merge(run_b, on="id", suffixes=("_a", "_b"))
    valid = merged[
        merged["llm_label_a"].isin(["YES", "NO"]) & merged["llm_label_b"].isin(["YES", "NO"])
    ].copy()
    if valid.empty:
        raise SystemExit("No overlapping YES/NO rows found between the two runs.")

    rep = agreement_report(
        valid["llm_label_a"].apply(normalize_bool_token),
        valid["llm_label_b"].apply(normalize_bool_token),
    )

    report_lines = [
        "# A/B Concordance",
        f"- Run A: `{args.run_a}`",
        f"- Run B: `{args.run_b}`",
        f"- Overlapping labeled rows: {len(valid)}",
        "",
        f"**Percent agreement:** {rep['percent_agreement']:.3f}",
        f"**Cohen's kappa:** {rep['cohen_kappa']:.3f}",
        "",
        "## Confusion matrix",
        f"Labels: {rep['labels']}",
        "```",
        *[str(row) for row in rep["confusion_matrix"]],
        "```",
    ]
    report_path = REPORTS_DIR / f"ab_compare_{now_tag()}.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print("\n".join(report_lines))
    print(f"\n Saved A/B report: {report_path}")


def cmd_sweep(args: argparse.Namespace) -> None:
    if not args.ref_column and not args.baseline_run:
        print("[INFO] No --ref-column or --baseline-run provided; sweep will only run inference.")
    for temp in args.temps:
        print(f"\n[SWEEP] temperature = {temp}")
        infer_args = argparse.Namespace(
            excel=args.excel,
            sheet=args.sheet,
            prompt_file=args.prompt_file,
            prompt_name=args.prompt_name,
            temperature=float(temp),
            no_cache=args.no_cache,
            limit=args.limit,
        )
        cmd_infer(infer_args)

        run_path = find_latest_run(args.prompt_name, float(temp))
        if not run_path:
            print("[WARN] Could not find the run file that was just created.")
            continue

        if args.ref_column:
            eval_args = argparse.Namespace(
                run_csv=str(run_path),
                excel=args.excel,
                sheet=args.sheet,
                ref_column=args.ref_column,
            )
            cmd_evaluate(eval_args)

        if args.baseline_run:
            compare_args = argparse.Namespace(run_a=args.baseline_run, run_b=str(run_path))
            cmd_compare(compare_args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Concordance workflow for the policy-claims review workbook."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    infer = sub.add_parser("infer", help="Run DeepSeek on the workbook and save run_*.csv.")
    infer.add_argument("--excel", default=str(DEFAULT_EXCEL), help=f"Workbook path (default: {DEFAULT_EXCEL})")
    infer.add_argument("--sheet", default=DEFAULT_SHEET, help=f"Workbook sheet (default: {DEFAULT_SHEET})")
    infer.add_argument("--prompt-file", default=None, help="Optional prompt file. Defaults to the built-in prompt.")
    infer.add_argument("--prompt-name", required=True, help="Short name for this run, e.g. updated_v3")
    infer.add_argument("--temperature", type=float, default=1.3, help="Sampling temperature (default: 1.3)")
    infer.add_argument("--no-cache", action="store_true", help="Ignore existing run_*.csv cache")
    infer.add_argument("--limit", type=int, default=None, help="Optional row limit for testing")
    infer.set_defaults(func=cmd_infer)

    evaluate = sub.add_parser("evaluate", help="Evaluate a run against a workbook review column.")
    evaluate.add_argument("--run-csv", required=True, help="Path to run_*.csv")
    evaluate.add_argument("--excel", default=str(DEFAULT_EXCEL), help=f"Workbook path (default: {DEFAULT_EXCEL})")
    evaluate.add_argument("--sheet", default=DEFAULT_SHEET, help=f"Workbook sheet (default: {DEFAULT_SHEET})")
    evaluate.add_argument("--ref-column", required=True, help="Reference column, e.g. 'MW review'")
    evaluate.set_defaults(func=cmd_evaluate)

    compare = sub.add_parser("compare", help="Compare two run_*.csv files.")
    compare.add_argument("--run-a", required=True, help="First run file")
    compare.add_argument("--run-b", required=True, help="Second run file")
    compare.set_defaults(func=cmd_compare)

    import_col = sub.add_parser("import-column", help="Convert a workbook column into a run_*.csv file.")
    import_col.add_argument("--excel", default=str(DEFAULT_EXCEL), help=f"Workbook path (default: {DEFAULT_EXCEL})")
    import_col.add_argument("--sheet", default=DEFAULT_SHEET, help=f"Workbook sheet (default: {DEFAULT_SHEET})")
    import_col.add_argument("--column", required=True, help="Workbook column to import")
    import_col.add_argument("--name", default=None, help="Optional run name override")
    import_col.set_defaults(func=cmd_import_column)

    sweep = sub.add_parser("sweep", help="Run inference at multiple temperatures, with optional evaluation/compare.")
    sweep.add_argument("--excel", default=str(DEFAULT_EXCEL), help=f"Workbook path (default: {DEFAULT_EXCEL})")
    sweep.add_argument("--sheet", default=DEFAULT_SHEET, help=f"Workbook sheet (default: {DEFAULT_SHEET})")
    sweep.add_argument("--prompt-file", default=None)
    sweep.add_argument("--prompt-name", required=True)
    sweep.add_argument("--temps", nargs="+", type=float, required=True, help="Example: 1.3 0.9 0.7 0.3")
    sweep.add_argument("--ref-column", default=None, help="Optional workbook column for concordance evaluation")
    sweep.add_argument("--baseline-run", default=None, help="Optional run_*.csv to compare each sweep output against")
    sweep.add_argument("--no-cache", action="store_true")
    sweep.add_argument("--limit", type=int, default=None, help="Optional row limit per temperature")
    sweep.set_defaults(func=cmd_sweep)

    bridge = sub.add_parser("bridge", help="Alias for import-column.")
    bridge.add_argument("--excel", default=str(DEFAULT_EXCEL), help=f"Workbook path (default: {DEFAULT_EXCEL})")
    bridge.add_argument("--sheet", default=DEFAULT_SHEET, help=f"Workbook sheet (default: {DEFAULT_SHEET})")
    bridge.add_argument("--column", required=True, help="Workbook column to import")
    bridge.add_argument("--name", default=None, help="Optional run name override")
    bridge.set_defaults(func=cmd_bridge)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
