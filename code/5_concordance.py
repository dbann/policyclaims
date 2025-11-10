#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OPTIMIZED VERSION - Key improvements:
1. Concurrent API calls (5x speedup potential)
2. Efficient cache management (load once, update incrementally)
3. Batch processing with progress saving
4. Reduced API pausing for cached items
"""

import os
import re
import json
import time
import argparse
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import concurrent.futures
from threading import Lock

import numpy as np
import pandas as pd
import httpx
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix
from dotenv import load_dotenv
from tqdm import tqdm

OUTPUTS_DIR = Path(__file__).resolve().parents[1] / "concordance" / "concordance_outputs"; OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = Path(__file__).resolve().parents[1] / "concordance" / "concordance_reports";  REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Performance settings
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0
RETRY_AFTER_FALLBACK = 5
LLM_PAUSE_SECONDS = 0.2  # Only for actual API calls
API_TIMEOUT_SECONDS = 90
NUM_CONCURRENT_REQUESTS = 5  # NEW: Parallel processing
SAVE_PROGRESS_EVERY_N = 50   # NEW: Save intermediate results

DEFAULT_PROMPT = """You are an expert policy analyst and academic researcher. Your sole function is to analyze a research abstract and determine if it makes a policy claim based on a strict set of rules and examples.

## Rules for Classification

A statement **IS** a policy claim if:
- It directly suggests or calls for action, regulation, or intervention.
- It can be **vague** (eg, "This has implications for policy") or **specific** (e.g., "States should ban X").
- It may be directed at a specific body or group (e.g., governments, public health organizations, healthcare professionals) or be vague (eg, "future policies should").
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

ABSTRACT: {abstract}
OUTPUT:"""

# Thread-safe cache and results storage
cache_lock = Lock()
results_lock = Lock()

def now_tag() -> str:
    return time.strftime("%Y-%m-%d_%H%M%S")

def prompt_hash(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()[:12]

def load_prompt(path: Optional[str]) -> str:
    if not path:
        return DEFAULT_PROMPT
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def normalize_bool_token(x: Any) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""

    if isinstance(x, (bool, np.bool_)):
        return "YES" if x else "NO"

    if isinstance(x, (int, np.integer)):
        return "YES" if int(x) == 1 else ("NO" if int(x) == 0 else str(x).upper())
    if isinstance(x, (float, np.floating)):
        if np.isnan(x):
            return ""
        if float(x) == 1.0: return "YES"
        if float(x) == 0.0: return "NO"

    s = str(x).replace("\u00a0"," ").replace("\u200b","").replace("\ufeff","")
    s = re.sub(r"\s+"," ",s).strip().lower()

    truthy = {"true","t","1","1.0","yes","y","是","对"}
    falsy  = {"false","f","0","0.0","no","n","否","不"}
    if s in truthy: return "YES"
    if s in falsy:  return "NO"

    return s.upper()

def postprocess_label_from_json_str(llm_json_str: str) -> Tuple[str, str]:
    raw = (llm_json_str or "").strip()
    if raw.startswith("```"):
        s = raw
        if s.startswith("```"):
            s = s[3:]
        s = s.strip()
        if s.lower().startswith("json"):
            s = s[4:].strip()
        if s.endswith("```"):
            s = s[:-3].strip()
        raw = s

    label = "OTHER"
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and "policy_claim" in parsed:
            val = parsed["policy_claim"]
            if isinstance(val, bool):
                label = "YES" if val else "NO"
            elif isinstance(val, str) and val.lower() in {"true", "false"}:
                label = "YES" if val.lower() == "true" else "NO"
    except json.JSONDecodeError:
        pass
    return raw, label

def agreement_report(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    labels = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # bootstrap 95% CI
    rng = np.random.default_rng(42)
    B = 1000
    n = len(y_true)
    idx = np.arange(n)
    acc_bs, kap_bs = [], []
    y_true_arr = np.array(y_true); y_pred_arr = np.array(y_pred)
    for _ in range(B):
        sample = rng.choice(idx, size=n, replace=True)
        acc_bs.append(accuracy_score(y_true_arr[sample], y_pred_arr[sample]))
        kap_bs.append(cohen_kappa_score(y_true_arr[sample], y_pred_arr[sample]))
    def pct_ci(arr): return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

    return {
        "percent_agreement": float(acc),
        "percent_agreement_ci": pct_ci(acc_bs),
        "cohen_kappa": float(kappa),
        "cohen_kappa_ci": pct_ci(kap_bs),
        "labels": labels,
        "confusion_matrix": cm.tolist(),
    }

def load_prev_runs() -> pd.DataFrame:
    """Load all previous runs ONCE and create an efficient lookup"""
    dfs = []
    for f in OUTPUTS_DIR.glob("run_*.csv"):
        try:
            dfs.append(pd.read_csv(f))
        except Exception:
            pass
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

# --------------- DeepSeek API ---------------

load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    print("Error: DEEPSEEK_API_KEY not found in .env file. Exiting.")
    exit(1)

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"

def call_deepseek_api(messages: List[Dict[str, str]], temperature: float) -> Optional[str]:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 60,
        "response_format": {"type": "json_object"}
    }
    for attempt in range(1, MAX_RETRIES + 1):
        time.sleep(LLM_PAUSE_SECONDS)
        try:
            r = httpx.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=API_TIMEOUT_SECONDS)
            if r.status_code == 429:
                retry_after = r.headers.get("Retry-After")
                wait_secs = int(retry_after) if retry_after and retry_after.isdigit() else RETRY_AFTER_FALLBACK
                print(f"[429] rate-limited. wait {wait_secs}s (attempt {attempt}/{MAX_RETRIES})")
                time.sleep(wait_secs)
                if attempt < MAX_RETRIES: continue
                return None
            if 500 <= r.status_code < 600:
                print(f"[{r.status_code}] server error. retry (attempt {attempt}/{MAX_RETRIES})")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF_BASE ** (attempt - 1))
                    continue
                return None
            r.raise_for_status()
            data = r.json()
            content = data["choices"][0]["message"]["content"]
            return content
        except Exception as e:
            print(f"[ERROR] API attempt {attempt}: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_BASE ** (attempt - 1))
                continue
            return None
    return None

def build_message(row, prompt_text: str) -> str:
    title = (row.get("title") or "").strip()
    abstract = (row.get("abstract") or "").strip()
    return (prompt_text
            .replace("{title}", title)
            .replace("{abstract}", abstract))

def read_excel_inputs(excel_path: str, sheet_name: str = "in") -> pd.DataFrame:
    df = pd.read_excel(excel_path, sheet_name=sheet_name, engine="openpyxl")
    wanted = [
        "scopus_id","doi","title","abstract","corresponding_author_country",
        "cited_by_count","policy_claim_via_terms","llm_policy_claim","MW review","MW notes"
    ]
    cols = [c for c in wanted if c in df.columns]
    df = df[cols].copy()
    
    if "scopus_id" in df.columns:
        df["id"] = df["scopus_id"]
    else:
        df["id"] = np.arange(1, len(df)+1)
    
    if "abstract" not in df.columns: df["abstract"] = ""
    if "title" not in df.columns: df["title"] = ""
    return df

def process_row_concurrent(row_data: Dict, prompt_text: str, p_hash: str, 
                          temperature: float, cache_dict: Dict) -> Dict:
    """Process a single row - can be called concurrently"""
    sample_id = row_data["id"]
    
    # Check cache
    cache_key = (str(sample_id), p_hash, DEEPSEEK_MODEL, temperature)
    if cache_key in cache_dict:
        return cache_dict[cache_key]
    
    # Build and call API
    message_text = build_message(row_data, prompt_text)
    messages = [
        {"role":"system","content":"You are an expert academic classifier that only returns JSON."},
        {"role":"user","content": message_text}
    ]
    
    raw = call_deepseek_api(messages, temperature)
    if raw is None:
        llm_out, llm_label = "[ERROR]", "OTHER"
    else:
        llm_out, llm_label = postprocess_label_from_json_str(raw)
    
    result = {
        "id": sample_id,
        "scopus_id": row_data.get("scopus_id",""),
        "doi": row_data.get("doi",""),
        "title": row_data.get("title",""),
        "prompt_name": row_data["prompt_name"],
        "prompt_hash": p_hash,
        "model": DEEPSEEK_MODEL,
        "temperature": temperature,
        "llm_output": llm_out,
        "llm_label": llm_label
    }
    
    return result

def cmd_infer(args: argparse.Namespace) -> None:
    """Optimized inference with concurrent processing"""
    df = read_excel_inputs(args.excel, sheet_name=args.sheet)

    # CORRECTED: Sort by a stable ID to ensure deterministic selection with --limit
    #if "id" in df.columns:
    #    df = df.sort_values(by="id").reset_index(drop=True)

    if hasattr(args, "limit") and args.limit is not None:
        df = df.iloc[:args.limit]
        
    prompt_text = load_prompt(args.prompt_file)
    p_hash = prompt_hash(prompt_text)
    
    # Build efficient cache lookup
    cache_dict = {}
    if not args.no_cache:
        print("Loading cache...")
        prev = load_prev_runs()
        if not prev.empty:
            # Create dictionary for O(1) lookups
            for _, row in prev.iterrows():
                cache_key = (
                    str(row["id"]),
                    row["prompt_hash"],
                    row["model"],
                    row["temperature"]
                )
                cache_dict[cache_key] = row.to_dict()
            print(f"Loaded {len(cache_dict)} cached results")
    
    # Prepare all rows with needed data
    rows_to_process = []
    cached_results = []
    
    for _, row in df.iterrows():
        row_data = row.to_dict()
        row_data["prompt_name"] = args.prompt_name
        
        # Check if cached
        cache_key = (str(row_data["id"]), p_hash, DEEPSEEK_MODEL, args.temperature)
        if cache_key in cache_dict:
            cached_results.append(cache_dict[cache_key])
        else:
            rows_to_process.append(row_data)
    
    print(f"Using {len(cached_results)} cached results, processing {len(rows_to_process)} new items")
    
    # Process new rows concurrently
    new_results = []
    if rows_to_process:
        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_CONCURRENT_REQUESTS) as executor:
            # Submit all tasks
            future_to_row = {
                executor.submit(
                    process_row_concurrent, 
                    row_data, 
                    prompt_text, 
                    p_hash, 
                    args.temperature,
                    cache_dict
                ): row_data
                for row_data in rows_to_process
            }
            
            # Process results with progress bar
            with tqdm(total=len(rows_to_process), desc="Processing") as pbar:
                for i, future in enumerate(concurrent.futures.as_completed(future_to_row)):
                    try:
                        result = future.result()
                        # CORRECTED: Use lock for thread-safe append
                        with results_lock:
                            new_results.append(result)
                        pbar.update(1)
                        
                        # Save progress periodically
                        if (i + 1) % SAVE_PROGRESS_EVERY_N == 0:
                            temp_df = pd.DataFrame(cached_results + new_results)
                            temp_path = OUTPUTS_DIR / f"run_{args.prompt_name}_temp{args.temperature}_partial.csv"
                            temp_df.to_csv(temp_path, index=False)
                            pbar.set_postfix_str(f"Saved {len(temp_df)} results")
                            
                    except Exception as e:
                        row_data = future_to_row[future]
                        print(f"Error processing ID {row_data['id']}: {e}")
                        # Add error result
                        new_results.append({
                            "id": row_data["id"],
                            "scopus_id": row_data.get("scopus_id",""),
                            "doi": row_data.get("doi",""),
                            "title": row_data.get("title",""),
                            "prompt_name": args.prompt_name,
                            "prompt_hash": p_hash,
                            "model": DEEPSEEK_MODEL,
                            "temperature": args.temperature,
                            "llm_output": f"[ERROR: {str(e)}]",
                            "llm_label": "OTHER"
                        })
                        pbar.update(1)
    
    # Combine all results
    all_results = cached_results + new_results
    out_df = pd.DataFrame(all_results)
    
    # Save final results
    out_path = OUTPUTS_DIR / f"run_{args.prompt_name}_temp{args.temperature}_{now_tag()}.csv"
    out_df.to_csv(out_path, index=False)
    
    # Clean up partial file if it exists
    temp_path = OUTPUTS_DIR / f"run_{args.prompt_name}_temp{args.temperature}_partial.csv"
    if temp_path.exists():
        temp_path.unlink()
    
    print(f"[OK] Saved: {out_path}")
    print(f"Total processed: {len(out_df)} ({len(cached_results)} from cache, {len(new_results)} new)")

def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s.strip())

def _find_latest_run(prompt_name: str, temperature: float) -> Optional[str]:
    patt = f"run_{prompt_name}_temp{temperature}"
    cands = sorted(OUTPUTS_DIR.glob(f"{patt}_*.csv"), key=lambda p: p.stat().st_mtime)
    return str(cands[-1]) if cands else None

# [Other command functions remain the same - cmd_import_column, cmd_sweep, cmd_evaluate, cmd_compare, cmd_bridge]
# ... [Include all the other cmd_* functions unchanged from the original] ...

def cmd_import_column(args: argparse.Namespace) -> None:
    df = read_excel_inputs(args.excel, sheet_name=args.sheet)
    col = args.column
    if col not in df.columns:
        print(f"[ERROR] column '{col}' not found in sheet '{args.sheet}'.\nAvailable: {list(df.columns)}")
        return

    rows = []
    for _, r in df.iterrows():
        sample_id = r.get("id", "")
        val = r.get(col, None)

        llm_label = ""
        llm_output = ""

        if isinstance(val, str) and "{" in val and "}" in val:
            raw, lab = postprocess_label_from_json_str(val)
            llm_output = raw
            llm_label = lab
        else:
            lab = normalize_bool_token(val)
            if lab in ("YES", "NO"):
                llm_label = lab
                llm_output = '{"policy_claim": %s}' % ("true" if lab == "YES" else "false")
            else:
                llm_label = "OTHER" if lab else ""
                llm_output = "" if (val is None or (isinstance(val, float) and np.isnan(val))) else str(val)

        rows.append({
            "id": sample_id,
            "scopus_id": r.get("scopus_id",""),
            "doi": r.get("doi",""),
            "title": r.get("title",""),
            "prompt_name": args.name or f"excel:{col}",
            "prompt_hash": "excel",
            "model": "excel-column",
            "temperature": np.nan,
            "llm_output": llm_output,
            "llm_label": llm_label,
        })

    out_df = pd.DataFrame(rows)
    out_path = OUTPUTS_DIR / f"run_from_{_slug(col)}_{now_tag()}.csv"
    out_df.to_csv(out_path, index=False)
    print(f"[OK] Imported column '{col}' → {out_path}")

def cmd_sweep(args: argparse.Namespace) -> None:
    if not args.ref_column and not args.baseline_run:
        print("[INFO] No --ref-column or --baseline-run provided: will only run inference at each temperature.")
    for t in args.temps:
        print(f"\n[SWEEP] temperature = {t}")

        ns_infer = argparse.Namespace(
            excel=args.excel,
            sheet=args.sheet,
            prompt_file=args.prompt_file,
            prompt_name=args.prompt_name,
            temperature=float(t),
            no_cache=args.no_cache,
            limit=getattr(args, "limit", None),
        )
        cmd_infer(ns_infer)

        run_path = _find_latest_run(args.prompt_name, float(t))
        if not run_path:
            print("[WARN] Cannot locate the output run file just produced.")
            continue
        print(f"[SWEEP] run = {run_path}")

        if args.ref_column:
            ns_eval = argparse.Namespace(
                run_csv=run_path,
                excel=args.excel,
                sheet=args.sheet,
                ref_column=args.ref_column,
            )
            cmd_evaluate(ns_eval)

        if args.baseline_run:
            ns_cmp = argparse.Namespace(run_a=args.baseline_run, run_b=run_path)
            cmd_compare(ns_cmp)

def cmd_evaluate(args: argparse.Namespace) -> None:
    run_df = pd.read_csv(args.run_csv)
    df = read_excel_inputs(args.excel, sheet_name=args.sheet)
    
    merged = run_df.merge(df, on="id", how="left", suffixes=("", "_excel"))

    ref_col = args.ref_column
    if ref_col not in merged.columns:
        print(f"[ERROR] ref-column '{ref_col}' not in Excel sheet '{args.sheet}'.")
        return

    merged["_ref_norm"]  = merged[ref_col].apply(normalize_bool_token)
    merged["_pred_norm"] = merged["llm_label"].apply(normalize_bool_token)
    labeled = merged[merged["_ref_norm"].isin(["YES", "NO"])].copy()
    if labeled.empty:
        print("[ERROR] No usable human labels (YES/NO) found.")
        return
    y_true = labeled["_ref_norm"].tolist()
    y_pred = labeled["_pred_norm"].tolist()
    rep = agreement_report(y_true, y_pred)

    scopus_col = "scopus_id_excel" if "scopus_id_excel" in labeled.columns else ("scopus_id" if "scopus_id" in labeled.columns else None)
    title_col  = "title_excel"     if "title_excel"     in labeled.columns else ("title"     if "title"     in labeled.columns else None)

    cols = ["id"]
    for c in [scopus_col, title_col, args.ref_column, "llm_label", "llm_output", "abstract"]:
        if c and c in labeled.columns:
            cols.append(c)

    mask = (np.array(y_true) != np.array(y_pred))
    mism = labeled.loc[mask, cols]

    mism_path = REPORTS_DIR / f"mismatches_{now_tag()}.csv"
    mism.to_csv(mism_path, index=False)

    md_lines = []
    md_lines.append(f"# Concordance Report")
    md_lines.append(f"- Run CSV: `{args.run_csv}`")
    md_lines.append(f"- Excel: `{args.excel}` (sheet: `{args.sheet}`)")
    md_lines.append(f"- Reference column: `{ref_col}`\n")
    md_lines.append(f"- Evaluated (labeled only): {len(labeled)} / {len(merged)} rows\n")
    md_lines.append(f"**Percent agreement:** {rep['percent_agreement']:.3f} "
                    f"(95% CI {rep['percent_agreement_ci'][0]:.3f}–{rep['percent_agreement_ci'][1]:.3f})")
    md_lines.append(f"**Cohen's kappa:** {rep['cohen_kappa']:.3f} "
                    f"(95% CI {rep['cohen_kappa_ci'][0]:.3f}–{rep['cohen_kappa_ci'][1]:.3f})\n")
    md_lines.append("## Confusion matrix")
    md_lines.append(f"Labels: {rep['labels']}")
    md_lines.append("```\n" + "\n".join([str(r) for r in rep["confusion_matrix"]]) + "\n```")
    md_lines.append(f"\nMismatches saved to `{mism_path}`.\n")

    report_path = REPORTS_DIR / f"summary_{now_tag()}.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"[OK] Saved report: {report_path}")
    print(f"[OK] Saved mismatches: {mism_path}")

def cmd_compare(args: argparse.Namespace) -> None:
    a = pd.read_csv(args.run_a)
    b = pd.read_csv(args.run_b)
    m = a.merge(b, on="id", suffixes=("_a","_b"))
    valid = m[m["llm_label_a"].isin(["YES","NO"]) & m["llm_label_b"].isin(["YES","NO"])]
    rep = agreement_report(
        valid["llm_label_a"].apply(normalize_bool_token).tolist(),
        valid["llm_label_b"].apply(normalize_bool_token).tolist()
    )
    md_lines = []
    md_lines.append(f"# A/B Concordance")
    md_lines.append(f"- Run A: `{args.run_a}`")
    md_lines.append(f"- Run B: `{args.run_b}`\n")
    md_lines.append(f"**Percent agreement:** {rep['percent_agreement']:.3f}")
    md_lines.append(f"**Cohen's kappa:** {rep['cohen_kappa']:.3f}\n")
    md_lines.append("## Confusion matrix")
    md_lines.append(f"Labels: {rep['labels']}")
    md_lines.append("```\n" + "\n".join([str(r) for r in rep["confusion_matrix"]]) + "\n```")
    
    # Store the report in a variable and print it to the console
    report_text = "\n".join(md_lines)
    print(report_text) 
    
    out = REPORTS_DIR / f"ab_compare_{now_tag()}.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n[OK] Saved A/B report: {out}")

def cmd_bridge(args: argparse.Namespace) -> None:
    df = read_excel_inputs(args.excel, sheet_name=args.sheet)
    col = args.column
    if col not in df.columns:
        print(f"[ERROR] column '{col}' not found in sheet '{args.sheet}'.")
        print("Available columns:", list(df.columns))
        return

    rows = []
    for _, r in df.iterrows():
        lab = normalize_bool_token(r[col])
        if lab not in {"YES", "NO"}:
            continue
        rows.append({
            "id": r["id"],
            "scopus_id": r.get("scopus_id",""),
            "doi": r.get("doi",""),
            "title": r.get("title",""),
            "prompt_name": args.name or f"from_excel__{col}",
            "prompt_hash": "excel",
            "model": "excel_column",
            "temperature": 0.0,
            "llm_output": json.dumps({"policy_claim": lab=="YES"}),
            "llm_label": lab
        })

    out_df = pd.DataFrame(rows)
    OUTPUTS_DIR.mkdir(exist_ok=True, parents=True)
    out_path = OUTPUTS_DIR / f"run_{(args.name or col)}_excel_{now_tag()}.csv"
    out_df.to_csv(out_path, index=False)
    print(f"[OK] Saved: {out_path}  rows={len(out_df)}")

# --------------- CLI ---------------

def main():
    p = argparse.ArgumentParser(description="EC pipeline: infer / evaluate / compare")
    sub = p.add_subparsers(dest="cmd", required=True)

    # infer
    p1 = sub.add_parser("infer", help="Read Excel(in) → DeepSeek → outputs/run_*.csv")
    p1.add_argument("--excel", required=True, help="Path to Excel")
    p1.add_argument("--sheet", default="in", help="Sheet name (default: in)")
    p1.add_argument("--prompt-file", default=None, help="Prompt template file; if omitted, use built-in prompt")
    p1.add_argument("--prompt-name", required=True, help="A short name for this prompt/run (e.g., updated_v3)")
    p1.add_argument("--temperature", type=float, default=1.3, help="Sampling temperature (default: 1.3)")
    p1.add_argument("--no-cache", action="store_true", help="Disable cache from previous runs")
    p1.add_argument("--limit", type=int, default=None, help="Limit the number of rows to process for testing")
    p1.set_defaults(func=cmd_infer)

    # evaluate
    p2 = sub.add_parser("evaluate", help="Compute concordance vs a reference column in Excel (e.g., 'MW review')")
    p2.add_argument("--run-csv", required=True, help="outputs/run_*.csv from infer")
    p2.add_argument("--excel", required=True, help="Same Excel file used for infer")
    p2.add_argument("--sheet", default="in", help="Sheet name (default: in)")
    p2.add_argument("--ref-column", required=True, help="Reference column name (e.g., 'MW review' or 'policy_claim_via_terms')")
    p2.set_defaults(func=cmd_evaluate)

    # compare
    p3 = sub.add_parser("compare", help="Compare two runs (A/B)")
    p3.add_argument("--run-a", required=True)
    p3.add_argument("--run-b", required=True)
    p3.set_defaults(func=cmd_compare)

    # import-column
    p4 = sub.add_parser("import-column", help="Convert an Excel boolean/JSON column into a run_*.csv (no copy/paste).")
    p4.add_argument("--excel", required=True)
    p4.add_argument("--sheet", default="in")
    p4.add_argument("--column", required=True, help="Excel column name (e.g., 'updated LLM prompt claim')")
    p4.add_argument("--name", default=None, help="Optional prompt-name override for the generated run")
    p4.set_defaults(func=cmd_import_column)

    # sweep
    p5 = sub.add_parser("sweep", help="Infer across multiple temperatures, then evaluate and/or compare automatically.")
    p5.add_argument("--excel", required=True)
    p5.add_argument("--sheet", default="in")
    p5.add_argument("--prompt-file", default=None)
    p5.add_argument("--prompt-name", required=True)
    p5.add_argument("--temps", nargs="+", type=float, required=True, help="e.g. 1.3 0.9 0.7 0.3")
    p5.add_argument("--ref-column", default=None, help="Evaluate against this Excel column (e.g., 'MW review')")
    p5.add_argument("--baseline-run", default=None, help="Compare each new run against this run CSV")
    p5.add_argument("--no-cache", action="store_true")
    p5.add_argument("--limit", type=int, default=None, help="Limit rows processed per temperature for testing")
    p5.set_defaults(func=cmd_sweep)

    # bridge
    p6 = sub.add_parser("bridge", help="Build a run_*.csv from an existing Excel boolean column")
    p6.add_argument("--excel", required=True)
    p6.add_argument("--sheet", default="in")
    p6.add_argument("--column", required=True, help="e.g., 'llm_policy_claim' or 'updated LLM prompt claim'")
    p6.add_argument("--name", default=None, help="optional run name shown in prompt_name")
    p6.set_defaults(func=cmd_bridge)


    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()