#!/usr/bin/env python3
"""
Processes merged abstracts using the Deepseek LLM API to classify
policy claims based on a specific prompt. Accepts a user-selected JSON file
and outputs JSON and CSV files. Supports full processing or reproducible
random subsampling.
"""

import os
import json
import time
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from tqdm.auto import tqdm # Progress bar for large loops
import concurrent.futures # Needed for threading
import argparse # For command-line arguments

# --- Environment & API Setup ---
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    print("Error: DEEPSEEK_API_KEY not found in .env file. Exiting.")
    exit(1)

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat" # Hardcoded model
LLM_TEMPERATURE = 0.1 # Hardcoded temperature

# --- Retry Settings ---
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0
RETRY_AFTER_FALLBACK = 5

# --- LLM Configuration ---
LLM_PROMPT_TEMPLATE = """You are an expert policy analyst and academic researcher. Your sole function is to analyze a research abstract and determine if it makes a policy claim based on a strict set of rules and examples.

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
OUTPUT: {{"policy_claim": true}}

ABSTRACT: "We conducted a randomized controlled trial of a new diabetes drug. While the drug showed promise, there was no statistically significant improvement over existing treatments. Further investigation with a larger sample size is warranted to determine its efficacy."
OUTPUT: {{"policy_claim": false}}

ABSTRACT: "Our study shows a strong correlation between green space exposure and reduced symptoms of anxiety. To improve public health, municipal governments should enact zoning policies that mandate the inclusion of parks and green areas in all new housing developments."
OUTPUT: {{"policy_claim": true}}

ABSTRACT: "This paper reviews the historical literature concerning the United Kingdom's housing crisis. The data show that housing affordability has declined steadily since the 1980s across all regions, presenting a significant challenge for young adults."
OUTPUT: {{"policy_claim": false}}

ABSTRACT: "These results have important policy implications..."
OUTPUT: {{"policy_claim": true}}

## Output Format
Your response MUST be a valid JSON object and nothing else. Follow this exact schema:
{{"policy_claim": true/false}}

## Task
Now, analyze the following abstract based on all the rules and examples provided.

ABSTRACT: {abstract}
OUTPUT:"""

# --- API Call Parameters ---
NUM_CONCURRENT_REQUESTS = 5
LLM_PAUSE_SECONDS = 0.2
API_TIMEOUT_SECONDS = 90
SAVE_PROGRESS_EVERY_N_ROWS = 100

# --- Utility Functions ---
def call_deepseek_api(abstract_text: str) -> bool | None:
    """Calls the API and returns the parsed boolean result or None on failure."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }
    final_prompt = LLM_PROMPT_TEMPLATE.format(abstract=abstract_text)
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": "You are an expert academic classifier that only returns JSON."},
            {"role": "user", "content": final_prompt}
        ],
        "temperature": LLM_TEMPERATURE,
        "max_tokens": 60,
        "response_format": {"type": "json_object"}
    }

    for attempt in range(1, MAX_RETRIES + 1):
        time.sleep(LLM_PAUSE_SECONDS)
        try:
            response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=API_TIMEOUT_SECONDS)

            if response.status_code == 429:
                if attempt < MAX_RETRIES: continue
                return None

            if 500 <= response.status_code < 600:
                if attempt < MAX_RETRIES: continue
                return None

            response.raise_for_status()

            result = response.json()
            if not result.get("choices") or not result["choices"][0].get("message"):
                if attempt < MAX_RETRIES: continue
                return None

            llm_output_str = result["choices"][0]["message"].get("content", "").strip()

            try:
                start_index = llm_output_str.find('{')
                end_index = llm_output_str.rfind('}') + 1
                if start_index != -1 and end_index != 0:
                    json_str = llm_output_str[start_index:end_index]
                    parsed = json.loads(json_str)
                    if isinstance(parsed, dict) and "policy_claim" in parsed:
                        val = parsed["policy_claim"]
                        if isinstance(val, bool):
                            return val
                        if isinstance(val, str) and val.lower() in {"true", "false"}:
                            return val.lower() == "true"
                if attempt < MAX_RETRIES: continue
                return None

            except json.JSONDecodeError:
                if attempt < MAX_RETRIES: continue
                return None

        except requests.exceptions.RequestException:
            if attempt < MAX_RETRIES: continue
            return None

    return None

def load_and_prepare_data(input_file: Path, output_file: Path, sample_percent: int | None, seed: int | None) -> pd.DataFrame:
    if output_file.exists():
        print(f"Resuming from existing output file: {output_file}")
        df = pd.read_json(output_file, orient="records", lines=False)
    else:
        if not input_file.exists():
            print(f"Error: Input file not found: {input_file}")
            exit(1)
        print(f"Loading initial data from: {input_file}")
        df_full = pd.read_json(input_file, orient="records")

        if sample_percent:
            frac = sample_percent / 100.0
            print(f"Taking a {sample_percent}% random sample with seed {seed}.")
            df = df_full.sample(frac=frac, random_state=seed).copy()
        else:
            df = df_full.copy()
    
    if 'llm_policy_claim' not in df.columns:
        df['llm_policy_claim'] = pd.NA

    df['llm_policy_claim'] = df['llm_policy_claim'].astype('boolean')

    initial_count = len(df)
    if 'abstract' not in df.columns: df['abstract'] = ''
    df = df.dropna(subset=['abstract'])
    df = df[df['abstract'].str.strip().str.len() >= 30]
    print(f"Removed {initial_count - len(df)} abstracts with short/missing abstracts.")
    return df.reset_index(drop=True)

def save_results(df: pd.DataFrame, json_path: Path, csv_path: Path):
    try:
        df_save = df.copy()
        df_save['llm_policy_claim'] = df_save['llm_policy_claim'].apply(lambda x: None if pd.isna(x) else bool(x))
        df_save.to_json(json_path, orient="records", indent=2, default_handler=str)
        df_save.to_csv(csv_path, index=False)
    except Exception as e:
        print(f"Error saving results: {e}")

def main(args):
    start_time = time.time()
    input_path = Path(args.input_file)
    output_suffix = "_LLM"
    if args.sample:
        output_suffix = f"_sample{args.sample}_seed{args.seed}{output_suffix}"

    output_file_llm = input_path.with_name(f"{input_path.stem}{output_suffix}.json")
    csv_output_llm = input_path.with_name(f"{input_path.stem}{output_suffix}.csv")
    
    print("=" * 50)
    print("🚀 Starting LLM Abstract Classification 🚀")
    print(f"Input file: {input_path}")
    print(f"Output files: \n  - {output_file_llm}\n  - {csv_output_llm}")
    print("=" * 50)

    df_prepared = load_and_prepare_data(input_path, output_file_llm, args.sample, args.seed)
    
    tasks_to_submit = []
    indices_to_process = []
    for index, row in df_prepared.iterrows():
        if pd.notna(row.get('llm_policy_claim')):
            continue
        if row.get('abstract', ''):
            tasks_to_submit.append(row['abstract'])
            indices_to_process.append(index)

    if not tasks_to_submit:
        print("✅ No new abstracts to process. All done!")
    else:
        print(f"Found {len(tasks_to_submit)} new abstracts to classify...")
        
        with tqdm(total=len(tasks_to_submit), desc="Classifying Abstracts") as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_CONCURRENT_REQUESTS) as executor:
                future_to_idx = {
                    executor.submit(call_deepseek_api, abstract_text): original_idx
                    for abstract_text, original_idx in zip(tasks_to_submit, indices_to_process)
                }
                for i, future in enumerate(concurrent.futures.as_completed(future_to_idx)):
                    original_idx = future_to_idx[future]
                    try:
                        result = future.result()
                        df_prepared.loc[original_idx, 'llm_policy_claim'] = result

                        # --- ADD THIS CODE BLOCK ---
                        # Get a snippet of the abstract to display
                        abstract_snippet = df_prepared.loc[original_idx, 'abstract'][:70] + "..."
                        # Update the progress bar with the latest result
                        pbar.set_postfix_str(f"Result: {str(result):<5} | Abstract: '{abstract_snippet}'")
                        # --- END OF ADDED CODE ---

                    except Exception as exc:
                        print(f"Critical error processing future for index {original_idx}: {exc}")
                        df_prepared.loc[original_idx, 'llm_policy_claim'] = pd.NA
                    
                    pbar.update(1)

                    if (i + 1) % SAVE_PROGRESS_EVERY_N_ROWS == 0:
                        save_results(df_prepared, output_file_llm, csv_output_llm)

        print("\n✅ Classification Complete!")
        save_results(df_prepared, output_file_llm, csv_output_llm)
    
    # Final Summary
    final_classified_count = df_prepared['llm_policy_claim'].notna().sum()
    final_true_count = (df_prepared['llm_policy_claim'] == True).sum()
    print(f"\nTotal abstracts in output file: {len(df_prepared)}")
    print(f"Successfully classified: {final_classified_count} ({final_true_count} True)")
    print(f"Failed/unclassified: {len(df_prepared) - final_classified_count}")
    print(f"Total script execution time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classify research abstracts for policy claims using the Deepseek API.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("input_file", type=str, help="Path to the input JSON file.")
    parser.add_argument("-s", "--sample", type=int, metavar="X", choices=range(1, 100), help="Process a random X%% subsample.")
    parser.add_argument("--seed", type=int, metavar="N", default=42, help="Random seed for sampling. Default: 42.")
    
    args = parser.parse_args()
    main(args)