#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluates evaluator performance on sanity check tasks (SC1 and SC2).

This script processes a judge's results file (JSON format) and filters for
tasks with idx 'SC1' and 'SC2', which have known correct answers.

- For task SC1, the correct answer is candidate 3.
- For task SC2, the correct answer is candidate 2.

The script checks if the evaluator ranked the correct candidate as #1.

Input:
  --results /path/to/results_A_V1.json  : Path to the results JSON file.

Output:
  A CSV file named 'evaluator_sanity_scores.csv' with the columns:
  - task_id: 'SC1' or 'SC2'
  - evaluator: The identifier for the evaluator (e.g., 'A_V1').
  - score: 1 if the evaluator's top choice was correct, 0 otherwise.
"""

import argparse
import json
import os
import re
from typing import List, Dict
import pandas as pd

# Copied from extract_model_ratings.py for robustness
def load_results(path: str) -> List[Dict]:
    """Load results supporting JSON array, JSON object, JSONL, or multi-JSON text.
    Return normalized rows with idx and parsed data.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    rows = []
    # Try parsing as a single JSON object or array
    try:
        obj = json.loads(raw)
        if isinstance(obj, list):
            rows = obj
        elif isinstance(obj, dict):
            rows = [obj]
    except json.JSONDecodeError:
        pass

    # If that fails, try JSONL
    if not rows:
        tmp = []
        ok = True
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                tmp.append(json.loads(line))
            except json.JSONDecodeError:
                ok = False
                tmp = []
                break
        if ok and tmp:
            rows = tmp

    # Fallback: multiple JSON blobs in one file
    if not rows:
        tmp = []
        stack, start = 0, None
        for i, ch in enumerate(raw):
            if ch == "{":
                if stack == 0:
                    start = i
                stack += 1
            elif ch == "}":
                stack -= 1
                if stack == 0 and start is not None:
                    block = raw[start:i+1]
                    try:
                        tmp.append(json.loads(block))
                    except json.JSONDecodeError:
                        pass
                    start = None
        rows = tmp

    # Normalize and filter for essential fields
    out = []
    for r in rows:
        idx = r.get("idx")
        p = r.get("parsed") or {}
        rating = p.get("rating")
        if idx and isinstance(rating, list):
            out.append({
                "idx": str(idx),
                "parsed": p,
            })
    return out


def main(results_path: str, outdir: str = None):
    # --- 1. Load Data ---
    if not os.path.exists(results_path):
        raise SystemExit(f"Error: Results file not found at '{results_path}'")

    results_fname = os.path.basename(results_path)
    if not results_fname.startswith("results_") or not results_fname.endswith(".json"):
        raise SystemExit(f"Error: Results filename '{results_fname}' must follow 'results_<EVALUATOR>.json' pattern.")

    evaluator = results_fname.replace("results_", "").replace(".json", "")
    results = load_results(results_path)

    # --- 2. Filter for Sanity Check Tasks ---
    sanity_results = [r for r in results if r.get("idx") in ("SC1", "SC2")]
    if not sanity_results:
        print(f"Info: No SC1 or SC2 tasks found in '{results_fname}'. Skipping.")
        return

    # --- 3. Process and Score Evaluator ---
    output_rows = []
    correct_candidates = {"SC1": 3, "SC2": 2}

    for r in sanity_results:
        task_id = r["idx"]
        rating = r.get("parsed", {}).get("rating", [])
        
        score = 0
        if rating and rating[0] == correct_candidates.get(task_id):
            score = 1
        
        output_rows.append({
            "task_id": task_id,
            "evaluator": evaluator,
            "score": score,
        })

    # --- 4. Output CSV ---
    if not output_rows:
        print("Could not generate any scores.")
        return

    output_df = pd.DataFrame(output_rows)
    
    # Create a unique filename, e.g., evaluator_sanity_scores_A_V1.csv
    output_filename = f"evaluator_sanity_scores_{evaluator}.csv"
    
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        output_path = os.path.join(outdir, output_filename)
    else:
        output_path = output_filename

    output_df.to_csv(output_path, index=False)

    print(f"✔ Successfully scored evaluator '{evaluator}' for {len(output_rows)} sanity check(s).")
    print(f"✔ Wrote: {output_path}")
    # print("\nPreview:")
    # print(output_df.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate evaluator performance on sanity checks (SC1, SC2).")
    ap.add_argument(
        "--results",
        required=True,
        help="Path to results JSON file (e.g., 'judge_results/results_A_V1.json')"
    )
    ap.add_argument(
        "--outdir",
        help="Optional: Directory to save the output CSV files."
    )
    args = ap.parse_args()
    main(args.results, args.outdir)
