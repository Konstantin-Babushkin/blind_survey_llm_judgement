#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Merge all normalized_scores_by_task_model.csv files under a given directory
into one master CSV, then write 3 CSVs by task type:
  - all_normalized_scores_idea_to_text.csv
  - all_normalized_scores_summarise.csv
  - all_normalized_scores_interpret.csv

Each file is naturally sorted by task_id (letters, then number): A1, A2, ..., B1, B2

Usage:
  python merge_normalized_scores.py --indir judge_results/matched \
    --outdir judge_results/matched
"""

import argparse
import os
import re
import sys
import pandas as pd
import numpy as np

WANTED_TASK_TYPES = ["Idea-to-Text", "Summarise", "Interpret"]

def naturalize_task_id_columns(df: pd.DataFrame, task_id_col: str = "task_id") -> pd.DataFrame:
    """
    Add helper columns to naturally sort task_id by letter(s) then number.
    Example order: A1, A2, A10, B1, B2, ...
    If the number is missing, it sorts after numbered ones within the same prefix.
    """
    if task_id_col not in df.columns:
        raise KeyError(f"Expected '{task_id_col}' column in data.")

    s = df[task_id_col].astype(str)

    # Extract leading letters and the first number sequence
    letters = s.str.extract(r'^\s*([A-Za-z]+)', expand=False).str.upper()
    numbers = s.str.extract(r'(\d+)', expand=False)

    # Convert to numeric; missing numbers -> +inf so they come last within the same letters
    num_vals = pd.to_numeric(numbers, errors="coerce")
    num_vals = num_vals.astype(float).fillna(np.inf)

    # Missing letters -> empty string, so they sort before lettered IDs
    letters = letters.fillna("")

    df = df.copy()
    df["_task_alpha"] = letters
    df["_task_num"] = num_vals
    return df

def sort_by_task_id(df: pd.DataFrame, task_id_col: str = "task_id") -> pd.DataFrame:
    df = naturalize_task_id_columns(df, task_id_col=task_id_col)
    df = df.sort_values(by=["_task_alpha", "_task_num", task_id_col], kind="mergesort", ignore_index=True)
    return df.drop(columns=["_task_alpha", "_task_num"], errors="ignore")

def main(indir: str, outdir: str):
    all_dfs = []
    for root, _, files in os.walk(indir):
        for f in files:
            if f == "normalized_scores_by_task_model.csv":
                path = os.path.join(root, f)
                try:
                    df = pd.read_csv(path)

                    # Add evaluator from the folder name (results_matched_X)
                    evaluator = os.path.basename(root).replace("results_matched_", "")
                    df["evaluator"] = evaluator

                    all_dfs.append(df)
                    print(f"✔ Loaded {path} ({len(df)} rows)")
                except Exception as e:
                    print(f"!!! Failed to load {path}: {e}", file=sys.stderr)

    if not all_dfs:
        raise SystemExit("No normalized_scores_by_task_model.csv files found.")

    master = pd.concat(all_dfs, ignore_index=True)

    # Validate that we have the needed columns
    if "task_type" not in master.columns:
        raise SystemExit("Input CSVs must contain a 'task_type' column.")
    if "task_id" not in master.columns:
        raise SystemExit("Input CSVs must contain a 'task_id' column.")

    os.makedirs(outdir, exist_ok=True)

    # Filter and write one file per requested task type
    written = 0
    for tt in WANTED_TASK_TYPES:
        subset = master[master["task_type"] == tt].copy()
        if subset.empty:
            print(f"• Skipping '{tt}': no rows found.")
            continue

        subset = sort_by_task_id(subset, task_id_col="task_id")

        # File name: normalized and readable
        safe = tt.lower().replace(" ", "_").replace("-", "_")
        out_path = os.path.join(outdir, f"all_normalized_scores_{safe}.csv")

        subset.to_csv(out_path, index=False)
        print(f"✔ Wrote {out_path} ({len(subset)} rows)")
        written += 1

    if written == 0:
        raise SystemExit("No rows matched any of the expected task types: "
                         + ", ".join(WANTED_TASK_TYPES))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True, help="Directory to search (recursively)")
    ap.add_argument("--outdir", required=True, help="Directory to write the 3 CSVs")
    args = ap.parse_args()
    main(args.indir, args.outdir)
