#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute normalized scores from ratings_wide_by_task.csv.

Input format (typical):
  task_id, task_type, question_id, criterion, <model_1>, <model_2>, ... [, evaluator]
  where each <model_X> cell holds an integer rank (1 = best), or NaN if absent

Output (wide):
  task_id, task_type, model_name, evaluator, <criterion_1>, <criterion_2>, ...

Notes:
- n is computed per row as the count of non-null model ranks on that row.
- For rows with n <= 1 (degenerate), the score is set to 1.0 if rank == 1, else NaN.
- If there are multiple rows for the same (task_id, task_type, evaluator, criterion, model),
  scores are averaged.
"""

import argparse
import os
from typing import List, Optional
import pandas as pd


def main(input_csv: str, output_csv: str, long_output_csv: Optional[str] = None):
    # Read
    df = pd.read_csv(input_csv)

    # Identify id columns (case-insensitive map)
    cols_lower = {c.lower(): c for c in df.columns}

    id_cols: List[str] = []
    for key in ["task_id", "task_type", "question_id", "criterion", "evaluator"]:
        if key in cols_lower:
            id_cols.append(cols_lower[key])

    # All other columns are presumed to be model names (holding ranks)
    model_cols = [c for c in df.columns if c not in id_cols]
    if not model_cols:
        raise SystemExit("No model columns detected. Ensure the input has model name columns with rank values.")

    # Count models per row
    df["_n_models"] = df[model_cols].notna().sum(axis=1)

    # Melt to long
    long = df.melt(
        id_vars=id_cols + ["_n_models"],
        value_vars=model_cols,
        var_name="model_name",
        value_name="rank",
    ).dropna(subset=["rank"]).copy()

    # Ensure numeric rank
    long["rank"] = pd.to_numeric(long["rank"], errors="coerce")
    long = long.dropna(subset=["rank"])

    # Normalized score
    def norm_score(n, r):
        n = int(n)
        r = float(r)
        if n <= 1:
            return 1.0 if r == 1.0 else float("nan")
        return 1.0 - (r - 1.0) / (n - 1.0)

    long["normalized_score"] = long.apply(lambda row: norm_score(row["_n_models"], row["rank"]), axis=1)

    # Keep only needed columns
    keep_cols = [c for c in [
        cols_lower.get("task_id"),
        cols_lower.get("task_type"),
        cols_lower.get("question_id"),
        cols_lower.get("criterion"),
        cols_lower.get("evaluator"),
    ] if c]
    out_long_cols = keep_cols + ["model_name", "rank", "normalized_score"]
    long_clean = long[out_long_cols].copy()

    # Aggregate duplicates by mean
    group_keys = []
    if cols_lower.get("task_id"):    group_keys.append(cols_lower["task_id"])
    if cols_lower.get("task_type"):  group_keys.append(cols_lower["task_type"])
    if cols_lower.get("evaluator"):  group_keys.append(cols_lower["evaluator"])
    if cols_lower.get("criterion"):  group_keys.append(cols_lower["criterion"])
    group_keys.append("model_name")

    agg_long = (
        long_clean.groupby(group_keys, dropna=False)["normalized_score"]
        .mean()
        .reset_index()
    )

    # Pivot to wide: index = (task_id, task_type, model_name[, evaluator])
    index_keys = []
    if cols_lower.get("task_id"):    index_keys.append(cols_lower["task_id"])
    if cols_lower.get("task_type"):  index_keys.append(cols_lower["task_type"])
    index_keys.append("model_name")
    if cols_lower.get("evaluator"):  index_keys.append(cols_lower["evaluator"])

    criterion_col = cols_lower.get("criterion", "criterion")

    pivot = agg_long.pivot_table(
        index=index_keys,
        columns=criterion_col,
        values="normalized_score",
        aggfunc="mean",
    ).reset_index()

    # Order columns: ids first, then criteria in specified order
    id_order = index_keys
    desired_order = [
        "Fluency",
        "Coherence",
        "Conciseness",
        "Accuracy",
        "Sentiment",
        "Motivation",
        "Constructiveness",
        "FinalChoice",
    ]
    crit_cols = [c for c in desired_order if c in pivot.columns]
    extras = [c for c in pivot.columns if c not in id_order + crit_cols]
    pivot = pivot[id_order + crit_cols + extras]

    # Write outputs
    out_dir = os.path.dirname(output_csv) or "."
    os.makedirs(out_dir, exist_ok=True)
    pivot.to_csv(output_csv, index=False)
    print(f"✔ Wrote wide (per task × model × evaluator): {output_csv}")

    if long_output_csv:
        out_long_dir = os.path.dirname(long_output_csv) or "."
        os.makedirs(out_long_dir, exist_ok=True)
        long_clean.to_csv(long_output_csv, index=False)
        print(f"✔ Wrote long (per task × question × model): {long_output_csv}")

    # Peek
    print("\nPreview:")
    print(pivot.head(10).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="ratings_wide_by_task.csv", help="Path to ratings_wide_by_task.csv")
    ap.add_argument("--output", default="normalized_scores_by_task_model.csv", help="Output CSV path (wide table)")
    ap.add_argument("--output_long", default="", help="Optional: write long table too")
    args = ap.parse_args()

    long_out = args.output_long if args.output_long.strip() else None
    main(args.input, args.output, long_out)
