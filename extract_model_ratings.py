#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract per-model ratings from results, match to model names via matcher,
and output the ratings of each model for each task.

Inputs (override via CLI):
  --results /path/to/results_A_V3.json
  --matcher /path/to/matcher_A_V3.csv

Outputs (written under: OUTDIR/results_matched_<A_Vx>/):
  ratings_long.csv         # long form: task_id, question_id, criterion, model_name, rank, task_type, evaluator
  ratings_wide_by_task.csv # pivot: one row per (task_id, question_id, criterion), model columns = rank, plus task_type, evaluator

Assumptions:
- Results JSON has fields under parsed.{form_id, criterion, rating}.
- form_id looks like: "<who>|<task_type>|<TaskID>|<Qxx>", e.g. "A-Volunteer1|Summarise|A3|Q02".
- Matcher CSV has: task_id, prompt_label, candidate_index, model
  where prompt_label == "{task_id}_{Qxx}_{criterionWithoutSpaces}".

Sanity checks:
- Use --sanity_pattern to mark "special" task_ids (regex). For those, verify that
  all candidate_index present in matcher for each prompt_label exist in the results' rating list.
  If any mismatch is found, the script exits with an error explaining which prompt failed.
"""

import argparse
import json
import os
import re
from typing import List, Dict, Tuple
import pandas as pd

# ---------- Robust loaders ----------

def load_results(path: str) -> List[Dict]:
    """Load results supporting JSON array, JSON object, JSONL, or multi-JSON text.
    Return normalized rows with form_id, criterion, rating (list[int])."""
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    rows = []
    # Try JSON
    try:
        obj = json.loads(raw)
        if isinstance(obj, list):
            rows = obj
        elif isinstance(obj, dict):
            rows = [obj]
    except json.JSONDecodeError:
        pass

    # Try JSONL
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

    # Normalize
    out = []
    for r in rows:
        p = r.get("parsed") or {}
        form_id = p.get("form_id")
        criterion = p.get("criterion")
        rating = p.get("rating")
        if form_id and isinstance(rating, list) and rating:
            clean = []
            for x in rating:
                try:
                    clean.append(int(x))
                except Exception:
                    continue
            if clean:
                out.append({
                    "form_id": str(form_id),
                    "criterion": str(criterion) if criterion is not None else "",
                    "rating": clean,
                })
    return out


def load_matcher(path: str) -> pd.DataFrame:
    """Load matcher with expected columns: task_id, prompt_label, candidate_index, model."""
    df = pd.read_csv(path)
    expect = {"task_id", "prompt_label", "candidate_index", "model"}
    missing = expect - set(df.columns)
    if missing:
        raise ValueError(f"Matcher CSV missing columns: {sorted(missing)}")

    df = df.copy()
    df["task_id"] = df["task_id"].astype(str)
    df["candidate_index"] = df["candidate_index"].astype(int)
    df["model_name"] = df["model"].astype(str)
    # normalize prompt_label by removing spaces in the criterion tail
    df["prompt_label_norm"] = df["prompt_label"].astype(str).str.replace(" ", "", regex=False)
    return df[["task_id", "prompt_label", "prompt_label_norm", "candidate_index", "model_name"]]


# ---------- ID helpers ----------

QID_RE = re.compile(r"\bQ\d+\b")
TASK_RE = re.compile(r"\b[A-Z]\d+\b")  # A3, B12, C5...
# form_id expected structure: "<who>|<task_type>|<TaskID>|<Qxx>"
def extract_question_id(form_id: str) -> str:
    m = QID_RE.search(form_id)
    return m.group(0) if m else form_id

def extract_task_id(form_id: str) -> str:
    m = TASK_RE.search(form_id)
    return m.group(0) if m else form_id

def extract_task_type(form_id: str) -> str:
    """Extract task_type from form_id by taking the 2nd '|' segment when available.
    Fallback: first non TaskID/QID token that is alphabetic; else return empty."""
    parts = str(form_id).split("|")
    if len(parts) >= 2:
        # Often the 2nd segment is the task type (e.g., "Summarise", "Idea-to-Text")
        candidate = parts[1].strip()
        if candidate:
            return candidate
    # Fallback: search any segment that isn't TaskID/QID and looks alpha-ish
    for seg in parts:
        if not TASK_RE.fullmatch(seg) and not QID_RE.fullmatch(seg) and re.search(r"[A-Za-z]", seg):
            return seg.strip()
    return ""

def make_prompt_label(task_id: str, question_id: str, criterion: str) -> str:
    return f"{task_id}_{question_id}_{(criterion or '').replace(' ', '')}"



# ---------- Core ----------

def main(results_path: str, matcher_path: str, outdir: str, sanity_pattern: str):
    results = load_results(results_path)
    if not results:
        raise SystemExit("No usable rows parsed from results file.")

    matcher = load_matcher(matcher_path)

    # Build normalized results DF with join key + task_type
    res_rows = []
    for r in results:
        form_id = r["form_id"]
        criterion = r["criterion"]
        rating = r["rating"]
        task_id = extract_task_id(form_id)
        question_id = extract_question_id(form_id)
        task_type = extract_task_type(form_id)
        prompt_label = make_prompt_label(task_id, question_id, criterion)
        res_rows.append({
            "form_id": form_id,
            "task_id": task_id,
            "question_id": question_id,
            "task_type": task_type,
            "criterion": criterion,
            "rating": rating,
            "prompt_label": prompt_label,
        })
    res_df = pd.DataFrame(res_rows)

    # Join via prompt_label (left side already space-less; matcher uses *_norm)
    joined = res_df.merge(
        matcher,
        left_on="prompt_label",
        right_on="prompt_label_norm",
        how="inner",
        validate="many_to_many",
    )
    if joined.empty:
        sample = res_df[["prompt_label"]].head(10).to_dict(orient="records")
        raise SystemExit(
            "No matches found via prompt_label. "
            f"Example expected labels: {sample}. "
            "Ensure matcher 'prompt_label' equals '{task_id}_{Qxx}_{criterionWithoutSpaces}'."
        )

    # Expand ratings: map candidate_index -> rank (1 = best)
    expanded = []
    for _, row in joined.iterrows():
        rank_by_idx = {int(idx): pos for pos, idx in enumerate(row["rating"], start=1)}
        cand_idx = int(row["candidate_index"])
        if cand_idx in rank_by_idx:
            expanded.append({
                "task_id": row["task_id_x"],           # from res_df
                "question_id": row["question_id"],
                "task_type": row["task_type"],
                "criterion": row["criterion"],
                "model_name": row["model_name"],
                "candidate_index": cand_idx,
                "rank": rank_by_idx[cand_idx],          # 1 = best
            })

    if not expanded:
        raise SystemExit("Matched prompts found, but no candidate indexes overlapped the ratings.")

    long_df = pd.DataFrame(expanded).sort_values(
        ["task_id", "question_id", "criterion", "rank", "model_name"]
    ).reset_index(drop=True)

    # Pivot to wide per task/question with model columns showing rank
    wide_df = long_df.pivot_table(
        index=["task_id", "question_id", "task_type", "criterion"],
        columns="model_name",
        values="rank",
        aggfunc="min"  # there should be one value; min is a safe resolver
    ).reset_index()

    # Write outputs (add evaluator + task_type columns)
    base = os.path.basename(results_path)   # e.g. results_A_V1.json
    stem = os.path.splitext(base)[0]        # → results_A_V1
    evaluator = stem.split("results_")[-1]  # → A_V1
    subdir = os.path.join(outdir, f"results_matched_{evaluator}")
    os.makedirs(subdir, exist_ok=True)

    long_df["evaluator"] = evaluator
    wide_df["evaluator"] = evaluator

    long_path = os.path.join(subdir, "ratings_long.csv")
    wide_path = os.path.join(subdir, "ratings_wide_by_task.csv")
    long_df.to_csv(long_path, index=False)
    wide_df.to_csv(wide_path, index=False)

    print(f"✔ Wrote: {long_path}")
    print(f"✔ Wrote: {wide_path}")
    print("\nPreview (long):")
    print(long_df.head(10).to_string(index=False))
    print("\nPreview (wide):")
    print(wide_df.head(10).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="/mnt/data/results_A_V3.json", help="Path to results JSON")
    ap.add_argument("--matcher", default="/mnt/data/matcher_A_V3.csv", help="Path to matcher CSV")
    ap.add_argument("--outdir", default="./judge_results/matched", help="Output directory")
    ap.add_argument(
        "--sanity_pattern",
        default="^(SC1|SC2)$",
        help="Regex for special 'sanity' task_ids (e.g. '^(A0|SANITY)$' or '^[ABC]0$'). If set, strict checks run.",
    )
    args = ap.parse_args()
    main(args.results, args.matcher, args.outdir, args.sanity_pattern)
