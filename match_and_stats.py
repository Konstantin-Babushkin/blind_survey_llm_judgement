#!/usr/bin/env python3
"""
Match candidates from results with model names from matcher for each question,
then provide overall stats by question_id and model_name (NO data edits required).

Inputs (override via CLI if needed):
  --results ./judge_results/results_A_V1.json
  --matcher ./judge_prompts/matcher/matcher_A_V1.csv
Outputs:
  matched_rows.csv
  stats_by_q_model.csv
"""

import argparse
import json
import os
import re
from collections import Counter

import pandas as pd


# -----------------------------
# Robust loaders for your files
# -----------------------------
def load_results(results_path: str) -> list[dict]:
    """
    Your results JSON has relevant fields under 'parsed':
      parsed.form_id, parsed.criterion, parsed.rating (list of candidate indexes, 1=best order)
    Supports JSON array, JSON object, JSONL, or multiple JSON blobs.
    """
    with open(results_path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    # Try JSON
    rows = []
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
        jsonl_ok = True
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                tmp.append(json.loads(line))
            except json.JSONDecodeError:
                jsonl_ok = False
                tmp = []
                break
        if jsonl_ok and tmp:
            rows = tmp

    # Fallback: greedy brace extraction for multiple blobs
    if not rows:
        brace_stack, start = 0, None
        tmp = []
        for i, ch in enumerate(raw):
            if ch == "{":
                if brace_stack == 0:
                    start = i
                brace_stack += 1
            elif ch == "}":
                brace_stack -= 1
                if brace_stack == 0 and start is not None:
                    block = raw[start : i + 1]
                    try:
                        tmp.append(json.loads(block))
                    except json.JSONDecodeError:
                        pass
                    start = None
        rows = tmp

    # Normalize to a simple list of dicts we actually use
    out = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        p = r.get("parsed") or {}
        form_id = p.get("form_id")
        criterion = p.get("criterion")
        rating = p.get("rating")
        if form_id and isinstance(rating, list) and rating:
            cleaned_rating = []
            for x in rating:
                try:
                    cleaned_rating.append(int(x))
                except Exception:
                    continue
            if cleaned_rating:
                out.append(
                    {
                        "form_id": str(form_id),
                        "criterion": str(criterion) if criterion is not None else "",
                        "rating": cleaned_rating,
                    }
                )
    return out


def load_matcher(matcher_path: str) -> pd.DataFrame:
    """
    Your matcher CSV columns: task_id, prompt_label, candidate_index, model
    Returns DataFrame with those columns + a normalized prompt_label (spaces removed at the tail).
    """
    m = pd.read_csv(matcher_path)
    required = {"task_id", "prompt_label", "candidate_index", "model"}
    missing = required - set(m.columns)
    if missing:
        raise ValueError(f"Matcher CSV missing required columns: {sorted(missing)}")

    m = m.copy()
    m["task_id"] = m["task_id"].astype(str)
    m["candidate_index"] = m["candidate_index"].astype(int)
    m["model_name"] = m["model"].astype(str)

    # normalize prompt_label by removing spaces (criterion tail sometimes has spaces)
    m["prompt_label_norm"] = m["prompt_label"].astype(str).str.replace(" ", "", regex=False)
    return m[["task_id", "prompt_label", "prompt_label_norm", "candidate_index", "model_name"]]


# -----------------------------
# Helpers
# -----------------------------
QID_RE = re.compile(r"\bQ\d+\b")
TASK_RE = re.compile(r"\bA\d+\b")


def extract_question_id(form_id: str) -> str:
    m = QID_RE.search(form_id)
    return m.group(0) if m else form_id


def extract_task_id(form_id: str) -> str:
    m = TASK_RE.search(form_id)
    return m.group(0) if m else form_id


def make_prompt_label(task_id: str, question_id: str, criterion: str) -> str:
    # matcher uses no spaces in the criterion part
    return f"{task_id}_{question_id}_{(criterion or '').replace(' ', '')}"


# -----------------------------
# Core pipeline
# -----------------------------
def main(results_path: str, matcher_path: str, out_dir: str = "."):
    results = load_results(results_path)
    if not results:
        raise RuntimeError("No usable rows parsed from results file.")

    matcher_df = load_matcher(matcher_path)

    # Build normalized results df
    res_rows = []
    for r in results:
        form_id = r["form_id"]
        criterion = r["criterion"]
        rating = r["rating"]  # list of candidate indexes in ranked order
        qid = extract_question_id(form_id)
        tid = extract_task_id(form_id)
        prompt_label = make_prompt_label(tid, qid, criterion)
        res_rows.append(
            {
                "form_id": form_id,
                "question_id": qid,
                "task_id": tid,
                "criterion": criterion,
                "rating": rating,
                "prompt_label": prompt_label,
            }
        )
    res_df = pd.DataFrame(res_rows)

    # Join via prompt_label (after removing spaces on matcher side)
    joined = res_df.merge(
        matcher_df,
        left_on="prompt_label",
        right_on="prompt_label_norm",
        how="inner",
        validate="many_to_many",
    )

    if joined.empty:
        sample = res_df[["prompt_label"]].head(10).to_dict(orient="records")
        raise RuntimeError(
            "No matches found via prompt_label. "
            f"Example expected labels: {sample}. "
            "Ensure matcher 'prompt_label' follows '{task_id}_{Qxx}_{criterionWithoutSpaces}'."
        )

    # Expand rankings into one row per candidate with model_name + rank
    expanded = []
    for _, row in joined.iterrows():
        # convert rating to rank map: candidate_index -> position (1 = best)
        rank_by_idx = {int(idx): pos for pos, idx in enumerate(row["rating"], start=1)}
        cand_idx = int(row["candidate_index"])
        if cand_idx in rank_by_idx:
            expanded.append(
                {
                    "form_id": row["form_id"],
                    "question_id": row["question_id"],
                    "criterion": row["criterion"],
                    "candidate_index": cand_idx,
                    "model_name": row["model_name"],
                    "rank": rank_by_idx[cand_idx],  # 1 = best
                }
            )
    if not expanded:
        raise RuntimeError("Matched prompts found, but no candidate indexes overlapped the ratings.")

    matched_df = (
        pd.DataFrame(expanded)
        .sort_values(["question_id", "criterion", "rank"])
        .reset_index(drop=True)
    )

    # Aggregate stats by (question_id, model_name)
    grouped = matched_df.groupby(["question_id", "model_name"])
    stats_df = (
        grouped.agg(
            n=("rank", "size"),
            wins=("rank", lambda s: int((s == 1).sum())),
            mean_rank=("rank", "mean"),
            median_rank=("rank", "median"),
        )
        .reset_index()
        .sort_values(["question_id", "wins", "mean_rank", "model_name"], ascending=[True, False, True, True])
    )

    # ---- FIX: robust per-position rank counts (no NaN/float 'pos_counts') ----
    pos_counts = (
        grouped["rank"]
        .apply(lambda s: dict(Counter(s)))
        .reset_index(name="pos_counts")
    )
    # Ensure every cell is a dict (guard against NaNs)
    pos_counts["pos_counts"] = pos_counts["pos_counts"].apply(
        lambda d: d if isinstance(d, dict) else {}
    )
    # Collect all rank positions that appear anywhere
    all_ranks = sorted({int(k) for d in pos_counts["pos_counts"] for k in d})
    # Expand into wide columns rank_1, rank_2, ...
    for rnk in all_ranks:
        pos_counts[f"rank_{rnk}"] = pos_counts["pos_counts"].apply(lambda d, r=rnk: int(d.get(r, 0)))
    pos_counts = pos_counts.drop(columns=["pos_counts"])

    stats_df = stats_df.merge(pos_counts, on=["question_id", "model_name"], how="left")

    # Output
    os.makedirs(out_dir, exist_ok=True)
    matched_path = os.path.join(out_dir, "matched_rows.csv")
    stats_path = os.path.join(out_dir, "stats_by_q_model.csv")
    matched_df.to_csv(matched_path, index=False)
    stats_df.to_csv(stats_path, index=False)

    print(f"✔ Wrote: {matched_path}")
    print(f"✔ Wrote: {stats_path}")
    print("Preview (first 10 stats rows):")
    print(stats_df.head(10).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="./judge_results/results_A_V1.json", help="Path to results JSON")
    ap.add_argument("--matcher", default="./judge_prompts/matcher/matcher_A_V1.csv", help="Path to matcher CSV")
    ap.add_argument("--outdir", default="./stats", help="Where to save CSV outputs")
    args = ap.parse_args()
    main(args.results, args.matcher, args.outdir)
