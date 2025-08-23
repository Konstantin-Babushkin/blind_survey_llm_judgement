#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build LLM-judge prompts from a CSV of tasks.

Input CSV (example columns, as in your sample):
- Volunteer, Category, Task ID, Original, Gemini, DeepSeek, Flash, Qwen

Usage:
  python build_prompts.py \
      --csv "handpick - tasks A.csv" \
      --volunteer "Volunteer 1" \
      --group A \
      --outdir ./prompts_A_V1 \
      --one-file all_prompts.txt \
      --shuffle \
      --seed 42

Notes:
- For ranking criteria, the LLM should return a strict permutation in "rating": [..].
- For Accuracy, the LLM should return a subset in "rating": [..] (e.g., [1,3,4]).
- Candidate order can be shuffled (default off; enable with --shuffle).
"""

import argparse
import os
import random
import sys
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple

# ---- Criteria definitions (text and JSON expectations) ----

CRITERIA = [
    ("Fluency", "Rank from most to least smooth and error-free.", "rank"),
    ("Coherence", "Rank from most to least logical and easy to follow.", "rank"),
    ("Conciseness", "Rank from most to least clear and compact.", "rank"),
    ("Accuracy", "Select all texts that are factually correct given the Original.", "subset"),
    ("Sentiment match", "Rank from most to least similar to the Original in style and sentiment.", "rank"),
    ("Motivational tone", "Rank from most to least empathetic and encouraging.", "rank"),
    ("Constructiveness", "Rank from most to least helpful and constructive.", "rank"),
    ("Final choice", "Rank from most to least suitable as a final version.", "rank"),
]

CANDIDATE_COLUMNS = ["Gemini", "DeepSeek", "Flash", "Qwen"]  # adjust if your headers differ


def load_tasks(csv_path: str, volunteer: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    needed = ["Volunteer", "Category", "Task ID", "Original"] + CANDIDATE_COLUMNS
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    df = df[df["Volunteer"].astype(str).str.strip() == volunteer.strip()]
    if df.empty:
        raise ValueError(f"No rows found for Volunteer='{volunteer}'")
    return df


def normalize_evaluator_id(group: str, volunteer: str) -> str:
    # e.g., group="A", volunteer="Volunteer 1" -> "A-Volunteer1"
    v = volunteer.replace(" ", "")
    return f"{group}-{v}"


def make_form_id(evaluator_id: str, category: str, task_id: str, q_index_1based: int) -> str:
    return f"{evaluator_id}|{category}|{task_id}|Q{q_index_1based:02d}"


def format_candidates(cands: List[str]) -> str:
    lines = []
    for i, c in enumerate(cands, start=1):
        lines.append(f"[{i}] {c}".rstrip())
    return "\n".join(lines)


def build_prompt(
    evaluator_id: str,
    group: str,
    volunteer: str,
    category: str,
    task_id: str,
    original: str,
    candidates: List[str],
    criterion_name: str,
    criterion_instruction: str,
    criterion_kind: str,  # "rank" or "subset"
) -> Tuple[str, str]:
    """
    Returns (title_for_file, prompt_text)
    """
    # Q index: locate criterion order
    q_index = [c[0] for c in CRITERIA].index(criterion_name) + 1
    form_id = make_form_id(evaluator_id, category, task_id, q_index)

    # Instruction tail and JSON stub
    if criterion_kind == "rank":
        tail = (
            "Please rank outputs strictly from best (1) to worst (4). No ties.\n"
            "Return JSON only.\n\n"
            "{\n"
            f'  "form_id": "{form_id}",\n'
            f'  "criterion": "{criterion_name}",\n'
            '  "rating": [1, 2, 3, 4],\n'
            '  "justifications": {\n'
            '    "1": "one sentence justification for candidate 1",\n'
            '    "2": "one sentence justification for candidate 2",\n'
            '    "3": "one sentence justification for candidate 3",\n'
            '    "4": "one sentence justification for candidate 4"\n'
            "  }\n"
            "}\n"
        )
    else:  # subset (Accuracy)
        tail = (
            "Please select all outputs that are factually correct given the Original.\n"
            "Return JSON only.\n\n"
            "{\n"
            f'  "form_id": "{form_id}",\n'
            f'  "criterion": "{criterion_name}",\n'
            '  "rating": [1, 3],\n'
            '  "justifications": {\n'
            '    "1": "why candidate 1 is correct/incorrect in one sentence",\n'
            '    "2": "why candidate 2 is correct/incorrect in one sentence",\n'
            '    "3": "why candidate 3 is correct/incorrect in one sentence",\n'
            '    "4": "why candidate 4 is correct/incorrect in one sentence"\n'
            "  }\n"
            "}\n"
        )

    prompt = (
        f"You are {volunteer} in Group {group}.\n"
        f"TASK TYPE: {category}\n"
        f"SAMPLE: {task_id}\n"
        f"CRITERION: {criterion_name}\n\n"
        f"INSTRUCTION:\n{criterion_instruction}\n\n"
        f"ORIGINAL:\n{original}\n\n"
        f"CANDIDATES:\n{format_candidates(candidates)}\n\n"
        f"{tail}"
    )

    title = f"{task_id}_Q{q_index:02d}_{criterion_name.replace(' ', '')}"
    return title, prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to tasks CSV, e.g., 'handpick - tasks A.csv'")
    ap.add_argument("--volunteer", required=True, help='Volunteer name as in CSV, e.g., "Volunteer 1"')
    ap.add_argument("--group", default="A", help="Group label to embed in form_id, e.g., A/B/C")
    ap.add_argument("--outdir", default="./prompts_out", help="Directory to write prompts")
    ap.add_argument("--one-file", default=None, help="Optional single file name to collect all prompts")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle candidate order per task/criterion")
    ap.add_argument("--seed", type=int, default=None, help="RNG seed for shuffling (for reproducibility)")
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    # Load and filter
    try:
        df = load_tasks(args.csv, args.volunteer)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    evaluator_id = normalize_evaluator_id(args.group, args.volunteer)

    all_prompts: List[Tuple[str, str]] = []

    for _, row in df.iterrows():
        category = str(row["Category"]).strip()
        task_id = str(row["Task ID"]).strip()
        original = str(row["Original"]).strip()

        # Pull candidate texts
        raw_candidates = [str(row[c]).strip() for c in CANDIDATE_COLUMNS]

        # We’ll create one prompt per criterion (8 total)
        for (crit_name, crit_text, crit_kind) in CRITERIA:
            # Optionally shuffle the candidates per form
            if args.shuffle:
                idxs = [0, 1, 2, 3]
                random.shuffle(idxs)
                cands = [raw_candidates[i] for i in idxs]
            else:
                cands = list(raw_candidates)

            title, prompt_text = build_prompt(
                evaluator_id=evaluator_id,
                group=args.group,
                volunteer=args.volunteer,
                category=category,
                task_id=task_id,
                original=original,
                candidates=cands,
                criterion_name=crit_name,
                criterion_instruction=crit_text,
                criterion_kind=crit_kind,
            )
            all_prompts.append((title, prompt_text))

    # Write outputs
    if args.one_file:
        one_path = outdir / args.one_file
        with one_path.open("w", encoding="utf-8") as f:
            for i, (title, prompt_text) in enumerate(all_prompts, start=1):
                f.write(f"----- PROMPT {i}: {title} -----\n")
                f.write(prompt_text)
                f.write("\n\n")
        print(f"Wrote {len(all_prompts)} prompts to: {one_path}")
    else:
        for title, prompt_text in all_prompts:
            file_path = outdir / f"{title}.txt"
            with file_path.open("w", encoding="utf-8") as f:
                f.write(prompt_text)
        print(f"Wrote {len(all_prompts)} prompts to directory: {outdir}")


if __name__ == "__main__":
    main()
