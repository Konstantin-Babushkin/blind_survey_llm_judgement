#!/usr/bin/env python3
# simple_matcher.py
#
# Usage:
#   python simple_matcher.py \
#     --prompts "/path/prompts_A_V1.txt" \
#     --handpick "/path/handpick - tasks A.csv" \
#     --out "/path/mapping_A.csv"

import argparse
import re
import unicodedata
from pathlib import Path
import pandas as pd

# -- Regex for blocks and candidates (kept simple) ----------------------------
PROMPT_BLOCK_RE = re.compile(
    r"----- PROMPT \d+: (?P<label>(?P<task>[A-C]\d+)_Q\d{2}_[A-Za-z]+) -----"
    r".*?CANDIDATES:\s*(?P<cands>(?:\[\d\].*?(?:\n|$))+?)\s*(?:Please rank|Please select)",
    re.DOTALL,
)
CAND_RE = re.compile(r"\[(\d)\]\s*(.*?)(?=\n\[\d\]|\Z)", re.DOTALL)

def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s or "")).replace("\u00a0"," ")
    s = s.replace("\u2019","'").replace("\u2014","-").replace("\u2013","-")
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def parse_prompts(path: Path):
    """Yield (task_id, prompt_label, {idx->candidate_text_normalized})."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    for m in PROMPT_BLOCK_RE.finditer(text):
        label = m.group("label")
        task  = m.group("task")
        cands = {int(i): normalize(t) for i, t in CAND_RE.findall(m.group("cands"))}
        yield task, label, cands

def main():
    ap = argparse.ArgumentParser(description="Match candidates in a single-group prompts file to model names.")
    ap.add_argument("--prompts", required=True, help="prompts_*.txt (one group)")
    ap.add_argument("--handpick", required=True, help="'handpick - tasks X.csv' for the same group")
    ap.add_argument("--out", required=True, help="output CSV")
    args = ap.parse_args()

    # Load handpick CSV and normalize model texts
    hp = pd.read_csv(args.handpick)
    hp["Task ID"] = hp["Task ID"].astype(str).str.strip()
    for col in ["Gemini","DeepSeek","Flash","Qwen"]:
        hp[col + "__norm"] = hp[col].astype(str).map(normalize)

    rows = []
    for task_id, prompt_label, cand_map in parse_prompts(Path(args.prompts)):
        row = hp.loc[hp["Task ID"] == task_id]
        if row.empty:
            # If the task isn't in the CSV, mark all 1..4 as UNMATCHED
            for idx, _ in cand_map.items():
                rows.append({
                    "task_id": task_id,
                    "prompt_label": prompt_label,
                    "candidate_index": idx,
                    "model": "UNMATCHED",
                })
            continue

        r = row.iloc[0]
        # Reverse map normalized text -> model name
        rev = {
            r["Gemini__norm"]: "Gemini",
            r["DeepSeek__norm"]: "DeepSeek",
            r["Flash__norm"]:   "Flash",
            r["Qwen__norm"]:    "Qwen",
        }

        for idx, cand_text in sorted(cand_map.items()):
            model = rev.get(cand_text, "UNMATCHED")
            rows.append({
                "task_id": task_id,
                "prompt_label": prompt_label,
                "candidate_index": idx,
                "model": model,
            })

    pd.DataFrame(rows).to_csv(args.out, index=False)

if __name__ == "__main__":
    main()
