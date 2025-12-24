"""
Compute criterion importance for HUMAN evaluators directly from multiple raw-ranking CSVs.

Input (raw rows), per model output:
task_id, task_type, model_name, evaluator, Fluency, Coherence, Conciseness, Accuracy,
Constructiveness, Final, Motivational, Sentiment

Evaluator group is inferred from evaluator prefix:
- evaluator like "c3_..." -> group "C"
- "a1_..." -> "A", "b2_..." -> "B"

Outputs:
- human_importance_corr.csv  (Spearman ρ with Final)
- human_importance_reg.csv   (standardized regression β predicting Final)
- group_topk_corr.csv        (top-k criteria per group by |ρ| or ρ)
- group_topk_reg.csv         (top-k criteria per group by |β| or β)

Scopes computed:
- ALL (pooled)
- TASK (per task_type)
- GROUP (A/B/C pooled across tasks)
- GROUP_TASK (A/B/C per task_type)
- EVALUATOR (per evaluator_id prefix like a1/b3/c2)
- EVALUATOR_TASK (per evaluator × task)

Run:
  python human_importance_from_raw.py --input_glob "/path/*.csv" --outdir out --topk 3

Notes:
- Spearman uses ranks, robust to scaling.
- Regression is OLS on z-scored predictors and target (standardized β).
- Regression uses complete cases and drops constant predictors within each scope.

"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


CRITERIA = ["Fluency", "Coherence", "Conciseness", "Accuracy", "Constructiveness", "Motivational", "Sentiment"]
FINAL_COL = "Final"


def infer_group(evaluator: pd.Series) -> pd.Series:
    return evaluator.astype(str).str.strip().str[0].str.upper()


def infer_evaluator_index(evaluator: pd.Series) -> pd.Series:
    """
    Extract only the index before underscore, e.g. 'c3_ISHE...' -> 'c3'.
    If no underscore, returns leading alnum chunk.
    """
    s = evaluator.astype(str).str.strip()
    m = s.str.extract(r"^([A-Za-z]+\d+)_")
    out = m[0].fillna(s.str.extract(r"^([A-Za-z]+\d+)")[0])
    return out.fillna(s)


def list_files(input_dir: str | None, input_glob: str | None) -> list[Path]:
    if input_glob:
        # supports absolute globs
        import glob as _glob

        files = [Path(p) for p in sorted(_glob.glob(input_glob))]
    elif input_dir:
        files = sorted(Path(input_dir).glob("*.csv"))
    else:
        raise ValueError("Provide --input_dir or --input_glob")
    if not files:
        raise FileNotFoundError("No CSV files found.")
    return files


def read_raw_csvs(files: list[Path]) -> pd.DataFrame:
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df.columns = [str(c).strip().strip('"') for c in df.columns]
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)

    required = {"task_id", "task_type", "model_name", "evaluator"} | set(CRITERIA) | {FINAL_COL}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Coerce numeric
    for c in CRITERIA + [FINAL_COL]:
        data[c] = pd.to_numeric(data[c], errors="coerce")

    data["group"] = infer_group(data["evaluator"])
    data["evaluator_id"] = infer_evaluator_index(data["evaluator"])

    # Keep relevant cols
    keep = ["task_id", "task_type", "model_name", "evaluator", "evaluator_id", "group"] + CRITERIA + [FINAL_COL]
    return data[keep].copy()


def spearman_corr(x: pd.Series, y: pd.Series) -> float:
    tmp = pd.concat([x, y], axis=1).dropna()
    if len(tmp) < 3:
        return np.nan
    rx = tmp.iloc[:, 0].rank(method="average")
    ry = tmp.iloc[:, 1].rank(method="average")
    if rx.nunique() < 2 or ry.nunique() < 2:
        return np.nan
    return float(rx.corr(ry))


def standardized_betas(sub: pd.DataFrame) -> pd.DataFrame:
    tmp = sub[CRITERIA + [FINAL_COL]].dropna().copy()
    n = len(tmp)
    if n < max(6, len(CRITERIA) + 2):
        return pd.DataFrame({"criterion": CRITERIA, "std_beta": [np.nan] * len(CRITERIA), "N": n, "note": "insufficient_data"})

    X = tmp[CRITERIA].astype(float)
    y = tmp[FINAL_COL].astype(float)

    # Drop constant predictors
    stds = X.std(ddof=0)
    keep = stds[stds > 0].index.tolist()
    dropped = [c for c in CRITERIA if c not in keep]

    X = X[keep]
    Xz = (X - X.mean()) / X.std(ddof=0)

    ystd = y.std(ddof=0)
    if ystd == 0 or np.isnan(ystd):
        # Final is constant in this scope
        rows = [{"criterion": c, "std_beta": np.nan, "N": n, "note": "final_constant"} for c in CRITERIA]
        return pd.DataFrame(rows)

    yz = (y - y.mean()) / ystd

    Xmat = np.column_stack([np.ones(len(Xz)), Xz.to_numpy()])
    coef, *_ = np.linalg.lstsq(Xmat, yz.to_numpy(), rcond=None)

    rows = []
    for c, b in zip(keep, coef[1:]):
        rows.append({"criterion": c, "std_beta": float(b), "N": n, "note": ""})
    for c in dropped:
        rows.append({"criterion": c, "std_beta": np.nan, "N": n, "note": "dropped_constant"})

    return pd.DataFrame(rows)


def compute_corr(sub: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in CRITERIA:
        rows.append({
            "criterion": c,
            "spearman_rho": spearman_corr(sub[c], sub[FINAL_COL]),
            "N": int(sub[[c, FINAL_COL]].dropna().shape[0]),
        })
    return pd.DataFrame(rows)


def add_scope_rows(
    out_corr: list[dict],
    out_reg: list[dict],
    scope: str,
    sub: pd.DataFrame,
    evaluator_id: str = "ALL",
    group: str = "ALL",
    task_type: str = "ALL",
):
    corr_df = compute_corr(sub)
    for _, r in corr_df.iterrows():
        out_corr.append({
            "scope": scope,
            "evaluator_id": evaluator_id,
            "group": group,
            "task_type": task_type,
            "criterion": r["criterion"],
            "spearman_rho": r["spearman_rho"],
            "N": int(r["N"]),
        })

    reg_df = standardized_betas(sub)
    for _, r in reg_df.iterrows():
        out_reg.append({
            "scope": scope,
            "evaluator_id": evaluator_id,
            "group": group,
            "task_type": task_type,
            "criterion": r["criterion"],
            "std_beta": r["std_beta"],
            "N": int(r["N"]),
            "note": r["note"],
        })


def topk(df: pd.DataFrame, value_col: str, k: int, use_abs: bool = False) -> pd.DataFrame:
    d = df.copy()
    d["_key"] = d[value_col].abs() if use_abs else d[value_col]
    return (
        d.sort_values(["scope", "group", "task_type", "evaluator_id", "_key"], ascending=[True, True, True, True, False])
        .groupby(["scope", "group", "task_type", "evaluator_id"], as_index=False)
        .head(k)
        .drop(columns=["_key"])
        .reset_index(drop=True)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default='./data')
    ap.add_argument("--input_glob", default=None)
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--topk_abs", action="store_true", help="Rank top-k by absolute |rho| / |beta|.")
    args = ap.parse_args()

    files = list_files(args.input_dir, args.input_glob)
    data = read_raw_csvs(files)

    out_corr: list[dict] = []
    out_reg: list[dict] = []

    # ALL pooled
    add_scope_rows(out_corr, out_reg, scope="ALL", sub=data)

    # TASK (pooled)
    for task, sub in data.groupby("task_type", dropna=False):
        add_scope_rows(out_corr, out_reg, scope="TASK", sub=sub, task_type=str(task))

    # GROUP (A/B/C pooled)
    for grp, sub in data.groupby("group", dropna=False):
        add_scope_rows(out_corr, out_reg, scope="GROUP", sub=sub, group=str(grp))

    # GROUP_TASK
    for (grp, task), sub in data.groupby(["group", "task_type"], dropna=False):
        add_scope_rows(out_corr, out_reg, scope="GROUP_TASK", sub=sub, group=str(grp), task_type=str(task))

    # EVALUATOR (a1/b2/c3 pooled)
    for ev, sub in data.groupby("evaluator_id", dropna=False):
        add_scope_rows(out_corr, out_reg, scope="EVALUATOR", sub=sub, evaluator_id=str(ev), group=str(ev)[0].upper())

    # EVALUATOR_TASK
    for (ev, task), sub in data.groupby(["evaluator_id", "task_type"], dropna=False):
        add_scope_rows(out_corr, out_reg, scope="EVALUATOR_TASK", sub=sub, evaluator_id=str(ev), group=str(ev)[0].upper(), task_type=str(task))

    corr_df = pd.DataFrame(out_corr)
    reg_df = pd.DataFrame(out_reg)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    corr_path = outdir / "human_importance_corr.csv"
    reg_path = outdir / "human_importance_reg.csv"
    corr_df.to_csv(corr_path, index=False)
    reg_df.to_csv(reg_path, index=False)

    # Top-k per scope×group×task×evaluator_id
    corr_top = topk(corr_df, "spearman_rho", args.topk, use_abs=args.topk_abs)
    reg_top = topk(reg_df, "std_beta", args.topk, use_abs=args.topk_abs)

    corr_top_path = outdir / f"human_top{args.topk}_corr.csv"
    reg_top_path = outdir / f"human_top{args.topk}_reg.csv"
    corr_top.to_csv(corr_top_path, index=False)
    reg_top.to_csv(reg_top_path, index=False)

    print(f"Read {len(files)} files, {len(data)} rows.")
    print("Saved:", corr_path)
    print("Saved:", reg_path)
    print("Saved:", corr_top_path)
    print("Saved:", reg_top_path)


if __name__ == "__main__":
    main()
